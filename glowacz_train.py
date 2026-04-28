#!/usr/bin/env python3
"""
Glowacz et al. Full Replication: CNN Training, Evaluation & Paper Figures

Replicates the acoustic fault diagnosis pipeline end-to-end:
  - Train DenseNet-201, ResNet-18, ResNet-50, EfficientNet-B0 on 224x224x3
    acoustic images produced by paper1_pipeline.py.
  - Evaluate with ER (overall accuracy) and MER (mean per-class accuracy).
  - Stratified K-fold cross-validation.
  - Generate all four paper figures:
      1. Time-domain waveforms per class
      2. FFT spectra (1-1000 Hz) per class
      3. Word-vector stem maps (519-807 Hz) per class
      4. 17x17 acoustic image grids per class

Usage:
  python glowacz_train.py --image_dir output_paper1/images --dataset_dir dataset

  # Plots only (no training):
  python glowacz_train.py --generate_plots --dataset_dir dataset

  # Specific models:
  python glowacz_train.py --models resnet18 resnet50 --epochs 100
"""

import argparse
import copy
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

sys.path.insert(0, str(Path(__file__).parent))
from paper1_pipeline import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    FFT_CROP_HIGH,
    FFT_CROP_LOW,
    MATRIX_SIZE,
    SEGMENT_LENGTH,
    TARGET_SR,
    WORD_CODING_K,
    compute_fft_magnitude,
    crop_spectrum,
    extract_dwv_range,
    load_audio,
    normalize_amplitude,
    pre_filter,
    resample_signal,
    segment_signal,
    word_coding,
)

PAPER_DWV_LOW = 519
PAPER_DWV_HIGH = 807

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ALL_MODELS = ["densenet201", "resnet18", "resnet50", "efficientnet_b0"]


def get_device(device_str):
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def create_model(model_name, num_classes):
    if model_name == "densenet201":
        model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def get_transforms(augment=False):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if augment:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                norm,
            ]
        )
    return transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), norm]
    )


def split_dataset(targets, train_per_class, seed=42):
    rng = np.random.RandomState(seed)
    classes = np.unique(targets)
    train_idx, test_idx = [], []
    for cls in classes:
        cls_idx = np.where(targets == cls)[0]
        rng.shuffle(cls_idx)
        n = min(train_per_class, len(cls_idx))
        train_idx.extend(cls_idx[:n].tolist())
        test_idx.extend(cls_idx[n:].tolist())
    return train_idx, test_idx


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / max(total, 1), 100.0 * correct / max(total, 1)


def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            running_loss += criterion(outputs, labels).item() * images.size(0)
            all_preds.extend(outputs.max(1)[1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    n_classes = len(set(labels))

    er = 100.0 * np.sum(preds == labels) / max(len(labels), 1)

    per_class = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class.append(100.0 * np.sum(preds[mask] == c) / mask.sum())
    mer = float(np.mean(per_class)) if per_class else 0.0

    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    return running_loss / max(len(labels), 1), er, mer, cm


def train_and_evaluate(
    model_name, train_loader, test_loader, num_classes, epochs, lr, device
):
    model = create_model(model_name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_state, best_er = None, 0.0
    history = {"train_loss": [], "train_acc": [], "test_er": [], "test_mer": []}

    for epoch in range(epochs):
        t_loss, t_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        _, er, mer, _ = evaluate_model(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["test_er"].append(er)
        history["test_mer"].append(mer)

        if er > best_er:
            best_er = er
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "[%s] Epoch %3d/%d  loss=%.4f  acc=%.1f%%  ER=%.1f%%  MER=%.1f%%",
                model_name,
                epoch + 1,
                epochs,
                t_loss,
                t_acc,
                er,
                mer,
            )

    if best_state:
        model.load_state_dict(best_state)
    return model, history


def run_kfold(
    model_name,
    train_ds,
    eval_ds,
    targets,
    num_classes,
    k,
    epochs,
    lr,
    batch_size,
    device,
    seed,
):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        logger.info("  [%s] Fold %d/%d", model_name, fold + 1, k)
        tr_loader = DataLoader(
            Subset(train_ds, tr_idx.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        te_loader = DataLoader(
            Subset(eval_ds, te_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        _, _, er, mer, cm = None, None, 0.0, 0.0, None
        model, hist = train_and_evaluate(
            model_name, tr_loader, te_loader, num_classes, epochs, lr, device
        )
        criterion = nn.CrossEntropyLoss()
        _, er, mer, cm = evaluate_model(model, te_loader, criterion, device)
        fold_results.append({"er": er, "mer": mer, "cm": cm, "history": hist})
        logger.info("    Fold %d: ER=%.1f%%  MER=%.1f%%", fold + 1, er, mer)

    return fold_results


def plot_training_curves(history, model_name, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["test_er"], label="Test ER")
    axes[1].set_title("Accuracy / ER (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[2].plot(history["test_mer"])
    axes[2].set_title("Test MER (%)")
    axes[2].set_xlabel("Epoch")
    fig.suptitle(model_name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, class_names, model_name, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    ax.set_title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _load_one_sample_per_class(dataset_dir):
    class_dirs = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    samples = {}
    for cd in class_dirs:
        wavs = sorted(
            [
                f
                for f in cd.iterdir()
                if f.suffix.lower() == ".wav" and not f.name.startswith(".")
            ]
        )
        if wavs:
            sig, sr = load_audio(str(wavs[0]))
            sig = resample_signal(sig, sr, TARGET_SR)
            samples[cd.name] = sig
    return samples


def _process_sample(signal, dwv_low=PAPER_DWV_LOW, dwv_high=PAPER_DWV_HIGH):
    segments = segment_signal(signal, SEGMENT_LENGTH)
    seg = segments[0] if segments else np.zeros(SEGMENT_LENGTH)
    filtered = pre_filter(seg, BANDPASS_LOW, BANDPASS_HIGH, TARGET_SR)
    normalized = normalize_amplitude(filtered)
    spectrum = compute_fft_magnitude(normalized)
    cropped, base_freq, freq_res = crop_spectrum(
        spectrum, FFT_CROP_LOW, FFT_CROP_HIGH, TARGET_SR
    )
    max_mag = np.max(cropped)
    if max_mag > 0:
        cropped = cropped / max_mag
    wv = word_coding(cropped, WORD_CODING_K)
    word_slice = extract_dwv_range(wv, base_freq, freq_res, dwv_low, dwv_high)
    return seg, cropped, base_freq, freq_res, word_slice


def plot_time_domain(dataset_dir, output_path):
    samples = _load_one_sample_per_class(dataset_dir)
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), squeeze=False)
    for ax, (cls, sig) in zip(axes.flat, samples.items()):
        t = np.arange(len(sig)) / TARGET_SR
        axes_flat = axes.flatten()
        idx = list(samples.keys()).index(cls)
        axes_flat[idx].plot(t, sig, linewidth=0.3, color="steelblue")
        axes_flat[idx].set_title(cls, fontsize=10)
        axes_flat[idx].set_xlabel("Time (s)")
        axes_flat[idx].set_ylabel("Amplitude")
        axes_flat[idx].set_xlim(0, t[-1])
    plt.suptitle("Time-Domain Signals", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_fft_spectra(dataset_dir, output_path):
    samples = _load_one_sample_per_class(dataset_dir)
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), squeeze=False)
    for idx, (cls, sig) in enumerate(samples.items()):
        _, cropped, base_freq, freq_res, _ = _process_sample(sig)
        freqs = np.arange(len(cropped)) * freq_res + base_freq
        axes[idx, 0].bar(freqs, cropped, width=freq_res, align="center")
        axes[idx, 0].set_title(cls, fontsize=10)
        axes[idx, 0].set_xlabel("Frequency (Hz)")
        axes[idx, 0].set_ylabel("Normalized Amplitude")
        axes[idx, 0].set_xlim(FFT_CROP_LOW, FFT_CROP_HIGH)
        axes[idx, 0].set_ylim(bottom=0)
        axes[idx, 0].grid(True, alpha=0.3)
    plt.suptitle("FFT Spectra (1–1000 Hz)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_word_vectors(
    dataset_dir, output_path, dwv_low=PAPER_DWV_LOW, dwv_high=PAPER_DWV_HIGH
):
    samples = _load_one_sample_per_class(dataset_dir)
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), squeeze=False)
    axes_flat = axes.flatten()
    for idx, (cls, sig) in enumerate(samples.items()):
        _, _, _, _, word_slice = _process_sample(sig, dwv_low, dwv_high)
        freqs = np.linspace(dwv_low, dwv_high, len(word_slice))
        axes_flat[idx].bar(
            freqs, word_slice, width=(freqs[1] - freqs[0]) if len(freqs) > 1 else 1
        )
        axes_flat[idx].set_title(cls, fontsize=10)
        axes_flat[idx].set_xlabel("Frequency (Hz)")
        axes_flat[idx].set_ylabel("D coefficient")
        axes_flat[idx].set_xlim(dwv_low, dwv_high)
    plt.suptitle(f"Word Vector Maps ({dwv_low}–{dwv_high} Hz)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_acoustic_images(
    dataset_dir, output_path, dwv_low=PAPER_DWV_LOW, dwv_high=PAPER_DWV_HIGH
):
    samples = _load_one_sample_per_class(dataset_dir)
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (cls, sig) in zip(axes, samples.items()):
        _, _, _, _, word_slice = _process_sample(sig, dwv_low, dwv_high)
        expected = MATRIX_SIZE * MATRIX_SIZE
        if len(word_slice) < expected:
            padded = np.zeros(expected)
            padded[: len(word_slice)] = word_slice
            word_slice = padded
        else:
            word_slice = word_slice[:expected]
        matrix_a = word_slice.reshape(MATRIX_SIZE, MATRIX_SIZE)
        ax.imshow(matrix_a, cmap="gray", aspect="equal", interpolation="nearest")
        ax.set_title(cls, fontsize=9)
        ax.axis("off")
    plt.suptitle("Acoustic Images (17×17)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Glowacz et al. Replication – Training & Evaluation"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="output_paper1/images",
        help="Directory of class-sorted acoustic images (from paper1_pipeline.py)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Raw dataset directory (for plot generation)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_glowacz",
        help="Output directory for models, plots, and results",
    )
    parser.add_argument(
        "--train_per_class",
        type=int,
        default=12,
        help="Training samples per class (paper default: 12)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        choices=ALL_MODELS,
        help="Which models to train",
    )
    parser.add_argument(
        "--kfold",
        type=int,
        default=0,
        help="K-fold CV folds (0 = disabled, paper uses 5)",
    )
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        help="Generate all 4 paper figures",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, mps",
    )
    parser.add_argument(
        "--auto_range",
        action="store_true",
        help="Read auto-detected DWV range from output_paper1/dwv_analysis/dwv_report.txt",
    )
    parser.add_argument(
        "--dwv_range_low",
        type=int,
        default=PAPER_DWV_LOW,
        help="DWV frequency range lower bound in Hz (paper: 519)",
    )
    parser.add_argument(
        "--dwv_range_high",
        type=int,
        default=PAPER_DWV_HIGH,
        help="DWV frequency range upper bound in Hz (paper: 807)",
    )
    args = parser.parse_args()

    if args.auto_range:
        report_path = Path("output_paper1") / "dwv_analysis" / "dwv_report.txt"
        if report_path.exists():
            with open(report_path) as fh:
                for line in fh:
                    if line.startswith("Auto-detected optimal range:"):
                        parts = line.split(":")[1].strip()
                        range_part = parts.split()[0]
                        low_s, high_s = range_part.split("-")
                        args.dwv_range_low = int(low_s)
                        args.dwv_range_high = int(high_s)
                        break
            logger.info(
                "Auto-detected DWV range: %d–%d Hz",
                args.dwv_range_low,
                args.dwv_range_high,
            )
        else:
            logger.warning(
                "DWV report not found at %s – falling back to --dwv_range_low/high",
                report_path,
            )

    dwv_low = args.dwv_range_low
    dwv_high = args.dwv_range_high
    logger.info("Using DWV range: %d–%d Hz", dwv_low, dwv_high)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Generate paper figures
    # ------------------------------------------------------------------
    if args.generate_plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = Path(args.dataset_dir)

        logger.info("Generating paper figures...")
        plot_time_domain(dataset_dir, plots_dir / "01_time_domain_signals.png")
        logger.info("  Saved time-domain signals")
        plot_fft_spectra(dataset_dir, plots_dir / "02_fft_spectra.png")
        logger.info("  Saved FFT spectra")
        plot_word_vectors(
            dataset_dir, plots_dir / "03_word_vector_maps.png", dwv_low, dwv_high
        )
        logger.info("  Saved word vector maps")
        plot_acoustic_images(
            dataset_dir, plots_dir / "04_acoustic_images.png", dwv_low, dwv_high
        )
        logger.info("  Saved acoustic images")

        if not (Path(args.image_dir).exists()):
            logger.info("No image_dir found – plots only, done.")
        else:
            logger.info("Plots saved. Use without --generate_plots to train models.")
        return

    device = get_device(args.device)
    logger.info("Device: %s", device)

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        logger.error(
            "Image directory not found: %s  –  run paper1_pipeline.py first.",
            image_dir,
        )
        sys.exit(1)

    eval_ds = datasets.ImageFolder(str(image_dir), transform=get_transforms(False))
    train_ds = datasets.ImageFolder(str(image_dir), transform=get_transforms(True))
    class_names = eval_ds.classes
    num_classes = len(class_names)
    targets = np.array(eval_ds.targets)

    logger.info(
        "Dataset: %d images, %d classes: %s",
        len(eval_ds),
        num_classes,
        class_names,
    )

    train_idx, test_idx = split_dataset(targets, args.train_per_class, args.seed)
    logger.info(
        "Split: %d train, %d test  (%d per class for train)",
        len(train_idx),
        len(test_idx),
        args.train_per_class,
    )

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        Subset(eval_ds, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name in args.models:
        logger.info("=" * 60)
        logger.info(
            "Training %s  (%d epochs, lr=%.5f)", model_name, args.epochs, args.lr
        )
        logger.info("=" * 60)

        model, history = train_and_evaluate(
            model_name,
            train_loader,
            test_loader,
            num_classes,
            args.epochs,
            args.lr,
            device,
        )

        torch.save(model.state_dict(), models_dir / f"{model_name}_best.pth")
        plot_training_curves(
            history, model_name, results_dir / f"{model_name}_curves.png"
        )
        np.savez(
            results_dir / f"{model_name}_history.npz",
            **{k: np.array(v) for k, v in history.items()},
        )

        criterion = nn.CrossEntropyLoss()
        _, er, mer, cm = evaluate_model(model, test_loader, criterion, device)
        plot_confusion_matrix(
            cm, class_names, model_name, results_dir / f"{model_name}_confusion.png"
        )

        all_results[model_name] = {
            "er": er,
            "mer": mer,
            "cm": cm,
            "history": history,
        }

        logger.info("[%s] Final — ER: %.1f%%   MER: %.1f%%", model_name, er, mer)

    # ------------------------------------------------------------------
    # K-fold cross-validation
    # ------------------------------------------------------------------
    kfold_results = {}
    if args.kfold > 0:
        kf_dir = output_dir / "kfold"
        kf_dir.mkdir(parents=True, exist_ok=True)

        for model_name in args.models:
            logger.info("=" * 60)
            logger.info("K-fold CV (%d folds): %s", args.kfold, model_name)
            logger.info("=" * 60)

            folds = run_kfold(
                model_name,
                train_ds,
                eval_ds,
                targets,
                num_classes,
                args.kfold,
                args.epochs,
                args.lr,
                args.batch_size,
                device,
                args.seed,
            )

            ers = [f["er"] for f in folds]
            mers = [f["mer"] for f in folds]
            kfold_results[model_name] = {"ers": ers, "mers": mers, "folds": folds}

            logger.info(
                "[%s] K-fold  ER: %.1f ± %.1f%%   MER: %.1f ± %.1f%%",
                model_name,
                np.mean(ers),
                np.std(ers),
                np.mean(mers),
                np.std(mers),
            )

            np.savez(
                kf_dir / f"{model_name}_kfold.npz",
                ers=np.array(ers),
                mers=np.array(mers),
            )

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    report_path = output_dir / "summary_report.txt"
    with open(report_path, "w") as fh:
        fh.write("Glowacz et al. Replication – Summary Report\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(f"Classes ({num_classes}): {class_names}\n")
        fh.write(f"Total images: {len(eval_ds)}\n")
        fh.write(f"Train/test split: {args.train_per_class} per class for train\n")
        fh.write(f"  Train: {len(train_idx)}   Test: {len(test_idx)}\n")
        fh.write(
            f"Epochs: {args.epochs}   LR: {args.lr}   Batch: {args.batch_size}\n\n"
        )

        fh.write("-" * 50 + "\n")
        fh.write("Fixed-Split Results\n")
        fh.write("-" * 50 + "\n")
        for mn in args.models:
            r = all_results[mn]
            fh.write(f"  {mn:20s}  ER = {r['er']:6.1f}%   MER = {r['mer']:6.1f}%\n")

        if kfold_results:
            fh.write("\n" + "-" * 50 + "\n")
            fh.write(f"K-fold CV Results ({args.kfold} folds)\n")
            fh.write("-" * 50 + "\n")
            for mn in args.models:
                kr = kfold_results[mn]
                fh.write(
                    f"  {mn:20s}  ER = {np.mean(kr['ers']):5.1f} ± {np.std(kr['ers']):4.1f}%"
                    f"   MER = {np.mean(kr['mers']):5.1f} ± {np.std(kr['mers']):4.1f}%\n"
                )

    logger.info("Summary report saved to %s", report_path)
    logger.info("All done.")


if __name__ == "__main__":
    main()
