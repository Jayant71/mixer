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
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    compute_dwv_analysis,
    compute_fft_magnitude,
    create_acoustic_image,
    crop_spectrum,
    extract_dwv_range,
    find_optimal_range,
    load_audio,
    normalize_amplitude,
    pre_filter,
    process_segment,
    resample_signal,
    save_image,
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
            return torch.device("cuda:1")
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


def split_dataset(targets, test_size=0.15, val_size=0.15, seed=42):
    indices = np.arange(len(targets))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=targets, random_state=seed
    )
    train_val_targets = targets[train_val_idx]
    val_fraction = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_fraction, stratify=train_val_targets, random_state=seed
    )
    return train_idx, val_idx, test_idx


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
    model_name, train_loader, val_loader, num_classes, epochs, lr, device
):
    model = create_model(model_name, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_state, best_er = None, 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_er": [],
        "val_mer": [],
    }

    for epoch in range(epochs):
        start_time = time.time()
        t_loss, t_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        v_loss, er, mer, _ = evaluate_model(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - start_time
        curr_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_er"].append(er)
        history["val_mer"].append(mer)

        if er > best_er:
            best_er = er
            best_state = copy.deepcopy(model.state_dict())

        logger.info(
            "[%s] Epoch %3d/%d | Time: %.1fs | LR: %.6f | Loss: %.4f | Acc: %.1f%% | Val Loss: %.4f | Val ER: %.1f%% | Val MER: %.1f%%",
            model_name,
            epoch + 1,
            epochs,
            epoch_time,
            curr_lr,
            t_loss,
            t_acc,
            v_loss,
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
    val_fraction=0.15,
):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_val_idx, te_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        logger.info("  [%s] Fold %d/%d", model_name, fold + 1, k)
        train_val_targets = targets[train_val_idx]
        tr_idx, va_idx = train_test_split(
            train_val_idx, test_size=val_fraction,
            stratify=train_val_targets, random_state=seed,
        )
        tr_loader = DataLoader(
            Subset(train_ds, tr_idx.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        va_loader = DataLoader(
            Subset(eval_ds, va_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        te_loader = DataLoader(
            Subset(eval_ds, te_idx.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        model, hist = train_and_evaluate(
            model_name, tr_loader, va_loader, num_classes, epochs, lr, device
        )
        criterion = nn.CrossEntropyLoss()
        _, val_er, val_mer, _ = evaluate_model(model, va_loader, criterion, device)
        _, test_er, test_mer, cm = evaluate_model(model, te_loader, criterion, device)
        fold_results.append({
            "val_er": val_er, "val_mer": val_mer,
            "test_er": test_er, "test_mer": test_mer,
            "cm": cm, "history": hist,
        })
        logger.info(
            "    Fold %d: Val ER=%.1f%%  Test ER=%.1f%%", fold + 1, val_er, test_er,
        )

    return fold_results


def plot_training_curves(history, model_name, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss (Error)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_er"], label="Val ER")
    axes[1].set_title("Accuracy / Val ER (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history["val_mer"])
    axes[2].set_title("Val MER (%)")
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


def _detect_native_sr(dataset_dir):
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        for f in class_dir.iterdir():
            if f.suffix.lower() == ".wav" and not f.name.startswith("."):
                try:
                    _, sr = load_audio(str(f))
                    return sr
                except Exception:
                    continue
    return None


def _load_one_sample_per_class(dataset_dir, target_sr=TARGET_SR, no_resample=False):
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
            if no_resample:
                sig = resample_signal(sig, sr, sr)
            else:
                sig = resample_signal(sig, sr, target_sr)
            samples[cd.name] = sig
    return samples


def _process_sample(
    signal,
    dwv_low=PAPER_DWV_LOW,
    dwv_high=PAPER_DWV_HIGH,
    filter_low=BANDPASS_LOW,
    filter_high=BANDPASS_HIGH,
    filter_order=5,
    fft_crop_low=FFT_CROP_LOW,
    fft_crop_high=FFT_CROP_HIGH,
    word_k=WORD_CODING_K,
    segment_length=SEGMENT_LENGTH,
    target_sr=TARGET_SR,
):
    segments = segment_signal(signal, segment_length)
    seg = segments[0] if segments else np.zeros(segment_length)
    filtered = pre_filter(seg, filter_low, filter_high, target_sr, order=filter_order)
    normalized = normalize_amplitude(filtered)
    spectrum = compute_fft_magnitude(normalized)
    cropped, base_freq, freq_res = crop_spectrum(
        spectrum, fft_crop_low, fft_crop_high, target_sr
    )
    max_mag = np.max(cropped)
    if max_mag > 0:
        cropped = cropped / max_mag
    wv = word_coding(cropped, word_k)
    word_slice = extract_dwv_range(wv, base_freq, freq_res, dwv_low, dwv_high)
    return seg, cropped, base_freq, freq_res, word_slice


def _compute_auto_dwv_range(dataset_dir, matrix_size, segment_length=SEGMENT_LENGTH, **proc_kwargs):
    from collections import defaultdict

    dataset_path = Path(dataset_dir)
    class_dirs = sorted(
        [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

    all_word_vectors = defaultdict(list)
    target_sr = proc_kwargs.pop("target_sr", TARGET_SR)
    no_resample = proc_kwargs.pop("no_resample", False)
    for class_dir in class_dirs:
        wav_files = sorted(
            [f for f in class_dir.iterdir() if f.suffix.lower() == ".wav" and not f.name.startswith(".")]
        )
        for wav_file in wav_files:
            try:
                signal, sr = load_audio(str(wav_file))
            except Exception:
                continue
            if no_resample:
                signal = resample_signal(signal, sr, sr)
            else:
                signal = resample_signal(signal, sr, target_sr)
            segments = segment_signal(signal, segment_length)
            for segment in segments:
                try:
                    wv, _, _ = process_segment(segment, target_sr, **proc_kwargs)
                except Exception:
                    continue
                all_word_vectors[class_dir.name].append(wv)

    if not all_word_vectors:
        return None, None

    dwv = compute_dwv_analysis(dict(all_word_vectors))
    if len(dwv) == 0:
        return None, None

    expected = matrix_size * matrix_size
    best_start, best_end, best_sum = find_optimal_range(dwv, expected)
    freq_resolution = target_sr / segment_length
    fft_crop_low = proc_kwargs.get("fft_crop_low", FFT_CROP_LOW)
    auto_low = int(fft_crop_low + best_start * freq_resolution)
    auto_high = int(fft_crop_low + best_end * freq_resolution)

    logger.info(
        "Auto-detected DWV range from dataset: %d-%d Hz (sum=%.4f)",
        auto_low, auto_high, best_sum,
    )
    return auto_low, auto_high


def _run_pipeline(image_dir, dataset_dir, dwv_low, dwv_high, matrix_size, output_size, segment_length=SEGMENT_LENGTH, **proc_kwargs):
    from collections import defaultdict

    dataset_path = Path(dataset_dir)
    class_dirs = sorted(
        [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    if not class_dirs:
        logger.error("No class directories found in %s", dataset_path)
        return False

    all_word_vectors = defaultdict(list)
    segment_registry = []
    target_sr = proc_kwargs.pop("target_sr", TARGET_SR)
    no_resample = proc_kwargs.pop("no_resample", False)

    for class_dir in class_dirs:
        class_name = class_dir.name
        wav_files = sorted(
            [f for f in class_dir.iterdir() if f.suffix.lower() == ".wav" and not f.name.startswith(".")]
        )
        logger.info("  Processing class '%s': %d WAV files", class_name, len(wav_files))
        for wav_file in wav_files:
            try:
                signal, sr = load_audio(str(wav_file))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", wav_file, exc)
                continue
            if no_resample:
                signal = resample_signal(signal, sr, sr)
            else:
                signal = resample_signal(signal, sr, target_sr)
            segments = segment_signal(signal, segment_length)
            for seg_idx, segment in enumerate(segments):
                try:
                    wv, base_freq, freq_res = process_segment(segment, target_sr, **proc_kwargs)
                except Exception as exc:
                    logger.warning("Failed on segment %d of %s: %s", seg_idx, wav_file.name, exc)
                    continue
                all_word_vectors[class_name].append(wv)
                segment_registry.append((class_name, wav_file, seg_idx, wv, base_freq, freq_res))

    if not segment_registry:
        logger.error("No segments generated from dataset")
        return False

    images_path = Path(image_dir)
    total = 0
    for entry in segment_registry:
        class_name, source_file, seg_idx, wv, base_freq, freq_res = entry
        word_slice = extract_dwv_range(wv, base_freq, freq_res, dwv_low, dwv_high)
        acoustic_img = create_acoustic_image(word_slice, matrix_size=matrix_size, output_size=output_size)
        class_img_dir = images_path / class_name
        class_img_dir.mkdir(parents=True, exist_ok=True)
        img_filename = f"{source_file.stem}_seg{seg_idx:03d}.png"
        save_image(acoustic_img, class_img_dir / img_filename)
        total += 1

    logger.info("Generated %d acoustic images in %s", total, images_path)
    return True


def plot_time_domain(dataset_dir, output_path, target_sr=TARGET_SR, no_resample=False):
    samples = _load_one_sample_per_class(dataset_dir, target_sr=target_sr, no_resample=no_resample)
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), squeeze=False)
    for ax, (cls, sig) in zip(axes.flat, samples.items()):
        t = np.arange(len(sig)) / target_sr
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


def plot_fft_spectra(dataset_dir, output_path, fft_crop_low=FFT_CROP_LOW, fft_crop_high=FFT_CROP_HIGH, **proc_kwargs):
    target_sr = proc_kwargs.get("target_sr", TARGET_SR)
    no_resample = proc_kwargs.get("no_resample", False)
    samples = _load_one_sample_per_class(dataset_dir, target_sr=target_sr, no_resample=no_resample)
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), squeeze=False)
    for idx, (cls, sig) in enumerate(samples.items()):
        _, cropped, base_freq, freq_res, _ = _process_sample(sig, fft_crop_low=fft_crop_low, fft_crop_high=fft_crop_high, **proc_kwargs)
        freqs = np.arange(len(cropped)) * freq_res + base_freq
        axes[idx, 0].bar(freqs, cropped, width=freq_res, align="center")
        axes[idx, 0].set_title(cls, fontsize=10)
        axes[idx, 0].set_xlabel("Frequency (Hz)")
        axes[idx, 0].set_ylabel("Normalized Amplitude")
        axes[idx, 0].set_xlim(fft_crop_low, fft_crop_high)
        axes[idx, 0].set_ylim(bottom=0)
        axes[idx, 0].grid(True, alpha=0.3)
    plt.suptitle(f"FFT Spectra ({fft_crop_low}–{fft_crop_high} Hz)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_fft_full_audio(dataset_dir, output_path, fft_crop_low=FFT_CROP_LOW, fft_crop_high=FFT_CROP_HIGH, **proc_kwargs):
    target_sr = proc_kwargs.get("target_sr", TARGET_SR)
    no_resample = proc_kwargs.get("no_resample", False)
    samples = _load_one_sample_per_class(dataset_dir, target_sr=target_sr, no_resample=no_resample)
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), squeeze=False)
    for idx, (cls, sig) in enumerate(samples.items()):
        filter_high = proc_kwargs.get("filter_high", BANDPASS_HIGH)
        filter_order = proc_kwargs.get("filter_order", 5)
        filtered = pre_filter(sig, proc_kwargs.get("filter_low", BANDPASS_LOW), filter_high, target_sr, order=filter_order)
        normalized = normalize_amplitude(filtered)
        spectrum = compute_fft_magnitude(normalized)
        cropped, base_freq, freq_res = crop_spectrum(spectrum, fft_crop_low, fft_crop_high, target_sr)
        freqs = np.arange(len(cropped)) * freq_res + base_freq
        axes[idx, 0].plot(freqs, cropped, linewidth=0.5, color="steelblue")
        axes[idx, 0].set_title(cls, fontsize=10)
        axes[idx, 0].set_xlabel("Frequency (Hz)")
        axes[idx, 0].set_ylabel("Normalized Amplitude")
        axes[idx, 0].set_xlim(fft_crop_low, fft_crop_high)
        axes[idx, 0].set_ylim(bottom=0)
        axes[idx, 0].grid(True, alpha=0.3)
    plt.suptitle(f"Full-Audio FFT ({fft_crop_low}–{fft_crop_high} Hz)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_word_vectors(
    dataset_dir, output_path, dwv_low=PAPER_DWV_LOW, dwv_high=PAPER_DWV_HIGH, **proc_kwargs
):
    samples = _load_one_sample_per_class(dataset_dir)
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), squeeze=False)
    axes_flat = axes.flatten()
    for idx, (cls, sig) in enumerate(samples.items()):
        _, _, _, _, word_slice = _process_sample(sig, dwv_low=dwv_low, dwv_high=dwv_high, **proc_kwargs)
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
    dataset_dir, output_path, dwv_low=PAPER_DWV_LOW, dwv_high=PAPER_DWV_HIGH,
    matrix_size=MATRIX_SIZE, **proc_kwargs
):
    samples = _load_one_sample_per_class(dataset_dir)
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (cls, sig) in zip(axes, samples.items()):
        _, _, _, _, word_slice = _process_sample(sig, dwv_low=dwv_low, dwv_high=dwv_high, **proc_kwargs)
        expected = matrix_size * matrix_size
        if len(word_slice) < expected:
            padded = np.zeros(expected)
            padded[: len(word_slice)] = word_slice
            word_slice = padded
        else:
            word_slice = word_slice[:expected]
        matrix_a = word_slice.reshape(matrix_size, matrix_size)
        ax.imshow(matrix_a, cmap="gray", aspect="equal", interpolation="nearest")
        ax.set_title(cls, fontsize=9)
        ax.axis("off")
    plt.suptitle(f"Acoustic Images ({matrix_size}×{matrix_size})", fontsize=13)
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
        "--test_size",
        type=float,
        default=0.15,
        help="Test set fraction (default: 0.15)",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Validation set fraction (default: 0.15)",
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
        "--filter_low",
        type=int,
        default=BANDPASS_LOW,
        help="Low-pass/bandpass lower cutoff frequency in Hz (default: %(default)s)",
    )
    parser.add_argument(
        "--filter_high",
        type=int,
        default=BANDPASS_HIGH,
        help="Low-pass upper cutoff frequency in Hz (default: %(default)s)",
    )
    parser.add_argument(
        "--filter_order",
        type=int,
        default=5,
        help="Butterworth filter order (default: %(default)s)",
    )
    parser.add_argument(
        "--fft_crop_low",
        type=int,
        default=FFT_CROP_LOW,
        help="FFT crop lower bound in Hz (default: %(default)s)",
    )
    parser.add_argument(
        "--fft_crop_high",
        type=int,
        default=FFT_CROP_HIGH,
        help="FFT crop upper bound in Hz (default: %(default)s)",
    )
    parser.add_argument(
        "--word_k",
        type=float,
        default=WORD_CODING_K,
        help="Word coding discretization step k (default: %(default)s)",
    )
    parser.add_argument(
        "--matrix_size",
        type=int,
        default=MATRIX_SIZE,
        help="Square matrix side length; bins = matrix_size^2 (default: %(default)s)",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=224,
        help="Resized output image dimension in px (default: %(default)s)",
    )
    parser.add_argument(
        "--segment_duration",
        type=float,
        default=1.0,
        help="Segment length in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--auto_range",
        action="store_true",
        help="Auto-detect DWV range from dataset using current parameters",
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
    parser.add_argument(
        "--target_sr",
        type=int,
        default=TARGET_SR,
        help="Target sample rate in Hz for resampling (default: %(default)s)",
    )
    parser.add_argument(
        "--no_resample",
        action="store_true",
        help="Do not resample audio; use native sample rate of each file. "
             "Overrides --target_sr with the first file's native SR.",
    )
    args = parser.parse_args()

    target_sr = args.target_sr
    if args.no_resample:
        detected_sr = _detect_native_sr(Path(args.dataset_dir))
        if detected_sr is not None:
            target_sr = detected_sr
            logger.info("--no_resample: using native sample rate %d Hz", target_sr)
        else:
            logger.warning("--no_resample but no WAV files found; using --target_sr %d", target_sr)

    segment_length = int(target_sr * args.segment_duration)

    proc_kwargs = dict(
        filter_low=args.filter_low,
        filter_high=args.filter_high,
        filter_order=args.filter_order,
        fft_crop_low=args.fft_crop_low,
        fft_crop_high=args.fft_crop_high,
        word_k=args.word_k,
        segment_length=segment_length,
        target_sr=target_sr,
        no_resample=args.no_resample,
    )

    if args.auto_range:
        logger.info("Running DWV analysis on dataset with current parameters...")
        auto_low, auto_high = _compute_auto_dwv_range(
            args.dataset_dir, args.matrix_size, **proc_kwargs
        )
        if auto_low is not None and auto_high is not None:
            args.dwv_range_low = auto_low
            args.dwv_range_high = auto_high
        else:
            logger.warning(
                "DWV analysis failed – falling back to --dwv_range_low/high"
            )

    dwv_low = args.dwv_range_low
    dwv_high = args.dwv_range_high

    if dwv_low < args.fft_crop_low or dwv_high > args.fft_crop_high:
        logger.warning(
            "DWV range [%d-%d Hz] is OUTSIDE FFT crop [%d-%d Hz]. "
            "Clamping to crop bounds. Set --dwv_range_low/high within the crop range.",
            dwv_low, dwv_high, args.fft_crop_low, args.fft_crop_high,
        )
        dwv_low = max(dwv_low, args.fft_crop_low)
        dwv_high = min(dwv_high, args.fft_crop_high)

    logger.info("Using DWV range: %d–%d Hz", dwv_low, dwv_high)
    logger.info(
        "Parameters: target_sr=%d Hz, resample=%s, filter=%d-%d Hz (order %d), fft_crop=%d-%d Hz, k=%.4f, matrix=%dx%d, segment=%.1fs",
        target_sr, not args.no_resample,
        args.filter_low, args.filter_high, args.filter_order,
        args.fft_crop_low, args.fft_crop_high, args.word_k,
        args.matrix_size, args.matrix_size, args.segment_duration,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    # Each run gets a unique timestamped folder
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory for this run: %s", output_dir)

    # ------------------------------------------------------------------
    # Generate paper figures
    # ------------------------------------------------------------------
    if args.generate_plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = Path(args.dataset_dir)

        logger.info("Generating paper figures...")
        plot_time_domain(dataset_dir, plots_dir / "01_time_domain_signals.png",
                         target_sr=target_sr, no_resample=args.no_resample)
        logger.info("  Saved time-domain signals")
        plot_fft_spectra(
            dataset_dir, plots_dir / "02_fft_spectra.png",
            **proc_kwargs,
        )
        logger.info("  Saved FFT spectra")
        plot_fft_full_audio(
            dataset_dir, plots_dir / "02b_fft_full_audio.png",
            **proc_kwargs,
        )
        logger.info("  Saved full-audio FFT spectra")
        plot_word_vectors(
            dataset_dir, plots_dir / "03_word_vector_maps.png", dwv_low, dwv_high,
            **proc_kwargs,
        )
        logger.info("  Saved word vector maps")
        plot_acoustic_images(
            dataset_dir, plots_dir / "04_acoustic_images.png", dwv_low, dwv_high,
            matrix_size=args.matrix_size, **proc_kwargs,
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
        logger.info(
            "Image directory not found: %s – auto-generating from dataset...",
            image_dir,
        )
        image_dir.mkdir(parents=True, exist_ok=True)
        ok = _run_pipeline(
            image_dir, args.dataset_dir, dwv_low, dwv_high,
            args.matrix_size, args.output_size,
            **proc_kwargs,
        )
        if not ok:
            logger.error("Auto-generation failed")
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

    train_idx, val_idx, test_idx = split_dataset(
        targets, args.test_size, args.val_size, args.seed
    )
    logger.info(
        "Split: %d train, %d val, %d test (%.0f%%/%.0f%%/%.0f%%)",
        len(train_idx), len(val_idx), len(test_idx),
        100 * len(train_idx) / len(targets),
        100 * len(val_idx) / len(targets),
        100 * len(test_idx) / len(targets),
    )

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(eval_ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
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
            val_loader,
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
        _, val_er, val_mer, _ = evaluate_model(model, val_loader, criterion, device)
        _, test_er, test_mer, test_cm = evaluate_model(model, test_loader, criterion, device)
        plot_confusion_matrix(
            test_cm, class_names, model_name, results_dir / f"{model_name}_confusion.png"
        )

        all_results[model_name] = {
            "val_er": val_er,
            "val_mer": val_mer,
            "test_er": test_er,
            "test_mer": test_mer,
            "cm": test_cm,
            "history": history,
        }

        logger.info(
            "[%s] Final — Val ER: %.1f%%  Val MER: %.1f%%  |  Test ER: %.1f%%  Test MER: %.1f%%",
            model_name, val_er, val_mer, test_er, test_mer,
        )

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
                val_fraction=args.val_size,
            )

            test_ers = [f["test_er"] for f in folds]
            test_mers = [f["test_mer"] for f in folds]
            val_ers = [f["val_er"] for f in folds]
            val_mers = [f["val_mer"] for f in folds]
            kfold_results[model_name] = {
                "test_ers": test_ers, "test_mers": test_mers,
                "val_ers": val_ers, "val_mers": val_mers,
                "folds": folds,
            }

            logger.info(
                "[%s] K-fold  Val ER: %.1f ± %.1f%%  |  Test ER: %.1f ± %.1f%%",
                model_name,
                np.mean(val_ers), np.std(val_ers),
                np.mean(test_ers), np.std(test_ers),
            )

            np.savez(
                kf_dir / f"{model_name}_kfold.npz",
                test_ers=np.array(test_ers),
                test_mers=np.array(test_mers),
                val_ers=np.array(val_ers),
                val_mers=np.array(val_mers),
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
        fh.write(
            f"Train/Val/Test split: {len(train_idx)}/{len(val_idx)}/{len(test_idx)} "
            f"({100*len(train_idx)/len(targets):.0f}%/"
            f"{100*len(val_idx)/len(targets):.0f}%/"
            f"{100*len(test_idx)/len(targets):.0f}%)\n"
        )
        fh.write(
            f"Epochs: {args.epochs}   LR: {args.lr}   Batch: {args.batch_size}\n\n"
        )

        fh.write("-" * 50 + "\n")
        fh.write("Fixed-Split Results\n")
        fh.write("-" * 50 + "\n")
        for mn in args.models:
            r = all_results[mn]
            fh.write(
                f"  {mn:20s}  Val ER = {r['val_er']:5.1f}%  Val MER = {r['val_mer']:5.1f}%"
                f"  |  Test ER = {r['test_er']:5.1f}%  Test MER = {r['test_mer']:5.1f}%\n"
            )

        if kfold_results:
            fh.write("\n" + "-" * 50 + "\n")
            fh.write(f"K-fold CV Results ({args.kfold} folds)\n")
            fh.write("-" * 50 + "\n")
            for mn in args.models:
                kr = kfold_results[mn]
                fh.write(
                    f"  {mn:20s}  Val ER = {np.mean(kr['val_ers']):5.1f} ± {np.std(kr['val_ers']):4.1f}%"
                    f"  |  Test ER = {np.mean(kr['test_ers']):5.1f} ± {np.std(kr['test_ers']):4.1f}%\n"
                )

    logger.info("Summary report saved to %s", report_path)

    best_model_name = ""
    best_test_er = -1
    for mn, res in all_results.items():
        if res["test_er"] > best_test_er:
            best_test_er = res["test_er"]
            best_model_name = mn

    if best_model_name:
        res = all_results[best_model_name]
        model = create_model(best_model_name, num_classes)
        num_params = sum(p.numel() for p in model.parameters()) / 1e6

        print("\n---")
        print(f"val_acc:          {res['val_er']/100.0:.4f}  (Val ER)")
        print(f"val_mer:          {res['val_mer']/100.0:.4f}  (Val MER)")
        print(f"test_acc:         {res['test_er']/100.0:.4f}  (Test ER)")
        print(f"test_mer:         {res['test_mer']/100.0:.4f}  (Test MER)")
        print(f"num_params_M:     {num_params:.1f}")
        print(f"best_model:       {best_model_name}")
        print("---")

    logger.info("All done.")


if __name__ == "__main__":
    main()
