#!/usr/bin/env python3
"""
Aziz et al. Full Replication: Classification, Evaluation & Paper Figures

Replicates the bearing fault diagnosis pipeline end-to-end:
  - Train 12 classifiers: SVM-L/Q/C/G, KNN-1/W, RF, AdaB, NN-N/W/BL/TL
  - 10-fold stratified cross-validation
  - Metrics: Accuracy, Sensitivity, Specificity, PPV, NPV (per-class + macro)
  - SNR robustness chart (10, 20, 30, 40 dB + Original)
  - All 7 paper figures

Usage:
  python aziz_train.py --features_file output_paper2/selected_features.npz

  # Plots only (no training):
  python aziz_train.py --generate_plots --dataset_dir dataset

  # Full run with robustness test:
  python aziz_train.py --features_file output_paper2/selected_features.npz --robustness

  # Specific classifiers:
  python aziz_train.py --classifiers SVM-G RF NN-TL
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))
from paper2_pipeline import (
    MEL_HIGH,
    MEL_LOW,
    NFFT,
    NOTCH_FREQ,
    NUM_MEL_FILTERS,
    TARGET_SR,
    create_mel_filterbank,
    leemdr,
    load_audio,
    notch_filter,
    resample_signal,
    z_score_normalize,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ALL_CLASSIFIERS = [
    "SVM-L",
    "SVM-Q",
    "SVM-C",
    "SVM-G",
    "KNN-1",
    "KNN-W",
    "RF",
    "AdaB",
    "NN-N",
    "NN-W",
    "NN-BL",
    "NN-TL",
]


def make_classifier(name, seed=42):
    if name == "SVM-L":
        return SVC(kernel="linear", C=1.0)
    if name == "SVM-Q":
        return SVC(kernel="poly", degree=2, C=1.0)
    if name == "SVM-C":
        return SVC(kernel="poly", degree=3, C=1.0)
    if name == "SVM-G":
        return SVC(kernel="rbf", gamma="scale", C=1.0)
    if name == "KNN-1":
        return KNeighborsClassifier(n_neighbors=1)
    if name == "KNN-W":
        return KNeighborsClassifier(n_neighbors=5, weights="distance")
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed)
    if name == "AdaB":
        return AdaBoostClassifier(n_estimators=50, random_state=seed)
    if name == "NN-N":
        return MLPClassifier(
            hidden_layer_sizes=(10,),
            activation="relu",
            max_iter=1000,
            random_state=seed,
        )
    if name == "NN-W":
        return MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="relu",
            max_iter=1000,
            random_state=seed,
        )
    if name == "NN-BL":
        return MLPClassifier(
            hidden_layer_sizes=(100, 100),
            activation="relu",
            max_iter=1000,
            random_state=seed,
        )
    if name == "NN-TL":
        return MLPClassifier(
            hidden_layer_sizes=(100, 100, 100),
            activation="relu",
            max_iter=1000,
            random_state=seed,
        )
    raise ValueError(f"Unknown classifier: {name}")


def compute_metrics(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    total = cm.sum()
    per_class = {}
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        per_class[i] = {
            "accuracy": (tp + tn) / total if total > 0 else 0.0,
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "npv": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        }
    overall_acc = np.trace(cm) / total if total > 0 else 0.0
    macro = {}
    for m in ["accuracy", "sensitivity", "specificity", "ppv", "npv"]:
        macro[m] = np.mean([per_class[i][m] for i in range(n_classes)])
    return {
        "cm": cm,
        "overall_accuracy": overall_acc,
        "per_class": per_class,
        "macro": macro,
    }


def run_kfold_cv(clf_name, features, labels, n_classes, k=10, seed=42):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(features, labels)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(features[tr_idx])
        X_te = scaler.transform(features[te_idx])
        clf = make_classifier(clf_name, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, labels[tr_idx])
        y_pred = clf.predict(X_te)
        m = compute_metrics(labels[te_idx], y_pred, n_classes)
        fold_metrics.append(m)
        if (fold + 1) % 5 == 0:
            logger.info(
                "    %s fold %d/%d  acc=%.2f%%",
                clf_name,
                fold + 1,
                k,
                m["overall_accuracy"] * 100,
            )
    agg_cm = sum(m["cm"] for m in fold_metrics)
    avg_acc = np.mean([m["overall_accuracy"] for m in fold_metrics])
    avg_macro = {
        k_: np.mean([m["macro"][k_] for m in fold_metrics])
        for k_ in fold_metrics[0]["macro"]
    }
    per_class_acc = np.array(
        [
            [m["per_class"][c]["accuracy"] for m in fold_metrics]
            for c in range(n_classes)
        ]
    ).mean(axis=1)
    return {
        "agg_cm": agg_cm,
        "avg_accuracy": avg_acc,
        "avg_macro": avg_macro,
        "per_class_acc": per_class_acc,
        "fold_metrics": fold_metrics,
    }


def add_noise_at_snr(features, snr_db, seed=42):
    rng = np.random.RandomState(seed)
    noisy = features.copy()
    for col in range(features.shape[1]):
        sig_pow = np.var(features[:, col])
        if sig_pow > 0:
            noise_pow = sig_pow / (10 ** (snr_db / 10))
            noisy[:, col] += rng.randn(features.shape[0]) * np.sqrt(noise_pow)
    return noisy


def _load_one_per_class(dataset_dir, max_seconds=5):
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
        if not wavs:
            continue
        sig, sr = load_audio(str(wavs[0]))
        sig = resample_signal(sig, sr, TARGET_SR)
        n_samples = int(max_seconds * TARGET_SR)
        samples[cd.name] = sig[:n_samples]
    return samples


def _preprocess_for_plot(signal):
    normed = z_score_normalize(signal)
    filtered = notch_filter(normed, NOTCH_FREQ, TARGET_SR)
    return filtered


def plot_raw_signals(dataset_dir, output_path):
    samples = _load_one_per_class(dataset_dir, max_seconds=5)
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), squeeze=False)
    for idx, (cls, sig) in enumerate(samples.items()):
        t = np.arange(len(sig)) / TARGET_SR
        axes[idx, 0].plot(t, sig, linewidth=0.3, color="steelblue")
        axes[idx, 0].set_title(cls)
        axes[idx, 0].set_xlabel("Time (s)")
        axes[idx, 0].set_ylabel("Amplitude")
    plt.suptitle("Raw Sound Signals (0–5 s)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_imf_visualization(dataset_dir, output_path):
    samples = _load_one_per_class(dataset_dir, max_seconds=1)
    if not samples:
        return

    for cls_idx, (cls, sig) in enumerate(samples.items()):
        preprocessed = _preprocess_for_plot(sig)
        logger.info("  [2/7] IMF: decomposing %s ...", cls)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from PyEMD import EMD as _EMD

            emd = _EMD()
            emd.FIXE = 5
            emd.MAX_ITERATION = 30
            emd.MAX_IMFS = 10
            imfs = emd(preprocessed, max_imf=8)

        n_imfs = imfs.shape[0] if imfs.ndim == 2 else 1
        ds = max(1, len(preprocessed) // 2000)
        t = np.arange(0, len(preprocessed), ds) / TARGET_SR

        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection="3d")
        for i in range(n_imfs):
            imf_ds = imfs[i, ::ds] if imfs.ndim == 2 else preprocessed[::ds]
            ax.plot(np.full_like(t, i), t, imf_ds, linewidth=0.5)
        ax.set_xlabel("Mode Number")
        ax.set_ylabel("Time (s)")
        ax.set_zlabel("Amplitude")
        ax.set_title(f"IMF Decomposition – {cls}")
        ax.view_init(elev=20.17061125542267, azim=65.07654988838055)

        stem = output_path.stem
        cls_path = output_path.with_name(
            f"{stem}_{cls.replace('+', 'plus').replace(' ', '_')}.png"
        )
        plt.tight_layout()
        plt.savefig(cls_path, dpi=150, bbox_inches="tight")
        plt.close()


def plot_preprocessed_signals(dataset_dir, output_path):
    samples = _load_one_per_class(dataset_dir, max_seconds=5)
    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), squeeze=False)
    for idx, (cls, sig) in enumerate(samples.items()):
        preprocessed = _preprocess_for_plot(sig)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            denoised = leemdr(preprocessed)
        t = np.arange(len(denoised)) / TARGET_SR
        axes[idx, 0].plot(t, denoised, linewidth=0.3, color="darkorange")
        axes[idx, 0].set_title(cls)
        axes[idx, 0].set_xlabel("Time (s)")
        axes[idx, 0].set_ylabel("Amplitude")
    plt.suptitle("LEEMDR Preprocessed Signals p(t) (0–5 s)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mel_filterbank(output_path):
    mel_fb = create_mel_filterbank(NUM_MEL_FILTERS, NFFT, TARGET_SR, MEL_LOW, MEL_HIGH)
    num_bins = mel_fb.shape[1]
    freqs = np.linspace(MEL_LOW, MEL_HIGH, num_bins)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(NUM_MEL_FILTERS):
        ax.plot(freqs, mel_fb[i], linewidth=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Triangular Mel Filter Bank ({NUM_MEL_FILTERS} filters, 0–5000 Hz)")
    ax.set_xlim(MEL_LOW, MEL_HIGH)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_boxplots(features, labels, class_names, output_path, n_feats=6):
    n_classes = len(class_names)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for idx in range(min(n_feats, features.shape[1])):
        ax = axes[idx // 3, idx % 3]
        data = [features[labels == c, idx] for c in range(n_classes)]
        bp = ax.boxplot(data, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_title(f"MMFCC {idx + 1}")
        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Value")
    plt.suptitle("MMFCC Distributions Across Classes", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, class_names, clf_name, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Actual",
        xlabel="Predicted",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
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
    ax.set_title(f"Confusion Matrix – {clf_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_robustness_snr(
    features, labels, n_classes, class_names, snr_levels, output_path, seed=42
):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(skf.split(features, labels))
    conditions = [f"{s} dB" for s in snr_levels] + ["Original"]
    n_cond = len(conditions)

    overall_accs = np.zeros(n_cond)
    class_accs = np.zeros((n_classes, n_cond))

    for ci, cond in enumerate(conditions):
        if cond == "Original":
            noisy = features.copy()
        else:
            snr = float(cond.replace(" dB", ""))
            noisy = add_noise_at_snr(features, snr, seed)

        fold_accs = []
        fold_class = [[] for _ in range(n_classes)]
        for tr_idx, te_idx in splits:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(noisy[tr_idx])
            X_te = scaler.transform(noisy[te_idx])
            clf = SVC(kernel="rbf", gamma="scale", C=1.0)
            clf.fit(X_tr, labels[tr_idx])
            y_pred = clf.predict(X_te)
            cm = confusion_matrix(labels[te_idx], y_pred, labels=list(range(n_classes)))
            fold_accs.append(np.trace(cm) / cm.sum())
            for c in range(n_classes):
                mask = labels[te_idx] == c
                if mask.sum() > 0:
                    fold_class[c].append(np.sum((y_pred == c) & mask) / mask.sum())
                else:
                    fold_class[c].append(0.0)

        overall_accs[ci] = np.mean(fold_accs) * 100
        for c in range(n_classes):
            class_accs[c, ci] = np.mean(fold_class[c]) * 100
        logger.info("  SNR %s → acc=%.1f%%", cond, overall_accs[ci])

    x = np.arange(n_cond)
    width = 0.8 / (n_classes + 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width * n_classes / 2,
        overall_accs,
        width,
        label="Overall",
        color="steelblue",
        edgecolor="black",
    )
    colors = plt.cm.Set2(np.linspace(0.2, 1, n_classes))
    for c in range(n_classes):
        ax.bar(
            x - width * n_classes / 2 + (c + 1) * width,
            class_accs[c],
            width,
            label=class_names[c],
            color=colors[c],
            edgecolor="black",
        )
    ax.set_xlabel("SNR Condition")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Robustness Chart: Accuracy vs. SNR (SVM-G)")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Aziz et al. Replication – Classification & Evaluation"
    )
    parser.add_argument(
        "--features_file",
        type=str,
        default="output_paper2/selected_features.npz",
        help="Path to GA-selected features (from paper2_pipeline.py)",
    )
    parser.add_argument(
        "--all_features_file",
        type=str,
        default="output_paper2/all_mmfcc_features.npz",
        help="Fallback: path to all 40 MMFCC features",
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
        default="output_aziz",
        help="Output directory",
    )
    parser.add_argument(
        "--kfold",
        type=int,
        default=10,
        help="Number of CV folds (paper: 10)",
    )
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        help="Generate all 7 paper figures",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=ALL_CLASSIFIERS,
        choices=ALL_CLASSIFIERS,
        help="Which classifiers to train",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Run SNR robustness test",
    )
    parser.add_argument(
        "--snr_levels",
        nargs="+",
        type=float,
        default=[10, 20, 30, 40],
        help="SNR levels (dB) for robustness chart",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Generate paper figures
    # ------------------------------------------------------------------
    if args.generate_plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        ds_dir = Path(args.dataset_dir)

        logger.info("Generating paper figures...")
        plot_raw_signals(ds_dir, plots_dir / "01_raw_signals.png")
        logger.info("  [1/7] Raw signals")
        plot_imf_visualization(ds_dir, plots_dir / "02_imf_visualization.png")
        logger.info("  [2/7] IMF visualization")
        plot_preprocessed_signals(ds_dir, plots_dir / "03_preprocessed_signals.png")
        logger.info("  [3/7] Preprocessed signals")
        plot_mel_filterbank(plots_dir / "04_mel_filterbank.png")
        logger.info("  [4/7] Mel filter bank")

        sel_path = Path(args.features_file)
        all_path = Path(args.all_features_file)
        if sel_path.exists():
            data = np.load(sel_path, allow_pickle=True)
        elif all_path.exists():
            data = np.load(all_path, allow_pickle=True)
        else:
            logger.info("No features file – remaining plots need features. Done.")
            return

        features = data["features"]
        labels = data["labels"].astype(int)
        class_names = list(data["class_names"])
        n_classes = len(class_names)
        plot_feature_boxplots(
            features, labels, class_names, plots_dir / "05_feature_boxplots.png"
        )
        logger.info("  [5/7] Feature box plots")

        cm_dir = output_dir / "confusion_matrices"
        cm_dir.mkdir(parents=True, exist_ok=True)
        for clf_name in args.classifiers:
            result = run_kfold_cv(
                clf_name, features, labels, n_classes, k=args.kfold, seed=args.seed
            )
            plot_confusion_matrix(
                result["agg_cm"],
                class_names,
                clf_name,
                cm_dir / f"{clf_name}_cm.png",
            )
        logger.info("  [6/7] Confusion matrices")

        if args.robustness:
            plot_robustness_snr(
                features,
                labels,
                n_classes,
                class_names,
                args.snr_levels,
                plots_dir / "07_robustness_snr.png",
                seed=args.seed,
            )
            logger.info("  [7/7] Robustness chart")
        else:
            logger.info("  [7/7] Skipped (use --robustness to enable)")

        logger.info("Plots done. Use without --generate_plots to run training.")
        return

    # ------------------------------------------------------------------
    # Load features
    # ------------------------------------------------------------------
    sel_path = Path(args.features_file)
    all_path = Path(args.all_features_file)

    if sel_path.exists():
        data = np.load(sel_path, allow_pickle=True)
        features = data["features"]
        labels = data["labels"].astype(int)
        class_names = list(data["class_names"])
        logger.info(
            "Loaded GA-selected features: %s (%d samples)", sel_path, len(features)
        )
    elif all_path.exists():
        data = np.load(all_path, allow_pickle=True)
        features = data["features"]
        labels = data["labels"].astype(int)
        class_names = list(data["class_names"])
        logger.info(
            "Loaded all MMFCC features: %s (%d samples)", all_path, len(features)
        )
    else:
        if args.generate_plots:
            logger.info("No features file – plots only.")
            return
        logger.error(
            "No features file found. Run paper2_pipeline.py first:\n"
            "  python paper2_pipeline.py --dataset_dir dataset --output_dir output_paper2"
        )
        sys.exit(1)

    n_classes = len(class_names)
    logger.info(
        "Dataset: %d samples, %d features, %d classes: %s",
        features.shape[0],
        features.shape[1],
        n_classes,
        class_names,
    )

    # ------------------------------------------------------------------
    # Train & evaluate all classifiers with K-fold CV
    # ------------------------------------------------------------------
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for clf_name in args.classifiers:
        logger.info("=" * 60)
        logger.info(
            "%s  –  %d-fold CV  (%d samples, %d features)",
            clf_name,
            args.kfold,
            len(features),
            features.shape[1],
        )
        logger.info("=" * 60)

        result = run_kfold_cv(
            clf_name, features, labels, n_classes, k=args.kfold, seed=args.seed
        )
        all_results[clf_name] = result

        plot_confusion_matrix(
            result["agg_cm"],
            class_names,
            clf_name,
            cm_dir / f"{clf_name}_cm.png",
        )

        logger.info(
            "  %s  Accuracy=%.2f%%  Sens=%.2f%%  Spec=%.2f%%  PPV=%.2f%%  NPV=%.2f%%",
            clf_name,
            result["avg_accuracy"] * 100,
            result["avg_macro"]["sensitivity"] * 100,
            result["avg_macro"]["specificity"] * 100,
            result["avg_macro"]["ppv"] * 100,
            result["avg_macro"]["npv"] * 100,
        )

    # ------------------------------------------------------------------
    # SNR Robustness
    # ------------------------------------------------------------------
    if args.robustness:
        logger.info("=" * 60)
        logger.info("SNR Robustness Test (SVM-G)")
        logger.info("=" * 60)
        plot_robustness_snr(
            features,
            labels,
            n_classes,
            class_names,
            args.snr_levels,
            output_dir / "plots" / "07_robustness_snr.png",
            seed=args.seed,
        )

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    report_path = output_dir / "summary_report.txt"
    with open(report_path, "w") as fh:
        fh.write("Aziz et al. Replication – Summary Report\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(f"Features: {features.shape[1]}-dimensional\n")
        fh.write(f"Samples: {features.shape[0]}\n")
        fh.write(f"Classes ({n_classes}): {class_names}\n")
        fh.write(f"Validation: {args.kfold}-fold stratified CV\n\n")

        fh.write("-" * 80 + "\n")
        fh.write(
            f"{'Classifier':>10}  {'Acc%':>7}  {'Sens%':>7}  {'Spec%':>7}"
            f"  {'PPV%':>7}  {'NPV%':>7}\n"
        )
        fh.write("-" * 80 + "\n")
        for name in args.classifiers:
            r = all_results[name]
            fh.write(
                f"{name:>10}  {r['avg_accuracy'] * 100:7.2f}  "
                f"{r['avg_macro']['sensitivity'] * 100:7.2f}  "
                f"{r['avg_macro']['specificity'] * 100:7.2f}  "
                f"{r['avg_macro']['ppv'] * 100:7.2f}  "
                f"{r['avg_macro']['npv'] * 100:7.2f}\n"
            )
        fh.write("-" * 80 + "\n")

        fh.write("\nPer-Class Accuracy (%):\n")
        for name in args.classifiers:
            r = all_results[name]
            per_cls = "  ".join(
                f"{class_names[i]}={r['per_class_acc'][i] * 100:.1f}"
                for i in range(n_classes)
            )
            fh.write(f"  {name:>10} : {per_cls}\n")

    logger.info("Summary report saved to %s", report_path)
    logger.info("All done.")


if __name__ == "__main__":
    main()
