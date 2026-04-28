#!/usr/bin/env python3
"""
Paper 2 Pipeline: Bearing Faults Classification using LEEMDR and MMFCCs
Replication of Aziz et al.'s pipeline for extracting 24-dimension feature
vectors from acoustic signals for SVM-based fault classification.

Steps replicated:
  1. Data Acquisition & Formatting (resample 10 kHz, 10s segments,
     z-score normalize, Butterworth notch filter)
  2. LEEMDR Denoising (EMD → RLE thresholding → reconstruct)
  3. MMFCC Extraction (pre-emphasis → windowing → DFT → power spectrum →
     Mel filter bank 0-5 kHz → log → DCT → 40 coefficients)
  4. GA Feature Selection (40 MMFCCs → 24 features via genetic algorithm)
"""

import argparse
import logging
import sys
from pathlib import Path

import warnings

import numpy as np
import scipy.fft
import scipy.signal
import soundfile as sf

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from PyEMD import EMD

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TARGET_SR = 10000
SEGMENT_DURATION = 10.0
SEGMENT_LENGTH = int(TARGET_SR * SEGMENT_DURATION)
NOTCH_FREQ = 50
NOTCH_Q = 30
RLE_THRESHOLD = 0.01
PRE_EMPHASIS_COEFF = 0.97
FRAME_SIZE = 512
HOP_SIZE = FRAME_SIZE // 2
NFFT = 512
NUM_MEL_FILTERS = 40
MEL_LOW = 0
MEL_HIGH = 5000
NUM_MMFCC = 40
GA_NUM_CHROMOSOMES = 10
GA_MAX_GENERATIONS = 50
GA_CROSSOVER_RATE = 0.8
GA_MUTATION_RATE = 0.1
GA_NUM_SELECTED = 24
EMD_MAX_IMFS = 8
EMD_MAX_SIFT_ITER = 30


def load_audio(filepath):
    signal, sr = sf.read(filepath)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    return signal.astype(np.float64), sr


def resample_signal(signal, orig_sr, target_sr):
    if orig_sr == target_sr:
        return signal
    num_samples = int(len(signal) * target_sr / orig_sr)
    return scipy.signal.resample(signal, num_samples)


def segment_signal(signal, segment_length):
    num_segments = len(signal) // segment_length
    segments = []
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segments.append(signal[start:end].copy())
    return segments


def z_score_normalize(signal):
    std = np.std(signal)
    if std < 1e-10:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


def notch_filter(signal, freq, fs, quality_factor=NOTCH_Q):
    b, a = scipy.signal.iirnotch(freq, quality_factor, fs)
    return scipy.signal.filtfilt(b, a, signal)


def leemdr(signal, rle_threshold=RLE_THRESHOLD):
    """
    Log Energy-based Empirical Mode Decomposition and Reconstruction.

    1. Decompose signal into IMFs via EMD.
    2. Compute RLE_j = log(sum|IMF_j|^2) / log(sum|v|^2) for each IMF.
    3. Keep IMFs with RLE_j > threshold (default 1%).
    4. Reconstruct by summing surviving IMFs.
    """
    emd = EMD()
    emd.FIXE = 5
    emd.MAX_ITERATION = EMD_MAX_SIFT_ITER
    emd.MAX_IMFS = EMD_MAX_IMFS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imfs = emd(signal, max_imf=EMD_MAX_IMFS)

    signal_energy = np.sum(signal**2)
    if signal_energy <= 0:
        return signal
    log_signal_energy = np.log(signal_energy)

    reconstructed = np.zeros_like(signal)
    kept_count = 0
    for imf in imfs:
        imf_energy = np.sum(imf**2)
        if imf_energy <= 0:
            continue
        rle = np.log(imf_energy) / log_signal_energy
        if rle > rle_threshold:
            reconstructed += imf
            kept_count += 1

    if kept_count == 0:
        logger.warning(
            "No IMFs passed RLE threshold; using all IMFs for reconstruction."
        )
        reconstructed = np.sum(imfs, axis=0)

    return reconstructed


def pre_emphasis(signal, coeff=PRE_EMPHASIS_COEFF):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(num_filters, nfft, fs, low_freq, high_freq):
    """
    Create triangular Mel-spaced filter bank.

    Returns array of shape (num_filters, nfft // 2 + 1).
    """
    mel_low = hz_to_mel(low_freq)
    mel_high = hz_to_mel(high_freq)
    mel_points = np.linspace(mel_low, mel_high, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((nfft + 1) * hz_points / fs).astype(int)

    num_bins = nfft // 2 + 1
    filterbank = np.zeros((num_filters, num_bins))

    for i in range(num_filters):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        for k in range(left, center):
            if center > left:
                filterbank[i, k] = (k - left) / (center - left)

        for k in range(center, right):
            if right > center:
                filterbank[i, k] = (right - k) / (right - center)

    return filterbank


def extract_mmfcc(
    signal,
    fs,
    num_mfcc=NUM_MMFCC,
    frame_size=FRAME_SIZE,
    hop_size=HOP_SIZE,
    nfft=NFFT,
    num_filters=NUM_MEL_FILTERS,
    low_freq=MEL_LOW,
    high_freq=MEL_HIGH,
    pre_emph_coeff=PRE_EMPHASIS_COEFF,
):
    """
    Extract Machine Mel-Frequency Cepstral Coefficients (MMFCCs).

    Steps A-G from the paper:
      A. Pre-emphasis filtering
      B. Windowing (Hamming, half-frame overlap)
      C. DFT
      D. Power spectrum  P_f(k) = (1/M)|X_f(k)|^2
      E. Mel filter bank (triangular, 0-5000 Hz)
      F. Log spectrum   S(c) = ln[sum P_f(k)*H_c(k)]
      G. DCT            C_M(n), n=1..O, O=40

    Returns the mean MMFCC vector across all frames (1D array of length num_mfcc).
    """
    emphasized = pre_emphasis(signal, pre_emph_coeff)

    mel_fb = create_mel_filterbank(num_filters, nfft, fs, low_freq, high_freq)
    window = np.hamming(frame_size)

    num_frames = max(0, (len(emphasized) - frame_size) // hop_size + 1)
    if num_frames == 0:
        return np.zeros(num_mfcc)

    all_mfccs = np.zeros((num_frames, num_mfcc))

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = emphasized[start:end] * window

        spectrum = np.fft.rfft(frame, n=nfft)

        power = (1.0 / nfft) * np.abs(spectrum) ** 2

        mel_energies = np.dot(mel_fb, power)
        mel_energies = np.maximum(mel_energies, np.finfo(float).eps)

        log_energies = np.log(mel_energies)

        dct_coeffs = scipy.fft.dct(log_energies, type=2, norm="ortho")[:num_mfcc]
        all_mfccs[i] = dct_coeffs

    return np.mean(all_mfccs, axis=0)


def ga_feature_selection(
    features,
    labels,
    num_features=NUM_MMFCC,
    num_selected=GA_NUM_SELECTED,
    num_chromosomes=GA_NUM_CHROMOSOMES,
    max_generations=GA_MAX_GENERATIONS,
    crossover_rate=GA_CROSSOVER_RATE,
    mutation_rate=GA_MUTATION_RATE,
):
    """
    Genetic Algorithm for feature selection.

    Each chromosome is a binary vector of length num_features.
    Constraint: exactly num_selected bits set to 1.
    Fitness: 5-fold CV accuracy of an RBF-SVM on the selected features.

    GA params from paper: 10 chromosomes, 50 generations,
    crossover rate 0.8, mutation rate 0.1.

    Returns (best_chromosome, best_fitness, fitness_history).
    """
    n_classes = len(set(labels))
    cv_folds = min(5, n_classes)

    def fitness(chromosome):
        selected_idx = np.where(chromosome == 1)[0]
        if len(selected_idx) < 2:
            return 0.0
        X = features[:, selected_idx]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        svm = SVC(kernel="rbf", gamma="scale")
        try:
            scores = cross_val_score(
                svm, X_scaled, labels, cv=cv_folds, scoring="accuracy"
            )
            return float(np.mean(scores))
        except Exception:
            return 0.0

    def repair(chromosome):
        chrom = chromosome.copy()
        ones = int(np.sum(chrom))
        while ones > num_selected:
            idx = np.random.choice(np.where(chrom == 1)[0])
            chrom[idx] = 0
            ones -= 1
        while ones < num_selected:
            idx = np.random.choice(np.where(chrom == 0)[0])
            chrom[idx] = 1
            ones += 1
        return chrom

    population = []
    for _ in range(num_chromosomes):
        chrom = np.zeros(num_features, dtype=int)
        selected = np.random.choice(num_features, num_selected, replace=False)
        chrom[selected] = 1
        population.append(chrom)

    best_chromosome = population[0].copy()
    best_fitness = -1.0
    fitness_history = []

    for gen in range(max_generations):
        fitnesses = np.array([fitness(chrom) for chrom in population])

        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_chromosome = population[gen_best_idx].copy()

        fitness_history.append((float(np.max(fitnesses)), float(np.mean(fitnesses))))

        logger.info(
            "  GA Gen %3d/%d  best=%.4f  mean=%.4f",
            gen + 1,
            max_generations,
            fitness_history[-1][0],
            fitness_history[-1][1],
        )

        new_population = [best_chromosome.copy()]

        while len(new_population) < num_chromosomes:
            i1, i2 = np.random.choice(num_chromosomes, 2, replace=False)
            parent1 = (
                population[i1] if fitnesses[i1] > fitnesses[i2] else population[i2]
            )
            i3, i4 = np.random.choice(num_chromosomes, 2, replace=False)
            parent2 = (
                population[i3] if fitnesses[i3] > fitnesses[i4] else population[i4]
            )

            if np.random.random() < crossover_rate:
                point = np.random.randint(1, num_features)
                child = np.concatenate([parent1[:point], parent2[point:]])
            else:
                child = parent1.copy()

            for j in range(num_features):
                if np.random.random() < mutation_rate:
                    child[j] = 1 - child[j]

            child = repair(child)
            new_population.append(child)

        population = new_population[:num_chromosomes]

    return best_chromosome, best_fitness, fitness_history


def main():
    parser = argparse.ArgumentParser(
        description="Paper 2 Pipeline – LEEMDR + MMFCC + GA Feature Selection"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Path to dataset directory (subfolders = classes)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_paper2",
        help="Path to output directory",
    )
    parser.add_argument(
        "--notch_freq",
        type=float,
        default=NOTCH_FREQ,
        help="Powerline notch filter frequency (Hz)",
    )
    parser.add_argument(
        "--pre_emph",
        type=float,
        default=PRE_EMPHASIS_COEFF,
        help="Pre-emphasis coefficient (0.9-1.0)",
    )
    parser.add_argument(
        "--skip_ga",
        action="store_true",
        help="Skip GA feature selection (save all 40 MMFCCs only)",
    )
    parser.add_argument(
        "--skip_leemdr",
        action="store_true",
        help="Skip LEEMDR denoising step",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    if not dataset_dir.exists():
        logger.error("Dataset directory not found: %s", dataset_dir)
        sys.exit(1)

    class_dirs = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )
    if not class_dirs:
        logger.error("No class directories found in %s", dataset_dir)
        sys.exit(1)

    logger.info("Found %d classes: %s", len(class_dirs), [d.name for d in class_dirs])

    # ------------------------------------------------------------------
    # Phase 1 – Data Acquisition & Basic Formatting
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 1: Data Acquisition & Basic Formatting")
    logger.info("=" * 60)

    all_features = []
    all_labels = []
    all_filenames = []
    class_names = sorted([d.name for d in class_dirs])
    label_map = {name: idx for idx, name in enumerate(class_names)}

    total_files = sum(
        len(
            [
                f
                for f in cd.iterdir()
                if f.suffix.lower() == ".wav" and not f.name.startswith(".")
            ]
        )
        for cd in class_dirs
    )
    processed = 0
    total_segments = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        wav_files = sorted(
            [
                f
                for f in class_dir.iterdir()
                if f.suffix.lower() == ".wav" and not f.name.startswith(".")
            ]
        )
        logger.info("Class '%s': %d WAV files", class_name, len(wav_files))

        for wav_idx, wav_file in enumerate(wav_files):
            try:
                signal, sr = load_audio(str(wav_file))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", wav_file, exc)
                continue

            processed += 1
            signal = resample_signal(signal, sr, TARGET_SR)
            segments = segment_signal(signal, SEGMENT_LENGTH)

            for seg_idx, segment in enumerate(segments):
                total_segments += 1
                logger.info(
                    "  [%d/%d] %s  seg %d/%d",
                    processed,
                    total_files,
                    wav_file.name,
                    seg_idx + 1,
                    len(segments),
                )
                normalized = z_score_normalize(segment)
                filtered = notch_filter(normalized, args.notch_freq, TARGET_SR)

                # ----------------------------------------------------------
                # Phase 2 – LEEMDR Denoising
                # ----------------------------------------------------------
                if not args.skip_leemdr:
                    try:
                        denoised = leemdr(filtered)
                    except Exception as exc:
                        logger.warning(
                            "LEEMDR failed on %s seg %d: %s – using filtered signal",
                            wav_file.name,
                            seg_idx,
                            exc,
                        )
                        denoised = filtered
                else:
                    denoised = filtered

                # ----------------------------------------------------------
                # Phase 3 – MMFCC Extraction
                # ----------------------------------------------------------
                mmfcc = extract_mmfcc(
                    denoised,
                    TARGET_SR,
                    pre_emph_coeff=args.pre_emph,
                )

                all_features.append(mmfcc)
                all_labels.append(label_map[class_name])
                all_filenames.append(f"{wav_file.stem}_seg{seg_idx:03d}")

        logger.info("  '%s' done", class_name)

    if len(all_features) == 0:
        logger.error("No features extracted. Exiting.")
        sys.exit(1)

    features_matrix = np.array(all_features)
    labels_array = np.array(all_labels)
    logger.info(
        "Extracted %d feature vectors of dimension %d",
        features_matrix.shape[0],
        features_matrix.shape[1],
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_dir / "all_mmfcc_features.npz",
        features=features_matrix,
        labels=labels_array,
        filenames=np.array(all_filenames),
        class_names=np.array(class_names),
    )
    logger.info("Saved all 40-dim MMFCCs to %s", output_dir / "all_mmfcc_features.npz")

    # ------------------------------------------------------------------
    # Phase 4 – GA Feature Selection
    # ------------------------------------------------------------------
    if not args.skip_ga:
        logger.info("=" * 60)
        logger.info(
            "Phase 4: GA Feature Selection (%d → %d)", NUM_MMFCC, GA_NUM_SELECTED
        )
        logger.info("=" * 60)

        best_chrom, best_fit, fit_history = ga_feature_selection(
            features_matrix, labels_array.tolist()
        )

        selected_indices = np.where(best_chrom == 1)[0]
        selected_features = features_matrix[:, selected_indices]

        np.savez(
            output_dir / "selected_features.npz",
            features=selected_features,
            labels=labels_array,
            filenames=np.array(all_filenames),
            class_names=np.array(class_names),
            selected_indices=selected_indices,
            best_fitness=best_fit,
        )
        np.save(output_dir / "ga_fitness_history.npy", np.array(fit_history))

        logger.info(
            "GA complete. Best fitness: %.4f  Selected features: %s",
            best_fit,
            selected_indices.tolist(),
        )

        report_path = output_dir / "ga_report.txt"
        with open(report_path, "w") as fh:
            fh.write("GA Feature Selection Report\n")
            fh.write("=" * 40 + "\n\n")
            fh.write(f"Input features: {NUM_MMFCC}\n")
            fh.write(f"Selected features: {GA_NUM_SELECTED}\n")
            fh.write(f"Chromosomes: {GA_NUM_CHROMOSOMES}\n")
            fh.write(f"Generations: {GA_MAX_GENERATIONS}\n")
            fh.write(f"Crossover rate: {GA_CROSSOVER_RATE}\n")
            fh.write(f"Mutation rate: {GA_MUTATION_RATE}\n\n")
            fh.write(f"Best fitness (CV accuracy): {best_fit:.4f}\n")
            fh.write(f"Selected feature indices: {selected_indices.tolist()}\n\n")
            fh.write("Fitness history (best, mean):\n")
            for gen, (best, mean) in enumerate(fit_history, 1):
                fh.write(f"  Gen {gen:3d}: best={best:.4f}  mean={mean:.4f}\n")

        logger.info("GA report saved to %s", report_path)

    logger.info("Pipeline complete. Output directory: %s", output_dir)


if __name__ == "__main__":
    main()
