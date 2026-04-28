#!/usr/bin/env python3
"""
Paper 1 Pipeline: Acoustic Fault Diagnosis of Induction Motors
Replication of Glowacz et al.'s audio transformation pipeline for converting
1D acoustic signals into 2D normalized images for CNN-based classification.

Steps replicated:
  1. Data Acquisition and Segmentation  (resample to 44100 Hz, 1-second segments)
  2. Pre-Filtering & Normalization      (low-pass 1225 Hz, amplitude normalization)
  3. FFT                                (magnitude spectrum, crop 1-1000 Hz)
  4. Word Coding                        (discretization with k=0.01)
  5. DWV & Region Selection             (discriminative frequency band identification)
  6. Acoustic Image Formation           (17x17 matrix -> 224x224x3, normalized)
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TARGET_SR = 44100
SEGMENT_LENGTH = TARGET_SR
BANDPASS_LOW = 1
BANDPASS_HIGH = 1225
FFT_CROP_LOW = 1
FFT_CROP_HIGH = 1000
WORD_CODING_K = 0.01
DWV_RANGE_LOW = 519
DWV_RANGE_HIGH = 807
MATRIX_SIZE = 17
OUTPUT_SIZE = 224
NUM_CHANNELS = 3
FILTER_ORDER = 5


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


def pre_filter(signal, low_freq, high_freq, fs, order=FILTER_ORDER):
    """
    Apply low-pass filter at high_freq Hz to the signal.

    The paper specifies a filter passband of 1-1225 Hz. Since a bandpass with
    a 1 Hz lower cutoff is numerically unstable (very low normalised frequency)
    and the subsequent FFT crop to 1-1000 Hz removes DC anyway, we use a simple
    low-pass at 1225 Hz which achieves the same effective result.
    """
    nyquist = fs / 2.0
    high_norm = high_freq / nyquist
    sos = scipy.signal.butter(order, high_norm, btype="low", output="sos")
    return scipy.signal.sosfiltfilt(sos, signal)


def normalize_amplitude(signal):
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val
    return signal


def compute_fft_magnitude(signal):
    N = len(signal)
    return np.abs(np.fft.rfft(signal)) * 2.0 / N


def crop_spectrum(spectrum, low_freq, high_freq, fs):
    N = (len(spectrum) - 1) * 2
    freq_resolution = fs / N
    low_bin = max(int(np.round(low_freq / freq_resolution)), 0)
    high_bin = min(int(np.round(high_freq / freq_resolution)), len(spectrum) - 1)
    cropped = spectrum[low_bin : high_bin + 1]
    base_freq = low_bin * freq_resolution
    return cropped, base_freq, freq_resolution


def word_coding(magnitudes, k=WORD_CODING_K):
    """
    Discretize magnitude spectrum into word vector.

    For each amplitude f_n > 0, the word coefficient is:
        D_p = ceil(f_n / k)
    where p indicates the bin index. Values <= 0 map to 0.
    """
    words = np.zeros_like(magnitudes, dtype=np.float64)
    mask = magnitudes > 0
    words[mask] = np.ceil(magnitudes[mask] / k)
    return words


def extract_dwv_range(word_vector, base_freq, freq_resolution, low_freq, high_freq):
    low_idx = int(np.round((low_freq - base_freq) / freq_resolution))
    high_idx = int(np.round((high_freq - base_freq) / freq_resolution))
    low_idx = max(low_idx, 0)
    high_idx = min(high_idx, len(word_vector) - 1)
    return word_vector[low_idx : high_idx + 1]


def compute_dwv_analysis(class_word_vectors):
    """
    Compute Differences of Word Vectors across all class pairs.

    For each pair of classes the absolute difference of their mean word vectors
    is computed. The element-wise maximum across all pairs is returned.
    """
    class_means = {}
    for cls, vectors in class_word_vectors.items():
        if len(vectors) > 0:
            class_means[cls] = np.mean(vectors, axis=0)

    classes = list(class_means.keys())
    if len(classes) < 2:
        if classes:
            return np.zeros_like(class_means[classes[0]])
        return np.array([])

    max_diff = np.zeros_like(class_means[classes[0]])
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            diff = np.abs(class_means[classes[i]] - class_means[classes[j]])
            max_diff = np.maximum(max_diff, diff)

    return max_diff


def find_optimal_range(dwv, range_size):
    cumsum = np.concatenate(([0], np.cumsum(dwv)))
    best_start = 0
    best_sum = -1.0
    for i in range(len(dwv) - range_size + 1):
        current_sum = cumsum[i + range_size] - cumsum[i]
        if current_sum > best_sum:
            best_sum = current_sum
            best_start = i
    return best_start, best_start + range_size - 1, best_sum


def create_acoustic_image(
    word_vector_slice,
    matrix_size=MATRIX_SIZE,
    output_size=OUTPUT_SIZE,
    num_channels=NUM_CHANNELS,
):
    """
    Create a 224x224x3 acoustic image from a word-vector frequency slice.

    Steps:
      1. Pad/trim to matrix_size^2 elements and reshape to (17, 17).
      2. Resize to (224, 224) via bilinear interpolation.
      3. Replicate across 3 channels.
      4. Normalize: W_norm = W / max(W).
    """
    expected = matrix_size * matrix_size
    if len(word_vector_slice) < expected:
        padded = np.zeros(expected)
        padded[: len(word_vector_slice)] = word_vector_slice
        word_vector_slice = padded
    elif len(word_vector_slice) > expected:
        word_vector_slice = word_vector_slice[:expected]

    matrix_a = word_vector_slice.reshape(matrix_size, matrix_size)

    img = Image.fromarray(matrix_a.astype(np.float32), mode="F")
    img_resized = img.resize((output_size, output_size), Image.BILINEAR)
    matrix_w = np.array(img_resized, dtype=np.float64)

    matrix_w = np.stack([matrix_w] * num_channels, axis=-1)

    max_val = np.max(matrix_w)
    if max_val > 0:
        matrix_w = matrix_w / max_val

    return matrix_w


def process_segment(segment, fs):
    """Run steps 2-4 on a single 1-second segment; return word vector + freq info."""
    filtered = pre_filter(segment, BANDPASS_LOW, BANDPASS_HIGH, fs)
    normalized = normalize_amplitude(filtered)
    spectrum = compute_fft_magnitude(normalized)
    cropped, base_freq, freq_res = crop_spectrum(
        spectrum, FFT_CROP_LOW, FFT_CROP_HIGH, fs
    )
    max_mag = np.max(cropped)
    if max_mag > 0:
        cropped = cropped / max_mag
    wv = word_coding(cropped, WORD_CODING_K)
    return wv, base_freq, freq_res


def save_image(matrix_w, output_path):
    img_uint8 = (matrix_w * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_uint8, mode="RGB")
    img.save(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Paper 1 Pipeline – Acoustic Image Generation"
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
        default="output_paper1",
        help="Path to output directory",
    )
    parser.add_argument(
        "--dwv_range_low",
        type=int,
        default=DWV_RANGE_LOW,
        help="DWV frequency range lower bound (Hz)",
    )
    parser.add_argument(
        "--dwv_range_high",
        type=int,
        default=DWV_RANGE_HIGH,
        help="DWV frequency range upper bound (Hz)",
    )
    parser.add_argument(
        "--auto_range",
        action="store_true",
        help="Auto-detect optimal frequency range via DWV analysis",
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="Also save images as .npz arrays for training",
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
    # Phase 1 – Load audio, segment, compute word vectors per segment
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 1: Processing audio files & computing word vectors")
    logger.info("=" * 60)

    all_word_vectors = defaultdict(list)
    segment_registry = []

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

        for wav_file in wav_files:
            try:
                signal, sr = load_audio(str(wav_file))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", wav_file, exc)
                continue

            signal = resample_signal(signal, sr, TARGET_SR)
            segments = segment_signal(signal, SEGMENT_LENGTH)

            for seg_idx, segment in enumerate(segments):
                try:
                    wv, base_freq, freq_res = process_segment(segment, TARGET_SR)
                except Exception as exc:
                    logger.warning(
                        "Failed on segment %d of %s: %s", seg_idx, wav_file.name, exc
                    )
                    continue

                all_word_vectors[class_name].append(wv)
                segment_registry.append(
                    (class_name, wav_file, seg_idx, wv, base_freq, freq_res)
                )

        logger.info(
            "  '%s': %d total segments", class_name, len(all_word_vectors[class_name])
        )

    logger.info("Total segments collected: %d", len(segment_registry))

    # ------------------------------------------------------------------
    # Phase 2 – DWV analysis
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 2: DWV Analysis")
    logger.info("=" * 60)

    dwv_output_dir = output_dir / "dwv_analysis"
    dwv_output_dir.mkdir(parents=True, exist_ok=True)

    dwv = compute_dwv_analysis(dict(all_word_vectors))

    auto_low = args.dwv_range_low
    auto_high = args.dwv_range_high

    if len(dwv) > 0:
        np.save(dwv_output_dir / "dwv_values.npy", dwv)
        logger.info("DWV max difference: %.4f", np.max(dwv))

        expected_components = MATRIX_SIZE * MATRIX_SIZE
        best_start, best_end, best_sum = find_optimal_range(dwv, expected_components)
        freq_resolution = TARGET_SR / SEGMENT_LENGTH
        auto_low = int(FFT_CROP_LOW + best_start * freq_resolution)
        auto_high = int(FFT_CROP_LOW + best_end * freq_resolution)
        logger.info(
            "Auto-detected optimal range: %d-%d Hz  (sum=%.4f)",
            auto_low,
            auto_high,
            best_sum,
        )
        logger.info(
            "Paper's specified range: %d-%d Hz", args.dwv_range_low, args.dwv_range_high
        )

        report_path = dwv_output_dir / "dwv_report.txt"
        with open(report_path, "w") as fh:
            fh.write("DWV Analysis Report\n")
            fh.write("=" * 40 + "\n")
            fh.write(f"Number of classes: {len(class_dirs)}\n")
            for cls in sorted(all_word_vectors.keys()):
                fh.write(f"  {cls}: {len(all_word_vectors[cls])} segments\n")
            fh.write(
                f"\nAuto-detected optimal range: {auto_low}-{auto_high} Hz "
                f"({best_end - best_start + 1} bins)\n"
            )
            fh.write(
                f"Paper's specified range: {args.dwv_range_low}-{args.dwv_range_high} Hz "
                f"({args.dwv_range_high - args.dwv_range_low + 1} bins)\n"
            )
            fh.write(f"\nMax DWV value: {np.max(dwv):.4f}\n")
            fh.write(f"Mean DWV value: {np.mean(dwv):.4f}\n")

    if args.auto_range:
        range_low = auto_low
        range_high = auto_high
        logger.info("Using AUTO-DETECTED range: %d-%d Hz", range_low, range_high)
    else:
        range_low = args.dwv_range_low
        range_high = args.dwv_range_high
        logger.info("Using PAPER range: %d-%d Hz", range_low, range_high)

    # ------------------------------------------------------------------
    # Phase 3 – Generate acoustic images
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 3: Generating acoustic images")
    logger.info("=" * 60)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    npz_dir = None
    if args.save_npz:
        npz_dir = output_dir / "npz"
        npz_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0

    for entry in segment_registry:
        class_name, source_file, seg_idx, wv, base_freq, freq_res = entry

        word_slice = extract_dwv_range(wv, base_freq, freq_res, range_low, range_high)
        acoustic_img = create_acoustic_image(word_slice)

        class_img_dir = images_dir / class_name
        class_img_dir.mkdir(parents=True, exist_ok=True)
        img_filename = f"{source_file.stem}_seg{seg_idx:03d}.png"
        save_image(acoustic_img, class_img_dir / img_filename)

        if npz_dir is not None:
            class_npz_dir = npz_dir / class_name
            class_npz_dir.mkdir(parents=True, exist_ok=True)
            npz_filename = f"{source_file.stem}_seg{seg_idx:03d}.npz"
            np.savez(
                class_npz_dir / npz_filename,
                image=acoustic_img,
                class_name=class_name,
                source=str(source_file),
                segment=seg_idx,
            )

        total_images += 1

    logger.info("Generated %d acoustic images in %s", total_images, images_dir)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
