#!/usr/bin/env python3
"""
Script to plot FFT transforms of one sample from each class in subplots.
Each subplot shows the magnitude spectrum up to the Nyquist frequency (fs/2).
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path

# Sampling rate
fs = 16000

# Classes
classes = [
    "elec+mech",
    "elec+therm",
    "electrical",
    "healthy",
    "mechanical-bearing",
    "mechanical-bush",
    "thermal",
]

# Dataset directory
dataset_dir = Path("dataset")

# Create figure with subplots
fig, axes = plt.subplots(len(classes), 1, figsize=(12, 18))

for i, cls in enumerate(classes):
    folder = dataset_dir / cls
    files = list(folder.glob("*.wav"))
    if not files:
        print(f"No files in {cls}")
        continue
    # Pick the first file
    file_path = files[0]
    print(f"Loading {file_path}")

    # Load audio
    signal, sr = sf.read(file_path)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)  # Convert to mono
    signal = signal.astype(np.float64)

    # If sr != fs, resample (though assuming all are 16000 Hz)
    if sr != fs:
        from scipy.signal import resample

        signal = resample(signal, int(len(signal) * fs / sr))

    # Compute FFT
    N = len(signal)
    freqs = np.fft.rfftfreq(N, 1 / fs)
    mag = np.abs(np.fft.rfft(signal)) * 2 / N  # Magnitude spectrum

    # Plot
    axes[i].plot(freqs, mag)
    axes[i].set_title(f"{cls} - Sample: {file_path.name} - Duration: {N / fs:.1f}s")
    axes[i].set_xlabel("Frequency (Hz)")
    axes[i].set_ylabel("Magnitude")
    axes[i].set_xlim(0, fs / 2)
    axes[i].grid(True)

plt.tight_layout()
plt.savefig("fft_samples_plot.png", dpi=300)
plt.show()
