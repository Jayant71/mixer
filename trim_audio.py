#!/usr/bin/env python3
"""
Trim the first 1 second and last 1 second from WAV files in a specific class folder.
Overwrites files in place.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def trim_wav(filepath, trim_start_sec=1.0, trim_end_sec=1.0):
    signal, sr = sf.read(str(filepath))
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    start_sample = int(trim_start_sec * sr)
    end_sample = len(signal) - int(trim_end_sec * sr)

    if end_sample <= start_sample:
        print(f"  SKIP (too short after trim): {filepath}")
        return False

    trimmed = signal[start_sample:end_sample]
    sf.write(str(filepath), trimmed, sr)
    duration = len(trimmed) / sr
    print(f"  Trimmed {filepath.name}: {len(signal)/sr:.2f}s -> {duration:.2f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Trim audio files in a dataset class folder")
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--class_name", type=str, default="healthy")
    parser.add_argument("--trim_start", type=float, default=1.0, help="Seconds to trim from start")
    parser.add_argument("--trim_end", type=float, default=1.0, help="Seconds to trim from end")
    args = parser.parse_args()

    class_dir = Path(args.dataset_dir) / args.class_name
    if not class_dir.exists():
        print(f"Class folder not found: {class_dir}")
        return

    wav_files = sorted(f for f in class_dir.iterdir() if f.suffix.lower() == ".wav")
    if not wav_files:
        print(f"No WAV files in {class_dir}")
        return

    print(f"Trimming {len(wav_files)} files in {class_dir} (start={args.trim_start}s, end={args.trim_end}s)")
    trimmed, skipped = 0, 0
    for f in wav_files:
        if trim_wav(f, args.trim_start, args.trim_end):
            trimmed += 1
        else:
            skipped += 1
    print(f"Done: {trimmed} trimmed, {skipped} skipped")


if __name__ == "__main__":
    main()
