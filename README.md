# Acoustic Fault Diagnosis – Mixer/Grinder Dataset

Replication of two acoustic fault diagnosis papers applied to a custom mixer/grinder dataset with 6 fault classes.

## Dataset Structure

```
dataset/
  elec+mech/           # Electrical + Mechanical combined fault
  elec+therm/          # Electrical + Thermal combined fault
  electrical/          # Electrical fault
  mechanical-bearing/  # Mechanical bearing fault
  mechanical-bush/     # Mechanical bush fault
  thermal/             # Thermal fault
```

Each class folder contains `.wav` audio recordings (originally 16 kHz, 30 seconds each).

---

## Paper 1: Glowacz et al. – CNN-Based Acoustic Image Classification

Converts 1D audio into 2D acoustic images for deep CNN classifiers (DenseNet-201, ResNet-18/50, EfficientNet-B0).

### Pipeline Scripts

| Script | Purpose |
|---|---|
| `paper1_pipeline.py` | Preprocessing: resample, segment, filter, FFT, word coding, acoustic image generation |
| `glowacz_train.py` | Train CNNs, K-fold CV, generate all 4 paper figures |

### Quick Start

```bash
# Step 1: Generate acoustic images from raw audio
python paper1_pipeline.py --dataset_dir dataset --output_dir output_paper1

# Step 2: Generate paper figures only (no training)
python glowacz_train.py --generate_plots --dataset_dir dataset

# Step 3: Train CNNs on the generated images
python glowacz_train.py --image_dir output_paper1/images --dataset_dir dataset --epochs 50

# Step 3 (alternative): Train with K-fold cross-validation
python glowacz_train.py --image_dir output_paper1/images --kfold 5
```

### Auto-Detected Frequency Range

The paper uses a fixed 519–807 Hz range for its induction motor data. Your dataset may have a different optimal range. Use `--auto_range` to let the DWV analysis find the best range:

```bash
# Generate images with auto-detected range
python paper1_pipeline.py --auto_range --dataset_dir dataset --output_dir output_paper1

# Use the same auto-detected range for plots and training
python glowacz_train.py --generate_plots --dataset_dir dataset --auto_range

# Or manually specify a range
python paper1_pipeline.py --dwv_range_low 100 --dwv_range_high 388 --dataset_dir dataset
python glowacz_train.py --generate_plots --dataset_dir dataset --dwv_range_low 100 --dwv_range_high 388
```

### Pipeline Steps (Paper 1)

1. **Resample** to 44,100 Hz → segment into 1-second chunks
2. **Low-pass filter** at 1–1,225 Hz + amplitude normalization
3. **FFT** magnitude spectrum, crop to 1–1,000 Hz, normalize
4. **Word coding** with k=0.01 (discretize amplitudes into integer coefficients)
5. **DWV analysis** to find the most discriminative frequency band (paper: 519–807 Hz)
6. **Acoustic image** formation: 289 components → 17×17 matrix → resize to 224×224×3 → normalize

### Paper Figures Generated

| # | Figure | File |
|---|---|---|
| 1 | Time-domain waveforms per class | `01_time_domain_signals.png` |
| 2 | FFT spectra (1–1000 Hz) bar graphs per class | `02_fft_spectra.png` |
| 3 | Word vector bar maps per class | `03_word_vector_maps.png` |
| 4 | 17×17 acoustic image grids per class | `04_acoustic_images.png` |

### Evaluation Metrics

- **ER** (Efficiency of Recognition): overall accuracy = correct / total × 100%
- **MER** (Mean Efficiency of Recognition): mean per-class accuracy

---

## Paper 2: Aziz et al. – LEEMDR + MMFCC Feature-Based Classification

Denoises audio via EMD, extracts 40 MMFCC features, reduces to 24 via genetic algorithm, then classifies with SVM/KNN/ensemble/ANN classifiers.

### Pipeline Scripts

| Script | Purpose |
|---|---|
| `paper2_pipeline.py` | Preprocessing: LEEMDR denoising, MMFCC extraction, GA feature selection |
| `aziz_train.py` | Train 12 classifiers, 10-fold CV, generate all 7 paper figures |

### Quick Start

```bash
# Step 1: Extract features (LEEMDR is slow, use --skip_leemdr to skip)
python paper2_pipeline.py --dataset_dir dataset --output_dir output_paper2

# Step 1 (faster, skip EMD):
python paper2_pipeline.py --dataset_dir dataset --output_dir output_paper2 --skip_leemdr

# Step 1 (skip GA feature selection too):
python paper2_pipeline.py --dataset_dir dataset --output_dir output_paper2 --skip_leemdr --skip_ga

# Step 2: Generate paper figures only (no training)
python aziz_train.py --generate_plots --dataset_dir dataset

# Step 3: Train all 12 classifiers with 10-fold CV
python aziz_train.py --features_file output_paper2/selected_features.npz

# Step 3 (specific classifiers + robustness test):
python aziz_train.py --classifiers SVM-G RF NN-TL --robustness
```

### Pipeline Steps (Paper 2)

1. **Resample** to 10,000 Hz → segment into 10-second chunks → z-score normalize → notch filter (50 Hz)
2. **LEEMDR**: EMD decomposition → RLE thresholding (>1%) → reconstruct from surviving IMFs
3. **MMFCC extraction** (40 coefficients): pre-emphasis → Hamming windowing → DFT → power spectrum → Mel filter bank (0–5 kHz) → log → DCT
4. **GA feature selection**: 40 → 24 features (10 chromosomes, 50 generations, crossover=0.8, mutation=0.1)

### Classifiers (12 total)

| Category | Classifiers |
|---|---|
| SVM | Linear (SVM-L), Quadratic (SVM-Q), Cubic (SVM-C), Gaussian (SVM-G) |
| KNN | Fine k=1 (KNN-1), Weighted (KNN-W) |
| Ensemble | Random Forest (RF), AdaBoost (AdaB) |
| ANN | Narrow-10 (NN-N), Wide-100 (NN-W), Bi-layered (NN-BL), Tri-layered (NN-TL) |

### Paper Figures Generated

| # | Figure | File |
|---|---|---|
| 1 | Raw sound signals (0–5 s) per class | `01_raw_signals.png` |
| 2 | IMF 3D decomposition (one per class) | `02_imf_visualization_<class>.png` |
| 3 | LEEMDR preprocessed signals (0–5 s) | `03_preprocessed_signals.png` |
| 4 | Triangular Mel filter bank (40 filters, 0–5 kHz) | `04_mel_filterbank.png` |
| 5 | MMFCC box-and-whisker plots across classes | `05_feature_boxplots.png` |
| 6 | Confusion matrices per classifier | `confusion_matrices/<clf>_cm.png` |
| 7 | SNR robustness chart (accuracy vs noise) | `07_robustness_snr.png` |

### Evaluation Metrics

Per-class and macro-average: **Accuracy, Sensitivity, Specificity, PPV, NPV**

---

## Output Directory Structure

```
output_paper1/              # Paper 1 pipeline outputs
  images/<class>/*.png      # 224×224×3 acoustic images
  dwv_analysis/             # DWV report and detected frequency range
  npz/<class>/*.npz         # Raw numpy arrays (if --save_npz)

output_paper2/              # Paper 2 pipeline outputs
  all_mmfcc_features.npz    # 40-dim MMFCCs + labels
  selected_features.npz     # 24-dim GA-selected features + labels
  ga_fitness_history.npy    # GA optimization trace
  ga_report.txt             # GA parameters and results

output_glowacz/             # Paper 1 training outputs
  plots/                    # 4 paper figures
  models/*.pth              # Best model weights
  results/                  # Training curves, confusion matrices
  kfold/                    # K-fold results (if enabled)
  summary_report.txt        # ER/MER for all models

output_aziz/                # Paper 2 training outputs
  plots/                    # 7 paper figures
  confusion_matrices/       # Per-classifier heatmaps
  summary_report.txt        # All metrics for all classifiers
```

---

## CLI Reference

### `paper1_pipeline.py`

```
--dataset_dir PATH     Raw audio directory (default: dataset)
--output_dir PATH      Output directory (default: output_paper1)
--auto_range           Auto-detect optimal DWV frequency range
--dwv_range_low N      Manual lower bound in Hz (default: 519)
--dwv_range_high N     Manual upper bound in Hz (default: 807)
--save_npz             Also save images as .npz arrays
```

### `glowacz_train.py`

```
--image_dir PATH       Acoustic images directory (default: output_paper1/images)
--dataset_dir PATH     Raw audio for plots (default: dataset)
--output_dir PATH      Output directory (default: output_glowacz)
--auto_range           Read auto-detected range from DWV report
--dwv_range_low N      Manual lower bound in Hz (default: 519)
--dwv_range_high N     Manual upper bound in Hz (default: 807)
--generate_plots       Generate 4 paper figures and exit
--train_per_class N    Training samples per class (default: 12)
--epochs N             Training epochs (default: 50)
--batch_size N         Batch size (default: 16)
--lr FLOAT             Learning rate (default: 0.001)
--models LIST          Which models to train (default: all)
--kfold N              K-fold CV folds, 0=off (default: 0)
--device STR           Device: auto, cpu, cuda, mps (default: auto)
--seed N               Random seed (default: 42)
```

### `paper2_pipeline.py`

```
--dataset_dir PATH     Raw audio directory (default: dataset)
--output_dir PATH      Output directory (default: output_paper2)
--notch_freq FLOAT     Powerline frequency in Hz (default: 50)
--pre_emph FLOAT       Pre-emphasis coefficient (default: 0.97)
--skip_leemdr          Skip EMD denoising (much faster)
--skip_ga              Skip GA feature selection
```

### `aziz_train.py`

```
--features_file PATH   GA-selected features (default: output_paper2/selected_features.npz)
--all_features_file PATH  Fallback: all 40 MMFCCs (default: output_paper2/all_mmfcc_features.npz)
--dataset_dir PATH     Raw audio for plots (default: dataset)
--output_dir PATH      Output directory (default: output_aziz)
--generate_plots       Generate 7 paper figures and exit
--classifiers LIST     Which classifiers to train (default: all 12)
--kfold N              CV folds (default: 10)
--robustness           Run SNR robustness test
--snr_levels LIST      SNR levels in dB (default: 10 20 30 40)
--seed N               Random seed (default: 42)
```

---

## Setup

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Dependencies

- numpy, scipy – numerical computing, signal processing, FFT
- SoundFile – WAV audio I/O
- Pillow – image processing and resizing
- EMD-signal (PyEMD) – Empirical Mode Decomposition for LEEMDR
- scikit-learn – SVM, KNN, RF, AdaBoost, ANN, metrics, cross-validation
- torch, torchvision – deep CNN training (DenseNet, ResNet, EfficientNet)
- matplotlib – all plot generation

Requires Python >= 3.14.

---

## References

- **Paper 1 (Glowacz et al.):** Acoustic fault diagnosis of three-phase induction motors using acoustic images and deep CNNs. Uses the DWV method to convert 1D audio into 2D normalized images classified by DenseNet-201, ResNet-18/50, and EfficientNet-B0.

- **Paper 2 (Aziz et al.):** Bearing faults classification using LEEMDR denoising and Machine Mel-Frequency Cepstral Coefficients (MMFCCs). Uses EMD-based signal cleaning, 40 custom MFCCs reduced to 24 via genetic algorithm, classified by SVM, KNN, ensemble, and ANN models with 10-fold CV achieving up to 99.26% accuracy (SVM-Gaussian).
