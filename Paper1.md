# Audio Transformations for Model Training Replication

This document provides the exact sequence of transformations applied to the raw audio files in the two provided papers, complete with mathematical formulas and parameter values. You can use this guide to build a robust replication pipeline for your datasets.

---

## Paper 1: Acoustic Fault Diagnosis of Induction Motors (Glowacz et al.)

This paper transforms 1D acoustic signals into 2D normalized images to be processed by deep convolutional neural networks (DenseNet, ResNet, EfficientNet).

### 1. Data Acquisition and Segmentation
* **Format Conversion:** The raw audio is captured in Advanced Audio Coding (AAC) format at 48,000 Hz and converted to WAVE format with a sampling frequency of **44,100 Hz**.
* **Segmentation:** The audio stream is divided into non-overlapping **1-second samples**.

### 2. Pre-Filtering & Normalization
* **Low-Pass Filter:** In the verified optimal method, a low-pass filter with a cutoff frequency range of **1–1225 Hz** is applied to the segmented signal.
* **Amplitude Normalization:** The time-domain amplitude of each 1-second sample is normalized.

### 3. Fast Fourier Transform (FFT)
* **FFT Application:** The FFT spectrum of the normalized 1-second sample is computed.
* **Frequency Cropping:** The spectrum is cropped to focus on the essential frequency components (e.g., $1\text{–}1000\text{ Hz}$). Let the spectrum be denoted as $f = [f_1, f_2, f_3, \dots, f_{n}]$.

### 4. Word Coding (Feature Discretization)
The continuous FFT spectrum $f$ is mapped into a discrete "word vector" $v = [v_1, v_2, v_3, \dots, v_n]$ using a defined threshold parameter $k$ (default value $k = 0.01$).
The conversion logic maps the amplitude $f_n$ to a specific word (coefficient) $D_p$:

$$ 
\begin{cases} 
\text{if } (f_n < k) \text{ and } (f_n > 0) & \Rightarrow \text{convert to } D_1 \\
\text{if } (f_n < 2k) \text{ and } (f_n > k) & \Rightarrow \text{convert to } D_2 \\
\text{if } (f_n < 3k) \text{ and } (f_n > 2k) & \Rightarrow \text{convert to } D_3 \\
\vdots \\
\text{if } (f_n < pk) \text{ and } (f_n > (p-1)k) & \Rightarrow \text{convert to } D_p 
\end{cases} 
$$
Where $p$ is a positive integer.

### 5. Differences of Word Vectors (DWV) & Region Selection
To isolate the most discriminative acoustic features, the DWV method limits the vectors to a specific frequency range.
* By computing the absolute differences between word vectors of different classes (e.g., Healthy vs. Broken Bars) and taking the maximum differences, a frequency band of interest is identified. 
* In the paper, the range of **519–807 Hz** is selected to form a matrix.

### 6. Acoustic Image Formation
* **Matrix Construction:** The selected frequency components (519–807 Hz) of the word vector are mapped into a square matrix $A$ of dimensions **$17 \times 17$** (since $17 \times 17 = 289$ components).
* **Reshaping for CNNs:** Matrix $A$ is resized to matrix $W$ with dimensions **$224 \times 224 \times 3$** (standard input size for ResNet/DenseNet).
* **Image Normalization:** Matrix $W$ is normalized by its maximum value to cap pixel intensities:
    $$W_{\text{Normalized}} = \frac{W}{\max(W)}$$

