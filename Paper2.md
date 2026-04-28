

## Paper 2: Bearing Faults Classification using LEEMDR and MMFCCs (Aziz et al.)

This paper cleans the audio heavily via a specialized decomposition method before extracting custom Machine Mel-frequency Cepstral Coefficients (MMFCC) as 1D features for machine learning classifiers (like SVMs).

### 1. Data Acquisition and Basic Formatting
* **Sampling:** The audio is captured with a sampling frequency of **10,000 Hz**.
* **Segmentation:** The signal is divided into **10-second segments**.
* **Standard Normalization:** Raw signals are normalized to eliminate gain dependencies and scale variances.
* **Powerline Filtering:** A **Butterworth notch filter** is applied to remove electrical powerline interference.

### 2. Signal Denoising: LEEMDR Method
The Log Energy-based Empirical Mode Decomposition and Reconstruction (LEEMDR) isolates essential features by decomposing the signal $v(t)$ into Intrinsic Mode Functions (IMFs).
* **Decomposition:** The signal is broken down into $IMFs$.
* **Relative Log Energy (RLE) Thresholding:** The importance of each $j^{\text{th}}$ IMF is evaluated against the original signal $v(t)$:
    $$RLE_j = \frac{\log\left(\sum_{n=1}^N |IMF_j[n]|^2\right)}{\log\left(\sum_{n=1}^N |v[n]|^2\right)}$$
* **Reconstruction:** Only IMFs where $RLE_j > 1\%$ are kept. The preprocessed signal $p(t)$ is reconstructed by summing these surviving, high-energy IMFs.

### 3. Feature Extraction: Machine Mel-Frequency Cepstral Coefficients (MMFCC)
40 custom MMFCCs are extracted from $p(t)$ focusing on the motor-specific fault spectrum ($0\text{–}5000\text{ Hz}$).

**Step A: Pre-emphasis Filtering**
Amplifies high-frequency elements using filter $PE(z)$:
$$PE(z) = 1 - \mu z^{-1}$$
*(Where $\mu$ usually ranges from 0.9 to 1).*

**Step B: Windowing**
The signal is divided into frames with a half-frame overlap. A **Hamming window** is applied to combat spectral leakage.

**Step C: Discrete Fourier Transform (DFT)**
The time-domain signal $x_f(n)$ is converted to the frequency spectrum $X_f(k)$ for each frame of size $M$:
$$X_f(k) = \sum_{n=0}^{M-1} x_f(n) e^{-j\frac{2\pi nk}{M}}$$

**Step D: Power Spectrum**
$$P_f(k) = \frac{1}{M}|X_f(k)|^2$$

**Step E: Mel Frequency Filter Bank**
A set of triangular filters tuned specifically to $0\text{–}5000\text{ Hz}$. For the $c^{\text{th}}$ filter with center frequency $f_o(c)$:
$$ H_c(k) = 
\begin{cases} 
0, & k < f_o(c-1) \\ 
\frac{k - f_o(c-1)}{f_o(c) - f_o(c-1)}, & f_o(c-1) \leq k \leq f_o(c) \\ 
\frac{f_o(c+1) - k}{f_o(c+1) - f_o(c)}, & f_o(c) \leq k \leq f_o(c+1) \\ 
0, & k > f_o(c+1) 
\end{cases} 
$$

**Step F: Log Spectrum**
$$S(c) = \ln\left[ \sum_{k=0}^{N-1} P_f(k)H_c(k) \right], \quad 0 \leq c \leq C$$

**Step G: Discrete Cosine Transform (DCT)**
Computes the final $O$ coefficients $C_M(n)$ (where $O=40$ for this pipeline):
$$C_M(n) = \sum_{c=0}^{N-1} S(c) \cos\left(\frac{\pi n (c - 0.5)}{C}\right), \quad n = 1, 2, \dots, O$$

### 4. Feature Selection (Genetic Algorithm)
* **Genetic Algorithm (GA):** A GA is applied to reduce the dimensionality of the feature set from the 40 extracted MMFCCs down to the **24 most discriminative features**.
* **GA Parameters:** Number of chromosomes = 10, Maximum generations = 50, Crossover rate = 0.8, Mutation rate = 0.1.
* **Final Output:** A robust 24-dimension feature vector passed into classifiers (like SVM with a Gaussian kernel).
