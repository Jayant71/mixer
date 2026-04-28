# Replication Guide: Acoustic Fault Diagnosis of Three-Phase Induction Motors (Glowacz et al.)

## 1. Dataset Characteristics & Splitting
* **Classes:** 4 conditions (Healthy, 1 broken rotor bar, 2 broken rotor bars, 3 broken rotor bars).
* **Sampling:** Captured at 48,000 Hz (AAC format), converted to **44,100 Hz (WAVE format)**.
* **Segmentation:** Audio streams are divided into strictly **1-second non-overlapping samples**.
* **Dataset Size:** * Total Samples: **280**
  * Training Set: **48 samples** (12 per class).
  * Test Set: **232 samples** (58 per class).
* **Validation Strategy:** K-fold cross-validation.

## 2. Preprocessing & Feature Extraction (DWV Method)
* **Pre-filter:** A low-pass filter from $1\text{–}1225\text{ Hz}$ is applied to the raw signal.
* **Normalization:** Amplitude normalization of each 1-second sample.
* **FFT:** Compute the FFT spectrum and evaluate components.
* **Word Coding:** Mapping FFT amplitude $f_n$ to sequence coefficients $D_p$ using a threshold parameter $k = 0.01$.
* **DWV Frequency Range Identification:** Based on the Differences of Word Vectors (DWV), the highest variant frequency range across classes is isolated. The specific optimal range determined is **$519\text{–}807\text{ Hz}$**.

## 3. Matrix Transformations & Acoustic Images
* **Initial Matrix ($A$):** The word vector components within the $519\text{–}807\text{ Hz}$ range are formed into a square matrix $A$ of dimensions **$17 \times 17$** ($17 \times 17 = 289$ frequency components).
* **Input Image Matrix ($W$):** Matrix $A$ is resized/interpolated to standard CNN dimensions of **$224 \times 224 \times 3$**.
* **Matrix Normalization:** The pixel intensities of the image are capped using the matrix's maximum value.
  $$W_{\text{Normalized}} = \frac{W}{\max(W)}$$

## 4. Model Architectures Evaluated
To match the study, evaluate the $224 \times 224 \times 3$ acoustic images using the following deep Convolutional Neural Networks (initialized with standard weights):
* **DenseNet-201** (201 layers, dense connectivity)
* **ResNet-18** (18 layers, residual blocks)
* **ResNet-50** (50 layers, residual bottleneck blocks)
* **EfficientNet-b0** (inverted bottleneck residual blocks, squeeze and excitation)

## 5. Evaluation Metrics
* **Efficiency of Recognition (ER):**
  $$ER = 100\% \times \left(\frac{R}{A}\right)$$
  Where $R$ = correctly recognized test samples, $A$ = total test samples.
* **Mean Efficiency of Recognition (MER):**
  $$MER = \frac{ER_1 + ER_2 + ER_3 + ER_4}{4}$$

## 6. Plots and Figures to Replicate
Generate the following visual plots to confirm parity with the paper's analyses:
1. **Time-domain signals:** Raw acoustic waveforms for the 4 distinct classes.
2. **FFT Spectra Plots:** Normalized amplitude vs. Frequency ($1\text{–}1000\text{ Hz}$) for the 4 classes.
3. **Word Vector Maps:** Stem/Bar plots of the computed $D$ coefficient against Frequency ($519\text{–}807\text{ Hz}$).
4. **Acoustic Images:** Grayscale visualization of the $17 \times 17$ matrix $W$ showing distinct cluster patterns per motor condition.
