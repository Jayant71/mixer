# Replication Guide: Bearing Faults Classification using LEEMDR and MMFCCs (Aziz et al.)

## 1. Dataset Characteristics & Splitting (SUBF v2.0)
* **Motor Specs:** 3-phase AC motor, 1440 RPM, 50 Hz, 0.25 HP. Microphone positioned 6 inches above ground, non-contact.
* **Classes:** 3 conditions (Healthy, Inner Race Fault (IR), Outer Race Fault (OR)).
* **Sampling Rate:** **10,000 Hz**.
* **Segmentation:** Audio divided into **10-second segments**.
* **Dataset Size:** 18 hours recording (6 hours per class). Yields 2160 signals per category $\rightarrow$ **6480 signals total**.
* **Validation Strategy:** **10-fold cross-validation**. 
  * Training pool per fold: **5800 observations**.
  * Testing pool per fold: **648 observations**.

## 2. Preprocessing & Feature Extraction
* **LEEMDR Process (Log Energy-based Empirical Mode Decomposition and Reconstruction):**
  * Decompose incoming sound into Intrinsic Mode Functions (IMFs).
  * Filter out IMFs using a Relative Log Energy threshold. Only keep IMFs where $RLE_j > 1\%$:
  $$RLE_j = \frac{\log\left(\sum_{n=1}^N |IMF_j[n]|^2\right)}{\log\left(\sum_{n=1}^N |v[n]|^2\right)}$$
  * Reconstruct the denoised signal $p(t)$ by summing surviving IMFs.
* **MMFCC Extraction:**
  * Apply a pre-emphasis filter: $PE(z) = 1 - \mu z^{-1}$.
  * Apply Hamming window with a half-frame overlap.
  * Extract **$40$ MMFCCs** using a Mel triangular filter bank constrained specifically to the **$0\text{–}5000\text{ Hz}$** range (motor fault range).
* **Feature Selection (Genetic Algorithm):**
  * Inputs: 40 MMFCC features.
  * Optimization output: Reduced to **$24$ features** preserving maximal variance.
  * GA Hyperparameters: Population = 10 chromosomes, Max generations = 50, Crossover rate = 0.8, Mutation rate = 0.1.

## 3. Model Architectures & Configurations
Train the following classifiers with the 24 selected features. *(Note: SVM-G yields the highest accuracy of 99.26% in the original study)*.
* **Support Vector Machines (SVM):** Linear (SVM-L), Quadratic (SVM-Q), Cubic (SVM-C), and Gaussian Kernel (SVM-G). 
* **K-Nearest Neighbors (KNN):** KNN-Fine ($k=1$), Weighted KNN (KNN-W).
* **Ensemble Models:** Random Forest (RF), AdaBoost (AdaB).
* **Artificial Neural Networks (ANN):**
  * Training parameters: **ReLU** activation, Backpropagation, and Max Iterations = **1000**.
  * **NN-N (Narrow):** 1 hidden layer, 10 neurons.
  * **NN-W (Wide):** 1 hidden layer, 100 neurons.
  * **NN-BL (Bi-layered):** 2 hidden layers, 100 neurons each.
  * **NN-TL (Tri-layered):** 3 hidden layers, 100 neurons each.

## 4. Evaluation Metrics
Calculate the following metrics across the 3 classes using standard confusion matrix values ($TP, TN, FP, FN$):
* **Accuracy** $= \frac{TP + TN}{TP + TN + FP + FN}$
* **Sensitivity** $= \frac{TP}{TP + FN}$
* **Specificity** $= \frac{TN}{TN + FP}$
* **Positive Predictive Value (PPV)** $= \frac{TP}{TP + FP}$
* **Negative Predictive Value (NPV)** $= \frac{TN}{TN + FN}$

## 5. Plots, Matrices, and Figures to Replicate
Generate the following visuals:
1. **Raw Sound Signals:** Line plot of Amplitude vs. Time ($0\text{–}5\text{ seconds}$) for Healthy, IR, and OR conditions.
2. **IMF Visualization:** 3D line plot of Mode Amplitude vs. Time vs. Mode Number for decomposed signals.
3. **Preprocessed Signals:** Line plot of Amplitude vs. Time ($0\text{–}5\text{ seconds}$) showing the cleaned $p(t)$ signal after LEEMDR.
4. **Triangular Filter Bank:** Graph Amplitude vs. Frequency ($0\text{–}5000\text{ Hz}$) plotting the overlapping Mel triangular filters.
5. **Feature Box Plots:** Box-and-whisker plots comparing distributions of specific MMFCCs across the three classes.
6. **Confusion Matrices:** $3 \times 3$ heatmaps (Actual vs. Predicted class) for SVMs, KNNs, Ensembles, and ANNs.
7. **Robustness Chart (SNR):** A grouped bar chart plotting Accuracy vs. SNR (at $10\text{ dB}, 20\text{ dB}, 30\text{ dB}, 40\text{ dB}, \text{Original}$) for the overall accuracy and individually for the 3 classes.
