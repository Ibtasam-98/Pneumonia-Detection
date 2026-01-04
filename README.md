# Comparative Analysis of Machine Learning Architectures for Pneumonia Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ML-Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the **official implementation** and experimental results for the research paper:
> **"Robust Detection of Pneumonia in Chest X-Rays via Dimensionality Reduction and Support Vector Machines."**

Our research evaluates the efficacy of classical Machine Learning (ML) architectures against the Chest X-Ray (Pneumonia) dataset, demonstrating that optimized traditional models can achieve clinical-grade accuracy with superior computational efficiency compared to deep learning alternatives.

---

## 📌 Abstract
Early diagnosis of pneumonia is a critical factor in patient mortality rates. This study explores a high-efficiency diagnostic pipeline using **Principal Component Analysis (PCA)** and three primary classifiers: **SVM, KNN, and Random Forest**. By reducing the feature space to 50 principal components (retaining **81.27%** of variance), our **SVM model achieved an Accuracy of 94.97% and a Sensitivity of 97.54%**, establishing a robust framework for rapid medical screening in resource-constrained environments.

---

## 📊 Experimental Results

### 1. Model Performance Benchmarking
The Support Vector Machine (SVM) outperformed all other architectures, particularly in **Recall (Sensitivity)** and **Inference Speed**, which are the most critical metrics for clinical applications.

| Metric | SVM (Rank 1) | KNN | Random Forest |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **0.9497** | 0.9258 | 0.9206 |
| **Recall (Sensitivity)** | **0.9754** | 0.9743 | 0.9731 |
| **Precision (PPV)** | **0.9564** | 0.9276 | 0.9224 |
| **F1-Score** | **0.9492** | 0.9240 | 0.9185 |
| **AUC-ROC** | **0.9846** | 0.9727 | 0.9744 |
| **Inference Time** | **0.16 ms/img** | 0.74 ms/img | 3.93 ms/img |



### 2. SVM Diagnostic Matrix (Test Set)
The SVM model demonstrated exceptional reliability in identifying positive cases while maintaining a low Type II error rate.

* **True Positives (TP):** 834 (Pneumonia correctly identified)
* **True Negatives (TN):** 279 (Normal correctly identified)
* **False Negatives (FN):** 21 (Missed cases)
* **Specificity:** 0.8801
* **NPV (Negative Predictive Value):** 0.9300



---

## 🛠️ Methodology & Implementation

### Data Pipeline
1.  **Preprocessing:** Images normalized to grayscale and resized to a uniform feature vector.
2.  **Dimensionality Reduction:** Applied PCA to reduce the input space to $n=50$ components, significantly lowering computational costs while maintaining an 81.27% explained variance ratio.
3.  **Optimization:** Utilized `GridSearchCV` for hyperparameter tuning. The optimal SVM configuration used an **RBF kernel** with `C=1`.

### Statistical Validation
To ensure the robustness of the reported metrics, the implementation includes:
* **5-Fold Stratified Cross-Validation:** (SVM Mean: 0.9532, Std Dev: 0.0052).
* **Bootstrapping ($n=50$):** SVM 95% Confidence Interval for Accuracy: **[0.9501, 0.9723]**.
* **Learning Curves:** Analysis confirmed low overfitting (Gap: 0.0217) and strong generalization.



---

## 💻 Usage

### Prerequisites
```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
