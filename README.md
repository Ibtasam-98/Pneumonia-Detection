# Comparative Analysis of Machine Learning Architectures for Pneumonia Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ML-Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the **official implementation** and experimental results for the research paper:
> **"Deployable AI Solution for Pneumonia Detection: A Contrastive Analysis of Machine Learning Ensemble Classifiers on Chest Radiographs."**

Our research presents a comprehensive machine learning pipeline for automated pneumonia detection from chest X-ray images, comparing three classical ML models with extensive clinical validation and a deployable web application.

---

## 📌 Abstract
Early diagnosis of pneumonia is critical for reducing mortality rates. This study implements a high-efficiency diagnostic pipeline using **Principal Component Analysis (PCA)** with three optimized classifiers: **Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Random Forest**. By reducing the feature space to 50 principal components (retaining **81.27%** of variance), our **SVM model achieved superior performance** across all metrics: **94.97% Accuracy, 97.54% Sensitivity, and 88.01% Specificity**. The system includes a fully-functional web application for real-time clinical decision support.

---

## 📊 Experimental Results

### 1. Model Performance Benchmarking
SVM outperformed all other architectures, particularly in **clinical metrics** and **computational efficiency**:

| Metric | SVM (Rank 1) | KNN | Random Forest |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **0.9497** | 0.9258 | 0.9206 |
| **Recall (Sensitivity)** | **0.9754** | 0.9743 | 0.9731 |
| **Precision (PPV)** | **0.9564** | 0.9276 | 0.9224 |
| **F1-Score** | **0.9492** | 0.9240 | 0.9185 |
| **Specificity** | **0.8801** | 0.7950 | 0.7792 |
| **AUC-ROC** | **0.9846** | 0.9727 | 0.9744 |
| **AUC-PR** | **0.9939** | 0.9849 | 0.9900 |
| **MCC** | **0.8709** | 0.8073 | 0.7936 |
| **Calibration Error** | **0.0554** | 0.0966 | 0.1262 |
| **Training Time** | 33.08 s | **1.28 s** | 102.55 s |
| **Inference Time** | **0.16 ms/img** | 0.74 ms/img | 3.93 ms/img |

### 2. Confusion Matrix Analysis
#### SVM - Confusion Matrix (Test Set: 1172 images)
* **True Positives (TP):** 834 (Pneumonia correctly identified)
* **True Negatives (TN):** 279 (Normal correctly identified)
* **False Positives (FP):** 38 (Normal incorrectly predicted as pneumonia)
* **False Negatives (FN):** 21 (Pneumonia incorrectly predicted as normal)
* **Specificity:** 0.8801
* **Negative Predictive Value (NPV):** 0.9300

### 3. Statistical Validation
* **5-Fold Stratified Cross-Validation:** SVM Mean Accuracy: 0.9532 ± 0.0052
* **Bootstrapping (n=50):** SVM 95% CI for Accuracy: **[0.9501, 0.9723]**
* **Overfitting Analysis:** SVM shows minimal overfitting (Gap: 0.0217) compared to Random Forest (Gap: 0.0794)

---
## 🏥 Clinical Significance

### Diagnostic Performance
1. **High Sensitivity:** All models achieved >97% sensitivity, ensuring reliable pneumonia detection
2. **Superior Specificity:** SVM's 88.01% specificity reduces false positives by 20-26 cases per 1000 patients compared to alternatives
3. **Reliable Probability Estimates:** SVM's low calibration error (0.0554) provides trustworthy risk assessment

### Computational Efficiency
* **Fastest Inference:** SVM processes images in 0.16 ms/image - suitable for real-time clinical use
* **Memory Efficient:** Compact model size (2.1 MB) enables deployment in resource-constrained settings
* **Web-Ready:** Full integration into interactive Streamlit application

---

## 🛠️ Methodology & Implementation

### Data Pipeline
1. **Dataset:** 5,856 pediatric chest X-rays from Guangzhou Women and Children's Medical Center
   - Training: 5,216 images (74.3% Pneumonia, 25.7% Normal)
   - Testing: 624 images (62.5% Pneumonia, 37.5% Normal)
   - Class Imbalance Ratio: 2.70:1 (Pneumonia:Normal)

2. **Preprocessing:**
   - Resize to 100×100 pixels
   - Grayscale conversion and intensity normalization
   - Data augmentation (rotation, flipping, brightness adjustment)

3. **Feature Engineering:**
   - Extract statistical, gradient, and texture features
   - Dimensionality Reduction: PCA to 50 components (81.27% variance explained)
   - Feature standardization for model compatibility

### Model Training
- **SVM:** RBF kernel with C=1.0, gamma='scale'
- **KNN:** k=11 neighbors, Manhattan distance, uniform weights
- **Random Forest:** 100 estimators, no max depth limitation
- **Hyperparameter Tuning:** GridSearchCV with 3-fold cross-validation
- **Class Imbalance Handling:** Balanced weighting strategies

---

## Web Application

### Features
- **Real-time Prediction:** Upload chest X-ray for instant analysis
- **Multi-Model Consensus:** Compare predictions from SVM, KNN, and Random Forest
- **Clinical Visualizations:** Probability distributions, confidence scores, risk indicators
- **Export Results:** Download predictions and visualizations for medical records
