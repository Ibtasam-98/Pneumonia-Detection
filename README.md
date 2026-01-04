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
+-----------------+----------------------+----------------------+
| | Predicted Positive | Predicted Negative |
+=================+======================+======================+
| Actual Positive | TP: 834 (Pneumonia) | FP: 38 |
+-----------------+----------------------+----------------------+
| Actual Negative | FN: 21 | TN: 279 (Normal) |
+-----------------+----------------------+----------------------+
