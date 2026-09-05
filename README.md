# Bladder-Cancer-Classification

# MRI Radiomics-Based Machine Learning for Bladder Cancer

This repository contains the source code used in the study of **MRI T2-weighted (T2WI) radiomics for bladder cancer classification using machine learning**.

## Overview

The study evaluates radiomics-based machine learning models for bladder cancer classification using MRI T2WI data collected from four imaging centers.

The machine learning models evaluated in this study include:

* **XGBoost**
* **Support Vector Machine (SVM) with RBF kernel**
* **Multilayer Perceptron (MLP)**
* **Stacking ensemble with Logistic Regression as the meta-learner**

The analysis includes independent test-set evaluation and **Leave-One-Center-Out (LOCO)** analysis to assess center-wise robustness and generalization.

## Dataset

The study included **215 patients** from four imaging centers.

| Dataset          | Patients |
| ---------------- | -------: |
| Training         |      171 |
| Independent Test |       44 |
| **Total**        |  **215** |

The raw MRI images and patient-level clinical data are **not included in this repository** due to data privacy and institutional restrictions.

## Analysis Pipeline

```text
MRI T2WI
   ↓
Radiomics Feature Extraction
   ↓
Preprocessing / Harmonization
   ↓
Boruta Feature Selection
   ↓
Machine Learning
   ├── XGBoost
   ├── SVM (RBF)
   └── MLP
   ↓
Stacking Ensemble
   ↓
Independent Test Evaluation
   ↓
LOCO Validation
```

## Radiomics

Radiomics features were extracted using **PyRadiomics**.

The analysis included:

* First-order features
* Shape features
* GLCM
* GLRLM
* GLSZM
* GLDM
* NGTDM

Main preprocessing parameters:

* Voxel spacing: **1 × 1 × 1 mm**
* Bin width: **25**
* Intensity normalization: **enabled**
* Normalization scale: **100**

## Feature Selection

Feature selection was performed using **Boruta**.

The selected features were subsequently used for machine learning model development and evaluation.

## Model Optimization

Hyperparameter optimization was performed using randomized search.

For the LOCO analysis:

* 200 randomized configurations
* 5-fold Stratified Cross-Validation
* ROC-AUC as the optimization criterion
* Hyperparameter optimization performed within each LOCO training fold

For XGBoost, `scale_pos_weight` was fixed at **1.7581** based on the training-set class distribution.

## LOCO Validation

Leave-One-Center-Out (LOCO) validation was performed to evaluate **center-wise robustness and generalization**.

Each imaging center was sequentially held out as the validation set:

```text
Fold 1 → Center 1 held out
Fold 2 → Center 2 held out
Fold 3 → Center 3 held out
Fold 4 → Center 4 held out
```

The independent test set (44 patients) was kept untouched during the LOCO analysis.

## Evaluation Metrics

Model performance was evaluated using:

* Accuracy
* Recall (Sensitivity)
* Precision
* F1-score
* ROC-AUC
* Confusion Matrix

## Repository Contents

The repository contains the code used for:

* Radiomics processing
* Feature selection
* Machine learning model development
* Hyperparameter optimization
* Stacking
* Independent test evaluation
* LOCO validation

## Requirements

The analysis was implemented in Python using libraries including:

```text
Python
NumPy
pandas
scikit-learn
XGBoost
PyRadiomics
SimpleITK
BorutaPy
Matplotlib
Seaborn
```

Package versions are provided in `requirements.txt`.

## Reproducibility

Random seeds were fixed where applicable to improve reproducibility.

The provided code reproduces the machine learning analysis described in the associated manuscript. However, the original patient-level MRI data are not publicly included in this repository.

## Contact

For questions regarding the code or methodology, please refer to the corresponding author of the associated publication.
