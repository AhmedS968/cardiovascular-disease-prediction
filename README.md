# Cardiovascular Disease Prediction

A machine learning classification project analysing clinical and lifestyle data from 70,000 patients to predict the presence of cardiovascular disease (CVD).

---

## Project Overview

Cardiovascular disease is the leading cause of global mortality, responsible for approximately 17.9 million deaths annually (WHO, 2021). Early, data-driven identification of at-risk individuals holds significant clinical utility. This project applies exploratory data analysis (EDA) and supervised machine learning to a dataset of 70,000 anonymised patient records, evaluating the predictive power of clinical features including blood pressure, cholesterol, glucose, BMI, and lifestyle variables.

This project was completed as part of a personal health data portfolio, drawing on skills in Python, pandas, seaborn, and scikit-learn.

---

## Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) |
| Records | 70,000 patients |
| Features | 12 (after ID removal) |
| Target | `cardio` (0 = no CVD, 1 = CVD present) |

**Feature descriptions:**

| Feature | Description | Type |
|---|---|---|
| `age` | Age in days (converted to years) | Numerical |
| `gender` | 1 = Female, 2 = Male | Categorical |
| `height` | Height in cm | Numerical |
| `weight` | Weight in kg | Numerical |
| `ap_hi` | Systolic blood pressure (mmHg) | Numerical |
| `ap_lo` | Diastolic blood pressure (mmHg) | Numerical |
| `cholesterol` | 1 = Normal, 2 = Above normal, 3 = Well above normal | Categorical |
| `gluc` | Glucose level (same scale as cholesterol) | Categorical |
| `smoke` | Smoking status (0/1) | Binary |
| `alco` | Alcohol intake (0/1) | Binary |
| `active` | Physical activity (0/1) | Binary |
| `bmi` | Body Mass Index (derived feature) | Numerical |

---

## Methods

### Data Cleaning
- Removed physiologically implausible blood pressure readings (systolic < 80 or > 250 mmHg; diastolic < 60 or > 200 mmHg)
- Removed records where diastolic pressure exceeded systolic pressure
- Filtered extreme height (< 140 cm or > 210 cm) and weight (< 30 kg or > 200 kg) values

### Feature Engineering
- Converted age from days to years
- Derived BMI from height and weight

### Exploratory Data Analysis
- Target variable distribution
- Age distribution by CVD status
- Gender stratification
- Cholesterol and glucose level analysis
- Full correlation matrix heatmap

### Modelling
- 70/30 stratified train-test split (`random_state=42`)
- Random Forest Classifier (100 estimators)
- Decision Tree Classifier (max depth = 8)
- Evaluation: accuracy, precision, recall, F1-score, confusion matrix, feature importance

---

## Results

| Model | Accuracy |
|---|---|
| Random Forest Classifier | ~72–73% |
| Decision Tree Classifier | ~69–71% |

**Top predictive features (Random Forest):**
1. Systolic blood pressure (`ap_hi`)
2. Age
3. BMI
4. Diastolic blood pressure (`ap_lo`)
5. Cholesterol level

---

## Repository Structure

```
cardiovascular-disease/
├── Cardiovascular_Disease.ipynb   # Full analysis notebook
├── cardio_train.csv               # Dataset (semicolon-delimited)
└── README.md                      # This file
```

---

## How to Run

### Option 1 — Google Colab (recommended, no installation required)
1. Upload both `Cardiovascular_Disease.ipynb` and `cardio_train.csv` to [Google Colab](https://colab.research.google.com)
2. Run all cells in order

### Option 2 — Local environment
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Cardiovascular_Disease.ipynb
```

---

## Skills Demonstrated

- Data cleaning and validation (outlier detection, physiological plausibility checks)
- Feature engineering (unit conversion, derived variables)
- Exploratory data analysis with seaborn and matplotlib
- Supervised machine learning classification with scikit-learn
- Model evaluation (accuracy, classification report, confusion matrix, feature importance)
- Python (pandas, numpy, seaborn, matplotlib, scikit-learn)

---

## Author

**AhmedS968**
[GitHub Profile](https://github.com/AhmedS968)

---

## Limitations and Future Work

- CVD diagnosis is self-reported; misclassification bias cannot be excluded
- No cross-validation applied; k-fold CV would produce more robust performance estimates
- Hyperparameter tuning (GridSearchCV) could improve model accuracy
- Future extensions: Logistic Regression baseline, ROC-AUC comparison, SHAP value analysis for clinical interpretability
