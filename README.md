# Diabetes Prediction Model

A machine learning project that predicts the likelihood of diabetes for patients using clinical and demographic features. This repository includes a Jupyter notebook with an end-to-end walkthrough and a modular Python script for training and making predictions.

## Table of contents

- Overview
- Dataset
- Key features
- Installation
- Usage
  - Train the model
  - Make predictions
- Model details & evaluation
- Project structure
- Reproducibility
- Notebook
- Disclaimer
- Author

## Overview

This project builds and evaluates classifiers to predict a binary diabetes outcome (Diabetes / No Diabetes) from features such as age, gender, BMI, blood glucose, and HbA1c level. It contains exploratory data analysis (EDA), preprocessing, model training, evaluation, and a prediction utility.

## Dataset

- File: `diabetes_prediction_dataset.csv`
- Size: ~3.8 MB
- Target: Binary label indicating diabetes
- Typical features: gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, blood glucose level

Tip: Inspect `diabetes_prediction_dataset.csv` to confirm feature names and data types before running the pipeline.

## Key features

- End-to-end notebook that demonstrates EDA, preprocessing, model training, and evaluation
- A reusable `model_training.py` script that exposes a `predict_diabetes()` function for single-instance predictions
- Comparison between Random Forest and Logistic Regression models
- Standard evaluation metrics: accuracy, precision, recall, F1-score, confusion matrix, and ROC AUC

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Krishma-adhikari/diabetes_prediction_model.git
cd diabetes_prediction_model
```

2. Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Usage

### Train the model

Run the training script to preprocess data, train models, and save the selected model to disk:

```bash
python model_training.py
```

What the script does:
- Loads and inspects the dataset
- Performs EDA and visualizations (in the notebook)
- Cleans data and handles missing values
- Encodes categorical variables and scales numeric features
- Trains and compares multiple classifiers (Random Forest, Logistic Regression)
- Evaluates models and prints classification reports and metrics
- Persists the chosen model for inference

### Make predictions

The script exposes a convenience function `predict_diabetes(...)` for single-instance predictions. Example usage:

```python
from model_training import predict_diabetes

result = predict_diabetes(
    gender="Female",
    age=80,
    hypertension=0,
    heart_disease=1,
    smoking_history="never",
    bmi=25.19,
    HbA1c_level=6.6,
    blood_glucose_level=140
)

print(result)  # "Diabetes" or "No diabetes"
```

If you modify feature names in the dataset, update `predict_diabetes()` accordingly.

## Model details & evaluation

- Primary model: Random Forest Classifier (selected after comparison)
  - Typical hyperparameters used: n_estimators=200, max_depth=8
- Baseline model: Logistic Regression

Evaluation metrics used:
- Accuracy, precision, recall, F1-score
- Confusion matrix
- ROC AUC score

For reproducible experiments, set a random seed in `model_training.py` before splitting or training.

## Project structure

```
diabetes_prediction_model/
├── diabetes_prediction_dataset.csv      # Training dataset
├── model_training.py                    # Training script and predict_diabetes() function
├── sugar_disease_prediction_model.ipynb # Interactive Jupyter notebook (EDA + experiments)
├── requirements.txt                     # Dependencies
└── README.md                            # This file
```

## Reproducibility

- Use the included `requirements.txt` to create a matching environment
- If you want deterministic results, set the random seed in `model_training.py` (for example, `random_state=42`) and ensure scikit-learn operations use it
- To reproduce results from the notebook, open `sugar_disease_prediction_model.ipynb` in Jupyter and run cells in order

## Notebook

Open `sugar_disease_prediction_model.ipynb` for a step-by-step walkthrough of data exploration, visualization, preprocessing, and model evaluation. The notebook contains plots and explanations that complement the script.

## Disclaimer

This project is intended for educational and research purposes only. It is NOT medical advice and should not be used for clinical decision-making. Consult qualified healthcare professionals for medical concerns.

## Author

Krishma Adhikari — https://github.com/Krishma-adhikari
