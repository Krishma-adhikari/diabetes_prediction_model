# 🩺 Diabetes Prediction Model

A machine learning project to predict diabetes in patients using classification algorithms. This model analyzes various health metrics to determine the likelihood of diabetes.

---

## 📋 Overview

This project builds a diabetes prediction system using health features like gender, age, BMI, blood glucose level, and HbA1c level.

It includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Machine learning model training
- Model evaluation
- Streamlit web application for real-time prediction

---

## 📊 Dataset

- File: `diabetes_prediction_dataset.csv`
- Type: Binary classification
- Target: diabetes (0 = No, 1 = Yes)

Features:
- gender
- age
- hypertension
- heart_disease
- smoking_history
- bmi
- HbA1c_level
- blood_glucose_level

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

## 🤖 Machine Learning Models

- Random Forest Classifier (Final Model)
- Logistic Regression (Baseline Model)

---

## 📦 Installation

```bash
git clone https://github.com/Krishma-adhikari/diabetes_prediction_model.git
cd diabetes_prediction_model
pip install -r requirements.txt
```

---

## 📁 Project Structure

```text
diabetes_prediction_model/
│
├── diabetes_prediction_dataset.csv
├── model_training.py
├── sugar_disease_prediction_model.ipynb
├── app.py
├── diabetes_pred.pkl
├── requirements.txt
└── README.md
```

---

## 🚀 Running the Streamlit App

```bash
streamlit run app.py
```
## Author
Krishma Adhikari
