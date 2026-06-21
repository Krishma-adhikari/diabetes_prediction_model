# Diabetes Prediction Model

A machine learning project to predict diabetes in patients using classification algorithms. This model analyzes various health metrics to determine the likelihood of diabetes.

## 📋 Overview

This project implements a diabetes prediction system using health-related features such as gender, age, BMI, blood glucose level, and HbA1c level. Multiple machine learning classifiers are trained and evaluated, with Random Forest being selected as the primary model due to its superior accuracy.

## 📊 Dataset

- **File**: `diabetes_prediction_dataset.csv`
- **Size**: ~3.8 MB
- **Features**: Patient demographics and health indicators
- **Target**: Binary classification (Diabetes / No Diabetes)

### Key Features

- **Gender**: Patient gender
- **Age**: Patient age
- **Hypertension**: Whether patient has hypertension (0/1)
- **Heart Disease**: Whether patient has heart disease (0/1)
- **Smoking History**: Smoking status
- **BMI**: Body Mass Index
- **HbA1c Level**: Hemoglobin A1c level
- **Blood Glucose Level**: Current blood glucose level

## 🛠️ Technology Stack

- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Krishma-adhikari/diabetes_prediction_model.git
cd diabetes_prediction_model
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Training the Model

Run the training script:
```bash
python model_training.py
```

This will:
- Load and explore the dataset
- Perform exploratory data analysis (EDA)
- Preprocess the data (handle categorical variables, scale features)
- Train multiple classifiers (Random Forest and Logistic Regression)
- Evaluate model performance
- Display accuracy metrics and classification reports

### Making Predictions

Use the `predict_diabetes()` function to make predictions:

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

print(result)  # Output: "Diabetes" or "No diabetes"
```

## 📈 Model Performance

The project trains and compares two classifiers:

1. **Random Forest Classifier** (Selected Model)
   - Estimators: 200
   - Max Depth: 8
   - Better accuracy and AUC score

2. **Logistic Regression**
   - Alternative baseline model
   - Good for linear relationships

### Evaluation Metrics

- **Accuracy**: Overall prediction correctness
- **Precision & Recall**: Class-specific performance
- **Confusion Matrix**: Breakdown of predictions
- **ROC AUC Score**: Model's ability to distinguish between classes
- **Classification Report**: Detailed performance metrics

## 📝 Project Structure

```
diabetes_prediction_model/
├── diabetes_prediction_dataset.csv      # Training dataset
├── model_training.py                    # Main training script
├── sugar_disease_prediction_model.ipynb # Jupyter notebook
├── requirements.txt                     # Project dependencies
└── README.md                            # This file
```

## 🔄 Workflow

1. **Data Loading**: Load CSV dataset
2. **Exploratory Data Analysis**: Visualize distributions and correlations
3. **Data Cleaning**: Remove duplicates, handle missing values
4. **Preprocessing**: 
   - Split data into train/test sets
   - Scale numerical features
   - Encode categorical variables
5. **Model Training**: Train classifiers on preprocessed data
6. **Evaluation**: Assess model performance on test set
7. **Prediction**: Make predictions on new patient data

## 🎯 Key Features

- ✅ Comprehensive EDA with visualizations
- ✅ Data preprocessing pipeline
- ✅ Multiple model comparisons
- ✅ Robust evaluation metrics
- ✅ Easy-to-use prediction function
- ✅ Feature correlation analysis

## 📚 Notebooks

The project includes a Jupyter notebook (`sugar_disease_prediction_model.ipynb`) with interactive exploration and step-by-step implementation.

## ⚠️ Disclaimer

This model is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with healthcare professionals for medical decisions.

## 👤 Author

[Krishma-adhikari](https://github.com/Krishma-adhikari)
