import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load("diabetes_pred.pkl")


st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk")


gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 1, 120, 30)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
smoking_history = st.selectbox(
    "Smoking History",
    ["never", "former", "current", "not current", "ever"]
)
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.5)
glucose = st.number_input("Blood Glucose Level", 50, 300, 100)


if st.button("Predict"):

    input_data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Diabetes Detected")
    else:
        st.success("✅ No Diabetes Detected")