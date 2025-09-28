import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pathlib

# Set base directory to project root
BASE_DIR = pathlib.Path(__file__).parent.parent

# Load the full pipeline (preprocessor + model together)
pipeline = joblib.load(BASE_DIR / "models" / "final_pipeline.pkl")

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient information below to check risk of heart disease:")

# User inputs
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [("Yes", 1), ("No", 0)])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 70, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [("Yes", 1), ("No", 0)])
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [3, 6, 7])  # dataset values

# Collect features
features = np.array([[age, sex[1], cp, trestbps, chol, fbs[1],
                      restecg, thalach, exang[1], oldpeak, slope, ca, thal]])

# Predict
if st.button("Predict"):
    input_df = pd.DataFrame(features, columns=[
        "age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal"
    ])
    prediction = pipeline.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ The model predicts **Heart Disease** risk.")
    else:
        st.success("✅ The model predicts **No Heart Disease** risk.")
