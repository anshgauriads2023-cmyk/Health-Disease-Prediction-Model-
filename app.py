
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("Heart Disease Prediction By Ansh Gauria")
st.markdown("Provide the following details")
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type",["ATA", "NAP", "ASY", "TA"])
resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, 1.0)
ST_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        "age": age,
        "Sex": 1 if sex == "Male" else 0,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,

        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == "Yes" else 0,

        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": 1 if exercise_angina == "Yes" else 0,
        "Oldpeak": oldpeak,
        "ST_Slope": ST_slope
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("High risk of heart disease. Please consult a doctor.")
    else:
        st.success("Low risk of heart disease. Keep maintaining a healthy lifestyle.")