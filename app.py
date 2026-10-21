import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Cardiovascular Disease Risk Predictor")
st.write("Enter clinical parameters to predict heart disease risk.")

# Load trained artifacts
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

# User Input Form
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
chol = st.number_input("Serum Cholestoral (mg/dl)", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.selectbox("Resting ECG Results (0-2)", options=[0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", value=1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
thal = st.selectbox("Thal (1 = normal; 2 = fixed defect; 3 = reversable defect)", options=[1, 2, 3])

if st.button("Predict Disease Risk"):
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    imputed_features = imputer.transform(features)
    scaled_features = scaler.transform(imputed_features)
    prediction = model.predict(scaled_features)
    
    if prediction[0] == 1:
        st.error("High Risk of Cardiovascular Disease detected.")
    else:
        st.success("Low Risk of Cardiovascular Disease detected.")
