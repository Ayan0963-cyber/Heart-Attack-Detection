# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Heart Attack Predictor", layout="centered")
st.title("❤ Heart Attack Prediction App")
st.markdown("Enter patient data in the sidebar to predict heart attack risk.")

# Load the trained model
model = joblib.load("heart_model.pkl")

# Sidebar for user input
def user_input_features():
    age = st.sidebar.slider("Age", 20, 80, 40)
    sex = st.sidebar.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.sidebar.radio("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: {
        1: "Normal",
        2: "Fixed Defect",
        3: "Reversible Defect"
    }[x])
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Display user input
st.subheader("Your Input:")
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display prediction result
st.subheader("Prediction:")
if prediction[0] == 1:
    st.error("💔 High Risk of Heart Attack")
else:
    st.success("✅ Low Risk of Heart Attack")

# Display confidence
st.subheader("Prediction Confidence:")
st.write(f"Model Confidence: **{np.max(prediction_proba) * 100:.2f}%**")
