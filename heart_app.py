import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(page_title="Heart Attack Predictor", layout="centered")
st.title("❤ Heart Attack Prediction App")
st.markdown("Enter patient data in the sidebar to predict heart attack risk.")

# Check if model file exists
MODEL_PATH = "heart_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please run `train_model.py` first to generate 'heart_model.pkl'.")
    st.stop()

# Load the trained model
model = joblib.load(MODEL_PATH)

# Sidebar inputs
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
    slope = st.sidebar.selectbox("Slope of the ST Segment", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
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

# Get input
input_df = user_input_features()

# Show user input
st.subheader("Your Input:")
st.write(input_df)

# Add predict button
if st.button("🔍 Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction:")
    if prediction[0] == 1:
        st.error("💔 High Risk of Heart Attack")
    else:
        st.success("✅ Low Risk of Heart Attack")

    st.subheader("Prediction Confidence:")
    st.write(f"Model Confidence: **{np.max(prediction_proba) * 100:.2f}%**")

else:
    st.info("👈 Enter the details and click **Predict** to see the result.")
