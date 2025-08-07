import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Heart Attack Predictor", layout="centered")

st.title("❤ Heart Attack Prediction App")
st.markdown("Enter patient data in the sidebar to predict heart attack risk.")


def user_input_features():
    age = st.sidebar.slider("Age", 20, 80, 40)
    sex = st.sidebar.radio("Sex", [0, 1])  # 0: Female, 1: Male
    cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting BP", 80, 200, 120)
    chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.selectbox("Rest ECG", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate", 70, 210, 150)
    exang = st.sidebar.radio("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope", [0, 1, 2])
    ca = st.sidebar.selectbox("Major Vessels", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thal", [1, 2, 3])
    
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

input_df = user_input_features()

#
df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]


model = RandomForestClassifier()
model.fit(X, y)


prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)


st.subheader("Your Input:")
st.write(input_df)

st.subheader("Prediction:")
st.write("💔 Heart Attack Risk" if prediction[0] == 1 else "✅ No Heart Attack Risk")

st.subheader("Prediction Confidence:")
st.write(f"{np.max(prediction_proba)*100:.2f}%")
