import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️")

@st.cache_resource
def load_model():
    return joblib.load("heart_model.pkl")

model = load_model()

st.title("❤️ Heart Disease Risk Predictor")
st.write("Tuned SVM trained on the UCI Heart Disease dataset (920 patients). "
         "Test accuracy 84 percent, recall 92 percent, ROC-AUC 0.89.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 80, 50)
    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    cp = st.selectbox("Chest pain type",
                      ["asymptomatic", "atypical angina", "non-anginal", "typical angina"])
    trestbps = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 220)
    fbs = st.checkbox("Fasting blood sugar above 120 mg/dl")

with col2:
    restecg = st.selectbox("Resting ECG result",
                           ["normal", "lv hypertrophy", "st-t abnormality"])
    thalch = st.slider("Max heart rate during exercise", 60, 210, 150)
    exang = st.checkbox("Chest pain during exercise")
    oldpeak = st.slider("Oldpeak (ECG change during exercise)", 0.0, 6.0, 1.0, 0.1)
    slope = st.selectbox("ECG slope during exercise", ["upsloping", "flat", "downsloping"])

if st.button("Predict", type="primary", use_container_width=True):
    patient = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalch": thalch, "exang": exang,
        "oldpeak": oldpeak, "slope": slope,
    }])
    risk = model.predict_proba(patient)[0, 1]

    if risk >= 0.5:
        st.error(f"Heart disease likely. Predicted risk: {risk*100:.1f}%")
    else:
        st.success(f"Heart disease unlikely. Predicted risk: {risk*100:.1f}%")
    st.progress(float(risk))

st.caption("Learning project only. Not a medical tool.")
