import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️")


@st.cache_resource
def load_model():
    df = pd.read_csv("heart_disease_uci.csv")
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["id", "dataset", "ca", "thal", "num"])
    df["chol"] = df["chol"].replace(0, np.nan)
    df["trestbps"] = df["trestbps"].replace(0, np.nan)

    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope"]

    preprocess = ColumnTransformer([
        ("num", Pipeline([("fill_blanks", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("fill_blanks", SimpleImputer(strategy="most_frequent")),
                          ("to_numbers", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    model = Pipeline([
        ("prep", preprocess),
        ("clf", SVC(C=1, gamma=0.1, probability=True, random_state=42)),
    ])
    model.fit(X_train, y_train)
    return model


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
    patient
