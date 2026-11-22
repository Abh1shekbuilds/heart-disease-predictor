# ❤️ Heart Disease Risk Predictor

A machine learning system that predicts the risk of heart disease from 11 routine clinical measurements, deployed as a live web app.

### 🔗 [Try the live app](https://hdai-predictor.streamlit.app/)

> ⚠️ Learning project only. Not a medical diagnostic tool.

---

## 📌 Overview

Given a patient's age, sex, chest pain type, blood pressure, cholesterol, ECG results and exercise test results, the model predicts whether heart disease is likely and outputs a risk percentage.

Because a **missed disease case is far more dangerous than a false alarm**, the model was selected with a focus on **recall** (catching as many true disease cases as possible), not just accuracy.

## 📊 Dataset

**UCI Heart Disease dataset** (via Kaggle): 920 patients combined from 4 hospitals (Cleveland, Hungary, Switzerland, VA Long Beach).

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Male / Female |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar above 120 mg/dl |
| `restecg` | Resting ECG result |
| `thalch` | Maximum heart rate during exercise |
| `exang` | Exercise-induced chest pain |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |

**Target:** the original `num` (0 to 4 severity) was converted to binary: **0 = no disease (411 patients)**, **1 = disease (509 patients)**.

## 🛠️ Approach

**1. Data cleaning**
- Dropped `ca` (66% missing) and `thal` (53% missing): too sparse to impute reliably
- Dropped `id` and `dataset` (hospital name), since a new patient would not come with a hospital label
- Replaced physiologically impossible values (cholesterol = 0, blood pressure = 0) with missing values

**2. Train/test split**
- 80/20 stratified split (736 train / 184 test), keeping the 55% disease ratio in both sets
- The test set was held out until the final evaluation

**3. Preprocessing pipeline** (scikit-learn `Pipeline` + `ColumnTransformer`)
- Numeric columns: median imputation + standard scaling
- Categorical columns: most-frequent imputation + one-hot encoding
- All preprocessing is fitted on training data only, preventing data leakage

**4. Model comparison** (5-fold stratified cross-validation on the training set)

| Model | Accuracy | Recall | F1 |
|---|---|---|---|
| **SVM** | 0.811 | **0.894** | **0.841** |
| KNN | 0.814 | 0.877 | 0.840 |
| Random Forest | 0.807 | 0.857 | 0.831 |
| Gradient Boosting | 0.806 | 0.855 | 0.830 |
| Logistic Regression | 0.794 | 0.838 | 0.817 |

**5. Hyperparameter tuning**
- `GridSearchCV` over SVM `C` and `gamma`, optimizing F1
- Best: `C = 1`, `gamma = 0.1`

## ✅ Final Results (held-out test set, 184 patients)

| Metric | Baseline (Logistic Regression) | Final (Tuned SVM) |
|---|---|---|
| Accuracy | 79.3% | **83.7%** |
| Recall (Disease) | 0.84 | **0.92** |
| Missed disease cases | 16 | **8** |
| ROC-AUC | n/a | **0.891** |

The final model catches **94 of 102** true disease cases, **halving** the missed cases compared with the baseline.

## 💻 Tech Stack

- **Python**, **pandas**, **NumPy**
- **scikit-learn** (pipelines, SVM, cross-validation, grid search)
- **Matplotlib** (confusion matrix, ROC curve)
- **Streamlit** (web app) on **Streamlit Community Cloud** (hosting)
- **Google Colab** (model development)

## 🚀 Run Locally

```bash
git clone https://github.com/Abh1shekbuilds/heart-disease-predictor.git
cd heart-disease-predictor
pip install -r requirements.txt streamlit
streamlit run app.py
```

## 📁 Project Structure

```
├── app.py                         # Streamlit web app (trains the model on startup)
├── heart_disease_uci.csv          # Dataset
├── heart_disease_model.ipynb      # Full analysis notebook (EDA, model comparison, tuning)
├── requirements.txt               # Dependencies
└── README.md
```

## ⚠️ Limitations

- Small dataset (920 patients) from older hospital records
- Two potentially strong features (`ca`, `thal`) were dropped because of missing data
- The app retrains on startup with the latest scikit-learn, so metrics may differ very slightly from those reported above
- Not clinically validated, so it must not be used for real medical decisions

## 🔮 Future Work

- Explain predictions with SHAP (which features drove each prediction)
- Tune the decision threshold to push recall higher
- Try deep learning on ECG signal data

---

**Author:** [Abh1shekbuilds](https://github.com/Abh1shekbuilds)
