# =====================================================
# DEPLOYMENT PREP (FINAL FIXED VERSION)
# =====================================================

import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="ML Model Deployment",
    layout="wide"
)

st.title("🚀 End-to-End ML Deployment")

# -----------------------------------------------------
# PATHS
# -----------------------------------------------------
MODEL_PATH = "final_model.pkl"
X_TEST_PATH = "X_test.pkl"
Y_TEST_PATH = "y_test.pkl"

# -----------------------------------------------------
# LOAD FILES
# -----------------------------------------------------
model = joblib.load(MODEL_PATH)
X_test = joblib.load(X_TEST_PATH)
y_test = joblib.load(Y_TEST_PATH)

# -----------------------------------------------------
# FEATURE METADATA (CRITICAL FIX)
# -----------------------------------------------------
FEATURE_COLUMNS = list(X_test.columns)

NUMERIC_COLS = X_test.select_dtypes(include=["int64", "float64"]).columns.tolist()
CATEGORICAL_COLS = X_test.select_dtypes(include=["object", "category"]).columns.tolist()

# -----------------------------------------------------
# MODEL EVALUATION
# -----------------------------------------------------
st.header("📊 Model Evaluation")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.metric("Accuracy", f"{accuracy:.2%}")
st.text(classification_report(y_test, y_pred))

# -----------------------------------------------------
# SAFE PREDICTION API
# -----------------------------------------------------
def predict_api(input_dict):
    input_df = pd.DataFrame([input_dict])
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0].max()
    return pred, prob

# -----------------------------------------------------
# STREAMLIT FRONTEND (FIXED)
# -----------------------------------------------------
st.header("🧪 Live Prediction")

user_input = {}

with st.form("prediction_form"):
    cols = st.columns(3)

    for i, col in enumerate(FEATURE_COLUMNS):
        with cols[i % 3]:
            if col in NUMERIC_COLS:
                user_input[col] = st.number_input(col, value=0.0)
            else:
                categories = sorted(X_test[col].dropna().unique())
                user_input[col] = st.selectbox(col, categories)

    submit = st.form_submit_button("Predict")

if submit:
    prediction, confidence = predict_api(user_input)
    st.success(f"Prediction: {prediction}")
    st.info(f"Confidence: {confidence:.2%}")

# -----------------------------------------------------
# DEBUG PANEL
# -----------------------------------------------------
with st.expander("🔍 Debug Info"):
    st.write("Numeric Columns:", NUMERIC_COLS)
    st.write("Categorical Columns:", CATEGORICAL_COLS)
    st.write("Input Data:", user_input)

st.caption("Production-Ready Streamlit ML Deployment")