from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

from features import engineer_features
from train_model import (
    DATA_PATH, MODEL_PATH, METADATA_PATH, train_and_save
)

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered",
)

st.title("🏦 Loan Approval Prediction")
st.caption("Logistic Regression • End-to-End Machine Learning Project")

# Important improvement:
# If model files are missing but the dataset exists, train automatically.
if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    if DATA_PATH.exists():
        with st.spinner("First run detected. Training Logistic Regression model..."):
            try:
                train_and_save()
                st.success("Model trained and saved successfully.")
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.stop()
    else:
        st.error("Dataset is missing.")
        st.info(
            "Place the Kaggle CSV at:\n\n"
            "`data/loan_approval_dataset.csv`\n\n"
            "Then refresh this page."
        )
        st.stop()

try:
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()

st.write("Enter applicant information below.")

with st.form("prediction_form"):
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

    income_annum = st.number_input("Annual Income (₹)", min_value=0, value=5000000, step=100000)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=15000000, step=100000)
    loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=40, value=10, step=1)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700, step=1)

    residential_assets = st.number_input("Residential Assets Value (₹)", min_value=0, value=5000000, step=100000)
    commercial_assets = st.number_input("Commercial Assets Value (₹)", min_value=0, value=3000000, step=100000)
    luxury_assets = st.number_input("Luxury Assets Value (₹)", min_value=0, value=7000000, step=100000)
    bank_assets = st.number_input("Bank Asset Value (₹)", min_value=0, value=3000000, step=100000)

    submitted = st.form_submit_button("Predict Loan Approval", use_container_width=True)

if submitted:
    applicant = pd.DataFrame([{
        "no_of_dependents": dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets,
        "commercial_assets_value": commercial_assets,
        "luxury_assets_value": luxury_assets,
        "bank_asset_value": bank_assets,
    }])

    applicant = engineer_features(applicant)

    prediction = int(model.predict(applicant)[0])
    probability = float(model.predict_proba(applicant)[0, 1])
    threshold = float(metadata.get("threshold", 0.5))

    st.divider()
    if prediction == 1:
        st.success("✅ Predicted Result: LOAN APPROVED")
    else:
        st.error("❌ Predicted Result: LOAN REJECTED")

    st.metric("Approval Probability", f"{probability:.2%}")
    st.caption(f"Classification threshold: {threshold:.2f}")

with st.expander("Model evaluation"):
    m = metadata.get("metrics", {})
    if m:
        st.write(f"Accuracy: **{m.get('accuracy', 0):.3f}**")
        st.write(f"Precision: **{m.get('precision', 0):.3f}**")
        st.write(f"Recall: **{m.get('recall', 0):.3f}**")
        st.write(f"F1-score: **{m.get('f1', 0):.3f}**")
        st.write(f"ROC-AUC: **{m.get('roc_auc', 0):.3f}**")
