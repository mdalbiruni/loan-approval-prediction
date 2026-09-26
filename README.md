# 🏦 Loan Approval Prediction

An end-to-end Machine Learning classification project that predicts whether a loan application will be approved or rejected.

## 🚀 Live Demo
[Open Live App](https://loan-approval-prediction-hmdejmabx9k5xtwklqruck.streamlit.app/)

## 📊 Model Performance

- Accuracy: 91.33%s
- Precision: 92.08%
- Recall: 94.16%
- F1-score: 93.11%
- ROC-AUC: 97.30%

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

## 🤖 Model

Logistic Regression

## 🎯 Objective

The goal of this project is to predict loan approval or rejection using applicant and financial information.

## 📌 Project Workflow

Kaggle Dataset → EDA → Data Cleaning → Feature Engineering → Feature Selection → Train/Test Split → Logistic Regression → Evaluation → Prediction → Streamlit

## ✨ Features Used

- Number of Dependents
- Education
- Self Employed
- Annual Income
- Loan Amount
- Loan Term
- CIBIL Score
- Residential Assets Value
- Commercial Assets Value
- Luxury Assets Value
- Bank Asset Value

## ⚙️ Feature Engineering

Two additional features were created:

- Total Assets
- Loan-to-Income Ratio

## 📈 Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

## 💻 Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
