# 🏦 Loan Approval Prediction

End-to-End Machine Learning project using **Logistic Regression**.

## Assignment workflow covered

Kaggle Dataset → Data Understanding → EDA → Data Cleaning → Feature Engineering → Feature Selection → Train/Test Split → Logistic Regression → Evaluation → Prediction → Streamlit

## Dataset

Use the Kaggle dataset:

**Loan Approval Prediction Dataset — Archit Sharma**

Kaggle:
https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

Download `loan_approval_dataset.csv` and put it here:

```text
data/loan_approval_dataset.csv
```

The project is built specifically for these columns:

- loan_id
- no_of_dependents
- education
- self_employed
- income_annum
- loan_amount
- loan_term
- cibil_score
- residential_assets_value
- commercial_assets_value
- luxury_assets_value
- bank_asset_value
- loan_status

## Important improvements in this clean version

- Uses one exact Kaggle dataset schema.
- Removes `loan_id` as an irrelevant identifier.
- Cleans whitespace consistently.
- Removes duplicates.
- Clips invalid negative asset values to zero.
- Creates two meaningful engineered features:
  - `total_assets`
  - `loan_to_income_ratio`
- Uses the same preprocessing for training and prediction.
- Uses Logistic Regression exactly as required.
- Evaluates Accuracy, Precision, Recall, F1-score, Confusion Matrix and ROC-AUC.
- Saves the complete preprocessing + model pipeline.
- Streamlit automatically trains the model if model files are missing but the dataset exists.

## Windows setup

Open CMD inside this folder:

```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Train from terminal

```cmd
python train_model.py
```

Successful output will create:

```text
model/loan_pipeline.joblib
model/metadata.json
```

## Run Streamlit

```cmd
streamlit run app.py
```

If the model files do not exist, the app will automatically train them as long as the dataset exists.

## Run Jupyter Notebook

```cmd
python -m notebook
```

Open:

```text
Loan_Approval_Prediction.ipynb
```

Then use **Run → Run All Cells**.

## Project files

```text
Loan_Approval_Prediction_Clean/
├── app.py
├── features.py
├── train_model.py
├── Loan_Approval_Prediction.ipynb
├── requirements.txt
├── README.md
├── setup.bat
├── run_app.bat
├── data/
│   └── PUT_DATASET_HERE.txt
└── model/
```
