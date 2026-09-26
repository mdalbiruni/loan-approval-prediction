from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import clean_strings, engineer_features, RAW_FEATURES

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "loan_approval_dataset.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "loan_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

TARGET = "loan_status"

def load_dataset(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download 'Loan Approval Prediction Dataset' from Kaggle and save it as "
            "data/loan_approval_dataset.csv"
        )

    df = pd.read_csv(path)
    df = clean_strings(df)

    required = ["loan_id", *RAW_FEATURES, TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset columns do not match the selected Kaggle dataset. "
            f"Missing columns: {missing}\nAvailable columns: {list(df.columns)}"
        )
    return df

def prepare_xy(df):
    df = df.drop_duplicates().copy()

    # Feature selection: loan_id is an identifier, not a predictive applicant attribute.
    df = df.drop(columns=["loan_id"])

    # Normalize target text and map to binary.
    df[TARGET] = df[TARGET].astype(str).str.strip()
    target_map = {"Approved": 1, "Rejected": 0}
    y = df[TARGET].map(target_map)
    if y.isna().any():
        unknown = sorted(df.loc[y.isna(), TARGET].unique().tolist())
        raise ValueError(f"Unexpected target labels: {unknown}")

    X = df.drop(columns=[TARGET])
    X = engineer_features(X)
    return X, y

def build_pipeline(X):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    return model, numeric_features, categorical_features

def train_and_save(data_path=DATA_PATH):
    df = load_dataset(data_path)
    X, y = prepare_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model, numeric_features, categorical_features = build_pipeline(X)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": {
            str(k): int(v) for k, v in y.value_counts().sort_index().items()
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "target": TARGET,
        "target_mapping": {"Rejected": 0, "Approved": 1},
        "raw_features": RAW_FEATURES,
        "engineered_features": ["total_assets", "loan_to_income_ratio"],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "metrics": metrics,
        "threshold": 0.5,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return model, metadata

if __name__ == "__main__":
    _, metadata = train_and_save()
    print("Training completed successfully.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print("\nEvaluation:")
    for key, value in metadata["metrics"].items():
        print(f"{key}: {value}")
