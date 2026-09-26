import pandas as pd
import numpy as np

RAW_FEATURES = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

ASSET_COLUMNS = [
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact same cleaning/feature engineering for training and prediction."""
    df = clean_strings(df)

    # Negative asset values are treated as invalid and clipped to zero.
    for col in ASSET_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)

    # Meaningful engineered feature 1: total declared asset value.
    existing_assets = [c for c in ASSET_COLUMNS if c in df.columns]
    if existing_assets:
        df["total_assets"] = df[existing_assets].sum(axis=1)

    # Meaningful engineered feature 2: requested loan relative to annual income.
    if "loan_amount" in df.columns and "income_annum" in df.columns:
        income = pd.to_numeric(df["income_annum"], errors="coerce")
        loan = pd.to_numeric(df["loan_amount"], errors="coerce")
        df["loan_to_income_ratio"] = loan / income.replace(0, np.nan)

    return df
