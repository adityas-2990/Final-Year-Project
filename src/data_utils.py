"""
data_utils.py
─────────────
Pure-Python / pandas helpers for loading, encoding, imputing, and
downloading data.  No Streamlit imports here — keeps logic testable.
"""

import base64

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────────────────────────────────────
#  Type conversion helpers
# ──────────────────────────────────────────────────────────────────────────────

def convert_booleans_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Cast all boolean columns to 0 / 1 integers."""
    df_out = df.copy()
    bool_cols = df_out.select_dtypes(include=[bool]).columns
    df_out[bool_cols] = df_out[bool_cols].astype(int)
    return df_out


def convert_categorical_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode every object / category column.
    Returns a *new* DataFrame; the original is untouched.
    """
    df_out = df.copy()
    cat_cols = df_out.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        df_out[col] = le.fit_transform(df_out[col].astype(str))
    return df_out


# ──────────────────────────────────────────────────────────────────────────────
#  Missing-value handlers
# ──────────────────────────────────────────────────────────────────────────────

def fill_missing(df: pd.DataFrame, column: str, method: str) -> pd.DataFrame:
    """
    Return a *new* DataFrame with missing values in `column` handled
    according to `method`:
        "mean" | "median" | "mode" | "drop"
    """
    df_out = df.copy()
    col = df_out[column]

    if method == "mean":
        df_out[column] = col.fillna(col.mean())
    elif method == "median":
        df_out[column] = col.fillna(col.median())
    elif method == "mode":
        df_out[column] = col.fillna(col.mode()[0])
    elif method == "drop":
        df_out = df_out.drop(columns=[column])
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return df_out


def knn_impute(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Impute *all* missing values with KNN.
    Categorical / boolean columns are encoded before imputation and the
    imputed values are written back only for the originally-missing columns.
    """
    df_encoded = convert_booleans_to_numeric(df)
    df_encoded = convert_categorical_to_numeric(df_encoded)

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_array = imputer.fit_transform(df_encoded)
    df_imputed = pd.DataFrame(imputed_array, columns=df.columns)

    # Write imputed values back for originally-missing columns only
    df_out = df.copy()
    for col in df.columns[df.isnull().any()]:
        df_out[col] = df_imputed[col]

    return df_out


# ──────────────────────────────────────────────────────────────────────────────
#  Outlier detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_outliers(df: pd.DataFrame, column: str, method: str) -> pd.DataFrame:
    """
    Return the subset of `df` that contains outliers in `column`.
    `method` is "zscore" or "iqr".
    """
    if method == "zscore":
        z = np.abs((df[column] - df[column].mean()) / df[column].std())
        return df[z > 3]
    elif method == "iqr":
        q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
        iqr = q3 - q1
        mask = (df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)
        return df[mask]
    else:
        raise ValueError(f"Unknown method: {method!r}")


# ──────────────────────────────────────────────────────────────────────────────
#  Download helper
# ──────────────────────────────────────────────────────────────────────────────

def df_to_csv_download_link(df: pd.DataFrame, filename: str = "data.csv") -> str:
    """Return an HTML anchor tag that triggers a CSV download."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return (
        f'<a href="data:file/csv;base64,{b64}" download="{filename}">'
        f"📥 Download CSV File</a>"
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Descriptive helpers
# ──────────────────────────────────────────────────────────────────────────────

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy DataFrame of columns with missing values."""
    counts = df.isnull().sum()
    counts = counts[counts > 0]
    if counts.empty:
        return pd.DataFrame(columns=["Column", "Missing Count", "Missing %"])
    summary = counts.reset_index()
    summary.columns = ["Column", "Missing Count"]
    summary["Missing %"] = (summary["Missing Count"] / len(df) * 100).round(2)
    return summary


def skewness_label(skew: float) -> str:
    if abs(skew) < 0.5:
        return "symmetric"
    return "right-skewed" if skew > 0 else "left-skewed"