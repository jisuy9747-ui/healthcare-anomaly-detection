from io import BytesIO

import pandas as pd


def read_uploaded_dataset(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        stream = BytesIO(file.getvalue()) if hasattr(file, "getvalue") else file
        return pd.read_csv(stream)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(BytesIO(file.getvalue()), engine="openpyxl")
    raise ValueError("Unsupported file type. Upload a CSV or Excel (.xlsx) file.")


def dataset_summary(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in df.columns if column not in numeric]
    missing = df.isna().sum()
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "missing_counts": missing[missing > 0].sort_values(ascending=False),
        "missing_percentage": (missing / len(df) * 100).round(2) if len(df) else missing,
        "constant_columns": [column for column in df.columns if df[column].nunique(dropna=False) <= 1],
    }

