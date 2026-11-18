from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass
class PreprocessingResult:
    data: pd.DataFrame
    pipeline: Pipeline
    fitted_columns: list
    dropped_constant_columns: list


def fit_preprocessing(
    df: pd.DataFrame,
    selected_columns: list,
    scaling: str = "Standard",
    imputation: str = "Median",
) -> PreprocessingResult:
    if not selected_columns:
        raise ValueError("Select at least one numerical feature.")

    missing = [column for column in selected_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing selected columns: {', '.join(missing)}")

    source = df[selected_columns].copy()
    if not all(pd.api.types.is_numeric_dtype(source[column]) for column in source.columns):
        raise ValueError("Every selected feature must contain numerical values.")

    constant = [column for column in source.columns if source[column].nunique(dropna=True) <= 1]
    active = [column for column in source.columns if column not in constant]
    if not active:
        raise ValueError("All selected features are constant; select at least one varying feature.")

    if imputation == "Mean":
        imputer = SimpleImputer(strategy="mean")
    elif imputation == "Most frequent":
        imputer = SimpleImputer(strategy="most_frequent")
    else:
        imputer = SimpleImputer(strategy="median")

    scaler = RobustScaler() if scaling == "Robust" else StandardScaler()
    pipeline = Pipeline([("imputer", imputer), ("scaler", scaler)])
    transformed = pipeline.fit_transform(source[active])
    processed = pd.DataFrame(
        transformed,
        index=source.index,
        columns=active,
    )
    return PreprocessingResult(processed, pipeline, active, constant)


def validate_analysis_structure(df: pd.DataFrame, selected_columns: list) -> list:
    warnings = []
    if len(df) < 20:
        warnings.append("The dataset has fewer than 20 records; model estimates may be unstable.")
    if len(selected_columns) == 0:
        warnings.append("No numerical features have been selected.")
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        warnings.append(
            f"{duplicate_count} duplicate rows are present. Review them instead of silently deleting them."
        )
    return warnings

