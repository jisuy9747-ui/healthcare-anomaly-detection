import pandas as pd


def explain_record(raw_row: pd.Series, reference: pd.DataFrame, selected_features: list) -> pd.DataFrame:
    rows = []
    for feature in selected_features:
        if feature not in raw_row.index or feature not in reference.columns:
            continue
        value = raw_row[feature]
        if not isinstance(value, (int, float)):
            continue
        mean = reference[feature].mean()
        std = reference[feature].std()
        z_score = (value - mean) / std if std and std > 0 else 0.0
        rows.append(
            {
                "Feature": feature,
                "Record value": value,
                "Reference mean": mean,
                "Reference standard deviation": std,
                "Deviation z-score": z_score,
                "Interpretation": (
                    "Far outside the reference distribution"
                    if abs(z_score) >= 3
                    else "Moderately outside the reference distribution"
                    if abs(z_score) >= 2
                    else "Within the usual reference range"
                ),
            }
        )
    return pd.DataFrame(rows)

