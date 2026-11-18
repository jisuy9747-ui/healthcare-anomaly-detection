import pandas as pd


def check_leave_state_out(df: pd.DataFrame, state_column: str = None) -> dict:
    if state_column is None:
        return {
            "available": False,
            "reason": "No state identifier column was supplied.",
        }
    if state_column not in df.columns:
        return {
            "available": False,
            "reason": f"State column '{state_column}' is not present.",
        }
    state_count = int(df[state_column].nunique(dropna=True))
    if state_count < 2:
        return {
            "available": False,
            "reason": "At least two distinct states are required.",
        }
    return {
        "available": True,
        "states": df[state_column].dropna().astype(str).unique().tolist(),
        "state_count": state_count,
    }
