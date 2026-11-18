import pandas as pd

from src.validation import check_leave_state_out


def test_leave_state_out_requires_state_column():
    df = pd.DataFrame({"age": [20, 30]})
    result = check_leave_state_out(df)
    assert result["available"] is False


def test_leave_state_out_accepts_multiple_states():
    df = pd.DataFrame({"state": ["A", "A", "B", "B"], "age": [20, 21, 30, 31]})
    result = check_leave_state_out(df, "state")
    assert result["available"] is True
    assert result["state_count"] == 2
