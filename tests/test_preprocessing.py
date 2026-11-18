import pandas as pd
import pytest
from io import BytesIO

from src.data_loader import dataset_summary, read_uploaded_dataset
from src.preprocessing import fit_preprocessing, validate_analysis_structure


class UploadedCsv(BytesIO):
    def __init__(self, content: str):
        super().__init__(content.encode("utf-8"))
        self.name = "sample.csv"


def test_dataset_summary_identifies_types_duplicates_and_missing():
    df = pd.DataFrame(
        {
            "age": [20, 21, 21, None],
            "sex": ["F", "M", "M", "F"],
            "bmi": [22.1, 23.2, 23.2, 24.0],
        }
    )
    summary = dataset_summary(df)
    assert summary["rows"] == 4
    assert summary["numeric_columns"] == ["age", "bmi"]
    assert summary["categorical_columns"] == ["sex"]
    assert summary["duplicates"] == 1


def test_csv_upload_reader():
    file = UploadedCsv("age,bmi\n20,22.1\n21,23.2\n")
    df = read_uploaded_dataset(file)
    assert list(df.columns) == ["age", "bmi"]
    assert len(df) == 2


def test_fit_preprocessing_handles_missing_values_and_scaling():
    df = pd.DataFrame({"age": [20.0, 30.0, None], "bmi": [22.0, 24.0, 26.0], "constant": [1, 1, 1]})
    result = fit_preprocessing(df, ["age", "bmi", "constant"], scaling="Standard", imputation="Median")
    assert result.fitted_columns == ["age", "bmi"]
    assert result.dropped_constant_columns == ["constant"]
    assert result.data.isna().sum().sum() == 0
    assert abs(result.data["age"].mean()) < 1e-12


def test_empty_feature_selection_is_rejected():
    with pytest.raises(ValueError):
        fit_preprocessing(pd.DataFrame({"age": [20, 21]}), [])


def test_structure_validation_warns_for_small_and_duplicate_data():
    df = pd.DataFrame({"age": [20, 20], "bmi": [22, 22]})
    warnings = validate_analysis_structure(df, ["age", "bmi"])
    assert any("fewer than 20" in warning for warning in warnings)
    assert any("duplicate" in warning for warning in warnings)
