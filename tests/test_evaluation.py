import numpy as np

from src.evaluation import calculate_metrics
from src.export import dataframe_to_csv, dataframe_to_excel
import pandas as pd


def test_metrics_calculate_confusion_matrix_and_rates():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    scores = np.array([0.1, 0.8, 0.9, 0.2])
    result = calculate_metrics(y_true, y_pred, scores)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["specificity"] == 0.5
    assert result["false_positive_rate"] == 0.5
    assert 0 <= result["pr_auc"] <= 1
    assert 0 <= result["roc_auc"] <= 1


def test_csv_and_excel_exports():
    df = pd.DataFrame({"age": [20, 30], "score": [0.2, 0.8]})
    assert b"age" in dataframe_to_csv(df)
    workbook = dataframe_to_excel(df)
    assert workbook.startswith(b"PK")
