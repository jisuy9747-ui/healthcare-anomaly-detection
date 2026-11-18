import pandas as pd

from models.dbscan import run_dbscan
from models.isolation_forest import run_isolation_forest
from models.knn import run_knn_distance
from models.lof import run_lof
from models.one_class_svm import run_one_class_svm
from src.preprocessing import fit_preprocessing


def test_synthetic_demo_preprocesses_and_runs_classical_models():
    df = pd.read_csv("data/sample/synthetic_healthcare_demo.csv")
    features = ["Age", "BMI", "Systolic_BP", "Diastolic_BP", "Heart_Rate", "Glucose"]
    result = fit_preprocessing(df, features)

    for runner in (
        lambda: run_isolation_forest(result.data.to_numpy()),
        lambda: run_lof(result.data.to_numpy()),
        lambda: run_one_class_svm(result.data.to_numpy()),
        lambda: run_knn_distance(result.data.to_numpy()),
        lambda: run_dbscan(result.data.to_numpy()),
    ):
        model_result = runner()
        assert len(model_result["predictions"]) == len(df)
        assert len(model_result["scores"]) == len(df)
