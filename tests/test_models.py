import numpy as np

from models.isolation_forest import run_isolation_forest
from models.lof import run_lof
from models.one_class_svm import run_one_class_svm
from models.autoencoder import run_autoencoder
from models.knn import run_knn_distance
from models.dbscan import run_dbscan


def sample_data():
    normal = np.random.default_rng(42).normal(0, 1, size=(100, 3))
    outliers = np.array([[8.0, 8.0, 8.0], [-8.0, -8.0, -8.0]])
    return np.vstack([normal, outliers])


def test_isolation_forest_returns_scores_and_predictions():
    result = run_isolation_forest(sample_data(), contamination=0.05)
    assert result["predictions"].shape == (102,)
    assert result["scores"].shape == (102,)
    assert result["predictions"].dtype == bool
    assert result["predictions"].sum() > 0


def test_lof_returns_scores_and_predictions():
    result = run_lof(sample_data(), neighbours=10)
    assert result["predictions"].shape == (102,)
    assert result["scores"].shape == (102,)


def test_one_class_svm_returns_scores_and_predictions():
    result = run_one_class_svm(sample_data(), kernel="rbf", nu=0.05)
    assert result["predictions"].shape == (102,)
    assert result["scores"].shape == (102,)
    assert result["predictions"].dtype == bool


def test_autoencoder_returns_reconstruction_scores_and_predictions():
    result = run_autoencoder(sample_data()[:50], epochs=2, batch_size=16)
    assert result["predictions"].shape == (50,)
    assert result["scores"].shape == (50,)
    assert result["threshold"] > 0
    assert result["predictions"].dtype == bool


def test_knn_distance_returns_scores_and_predictions():
    result = run_knn_distance(sample_data(), k=5)
    assert result["predictions"].shape == (102,)
    assert result["scores"].shape == (102,)
    assert result["threshold"] > 0


def test_dbscan_returns_noise_and_cluster_assignments():
    result = run_dbscan(sample_data(), eps=1.0, min_samples=5)
    assert result["predictions"].shape == (102,)
    assert result["cluster_labels"].shape == (102,)
    assert result["cluster_count"] >= 0
    assert result["noise_count"] == int(result["predictions"].sum())
