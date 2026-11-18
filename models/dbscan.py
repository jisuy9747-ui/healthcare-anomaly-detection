import time

import numpy as np
from sklearn.cluster import DBSCAN


def run_dbscan(X, eps: float = 0.5, min_samples: int = 5) -> dict:
    start = time.perf_counter()
    model = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = model.fit_predict(X)
    predictions = cluster_labels == -1
    cluster_count = len(set(cluster_labels.tolist())) - (1 if -1 in cluster_labels else 0)

    return {
        "name": "DBSCAN",
        "model": model,
        "predictions": np.asarray(predictions, dtype=bool),
        "scores": np.asarray(cluster_labels, dtype=float),
        "cluster_labels": np.asarray(cluster_labels),
        "cluster_count": cluster_count,
        "noise_count": int(predictions.sum()),
        "supports_score_auc": False,
        "processing_time": time.perf_counter() - start,
    }

