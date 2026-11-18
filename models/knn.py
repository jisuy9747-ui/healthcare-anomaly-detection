import time

import numpy as np
from sklearn.neighbors import NearestNeighbors


def run_knn_distance(X, k: int = 5) -> dict:
    start = time.perf_counter()
    k = min(max(1, k), len(X) - 1)
    model = NearestNeighbors(n_neighbors=k + 1)
    model.fit(X)
    distances, _ = model.kneighbors(X)
    scores = distances[:, -1]
    threshold = np.percentile(scores, 95)
    predictions = scores >= threshold

    return {
        "name": "K-Nearest Neighbour",
        "model": model,
        "predictions": np.asarray(predictions, dtype=bool),
        "scores": np.asarray(scores, dtype=float),
        "threshold": float(threshold),
        "processing_time": time.perf_counter() - start,
    }

