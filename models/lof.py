import time

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def run_lof(X, neighbours: int = 20, contamination: float | str = "auto") -> dict:
    start = time.perf_counter()
    neighbours = min(neighbours, max(2, len(X) - 1))
    model = LocalOutlierFactor(n_neighbors=neighbours, contamination=contamination)
    labels = model.fit_predict(X)
    predictions = labels == -1
    scores = -model.negative_outlier_factor_
    return {
        "name": "Local Outlier Factor",
        "model": model,
        "predictions": predictions,
        "scores": np.asarray(scores, dtype=float),
        "processing_time": time.perf_counter() - start,
    }

