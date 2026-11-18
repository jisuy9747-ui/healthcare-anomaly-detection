import time

import numpy as np
from sklearn.ensemble import IsolationForest


def run_isolation_forest(X, contamination: float = 0.05, random_state: int = 42) -> dict:
    start = time.perf_counter()
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    labels = model.fit_predict(X)
    predictions = labels == -1
    scores = -model.decision_function(X)
    return {
        "name": "Isolation Forest",
        "model": model,
        "predictions": predictions,
        "scores": np.asarray(scores, dtype=float),
        "processing_time": time.perf_counter() - start,
    }

