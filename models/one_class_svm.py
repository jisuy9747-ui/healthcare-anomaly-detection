import time

import numpy as np
from sklearn.svm import OneClassSVM


def run_one_class_svm(
    X,
    kernel: str = "rbf",
    nu: float = 0.05,
) -> dict:
    start = time.perf_counter()
    model = OneClassSVM(kernel=kernel, nu=nu)
    labels = model.fit_predict(X)
    predictions = labels == -1
    scores = -model.decision_function(X).ravel()
    return {
        "name": "One-Class SVM",
        "model": model,
        "predictions": predictions,
        "scores": np.asarray(scores, dtype=float),
        "processing_time": time.perf_counter() - start,
    }

