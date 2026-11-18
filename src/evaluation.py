import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
)


def calculate_metrics(y_true, y_pred, anomaly_score) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    anomaly_score = np.asarray(anomaly_score, dtype=float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_positive_rate = fp / (tn + fp) if (tn + fp) else 0.0

    result = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if len(np.unique(y_true)) == 2:
        result["pr_auc"] = float(average_precision_score(y_true, anomaly_score))
        result["roc_auc"] = float(roc_auc_score(y_true, anomaly_score))
    else:
        result["pr_auc"] = None
        result["roc_auc"] = None
    return result

