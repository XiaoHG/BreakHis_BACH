"""Binary classification metrics for BreakHis."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def binary_metrics(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5
) -> Dict[str, float]:
    """Compute accuracy, macro-F1, AUC, sensitivity, specificity, balanced accuracy."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return {}
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    if np.unique(y_true).size < 2:
        auc = float("nan")
        balanced_accuracy = float((y_pred == y_true).mean())
    else:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")
        balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))

    return {
        "n": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": balanced_accuracy,
        "auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
