# metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, log_loss


@dataclass
class MetricRow:
    step: int
    drift_strength: float
    accuracy: float
    macro_f1: float
    logloss: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    # Logloss expects probabilities; if shape mismatch, guard
    try:
        ll = log_loss(y_true, y_proba)
    except Exception:
        ll = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "logloss": float(ll),
    }
