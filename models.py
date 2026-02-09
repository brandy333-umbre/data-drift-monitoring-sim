# models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ModelBundle:
    model: Any
    name: str

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Some models may not implement predict_proba; handle gracefully
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        # fallback: make a pseudo-probability from hard predictions
        preds = self.model.predict(X)
        n_classes = int(preds.max()) + 1
        proba = np.zeros((len(preds), n_classes), dtype=float)
        proba[np.arange(len(preds)), preds] = 1.0
        return proba


def build_model(model_type: str, random_state: int = 42) -> ModelBundle:
    model_type = model_type.lower().strip()
    if model_type == "logreg":
        m = LogisticRegression(
            max_iter=500,
            n_jobs=None,
            solver="lbfgs",
        )
        return ModelBundle(model=m, name="LogisticRegression")
    elif model_type == "rf":
        m = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        )
        return ModelBundle(model=m, name="RandomForest")
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'logreg' or 'rf'.")
