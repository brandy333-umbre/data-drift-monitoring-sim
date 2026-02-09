# drift_generators.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class CovariateDrift:
    """
    Covariate drift: shift feature distribution over time.
    We apply an additive shift to a subset of features and optional scaling.
    """
    feature_idx: np.ndarray  # indices of features to drift
    direction: np.ndarray    # drift direction vector for those features (unit-ish)

    def apply(self, X: np.ndarray, strength: float) -> np.ndarray:
        X2 = X.copy()
        # Additive shift along the chosen direction
        X2[:, self.feature_idx] += strength * self.direction
        return X2


@dataclass
class LabelShift:
    """
    Label shift: change class prevalence over time by flipping a portion of labels.
    This simulates a change in base rate, not feature distribution.
    """
    flip_class: int = 0  # class to flip from (0 -> 1 by default)
    target_class: int = 1
    max_flip_rate: float = 0.30  # at max drift strength, flip up to 30%

    def apply(self, y: np.ndarray, strength: float) -> np.ndarray:
        y2 = y.copy()
        flip_rate = min(max(strength, 0.0), 1.0) * self.max_flip_rate
        idx = np.where(y2 == self.flip_class)[0]
        n_flip = int(len(idx) * flip_rate)
        if n_flip <= 0:
            return y2
        flip_idx = np.random.choice(idx, size=n_flip, replace=False)
        y2[flip_idx] = self.target_class
        return y2


def make_covariate_drift(n_features: int, n_drift_features: int = 5, seed: int = 42) -> CovariateDrift:
    rng = np.random.default_rng(seed)
    feature_idx = rng.choice(np.arange(n_features), size=min(n_drift_features, n_features), replace=False)
    direction = rng.normal(size=len(feature_idx))
    # normalize direction for stable scaling
    norm = np.linalg.norm(direction) + 1e-12
    direction = direction / norm
    return CovariateDrift(feature_idx=feature_idx, direction=direction)
