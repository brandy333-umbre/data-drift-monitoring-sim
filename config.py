# config.py
from __future__ import annotations

from dataclasses import dataclass

@dataclass
class SimConfig:
    # Dataset
    n_train: int = 6000
    n_val: int = 1500
    n_test: int = 2000
    n_features: int = 20
    n_classes: int = 2
    class_sep: float = 1.2
    random_state: int = 42

    # Drift simulation
    n_steps: int = 20                 # how many "time steps" we simulate
    drift_strength_max: float = 2.0   # max magnitude of drift at final step
    drift_type: str = "covariate"     # "covariate" or "label" or "both"

    # Monitoring
    batch_size: int = 500
    report_top_k_errors: int = 5

    # Model
    model_type: str = "logreg"        # "logreg" or "rf"
