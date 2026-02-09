# simulate_drift.py
from __future__ import annotations

import argparse
import csv
import os
from typing import List, Dict, Any

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from config import SimConfig
from models import build_model
from metrics import compute_metrics
from drift_generators import make_covariate_drift, LabelShift


def save_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="Simulate data drift and monitor model performance over time.")
    parser.add_argument("--out_csv", type=str, default="results/drift_metrics.csv")
    parser.add_argument("--model_type", type=str, default=None, help="logreg or rf (overrides config)")
    parser.add_argument("--drift_type", type=str, default=None, help="covariate, label, or both (overrides config)")
    parser.add_argument("--n_steps", type=int, default=None, help="override number of drift steps")
    parser.add_argument("--seed", type=int, default=None, help="override random seed")
    args = parser.parse_args()

    cfg = SimConfig()
    if args.model_type:
        cfg.model_type = args.model_type
    if args.drift_type:
        cfg.drift_type = args.drift_type
    if args.n_steps is not None:
        cfg.n_steps = args.n_steps
    if args.seed is not None:
        cfg.random_state = args.seed

    rng = np.random.default_rng(cfg.random_state)
    np.random.seed(cfg.random_state)

    # -----------------------------
    # 1) Generate a base dataset (no drift)
    # -----------------------------
    X, y = make_classification(
        n_samples=cfg.n_train + cfg.n_val + cfg.n_test,
        n_features=cfg.n_features,
        n_informative=int(cfg.n_features * 0.6),
        n_redundant=int(cfg.n_features * 0.2),
        n_classes=cfg.n_classes,
        class_sep=cfg.class_sep,
        random_state=cfg.random_state,
    )

    # Train/val/test split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(cfg.n_val + cfg.n_test), random_state=cfg.random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=cfg.n_test, random_state=cfg.random_state, stratify=y_tmp
    )

    # -----------------------------
    # 2) Fit a baseline model (trained on "stable" data)
    # -----------------------------
    bundle = build_model(cfg.model_type, random_state=cfg.random_state)
    bundle.model.fit(X_train, y_train)

    # Baseline check on validation (no drift)
    yv_pred = bundle.predict(X_val)
    yv_proba = bundle.predict_proba(X_val)
    base = compute_metrics(y_val, yv_pred, yv_proba)
    print(f"\nModel: {bundle.name}")
    print(f"Baseline (no drift) on VAL: acc={base['accuracy']:.3f} macroF1={base['macro_f1']:.3f} logloss={base['logloss']:.3f}\n")

    # -----------------------------
    # 3) Create drift generators
    # -----------------------------
    cov_drift = make_covariate_drift(cfg.n_features, n_drift_features=min(6, cfg.n_features), seed=cfg.random_state)
    label_shift = LabelShift(flip_class=0, target_class=1, max_flip_rate=0.30)

    # We'll evaluate on the test set over time as drift increases.
    # Drift strength goes linearly from 0 -> cfg.drift_strength_max
    rows: List[Dict[str, Any]] = []

    print("Simulating drift over time...\n")
    for step in range(cfg.n_steps + 1):
        frac = step / max(cfg.n_steps, 1)
        drift_strength = frac * cfg.drift_strength_max

        Xt = X_test.copy()
        yt = y_test.copy()

        # Apply drift
        if cfg.drift_type in {"covariate", "both"}:
            Xt = cov_drift.apply(Xt, strength=drift_strength)

        if cfg.drift_type in {"label", "both"}:
            # Strength normalized to [0,1] for label flip rate
            strength01 = min(drift_strength / max(cfg.drift_strength_max, 1e-9), 1.0)
            yt = label_shift.apply(yt, strength=strength01)

        # Predict and compute metrics
        y_pred = bundle.predict(Xt)
        y_proba = bundle.predict_proba(Xt)
        m = compute_metrics(yt, y_pred, y_proba)

        row = {
            "step": step,
            "drift_strength": drift_strength,
            "drift_type": cfg.drift_type,
            "model": bundle.name,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "logloss": m["logloss"],
        }
        rows.append(row)

        print(
            f"step={step:>2} strength={drift_strength:>4.2f} "
            f"acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} logloss={m['logloss']:.3f}"
        )

    # -----------------------------
    # 4) Save metrics as CSV
    # -----------------------------
    save_csv(rows, args.out_csv)
    print(f"\nSaved metrics -> {args.out_csv}\n")

    # Optional: quick "monitoring trigger" example
    # (e.g., alert when accuracy drops by > 10% from baseline)
    acc0 = rows[0]["accuracy"]
    threshold = acc0 - 0.10
    triggered = [r for r in rows if r["accuracy"] < threshold]
    if triggered:
        first = triggered[0]
        print(
            f"Example alert: accuracy dropped >10% at step={first['step']} "
            f"(strength={first['drift_strength']:.2f}, acc={first['accuracy']:.3f})"
        )
    else:
        print("Example alert: accuracy never dropped >10% in this simulation.")


if __name__ == "__main__":
    main()
