"""Lightweight experiment runner for GraphCal vs Vanilla."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graphcal import GraphCalAugmenter
from src.graphcal.metrics import mae, tail_mae


def load_california(seed: int = 42):
    ds = fetch_california_housing(as_frame=False)
    X = ds.data.astype(np.float64)
    y = ds.target.astype(np.float64)
    return train_test_split(X, y, test_size=0.3, random_state=seed)


def evaluate_pair(X_train, X_test, y_train, y_test, seed: int):
    aug = GraphCalAugmenter(random_state=seed)
    aug_result = aug.fit_transform(X_train, y_train)

    mlp_base = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=50, random_state=seed)
    mlp_aug = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=50, random_state=seed)
    rf_base = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    rf_aug = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)

    mlp_base.fit(X_train, y_train)
    mlp_aug.fit(aug_result.X_aug, aug_result.y_aug)
    rf_base.fit(X_train, y_train)
    rf_aug.fit(aug_result.X_aug, aug_result.y_aug)

    out = {}
    for name, model in {
        "mlp_vanilla": mlp_base,
        "mlp_graphcal": mlp_aug,
        "rf_vanilla": rf_base,
        "rf_graphcal": rf_aug,
    }.items():
        pred = model.predict(X_test)
        out[name] = {
            "mae": mae(y_test, pred),
            "tail_mae_95": tail_mae(y_test, pred, q=0.95),
        }
    out["diagnostics"] = aug_result.diagnostics
    return out


def main() -> None:
    seeds = [42, 123, 456]
    rows = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = load_california(seed=seed)
        metrics = evaluate_pair(X_train, X_test, y_train, y_test, seed=seed)
        row = {"seed": seed, **metrics}
        rows.append(row)

    out_path = ROOT / "results_california_graphcal.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved results: {out_path}")


if __name__ == "__main__":
    main()
