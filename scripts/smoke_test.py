"""Smoke test with multiple inspections before full experiments."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graphcal import GraphCalAugmenter
from src.graphcal.metrics import mae, tail_mae


def make_synthetic_regression(n: int = 800, d: int = 8, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n, d))
    nonlinear = 0.5 * (X[:, 0] ** 2) - 0.25 * X[:, 1] * X[:, 2]
    skew_tail = rng.exponential(scale=1.5, size=n)
    y = 2.0 * X[:, 0] - 1.2 * X[:, 3] + nonlinear + 0.6 * skew_tail + rng.normal(0.0, 0.5, size=n)
    return X.astype(np.float64), y.astype(np.float64)


def main() -> None:
    print("=== GraphCal Smoke Test ===")
    X, y = make_synthetic_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("[Inspection 1] Data sanity")
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"  y skew proxy (mean-median): {np.mean(y_train) - np.median(y_train):.4f}")

    augmenter = GraphCalAugmenter(kappa=2.0, gamma=1.5, q_percentile=30.0, random_state=42)
    result = augmenter.fit_transform(X_train, y_train)

    print("[Inspection 2] Augmentation sanity")
    print(f"  Diagnostics: {result.diagnostics}")
    assert result.X_aug.shape[0] == result.y_aug.shape[0], "X/y augmented size mismatch"
    assert result.X_aug.shape[0] >= X_train.shape[0], "Augmentation should not shrink data"
    assert np.isfinite(result.X_aug).all() and np.isfinite(result.y_aug).all(), "NaN/Inf detected after augmentation"

    base_model = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", max_iter=2, random_state=42)
    aug_model = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu", max_iter=2, random_state=42)

    base_model.fit(X_train, y_train)
    aug_model.fit(result.X_aug, result.y_aug)

    pred_base = base_model.predict(X_test)
    pred_aug = aug_model.predict(X_test)

    print("[Inspection 3] Training/inference sanity")
    print(f"  Baseline MAE: {mae(y_test, pred_base):.4f}")
    print(f"  GraphCal MAE: {mae(y_test, pred_aug):.4f}")
    print(f"  Baseline Tail MAE@95%: {tail_mae(y_test, pred_base, q=0.95):.4f}")
    print(f"  GraphCal Tail MAE@95%: {tail_mae(y_test, pred_aug, q=0.95):.4f}")

    print("=== Smoke test completed successfully ===")


if __name__ == "__main__":
    main()
