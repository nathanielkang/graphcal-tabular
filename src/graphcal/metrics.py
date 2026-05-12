from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def tail_mae(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.95) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    threshold = np.quantile(y_true, q)
    mask = y_true >= threshold
    if np.sum(mask) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
