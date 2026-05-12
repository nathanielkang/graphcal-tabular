from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CalMixResult:
    X_aug: np.ndarray
    y_aug: np.ndarray
    diagnostics: dict[str, Any]


class CalibratedMixupAugmenter:
    """
    Lightweight calibrated mixup for tabular regression.

    - Anchors are sampled with higher probability from tail values.
    - Pairs are chosen from target-nearest neighbors to reduce label mismatch.
    - Synthetic points are convex combinations of paired samples.
    """

    def __init__(
        self,
        mixup_ratio: float = 0.5,
        alpha: float = 0.7,
        pair_k: int = 32,
        tail_power: float = 1.2,
        random_state: int = 42,
    ) -> None:
        self.mixup_ratio = float(mixup_ratio)
        self.alpha = float(alpha)
        self.pair_k = int(pair_k)
        self.tail_power = float(tail_power)
        self.rng = np.random.default_rng(random_state)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> CalMixResult:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n, d = X.shape
        n_new = max(0, int(round(self.mixup_ratio * n)))
        if n_new == 0:
            return CalMixResult(
                X_aug=X.copy(),
                y_aug=y.copy(),
                diagnostics={"num_generated": 0, "mixup_ratio": 0.0, "pair_k": min(self.pair_k, max(1, n - 1))},
            )

        # Tail-aware anchor sampling probabilities from target rank.
        ranks = np.argsort(np.argsort(y))
        rel = (ranks.astype(np.float64) + 1.0) / float(n)
        rel = np.power(rel, self.tail_power)
        probs = rel / rel.sum()

        # Target-nearest neighbor list (in y space only).
        order = np.argsort(y)
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(n)
        k = max(1, min(self.pair_k, n - 1))

        left = np.clip(inv_order - k, 0, n - 1)
        right = np.clip(inv_order + k, 0, n - 1)

        x_new = np.empty((n_new, d), dtype=np.float64)
        y_new = np.empty((n_new,), dtype=np.float64)
        lambdas = self.rng.beta(self.alpha, self.alpha, size=n_new)
        anchors = self.rng.choice(n, size=n_new, p=probs, replace=True)

        for t, i in enumerate(anchors):
            lidx = left[i]
            ridx = right[i]
            if ridx <= lidx:
                j = int(self.rng.integers(0, n))
            else:
                pos = int(self.rng.integers(lidx, ridx + 1))
                j = int(order[pos])
            lam = float(lambdas[t])
            x_new[t] = lam * X[i] + (1.0 - lam) * X[j]
            y_new[t] = lam * y[i] + (1.0 - lam) * y[j]

        X_aug = np.vstack([X, x_new])
        y_aug = np.concatenate([y, y_new], axis=0)
        return CalMixResult(
            X_aug=X_aug,
            y_aug=y_aug,
            diagnostics={
                "num_generated": int(n_new),
                "mixup_ratio": float(n_new / max(1, n)),
                "pair_k": int(k),
                "alpha": float(self.alpha),
                "tail_power": float(self.tail_power),
            },
        )
