from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class GraphCalResult:
    X_aug: np.ndarray
    y_aug: np.ndarray
    diagnostics: dict[str, Any]


class GraphCalAugmenter:
    """Graph-based manifold-native augmentation for imbalanced regression."""

    def __init__(
        self,
        k: int | None = None,
        q_percentile: float = 30.0,
        hop_count: int = 2,
        kappa: float = 2.0,
        gamma: float = 1.5,
        delta: float = 2.0,
        alpha_y: float = 1.0,
        min_multiplier: float = 2.0,
        random_state: int = 42,
    ) -> None:
        self.k = k
        self.q_percentile = q_percentile
        self.hop_count = hop_count
        self.kappa = kappa
        self.gamma = gamma
        self.delta = delta
        self.alpha_y = alpha_y
        self.min_multiplier = min_multiplier
        self.random_state = np.random.default_rng(random_state)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> GraphCalResult:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n, d = X.shape
        k = self.k or min(int(math.ceil(2.0 * math.sqrt(n))), 50)
        k = max(2, min(k, n - 1))

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        Xn = x_scaler.fit_transform(X)
        yn = y_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)

        z = np.concatenate([Xn, (self.alpha_y * yn).reshape(-1, 1)], axis=1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nbrs.fit(z)
        dists, idxs = nbrs.kneighbors(z)
        # Ignore self-neighbor at column 0.
        knn_dists = dists[:, 1:]
        knn_idxs = idxs[:, 1:]

        density = 1.0 / (np.mean(knn_dists, axis=1) + 1e-12)
        tau = float(np.percentile(density, self.q_percentile))
        sparse_mask = density < tau
        sparse_indices = np.where(sparse_mask)[0]

        if sparse_indices.size == 0:
            return GraphCalResult(
                X_aug=X.copy(),
                y_aug=y.copy(),
                diagnostics={
                    "k": k,
                    "tau": tau,
                    "num_sparse_nodes": 0,
                    "num_generated": 0,
                    "num_accepted": 0,
                },
            )

        # Symmetric adjacency for L-hop neighborhoods.
        adjacency: list[set[int]] = [set() for _ in range(n)]
        for i in range(n):
            for j in knn_idxs[i]:
                j_int = int(j)
                adjacency[i].add(j_int)
                adjacency[j_int].add(i)

        # Real-node median graph distance for rejection threshold.
        real_graph_dist = np.mean(knn_dists, axis=1)
        real_graph_med = float(np.median(real_graph_dist))

        generated_X: list[np.ndarray] = []
        generated_y: list[float] = []
        num_generated = 0
        num_accepted = 0

        for i in sparse_indices:
            rho_i = density[i]
            budget = int(math.ceil(self.kappa * max(0.0, (tau - rho_i) / max(tau, 1e-12))))
            if budget <= 0:
                continue
            candidate_count = int(math.ceil(self.gamma * budget))

            neighborhood = self._l_hop_neighbors(i, adjacency, self.hop_count)
            min_required = int(math.ceil(self.min_multiplier * (d + 1)))
            hop = self.hop_count
            while len(neighborhood) < min_required and hop < 4:
                hop += 1
                neighborhood = self._l_hop_neighbors(i, adjacency, hop)

            neigh_idx = np.array(sorted(neighborhood), dtype=int)
            z_local = z[neigh_idx]

            mu = np.mean(z_local, axis=0)
            lw = LedoitWolf().fit(z_local)
            cov = lw.covariance_

            # Over-generate, then verify + select diverse points.
            samples = self.random_state.multivariate_normal(mu, cov, size=candidate_count)
            num_generated += candidate_count
            accepted = []
            for cand in samples:
                if self._accept_candidate(cand, z, real_graph_med, k):
                    accepted.append(cand)

            if not accepted:
                continue
            selected = self._select_farthest(np.array(accepted), budget, z)
            num_accepted += selected.shape[0]

            x_sel = selected[:, :-1]
            y_sel = selected[:, -1] / max(self.alpha_y, 1e-12)
            x_back = x_scaler.inverse_transform(x_sel)
            y_back = y_scaler.inverse_transform(y_sel.reshape(-1, 1)).reshape(-1)

            generated_X.extend(x_back)
            generated_y.extend(y_back.tolist())

        if generated_X:
            X_syn = np.asarray(generated_X, dtype=np.float64)
            y_syn = np.asarray(generated_y, dtype=np.float64)
            X_aug = np.vstack([X, X_syn])
            y_aug = np.concatenate([y, y_syn])
        else:
            X_aug = X.copy()
            y_aug = y.copy()

        return GraphCalResult(
            X_aug=X_aug,
            y_aug=y_aug,
            diagnostics={
                "k": k,
                "tau": tau,
                "num_sparse_nodes": int(sparse_indices.size),
                "num_generated": int(num_generated),
                "num_accepted": int(num_accepted),
                "augmentation_ratio": float((len(y_aug) - len(y)) / max(len(y), 1)),
            },
        )

    def _l_hop_neighbors(self, src: int, adjacency: list[set[int]], hop_count: int) -> set[int]:
        visited = {src}
        frontier = {src}
        for _ in range(hop_count):
            nxt = set()
            for u in frontier:
                nxt.update(adjacency[u])
            nxt -= visited
            if not nxt:
                break
            visited.update(nxt)
            frontier = nxt
        return visited

    def _accept_candidate(self, cand: np.ndarray, z_real: np.ndarray, real_graph_med: float, k: int) -> bool:
        # Graph-distance check.
        d = np.linalg.norm(z_real - cand, axis=1)
        nearest = np.partition(d, k)[:k]
        graph_dist = float(np.mean(nearest))
        if graph_dist > self.delta * real_graph_med:
            return False

        # Feature-target consistency check with local linear fit.
        idx = np.argpartition(d, k)[:k]
        x_local = z_real[idx, :-1]
        y_local = z_real[idx, -1]
        reg = LinearRegression().fit(x_local, y_local)
        y_hat = float(reg.predict(cand[:-1].reshape(1, -1))[0])
        resid = y_local - reg.predict(x_local)
        resid_std = float(np.std(resid) + 1e-8)
        return abs(float(cand[-1]) - y_hat) <= 3.0 * resid_std

    def _select_farthest(self, accepted: np.ndarray, budget: int, anchors: np.ndarray) -> np.ndarray:
        if accepted.shape[0] <= budget:
            return accepted

        selected = [accepted[0]]
        remaining = accepted[1:].copy()
        while len(selected) < budget and remaining.shape[0] > 0:
            base = np.vstack([anchors, np.asarray(selected)])
            dist_mat = np.linalg.norm(remaining[:, None, :] - base[None, :, :], axis=2)
            min_dist = np.min(dist_mat, axis=1)
            pick = int(np.argmax(min_dist))
            selected.append(remaining[pick])
            remaining = np.delete(remaining, pick, axis=0)
        return np.asarray(selected)
