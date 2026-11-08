from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf, empirical_covariance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def k_nearest_indices(distances: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k smallest distances. Exact ties keep the smallest index."""
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    if k < 1 or k > distances.size:
        raise ValueError("k must lie between 1 and the number of distances")
    order = np.lexsort((np.arange(distances.size), distances))
    return order[:k]


def voronoi_indices(query: np.ndarray, reference: np.ndarray, block: int = 4096) -> np.ndarray:
    """Nearest reference row for each query.

    Equal Euclidean distances resolve to the smallest reference index.
    Squared distances are compared in reference order, and a later block
    replaces an earlier one only when it is strictly closer.
    """
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("Voronoi queries and references must share a coordinate dimension")
    if reference.shape[0] < 1 or query.shape[0] < 1:
        raise ValueError("Voronoi assignment needs at least one query and one reference row")
    best_sq = np.full(query.shape[0], np.inf)
    best_ix = np.zeros(query.shape[0], dtype=int)
    q_sq = np.einsum("ij,ij->i", query, query)
    r_sq = np.einsum("ij,ij->i", reference, reference)
    for start in range(0, reference.shape[0], block):
        stop = min(start + block, reference.shape[0])
        dist_sq = q_sq[:, None] + r_sq[start:stop][None, :] - 2.0 * query @ reference[start:stop].T
        local = np.argmin(dist_sq, axis=1)
        local_sq = dist_sq[np.arange(query.shape[0]), local]
        better = local_sq < best_sq
        best_sq[better] = local_sq[better]
        best_ix[better] = local[better] + start
    return best_ix


def scaled_identity_ledoit_wolf(samples: np.ndarray) -> np.ndarray:
    """MLE covariance shrunk toward (trace(S) / p) I.

    The coefficient is the Ledoit-Wolf intensity. The target is the scaled
    identity, not diag(S) and not the OAS coefficient.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("Ledoit-Wolf needs a nonempty neighborhood matrix")
    n, p = x.shape
    if n < 2:
        return np.zeros((p, p), dtype=np.float64)
    S = empirical_covariance(x)
    scale = float(np.trace(S) / p)
    if not np.isfinite(S).all() or scale <= 0.0:
        return np.zeros((p, p), dtype=np.float64)
    lam = float(LedoitWolf(store_precision=False).fit(x).shrinkage_)
    if not np.isfinite(lam):
        lam = 1.0
    lam = float(np.clip(lam, 0.0, 1.0))
    shrunk = (1.0 - lam) * S
    shrunk.flat[:: p + 1] += lam * scale
    return shrunk


def _neighbor_tables(points: np.ndarray, k: int, block: int = 512):
    """k nearest other rows. Exact distance ties keep the smallest index."""
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if k < 1 or k > n - 1:
        raise ValueError("Neighborhood size must exclude self and stay inside the sample")
    nn_idx = np.empty((n, k), dtype=int)
    nn_dist = np.empty((n, k), dtype=np.float64)
    sq = np.einsum("ij,ij->i", points, points)
    for start in range(0, n, block):
        stop = min(start + block, n)
        block_x = points[start:stop]
        dist_sq = sq[start:stop, None] + sq[None, :] - 2.0 * block_x @ points.T
        rows = np.arange(stop - start)
        dist_sq[rows, rows + start] = np.inf
        kth = np.partition(dist_sq, k - 1, axis=1)[:, k - 1]
        for i in range(stop - start):
            row = dist_sq[i]
            closer = np.flatnonzero(row < kth[i])
            tied = np.flatnonzero(row == kth[i])
            tied.sort()
            need = k - closer.size
            if need < 0 or tied.size < need:
                raise RuntimeError("k-nearest tie resolution could not fill the neighborhood")
            chosen = np.concatenate((closer, tied[:need]))
            chosen = chosen[np.lexsort((chosen, row[chosen]))]
            nn_idx[start + i] = chosen
            nn_dist[start + i] = np.sqrt(np.maximum(row[chosen], 0.0))
    return nn_dist, nn_idx


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
        max_hops: int = 4,
        covariance: str = "ledoit_wolf",
        verify: bool = True,
        adaptive_budget: bool = True,
        graph_space: str = "joint",
        distance_scale: str = "global",
        residual_screen: bool = True,
        ridge_residual: bool = False,
        ridge_alpha: float = 1.0,
        residual_sigma_mult: float = 3.0,
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
        self.max_hops = max_hops
        self.covariance = covariance
        self.verify = verify
        self.adaptive_budget = adaptive_budget
        self.graph_space = graph_space
        self.distance_scale = distance_scale
        self.residual_screen = residual_screen
        self.ridge_residual = ridge_residual
        self.ridge_alpha = float(ridge_alpha)
        self.residual_sigma_mult = float(residual_sigma_mult)
        self.seed = random_state

    def encode_joint(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fitted joint coordinates, including the target weight alpha_y."""
        if not hasattr(self, "x_scaler_"):
            raise RuntimeError("GraphCal geometry is not fitted")
        xn = self.x_scaler_.transform(np.asarray(X, dtype=np.float64))
        yn = self.y_scaler_.transform(np.asarray(y, dtype=np.float64).reshape(-1, 1)).reshape(-1)
        return np.column_stack((xn, self.alpha_y * yn))

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> GraphCalResult:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.size or y.size < 3:
            raise ValueError("GraphCal requires at least three paired rows")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("GraphCal inputs must be finite")
        if not 0 < self.alpha_y <= 1 or not 0 < self.q_percentile < 100:
            raise ValueError("Invalid target weight or sparsity percentile")
        if self.distance_scale not in ("global", "local"):
            raise ValueError("distance_scale must be global or local")
        if self.kappa <= 0 or self.gamma < 1 or self.delta <= 0:
            raise ValueError("Invalid budget, over-generation or distance tolerance")
        if not 1 <= self.hop_count <= self.max_hops:
            raise ValueError("Hop count must lie between one and max_hops")
        if self.graph_space not in {"joint", "features", "target"}:
            raise ValueError("Unknown graph space")
        self.random_state = np.random.default_rng(self.seed)
        n, d = X.shape
        k = self.k or min(int(math.ceil(2.0 * math.sqrt(n))), 50)
        k = max(2, min(k, n - 1))

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        Xn = x_scaler.fit_transform(X)
        yn = y_scaler.fit_transform(y.reshape(-1, 1)).reshape(-1)
        z = np.concatenate([Xn, (self.alpha_y * yn).reshape(-1, 1)], axis=1)
        p = int(z.shape[1])
        # One rule in every dimension: at least 2p joint rows, or four hops, then shrink.
        required = 2 * p
        if self.graph_space == "joint":
            graph_z = z
        elif self.graph_space == "features":
            graph_z = Xn
        else:
            graph_z = yn[:, None]
        knn_dists, knn_idxs = _neighbor_tables(graph_z, k)

        density = 1.0 / (np.mean(knn_dists, axis=1) + 1e-12)
        tau = float(np.percentile(density, self.q_percentile, method="linear"))
        sparse_indices = np.flatnonzero(density < tau)

        if self.graph_space == "joint":
            real_graph_dist = np.mean(knn_dists, axis=1)
        else:
            real_graph_dist = np.mean(_neighbor_tables(z, k)[0], axis=1)
        real_graph_med = float(np.median(real_graph_dist))
        self.train_mean_dist_ = np.asarray(real_graph_dist, dtype=np.float64)
        self._set_fitted_state(
            x_scaler, y_scaler, z, density, sparse_indices, k, self.delta * real_graph_med,
        )

        diagnostics = {
            "k": int(k),
            "tau": tau,
            "joint_dimension": p,
            "required_neighborhood": int(required),
            "max_hops": int(self.max_hops),
            "num_sparse_nodes": int(sparse_indices.size),
            "num_generated": 0,
            "num_accepted": 0,
            "anchors_below_2p": 0,
            "max_hops_used": 0,
            "augmentation_ratio": 0.0,
            "covariance": self.covariance,
            "distance_scale": self.distance_scale,
            "residual_screen": self.residual_screen,
        }
        if sparse_indices.size == 0:
            return GraphCalResult(X.copy(), y.copy(), diagnostics)

        adjacency: list[set[int]] = [set() for _ in range(n)]
        for i in range(n):
            for j in knn_idxs[i]:
                j_int = int(j)
                adjacency[i].add(j_int)
                adjacency[j_int].add(i)

        generated_X: list[np.ndarray] = []
        generated_y: list[float] = []
        num_generated = 0
        num_accepted = 0
        below_2p = 0
        max_hops_used = 0

        for i in sparse_indices[np.argsort(density[sparse_indices], kind="stable")]:
            rho_i = density[i]
            budget = int(math.ceil(self.kappa * max(0.0, (tau - rho_i) / max(tau, 1e-12))))
            if not self.adaptive_budget:
                budget = int(math.ceil(self.kappa))
            if budget <= 0:
                continue
            candidate_count = int(math.ceil(self.gamma * budget))

            hop = self.hop_count
            neighborhood = self._l_hop_neighbors(i, adjacency, hop)
            while len(neighborhood) < required and hop < self.max_hops:
                hop += 1
                neighborhood = self._l_hop_neighbors(i, adjacency, hop)
            max_hops_used = max(max_hops_used, hop)
            if len(neighborhood) < required:
                below_2p += 1

            neigh_idx = np.array(sorted(neighborhood), dtype=int)
            z_local = z[neigh_idx]
            mu = np.mean(z_local, axis=0)
            if self.covariance == "ledoit_wolf":
                cov = scaled_identity_ledoit_wolf(z_local)
            elif self.covariance == "empirical":
                cov = np.cov(z_local, rowvar=False, ddof=0) if z_local.shape[0] > 1 else np.zeros((p, p))
            elif self.covariance == "diagonal":
                cov = np.diag(np.var(z_local, axis=0))
            else:
                raise ValueError("Unknown covariance estimator")

            samples = self._draw_gaussian(mu, cov, candidate_count)
            num_generated += candidate_count
            accepted = []
            for cand in samples:
                # Both screens run before a candidate can be retained.
                if self.verify and not self.passes_screens(cand):
                    continue
                accepted.append(cand)

            if not accepted:
                continue
            previous = (
                np.column_stack((
                    x_scaler.transform(np.asarray(generated_X)),
                    self.alpha_y * y_scaler.transform(np.asarray(generated_y).reshape(-1, 1)),
                ))
                if generated_X else np.empty((0, d + 1))
            )
            selected = self._select_farthest(np.array(accepted), budget, np.vstack((z, previous)))
            num_accepted += selected.shape[0]

            x_sel = selected[:, :-1]
            y_sel = selected[:, -1] / max(self.alpha_y, 1e-12)
            x_back = x_scaler.inverse_transform(x_sel)
            y_back = y_scaler.inverse_transform(y_sel.reshape(-1, 1)).reshape(-1)
            generated_X.extend(x_back)
            generated_y.extend(y_back.tolist())

        if generated_X:
            X_aug = np.vstack([X, np.asarray(generated_X, dtype=np.float64)])
            y_aug = np.concatenate([y, np.asarray(generated_y, dtype=np.float64)])
        else:
            X_aug = X.copy()
            y_aug = y.copy()
        diagnostics.update({
            "num_generated": int(num_generated),
            "num_accepted": int(num_accepted),
            "anchors_below_2p": int(below_2p),
            "max_hops_used": int(max_hops_used),
            "augmentation_ratio": float((len(y_aug) - len(y)) / max(len(y), 1)),
        })
        return GraphCalResult(X_aug, y_aug, diagnostics)

    def _set_fitted_state(self, x_scaler, y_scaler, z, density, sparse_indices, k, threshold) -> None:
        self.x_scaler_ = x_scaler
        self.y_scaler_ = y_scaler
        self.reference_z_ = np.array(z, dtype=np.float64, copy=True)
        self.density_ = np.array(density, dtype=np.float64, copy=True)
        self.sparse_indices_ = np.array(sparse_indices, dtype=int, copy=True)
        self.k_ = int(k)
        self.threshold_ = float(threshold)

    def passes_screens(self, cand: np.ndarray) -> bool:
        """Global median distance screen, then the residual screen."""
        cand = np.asarray(cand, dtype=np.float64).reshape(-1)
        distances = np.linalg.norm(self.reference_z_ - cand, axis=1)
        idx = k_nearest_indices(distances, self.k_)
        scores = np.asarray(self.train_mean_dist_, dtype=np.float64)
        if self.distance_scale == "local":
            scale = float(np.median(scores[idx]))
        else:
            scale = float(np.median(scores))
        if float(distances[idx].mean()) > self.delta * max(scale, 1e-12):
            return False
        if not self.residual_screen:
            return True
        return self.residual_accepts(cand, idx)

    def residual_accepts(self, cand: np.ndarray, idx: np.ndarray) -> bool:
        x_local = self.reference_z_[idx, :-1]
        y_local = self.reference_z_[idx, -1]
        if self.ridge_residual:
            reg = Ridge(alpha=self.ridge_alpha).fit(x_local, y_local)
        else:
            reg = LinearRegression().fit(x_local, y_local)
        y_hat = float(reg.predict(np.asarray(cand[:-1], dtype=np.float64).reshape(1, -1))[0])
        resid = y_local - reg.predict(x_local)
        resid_std = float(np.std(resid) + 1e-8)
        return abs(float(cand[-1]) - y_hat) <= self.residual_sigma_mult * resid_std

    def _draw_gaussian(self, mu: np.ndarray, cov: np.ndarray, count: int) -> np.ndarray:
        sym = 0.5 * (np.asarray(cov, dtype=np.float64) + np.asarray(cov, dtype=np.float64).T)
        evals, evecs = np.linalg.eigh(sym)
        if np.isfinite(evals).all() and float(np.min(evals)) > 1e-12:
            return self.random_state.multivariate_normal(mu, cov, size=count)
        evals = np.clip(evals, 0.0, None)
        if not np.any(evals > 0.0):
            return np.repeat(np.asarray(mu, dtype=np.float64).reshape(1, -1), count, axis=0)
        noise = self.random_state.standard_normal((count, evals.size)) * np.sqrt(evals)
        return np.asarray(mu, dtype=np.float64) + noise @ evecs.T

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

    def _select_farthest(self, accepted: np.ndarray, budget: int, anchors: np.ndarray) -> np.ndarray:
        if accepted.shape[0] <= budget:
            return accepted

        nearest = NearestNeighbors(n_neighbors=1).fit(anchors)
        distances = nearest.kneighbors(accepted)[0].reshape(-1)
        first = int(np.argmax(distances))
        selected = [accepted[first]]
        remaining = np.delete(accepted, first, axis=0)
        while len(selected) < budget and remaining.shape[0] > 0:
            base = np.vstack([anchors, np.asarray(selected)])
            min_dist = NearestNeighbors(n_neighbors=1).fit(base).kneighbors(remaining)[0].reshape(-1)
            pick = int(np.argmax(min_dist))
            selected.append(remaining[pick])
            remaining = np.delete(remaining, pick, axis=0)
        return np.asarray(selected)
