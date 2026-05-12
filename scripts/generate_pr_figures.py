#!/usr/bin/env python3
"""
Generate matplotlib figures 3–6 for GraphCal manuscripts (optional).

Output directory: set environment variable GRAPHCAL_FIGURE_DIR to an absolute
path; otherwise files go to ``<repo>/figures/``. Basenames:
  - graphcal_fig3_sensitivity.{png,pdf}
  - graphcal_fig4_tsne.{png,pdf}
  - graphcal_fig5_perregion.{png,pdf}
  - graphcal_fig6_distributions.{png,pdf}

Styling: Times New Roman (serif), 300 DPI, thin spines; Fig.4 uses gridspec_kw
right margin + manual colorbar (no tight_layout + shared colorbar).

Data sources (see module docstrings per figure):
  - Fig.3: on-the-fly Abalone sweep (optional local cache JSON; gitignored).
  - Fig.4: sklearn/OpenML Abalone + local augmenters.
  - Fig.5: California housing + KDE-based rarity bins (SMOGN/PRIME not bundled).
  - Fig.6: nine datasets via src.graphcal.datasets loaders.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent.parent
_fig = os.environ.get("GRAPHCAL_FIGURE_DIR", "").strip()
FIG_DIR = Path(_fig) if _fig else (CODE_ROOT / "figures")
CACHE_SENS = CODE_ROOT / "results_fig3_sensitivity_abalone_cache.json"

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from sklearn.manifold import TSNE  # noqa: E402
from sklearn.neighbors import KernelDensity  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.graphcal import GraphCalAugmenter  # noqa: E402
from src.graphcal.datasets import load_dataset  # noqa: E402
from src.graphcal.metrics import mae  # noqa: E402


@dataclass
class _PairwiseMixupResult:
    """Minimal pairwise Beta-mixup augmentation (generic baseline for figures only)."""

    X_aug: np.ndarray
    y_aug: np.ndarray


def _pairwise_regression_mixup(
    X: np.ndarray,
    y: np.ndarray,
    *,
    mixup_ratio: float = 0.5,
    alpha: float = 0.3,
    random_state: int = 0,
) -> _PairwiseMixupResult:
    """Convex combinations of random training pairs (features and target jointly)."""
    rng = np.random.default_rng(random_state)
    n = int(X.shape[0])
    n_new = max(1, int(round(float(mixup_ratio) * n)))
    rows_x = [np.asarray(X, dtype=np.float64)]
    rows_y = [np.asarray(y, dtype=np.float64).ravel()]
    for _ in range(n_new):
        i, j = rng.choice(n, size=2, replace=False)
        lam = float(rng.beta(alpha, alpha))
        rows_x.append(lam * X[i] + (1.0 - lam) * X[j])
        rows_y.append(lam * y.ravel()[i] + (1.0 - lam) * y.ravel()[j])
    Xa = np.vstack(rows_x)
    ya = np.concatenate(rows_y)
    return _PairwiseMixupResult(X_aug=Xa, y_aug=ya)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Nimbus Roman"],
            "mathtext.fontset": "dejavuserif",
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )
    return plt


def _save(fig, stem: str, plt) -> tuple[Path, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    return png, pdf


def fig3_sensitivity(plt) -> tuple[Path, Path]:
    """
    Fig.3: MAE vs k, q, L, delta (one-at-a-time) on Abalone.
    No hyperparameter sweep exists in results_all_datasets_graphcal.json (only
    single diagnostics per run); this figure uses a fresh Abalone evaluation
    matching run_all_datasets MLP (128,128), max_iter=50, 3 seeds for mean±std.
    """
    dataset_name = "abalone"
    seeds = (42, 123, 456)
    grids = {
        "k": [5, 10, 20, 30, 50, 75, 100],
        "q": [10, 15, 20, 25, 30, 35, 40, 45, 50],
        "L": [1, 2, 3, 4],
        "d": [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
    }
    defaults = {"q_percentile": 30.0, "hop_count": 2, "delta": 2.0}

    def build_aug(param: str, val, seed: int) -> GraphCalAugmenter:
        kw: dict = {"random_state": seed}
        if param != "k":
            kw["k"] = None
        else:
            kw["k"] = int(val)
        if param != "q":
            kw["q_percentile"] = defaults["q_percentile"]
        else:
            kw["q_percentile"] = float(val)
        if param != "L":
            kw["hop_count"] = defaults["hop_count"]
        else:
            kw["hop_count"] = int(val)
        if param != "d":
            kw["delta"] = defaults["delta"]
        else:
            kw["delta"] = float(val)
        return GraphCalAugmenter(**kw)

    def vanilla_mae_for_seed(seed: int) -> float:
        X_tr, X_te, y_tr, y_te = load_dataset(dataset_name, seed)
        m = MLPRegressor(
            hidden_layer_sizes=(128, 128),
            max_iter=50,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
        m.fit(X_tr, y_tr)
        return mae(y_te, m.predict(X_te))

    if CACHE_SENS.exists():
        with CACHE_SENS.open(encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {"vanilla_mae": {}, "curves": {}}
        for s in seeds:
            cache["vanilla_mae"][str(s)] = vanilla_mae_for_seed(s)

        def curve_for(param: str, values: list):
            rows = []
            for val in values:
                ys = []
                for s in seeds:
                    X_tr, X_te, y_tr, y_te = load_dataset(dataset_name, s)
                    aug = build_aug(param, val, s)
                    res = aug.fit_transform(X_tr, y_tr)
                    mlp = MLPRegressor(
                        hidden_layer_sizes=(128, 128),
                        max_iter=50,
                        random_state=s,
                        early_stopping=True,
                        validation_fraction=0.1,
                    )
                    mlp.fit(res.X_aug, res.y_aug)
                    ys.append(mae(y_te, mlp.predict(X_te)))
                rows.append(
                    {
                        "x": val,
                        "mean": float(np.mean(ys)),
                        "std": float(np.std(ys, ddof=0)),
                    }
                )
            return rows

        cache["curves"]["k"] = curve_for("k", grids["k"])
        cache["curves"]["q"] = curve_for("q", grids["q"])
        cache["curves"]["L"] = curve_for("L", grids["L"])
        cache["curves"]["d"] = curve_for("d", grids["d"])
        with CACHE_SENS.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    vanilla_mean = float(np.mean([cache["vanilla_mae"][str(s)] for s in seeds]))

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), sharey=False)
    panels = [
        ("k", "k (neighbors)", axes[0, 0], 50),
        ("q", r"$q$ (sparsity percentile)", axes[0, 1], 30),
        ("L", r"$L$ (hop radius)", axes[1, 0], 2),
        ("d", r"$\delta$ (rejection multiplier)", axes[1, 1], 2.0),
    ]
    for key, xlabel, ax, def_x in panels:
        rows = cache["curves"][key]
        xs = [r["x"] for r in rows]
        means = [r["mean"] for r in rows]
        stds = [r["std"] for r in rows]
        ax.plot(xs, means, "o-", color="#1f77b4", lw=1.2, ms=4)
        ax.fill_between(xs, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2, color="#1f77b4")
        ax.axhline(vanilla_mean, color="#7f7f7f", ls="--", lw=1.0, label="Vanilla MLP (mean)")
        ax.axvline(def_x, color="#d62728", ls=":", lw=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test MAE")
        ax.grid(True, axis="y", alpha=0.35, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].legend(loc="upper right", fontsize=7)

    fig.text(
        0.99,
        0.01,
        "Dataset: Abalone (no sweep JSON; computed here). Shaded: ±1 std over seeds 42,123,456.",
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#444444",
    )
    for i, lab in enumerate("abcd"):
        ax = axes.flatten()[i]
        ax.text(0.02, 0.95, f"({lab})", transform=ax.transAxes, fontsize=9, va="top", fontweight="bold")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return _save(fig, "graphcal_fig3_sensitivity", plt)


def _joint_z(X: np.ndarray, y: np.ndarray, alpha_y: float = 1.0) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    ys = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
    return np.concatenate([Xs, (alpha_y * ys).reshape(-1, 1)], axis=1)


def fig4_tsne(plt) -> tuple[Path, Path]:
    """
    Fig.4: t-SNE of real + synthetic points (Abalone train).
    Two panels: a generic pairwise mixup visual baseline vs GraphCal (separate
    embeddings per panel). Perplexity 30, seed 42.
    """
    from matplotlib import gridspec
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    seed = 42
    perplexity = 30
    rng = np.random.default_rng(seed)
    X_tr, _, y_tr, _ = load_dataset("abalone", seed)
    n_real_cap = 1800
    if X_tr.shape[0] > n_real_cap:
        idx = rng.choice(X_tr.shape[0], size=n_real_cap, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    def subsample_syn(Xa: np.ndarray, ya: np.ndarray, n_real: int, cap: int = 500):
        syn_X, syn_y = Xa[n_real:], ya[n_real:]
        if syn_X.shape[0] <= cap:
            return syn_X, syn_y
        j = rng.choice(syn_X.shape[0], size=cap, replace=False)
        return syn_X[j], syn_y[j]

    n_real = X_tr.shape[0]
    mix_vis = _pairwise_regression_mixup(
        X_tr, y_tr, mixup_ratio=0.5, alpha=0.3, random_state=seed
    )
    gcal = GraphCalAugmenter(random_state=seed).fit_transform(X_tr, y_tr)

    panels = [
        ("(a) Original + pairwise mixup", mix_vis.X_aug, mix_vis.y_aug, "o", "^"),
        ("(b) Original + GraphCal", gcal.X_aug, gcal.y_aug, "o", "^"),
    ]

    fig = plt.figure(figsize=(5.2, 2.55))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.06], wspace=0.28, right=0.88)
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    cax = fig.add_subplot(gs[0, 2])

    y_all = np.concatenate([y_tr, mix_vis.y_aug[n_real:], gcal.y_aug[n_real:]])
    norm = Normalize(vmin=float(np.min(y_all)), vmax=float(np.max(y_all)))
    cmap = plt.cm.viridis

    for ax, (title, Xa, ya, m_real, m_syn) in zip(axes, panels):
        sx, sy = subsample_syn(Xa, ya, n_real)
        Z = _joint_z(np.vstack([X_tr, sx]), np.hstack([y_tr, sy]))
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, Z.shape[0] - 1),
            random_state=seed,
            init="pca",
            learning_rate="auto",
        )
        emb = tsne.fit_transform(Z)
        emb_real, emb_syn = emb[:n_real], emb[n_real:]
        ax.scatter(emb_real[:, 0], emb_real[:, 1], c=y_tr, cmap=cmap, norm=norm, s=10, marker=m_real, edgecolors="none", alpha=0.75)
        ax.scatter(emb_syn[:, 0], emb_syn[:, 1], c=sy, cmap=cmap, norm=norm, s=14, marker=m_syn, edgecolors="k", linewidths=0.15, alpha=0.85)
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.25, lw=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Target (rings)", fontsize=8)

    fig.suptitle("Abalone — training subset (70% split)", fontsize=9, y=1.02)
    fig.text(0.5, -0.02, f"t-SNE perplexity = {perplexity}, random_state = {seed}", ha="center", fontsize=7, color="#333333")

    # Manual layout: avoid tight_layout with colorbar
    fig.subplots_adjust(left=0.06, right=0.86, bottom=0.22, top=0.82, wspace=0.35)
    return _save(fig, "graphcal_fig4_tsne", plt)


def _rarity_bins_equal_count(y_train: np.ndarray, y_test: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Return bin id 0..n_bins-1 where 0 = rarest (low KDE), n_bins-1 = densest."""
    bw = max(np.std(y_train) * 0.12, 1e-6)
    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(y_train.reshape(-1, 1))
    logd = kde.score_samples(y_test.reshape(-1, 1))
    dens = np.exp(logd)
    rarity = 1.0 / (dens + 1e-12)
    order = np.argsort(-rarity)
    n = len(y_test)
    bin_ids = np.zeros(n, dtype=int)
    for b in range(n_bins):
        sl = order[b * n // n_bins : (b + 1) * n // n_bins]
        bin_ids[sl] = b
    return bin_ids


def fig5_perregion(plt) -> tuple[Path, Path]:
    """
    Per-bin MAE (KDE rarity bins: bin 1 = rare, 5 = dense center).
    California; methods: Vanilla, pairwise mixup (figure baseline), GraphCal.
    """
    seed = 42
    X_tr, X_te, y_tr, y_te = load_dataset("california", seed)
    bins = _rarity_bins_equal_count(y_tr, y_te, 5)

    def train_predict_mae(Xa, ya):
        m = MLPRegressor(
            hidden_layer_sizes=(256, 128),
            max_iter=120,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
        m.fit(Xa, ya)
        pred = m.predict(X_te)
        per_bin = []
        for b in range(5):
            msk = bins == b
            if msk.sum() == 0:
                per_bin.append(float("nan"))
            else:
                per_bin.append(mae(y_te[msk], pred[msk]))
        return per_bin

    mix_vis = _pairwise_regression_mixup(
        X_tr, y_tr, mixup_ratio=0.5, alpha=0.3, random_state=seed
    )
    gres = GraphCalAugmenter(random_state=seed).fit_transform(X_tr, y_tr)

    series = {
        "Vanilla": train_predict_mae(X_tr, y_tr),
        "Pairwise mixup": train_predict_mae(mix_vis.X_aug, mix_vis.y_aug),
        "GraphCal": train_predict_mae(gres.X_aug, gres.y_aug),
    }

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(1, 6)
    n_m = len(series)
    width = 0.22
    colors = ["#b0b0b0", "#1f77b4", "#d62728"]
    for i, (name, vals) in enumerate(series.items()):
        offset = (i - (n_m - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=name, color=colors[i % len(colors)], edgecolor="0.25", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Bin {j}" for j in x])
    ax.set_xlabel("Target rarity bin (1 = rare tail, 5 = dense center)")
    ax.set_ylabel("Test MAE")
    ax.legend(loc="upper right", fontsize=7.5)
    ax.grid(True, axis="y", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.02,
        0.02,
        "Bins: test-point KDE rarity vs. train target (California).\n"
        "Pairwise mixup: generic convex pair baseline for illustration (not manuscript-specific).",
        transform=ax.transAxes,
        fontsize=6.5,
        va="bottom",
        color="#333333",
    )
    fig.tight_layout()
    return _save(fig, "graphcal_fig5_perregion", plt)


def _skew(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    m = float(np.mean(a))
    s = float(np.std(a))
    if s < 1e-12:
        return 0.0
    z = (a - m) / s
    return float(np.mean(z**3))


def fig6_distributions(plt) -> tuple[Path, Path]:
    """3×3 histogram + KDE for nine paper datasets (training targets, 70% split)."""
    order = [
        ("abalone", "Abalone", "rings"),
        ("california", "California", r"MedHouseVal ($\times 10^{5}$ USD)"),
        ("bike", "Bike Sharing", "count (rentals)"),
        ("wine", "Wine Quality", "score (discrete)"),
        ("energy", "Energy", "Y1 (kWh, load)"),
        ("airfoil", "Airfoil", "scaled sound pressure"),
        ("superconductor", "Superconductivity", r"$T_c$ (K)"),
        ("cpu", "CPU", "relative perf."),
        ("forest_fires", "Forest Fires", r"burned area ($\sqrt{\mathrm{ha}}$)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(7.4, 6.6), sharex=False, sharey=False)
    axes = axes.ravel()

    for ax, (key, title, xunit) in zip(axes, order):
        X_tr, _, y_tr, _ = load_dataset(key, seed=42)
        y = y_tr.ravel()
        y_f = y[np.isfinite(y)]
        sk = _skew(y_f)
        # Forest fires often log1p in papers; keep raw per OpenML pipeline in datasets.py
        ax.hist(y_f, bins=40, density=True, color="#aec7e8", edgecolor="#7f7f7f", linewidth=0.3, label="Histogram")
        xs = np.linspace(np.min(y_f), np.max(y_f), 200)
        if len(np.unique(y_f)) < 12:
            hist, edges = np.histogram(y_f, bins=min(20, len(np.unique(y_f))), density=True)
            cx = (edges[:-1] + edges[1:]) / 2
            ax.plot(cx, hist, drawstyle="steps-mid", color="#1f3763", lw=1.4, label="Step density")
        else:
            bw = max(np.std(y_f) * 0.2, 1e-6)
            kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(y_f.reshape(-1, 1))
            logp = kde.score_samples(xs.reshape(-1, 1))
            ax.plot(xs, np.exp(logp), color="#1f3763", lw=1.4, label="KDE")

        q05, q95 = np.quantile(y_f, [0.05, 0.95])
        if sk > 0.5:
            ax.axvspan(q95, np.max(y_f), color="#ffb6c1", alpha=0.35, lw=0)
        elif sk < -0.5:
            ax.axvspan(np.min(y_f), q05, color="#ffb6c1", alpha=0.35, lw=0)
        else:
            ax.axvspan(np.min(y_f), q05, color="#ffb6c1", alpha=0.22, lw=0)
            ax.axvspan(q95, np.max(y_f), color="#ffb6c1", alpha=0.22, lw=0)

        med = float(np.median(y_f))
        ax.axvline(med, color="k", ls="--", lw=1.0)
        ylo, yhi = ax.get_ylim()
        ax.text(med, ylo + 0.04 * (yhi - ylo), "median", rotation=90, va="bottom", ha="right", fontsize=6)

        ax.set_title(f"{title}\n({xunit})", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25)

    from matplotlib.patches import Patch

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="#ffb6c1", alpha=0.45, edgecolor="none"))
    labels.append("Highlighted tail region")
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=8)
    fig.supxlabel("Target value", fontsize=9)
    fig.supylabel("Density", fontsize=9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        fig.tight_layout(rect=[0.02, 0.02, 0.95, 0.95])

    return _save(fig, "graphcal_fig6_distributions", plt)


def main() -> None:
    plt = _setup_matplotlib()
    out = []
    out.append(("Fig.3", fig3_sensitivity(plt)))
    out.append(("Fig.4", fig4_tsne(plt)))
    out.append(("Fig.5", fig5_perregion(plt)))
    out.append(("Fig.6", fig6_distributions(plt)))
    for name, (png, pdf) in out:
        print(f"{name}: {png}\n       {pdf}")


if __name__ == "__main__":
    main()
