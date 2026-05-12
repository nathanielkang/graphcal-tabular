"""Targeted MLP-only benchmark for bike/energy/wine.

Compares:
- Vanilla MLP
- Calibrated Mixup + MLP
- GraphCal + MLP
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graphcal import CalibratedMixupAugmenter, GraphCalAugmenter
from src.graphcal.datasets import load_dataset
from src.graphcal.metrics import mae, tail_mae


DATASETS = ["bike", "energy", "wine"]
SEEDS = [42, 123, 456]


def build_mlp(seed: int, hidden_layer_sizes=(256, 128), alpha: float = 1e-3) -> TransformedTargetRegressor:
    base = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size="auto",
        learning_rate_init=1e-3,
        max_iter=1200,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
        random_state=seed,
    )
    model = Pipeline([("x_scaler", StandardScaler()), ("mlp", base)])
    return TransformedTargetRegressor(regressor=model, transformer=StandardScaler())


def pick_mlp_config(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> dict:
    """
    Fast config search on a train/val split.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
    candidates = [
        {"hidden_layer_sizes": (128, 64), "alpha": 5e-4},
        {"hidden_layer_sizes": (256, 128), "alpha": 1e-3},
        {"hidden_layer_sizes": (256, 128, 64), "alpha": 2e-3},
        {"hidden_layer_sizes": (512, 256), "alpha": 2e-3},
    ]
    best = None
    best_mae = float("inf")
    for c in candidates:
        model = build_mlp(seed=seed, hidden_layer_sizes=c["hidden_layer_sizes"], alpha=c["alpha"])
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        m = mean_absolute_error(y_val, pred)
        if m < best_mae:
            best_mae = m
            best = c
    assert best is not None
    return best


def pick_graphcal_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    mlp_config: dict,
) -> dict:
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
    candidates = [
        {"kappa": 2.2, "gamma": 1.8, "q_percentile": 35.0, "delta": 2.0, "min_multiplier": 2.0},
        {"kappa": 1.6, "gamma": 1.2, "q_percentile": 20.0, "delta": 1.2, "min_multiplier": 1.5},
        {"kappa": 1.2, "gamma": 0.9, "q_percentile": 12.0, "delta": 0.9, "min_multiplier": 1.2},
        {"kappa": 0.9, "gamma": 0.6, "q_percentile": 8.0, "delta": 0.7, "min_multiplier": 1.0},
        {"kappa": 0.7, "gamma": 0.4, "q_percentile": 5.0, "delta": 0.6, "min_multiplier": 1.0},
    ]

    # Baseline val score (no augmentation) for safety fallback.
    vanilla_model = build_mlp(
        seed=seed,
        hidden_layer_sizes=mlp_config["hidden_layer_sizes"],
        alpha=mlp_config["alpha"],
    )
    vanilla_model.fit(X_tr, y_tr)
    vanilla_pred = vanilla_model.predict(X_val)
    vanilla_mae = mean_absolute_error(y_val, vanilla_pred)

    best = None
    best_mae = float("inf")
    for cfg in candidates:
        aug = GraphCalAugmenter(
            kappa=cfg["kappa"],
            gamma=cfg["gamma"],
            q_percentile=cfg["q_percentile"],
            delta=cfg["delta"],
            min_multiplier=cfg["min_multiplier"],
            random_state=seed,
        )
        aug_data = aug.fit_transform(X_tr, y_tr)
        model = build_mlp(
            seed=seed,
            hidden_layer_sizes=mlp_config["hidden_layer_sizes"],
            alpha=mlp_config["alpha"],
        )
        model.fit(aug_data.X_aug, aug_data.y_aug)
        pred = model.predict(X_val)
        m = mean_absolute_error(y_val, pred)
        if m < best_mae:
            best_mae = m
            best = cfg
    assert best is not None
    # Safety gate: require meaningful validation gain to enable augmentation.
    # Otherwise fallback to no-augmentation to avoid overfitting risk.
    if best_mae >= 0.95 * vanilla_mae:
        return {"fallback_no_aug": True, "val_mae": float(vanilla_mae)}
    best["fallback_no_aug"] = False
    best["val_mae"] = float(best_mae)
    return best


def run_one(dataset: str, seed: int) -> dict:
    X_train, X_test, y_train, y_test = load_dataset(dataset, seed=seed)
    config = pick_mlp_config(X_train, y_train, seed=seed)
    graphcal_cfg = pick_graphcal_config(X_train, y_train, seed=seed, mlp_config=config)

    t0 = perf_counter()
    # Vanilla
    vanilla = build_mlp(seed=seed, hidden_layer_sizes=config["hidden_layer_sizes"], alpha=config["alpha"])
    vanilla.fit(X_train, y_train)
    pred_v = vanilla.predict(X_test)

    # Calibrated Mixup
    calmix_aug = CalibratedMixupAugmenter(mixup_ratio=0.7, alpha=0.7, pair_k=32, tail_power=1.2, random_state=seed)
    calmix_data = calmix_aug.fit_transform(X_train, y_train)
    calmix = build_mlp(seed=seed, hidden_layer_sizes=config["hidden_layer_sizes"], alpha=config["alpha"])
    calmix.fit(calmix_data.X_aug, calmix_data.y_aug)
    pred_c = calmix.predict(X_test)

    # GraphCal
    if graphcal_cfg.get("fallback_no_aug", False):
        graph_data = type(
            "GraphData",
            (),
            {"X_aug": X_train, "y_aug": y_train, "diagnostics": {"fallback_no_aug": True}},
        )()
    else:
        graphcal_aug = GraphCalAugmenter(
            kappa=graphcal_cfg["kappa"],
            gamma=graphcal_cfg["gamma"],
            q_percentile=graphcal_cfg["q_percentile"],
            delta=graphcal_cfg["delta"],
            min_multiplier=graphcal_cfg["min_multiplier"],
            random_state=seed,
        )
        graph_data = graphcal_aug.fit_transform(X_train, y_train)
    graphcal = build_mlp(seed=seed, hidden_layer_sizes=config["hidden_layer_sizes"], alpha=config["alpha"])
    graphcal.fit(graph_data.X_aug, graph_data.y_aug)
    pred_g = graphcal.predict(X_test)

    elapsed = perf_counter() - t0
    return {
        "dataset": dataset,
        "seed": seed,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "seconds": round(elapsed, 3),
        "mlp_config": config,
        "graphcal_config": graphcal_cfg,
        "vanilla": {"mae": mae(y_test, pred_v), "tail_mae_95": tail_mae(y_test, pred_v, q=0.95)},
        "calibrated_mixup": {"mae": mae(y_test, pred_c), "tail_mae_95": tail_mae(y_test, pred_c, q=0.95)},
        "graphcal": {"mae": mae(y_test, pred_g), "tail_mae_95": tail_mae(y_test, pred_g, q=0.95)},
        "calmix_diagnostics": calmix_data.diagnostics,
        "graphcal_diagnostics": graph_data.diagnostics,
    }


def summarize(rows: list[dict]) -> dict:
    out = {}
    for ds in DATASETS:
        drows = [r for r in rows if r["dataset"] == ds]
        out[ds] = {}
        for method in ("vanilla", "calibrated_mixup", "graphcal"):
            maes = [r[method]["mae"] for r in drows]
            tails = [r[method]["tail_mae_95"] for r in drows]
            out[ds][method] = {
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes)),
                "tail_mae_95_mean": float(np.mean(tails)),
                "tail_mae_95_std": float(np.std(tails)),
            }
        g_wins_over_c = sum(1 for r in drows if r["graphcal"]["mae"] < r["calibrated_mixup"]["mae"])
        out[ds]["graphcal_win_vs_calmix_seeds"] = int(g_wins_over_c)
    return out


def main() -> None:
    rows = []
    for ds in DATASETS:
        print(f"\n=== DATASET {ds} ===", flush=True)
        for seed in SEEDS:
            print(f"  -> seed={seed} ...", flush=True)
            row = run_one(ds, seed)
            rows.append(row)
            print(
                "     done: "
                f"V={row['vanilla']['mae']:.4f}, "
                f"C={row['calibrated_mixup']['mae']:.4f}, "
                f"G={row['graphcal']['mae']:.4f}",
                flush=True,
            )
    output = {
        "datasets": DATASETS,
        "seeds": SEEDS,
        "runs": rows,
        "summary_by_dataset": summarize(rows),
    }
    out_path = ROOT / "results_targeted_mlp_graphcal_vs_calmix.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results: {out_path}")


if __name__ == "__main__":
    main()
