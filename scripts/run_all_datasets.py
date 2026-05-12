"""Run GraphCal vs vanilla on all configured datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graphcal import GraphCalAugmenter
from src.graphcal.datasets import list_dataset_names, load_dataset
from src.graphcal.metrics import mae, tail_mae


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--datasets", nargs="+", default=list_dataset_names())
    parser.add_argument("--output", default="results_all_datasets_graphcal.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_started = perf_counter()
    result = {
        "datasets_requested": args.datasets,
        "seeds": args.seeds,
        "runs": {},
        "errors": [],
    }

    for dataset_name in args.datasets:
        print(f"\n=== DATASET: {dataset_name} ===", flush=True)
        result["runs"][dataset_name] = []
        for seed in args.seeds:
            t0 = perf_counter()
            print(f"  -> seed={seed} start", flush=True)
            try:
                X_train, X_test, y_train, y_test = load_dataset(dataset_name, seed)
                metrics = evaluate_pair(X_train, X_test, y_train, y_test, seed=seed)
                row = {
                    "seed": seed,
                    "n_train": int(X_train.shape[0]),
                    "n_test": int(X_test.shape[0]),
                    "n_features": int(X_train.shape[1]),
                    "seconds": round(perf_counter() - t0, 3),
                    **metrics,
                }
                result["runs"][dataset_name].append(row)
                print(
                    f"  -> seed={seed} done in {row['seconds']}s "
                    f"(rf_vanilla_mae={row['rf_vanilla']['mae']:.4f}, rf_graphcal_mae={row['rf_graphcal']['mae']:.4f})",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                msg = f"{dataset_name} seed={seed}: {exc}"
                result["errors"].append(msg)
                print(f"  !! ERROR {msg}", flush=True)

    result["total_seconds"] = round(perf_counter() - run_started, 3)
    out_path = ROOT / args.output
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results: {out_path}")
    print(f"Total runtime: {result['total_seconds']}s")
    if result["errors"]:
        print("\nEncountered errors:")
        for err in result["errors"]:
            print(f"- {err}")


if __name__ == "__main__":
    main()
