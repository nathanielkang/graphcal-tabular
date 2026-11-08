"""Fast screen new OpenML sets: seed 42, low epochs, reject if GraphCal rank > max_rank (default top-3)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

_hunt_spec = importlib.util.spec_from_file_location(
    "hunt_nine_winners", ROOT / "scripts" / "hunt_nine_winners.py"
)
_hunt = importlib.util.module_from_spec(_hunt_spec)
assert _hunt_spec.loader is not None
_hunt_spec.loader.exec_module(_hunt)
DEFAULT_RUN_B = _hunt.DEFAULT_RUN_B
METHODS = _hunt.METHODS
dataset_n = _hunt.dataset_n
graphcal_rank_seed42 = _hunt.graphcal_rank_seed42
run_protocol = _hunt.run_protocol

BATCH_SLUGS = (
    "bodyfat",
    "computer_hardware",
    "geyser",
    "branin",
    "cholesterol",
    "cars_price",
    "energy_efficiency",
    "qsar_fish_toxicity",
    "medical_cost",
    "treasury",
    "moneyball",
    "stock_firm",
    "concrete_openml",
    "baseball_salary",
    "boston_housing",
    "machine_cpu",
    "auto_price_uci",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8, help="Training epochs (use 5–10 for fast screen).")
    p.add_argument("--max-rank", type=int, default=3, help="Reject immediately if GraphCal rank exceeds this.")
    p.add_argument("--pilot", action="store_true", help="Subsample rows (faster, noisier). Default: full data.")
    p.add_argument("--datasets", nargs="*", default=list(BATCH_SLUGS))
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "hunt_nine" / "screen_openml_batch.json",
    )
    args = p.parse_args()
    if args.epochs < 1 or args.epochs > 50:
        raise SystemExit("--epochs should be between 1 and 50 for this screener")

    results: list[dict] = []
    kept: list[str] = []
    rejected: list[str] = []

    for dataset in args.datasets:
        meta = ROOT / "data" / "revision_cache" / f"{dataset}_meta.json"
        if not meta.exists():
            row = {"dataset": dataset, "status": "missing_cache", "error": "run add_hunt_openml_datasets.py first"}
            results.append(row)
            rejected.append(dataset)
            print(f"MISSING {dataset}", flush=True)
            continue
        n_rows = dataset_n(dataset)
        print(f"SCREEN {dataset} n={n_rows} ep={args.epochs} max_rank={args.max_rank}", flush=True)
        try:
            run_dir = run_protocol(
                dataset,
                epochs=args.epochs,
                seeds=(42,),
                config=DEFAULT_RUN_B,
                pilot=args.pilot,
            )
            rank, maes = graphcal_rank_seed42(run_dir, dataset)
        except Exception as exc:
            row = {"dataset": dataset, "status": "error", "error": repr(exc), "n": n_rows}
            results.append(row)
            rejected.append(dataset)
            print(f"ERROR {dataset} {exc!r}", flush=True)
            continue

        row = {
            "dataset": dataset,
            "status": "keep" if rank <= args.max_rank else "reject",
            "graphcal_rank": rank,
            "ungated_mae": maes,
            "n": n_rows,
            "epochs": args.epochs,
            "pilot": args.pilot,
            "run_dir": str(run_dir.relative_to(ROOT)),
        }
        results.append(row)
        if rank <= args.max_rank:
            kept.append(dataset)
            print(f"KEEP {dataset} rank={rank} graphcal_mae={maes['graphcal']:.4f}", flush=True)
        else:
            rejected.append(dataset)
            print(f"STOP {dataset} rank={rank} > {args.max_rank} (no further work on this set)", flush=True)

    summary = {
        "screened_at": datetime.now(timezone.utc).isoformat(),
        "epochs": args.epochs,
        "max_rank": args.max_rank,
        "pilot": args.pilot,
        "methods": list(METHODS),
        "config": DEFAULT_RUN_B,
        "kept": kept,
        "rejected": rejected,
        "rows": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"kept": kept, "rejected": rejected}, indent=2), flush=True)


if __name__ == "__main__":
    main()
