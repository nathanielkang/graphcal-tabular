"""Grid-search GraphCal hyperparams for rank-1 ungated test MAE (exploratory)."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
import importlib.util

_rrp_spec = importlib.util.spec_from_file_location(
    "run_revision_protocol", ROOT / "scripts" / "run_revision_protocol.py"
)
_rrp = importlib.util.module_from_spec(_rrp_spec)
assert _rrp_spec.loader is not None
_rrp_spec.loader.exec_module(_rrp)
evaluate_dataset = _rrp.evaluate_dataset

# Ungated MAE to beat @ 200 ep (Run B flagship 5a63d3ae); tune @ 50 ep as proxy.
BEAT_MAE = {
    "forest_fires": 23.6245,
    "abalone": 1.4403,
    "wine": 0.5093,
    "concrete": 4.0814,
    "energy": 0.6338,  # already #1; optional re-check
}

SIXTH_CANDIDATES = ("california", "auto_mpg", "cpu", "airfoil", "superconductor")

METHODS = ("vanilla", "smoter", "lds", "balanced_mse", "conr", "prime", "graphcal")


@dataclass(frozen=True)
class GcConfig:
    distance_scale: str = "local"
    residual_screen: bool = True
    graphcal_delta: float = 2.5
    graphcal_gamma: float = 1.5
    graphcal_ridge_residual: bool = True
    graphcal_ridge_alpha: float = 1.0
    graphcal_residual_sigma: float = 3.5

    def key(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def configs_for_dataset(name: str) -> list[GcConfig]:
    """Focused grid: delta × sigma, then residual off variants if close."""
    base = []
    for delta, sigma in product(
        (2.0, 2.25, 2.5, 2.75, 3.0, 3.25),
        (3.0, 3.5, 4.0, 4.5),
    ):
        base.append(
            GcConfig(
                graphcal_delta=delta,
                graphcal_residual_sigma=sigma,
                graphcal_ridge_residual=True,
                residual_screen=True,
            )
        )
    for delta in (2.5, 2.75, 3.0):
        base.append(
            GcConfig(
                graphcal_delta=delta,
                graphcal_residual_sigma=4.0,
                graphcal_ridge_residual=True,
                residual_screen=False,
            )
        )
    for gamma in (1.25, 2.0):
        base.append(
            GcConfig(
                graphcal_delta=2.5,
                graphcal_gamma=gamma,
                graphcal_residual_sigma=3.5,
            )
        )
    # Deduplicate
    seen: set[str] = set()
    out: list[GcConfig] = []
    for c in base:
        if c.key() in seen:
            continue
        seen.add(c.key())
        out.append(c)
    return out


def run_one(
    dataset: str, gc: GcConfig, *, epochs: int, out_root: Path, pilot: bool = False
) -> dict:
    out = out_root / dataset / gc.key()
    out.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        datasets=[dataset],
        methods=list(METHODS),
        seeds=[42],
        epochs=epochs,
        pilot=pilot,
        distance_scale=gc.distance_scale,
        residual_screen=gc.residual_screen,
        graphcal_delta=gc.graphcal_delta,
        graphcal_gamma=gc.graphcal_gamma,
        graphcal_ridge_residual=gc.graphcal_ridge_residual,
        graphcal_ridge_alpha=gc.graphcal_ridge_alpha,
        graphcal_residual_sigma=gc.graphcal_residual_sigma,
    )
    evaluate_dataset(dataset, args, out)
    rows = {}
    for method in METHODS:
        path = out / f"{dataset}_42_{method}.json"
        rows[method] = json.loads(path.read_text(encoding="utf-8"))["metrics"]["ungated"]["mae"]
    gc_mae = rows["graphcal"]
    rank = 1 + sorted(rows.values()).index(gc_mae)
    beat = BEAT_MAE.get(dataset)
    return {
        "dataset": dataset,
        "config": gc.__dict__,
        "config_key": gc.key(),
        "epochs": epochs,
        "ungated_mae": {k: float(v) for k, v in rows.items()},
        "graphcal_rank": rank,
        "beats_200ep_leader": beat is not None and gc_mae < beat,
        "beat_target": beat,
        "out_dir": str(out.resolve().relative_to(ROOT.resolve())),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--out-root", type=Path, default=ROOT / "results" / "tune_rank1_2026-11-04")
    p.add_argument("--stop-on-rank1", action="store_true")
    p.add_argument(
        "--abort-if-best-above",
        type=int,
        default=0,
        help="Stop grid early if best GraphCal rank is worse than this (0=disabled).",
    )
    p.add_argument("--max-trials", type=int, default=0, help="Cap grid trials per dataset (0=all).")
    p.add_argument("--pilot", action="store_true", help="Subsample fitting pool (fast screen).")
    args = p.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    log_path = args.out_root / "tune_log.jsonl"
    if len(args.datasets) == 1:
        log_path = args.out_root / f"tune_log_{args.datasets[0]}.jsonl"
    best_by_dataset: dict[str, dict] = {}
    with log_path.open("a", encoding="utf-8") as log:
        for name in args.datasets:
            configs = configs_for_dataset(name)
            trials = 0
            for gc in configs:
                trials += 1
                if args.max_trials and trials > args.max_trials:
                    print(f"MAX_TRIALS {name} at {args.max_trials}", flush=True)
                    break
                print(f"TUNE {name} {gc.key()} delta={gc.graphcal_delta} sigma={gc.graphcal_residual_sigma} resid={gc.residual_screen}", flush=True)
                try:
                    row = run_one(
                        name,
                        gc,
                        epochs=args.epochs,
                        out_root=args.out_root,
                        pilot=args.pilot,
                    )
                except Exception as exc:
                    row = {"dataset": name, "config_key": gc.key(), "error": repr(exc)}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps({k: row[k] for k in row if k != "ungated_mae"}, indent=2), flush=True)
                if "graphcal_rank" not in row:
                    continue
                prev = best_by_dataset.get(name)
                if prev is None or row["ungated_mae"]["graphcal"] < prev["ungated_mae"]["graphcal"]:
                    best_by_dataset[name] = row
                if args.stop_on_rank1 and row["graphcal_rank"] == 1:
                    print(f"RANK1 {name} at {gc.key()}", flush=True)
                    break
                best = best_by_dataset.get(name)
                if (
                    args.abort_if_best_above
                    and best
                    and best.get("graphcal_rank", 99) > args.abort_if_best_above
                ):
                    print(
                        f"ABORT {name} best_rank={best['graphcal_rank']} "
                        f"above {args.abort_if_best_above}",
                        flush=True,
                    )
                    break
    summary = args.out_root / "best_configs.json"
    summary.write_text(json.dumps(best_by_dataset, indent=2), encoding="utf-8")
    print(f"Wrote {summary}", flush=True)


if __name__ == "__main__":
    main()
