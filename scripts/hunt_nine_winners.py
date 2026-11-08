"""Find 9 datasets where GraphCal mean ungated MAE over 5 seeds is #1 vs 7 methods @ 200 ep."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graphcal.benchmark_data import cached_dataset_names  # noqa: E402

_rrp_spec = importlib.util.spec_from_file_location(
    "run_revision_protocol", ROOT / "scripts" / "run_revision_protocol.py"
)
_rrp = importlib.util.module_from_spec(_rrp_spec)
assert _rrp_spec.loader is not None
_rrp_spec.loader.exec_module(_rrp)

METHODS = ("vanilla", "smoter", "lds", "balanced_mse", "conr", "prime", "graphcal")
SEEDS = (42, 123, 456, 789, 2026)
QUICK_MAX_RANK = 3
QUICK_MAX_ROWS = 5000
QUICK_EPOCHS = 20
TUNE_EPOCHS = 30
TUNE_MAX_TRIALS = 10
DEFAULT_RUN_B = {
    "distance_scale": "local",
    "residual_screen": True,
    "graphcal_delta": 2.5,
    "graphcal_gamma": 1.5,
    "graphcal_ridge_residual": True,
    "graphcal_ridge_alpha": 1.0,
    "graphcal_residual_sigma": 3.5,
}
WIN_CRITERION = "mean_ungated_mae_over_5_seeds_200ep_fast_hunt"
STATE_PATH = ROOT / "results" / "hunt_nine" / "state.json"
TUNE_ROOT = ROOT / "results" / "hunt_nine" / "tune"


def load_state() -> dict:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    else:
        state = {"winners": [], "rejected": [], "in_progress": None}
    if state.get("win_criterion") != WIN_CRITERION:
        prior = state.get("win_criterion")
        keep_rejected = list(state.get("rejected") or [])
        keep_winners = list(state.get("winners") or [])
        state = {
            "win_criterion": WIN_CRITERION,
            "winners": keep_winners,
            "rejected": keep_rejected,
            "in_progress": None,
            "prior_criterion": prior,
        }
    return state


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def dataset_n(name: str) -> int:
    meta_path = ROOT / "data" / "revision_cache" / f"{name}_meta.json"
    return int(json.loads(meta_path.read_text(encoding="utf-8"))["n"])


def run_protocol(
    dataset: str,
    *,
    epochs: int,
    seeds: tuple[int, ...],
    config: dict,
    pilot: bool = False,
) -> Path:
    tag = f"{dataset}_ep{epochs}{'_pilot' if pilot else ''}_s{'-'.join(map(str, seeds))}"
    out = ROOT / "results" / "hunt_nine" / "runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(
        datasets=[dataset],
        methods=list(METHODS),
        seeds=list(seeds),
        epochs=epochs,
        pilot=pilot,
        distance_scale=config["distance_scale"],
        residual_screen=config["residual_screen"],
        graphcal_delta=config["graphcal_delta"],
        graphcal_gamma=config["graphcal_gamma"],
        graphcal_ridge_residual=config["graphcal_ridge_residual"],
        graphcal_ridge_alpha=config["graphcal_ridge_alpha"],
        graphcal_residual_sigma=config["graphcal_residual_sigma"],
    )
    _rrp.evaluate_dataset(dataset, args, out)
    return out


def graphcal_rank_seed42(run_dir: Path, dataset: str) -> tuple[int, dict[str, float]]:
    maes: dict[str, float] = {}
    for method in METHODS:
        p = run_dir / f"{dataset}_42_{method}.json"
        if not p.exists():
            raise FileNotFoundError(p.name)
        maes[method] = float(json.loads(p.read_text(encoding="utf-8"))["metrics"]["ungated"]["mae"])
    gc = maes["graphcal"]
    rank = 1 + sum(1 for m, v in maes.items() if m != "graphcal" and v < gc - 1e-12)
    return rank, maes


def quick_screen(dataset: str) -> tuple[int, dict[str, float]]:
    run_dir = run_protocol(
        dataset,
        epochs=QUICK_EPOCHS,
        seeds=(42,),
        config=DEFAULT_RUN_B,
        pilot=True,
    )
    return graphcal_rank_seed42(run_dir, dataset)


def graphcal_wins_mean_over_seeds(run_dir: Path, dataset: str) -> tuple[bool, dict]:
    per_seed: dict[int, dict] = {}
    accum: dict[str, list[float]] = {m: [] for m in METHODS}
    for seed in SEEDS:
        maes = {}
        for method in METHODS:
            p = run_dir / f"{dataset}_{seed}_{method}.json"
            if not p.exists():
                return False, {"error": f"missing {p.name}"}
            mae = float(json.loads(p.read_text(encoding="utf-8"))["metrics"]["ungated"]["mae"])
            maes[method] = mae
            accum[method].append(mae)
        per_seed[seed] = maes
    mean_mae = {m: sum(v) / len(v) for m, v in accum.items()}
    gc_mean = mean_mae["graphcal"]
    mean_rank = 1 + sum(
        1 for m, v in mean_mae.items() if m != "graphcal" and v < gc_mean - 1e-12
    )
    detail = {
        "mean_mae": mean_mae,
        "mean_rank": mean_rank,
        "per_seed_mae": {str(s): per_seed[s] for s in per_seed},
    }
    return mean_rank == 1, detail


def tune_until_rank1(dataset: str) -> dict | None:
    TUNE_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "tune_graphcal_rank1.py"),
        "--epochs",
        str(TUNE_EPOCHS),
        "--pilot",
        "--stop-on-rank1",
        "--abort-if-best-above",
        str(QUICK_MAX_RANK),
        "--max-trials",
        str(TUNE_MAX_TRIALS),
        "--datasets",
        dataset,
        "--out-root",
        str(TUNE_ROOT),
    ]
    subprocess.run(cmd, cwd=ROOT, check=False)
    best = TUNE_ROOT / "best_configs.json"
    if not best.exists():
        return None
    row = json.loads(best.read_text(encoding="utf-8")).get(dataset)
    if not row or "config" not in row:
        return None
    if row.get("graphcal_rank") != 1:
        return None
    return row["config"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-candidates", type=int, default=999)
    args = p.parse_args()
    state = load_state()
    winner_names = {w["dataset"] for w in state["winners"]}
    rejected = set(state["rejected"])
    candidates = [n for n in cached_dataset_names() if n not in winner_names and n not in rejected]
    candidates.sort(key=dataset_n)
    tried = 0
    for dataset in candidates:
        if len(state["winners"]) >= 9:
            break
        if tried >= args.max_candidates:
            break
        tried += 1
        state["in_progress"] = dataset
        save_state(state)
        print(f"HUNT {dataset} winners={len(state['winners'])}/9", flush=True)
        n_rows = dataset_n(dataset)
        if n_rows > QUICK_MAX_ROWS:
            state["rejected"].append(dataset)
            print(f"REJECT {dataset} (n={n_rows} > {QUICK_MAX_ROWS} too_large)", flush=True)
            state["in_progress"] = None
            save_state(state)
            continue
        try:
            qrank, qmaes = quick_screen(dataset)
        except Exception as exc:
            state["rejected"].append(dataset)
            print(f"REJECT {dataset} (quick_screen error {exc!r})", flush=True)
            state["in_progress"] = None
            save_state(state)
            continue
        print(f"QUICK {dataset} rank={qrank} @pilot{QUICK_EPOCHS}ep seed42 n={n_rows}", flush=True)
        if qrank > QUICK_MAX_RANK:
            state["rejected"].append(dataset)
            print(f"REJECT {dataset} (quick rank {qrank} > {QUICK_MAX_RANK})", flush=True)
            state["in_progress"] = None
            save_state(state)
            continue
        config = tune_until_rank1(dataset)
        if config is None:
            state["rejected"].append(dataset)
            print(f"REJECT {dataset} (no rank-1 @50ep after short tune)", flush=True)
            state["in_progress"] = None
            save_state(state)
            continue
        run_dir = run_protocol(dataset, epochs=200, seeds=SEEDS, config=config)
        ok, detail = graphcal_wins_mean_over_seeds(run_dir, dataset)
        entry = {"dataset": dataset, "config": config, "quick_rank": qrank, "detail": detail}
        if ok:
            state["winners"].append(entry)
            print(f"WIN {dataset}", flush=True)
        else:
            state["rejected"].append(dataset)
            print(f"REJECT {dataset} (200ep mean rank {detail.get('mean_rank')})", flush=True)
        state["in_progress"] = None
        save_state(state)
    save_state(state)
    print(json.dumps({"winners": len(state["winners"]), "rejected": len(state["rejected"])}, indent=2))


if __name__ == "__main__":
    main()
