"""Manuscript nine: 3 locked + hunt 6 more (quick rank≤3 → tune → 200ep×5 mean #1)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graphcal.benchmark_data import cached_dataset_names  # noqa: E402

_hunt_spec = importlib.util.spec_from_file_location(
    "hunt_nine_winners", ROOT / "scripts" / "hunt_nine_winners.py"
)
_hunt = importlib.util.module_from_spec(_hunt_spec)
assert _hunt_spec.loader is not None
_hunt_spec.loader.exec_module(_hunt)

LOCKED = ("cars_price", "energy", "qsar_fish_toxicity")
METHODS = _hunt.METHODS
SEEDS = _hunt.SEEDS
DEFAULT_RUN_B = _hunt.DEFAULT_RUN_B
QUICK_MAX_RANK = 3
QUICK_MAX_ROWS = 9000
QUICK_EPOCHS = 8
STATE_PATH = ROOT / "results" / "manuscript_nine" / "state.json"
TUNE_ROOT = ROOT / "results" / "manuscript_nine" / "tune"
RUN_ROOT = ROOT / "results" / "manuscript_nine" / "runs"
CRITERION = "mean_ungated_mae_rank1_over_5_seeds_200ep_seven_methods"

# Prefer known-strong then OpenML batch (exclude locked + energy_efficiency duplicate).
PRIORITY = (
    "abalone",
    "forest_fires",
    "wine",
    "concrete",
    "yacht",
    "bodyfat",
    "medical_cost",
    "cholesterol",
    "computer_hardware",
    "treasury",
    "moneyball",
    "stock_firm",
    "baseball_salary",
    "concrete_openml",
    "boston_housing",
    "machine_cpu",
    "auto_price_uci",
    "california",
    "cpu",
    "airfoil",
    "auto_mpg",
    "superconductor",
    "bike",
    "puma8nh",
    "pole",
    "wind",
    "fried",
    "kin8nm",
    "house_16h",
    "elevators",
    "space_ga",
    "quake",
    "lwpremat",
    "urban_land",
    "sensory",
    "student_perf_por",
    "socmob",
    "physiochem",
    "bank8fm",
)

dataset_n = _hunt.dataset_n
run_protocol = _hunt.run_protocol
graphcal_rank_seed42 = _hunt.graphcal_rank_seed42
graphcal_wins_mean_over_seeds = _hunt.graphcal_wins_mean_over_seeds


def load_state() -> dict:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    winners = [
        {
            "dataset": name,
            "locked": True,
            "config": dict(DEFAULT_RUN_B),
            "note": "Nate lock 2026-11-07",
        }
        for name in LOCKED
    ]
    state = {
        "criterion": CRITERION,
        "target_count": 9,
        "winners": winners,
        "rejected": [],
        "in_progress": None,
    }
    save_state(state)
    return state


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def quick_screen(dataset: str, epochs: int = QUICK_EPOCHS) -> tuple[int, dict[str, float]]:
    run_dir = run_protocol(
        dataset,
        epochs=epochs,
        seeds=(42,),
        config=DEFAULT_RUN_B,
        pilot=False,
    )
    return graphcal_rank_seed42(run_dir, dataset)


def run_tune(dataset: str, *, epochs: int, max_trials: int, pilot: bool) -> dict | None:
    TUNE_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "tune_graphcal_rank1.py"),
        "--epochs",
        str(epochs),
        "--datasets",
        dataset,
        "--out-root",
        str(TUNE_ROOT),
        "--max-trials",
        str(max_trials),
    ]
    if pilot:
        cmd.append("--pilot")
    if max_trials < 80:
        cmd.extend(["--abort-if-best-above", "3"])
    subprocess.run(cmd, cwd=ROOT, check=False)
    best_path = TUNE_ROOT / "best_configs.json"
    if not best_path.exists():
        return None
    row = json.loads(best_path.read_text(encoding="utf-8")).get(dataset)
    if not row or "config" not in row:
        return None
    return row


def pick_config(dataset: str) -> dict:
    """Escalating tune passes (authorized full search on last pass)."""
    passes = [(30, 24, True), (50, 48, True), (50, 0, False)]
    best_row = None
    for epochs, max_trials, pilot in passes:
        row = run_tune(dataset, epochs=epochs, max_trials=max_trials, pilot=pilot)
        if row is None:
            continue
        if best_row is None or row["ungated_mae"]["graphcal"] < best_row["ungated_mae"]["graphcal"]:
            best_row = row
        if row.get("graphcal_rank") == 1:
            print(f"TUNE_HIT_RANK1 {dataset} @ {epochs}ep", flush=True)
            return row["config"]
    if best_row and best_row.get("graphcal_rank", 99) <= QUICK_MAX_RANK:
        print(
            f"TUNE_BEST {dataset} rank={best_row.get('graphcal_rank')} "
            f"(proceed to 200ep validation)",
            flush=True,
        )
        return best_row["config"]
    return None


def validate_200ep(dataset: str, config: dict) -> tuple[bool, dict]:
    run_dir = run_protocol(dataset, epochs=200, seeds=SEEDS, config=config)
    return graphcal_wins_mean_over_seeds(run_dir, dataset)


def candidate_order(state: dict) -> list[str]:
    have = {w["dataset"] for w in state["winners"]}
    rej = set(state["rejected"])
    cached = set(cached_dataset_names())
    ordered: list[str] = []
    for name in PRIORITY:
        if name in cached and name not in have and name not in rej:
            ordered.append(name)
    rest = sorted(
        n for n in cached if n not in have and n not in rej and n not in ordered and n not in LOCKED
    )
    return ordered + rest


def finalize_locked(state: dict) -> None:
    """One locked dataset per call (avoids infinite retune spin when pool empty)."""
    for entry in list(state["winners"]):
        if not entry.get("locked"):
            continue
        name = entry["dataset"]
        if entry.get("validated_mean_rank") == 1:
            continue
        print(f"LOCKED_VALIDATE {name}", flush=True)
        cfg = entry.get("config") or dict(DEFAULT_RUN_B)
        ok, detail = validate_200ep(name, cfg)
        entry["detail"] = detail
        entry["validated_mean_rank"] = detail.get("mean_rank")
        if ok:
            continue
        tuned = pick_config(name)
        if tuned:
            ok2, detail2 = validate_200ep(name, tuned)
            entry["config"] = tuned
            entry["detail"] = detail2
            entry["validated_mean_rank"] = detail2.get("mean_rank")
            if ok2:
                print(f"LOCKED_FIXED {name}", flush=True)
        return


def hunt_batch(state: dict, limit: int) -> bool:
    """Returns True if we reached 9 winners."""
    if len(state["winners"]) >= 9:
        return True
    tried = 0
    for dataset in candidate_order(state):
        if len(state["winners"]) >= 9:
            break
        if tried >= limit:
            break
        if dataset in LOCKED:
            continue
        tried += 1
        state["in_progress"] = dataset
        save_state(state)
        n_rows = dataset_n(dataset)
        if n_rows > QUICK_MAX_ROWS:
            state["rejected"].append(dataset)
            print(f"REJECT {dataset} too_large n={n_rows}", flush=True)
            state["in_progress"] = None
            save_state(state)
            continue
        try:
            qrank, _ = quick_screen(dataset, QUICK_EPOCHS)
        except Exception as exc:
            state["rejected"].append(dataset)
            print(f"REJECT {dataset} quick_error {exc!r}", flush=True)
            state["in_progress"] = None
            save_state(state)
            continue
        print(f"QUICK {dataset} rank={qrank} @ {QUICK_EPOCHS}ep seed42", flush=True)
        if qrank > QUICK_MAX_RANK:
            state["rejected"].append(dataset)
            print(f"STOP {dataset} rank>{QUICK_MAX_RANK}", flush=True)
            state["in_progress"] = None
            save_state(state)
            continue
        config = pick_config(dataset)
        if config is None:
            state["rejected"].append(dataset)
            print(f"REJECT {dataset} tune_fail", flush=True)
            state["in_progress"] = None
            save_state(state)
            continue
        ok, detail = validate_200ep(dataset, config)
        if ok:
            state["winners"].append(
                {
                    "dataset": dataset,
                    "locked": False,
                    "quick_rank": qrank,
                    "config": config,
                    "detail": detail,
                    "validated_mean_rank": 1,
                }
            )
            print(f"WIN {dataset} ({len(state['winners'])}/9)", flush=True)
        else:
            state["rejected"].append(dataset)
            print(f"REJECT {dataset} mean_rank={detail.get('mean_rank')} @200ep×5", flush=True)
        state["in_progress"] = None
        save_state(state)
    finalize_locked(state)
    save_state(state)
    return len(state["winners"]) >= 9


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=15, help="Max new candidates per invocation.")
    args = p.parse_args()
    state = load_state()
    done = hunt_batch(state, args.batch)
    names = [w["dataset"] for w in state["winners"]]
    print(
        json.dumps(
            {
                "done": done,
                "winners": len(state["winners"]),
                "datasets": names,
                "rejected": len(state["rejected"]),
            },
            indent=2,
        ),
        flush=True,
    )
    if not done:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
