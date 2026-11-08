"""Merge tune best configs with Run-B defaults so roster has nine datasets."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROSTER_IN = ROOT.parent / "3_Others" / "TUNE_NINE_ROSTER.json"
ROSTER_OUT = ROOT.parent / "3_Others" / "TUNE_NINE_ROSTER_FULL.json"
RUN_B = {
    "distance_scale": "local",
    "residual_screen": True,
    "graphcal_delta": 2.5,
    "graphcal_gamma": 1.5,
    "graphcal_ridge_residual": True,
    "graphcal_ridge_alpha": 1.0,
    "graphcal_residual_sigma": 3.5,
}
MAIN_NINE = (
    "energy",
    "forest_fires",
    "wine",
    "abalone",
    "concrete",
    "yacht",
    "california",
    "cpu",
    "bike",
)


def main() -> None:
    base = json.loads(ROSTER_IN.read_text(encoding="utf-8"))
    by_name = {e["dataset"]: e for e in base["datasets"]}
    all_best = base.get("all_twelve_best") or {}
    entries = []
    for name in MAIN_NINE:
        if name in by_name:
            entries.append(by_name[name])
        elif name in all_best:
            entries.append(
                {
                    "dataset": name,
                    "graphcal_rank": all_best[name]["rank"],
                    "graphcal_mae": all_best[name]["mae"],
                    "config": dict(RUN_B),
                    "config_key": "run_b_default",
                    "beats_200ep_leader": False,
                }
            )
        else:
            entries.append(
                {
                    "dataset": name,
                    "graphcal_rank": 99,
                    "graphcal_mae": None,
                    "config": dict(RUN_B),
                    "config_key": "run_b_default",
                    "beats_200ep_leader": False,
                }
            )
    entries.sort(key=lambda e: (e["graphcal_rank"], e["graphcal_mae"] or 1e9))
    mean = sum(e["graphcal_rank"] for e in entries[:9]) / 9
    out = {
        "n_pick": 9,
        "mean_graphcal_rank": mean,
        "datasets": entries[:9],
        "note": "Filled missing tune configs with Run B defaults for five-seed measured run.",
    }
    ROSTER_OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"mean_rank": mean, "picked": [e["dataset"] for e in entries[:9]]}, indent=2))


if __name__ == "__main__":
    main()
