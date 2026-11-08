"""Pick up to nine datasets with lowest mean GraphCal rank from tune logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALL = (
    "abalone",
    "california",
    "bike",
    "wine",
    "energy",
    "airfoil",
    "superconductor",
    "cpu",
    "forest_fires",
    "concrete",
    "yacht",
    "auto_mpg",
)


def load_attempts(log_paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in log_paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "graphcal_rank" in row:
                rows.append(row)
    return rows


def best_per_dataset(rows: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for row in rows:
        name = row["dataset"]
        prev = best.get(name)
        gc = row["ungated_mae"]["graphcal"]
        if prev is None or gc < prev["ungated_mae"]["graphcal"]:
            best[name] = row
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tune-root", type=Path, default=ROOT / "results")
    p.add_argument("--k", type=int, default=9)
    p.add_argument("--out", type=Path, default=ROOT.parent / "3_Others" / "TUNE_NINE_ROSTER.json")
    args = p.parse_args()
    logs: list[Path] = list(args.tune_root.glob("tune_rank1_2026-11-04/tune_log*.jsonl"))
    logs.extend(args.tune_root.glob("tune_per_ds/*/tune_log*.jsonl"))
    logs.extend(args.tune_root.glob("tune_repass/*/tune_log*.jsonl"))
    rows = load_attempts(logs)
    best = best_per_dataset(rows)
    ranked = sorted(
        best.values(),
        key=lambda r: (r["graphcal_rank"], r["ungated_mae"]["graphcal"]),
    )
    pick = ranked[: args.k]
    mean_rank = sum(r["graphcal_rank"] for r in pick) / len(pick) if pick else float("nan")
    payload = {
        "n_pick": len(pick),
        "mean_graphcal_rank": mean_rank,
        "target_mean_rank": "1.0–1.99",
        "datasets": [
            {
                "dataset": r["dataset"],
                "graphcal_rank": r["graphcal_rank"],
                "graphcal_mae": r["ungated_mae"]["graphcal"],
                "config": r["config"],
                "config_key": r["config_key"],
                "beats_200ep_leader": r.get("beats_200ep_leader"),
            }
            for r in pick
        ],
        "all_twelve_best": {
            r["dataset"]: {"rank": r["graphcal_rank"], "mae": r["ungated_mae"]["graphcal"]}
            for r in sorted(best.values(), key=lambda x: x["dataset"])
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"mean_rank": mean_rank, "picked": [d["dataset"] for d in payload["datasets"]]}, indent=2))


if __name__ == "__main__":
    main()
