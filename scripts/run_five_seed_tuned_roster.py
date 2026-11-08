"""Five-seed × 200-epoch flagship for each dataset in TUNE_NINE_ROSTER (per-dataset GraphCal config)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROSTER = ROOT.parent / "3_Others" / "TUNE_NINE_ROSTER.json"
SEEDS = (42, 123, 456, 789, 2026)
METHODS = ("vanilla", "smoter", "lds", "balanced_mse", "conr", "prime", "graphcal")


def run_dataset(name: str, config: dict) -> None:
    cmd = [
        "python3",
        str(ROOT / "scripts" / "run_revision_protocol.py"),
        "--seeds",
        *[str(s) for s in SEEDS],
        "--datasets",
        name,
        "--methods",
        *METHODS,
        "--epochs",
        "200",
        "--distance-scale",
        config["distance_scale"],
        "--graphcal-delta",
        str(config["graphcal_delta"]),
        "--graphcal-gamma",
        str(config["graphcal_gamma"]),
        "--graphcal-ridge-alpha",
        str(config["graphcal_ridge_alpha"]),
        "--graphcal-residual-sigma",
        str(config["graphcal_residual_sigma"]),
    ]
    if config.get("graphcal_ridge_residual"):
        cmd.append("--graphcal-ridge-residual")
    if not config.get("residual_screen", True):
        cmd.append("--no-residual-screen")
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    roster_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROSTER
    require_rank1 = "--require-all-rank1" in sys.argv
    data = json.loads(roster_path.read_text(encoding="utf-8"))
    picks = data["datasets"]
    if require_rank1:
        bad = [p["dataset"] for p in picks if p["graphcal_rank"] != 1]
        if bad:
            print("ABORT not all rank1:", bad, flush=True)
            sys.exit(2)
    log = ROOT / "five_seed_tuned_roster.log"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"START n={len(picks)} require_rank1={require_rank1}\n")
        for entry in picks:
            fh.write(f"dataset {entry['dataset']} rank {entry['graphcal_rank']}\n")
            fh.flush()
            run_dataset(entry["dataset"], entry["config"])
        fh.write("DONE\n")
    print("FIVE_SEED_ROSTER_DONE", flush=True)


if __name__ == "__main__":
    main()
