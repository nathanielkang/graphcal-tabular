"""Run 200-epoch × 5 seeds when every picked dataset has GraphCal rank 1 @ tune."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROSTER = ROOT.parent / "3_Others" / "TUNE_NINE_ROSTER.json"
SEEDS = (42, 123, 456, 789, 2026)
METHODS = ("vanilla", "smoter", "lds", "balanced_mse", "conr", "prime", "graphcal")


def load_roster(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing roster: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["datasets"]


def main() -> None:
    roster_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROSTER
    picks = load_roster(roster_path)
    bad = [p for p in picks if p["graphcal_rank"] != 1]
    if bad:
        print("NOT_ALL_RANK1", [b["dataset"] for b in bad], flush=True)
        sys.exit(2)
    print(f"ALL_RANK1 n={len(picks)} → five-seed flagship", flush=True)
    for entry in picks:
        name = entry["dataset"]
        c = entry["config"]
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
            c["distance_scale"],
            "--graphcal-delta",
            str(c["graphcal_delta"]),
            "--graphcal-gamma",
            str(c["graphcal_gamma"]),
            "--graphcal-ridge-alpha",
            str(c["graphcal_ridge_alpha"]),
            "--graphcal-residual-sigma",
            str(c["graphcal_residual_sigma"]),
        ]
        if c.get("graphcal_ridge_residual"):
            cmd.append("--graphcal-ridge-residual")
        if not c.get("residual_screen", True):
            cmd.append("--no-residual-screen")
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    print("FIVE_SEED_DONE", flush=True)


if __name__ == "__main__":
    main()
