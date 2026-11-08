"""Print one-line tune monitor summary (run locally or on VM)."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    tune_root = ROOT / "results"
    logs = list(tune_root.glob("tune_rank1_2026-11-04/tune_log.jsonl"))
    logs.extend(tune_root.glob("tune_per_ds/*/tune_log*.jsonl"))
    best: dict[str, dict] = {}
    rank1: set[str] = set()
    for path in logs:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "graphcal_rank" not in row:
                continue
            d = row["dataset"]
            gc = row["ungated_mae"]["graphcal"]
            if d not in best or gc < best[d]["mae"]:
                best[d] = {"rank": row["graphcal_rank"], "mae": gc}
            if row["graphcal_rank"] == 1:
                rank1.add(d)
    procs = subprocess.run(
        ["pgrep", "-f", "tune_graphcal_rank1"],
        capture_output=True,
        text=True,
    )
    nproc = len([x for x in procs.stdout.split() if x.strip()])
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = (
        f"{ts} tune_procs={nproc} rank1={sorted(rank1)} "
        f"n_best={len(best)} "
        + " ".join(f"{k}:r{best[k]['rank']}" for k in sorted(best))
    )
    print(line, flush=True)
    out = ROOT.parent / "3_Others" / "TUNE_MONITOR_2026-11-05.md"
    block = f"| {ts} | `{line}` |\n"
    if out.exists() and "| UTC check |" in out.read_text(encoding="utf-8"):
        text = out.read_text(encoding="utf-8")
        if block.strip() not in text:
            out.write_text(text.rstrip() + "\n" + block, encoding="utf-8")
    sys.exit(0 if nproc == 0 else 1)


if __name__ == "__main__":
    main()
