"""Export graphcal/vanilla rows from latest five-seed flagship runs into 3_Others ledger JSON."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLAGSHIP = ROOT / "results" / "revision_2026-10-24" / "flagship"
OUT = ROOT.parent / "3_Others" / "LEDGER_FIVE_SEED_TUNED_EXPLORATORY.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    roster_path = ROOT.parent / "3_Others" / "TUNE_NINE_ROSTER.json"
    datasets = []
    if roster_path.exists():
        datasets = [d["dataset"] for d in json.loads(roster_path.read_text())["datasets"]]
    rows = []
    dirs = sorted(FLAGSHIP.glob("*/protocol.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    seen: set[tuple[str, int, str]] = set()
    for prot_path in dirs:
        prot = json.loads(prot_path.read_text(encoding="utf-8"))
        args = prot["arguments"]
        if args.get("epochs") != 200:
            continue
        if len(args.get("seeds") or []) < 5:
            continue
        run_dir = prot_path.parent
        name = run_dir.name
        for json_path in sorted(run_dir.glob("*_*.json")):
            if "frozen" in json_path.name or json_path.name in ("protocol.json", "status.json"):
                continue
            obj = json.loads(json_path.read_text(encoding="utf-8"))
            if obj.get("method") not in (
                "graphcal",
                "vanilla",
                "smoter",
                "lds",
                "balanced_mse",
                "conr",
                "prime",
            ):
                continue
            if datasets and obj["dataset"] not in datasets:
                continue
            key = (obj["dataset"], obj["seed"], obj["method"])
            if key in seen:
                continue
            seen.add(key)
            u = obj["metrics"]["ungated"]
            rows.append(
                {
                    "run_id": name,
                    "dataset": obj["dataset"],
                    "seed": obj["seed"],
                    "method": obj["method"],
                    "epochs": obj["epochs"],
                    "ungated_mae": u["mae"],
                    "joint_sparse_mae": u.get("joint_sparse_mae"),
                    "graphcal_config": args if obj["method"] == "graphcal" else None,
                }
            )
    payload = {
        "status": "EXPLORATORY_MEASURED_FIVE_SEEDS",
        "note": "Pull after overnight five-seed tuned runs; not for PDF until Nate locks method vs TeX.",
        "n_rows": len(rows),
        "rows": sorted(rows, key=lambda r: (r["dataset"], r["seed"], r["method"])),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
