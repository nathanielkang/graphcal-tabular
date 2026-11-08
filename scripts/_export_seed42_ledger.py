"""Export the local seed-42 tarball into a named ledger. Not a paper script."""
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARBALL = ROOT / "results" / "revision_2026-10-24" / "seed42_json.tgz"
OUT = ROOT.parents[0] / "3_Others" / "LEDGER_SEED42_MEASURED_2026-11-04.json"

rows = []
with tarfile.open(TARBALL, "r:gz") as archive:
    for name in sorted(archive.getnames()):
        if not name.endswith(".json"):
            continue
        if name.endswith(("protocol.json", "status.json")):
            continue
        obj = json.loads(archive.extractfile(name).read())
        if "metrics" not in obj:
            continue
        augmentation = (obj.get("diagnostics") or {}).get("augmentation") or {}
        activation = obj.get("activation") or {}
        ungated = obj["metrics"]["ungated"]
        rows.append(
            {
                "dataset": obj["dataset"],
                "seed": obj["seed"],
                "method": obj["method"],
                "pilot": obj["pilot"],
                "epochs": obj["epochs"],
                "ungated_mae": ungated["mae"],
                "empirical_mae": obj["metrics"]["empirical_gate"]["mae"],
                "certified_mae": obj["metrics"]["certified_gate"]["mae"],
                "ungated_upper": ungated["upper_tail_mae"],
                "ungated_lower": ungated["lower_tail_mae"],
                "ungated_sparse": ungated["joint_sparse_mae"],
                "emp_diff": activation.get("empirical_difference"),
                "margin": activation.get("hoeffding_margin"),
                "certified_activated": activation.get("activated"),
                "n_activation": activation.get("n_activation"),
                "family_size": activation.get("family_size"),
                "num_generated": augmentation.get("num_generated"),
                "num_accepted": augmentation.get("num_accepted"),
                "k": augmentation.get("k"),
            }
        )

payload = {
    "source": "2_Code/results/revision_2026-10-24/seed42_json.tgz",
    "run_note": "Training seed 42 only. Printed method is the stored global screen. Not a five-seed mean.",
    "rows": rows,
}
OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"wrote {len(rows)} rows to {OUT}")
for row in rows:
    if row["method"] != "graphcal":
        continue
    print(
        f"{row['dataset']:16} ung {row['ungated_mae']:.4f} "
        f"emp {row['empirical_mae']:.4f} cert {row['certified_mae']:.4f} "
        f"act {row['certified_activated']} acc {row['num_accepted']}/{row['num_generated']} "
        f"diff {row['emp_diff']:.4f} margin {row['margin']:.3f}"
    )
