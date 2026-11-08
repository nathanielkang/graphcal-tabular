import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "results/manuscript_nine/state.json"
s = json.loads(p.read_text(encoding="utf-8"))
undo = {"wine", "bank8fm", "sensory", "socmob"}
s["rejected"] = [x for x in s["rejected"] if x not in undo]
p.write_text(json.dumps(s, indent=2), encoding="utf-8")
print("unrejected", sorted(undo), "rejected", len(s["rejected"]))
