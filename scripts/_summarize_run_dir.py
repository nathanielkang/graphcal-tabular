"""Summarize ungated test MAE from a flagship run directory."""
import json
import glob
import os
import re
import sys

log_path = sys.argv[1]
root_hint = sys.argv[2] if len(sys.argv) > 2 else ""
log = open(log_path, encoding="utf-8").read()
match = re.search(r"flagship/([a-f0-9]+)", log)
if match:
    root = os.path.join(
        "experiments/17_InformationSciences_GraphCal/2_Code/results/revision_2026-10-24/flagship",
        match.group(1),
    )
elif root_hint:
    root = root_hint
else:
    dirs = sorted(
        glob.glob(
            "experiments/17_InformationSciences_GraphCal/2_Code/results/revision_2026-10-24/flagship/*/protocol.json"
        ),
        key=os.path.getmtime,
    )
    root = os.path.dirname(dirs[-1])
print("ROOT", root)
prot = json.load(open(os.path.join(root, "protocol.json"), encoding="utf-8"))
print("epochs", prot["arguments"].get("epochs"), "methods", prot["arguments"].get("methods"))
rows = []
for path in sorted(glob.glob(os.path.join(root, "*_42_*.json"))):
    if "frozen" in path or path.endswith(("protocol.json", "status.json")):
        continue
    obj = json.load(open(path, encoding="utf-8"))
    rows.append((obj["dataset"], obj["method"], obj["metrics"]["ungated"]["mae"]))
by = {}
for dataset, method, mae in rows:
    by.setdefault(dataset, {})[method] = mae
methods = prot["arguments"]["methods"]
for dataset in sorted(by):
    vals = [(m, by[dataset][m]) for m in methods if m in by[dataset]]
    best = min(vals, key=lambda item: item[1])
    print(f"\n{dataset}:")
    for method, mae in vals:
        tag = " *best*" if method == best[0] else ""
        print(f"  {method:14} {mae:.4f}{tag}")
    graphcal = by[dataset].get("graphcal")
    if graphcal is not None:
        rank = 1 + sorted(v for _, v in vals).index(graphcal)
        print(f"  GraphCal rank {rank}/{len(vals)}")
