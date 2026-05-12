# GraphCal — reference code

Minimal Python implementation of **GraphCal** (graph-based, manifold-aware augmentation for **imbalanced tabular regression**). This repository contains **library and driver scripts only**; it does **not** ship experiment logs, result tables, or manuscript assets.

## Contents

- `src/graphcal/` — augmentation core, dataset loaders, metrics, optional Calibrated Mixup baseline helper
- `scripts/smoke_test.py` — small synthetic check (run this first)
- `scripts/run_experiments.py` — California housing demo (writes JSON locally; keep out of git)
- `scripts/run_all_datasets.py` — multi-dataset driver (writes JSON locally; keep out of git)
- `scripts/run_mlp_targeted_vs_calmix.py` — optional comparison harness
- `scripts/generate_pr_figures.py` — optional matplotlib figure export (see script docstring)
- `scripts/monitor_cloud_results.ps1` — optional GCP VM poll helper (**configure** project/VM paths before use)

## Setup

```bash
python -m pip install -r requirements.txt
python scripts/smoke_test.py
```

Use Python 3.10+ recommended (see `requirements.txt`).

## Figure export (optional)

By default, `scripts/generate_pr_figures.py` writes under `./figures/` inside this repo. Override with:

```bash
set GRAPHCAL_FIGURE_DIR=C:\path\to\output\figures
python scripts/generate_pr_figures.py
```

## Citation

If you use this code in research, cite the associated paper when it is publicly available (DOI / arXiv link). This README intentionally omits venue and author names for blind / de-identified distribution.

## License

Unless otherwise noted in upstream dependencies, code in this tree is provided for reproducibility; add a `LICENSE` file when you publish the canonical open release.
