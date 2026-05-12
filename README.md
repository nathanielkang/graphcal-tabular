# GraphCal — reference implementation

This repository provides a **reference implementation** of **GraphCal**, a graph-based, manifold-aware data augmentation procedure for **imbalanced tabular regression**. The code is intended to support **reproducibility checks** and **independent re-implementation verification** aligned with the accompanying manuscript.

## Scope of the release

The tree contains **source code and driver scripts** used to construct augmented training sets and to run the bundled diagnostics. **Numerical experiment logs, aggregated result tables, and manuscript figure assets are not versioned here**; reviewers and readers who wish to regenerate outputs should execute the scripts locally and retain the resulting files in their own working directories.

## Layout

- `src/graphcal/` — `GraphCalAugmenter` (core pipeline), dataset loaders, evaluation metrics
- `scripts/smoke_test.py` — small end-to-end sanity check (recommended first run)
- `scripts/run_experiments.py` — single-dataset demonstration driver (California housing)
- `scripts/run_all_datasets.py` — multi-dataset driver matching the paper’s evaluation protocol
- `scripts/generate_pr_figures.py` — optional matplotlib export for illustrative figures (writes to `figures/` or to `GRAPHCAL_FIGURE_DIR` if set)
- `scripts/monitor_cloud_results.ps1` — optional helper for polling remote machines (configure via environment variables documented in the script)

## Environment and quick verification

Python **3.10+** is recommended. Install dependencies and run the smoke test:

```bash
python -m pip install -r requirements.txt
python scripts/smoke_test.py
```

A successful smoke test confirms that augmentation, training, and evaluation execute without runtime errors on a standard benchmark split.

## Optional figure export

Figure export is **not required** for verifying the augmentation logic. When enabled, outputs are written under `./figures/` inside this repository unless overridden:

```bash
export GRAPHCAL_FIGURE_DIR=/absolute/path/to/output/figures   # Unix
python scripts/generate_pr_figures.py
```

On Windows (PowerShell), set `GRAPHCAL_FIGURE_DIR` before invoking the script in the same manner.

## Citation

If this implementation is useful in your work, please cite the associated article using the bibliographic record provided in the camera-ready / publisher version once available (DOI or stable arXiv identifier).

## License

Add a `LICENSE` file to match your intended distribution terms before tagging a formal release.
