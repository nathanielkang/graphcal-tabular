# GraphCal — official implementation (tabular regression)

This repository provides a **self-contained reference implementation** of **GraphCal**, a graph-based, manifold-aware data augmentation procedure for **imbalanced tabular regression**. The code is intended to support **independent verification** of the method described in the accompanying manuscript (training-data augmentation only; any downstream regressor may be used after augmentation).

## What is included

- **`src/graphcal/`** — GraphCal augmentation (`GraphCalAugmenter`), OpenML-based dataset loaders used in the evaluation protocol, and evaluation metrics (MAE, tail MAE).
- **`scripts/smoke_test.py`** — Fast end-to-end sanity check on a standard benchmark split (recommended first run).
- **`scripts/run_experiments.py`** — Minimal driver: California housing, comparing **training without augmentation** vs **training with GraphCal augmentation** (same regressor family for both).
- **`scripts/run_all_datasets.py`** — Multi-dataset evaluation driver matching the paper’s dataset list (writes JSON summaries to the working directory when executed).
- **`scripts/monitor_cloud_results.ps1`** — Optional helper to poll remote machines over SSH (configure GCP project, zone, VM names, and paths for your environment).

Baseline methods, external reimplementations, full result tables from the paper, and **any manuscript figure or plotting utilities** are **not** distributed here; reviewers should run the provided drivers to reproduce GraphCal-augmented training on the bundled datasets.

## Requirements

Python 3.10+ and dependencies listed in `requirements.txt`.

## Quick verification

```bash
python -m pip install -r requirements.txt
python scripts/smoke_test.py
```

A successful run confirms that augmentation, acceptance filtering, and downstream training complete without error on the packaged smoke configuration.

## Reproducing larger experiments

After the smoke test, use `scripts/run_all_datasets.py` with the documented CLI flags (see `--help`). Runtime and peak memory depend on dataset size and hardware; use the same random seeds as in the paper when comparing to reported numbers.

## Citation

Please cite the manuscript using the bibliographic information provided in the camera-ready or preprint version once available (DOI / arXiv).

## License

Add a `LICENSE` file consistent with your institution’s policy before the final public release if one is not already present.
