"""Add extra OpenML tabular regression sets to revision_cache for winner hunt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import openml
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "revision_cache"

# (slug, openml_id, target_column or None for OpenML default)
SPECS = (
    ("puma8nh", 519, None),
    ("kin8nm", 827, None),
    ("wind", 503, None),
    ("fried", 901, None),
    ("house_16h", 574, None),
    ("pole", 672, None),
    ("elevators", 4549, None),
    ("space_ga", 201, None),
    ("quake", 198, None),
    # Tier-A batch 2026-11-06 (see 3_Others/HUNT_DATASET_CANDIDATES_2026-11-06.md)
    ("bodyfat", 42837, "L7"),
    ("computer_hardware", 46287, "9"),
    ("geyser", 712, "col_3"),
    ("branin", 41907, "y"),
    ("cholesterol", 204, "chol"),
    ("cars_price", 44994, "Price"),
    ("energy_efficiency", 43918, "Y1"),
    ("qsar_fish_toxicity", 44970, "LC50"),
    ("medical_cost", 46289, "charges"),
    # Batch B — toward nine manuscript sets (see MANUSCRIPT_NINE_DATASET_PLAN)
    ("treasury", 42367, "1MonthCDRate"),
    ("moneyball", 41021, "RS"),
    ("stock_firm", 223, "company10"),
    ("concrete_openml", 44959, "strength"),
    ("baseball_salary", 495, "1987_annual_salary_on_opening_day_in_thousands_of_dollars"),
    ("boston_housing", 531, "MEDV"),
    ("machine_cpu", 230, "class"),
    ("auto_price_uci", 207, "class"),
    # Batch C — pool exhausted on 2026-11-07; extra regression slugs
    ("bank8fm", 287, None),
    ("pollen", 467, None),
    ("lwpremat", 538, None),
    ("urban_land", 632, None),
    ("sensory", 688, None),
    ("student_perf_por", 43926, "G3"),
    ("socmob", 42192, "social_mobility_2016_nanoscore"),
    ("physiochem", 194, "RMS"),
)


def openml_frame_and_target(data_id: int, target_override: str | None) -> tuple[pd.DataFrame, str]:
    """Return a single DataFrame with target column; handle NAs in target for sklearn fetch."""
    try:
        data = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
        frame = data.frame
        target = target_override or (data.target.name if hasattr(data.target, "name") else frame.columns[-1])
        return frame, target
    except ValueError as exc:
        if "missing" not in str(exc).lower() and "Missing" not in str(exc):
            raise
    ds = openml.datasets.get_dataset(data_id, download_data=True)
    target = target_override or ds.default_target_attribute
    X, y, _, _ = ds.get_data(target=target, dataset_format="dataframe")
    frame = X.copy()
    frame[target] = y
    return frame, target


def save(name: str, frame: pd.DataFrame, target: str, source: dict, categorical=()) -> None:
    frame = frame.copy()
    for col in frame.columns:
        if isinstance(frame[col].dtype, pd.CategoricalDtype):
            frame[col] = frame[col].astype("string")
    y = pd.to_numeric(frame[target], errors="coerce")
    mask = np.isfinite(y.to_numpy(dtype=float))
    frame = frame.loc[mask].reset_index(drop=True)
    y = y.loc[mask]
    frame[target] = y.to_numpy(dtype=float)
    # Hunt default was 200; manuscript expansion allows n ≥ 150 after cleaning.
    if len(frame) < 150:
        raise ValueError(f"{name}: too few rows after cleaning ({len(frame)})")
    path = CACHE / f"{name}.pkl"
    frame.to_pickle(path)
    meta = {
        "dataset": name,
        "target": target,
        "source": source,
        "n": len(frame),
        "d": len(frame.columns) - 1,
        "categorical": list(categorical),
        "excluded_columns": [],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (CACHE / f"{name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(name, meta["n"], meta["d"], flush=True)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for slug, data_id, target_override in SPECS:
        meta_path = CACHE / f"{slug}_meta.json"
        if meta_path.exists():
            print("skip", slug, flush=True)
            continue
        ds = openml.datasets.get_dataset(data_id, download_data=False)
        details_name = ds.name or slug
        frame, target = openml_frame_and_target(data_id, target_override)
        if target not in frame.columns:
            target = frame.columns[-1]
        if slug == "energy_efficiency" and "Y2" in frame.columns and target == "Y1":
            frame = frame.drop(columns=["Y2"])
        cats = [c for c in frame.columns if c != target and str(frame[c].dtype) == "category"]
        try:
            save(
                slug,
                frame,
                target,
                {"provider": "OpenML", "data_id": data_id, "name": details_name},
                cats,
            )
        except Exception as exc:
            print("fail", slug, exc, flush=True)


if __name__ == "__main__":
    main()
