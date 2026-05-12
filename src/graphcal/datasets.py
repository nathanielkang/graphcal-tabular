from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: Callable[[], tuple[np.ndarray, np.ndarray]]


def _to_numeric_frame(df: pd.DataFrame) -> np.ndarray:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype).startswith("category"):
            out[col] = pd.Categorical(out[col]).codes
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.to_numpy(dtype=np.float64)


def _drop_nan_rows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask]


def _load_openml(data_id: int, target_candidates: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ds = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    X_df = ds.data.copy()
    lower_map = {str(c).lower(): c for c in X_df.columns}
    target_col = None
    for t in target_candidates:
        if t.lower() in lower_map:
            target_col = lower_map[t.lower()]
            break

    if target_col is not None:
        y = pd.to_numeric(X_df[target_col], errors="coerce").to_numpy(dtype=np.float64)
        X_df = X_df.drop(columns=[target_col])
    else:
        y = pd.to_numeric(pd.Series(np.asarray(ds.target)), errors="coerce").to_numpy(dtype=np.float64)
    X = _to_numeric_frame(X_df)
    return _drop_nan_rows(X, y)


def _load_openml_name(name: str, version: int, target_candidates: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ds = fetch_openml(name=name, version=version, as_frame=True, parser="auto")
    X_df = ds.data.copy()
    lower_map = {str(c).lower(): c for c in X_df.columns}
    target_col = None
    for t in target_candidates:
        if t.lower() in lower_map:
            target_col = lower_map[t.lower()]
            break
    if target_col is not None:
        y = pd.to_numeric(X_df[target_col], errors="coerce").to_numpy(dtype=np.float64)
        X_df = X_df.drop(columns=[target_col])
    else:
        y = pd.to_numeric(pd.Series(np.asarray(ds.target)), errors="coerce").to_numpy(dtype=np.float64)
    X = _to_numeric_frame(X_df)
    return _drop_nan_rows(X, y)


def _load_abalone() -> tuple[np.ndarray, np.ndarray]:
    return _load_openml(183, ["Rings", "rings"])


def _load_california() -> tuple[np.ndarray, np.ndarray]:
    ds = fetch_california_housing(as_frame=False)
    X = np.asarray(ds.data, dtype=np.float64)
    y = np.asarray(ds.target, dtype=np.float64).ravel()
    return _drop_nan_rows(X, y)


def _load_bike() -> tuple[np.ndarray, np.ndarray]:
    return _load_openml(44063, ["cnt", "Count"])


def _load_cpu() -> tuple[np.ndarray, np.ndarray]:
    try:
        return _load_openml(573, ["usr", "user"])
    except Exception:
        return _load_openml(197, ["usr", "user"])


def _load_wine() -> tuple[np.ndarray, np.ndarray]:
    return _load_openml(287, ["quality", "Quality"])


def _load_energy() -> tuple[np.ndarray, np.ndarray]:
    try:
        return _load_openml(1472, ["Y1", "y1"])
    except Exception:
        return _load_openml(213, ["Y1", "y1"])


def _load_airfoil() -> tuple[np.ndarray, np.ndarray]:
    try:
        return _load_openml_name("airfoil_self_noise", 1, [])
    except Exception:
        return _load_openml(43919, [])


def _load_superconductor() -> tuple[np.ndarray, np.ndarray]:
    try:
        return _load_openml(44964, ["critical_temp", "critical temperature"])
    except Exception:
        return _load_openml(43443, ["critical_temp", "critical temperature"])


def _load_forest_fires() -> tuple[np.ndarray, np.ndarray]:
    raw = urlopen(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv",
        timeout=120,
    ).read()
    df = pd.read_csv(io.BytesIO(raw))
    if "area" not in df.columns:
        raise ValueError("Forest Fires target column 'area' not found")
    y = pd.to_numeric(df["area"], errors="coerce").to_numpy(dtype=np.float64)
    X_df = df.drop(columns=["area"])
    X = _to_numeric_frame(X_df)
    return _drop_nan_rows(X, y)


DATASET_SPECS: list[DatasetSpec] = [
    DatasetSpec("abalone", _load_abalone),
    DatasetSpec("california", _load_california),
    DatasetSpec("bike", _load_bike),
    DatasetSpec("cpu", _load_cpu),
    DatasetSpec("wine", _load_wine),
    DatasetSpec("energy", _load_energy),
    DatasetSpec("airfoil", _load_airfoil),
    DatasetSpec("superconductor", _load_superconductor),
    DatasetSpec("forest_fires", _load_forest_fires),
]


def list_dataset_names() -> list[str]:
    return [d.name for d in DATASET_SPECS]


def load_dataset(name: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec = next((d for d in DATASET_SPECS if d.name == name), None)
    if spec is None:
        raise ValueError(f"Unknown dataset: {name}")
    X, y = spec.loader()
    if X.shape[0] < 10:
        raise ValueError(f"Dataset {name} has too few samples ({X.shape[0]})")
    return train_test_split(X, y, test_size=0.3, random_state=seed)
