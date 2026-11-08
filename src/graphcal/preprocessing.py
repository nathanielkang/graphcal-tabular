"""Training-only numeric imputation and smoothed leave-one-out encoding."""
from __future__ import annotations
import numpy as np
import pandas as pd


class TrainingEncoder:
    def __init__(self, categorical=(), smoothing: float = 10.0):
        self.categorical = tuple(categorical)
        self.smoothing = smoothing

    def fit_transform(self, frame: pd.DataFrame, y) -> np.ndarray:
        frame = frame.copy()
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(frame) != len(y) or len(y) < 2 or not np.isfinite(y).all():
            raise ValueError("Encoding needs at least two finite training labels")
        self.columns_ = list(frame.columns)
        self.mean_ = float(y.mean())
        self.mapping_, self.medians_ = {}, {}
        output = []
        for col in self.columns_:
            if col in self.categorical or not pd.api.types.is_numeric_dtype(frame[col]):
                values = frame[col].astype('string').fillna('<missing>')
                stats = pd.DataFrame({'category': values.to_numpy(), 'y': y}).groupby('category')['y'].agg(['sum','count'])
                self.mapping_[col] = ((stats['sum'] + self.smoothing*self.mean_)
                                     / (stats['count'] + self.smoothing)).to_dict()
                sums = values.map(stats['sum']).to_numpy(dtype=float)
                counts = values.map(stats['count']).to_numpy(dtype=float)
                # Both the category sum AND global smoothing prior exclude y_i.
                prior = (y.sum() - y) / (len(y) - 1)
                encoded = (sums - y + self.smoothing * prior) / (counts - 1 + self.smoothing)
                output.append(encoded)
            else:
                values = pd.to_numeric(frame[col], errors='coerce').to_numpy(dtype=float)
                valid = np.isfinite(values)
                median = float(np.median(values[valid])) if valid.any() else 0.0
                self.medians_[col] = median
                output.append(np.where(valid, values, median))
        return np.column_stack(output)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        output = []
        for col in self.columns_:
            if col in self.mapping_:
                output.append(frame[col].astype('string').fillna('<missing>').map(
                    self.mapping_[col]).fillna(self.mean_).to_numpy(dtype=float))
            else:
                a = pd.to_numeric(frame[col], errors='coerce').to_numpy(dtype=float)
                output.append(np.where(np.isfinite(a), a, self.medians_[col]))
        return np.column_stack(output)
