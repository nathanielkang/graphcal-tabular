"""Count which screen rejects Bike candidates under the local distance scale."""
import numpy as np
from sklearn.model_selection import train_test_split

from src.graphcal.augmenter import GraphCalAugmenter, k_nearest_indices
from src.graphcal.benchmark_data import load_revision_dataset
from src.graphcal.preprocessing import TrainingEncoder


def fitting_index(n):
    remaining, _test = train_test_split(np.arange(n), test_size=0.3, random_state=20261024)
    fitting_tuning, _activation = train_test_split(remaining, test_size=1 / 7, random_state=20261025)
    fitting, _tuning = train_test_split(fitting_tuning, test_size=1 / 6, random_state=20261026)
    return fitting


def main():
    frame, y, meta = load_revision_dataset("bike")
    fit = fitting_index(len(y))
    encoder = TrainingEncoder(meta["categorical"])
    x_fit = encoder.fit_transform(frame.iloc[fit], y[fit])
    y_fit = y[fit]
    aug = GraphCalAugmenter(distance_scale="local", random_state=42)
    counts = {"distance": 0, "residual": 0, "pass": 0}

    def screen(cand):
        cand = np.asarray(cand, dtype=np.float64).reshape(-1)
        distances = np.linalg.norm(aug.reference_z_ - cand, axis=1)
        idx = k_nearest_indices(distances, aug.k_)
        scale = float(np.median(np.asarray(aug.train_mean_dist_)[idx]))
        if float(distances[idx].mean()) > aug.delta * max(scale, 1e-12):
            counts["distance"] += 1
            return False
        if not aug.residual_accepts(cand, idx):
            counts["residual"] += 1
            return False
        counts["pass"] += 1
        return True

    aug.passes_screens = screen
    result = aug.fit_transform(x_fit, y_fit)
    print("COUNTS", counts, flush=True)
    print("DIAG", result.diagnostics, flush=True)


if __name__ == "__main__":
    main()
