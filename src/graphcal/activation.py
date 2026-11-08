"""One-use activation of frozen models on independent observations.

The certificate controls capped absolute loss, not uncapped MAE. Neither model
may be trained, tuned, or refitted using the activation observations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ActivationDecision:
    empirical_difference: float
    upper_bound: float
    hoeffding_margin: float
    activated: bool
    n_activation: int
    loss_cap: float
    family_size: int
    failure_probability: float

    def to_dict(self) -> dict:
        return asdict(self)


def activation_decision(y, baseline_prediction, candidate_prediction, *,
                        loss_cap: float, family_size: int,
                        failure_probability: float = 0.05) -> ActivationDecision:
    """Decide one frozen contrast.

    ``family_size`` is K for this dataset and seed. The default failure
    probability 0.05 is simultaneous over those K contrasts only.
    """
    arrays = [np.asarray(a, dtype=float).reshape(-1)
              for a in (y, baseline_prediction, candidate_prediction)]
    if not arrays[0].size or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Activation requires nonempty arrays of equal length")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Activation observations and predictions must be finite")
    if not math.isfinite(loss_cap) or loss_cap <= 0:
        raise ValueError("The loss cap must be positive and fixed before activation")
    if not isinstance(family_size, int) or family_size < 1:
        raise ValueError("family_size must count all frozen candidate contrasts")
    if not 0 < failure_probability < 1:
        raise ValueError("failure_probability must be strictly between zero and one")
    truth, p0, p1 = arrays
    differences = (np.minimum(np.abs(truth - p1), loss_cap)
                   - np.minimum(np.abs(truth - p0), loss_cap))
    mean = float(differences.mean())
    margin = loss_cap * math.sqrt(
        2.0 * math.log(family_size / failure_probability) / len(truth))
    upper = mean + margin
    return ActivationDecision(mean, upper, margin, upper <= 0,
                              len(truth), loss_cap, family_size,
                              failure_probability)
