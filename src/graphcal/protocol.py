"""Paper tuning slots and the activation family size.

Two paired settings, not a Cartesian product of learning rate and kappa.
Slot 0 pairs learning rate 1e-3 with GraphCal kappa 2.
Slot 1 pairs learning rate 3e-3 with GraphCal kappa 1.
Other methods receive those same two learning rates and no extra slots.
"""
from __future__ import annotations

PAIRED_SETTINGS = (
    {"learning_rate": 1e-3, "kappa": 2.0},
    {"learning_rate": 3e-3, "kappa": 1.0},
)


def comparison_family_size(methods) -> int:
    """K for one dataset and one seed.

    The default 95 percent statement is simultaneous over these K frozen
    contrasts only. It is not a union bound across datasets or seeds.
    """
    names = tuple(methods)
    if names.count("vanilla") != 1:
        raise ValueError("The comparison family needs exactly one unaugmented reference")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate methods are not a comparison family")
    return len(names) - 1
