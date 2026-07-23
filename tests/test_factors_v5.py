import numpy as np
import pandas as pd

from ashare_alpha.factors_v5 import FACTOR_SETS_V5


def soft_min(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(-np.log(np.exp(-array).mean()))


def test_soft_min_is_symmetric_and_penalizes_one_weak_leg() -> None:
    assert abs(soft_min([1.0, 2.0]) - soft_min([2.0, 1.0])) < 1e-12
    assert soft_min([2.0, -1.0]) < soft_min([2.0, 1.0])


def test_every_v5_candidate_uses_at_least_two_constituents() -> None:
    assert all(len(columns) >= 2 for columns in FACTOR_SETS_V5.values())
