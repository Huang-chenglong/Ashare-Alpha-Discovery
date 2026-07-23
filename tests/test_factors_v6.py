import pandas as pd

from ashare_alpha.factors_v6 import SUPPLY_PENALTY_WEIGHT


def test_supply_penalty_is_one_sided() -> None:
    supply_z = pd.Series([-2.0, 0.0, 2.0])
    penalty = SUPPLY_PENALTY_WEIGHT * (-supply_z).clip(lower=0.0)
    assert penalty.tolist() == [0.5, 0.0, 0.0]


def test_equal_core_is_reduced_only_by_bad_supply() -> None:
    core = pd.Series([1.0, 1.0])
    supply_z = pd.Series([-1.0, 1.0])
    score = core - SUPPLY_PENALTY_WEIGHT * (-supply_z).clip(lower=0.0)
    assert score.iloc[0] < score.iloc[1]
