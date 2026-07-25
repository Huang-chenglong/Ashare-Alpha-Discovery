import numpy as np

from ashare_alpha.factors_v8 import _rolling_participation_dominance


def test_participation_dominance_is_one_for_complete_separation() -> None:
    returns = np.array([1.0, -1.0] * 10)
    turnover = np.where(returns > 0.0, 2.0, 1.0)
    dominance, balance, _, _ = _rolling_participation_dominance(
        returns, turnover, horizon=20
    )
    assert dominance[-1] == 1.0
    assert balance[-1] == 0.0


def test_participation_dominance_does_not_depend_on_return_magnitude() -> None:
    returns = np.array([1.0, -1.0] * 10)
    turnover = np.linspace(1.0, 2.0, 20)
    first, _, _, _ = _rolling_participation_dominance(returns, turnover, 20)
    returns[returns > 0.0] *= 100.0
    second, _, _, _ = _rolling_participation_dominance(returns, turnover, 20)
    assert first[-1] == second[-1]
