import numpy as np

from ashare_alpha.factors_v7 import _completed_event_rolling_mean


def test_completed_event_statistic_has_no_future_dependency() -> None:
    values = np.tile(np.array([0.02, 0.01, -0.005, -0.015, 0.004]), 30)
    directions = np.zeros(len(values))
    weights = np.zeros(len(values))
    for index in range(10, 120, 10):
        directions[index] = 1.0 if (index // 10) % 2 == 0 else -1.0
        weights[index] = 1.0
        values[index] = 0.02 * directions[index]
    original, _ = _completed_event_rolling_mean(
        values, weights, directions, lookback=60, survival_horizon=5
    )
    mutated_values = values.copy()
    mutated_values[101:] = mutated_values[101:] * -11.0
    mutated, _ = _completed_event_rolling_mean(
        mutated_values, weights, directions, lookback=60, survival_horizon=5
    )
    np.testing.assert_allclose(original[:101], mutated[:101], equal_nan=True)


def test_direction_balance_is_separate_from_survival_asymmetry() -> None:
    values = np.full(100, 0.001)
    directions = np.zeros(100)
    weights = np.zeros(100)
    for index in (10, 30, 50):
        directions[index] = 1.0
        weights[index] = 2.0
        values[index : index + 6] = 0.01
    for index in (20, 40, 60):
        directions[index] = -1.0
        weights[index] = 1.0
        values[index] = -0.01
        values[index + 1] = 0.02
    asymmetry, balance = _completed_event_rolling_mean(
        values, weights, directions, lookback=60, survival_horizon=5
    )
    assert asymmetry[65] > 0.5
    assert balance[65] > 0.0

