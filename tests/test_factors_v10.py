import numpy as np

from ashare_alpha.factors_v10 import REVERSE_MOTIF, _joint_path_statistics


def test_motif_reverse_is_an_involution() -> None:
    indices = np.arange(64)
    np.testing.assert_array_equal(REVERSE_MOTIF[REVERSE_MOTIF], indices)


def test_repeated_palindromic_motif_is_time_reversible() -> None:
    returns = np.tile(np.array([1.0, -1.0, 1.0]), 50)
    activity = np.tile(np.array([1.0, -1.0, 1.0]), 50)
    statistic, _, _, _, _ = _joint_path_statistics(returns, activity, 120)
    assert statistic[-1] <= 0.0
    assert np.isfinite(statistic[-1])
