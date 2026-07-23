import numpy as np
import pandas as pd

from ashare_alpha.statistics import benjamini_hochberg, hac_mean_test


def test_benjamini_hochberg_is_monotone_in_p_value_order() -> None:
    p_values = pd.Series([0.01, 0.04, 0.03, np.nan])
    adjusted = benjamini_hochberg(p_values)
    ordered = adjusted.loc[p_values.dropna().sort_values().index]
    assert ordered.is_monotonic_increasing
    assert np.allclose(adjusted.iloc[:3], [0.03, 0.04, 0.04])
    assert np.isnan(adjusted.iloc[3])


def test_hac_mean_test_detects_positive_constant_signal() -> None:
    values = pd.Series([0.01, 0.02, 0.015, 0.018, 0.013, 0.019])
    statistic, p_value = hac_mean_test(values)
    assert statistic > 0.0
    assert 0.0 <= p_value < 0.05
