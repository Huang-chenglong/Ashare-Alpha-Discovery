from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def mad_winsorize(values: pd.Series, scale: float = 3.0) -> pd.Series:
    median = values.median()
    mad = (values - median).abs().median()
    if not np.isfinite(mad) or mad <= 1e-12:
        return values.copy()
    width = scale * 1.4826 * mad
    return values.clip(median - width, median + width)


def zscore(values: pd.Series) -> pd.Series:
    standard_deviation = float(values.std(ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation <= 1e-12:
        return values * 0.0
    return (values - values.mean()) / standard_deviation


def hac_mean_test(values: pd.Series, max_lag: int = 3) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 3:
        return np.nan, np.nan
    centered = clean - clean.mean()
    long_run_variance = float(centered @ centered / len(clean))
    effective_lag = min(max_lag, len(clean) - 1)
    for lag in range(1, effective_lag + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / len(clean))
        weight = 1.0 - lag / (max_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    standard_error = np.sqrt(max(long_run_variance, 0.0) / len(clean))
    if standard_error <= 1e-15:
        return np.nan, np.nan
    statistic = float(clean.mean() / standard_error)
    return statistic, float(norm.sf(statistic))


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return output
    count = len(valid)
    adjusted = valid.to_numpy() * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
    output.loc[valid.index] = adjusted
    return output
