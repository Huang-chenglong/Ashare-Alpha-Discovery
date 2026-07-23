from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .baselines import compute_baselines
from .factors import Candidate


HORIZONS_V8 = (20, 60, 120)
MINIMUM_DAYS_PER_DIRECTION = 5
CANDIDATES_V8 = {
    f"pds_{horizon}": Candidate(
        "participation_distribution_dominance",
        (
            f"A stock whose turnover distribution on positive residual-return days "
            f"stochastically dominates its turnover distribution on negative days over "
            f"{horizon} sessions earns higher next-month residual returns."
        ),
    )
    for horizon in HORIZONS_V8
}
CANDIDATE_COLUMNS_V8 = list(CANDIDATES_V8)
COMMON_CONTROLS_V8 = [
    "low_turnover_20",
    "low_turnover_vol_60",
    "reversal_20",
    "low_volatility_60",
    "anti_max_return_20",
    "anti_skewness_60",
    "momentum_60_20",
    "momentum_120_20",
    "overnight_mean_20",
    "negative_intraday_mean_20",
    "diffuse_turnover_60",
]
CONTROL_COLUMNS_BY_CANDIDATE_V8 = {
    f"pds_{horizon}": [
        *COMMON_CONTROLS_V8,
        f"return_sign_balance_{horizon}",
        f"signed_turnover_share_{horizon}",
        f"linear_return_turnover_correlation_{horizon}",
    ]
    for horizon in HORIZONS_V8
}


def _rolling_participation_dominance(
    excess_return: np.ndarray,
    turnover: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(excess_return)
    dominance = np.full(count, np.nan)
    sign_balance = np.full(count, np.nan)
    signed_share = np.full(count, np.nan)
    linear_correlation = np.full(count, np.nan)
    for end in range(horizon - 1, count):
        start = end - horizon + 1
        returns = excess_return[start : end + 1]
        activity = turnover[start : end + 1]
        valid = np.isfinite(returns) & np.isfinite(activity) & (returns != 0.0) & (activity > 0.0)
        returns = returns[valid]
        activity = activity[valid]
        positive = returns > 0.0
        positive_count = int(positive.sum())
        negative_count = int((~positive).sum())
        if (
            positive_count < MINIMUM_DAYS_PER_DIRECTION
            or negative_count < MINIMUM_DAYS_PER_DIRECTION
        ):
            continue
        ranks = rankdata(activity, method="average")
        u_positive = float(ranks[positive].sum()) - positive_count * (positive_count + 1.0) / 2.0
        dominance[end] = 2.0 * u_positive / (positive_count * negative_count) - 1.0
        sign_balance[end] = (positive_count - negative_count) / (positive_count + negative_count)
        total_activity = float(activity.sum())
        signed_share[end] = (
            float(activity[positive].sum()) - float(activity[~positive].sum())
        ) / total_activity
        if np.std(returns) > 1e-12 and np.std(np.log(activity)) > 1e-12:
            linear_correlation[end] = np.corrcoef(returns, np.log(activity))[0, 1]
    return dominance, sign_balance, signed_share, linear_correlation


def compute_candidates_v8(daily: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date", "asset", "open", "close", "turnover_fraction", "float_market_cap",
        "is_member", "tradestatus", "is_st",
    }
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Daily panel is missing {sorted(missing)}")
    frame = compute_baselines(daily).sort_values(["asset", "date"]).reset_index(drop=True)
    close = frame["close"].where(frame["close"].gt(0.0))
    frame["pds_log_return"] = np.log(close).groupby(frame["asset"], sort=False).diff()
    market_eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    market = frame["pds_log_return"].where(market_eligible).groupby(
        frame["date"]
    ).transform("median")
    frame["pds_excess_return"] = frame["pds_log_return"] - market
    parts: list[pd.DataFrame] = []
    for _, rows in frame.groupby("asset", sort=False):
        rows = rows.copy()
        returns = rows["pds_excess_return"].to_numpy(dtype=float)
        turnover = rows["turnover_fraction"].to_numpy(dtype=float)
        for horizon in HORIZONS_V8:
            dominance, balance, share, correlation = _rolling_participation_dominance(
                returns, turnover, horizon
            )
            rows[f"pds_{horizon}"] = dominance
            rows[f"return_sign_balance_{horizon}"] = balance
            rows[f"signed_turnover_share_{horizon}"] = share
            rows[f"linear_return_turnover_correlation_{horizon}"] = correlation
        parts.append(rows)
    return (
        pd.concat(parts, ignore_index=True)
        .drop(columns=["pds_log_return", "pds_excess_return"])
        .sort_values(["asset", "date"])
    )
