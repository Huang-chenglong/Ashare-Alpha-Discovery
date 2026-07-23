from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import compute_baselines
from .factors import Candidate


HORIZONS_V10 = (60, 120)
MOTIF_COUNT = 64
JEFFREYS_PSEUDOCOUNT = 0.5
CANDIDATES_V10 = {
    f"joint_path_reversibility_{horizon}": Candidate(
        "return_participation_time_reversibility",
        (
            f"Lower time-arrow asymmetry in three-day joint return-sign and participation "
            f"motifs over {horizon} sessions predicts higher next-month returns."
        ),
    )
    for horizon in HORIZONS_V10
}
CANDIDATE_COLUMNS_V10 = list(CANDIDATES_V10)
COMMON_CONTROLS_V10 = [
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
CONTROL_COLUMNS_BY_CANDIDATE_V10 = {
    f"joint_path_reversibility_{horizon}": [
        *COMMON_CONTROLS_V10,
        f"joint_motif_entropy_{horizon}",
        f"joint_return_sign_balance_{horizon}",
        f"joint_activity_balance_{horizon}",
        f"joint_linear_return_turnover_correlation_{horizon}",
    ]
    for horizon in HORIZONS_V10
}


def _reverse_motif_indices() -> np.ndarray:
    indices = np.arange(MOTIF_COUNT)
    first = indices // 16
    middle = (indices // 4) % 4
    last = indices % 4
    return last * 16 + middle * 4 + first


REVERSE_MOTIF = _reverse_motif_indices()


def _joint_path_statistics(
    returns: np.ndarray,
    activity: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(returns)
    reversibility = np.full(count, np.nan)
    entropy = np.full(count, np.nan)
    return_balance = np.full(count, np.nan)
    activity_balance = np.full(count, np.nan)
    linear_correlation = np.full(count, np.nan)
    valid = np.isfinite(returns) & np.isfinite(activity)
    states = np.full(count, -1, dtype=int)
    states[valid] = (returns[valid] > 0.0).astype(int) * 2 + (activity[valid] > 0.0).astype(int)
    motifs = np.full(count, -1, dtype=int)
    for index in range(2, count):
        if np.all(states[index - 2 : index + 1] >= 0):
            motifs[index] = states[index - 2] * 16 + states[index - 1] * 4 + states[index]
    for end in range(horizon - 1, count):
        start = end - horizon + 1
        window_motifs = motifs[start + 2 : end + 1]
        window_motifs = window_motifs[window_motifs >= 0]
        window_valid = valid[start : end + 1]
        if len(window_motifs) < horizon // 2 or window_valid.sum() < horizon // 2:
            continue
        counts = np.bincount(window_motifs, minlength=MOTIF_COUNT).astype(float)
        probability = (counts + JEFFREYS_PSEUDOCOUNT) / (
            counts.sum() + JEFFREYS_PSEUDOCOUNT * MOTIF_COUNT
        )
        reversed_probability = probability[REVERSE_MOTIF]
        mixture = 0.5 * (probability + reversed_probability)
        js_divergence = 0.5 * np.sum(probability * np.log(probability / mixture))
        js_divergence += 0.5 * np.sum(
            reversed_probability * np.log(reversed_probability / mixture)
        )
        reversibility[end] = -js_divergence
        entropy[end] = -np.sum(probability * np.log(probability)) / np.log(MOTIF_COUNT)
        window_returns = returns[start : end + 1][window_valid]
        window_activity = activity[start : end + 1][window_valid]
        return_balance[end] = np.mean(np.where(window_returns > 0.0, 1.0, -1.0))
        activity_balance[end] = np.mean(np.where(window_activity > 0.0, 1.0, -1.0))
        if np.std(window_returns) > 1e-12 and np.std(window_activity) > 1e-12:
            linear_correlation[end] = np.corrcoef(window_returns, window_activity)[0, 1]
    return reversibility, entropy, return_balance, activity_balance, linear_correlation


def compute_candidates_v10(
    daily: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = HORIZONS_V10,
) -> pd.DataFrame:
    frame = compute_baselines(daily).sort_values(["asset", "date"]).reset_index(drop=True)
    close = frame["close"].where(frame["close"].gt(0.0))
    log_return = np.log(close).groupby(frame["asset"], sort=False).diff()
    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    market = log_return.where(eligible).groupby(frame["date"]).transform("median")
    excess = log_return - market
    log_turnover = np.log(
        frame["turnover_fraction"].where(frame["turnover_fraction"].gt(0.0))
    )
    trailing_median = log_turnover.groupby(frame["asset"], sort=False).transform(
        lambda values: values.rolling(60, min_periods=30).median().shift(1)
    )
    activity = log_turnover - trailing_median
    frame["joint_excess_return"] = excess
    frame["joint_activity_state"] = activity
    parts: list[pd.DataFrame] = []
    for _, rows in frame.groupby("asset", sort=False):
        rows = rows.copy()
        returns = rows["joint_excess_return"].to_numpy(dtype=float)
        participation = rows["joint_activity_state"].to_numpy(dtype=float)
        for horizon in horizons:
            reversible, entropy, return_balance, activity_balance, correlation = (
                _joint_path_statistics(returns, participation, horizon)
            )
            rows[f"joint_path_reversibility_{horizon}"] = reversible
            rows[f"joint_motif_entropy_{horizon}"] = entropy
            rows[f"joint_return_sign_balance_{horizon}"] = return_balance
            rows[f"joint_activity_balance_{horizon}"] = activity_balance
            rows[f"joint_linear_return_turnover_correlation_{horizon}"] = correlation
        parts.append(rows)
    return (
        pd.concat(parts, ignore_index=True)
        .drop(columns=["joint_excess_return", "joint_activity_state"])
        .sort_values(["asset", "date"])
    )
