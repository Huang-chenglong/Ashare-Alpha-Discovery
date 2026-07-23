from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import compute_baselines
from .factors import Candidate


ACTIVITY_Z_THRESHOLD = 0.75
ABS_EXCESS_RETURN_THRESHOLD = 0.005
MINIMUM_EVENTS_PER_DIRECTION = 2
PARAMETER_GRID = ((60, 5), (60, 10), (120, 5), (120, 10))

CANDIDATES_V7 = {
    f"disa_{lookback}_{survival}": Candidate(
        "directional_impact_survival_asymmetry",
        (
            f"Positive high-participation shocks surviving longer than negative shocks "
            f"over {lookback} observations and a {survival}-session first-passage horizon "
            "predict positive next-month residual returns."
        ),
    )
    for lookback, survival in PARAMETER_GRID
}
CANDIDATE_COLUMNS_V7 = list(CANDIDATES_V7)

COMMON_CONTROLS_V7 = [
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
CONTROL_COLUMNS_BY_CANDIDATE_V7 = {
    f"disa_{lookback}_{survival}": [
        *COMMON_CONTROLS_V7,
        f"event_direction_balance_{lookback}_{survival}",
    ]
    for lookback, survival in PARAMETER_GRID
}


def _completed_event_rolling_mean(
    values: np.ndarray,
    weights: np.ndarray,
    directions: np.ndarray,
    *,
    lookback: int,
    survival_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate only events whose first-passage path is observable by formation time."""
    count = len(values)
    positive_numerator = np.zeros(count)
    positive_denominator = np.zeros(count)
    positive_count = np.zeros(count)
    negative_numerator = np.zeros(count)
    negative_denominator = np.zeros(count)
    negative_count = np.zeros(count)

    for event_index in range(max(0, count - survival_horizon)):
        direction = directions[event_index]
        weight = weights[event_index]
        if direction == 0.0 or not np.isfinite(weight) or weight <= 0.0:
            continue
        cumulative = 0.0
        survived = 0
        for offset in range(survival_horizon + 1):
            step = values[event_index + offset]
            if not np.isfinite(step):
                break
            cumulative += step
            if direction * cumulative <= 0.0:
                break
            survived += 1
        survival_fraction = survived / (survival_horizon + 1.0)
        completion_index = event_index + survival_horizon
        if direction > 0.0:
            positive_numerator[completion_index] = weight * survival_fraction
            positive_denominator[completion_index] = weight
            positive_count[completion_index] = 1.0
        else:
            negative_numerator[completion_index] = weight * survival_fraction
            negative_denominator[completion_index] = weight
            negative_count[completion_index] = 1.0

    window = lookback - survival_horizon
    if window <= 0:
        raise ValueError("lookback must exceed the survival horizon")

    def rolling_sum(array: np.ndarray) -> np.ndarray:
        return (
            pd.Series(array)
            .rolling(window, min_periods=max(20, window // 2))
            .sum()
            .to_numpy()
        )

    positive_num = rolling_sum(positive_numerator)
    positive_den = rolling_sum(positive_denominator)
    positive_events = rolling_sum(positive_count)
    negative_num = rolling_sum(negative_numerator)
    negative_den = rolling_sum(negative_denominator)
    negative_events = rolling_sum(negative_count)
    enough = (
        (positive_events >= MINIMUM_EVENTS_PER_DIRECTION)
        & (negative_events >= MINIMUM_EVENTS_PER_DIRECTION)
        & (positive_den > 0.0)
        & (negative_den > 0.0)
    )
    positive_mean = np.full(count, np.nan)
    negative_mean = np.full(count, np.nan)
    total_denominator = positive_den + negative_den
    direction_balance = np.full(count, np.nan)
    np.divide(positive_num, positive_den, out=positive_mean, where=enough)
    np.divide(negative_num, negative_den, out=negative_mean, where=enough)
    np.divide(
        positive_den - negative_den,
        total_denominator,
        out=direction_balance,
        where=enough,
    )
    asymmetry = np.where(enough, positive_mean - negative_mean, np.nan)
    return asymmetry, direction_balance


def compute_candidates_v7(daily: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date", "asset", "open", "close", "turnover_fraction", "float_market_cap",
        "is_member", "tradestatus", "is_st",
    }
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Daily panel is missing {sorted(missing)}")
    frame = compute_baselines(daily).sort_values(["asset", "date"]).reset_index(drop=True)
    positive_close = frame["close"].where(frame["close"].gt(0.0))
    frame["disa_log_return"] = np.log(positive_close).groupby(
        frame["asset"], sort=False
    ).diff()
    market_eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    market_median = frame["disa_log_return"].where(market_eligible).groupby(
        frame["date"]
    ).transform("median")
    frame["disa_excess_return"] = frame["disa_log_return"] - market_median
    log_turnover = np.log(
        frame["turnover_fraction"].where(frame["turnover_fraction"].gt(0.0))
    )
    grouped = log_turnover.groupby(frame["asset"], sort=False)
    trailing_mean = grouped.transform(
        lambda values: values.rolling(60, min_periods=30).mean().shift(1)
    )
    trailing_std = grouped.transform(
        lambda values: values.rolling(60, min_periods=30).std(ddof=0).shift(1)
    )
    frame["disa_activity_z"] = (log_turnover - trailing_mean) / trailing_std.replace(
        0.0, np.nan
    )

    parts: list[pd.DataFrame] = []
    for _, rows in frame.groupby("asset", sort=False):
        rows = rows.copy()
        excess = rows["disa_excess_return"].to_numpy(dtype=float)
        activity = rows["disa_activity_z"].to_numpy(dtype=float)
        event = (
            np.isfinite(excess)
            & np.isfinite(activity)
            & (activity >= ACTIVITY_Z_THRESHOLD)
            & (np.abs(excess) >= ABS_EXCESS_RETURN_THRESHOLD)
        )
        directions = np.where(event, np.sign(excess), 0.0)
        weights = np.where(event, np.clip(activity, ACTIVITY_Z_THRESHOLD, 3.0), 0.0)
        for lookback, survival_horizon in PARAMETER_GRID:
            candidate, balance = _completed_event_rolling_mean(
                excess,
                weights,
                directions,
                lookback=lookback,
                survival_horizon=survival_horizon,
            )
            rows[f"disa_{lookback}_{survival_horizon}"] = candidate
            rows[f"event_direction_balance_{lookback}_{survival_horizon}"] = balance
        parts.append(rows)
    output = pd.concat(parts, ignore_index=True).sort_values(["asset", "date"])
    return output.drop(columns=["disa_log_return", "disa_excess_return", "disa_activity_z"])
