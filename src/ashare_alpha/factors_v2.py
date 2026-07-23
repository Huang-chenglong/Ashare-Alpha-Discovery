from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import Candidate, CONTROL_COLUMNS, _rolling, compute_candidates


CANDIDATES_V2 = {
    "active_positive_gap_reversal_20": Candidate(
        "gap_absorption",
        "High-activity positive overnight gaps that fade intraday reveal disagreement and predict higher returns.",
    ),
    "active_positive_gap_reversal_60": Candidate(
        "gap_absorption",
        "The quarterly high-activity positive-gap reversal fraction predicts higher returns.",
    ),
    "active_negative_gap_recovery_20": Candidate(
        "gap_absorption",
        "High-activity negative overnight gaps recovered intraday reveal latent demand.",
    ),
    "active_negative_gap_recovery_60": Candidate(
        "gap_absorption",
        "The quarterly high-activity negative-gap recovery fraction predicts higher returns.",
    ),
    "burst_close_strength_20": Candidate(
        "auction_strength",
        "Closing near the daily high specifically on turnover bursts reveals persistent demand.",
    ),
    "burst_close_strength_60": Candidate(
        "auction_strength",
        "Quarterly burst-day close strength reveals persistent demand.",
    ),
}
CANDIDATE_COLUMNS_V2 = list(CANDIDATES_V2)

# V2 must be incremental to both standard baselines and the principal V1 temporal
# participation signals. These columns are fixed before V2 outcomes are evaluated.
CONTROL_COLUMNS_V2 = [
    *CONTROL_COLUMNS,
    "diffuse_turnover_20",
    "diffuse_turnover_60",
    "turnover_burst_reversal_20",
    "turnover_burst_reversal_60",
]


def _rolling_ratio(
    frame: pd.DataFrame,
    numerator: str,
    denominator: str,
    horizon: int,
) -> pd.Series:
    numerator_sum = _rolling(frame, numerator, horizon, "sum")
    denominator_sum = _rolling(frame, denominator, horizon, "sum")
    return numerator_sum / denominator_sum.replace(0.0, np.nan)


def compute_candidates_v2(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"high", "low"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"V2 daily panel is missing {sorted(missing)}")

    frame = compute_candidates(daily)
    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    previous_close = frame.groupby("asset", sort=False)["close"].shift(1)
    valid_open = frame["open"].where(frame["open"].gt(0.0))
    valid_close = frame["close"].where(frame["close"].gt(0.0))
    frame["overnight_gap"] = np.log(valid_open / previous_close)
    frame["intraday_return"] = np.log(valid_close / valid_open)
    for column in ["overnight_gap", "intraday_return"]:
        market_component = frame[column].where(eligible).groupby(frame["date"]).transform("median")
        frame[column] = frame[column] - market_component

    usable_turnover = frame["turnover_fraction"].where(
        frame["turnover_fraction"].gt(0.0)
    )
    frame["usable_turnover_v2"] = usable_turnover
    rolling_turnover_median = _rolling(frame, "usable_turnover_v2", 60, "median")
    frame["positive_activity_surprise"] = np.log(
        usable_turnover / rolling_turnover_median.replace(0.0, np.nan)
    ).clip(lower=0.0)

    positive_gap = frame["overnight_gap"].clip(lower=0.0)
    positive_gap_reversal = np.minimum(
        positive_gap,
        (-frame["intraday_return"]).clip(lower=0.0),
    )
    negative_gap = (-frame["overnight_gap"]).clip(lower=0.0)
    negative_gap_recovery = np.minimum(
        negative_gap,
        frame["intraday_return"].clip(lower=0.0),
    )
    frame["positive_gap_reversal_num"] = (
        frame["positive_activity_surprise"] * positive_gap_reversal
    )
    frame["positive_gap_reversal_den"] = (
        frame["positive_activity_surprise"] * positive_gap
    )
    frame["negative_gap_recovery_num"] = (
        frame["positive_activity_surprise"] * negative_gap_recovery
    )
    frame["negative_gap_recovery_den"] = (
        frame["positive_activity_surprise"] * negative_gap
    )

    price_range = (frame["high"] - frame["low"]).where(
        frame["high"].gt(frame["low"])
    )
    frame["close_location"] = (frame["close"] - frame["low"]) / price_range - 0.5
    frame["turnover_squared_v2"] = usable_turnover.pow(2)
    frame["turnover_squared_close_location"] = (
        frame["turnover_squared_v2"] * frame["close_location"]
    )

    for horizon in (20, 60):
        frame[f"active_positive_gap_reversal_{horizon}"] = _rolling_ratio(
            frame,
            "positive_gap_reversal_num",
            "positive_gap_reversal_den",
            horizon,
        )
        frame[f"active_negative_gap_recovery_{horizon}"] = _rolling_ratio(
            frame,
            "negative_gap_recovery_num",
            "negative_gap_recovery_den",
            horizon,
        )
        burst_close = _rolling_ratio(
            frame,
            "turnover_squared_close_location",
            "turnover_squared_v2",
            horizon,
        )
        ordinary_close = _rolling(frame, "close_location", horizon, "mean")
        frame[f"burst_close_strength_{horizon}"] = burst_close - ordinary_close

    return frame.drop(
        columns=[
            "overnight_gap",
            "intraday_return",
            "usable_turnover_v2",
            "positive_activity_surprise",
            "positive_gap_reversal_num",
            "positive_gap_reversal_den",
            "negative_gap_recovery_num",
            "negative_gap_recovery_den",
            "close_location",
            "turnover_squared_v2",
            "turnover_squared_close_location",
        ]
    )
