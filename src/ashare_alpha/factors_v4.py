from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import Candidate, CONTROL_COLUMNS, _rolling, compute_candidates


CANDIDATES_V4 = {
    "diffuse_upside_variance_20": Candidate(
        "lottery_concentration",
        "Upside variance spread across many days is less lottery-like and predicts higher returns.",
    ),
    "diffuse_upside_variance_60": Candidate(
        "lottery_concentration",
        "Quarterly diffuse upside variance is less lottery-like and predicts higher returns.",
    ),
    "low_active_upside_variance_share_20": Candidate(
        "lottery_activity_alignment",
        "A lower share of active-day variance coming from upside jumps predicts higher returns.",
    ),
    "low_active_upside_variance_share_60": Candidate(
        "lottery_activity_alignment",
        "A lower quarterly active-day upside variance share predicts higher returns.",
    ),
    "low_turnover_upside_amplification_20": Candidate(
        "lottery_activity_alignment",
        "Less amplification of upside variance by turnover bursts predicts higher returns.",
    ),
    "low_turnover_upside_amplification_60": Candidate(
        "lottery_activity_alignment",
        "Less quarterly amplification of upside variance by turnover bursts predicts higher returns.",
    ),
}
CANDIDATE_COLUMNS_V4 = list(CANDIDATES_V4)
LOTTERY_CONTROL_COLUMNS = [
    "max_excess_return_20",
    "max_excess_return_60",
    "realized_skewness_20",
    "realized_skewness_60",
    "diffuse_turnover_20",
    "diffuse_turnover_60",
]
CONTROL_COLUMNS_V4 = [*CONTROL_COLUMNS, *LOTTERY_CONTROL_COLUMNS]


def compute_candidates_v4(daily: pd.DataFrame) -> pd.DataFrame:
    frame = compute_candidates(daily).reset_index(drop=True)
    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    positive_close = frame["close"].where(frame["close"].gt(0.0))
    frame["log_return_v4"] = np.log(positive_close).groupby(
        frame["asset"], sort=False
    ).diff()
    market_return = frame["log_return_v4"].where(eligible).groupby(
        frame["date"]
    ).transform("median")
    frame["excess_return_v4"] = frame["log_return_v4"] - market_return
    frame["positive_return_v4"] = frame["excess_return_v4"].clip(lower=0.0)
    frame["positive_variance_v4"] = frame["positive_return_v4"].pow(2)
    frame["positive_fourth_v4"] = frame["positive_return_v4"].pow(4)
    frame["total_variance_v4"] = frame["excess_return_v4"].pow(2)

    usable_turnover = frame["turnover_fraction"].where(
        frame["turnover_fraction"].gt(0.0)
    )
    frame["usable_turnover_v4"] = usable_turnover
    turnover_median = _rolling(frame, "usable_turnover_v4", 60, "median")
    frame["activity_surprise_v4"] = np.log(
        usable_turnover / turnover_median.replace(0.0, np.nan)
    ).clip(lower=0.0)
    frame["active_positive_variance_v4"] = (
        frame["activity_surprise_v4"] * frame["positive_variance_v4"]
    )
    frame["active_total_variance_v4"] = (
        frame["activity_surprise_v4"] * frame["total_variance_v4"]
    )
    frame["turnover_squared_v4"] = usable_turnover.pow(2)
    frame["turnover_positive_variance_v4"] = (
        frame["turnover_squared_v4"] * frame["positive_variance_v4"]
    )

    for horizon in (20, 60):
        positive_variance_sum = _rolling(
            frame, "positive_variance_v4", horizon, "sum"
        )
        positive_fourth_sum = _rolling(
            frame, "positive_fourth_v4", horizon, "sum"
        )
        upside_hhi = positive_fourth_sum / positive_variance_sum.pow(2).replace(
            0.0, np.nan
        )
        frame[f"diffuse_upside_variance_{horizon}"] = -np.log(
            (horizon * upside_hhi).where(upside_hhi.gt(0.0))
        )

        active_positive = _rolling(
            frame, "active_positive_variance_v4", horizon, "sum"
        )
        active_total = _rolling(frame, "active_total_variance_v4", horizon, "sum")
        frame[f"low_active_upside_variance_share_{horizon}"] = -(
            active_positive / active_total.replace(0.0, np.nan)
        )

        burst_positive = _rolling(
            frame, "turnover_positive_variance_v4", horizon, "sum"
        ) / _rolling(frame, "turnover_squared_v4", horizon, "sum").replace(
            0.0, np.nan
        )
        ordinary_positive = _rolling(
            frame, "positive_variance_v4", horizon, "mean"
        )
        ordinary_total = _rolling(frame, "total_variance_v4", horizon, "mean")
        frame[f"low_turnover_upside_amplification_{horizon}"] = -(
            (burst_positive - ordinary_positive) / ordinary_total.replace(0.0, np.nan)
        )

        frame[f"max_excess_return_{horizon}"] = _rolling(
            frame, "excess_return_v4", horizon, "max"
        )
        frame[f"realized_skewness_{horizon}"] = _rolling(
            frame, "excess_return_v4", horizon, "skew"
        )

    return frame.drop(
        columns=[
            "log_return_v4",
            "excess_return_v4",
            "positive_return_v4",
            "positive_variance_v4",
            "positive_fourth_v4",
            "total_variance_v4",
            "usable_turnover_v4",
            "activity_surprise_v4",
            "active_positive_variance_v4",
            "active_total_variance_v4",
            "turnover_squared_v4",
            "turnover_positive_variance_v4",
        ]
    )
