from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import Candidate
from .factors_v8 import (
    COMMON_CONTROLS_V8,
    compute_candidates_v8,
)


HORIZONS_V9 = (60, 120)
CANDIDATES_V9 = {
    f"dpda_{horizon}": Candidate(
        "directional_participation_dispersion_asymmetry",
        (
            f"Participation spread across positive-return days but concentrated in a few "
            f"negative-return shocks over {horizon} sessions predicts higher next-month returns."
        ),
    )
    for horizon in HORIZONS_V9
}
CANDIDATE_COLUMNS_V9 = list(CANDIDATES_V9)
CONTROL_COLUMNS_BY_CANDIDATE_V9 = {
    f"dpda_{horizon}": [
        *COMMON_CONTROLS_V8,
        f"return_sign_balance_{horizon}",
        f"signed_turnover_share_{horizon}",
        f"linear_return_turnover_correlation_{horizon}",
        f"pds_{horizon}",
    ]
    for horizon in HORIZONS_V9
}


def _rolling_sum(frame: pd.DataFrame, column: str, horizon: int) -> pd.Series:
    return (
        frame.groupby("asset", sort=False)[column]
        .rolling(horizon, min_periods=horizon)
        .sum()
        .reset_index(level=0, drop=True)
    )


def compute_candidates_v9(daily: pd.DataFrame) -> pd.DataFrame:
    frame = compute_candidates_v8(daily).sort_values(["asset", "date"]).reset_index(drop=True)
    close = frame["close"].where(frame["close"].gt(0.0))
    log_return = np.log(close).groupby(frame["asset"], sort=False).diff()
    market_eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    market = log_return.where(market_eligible).groupby(frame["date"]).transform("median")
    excess = log_return - market
    turnover = frame["turnover_fraction"].where(frame["turnover_fraction"].gt(0.0))
    positive = excess.gt(0.0) & excess.notna() & turnover.notna()
    negative = excess.lt(0.0) & excess.notna() & turnover.notna()
    frame["dpda_positive_count"] = positive.astype(float)
    frame["dpda_negative_count"] = negative.astype(float)
    frame["dpda_positive_turnover"] = turnover.where(positive, 0.0)
    frame["dpda_negative_turnover"] = turnover.where(negative, 0.0)
    frame["dpda_positive_turnover_sq"] = turnover.pow(2).where(positive, 0.0)
    frame["dpda_negative_turnover_sq"] = turnover.pow(2).where(negative, 0.0)
    for horizon in HORIZONS_V9:
        positive_count = _rolling_sum(frame, "dpda_positive_count", horizon)
        negative_count = _rolling_sum(frame, "dpda_negative_count", horizon)
        positive_sum = _rolling_sum(frame, "dpda_positive_turnover", horizon)
        negative_sum = _rolling_sum(frame, "dpda_negative_turnover", horizon)
        positive_squared = _rolling_sum(frame, "dpda_positive_turnover_sq", horizon)
        negative_squared = _rolling_sum(frame, "dpda_negative_turnover_sq", horizon)
        positive_concentration = (
            positive_count * positive_squared / positive_sum.pow(2).replace(0.0, np.nan)
        )
        negative_concentration = (
            negative_count * negative_squared / negative_sum.pow(2).replace(0.0, np.nan)
        )
        enough = positive_count.ge(10) & negative_count.ge(10)
        frame[f"dpda_{horizon}"] = np.log(
            negative_concentration.where(enough & negative_concentration.gt(0.0))
        ) - np.log(
            positive_concentration.where(enough & positive_concentration.gt(0.0))
        )
    return frame.drop(
        columns=[
            "dpda_positive_count", "dpda_negative_count", "dpda_positive_turnover",
            "dpda_negative_turnover", "dpda_positive_turnover_sq",
            "dpda_negative_turnover_sq",
        ]
    )
