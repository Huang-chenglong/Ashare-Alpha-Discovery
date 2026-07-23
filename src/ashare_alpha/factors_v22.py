from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import Candidate
from .factors_v21 import (
    COMMON_CONTROLS_V21,
    HORIZONS_V21,
    _rolling,
    compute_candidates_v21,
)


HORIZONS_V22 = HORIZONS_V21
CANDIDATES_V22 = {
    f"pwrr_{horizon}": Candidate(
        "participation_weighted_range_relaxation",
        (
            "Abnormal-turnover-weighted range relaxation after downside versus upside "
            f"tail events over {horizon} completed sessions predicts higher returns."
        ),
    )
    for horizon in HORIZONS_V22
}
CANDIDATE_COLUMNS_V22 = list(CANDIDATES_V22)
CONTROL_COLUMNS_BY_CANDIDATE_V22 = {
    f"pwrr_{horizon}": [
        *COMMON_CONTROLS_V21,
        f"drra_{horizon}",
        f"range_tail_event_balance_{horizon}",
        f"mean_tail_range_response_{horizon}",
        f"linear_residual_range_response_correlation_{horizon}",
        f"mean_tail_event_log_range_{horizon}",
        f"weighted_tail_event_balance_{horizon}",
        f"mean_tail_event_weight_{horizon}",
    ]
    for horizon in HORIZONS_V22
}


def compute_candidates_v22(daily: pd.DataFrame) -> pd.DataFrame:
    frame = compute_candidates_v21(daily).sort_values(["asset", "date"]).reset_index(drop=True)
    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    close = frame["close"].where(frame["close"].gt(0.0))
    log_return = np.log(close).groupby(frame["asset"], sort=False).diff()
    market = log_return.where(eligible).groupby(frame["date"]).transform("median")
    residual = log_return - market
    eligible_residual = residual.where(eligible)
    lower = eligible_residual.groupby(frame["date"]).transform("quantile", q=0.20)
    upper = eligible_residual.groupby(frame["date"]).transform("quantile", q=0.80)
    bottom_event = eligible & residual.le(lower) & residual.notna()
    top_event = eligible & residual.ge(upper) & residual.notna()
    valid_range = frame["high"].gt(0.0) & frame["low"].gt(0.0) & frame["high"].ge(frame["low"])
    log_range = np.log(frame["high"].where(valid_range) / frame["low"].where(valid_range))
    previous_range = log_range.groupby(frame["asset"], sort=False).shift(1)
    range_response = log_range - previous_range
    turnover = frame["turnover_fraction"].where(frame["turnover_fraction"].gt(0.0))
    trailing_median = turnover.groupby(frame["asset"], sort=False).transform(
        lambda values: values.rolling(60, min_periods=30).median().shift(1)
    )
    event_weight = (turnover / trailing_median.replace(0.0, np.nan)).clip(0.5, 3.0)
    previous_weight = event_weight.groupby(frame["asset"], sort=False).shift(1)
    previous_bottom = bottom_event.groupby(frame["asset"], sort=False).shift(1, fill_value=False)
    previous_top = top_event.groupby(frame["asset"], sort=False).shift(1, fill_value=False)
    completed_bottom = previous_bottom & range_response.notna() & previous_weight.notna()
    completed_top = previous_top & range_response.notna() & previous_weight.notna()
    frame["pwrr_bottom_count"] = completed_bottom.astype(float)
    frame["pwrr_top_count"] = completed_top.astype(float)
    frame["pwrr_bottom_weight"] = previous_weight.where(completed_bottom, 0.0)
    frame["pwrr_top_weight"] = previous_weight.where(completed_top, 0.0)
    frame["pwrr_bottom_weighted_response"] = (
        previous_weight * range_response
    ).where(completed_bottom, 0.0)
    frame["pwrr_top_weighted_response"] = (
        previous_weight * range_response
    ).where(completed_top, 0.0)
    frame["pwrr_tail_count"] = (completed_bottom | completed_top).astype(float)
    frame["pwrr_tail_weight"] = previous_weight.where(completed_bottom | completed_top, 0.0)
    for horizon in HORIZONS_V22:
        bottom_count = _rolling(frame, "pwrr_bottom_count", horizon, "sum")
        top_count = _rolling(frame, "pwrr_top_count", horizon, "sum")
        bottom_weight = _rolling(frame, "pwrr_bottom_weight", horizon, "sum")
        top_weight = _rolling(frame, "pwrr_top_weight", horizon, "sum")
        bottom_mean = _rolling(frame, "pwrr_bottom_weighted_response", horizon, "sum") / bottom_weight.replace(0.0, np.nan)
        top_mean = _rolling(frame, "pwrr_top_weighted_response", horizon, "sum") / top_weight.replace(0.0, np.nan)
        enough = bottom_count.ge(8) & top_count.ge(8)
        frame[f"pwrr_{horizon}"] = (top_mean - bottom_mean).where(enough)
        frame[f"weighted_tail_event_balance_{horizon}"] = (
            (bottom_weight - top_weight) / (bottom_weight + top_weight)
        ).where(enough)
        tail_count = _rolling(frame, "pwrr_tail_count", horizon, "sum")
        frame[f"mean_tail_event_weight_{horizon}"] = (
            _rolling(frame, "pwrr_tail_weight", horizon, "sum")
            / tail_count.replace(0.0, np.nan)
        )
    return frame.drop(columns=[
        "pwrr_bottom_count", "pwrr_top_count", "pwrr_bottom_weight",
        "pwrr_top_weight", "pwrr_bottom_weighted_response",
        "pwrr_top_weighted_response", "pwrr_tail_count", "pwrr_tail_weight",
    ])
