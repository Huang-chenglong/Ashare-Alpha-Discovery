from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import compute_baselines
from .factors import Candidate


HORIZONS_V21 = (60, 90)
CANDIDATES_V21 = {
    f"drra_{horizon}": Candidate(
        "directional_range_relaxation_asymmetry",
        (
            "Faster range relaxation after downside than upside residual-return tails "
            f"over {horizon} completed sessions predicts higher next-month returns."
        ),
    )
    for horizon in HORIZONS_V21
}
CANDIDATE_COLUMNS_V21 = list(CANDIDATES_V21)
COMMON_CONTROLS_V21 = [
    "low_turnover_20", "low_turnover_vol_60", "reversal_20", "low_volatility_60",
    "anti_max_return_20", "anti_skewness_60", "momentum_60_20", "momentum_120_20",
    "overnight_mean_20", "negative_intraday_mean_20", "diffuse_turnover_60",
]
CONTROL_COLUMNS_BY_CANDIDATE_V21 = {
    f"drra_{horizon}": [
        *COMMON_CONTROLS_V21,
        f"range_tail_event_balance_{horizon}",
        f"mean_tail_range_response_{horizon}",
        f"linear_residual_range_response_correlation_{horizon}",
        f"mean_tail_event_log_range_{horizon}",
    ]
    for horizon in HORIZONS_V21
}


def _rolling(frame: pd.DataFrame, column: str, horizon: int, method: str) -> pd.Series:
    roller = frame.groupby("asset", sort=False)[column].rolling(horizon, min_periods=horizon)
    return getattr(roller, method)().reset_index(level=0, drop=True)


def compute_candidates_v21(daily: pd.DataFrame) -> pd.DataFrame:
    frame = compute_baselines(daily).sort_values(["asset", "date"]).reset_index(drop=True)
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
    previous_residual = residual.groupby(frame["asset"], sort=False).shift(1)
    previous_bottom = bottom_event.groupby(frame["asset"], sort=False).shift(1, fill_value=False)
    previous_top = top_event.groupby(frame["asset"], sort=False).shift(1, fill_value=False)
    completed_bottom = previous_bottom & range_response.notna()
    completed_top = previous_top & range_response.notna()
    frame["drra_bottom_count"] = completed_bottom.astype(float)
    frame["drra_top_count"] = completed_top.astype(float)
    frame["drra_bottom_response"] = range_response.where(completed_bottom, 0.0)
    frame["drra_top_response"] = range_response.where(completed_top, 0.0)
    frame["drra_tail_count"] = (completed_bottom | completed_top).astype(float)
    frame["drra_tail_response"] = range_response.where(completed_bottom | completed_top, 0.0)
    frame["drra_tail_event_range"] = previous_range.where(completed_bottom | completed_top, 0.0)
    frame["drra_prior_residual"] = previous_residual
    frame["drra_range_response"] = range_response
    frame["drra_residual_response_product"] = previous_residual * range_response
    for horizon in HORIZONS_V21:
        bottom_count = _rolling(frame, "drra_bottom_count", horizon, "sum")
        top_count = _rolling(frame, "drra_top_count", horizon, "sum")
        bottom_mean = _rolling(frame, "drra_bottom_response", horizon, "sum") / bottom_count.replace(0.0, np.nan)
        top_mean = _rolling(frame, "drra_top_response", horizon, "sum") / top_count.replace(0.0, np.nan)
        enough = bottom_count.ge(8) & top_count.ge(8)
        frame[f"drra_{horizon}"] = (top_mean - bottom_mean).where(enough)
        frame[f"range_tail_event_balance_{horizon}"] = (
            (bottom_count - top_count) / (bottom_count + top_count)
        ).where(enough)
        tail_count = _rolling(frame, "drra_tail_count", horizon, "sum")
        frame[f"mean_tail_range_response_{horizon}"] = (
            _rolling(frame, "drra_tail_response", horizon, "sum")
            / tail_count.replace(0.0, np.nan)
        )
        frame[f"mean_tail_event_log_range_{horizon}"] = (
            _rolling(frame, "drra_tail_event_range", horizon, "sum")
            / tail_count.replace(0.0, np.nan)
        )
        covariance = _rolling(frame, "drra_residual_response_product", horizon, "mean") - (
            _rolling(frame, "drra_prior_residual", horizon, "mean")
            * _rolling(frame, "drra_range_response", horizon, "mean")
        )
        frame[f"linear_residual_range_response_correlation_{horizon}"] = covariance / (
            _rolling(frame, "drra_prior_residual", horizon, "std")
            * _rolling(frame, "drra_range_response", horizon, "std")
        ).replace(0.0, np.nan)
    return frame.drop(columns=[
        "drra_bottom_count", "drra_top_count", "drra_bottom_response",
        "drra_top_response", "drra_tail_count", "drra_tail_response",
        "drra_tail_event_range", "drra_prior_residual", "drra_range_response",
        "drra_residual_response_product",
    ])
