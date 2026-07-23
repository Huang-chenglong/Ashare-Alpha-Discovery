from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import compute_baselines
from .factors import Candidate


HORIZONS_V16 = (60, 90)
CANDIDATES_V16 = {
    f"dvca_{horizon}": Candidate(
        "directional_volume_clock_asymmetry",
        (
            "More front-loaded volume-clock geometry on positive than negative residual "
            f"intraday sessions over {horizon} days predicts higher next-month returns."
        ),
    )
    for horizon in HORIZONS_V16
}
CANDIDATE_COLUMNS_V16 = list(CANDIDATES_V16)
COMMON_CONTROLS_V16 = [
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
CONTROL_COLUMNS_BY_CANDIDATE_V16 = {
    f"dvca_{horizon}": [
        *COMMON_CONTROLS_V16,
        f"intraday_sign_balance_{horizon}",
        f"mean_volume_clock_{horizon}",
        f"mean_body_efficiency_{horizon}",
        f"mean_signed_vwap_midpoint_{horizon}",
        f"linear_residual_clock_correlation_{horizon}",
    ]
    for horizon in HORIZONS_V16
}


def _rolling(frame: pd.DataFrame, column: str, horizon: int, method: str) -> pd.Series:
    grouped = frame.groupby("asset", sort=False)[column]
    roller = grouped.rolling(horizon, min_periods=horizon)
    return getattr(roller, method)().reset_index(level=0, drop=True)


def compute_candidates_v16(daily: pd.DataFrame) -> pd.DataFrame:
    if "adjusted_vwap" not in daily:
        raise ValueError("V16 requires adjusted_vwap from actual TDX amount and volume")
    frame = compute_baselines(daily).sort_values(["asset", "date"]).reset_index(drop=True)
    valid_open = frame["open"].where(frame["open"].gt(0.0))
    valid_close = frame["close"].where(frame["close"].gt(0.0))
    intraday = np.log(valid_close / valid_open)
    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    market = intraday.where(eligible).groupby(frame["date"]).transform("median")
    residual = intraday - market
    price_range = (frame["high"] - frame["low"]).where(frame["high"].gt(frame["low"]))
    midpoint = (frame["open"] + frame["close"]) / 2.0
    signed_midpoint = (midpoint - frame["adjusted_vwap"]) / price_range
    direction = np.sign(frame["close"] - frame["open"])
    clock = direction * signed_midpoint
    body_efficiency = (frame["close"] - frame["open"]).abs() / price_range
    valid = residual.notna() & clock.notna() & direction.ne(0.0)
    positive = valid & residual.gt(0.0)
    negative = valid & residual.lt(0.0)
    frame["dvca_positive_count"] = positive.astype(float)
    frame["dvca_negative_count"] = negative.astype(float)
    frame["dvca_positive_clock"] = clock.where(positive, 0.0)
    frame["dvca_negative_clock"] = clock.where(negative, 0.0)
    frame["dvca_clock"] = clock
    frame["dvca_body_efficiency"] = body_efficiency
    frame["dvca_signed_midpoint"] = signed_midpoint
    frame["dvca_residual"] = residual
    frame["dvca_clock_residual"] = clock * residual
    for horizon in HORIZONS_V16:
        positive_count = _rolling(frame, "dvca_positive_count", horizon, "sum")
        negative_count = _rolling(frame, "dvca_negative_count", horizon, "sum")
        positive_mean = _rolling(frame, "dvca_positive_clock", horizon, "sum") / positive_count.replace(0.0, np.nan)
        negative_mean = _rolling(frame, "dvca_negative_clock", horizon, "sum") / negative_count.replace(0.0, np.nan)
        enough = positive_count.ge(10) & negative_count.ge(10)
        frame[f"dvca_{horizon}"] = (positive_mean - negative_mean).where(enough)
        frame[f"intraday_sign_balance_{horizon}"] = (
            (positive_count - negative_count) / (positive_count + negative_count)
        ).where(enough)
        frame[f"mean_volume_clock_{horizon}"] = _rolling(frame, "dvca_clock", horizon, "mean")
        frame[f"mean_body_efficiency_{horizon}"] = _rolling(
            frame, "dvca_body_efficiency", horizon, "mean"
        )
        frame[f"mean_signed_vwap_midpoint_{horizon}"] = _rolling(
            frame, "dvca_signed_midpoint", horizon, "mean"
        )
        covariance = _rolling(frame, "dvca_clock_residual", horizon, "mean") - (
            _rolling(frame, "dvca_clock", horizon, "mean")
            * _rolling(frame, "dvca_residual", horizon, "mean")
        )
        frame[f"linear_residual_clock_correlation_{horizon}"] = covariance / (
            _rolling(frame, "dvca_clock", horizon, "std")
            * _rolling(frame, "dvca_residual", horizon, "std")
        ).replace(0.0, np.nan)
    return frame.drop(
        columns=[
            "dvca_positive_count", "dvca_negative_count", "dvca_positive_clock",
            "dvca_negative_clock", "dvca_clock", "dvca_body_efficiency",
            "dvca_signed_midpoint", "dvca_residual", "dvca_clock_residual",
        ]
    )
