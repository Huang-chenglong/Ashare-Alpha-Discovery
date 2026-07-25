from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import compute_baselines
from .factors import Candidate


HORIZONS_V12 = (60, 120)
CANDIDATES_V12 = {
    f"drca_{horizon}": Candidate(
        "directional_range_conversion_asymmetry",
        (
            f"Greater conversion of daily high-low range into the close-to-open body on "
            f"positive than negative intraday days over {horizon} sessions predicts higher returns."
        ),
    )
    for horizon in HORIZONS_V12
}
CANDIDATE_COLUMNS_V12 = list(CANDIDATES_V12)
COMMON_CONTROLS_V12 = [
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
CONTROL_COLUMNS_BY_CANDIDATE_V12 = {
    f"drca_{horizon}": [
        *COMMON_CONTROLS_V12,
        f"intraday_sign_balance_{horizon}",
        f"mean_body_efficiency_{horizon}",
        f"mean_close_location_{horizon}",
        f"linear_intraday_efficiency_correlation_{horizon}",
    ]
    for horizon in HORIZONS_V12
}


def _rolling(frame: pd.DataFrame, column: str, horizon: int, method: str) -> pd.Series:
    grouped = frame.groupby("asset", sort=False)[column]
    roller = grouped.rolling(horizon, min_periods=horizon)
    return getattr(roller, method)().reset_index(level=0, drop=True)


def compute_candidates_v12(daily: pd.DataFrame) -> pd.DataFrame:
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
    body_efficiency = (frame["close"] - frame["open"]).abs() / price_range
    close_location = (frame["close"] - frame["low"]) / price_range - 0.5
    valid = residual.notna() & body_efficiency.notna()
    positive = valid & residual.gt(0.0)
    negative = valid & residual.lt(0.0)
    frame["drca_positive_count"] = positive.astype(float)
    frame["drca_negative_count"] = negative.astype(float)
    frame["drca_positive_efficiency"] = body_efficiency.where(positive, 0.0)
    frame["drca_negative_efficiency"] = body_efficiency.where(negative, 0.0)
    frame["drca_efficiency"] = body_efficiency
    frame["drca_close_location"] = close_location
    frame["drca_residual"] = residual
    frame["drca_efficiency_residual"] = body_efficiency * residual
    for horizon in HORIZONS_V12:
        positive_count = _rolling(frame, "drca_positive_count", horizon, "sum")
        negative_count = _rolling(frame, "drca_negative_count", horizon, "sum")
        positive_mean = _rolling(frame, "drca_positive_efficiency", horizon, "sum") / positive_count.replace(0.0, np.nan)
        negative_mean = _rolling(frame, "drca_negative_efficiency", horizon, "sum") / negative_count.replace(0.0, np.nan)
        enough = positive_count.ge(10) & negative_count.ge(10)
        frame[f"drca_{horizon}"] = (positive_mean - negative_mean).where(enough)
        frame[f"intraday_sign_balance_{horizon}"] = (
            (positive_count - negative_count) / (positive_count + negative_count)
        ).where(enough)
        frame[f"mean_body_efficiency_{horizon}"] = _rolling(
            frame, "drca_efficiency", horizon, "mean"
        )
        frame[f"mean_close_location_{horizon}"] = _rolling(
            frame, "drca_close_location", horizon, "mean"
        )
        covariance = _rolling(frame, "drca_efficiency_residual", horizon, "mean") - (
            _rolling(frame, "drca_efficiency", horizon, "mean")
            * _rolling(frame, "drca_residual", horizon, "mean")
        )
        efficiency_std = _rolling(frame, "drca_efficiency", horizon, "std")
        residual_std = _rolling(frame, "drca_residual", horizon, "std")
        frame[f"linear_intraday_efficiency_correlation_{horizon}"] = covariance / (
            efficiency_std * residual_std
        ).replace(0.0, np.nan)
    return frame.drop(
        columns=[
            "drca_positive_count", "drca_negative_count", "drca_positive_efficiency",
            "drca_negative_efficiency", "drca_efficiency", "drca_close_location",
            "drca_residual", "drca_efficiency_residual",
        ]
    )
