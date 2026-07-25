from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import compute_baselines
from .factors import Candidate


HORIZONS_V17 = (60, 90)
CANDIDATES_V17 = {
    f"draa_{horizon}": Candidate(
        "directional_range_acceptance_asymmetry",
        (
            "Greater consecutive high-low range acceptance during positive than negative "
            f"residual center migration over {horizon} sessions predicts higher returns."
        ),
    )
    for horizon in HORIZONS_V17
}
CANDIDATE_COLUMNS_V17 = list(CANDIDATES_V17)
COMMON_CONTROLS_V17 = [
    "low_turnover_20", "low_turnover_vol_60", "reversal_20", "low_volatility_60",
    "anti_max_return_20", "anti_skewness_60", "momentum_60_20", "momentum_120_20",
    "overnight_mean_20", "negative_intraday_mean_20", "diffuse_turnover_60",
]
CONTROL_COLUMNS_BY_CANDIDATE_V17 = {
    f"draa_{horizon}": [
        *COMMON_CONTROLS_V17,
        f"center_sign_balance_{horizon}",
        f"mean_range_acceptance_{horizon}",
        f"mean_absolute_center_displacement_{horizon}",
        f"mean_relative_range_{horizon}",
        f"linear_center_acceptance_correlation_{horizon}",
    ]
    for horizon in HORIZONS_V17
}


def _rolling(frame: pd.DataFrame, column: str, horizon: int, method: str) -> pd.Series:
    roller = frame.groupby("asset", sort=False)[column].rolling(horizon, min_periods=horizon)
    return getattr(roller, method)().reset_index(level=0, drop=True)


def compute_candidates_v17(daily: pd.DataFrame) -> pd.DataFrame:
    frame = compute_baselines(daily).sort_values(["asset", "date"]).reset_index(drop=True)
    center = (frame["high"] + frame["low"]) / 2.0
    center = center.where(center.gt(0.0))
    grouped_asset = frame["asset"]
    prior_center = center.groupby(grouped_asset, sort=False).shift(1)
    prior_high = frame["high"].groupby(grouped_asset, sort=False).shift(1)
    prior_low = frame["low"].groupby(grouped_asset, sort=False).shift(1)
    center_change = np.log(center / prior_center)
    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    market = center_change.where(eligible).groupby(frame["date"]).transform("median")
    residual = center_change - market
    intersection = (
        pd.concat([frame["high"], prior_high], axis=1).min(axis=1)
        - pd.concat([frame["low"], prior_low], axis=1).max(axis=1)
    ).clip(lower=0.0)
    union = (
        pd.concat([frame["high"], prior_high], axis=1).max(axis=1)
        - pd.concat([frame["low"], prior_low], axis=1).min(axis=1)
    ).where(lambda values: values.gt(0.0))
    acceptance = intersection / union
    relative_range = (frame["high"] - frame["low"]) / center
    valid = residual.notna() & acceptance.notna()
    positive = valid & residual.gt(0.0)
    negative = valid & residual.lt(0.0)
    frame["draa_positive_count"] = positive.astype(float)
    frame["draa_negative_count"] = negative.astype(float)
    frame["draa_positive_acceptance"] = acceptance.where(positive, 0.0)
    frame["draa_negative_acceptance"] = acceptance.where(negative, 0.0)
    frame["draa_acceptance"] = acceptance
    frame["draa_abs_center"] = residual.abs()
    frame["draa_relative_range"] = relative_range
    frame["draa_residual"] = residual
    frame["draa_acceptance_residual"] = acceptance * residual
    for horizon in HORIZONS_V17:
        positive_count = _rolling(frame, "draa_positive_count", horizon, "sum")
        negative_count = _rolling(frame, "draa_negative_count", horizon, "sum")
        positive_mean = _rolling(frame, "draa_positive_acceptance", horizon, "sum") / positive_count.replace(0.0, np.nan)
        negative_mean = _rolling(frame, "draa_negative_acceptance", horizon, "sum") / negative_count.replace(0.0, np.nan)
        enough = positive_count.ge(10) & negative_count.ge(10)
        frame[f"draa_{horizon}"] = (positive_mean - negative_mean).where(enough)
        frame[f"center_sign_balance_{horizon}"] = (
            (positive_count - negative_count) / (positive_count + negative_count)
        ).where(enough)
        frame[f"mean_range_acceptance_{horizon}"] = _rolling(frame, "draa_acceptance", horizon, "mean")
        frame[f"mean_absolute_center_displacement_{horizon}"] = _rolling(frame, "draa_abs_center", horizon, "mean")
        frame[f"mean_relative_range_{horizon}"] = _rolling(frame, "draa_relative_range", horizon, "mean")
        covariance = _rolling(frame, "draa_acceptance_residual", horizon, "mean") - (
            _rolling(frame, "draa_acceptance", horizon, "mean")
            * _rolling(frame, "draa_residual", horizon, "mean")
        )
        frame[f"linear_center_acceptance_correlation_{horizon}"] = covariance / (
            _rolling(frame, "draa_acceptance", horizon, "std")
            * _rolling(frame, "draa_residual", horizon, "std")
        ).replace(0.0, np.nan)
    return frame.drop(columns=[
        "draa_positive_count", "draa_negative_count", "draa_positive_acceptance",
        "draa_negative_acceptance", "draa_acceptance", "draa_abs_center",
        "draa_relative_range", "draa_residual", "draa_acceptance_residual",
    ])
