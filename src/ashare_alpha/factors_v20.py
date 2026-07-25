from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import compute_baselines
from .factors import Candidate


HORIZONS_V20 = (60, 90)
CANDIDATES_V20 = {
    f"strq_{horizon}": Candidate(
        "signed_tail_response_quality",
        (
            "Stronger downside-tail repair plus upside-tail persistence over completed "
            f"events in {horizon} sessions predicts higher next-month returns."
        ),
    )
    for horizon in HORIZONS_V20
}
CANDIDATE_COLUMNS_V20 = list(CANDIDATES_V20)
COMMON_CONTROLS_V20 = [
    "low_turnover_20", "low_turnover_vol_60", "reversal_20", "low_volatility_60",
    "anti_max_return_20", "anti_skewness_60", "momentum_60_20", "momentum_120_20",
    "overnight_mean_20", "negative_intraday_mean_20", "diffuse_turnover_60",
]
CONTROL_COLUMNS_BY_CANDIDATE_V20 = {
    f"strq_{horizon}": [
        *COMMON_CONTROLS_V20,
        f"tail_event_balance_{horizon}",
        f"tail_response_asymmetry_{horizon}",
        f"lag_one_residual_correlation_{horizon}",
        f"mean_tail_event_magnitude_{horizon}",
    ]
    for horizon in HORIZONS_V20
}


def _rolling(frame: pd.DataFrame, column: str, horizon: int, method: str) -> pd.Series:
    roller = frame.groupby("asset", sort=False)[column].rolling(horizon, min_periods=horizon)
    return getattr(roller, method)().reset_index(level=0, drop=True)


def compute_candidates_v20(daily: pd.DataFrame) -> pd.DataFrame:
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
    previous_residual = residual.groupby(frame["asset"], sort=False).shift(1)
    previous_bottom = bottom_event.groupby(frame["asset"], sort=False).shift(1, fill_value=False)
    previous_top = top_event.groupby(frame["asset"], sort=False).shift(1, fill_value=False)
    response_ratio = (residual / previous_residual.abs().replace(0.0, np.nan)).clip(-2.0, 2.0)
    completed_bottom = previous_bottom & response_ratio.notna()
    completed_top = previous_top & response_ratio.notna()
    frame["strq_bottom_count"] = completed_bottom.astype(float)
    frame["strq_top_count"] = completed_top.astype(float)
    frame["strq_bottom_response"] = response_ratio.where(completed_bottom, 0.0)
    frame["strq_top_response"] = response_ratio.where(completed_top, 0.0)
    frame["strq_tail_magnitude"] = previous_residual.abs().where(
        completed_bottom | completed_top, 0.0
    )
    frame["strq_tail_count"] = (completed_bottom | completed_top).astype(float)
    frame["strq_residual"] = residual
    frame["strq_previous_residual"] = previous_residual
    frame["strq_residual_product"] = residual * previous_residual
    for horizon in HORIZONS_V20:
        bottom_count = _rolling(frame, "strq_bottom_count", horizon, "sum")
        top_count = _rolling(frame, "strq_top_count", horizon, "sum")
        bottom_mean = _rolling(frame, "strq_bottom_response", horizon, "sum") / bottom_count.replace(0.0, np.nan)
        top_mean = _rolling(frame, "strq_top_response", horizon, "sum") / top_count.replace(0.0, np.nan)
        enough = bottom_count.ge(8) & top_count.ge(8)
        frame[f"strq_{horizon}"] = (bottom_mean + top_mean).where(enough)
        frame[f"tail_event_balance_{horizon}"] = (
            (bottom_count - top_count) / (bottom_count + top_count)
        ).where(enough)
        frame[f"tail_response_asymmetry_{horizon}"] = (bottom_mean - top_mean).where(enough)
        tail_count = _rolling(frame, "strq_tail_count", horizon, "sum")
        frame[f"mean_tail_event_magnitude_{horizon}"] = (
            _rolling(frame, "strq_tail_magnitude", horizon, "sum")
            / tail_count.replace(0.0, np.nan)
        )
        covariance = _rolling(frame, "strq_residual_product", horizon, "mean") - (
            _rolling(frame, "strq_residual", horizon, "mean")
            * _rolling(frame, "strq_previous_residual", horizon, "mean")
        )
        frame[f"lag_one_residual_correlation_{horizon}"] = covariance / (
            _rolling(frame, "strq_residual", horizon, "std")
            * _rolling(frame, "strq_previous_residual", horizon, "std")
        ).replace(0.0, np.nan)
    return frame.drop(columns=[
        "strq_bottom_count", "strq_top_count", "strq_bottom_response",
        "strq_top_response", "strq_tail_magnitude", "strq_tail_count",
        "strq_residual", "strq_previous_residual", "strq_residual_product",
    ])
