from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import Candidate
from .factors_v21 import COMMON_CONTROLS_V21, compute_candidates_v21


HORIZONS_V29 = (60, 90)
MINIMUM_EVENTS_PER_TAIL_V29 = 8
CANDIDATES_V29 = {
    f"tsra_{horizon}": Candidate(
        "tail_shadow_rejection_asymmetry",
        (
            "Stronger next-session lower-shadow rejection after downside tails than "
            f"upper-shadow rejection after upside tails over {horizon} sessions predicts higher returns."
        ),
    )
    for horizon in HORIZONS_V29
}
CANDIDATE_COLUMNS_V29 = list(CANDIDATES_V29)
CONTROL_COLUMNS_BY_CANDIDATE_V29 = {
    f"tsra_{horizon}": [
        *COMMON_CONTROLS_V21,
        f"drra_{horizon}",
        f"range_tail_event_balance_{horizon}",
        f"mean_tail_range_response_{horizon}",
        f"linear_residual_range_response_correlation_{horizon}",
        f"mean_tail_event_log_range_{horizon}",
        f"mean_lower_shadow_share_{horizon}",
        f"mean_upper_shadow_share_{horizon}",
        f"mean_body_efficiency_share_{horizon}",
    ]
    for horizon in HORIZONS_V29
}


def _rolling(frame: pd.DataFrame, column: str, horizon: int, method: str) -> pd.Series:
    roller = frame.groupby("asset", sort=False)[column].rolling(
        horizon, min_periods=horizon
    )
    return getattr(roller, method)().reset_index(level=0, drop=True)


def compute_candidates_v29(daily: pd.DataFrame) -> pd.DataFrame:
    frame = compute_candidates_v21(daily).sort_values(["asset", "date"]).reset_index(
        drop=True
    )
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
    lower_tail = eligible_residual.groupby(frame["date"]).transform("quantile", q=0.20)
    upper_tail = eligible_residual.groupby(frame["date"]).transform("quantile", q=0.80)
    bottom_event = eligible & residual.le(lower_tail) & residual.notna()
    top_event = eligible & residual.ge(upper_tail) & residual.notna()
    previous_bottom = bottom_event.groupby(frame["asset"], sort=False).shift(
        1, fill_value=False
    )
    previous_top = top_event.groupby(frame["asset"], sort=False).shift(
        1, fill_value=False
    )

    price_range = (frame["high"] - frame["low"]).where(frame["high"].gt(frame["low"]))
    lower_shadow = (
        pd.concat([frame["open"], frame["close"]], axis=1).min(axis=1) - frame["low"]
    ) / price_range
    upper_shadow = (
        frame["high"] - pd.concat([frame["open"], frame["close"]], axis=1).max(axis=1)
    ) / price_range
    body_efficiency = (frame["close"] - frame["open"]).abs() / price_range
    valid_shadow = (
        eligible
        & lower_shadow.between(0.0, 1.0)
        & upper_shadow.between(0.0, 1.0)
        & body_efficiency.between(0.0, 1.0)
    )
    completed_bottom = previous_bottom & valid_shadow
    completed_top = previous_top & valid_shadow
    frame["tsra_bottom_count"] = completed_bottom.astype(float)
    frame["tsra_top_count"] = completed_top.astype(float)
    frame["tsra_bottom_rejection"] = lower_shadow.where(completed_bottom, 0.0)
    frame["tsra_top_rejection"] = upper_shadow.where(completed_top, 0.0)
    frame["tsra_lower_shadow"] = lower_shadow.where(valid_shadow)
    frame["tsra_upper_shadow"] = upper_shadow.where(valid_shadow)
    frame["tsra_body_efficiency"] = body_efficiency.where(valid_shadow)

    for horizon in HORIZONS_V29:
        bottom_count = _rolling(frame, "tsra_bottom_count", horizon, "sum")
        top_count = _rolling(frame, "tsra_top_count", horizon, "sum")
        enough = (
            bottom_count.ge(MINIMUM_EVENTS_PER_TAIL_V29)
            & top_count.ge(MINIMUM_EVENTS_PER_TAIL_V29)
        )
        bottom_mean = _rolling(
            frame, "tsra_bottom_rejection", horizon, "sum"
        ) / bottom_count.replace(0.0, np.nan)
        top_mean = _rolling(
            frame, "tsra_top_rejection", horizon, "sum"
        ) / top_count.replace(0.0, np.nan)
        frame[f"tsra_{horizon}"] = (bottom_mean - top_mean).where(enough)
        frame[f"mean_lower_shadow_share_{horizon}"] = _rolling(
            frame, "tsra_lower_shadow", horizon, "mean"
        )
        frame[f"mean_upper_shadow_share_{horizon}"] = _rolling(
            frame, "tsra_upper_shadow", horizon, "mean"
        )
        frame[f"mean_body_efficiency_share_{horizon}"] = _rolling(
            frame, "tsra_body_efficiency", horizon, "mean"
        )

    return frame.drop(columns=[
        "tsra_bottom_count", "tsra_top_count", "tsra_bottom_rejection",
        "tsra_top_rejection", "tsra_lower_shadow", "tsra_upper_shadow",
        "tsra_body_efficiency",
    ])

