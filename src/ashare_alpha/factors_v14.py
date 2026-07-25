from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import compute_baselines
from .factors import Candidate


HORIZONS_V14 = (60, 120)
CANDIDATES_V14 = {
    f"vsti_{horizon}": Candidate(
        "vwap_state_transition_imbalance",
        (
            "A higher sequential imbalance of below-to-up-cross-to-above VWAP states "
            f"versus above-to-down-cross-to-below states over {horizon} sessions predicts "
            "higher next-month returns."
        ),
    )
    for horizon in HORIZONS_V14
}
CANDIDATE_COLUMNS_V14 = list(CANDIDATES_V14)
COMMON_CONTROLS_V14 = [
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
CONTROL_COLUMNS_BY_CANDIDATE_V14 = {
    f"vsti_{horizon}": [
        *COMMON_CONTROLS_V14,
        f"mean_open_vwap_position_{horizon}",
        f"mean_close_vwap_position_{horizon}",
        f"cross_direction_balance_{horizon}",
        f"state_transition_activity_{horizon}",
    ]
    for horizon in HORIZONS_V14
}


def _rolling(frame: pd.DataFrame, column: str, horizon: int, method: str) -> pd.Series:
    grouped = frame.groupby("asset", sort=False)[column]
    roller = grouped.rolling(horizon, min_periods=horizon)
    return getattr(roller, method)().reset_index(level=0, drop=True)


def _vwap_state(frame: pd.DataFrame) -> pd.Series:
    valid = (
        frame["adjusted_vwap"].gt(0.0)
        & frame["open"].gt(0.0)
        & frame["close"].gt(0.0)
        & frame["tradestatus"].fillna(0).eq(1)
    )
    state = pd.Series(pd.NA, index=frame.index, dtype="Int8")
    vwap = frame["adjusted_vwap"]
    below = valid & frame["open"].lt(vwap) & frame["close"].lt(vwap)
    above = valid & frame["open"].gt(vwap) & frame["close"].gt(vwap)
    up_cross = (
        valid
        & frame["open"].le(vwap)
        & frame["close"].ge(vwap)
        & frame["close"].gt(frame["open"])
    )
    down_cross = (
        valid
        & frame["open"].ge(vwap)
        & frame["close"].le(vwap)
        & frame["close"].lt(frame["open"])
    )
    state.loc[below] = 0
    state.loc[up_cross] = 1
    state.loc[above] = 2
    state.loc[down_cross] = 3
    return state


def compute_candidates_v14(daily: pd.DataFrame) -> pd.DataFrame:
    if "adjusted_vwap" not in daily:
        raise ValueError("V14 requires adjusted_vwap from actual TDX amount and volume")
    frame = compute_baselines(daily).sort_values(["asset", "date"]).reset_index(drop=True)
    state = _vwap_state(frame)
    previous = state.groupby(frame["asset"], sort=False).shift(1)
    adjacent = state.notna() & previous.notna()
    positive = adjacent & (
        (previous.eq(0) & state.eq(1)) | (previous.eq(1) & state.eq(2))
    )
    negative = adjacent & (
        (previous.eq(2) & state.eq(3)) | (previous.eq(3) & state.eq(0))
    )
    any_change = adjacent & state.ne(previous)
    price_range = (frame["high"] - frame["low"]).where(frame["high"].gt(frame["low"]))
    frame["vsti_score"] = positive.astype(float) - negative.astype(float)
    frame["vsti_directional_event"] = (positive | negative).astype(float)
    frame["vsti_adjacent"] = adjacent.astype(float)
    frame["vsti_state_change"] = any_change.astype(float)
    frame["vsti_open_position"] = (frame["open"] - frame["adjusted_vwap"]) / price_range
    frame["vsti_close_position"] = (frame["close"] - frame["adjusted_vwap"]) / price_range
    frame["vsti_up_cross"] = state.eq(1).astype(float)
    frame["vsti_down_cross"] = state.eq(3).astype(float)
    for horizon in HORIZONS_V14:
        directional = _rolling(frame, "vsti_directional_event", horizon, "sum")
        adjacent_count = _rolling(frame, "vsti_adjacent", horizon, "sum")
        score = _rolling(frame, "vsti_score", horizon, "sum")
        frame[f"vsti_{horizon}"] = (score / adjacent_count.replace(0.0, np.nan)).where(
            directional.ge(5)
        )
        frame[f"mean_open_vwap_position_{horizon}"] = _rolling(
            frame, "vsti_open_position", horizon, "mean"
        )
        frame[f"mean_close_vwap_position_{horizon}"] = _rolling(
            frame, "vsti_close_position", horizon, "mean"
        )
        frame[f"cross_direction_balance_{horizon}"] = (
            _rolling(frame, "vsti_up_cross", horizon, "sum")
            - _rolling(frame, "vsti_down_cross", horizon, "sum")
        ) / adjacent_count.replace(0.0, np.nan)
        frame[f"state_transition_activity_{horizon}"] = _rolling(
            frame, "vsti_state_change", horizon, "sum"
        ) / adjacent_count.replace(0.0, np.nan)
    return frame.drop(
        columns=[
            "vsti_score",
            "vsti_directional_event",
            "vsti_adjacent",
            "vsti_state_change",
            "vsti_open_position",
            "vsti_close_position",
            "vsti_up_cross",
            "vsti_down_cross",
        ]
    )
