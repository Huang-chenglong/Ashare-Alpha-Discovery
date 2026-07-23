from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Candidate:
    family: str
    hypothesis: str


CANDIDATES = {
    "diffuse_turnover_20": Candidate(
        "participation_concentration", "Diffuse monthly participation avoids episodic attention."
    ),
    "diffuse_turnover_60": Candidate(
        "participation_concentration", "Diffuse quarterly participation avoids episodic attention."
    ),
    "turnover_burst_reversal_20": Candidate(
        "burst_direction", "The return concentrated in monthly turnover bursts reverses."
    ),
    "turnover_burst_reversal_60": Candidate(
        "burst_direction", "The return concentrated in quarterly turnover bursts reverses."
    ),
    "attention_normalization_20_60": Candidate(
        "participation_change", "Recent participation normalization follows a temporary attention burst."
    ),
    "float_supply_days_120": Candidate(
        "float_supply", "New float that requires more normal-volume days to absorb creates an overhang."
    ),
}
CANDIDATE_COLUMNS = list(CANDIDATES)
CONTROL_COLUMNS = [
    "low_turnover_20",
    "low_turnover_vol_60",
    "reversal_20",
    "low_volatility_60",
]


def _rolling(frame: pd.DataFrame, column: str, window: int, method: str) -> pd.Series:
    grouped = frame.groupby("asset", sort=False)[column]
    roller = grouped.rolling(window, min_periods=max(10, window // 2))
    return getattr(roller, method)().reset_index(level=0, drop=True)


def compute_candidates(daily: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date", "asset", "open", "close", "volume", "turnover_fraction",
        "float_shares", "float_market_cap", "is_member", "tradestatus", "is_st",
    }
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Daily panel is missing {sorted(missing)}")

    frame = daily.sort_values(["asset", "date"]).copy()
    frame["asset"] = frame["asset"].astype("string").str.zfill(6)
    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    positive_close = frame["close"].where(frame["close"].gt(0.0))
    frame["log_return"] = np.log(positive_close).groupby(frame["asset"], sort=False).diff()
    market_median = frame["log_return"].where(eligible).groupby(frame["date"]).transform("median")
    frame["excess_log_return"] = frame["log_return"] - market_median
    frame["usable_turnover"] = frame["turnover_fraction"].where(
        frame["turnover_fraction"].gt(0.0)
    )
    frame["turnover_squared"] = frame["usable_turnover"].pow(2)
    frame["turnover_weighted_return"] = frame["turnover_squared"] * frame["excess_log_return"]

    hhi: dict[int, pd.Series] = {}
    for horizon in (20, 60):
        turnover_sum = _rolling(frame, "usable_turnover", horizon, "sum")
        squared_sum = _rolling(frame, "turnover_squared", horizon, "sum")
        hhi[horizon] = squared_sum / turnover_sum.pow(2).replace(0.0, np.nan)
        frame[f"diffuse_turnover_{horizon}"] = -np.log(
            (horizon * hhi[horizon]).where(hhi[horizon].gt(0.0))
        )
        burst_return = (
            _rolling(frame, "turnover_weighted_return", horizon, "sum")
            / squared_sum.replace(0.0, np.nan)
        )
        ordinary_return = _rolling(frame, "excess_log_return", horizon, "mean")
        frame[f"turnover_burst_reversal_{horizon}"] = -(burst_return - ordinary_return)

    frame["attention_normalization_20_60"] = (
        np.log((60.0 * hhi[60]).where(hhi[60].gt(0.0)))
        - np.log((20.0 * hhi[20]).where(hhi[20].gt(0.0)))
    )
    previous_float = frame.groupby("asset", sort=False)["float_shares"].shift(120)
    new_float = (frame["float_shares"] - previous_float).clip(lower=0.0)
    average_volume = _rolling(frame, "volume", 20, "mean")
    frame["float_supply_days_120"] = -new_float / average_volume.replace(0.0, np.nan)

    turnover_20 = _rolling(frame, "usable_turnover", 20, "mean")
    frame["low_turnover_20"] = -np.log(turnover_20.where(turnover_20.gt(0.0)))
    frame["log_turnover"] = np.log(frame["usable_turnover"])
    frame["low_turnover_vol_60"] = -_rolling(frame, "log_turnover", 60, "std")
    frame["reversal_20"] = -frame.groupby("asset", sort=False)["close"].pct_change(
        20, fill_method=None
    )
    frame["low_volatility_60"] = -_rolling(frame, "log_return", 60, "std")
    return frame.drop(
        columns=[
            "log_return", "excess_log_return", "usable_turnover", "turnover_squared",
            "turnover_weighted_return", "log_turnover",
        ]
    )
