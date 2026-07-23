from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import Candidate, _rolling
from .factors_v4 import compute_candidates_v4


BASELINES = {
    "low_turnover_20": Candidate("known_baseline", "Known low-turnover benchmark."),
    "low_turnover_vol_60": Candidate("known_baseline", "Known turnover-stability benchmark."),
    "reversal_20": Candidate("known_baseline", "Known short-term reversal benchmark."),
    "low_volatility_60": Candidate("known_baseline", "Known low-volatility benchmark."),
    "anti_max_return_20": Candidate("known_baseline", "Known anti-MAX lottery benchmark."),
    "anti_max_return_60": Candidate("known_baseline", "Known quarterly anti-MAX benchmark."),
    "anti_skewness_20": Candidate("known_baseline", "Known low-skewness benchmark."),
    "anti_skewness_60": Candidate("known_baseline", "Known quarterly low-skewness benchmark."),
    "momentum_60_20": Candidate("known_baseline", "Intermediate 60-to-20-day momentum benchmark."),
    "momentum_120_20": Candidate("known_baseline", "Intermediate 120-to-20-day momentum benchmark."),
    "overnight_mean_20": Candidate("known_baseline", "Known overnight-return component benchmark."),
    "overnight_mean_60": Candidate("known_baseline", "Quarterly overnight-return component benchmark."),
    "negative_intraday_mean_20": Candidate("known_baseline", "Known negative intraday-return component benchmark."),
    "negative_intraday_mean_60": Candidate("known_baseline", "Quarterly negative intraday-return component benchmark."),
}
BASELINE_COLUMNS = list(BASELINES)


def compute_baselines(daily: pd.DataFrame) -> pd.DataFrame:
    frame = compute_candidates_v4(daily).reset_index(drop=True)
    frame["anti_max_return_20"] = -frame["max_excess_return_20"]
    frame["anti_max_return_60"] = -frame["max_excess_return_60"]
    frame["anti_skewness_20"] = -frame["realized_skewness_20"]
    frame["anti_skewness_60"] = -frame["realized_skewness_60"]
    grouped_close = frame.groupby("asset", sort=False)["close"]
    frame["momentum_60_20"] = grouped_close.shift(20) / grouped_close.shift(60) - 1.0
    frame["momentum_120_20"] = grouped_close.shift(20) / grouped_close.shift(120) - 1.0

    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    previous_close = grouped_close.shift(1)
    frame["overnight_component"] = np.log(
        frame["open"].where(frame["open"].gt(0.0)) / previous_close
    )
    frame["intraday_component"] = np.log(
        frame["close"].where(frame["close"].gt(0.0))
        / frame["open"].where(frame["open"].gt(0.0))
    )
    for column in ["overnight_component", "intraday_component"]:
        market = frame[column].where(eligible).groupby(frame["date"]).transform("median")
        frame[column] = frame[column] - market
    for horizon in (20, 60):
        frame[f"overnight_mean_{horizon}"] = _rolling(
            frame, "overnight_component", horizon, "mean"
        )
        frame[f"negative_intraday_mean_{horizon}"] = -_rolling(
            frame, "intraday_component", horizon, "mean"
        )
    return frame.drop(columns=["overnight_component", "intraday_component"])
