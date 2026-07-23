from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import compute_baselines
from .factors import Candidate
from .factors_v3 import _remaining_inventory


HORIZONS_V26 = (100, 120)
SUPPLY_EVENT_THRESHOLD_V26 = 0.001
EARLY_WINDOW_V26 = (1, 5)
LATE_WINDOW_V26 = (6, 20)
MINIMUM_EARLY_OBSERVATIONS_V26 = 4
MINIMUM_LATE_OBSERVATIONS_V26 = 12
CANDIDATES_V26 = {
    f"fsid_{horizon}": Candidate(
        "float_supply_impact_decay",
        (
            "Improving residual price impact per unit turnover from early to late stages "
            f"after completed float-supply events over {horizon} sessions predicts higher returns."
        ),
    )
    for horizon in HORIZONS_V26
}
CANDIDATE_COLUMNS_V26 = list(CANDIDATES_V26)
COMMON_CONTROLS_V26 = [
    "low_turnover_20", "low_turnover_vol_60", "reversal_20", "low_volatility_60",
    "anti_max_return_20", "anti_skewness_60", "momentum_60_20", "momentum_120_20",
    "overnight_mean_20", "negative_intraday_mean_20", "diffuse_turnover_60",
]
CONTROL_COLUMNS_BY_CANDIDATE_V26 = {
    f"fsid_{horizon}": [
        *COMMON_CONTROLS_V26,
        "remaining_float_inventory_days_025_control",
        f"mean_completed_supply_fraction_{horizon}",
        f"mean_completed_residual_return_{horizon}",
        f"mean_completed_turnover_{horizon}",
        f"mean_raw_absorption_slope_{horizon}",
        f"mean_completed_event_age_{horizon}",
    ]
    for horizon in HORIZONS_V26
}


def _rolling(frame: pd.DataFrame, column: str, horizon: int, method: str) -> pd.Series:
    roller = frame.groupby("asset", sort=False)[column].rolling(
        horizon, min_periods=horizon
    )
    return getattr(roller, method)().reset_index(level=0, drop=True)


def _completed_supply_events(
    rows: pd.DataFrame,
    residual: np.ndarray,
    turnover: np.ndarray,
    supply_fraction: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, np.ndarray]:
    count = len(rows)
    output = {
        "count": np.zeros(count),
        "weight": np.zeros(count),
        "weighted_score": np.zeros(count),
        "weighted_supply": np.zeros(count),
        "weighted_return": np.zeros(count),
        "weighted_turnover": np.zeros(count),
        "weighted_raw_slope": np.zeros(count),
        "weighted_completion_position": np.zeros(count),
    }
    for event in np.flatnonzero(supply_fraction > SUPPLY_EVENT_THRESHOLD_V26):
        completion = event + LATE_WINDOW_V26[1]
        if completion >= count:
            continue
        early_slice = slice(event + EARLY_WINDOW_V26[0], event + EARLY_WINDOW_V26[1] + 1)
        late_slice = slice(event + LATE_WINDOW_V26[0], event + LATE_WINDOW_V26[1] + 1)
        early_valid = (
            eligible[early_slice]
            & np.isfinite(residual[early_slice])
            & np.isfinite(turnover[early_slice])
            & (turnover[early_slice] > 0.0)
        )
        late_valid = (
            eligible[late_slice]
            & np.isfinite(residual[late_slice])
            & np.isfinite(turnover[late_slice])
            & (turnover[late_slice] > 0.0)
        )
        if (
            int(early_valid.sum()) < MINIMUM_EARLY_OBSERVATIONS_V26
            or int(late_valid.sum()) < MINIMUM_LATE_OBSERVATIONS_V26
        ):
            continue
        early_return = float(residual[early_slice][early_valid].sum())
        late_return = float(residual[late_slice][late_valid].sum())
        early_turnover = float(turnover[early_slice][early_valid].sum())
        late_turnover = float(turnover[late_slice][late_valid].sum())
        if early_turnover <= 0.0 or late_turnover <= 0.0:
            continue
        score = np.clip(
            late_return / late_turnover - early_return / early_turnover,
            -5.0,
            5.0,
        )
        raw_slope = late_return / int(late_valid.sum()) - early_return / int(
            early_valid.sum()
        )
        event_supply = float(supply_fraction[event])
        weight = float(np.clip(event_supply, 0.001, 0.50))
        output["count"][completion] = 1.0
        output["weight"][completion] = weight
        output["weighted_score"][completion] = weight * score
        output["weighted_supply"][completion] = weight * event_supply
        output["weighted_return"][completion] = weight * (early_return + late_return)
        output["weighted_turnover"][completion] = weight * (early_turnover + late_turnover)
        output["weighted_raw_slope"][completion] = weight * raw_slope
        output["weighted_completion_position"][completion] = weight * completion
    return output


def compute_candidates_v26(daily: pd.DataFrame) -> pd.DataFrame:
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
    turnover = frame["turnover_fraction"].where(frame["turnover_fraction"].gt(0.0))
    previous_float = frame.groupby("asset", sort=False)["float_shares"].shift(1)
    supply_fraction = (
        (frame["float_shares"] - previous_float).clip(lower=0.0)
        / previous_float.where(previous_float.gt(0.0))
    )

    inventory = _remaining_inventory(frame, supply_fraction, turnover, 0.25)
    normal_turnover = frame.groupby("asset", sort=False)["turnover_fraction"].transform(
        lambda values: values.where(values.gt(0.0)).rolling(20, min_periods=10).mean()
    )
    frame["remaining_float_inventory_days_025_control"] = -np.log1p(
        inventory / normal_turnover.replace(0.0, np.nan)
    )
    event_columns = [
        "count", "weight", "weighted_score", "weighted_supply", "weighted_return",
        "weighted_turnover", "weighted_raw_slope", "weighted_completion_position",
    ]
    for column in event_columns:
        frame[f"fsid_{column}"] = 0.0
    frame["fsid_asset_position"] = frame.groupby("asset", sort=False).cumcount().astype(float)

    for _, indices in frame.groupby("asset", sort=False).indices.items():
        positions = np.asarray(indices)
        completed = _completed_supply_events(
            frame.iloc[positions],
            residual.iloc[positions].to_numpy(dtype=float),
            turnover.iloc[positions].to_numpy(dtype=float),
            supply_fraction.iloc[positions].fillna(0.0).to_numpy(dtype=float),
            eligible.iloc[positions].to_numpy(dtype=bool),
        )
        for column, values in completed.items():
            frame.loc[positions, f"fsid_{column}"] = values

    for horizon in HORIZONS_V26:
        event_count = _rolling(frame, "fsid_count", horizon, "sum")
        weight_sum = _rolling(frame, "fsid_weight", horizon, "sum")
        enough = event_count.ge(1.0) & weight_sum.gt(0.0)
        frame[f"fsid_{horizon}"] = (
            _rolling(frame, "fsid_weighted_score", horizon, "sum")
            / weight_sum.replace(0.0, np.nan)
        ).where(enough)
        frame[f"mean_completed_supply_fraction_{horizon}"] = (
            _rolling(frame, "fsid_weighted_supply", horizon, "sum")
            / weight_sum.replace(0.0, np.nan)
        ).where(enough)
        frame[f"mean_completed_residual_return_{horizon}"] = (
            _rolling(frame, "fsid_weighted_return", horizon, "sum")
            / weight_sum.replace(0.0, np.nan)
        ).where(enough)
        frame[f"mean_completed_turnover_{horizon}"] = (
            _rolling(frame, "fsid_weighted_turnover", horizon, "sum")
            / weight_sum.replace(0.0, np.nan)
        ).where(enough)
        frame[f"mean_raw_absorption_slope_{horizon}"] = (
            _rolling(frame, "fsid_weighted_raw_slope", horizon, "sum")
            / weight_sum.replace(0.0, np.nan)
        ).where(enough)
        mean_completion_position = (
            _rolling(frame, "fsid_weighted_completion_position", horizon, "sum")
            / weight_sum.replace(0.0, np.nan)
        )
        frame[f"mean_completed_event_age_{horizon}"] = (
            frame["fsid_asset_position"] - mean_completion_position
        ).where(enough)

    return frame.drop(columns=[
        *(f"fsid_{column}" for column in event_columns),
        "fsid_asset_position",
    ])

