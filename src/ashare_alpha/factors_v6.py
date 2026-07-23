from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import compute_baselines
from .factors import Candidate, _rolling
from .factors_v3 import _remaining_inventory
from .statistics import mad_winsorize, zscore


CORE_COMPONENTS_V6 = [
    "anti_max_return_20",
    "low_turnover_20",
    "negative_intraday_mean_20",
    "low_volatility_60",
    "reversal_20",
]
SUPPLY_COMPONENT_V6 = "remaining_float_inventory_days_025_v6"
SUPPLY_PENALTY_WEIGHT = 0.25
CANDIDATES_V6 = {
    "supply_aware_defensive_attention": Candidate(
        "engineered_composite",
        "An equal-weight defensive-attention composite with a one-sided dynamic float-supply penalty predicts higher returns.",
    )
}
CANDIDATE_COLUMNS_V6 = list(CANDIDATES_V6)


def _cross_section_zscore(
    frame: pd.DataFrame, column: str, eligible: pd.Series
) -> pd.Series:
    values = frame[column].where(eligible)

    def transform(rows: pd.Series) -> pd.Series:
        if rows.notna().sum() < 2:
            return rows
        return zscore(mad_winsorize(rows)).clip(-5.0, 5.0)

    return values.groupby(frame["date"]).transform(transform)


def compute_candidates_v6(daily: pd.DataFrame) -> pd.DataFrame:
    frame = compute_baselines(daily).reset_index(drop=True)
    previous_float = frame.groupby("asset", sort=False)["float_shares"].shift(1)
    new_float_fraction = (
        (frame["float_shares"] - previous_float).clip(lower=0.0)
        / frame["float_shares"].where(frame["float_shares"].gt(0.0))
    )
    usable_turnover = frame["turnover_fraction"].where(
        frame["turnover_fraction"].gt(0.0)
    )
    frame["usable_turnover_v6"] = usable_turnover
    inventory = _remaining_inventory(
        frame, new_float_fraction, usable_turnover, absorption_scale=0.25
    )
    normal_turnover = _rolling(frame, "usable_turnover_v6", 20, "mean")
    frame[SUPPLY_COMPONENT_V6] = -np.log1p(
        inventory / normal_turnover.replace(0.0, np.nan)
    )

    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    component_z = [
        _cross_section_zscore(frame, column, eligible)
        for column in CORE_COMPONENTS_V6
    ]
    supply_z = _cross_section_zscore(frame, SUPPLY_COMPONENT_V6, eligible)
    core_score = pd.concat(component_z, axis=1).mean(axis=1, skipna=False)
    supply_penalty = (-supply_z).clip(lower=0.0)
    frame["supply_aware_defensive_attention"] = (
        core_score - SUPPLY_PENALTY_WEIGHT * supply_penalty
    )
    return frame.drop(columns=["usable_turnover_v6"])
