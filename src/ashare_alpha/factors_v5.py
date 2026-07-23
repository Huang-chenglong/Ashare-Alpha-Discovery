from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import compute_baselines
from .factors import Candidate
from .statistics import mad_winsorize, zscore


MAIN_EFFECTS_V5 = [
    "anti_max_return_20",
    "low_turnover_20",
    "negative_intraday_mean_20",
    "low_volatility_60",
    "reversal_20",
]
FACTOR_SETS_V5 = {
    "joint_lottery_liquidity": (
        "anti_max_return_20",
        "low_turnover_20",
    ),
    "joint_lottery_intraday": (
        "anti_max_return_20",
        "negative_intraday_mean_20",
    ),
    "joint_liquidity_intraday": (
        "low_turnover_20",
        "negative_intraday_mean_20",
    ),
    "joint_lottery_liquidity_intraday": (
        "anti_max_return_20",
        "low_turnover_20",
        "negative_intraday_mean_20",
    ),
    "joint_defensive_fourway": (
        "anti_max_return_20",
        "low_turnover_20",
        "negative_intraday_mean_20",
        "low_volatility_60",
    ),
    "joint_reversal_fourway": (
        "anti_max_return_20",
        "low_turnover_20",
        "negative_intraday_mean_20",
        "reversal_20",
    ),
}
CANDIDATES_V5 = {
    name: Candidate(
        "nonlinear_defensive_bottleneck",
        "A soft-min joint state contains incremental prediction beyond all constituent main effects.",
    )
    for name in FACTOR_SETS_V5
}
CANDIDATE_COLUMNS_V5 = list(CANDIDATES_V5)
CONTROL_COLUMNS_V5 = MAIN_EFFECTS_V5


def _eligible_cross_section_zscore(
    frame: pd.DataFrame,
    column: str,
    eligible: pd.Series,
) -> pd.Series:
    values = frame[column].where(eligible)
    def transform(rows: pd.Series) -> pd.Series:
        if rows.notna().sum() < 2:
            return rows
        return zscore(mad_winsorize(rows)).clip(-5.0, 5.0)

    return values.groupby(frame["date"]).transform(transform)


def compute_candidates_v5(daily: pd.DataFrame) -> pd.DataFrame:
    frame = compute_baselines(daily).reset_index(drop=True)
    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    transformed: dict[str, pd.Series] = {
        column: _eligible_cross_section_zscore(frame, column, eligible)
        for column in MAIN_EFFECTS_V5
    }
    for candidate, constituents in FACTOR_SETS_V5.items():
        matrix = np.column_stack(
            [transformed[column].to_numpy(dtype=float) for column in constituents]
        )
        # Smooth minimum: a high score requires every constituent to be high. Dividing
        # by the constituent count keeps the scale comparable across arities.
        frame[candidate] = -np.log(np.exp(-matrix).mean(axis=1))
    return frame
