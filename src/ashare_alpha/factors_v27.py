from __future__ import annotations

import pandas as pd

from .factors import Candidate
from .factors_v26 import COMMON_CONTROLS_V26, compute_candidates_v26


CANDIDATES_V27 = {
    "mfsic_equal_weight": Candidate(
        "multiscale_float_supply_impact_consensus",
        (
            "Availability-preserving equal-weight rank consensus across 100- and "
            "120-session float-supply impact-decay scores predicts higher returns."
        ),
    )
}
CANDIDATE_COLUMNS_V27 = list(CANDIDATES_V27)
EVENT_CONTROL_STEMS_V27 = [
    "mean_completed_supply_fraction",
    "mean_completed_residual_return",
    "mean_completed_turnover",
    "mean_raw_absorption_slope",
    "mean_completed_event_age",
]
CONTROL_COLUMNS_BY_CANDIDATE_V27 = {
    "mfsic_equal_weight": [
        *COMMON_CONTROLS_V26,
        "remaining_float_inventory_days_025_control",
        *(f"multiscale_{stem}" for stem in EVENT_CONTROL_STEMS_V27),
        "fsid_100_available",
    ]
}


def _eligible_rank(
    values: pd.Series,
    dates: pd.Series,
    eligible: pd.Series,
) -> pd.Series:
    return values.where(eligible).groupby(dates).rank(method="average", pct=True)


def compute_candidates_v27(daily: pd.DataFrame) -> pd.DataFrame:
    frame = compute_candidates_v26(daily).sort_values(["asset", "date"]).reset_index(
        drop=True
    )
    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    rank_100 = _eligible_rank(frame["fsid_100"], frame["date"], eligible)
    rank_120 = _eligible_rank(frame["fsid_120"], frame["date"], eligible)
    short_available = rank_100.notna() & rank_120.notna()
    frame["mfsic_equal_weight"] = (
        0.5 * rank_120 + 0.5 * rank_100.where(short_available, rank_120)
    ).where(rank_120.notna())
    frame["fsid_100_available"] = short_available.astype(float).where(rank_120.notna())
    for stem in EVENT_CONTROL_STEMS_V27:
        long_control = frame[f"{stem}_120"]
        short_control = frame[f"{stem}_100"]
        frame[f"multiscale_{stem}"] = (
            0.5 * long_control
            + 0.5 * short_control.where(short_available, long_control)
        ).where(rank_120.notna())
    return frame

