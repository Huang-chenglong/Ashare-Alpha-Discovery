from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import compute_baselines
from .factors import Candidate
from .factors_v8 import COMMON_CONTROLS_V8, _rolling_participation_dominance
from .factors_v10 import _joint_path_statistics


CANDIDATES_V18 = {
    "spic_equal_weight": Candidate(
        "structural_participation_irreversibility_composite",
        (
            "An equal-weight rank synthesis of participation distribution dominance and "
            "joint return-participation path reversibility predicts higher returns."
        ),
    )
}
CANDIDATE_COLUMNS_V18 = list(CANDIDATES_V18)
CONTROL_COLUMNS_BY_CANDIDATE_V18 = {
    "spic_equal_weight": [
        *COMMON_CONTROLS_V8,
        "return_sign_balance_120",
        "signed_turnover_share_120",
        "linear_return_turnover_correlation_120",
        "joint_motif_entropy_60",
        "joint_return_sign_balance_60",
        "joint_activity_balance_60",
        "joint_linear_return_turnover_correlation_60",
    ]
}


def _eligible_rank(values: pd.Series, dates: pd.Series, eligible: pd.Series) -> pd.Series:
    ranked = values.where(eligible).groupby(dates).rank(method="average", pct=True)
    return ranked.where(eligible)


def compute_candidates_v18(daily: pd.DataFrame) -> pd.DataFrame:
    frame = compute_baselines(daily).sort_values(["asset", "date"]).reset_index(drop=True)
    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    close = frame["close"].where(frame["close"].gt(0.0))
    log_return = np.log(close).groupby(frame["asset"], sort=False).diff()
    market = log_return.where(eligible).groupby(frame["date"]).transform("median")
    frame["spic_excess_return"] = log_return - market
    frame["spic_turnover"] = frame["turnover_fraction"].where(
        frame["turnover_fraction"].gt(0.0)
    )
    log_turnover = np.log(frame["spic_turnover"])
    trailing_median = log_turnover.groupby(frame["asset"], sort=False).transform(
        lambda values: values.rolling(60, min_periods=30).median().shift(1)
    )
    frame["spic_activity"] = log_turnover - trailing_median
    parts: list[pd.DataFrame] = []
    for _, rows in frame.groupby("asset", sort=False):
        rows = rows.copy()
        returns = rows["spic_excess_return"].to_numpy(dtype=float)
        turnover = rows["spic_turnover"].to_numpy(dtype=float)
        activity = rows["spic_activity"].to_numpy(dtype=float)
        pds, sign_balance, signed_share, pds_correlation = (
            _rolling_participation_dominance(returns, turnover, 120)
        )
        reversible, entropy, return_balance, activity_balance, joint_correlation = (
            _joint_path_statistics(returns, activity, 60)
        )
        rows["pds_120"] = pds
        rows["return_sign_balance_120"] = sign_balance
        rows["signed_turnover_share_120"] = signed_share
        rows["linear_return_turnover_correlation_120"] = pds_correlation
        rows["joint_path_reversibility_60"] = reversible
        rows["joint_motif_entropy_60"] = entropy
        rows["joint_return_sign_balance_60"] = return_balance
        rows["joint_activity_balance_60"] = activity_balance
        rows["joint_linear_return_turnover_correlation_60"] = joint_correlation
        parts.append(rows)
    frame = pd.concat(parts, ignore_index=True).sort_values(["asset", "date"])
    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    pds_rank = _eligible_rank(frame["pds_120"], frame["date"], eligible)
    joint_rank = _eligible_rank(
        frame["joint_path_reversibility_60"], frame["date"], eligible
    )
    frame["spic_equal_weight"] = 0.5 * pds_rank + 0.5 * joint_rank
    return frame.drop(columns=["spic_excess_return", "spic_turnover", "spic_activity"])
