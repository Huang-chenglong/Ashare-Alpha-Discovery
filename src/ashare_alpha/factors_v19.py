from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import compute_baselines
from .factors import Candidate
from .factors_v8 import COMMON_CONTROLS_V8, _rolling_participation_dominance
from .factors_v10 import _joint_path_statistics
from .factors_v18 import _eligible_rank


CANDIDATES_V19 = {
    "mspc_equal_weight": Candidate(
        "multiscale_structural_participation_consensus",
        (
            "Equal-weight consensus across turnover-distribution dominance and 60/120-day "
            "joint path reversibility predicts higher next-month returns."
        ),
    )
}
CANDIDATE_COLUMNS_V19 = list(CANDIDATES_V19)
CONTROL_COLUMNS_BY_CANDIDATE_V19 = {
    "mspc_equal_weight": [
        *COMMON_CONTROLS_V8,
        "return_sign_balance_120",
        "signed_turnover_share_120",
        "linear_return_turnover_correlation_120",
        "joint_motif_entropy_60",
        "joint_return_sign_balance_60",
        "joint_activity_balance_60",
        "joint_linear_return_turnover_correlation_60",
        "joint_motif_entropy_120",
        "joint_return_sign_balance_120",
        "joint_activity_balance_120",
        "joint_linear_return_turnover_correlation_120",
    ]
}


def compute_candidates_v19(daily: pd.DataFrame) -> pd.DataFrame:
    frame = compute_baselines(daily).sort_values(["asset", "date"]).reset_index(drop=True)
    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    close = frame["close"].where(frame["close"].gt(0.0))
    log_return = np.log(close).groupby(frame["asset"], sort=False).diff()
    market = log_return.where(eligible).groupby(frame["date"]).transform("median")
    frame["mspc_excess_return"] = log_return - market
    frame["mspc_turnover"] = frame["turnover_fraction"].where(
        frame["turnover_fraction"].gt(0.0)
    )
    log_turnover = np.log(frame["mspc_turnover"])
    trailing_median = log_turnover.groupby(frame["asset"], sort=False).transform(
        lambda values: values.rolling(60, min_periods=30).median().shift(1)
    )
    frame["mspc_activity"] = log_turnover - trailing_median
    parts: list[pd.DataFrame] = []
    for _, rows in frame.groupby("asset", sort=False):
        rows = rows.copy()
        returns = rows["mspc_excess_return"].to_numpy(dtype=float)
        turnover = rows["mspc_turnover"].to_numpy(dtype=float)
        activity = rows["mspc_activity"].to_numpy(dtype=float)
        pds, sign_balance, signed_share, pds_correlation = (
            _rolling_participation_dominance(returns, turnover, 120)
        )
        rows["pds_120"] = pds
        rows["return_sign_balance_120"] = sign_balance
        rows["signed_turnover_share_120"] = signed_share
        rows["linear_return_turnover_correlation_120"] = pds_correlation
        for horizon in (60, 120):
            reversible, entropy, return_balance, activity_balance, correlation = (
                _joint_path_statistics(returns, activity, horizon)
            )
            rows[f"joint_path_reversibility_{horizon}"] = reversible
            rows[f"joint_motif_entropy_{horizon}"] = entropy
            rows[f"joint_return_sign_balance_{horizon}"] = return_balance
            rows[f"joint_activity_balance_{horizon}"] = activity_balance
            rows[f"joint_linear_return_turnover_correlation_{horizon}"] = correlation
        parts.append(rows)
    frame = pd.concat(parts, ignore_index=True).sort_values(["asset", "date"])
    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    ranks = [
        _eligible_rank(frame[column], frame["date"], eligible)
        for column in [
            "pds_120", "joint_path_reversibility_60", "joint_path_reversibility_120"
        ]
    ]
    frame["mspc_equal_weight"] = sum(ranks) / 3.0
    return frame.drop(columns=["mspc_excess_return", "mspc_turnover", "mspc_activity"])
