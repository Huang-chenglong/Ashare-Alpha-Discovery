from __future__ import annotations

import pandas as pd

from .factors import Candidate
from .factors_v10 import COMMON_CONTROLS_V10, compute_candidates_v10


CALIBRATED_HORIZON_V11 = 90
CANDIDATES_V11 = {
    "joint_path_reversibility_90": Candidate(
        "return_participation_time_reversibility",
        "The pre-frozen midpoint 90-session joint-path reversibility score predicts higher next-month returns.",
    )
}
CANDIDATE_COLUMNS_V11 = list(CANDIDATES_V11)
CONTROL_COLUMNS_BY_CANDIDATE_V11 = {
    "joint_path_reversibility_90": [
        *COMMON_CONTROLS_V10,
        "joint_motif_entropy_90",
        "joint_return_sign_balance_90",
        "joint_activity_balance_90",
        "joint_linear_return_turnover_correlation_90",
    ]
}


def compute_candidates_v11(daily: pd.DataFrame) -> pd.DataFrame:
    return compute_candidates_v10(daily, horizons=(CALIBRATED_HORIZON_V11,))
