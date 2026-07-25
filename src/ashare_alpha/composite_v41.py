from __future__ import annotations

import pandas as pd


CANDIDATE_V41 = "scfc_v41"
COMPONENT_COLUMNS_V41 = ["nsim_v30_score", "cfma_v40_score"]
COMPONENT_WEIGHTS_V41 = {
    "nsim_v30_score": 0.5,
    "cfma_v40_score": 0.5,
}


def combine_component_scores_v41(frame: pd.DataFrame) -> pd.Series:
    missing = set(COMPONENT_COLUMNS_V41).difference(frame.columns)
    if missing:
        raise ValueError(f"V41 component frame is missing {sorted(missing)}")
    score = pd.Series(0.0, index=frame.index, dtype=float)
    for column in COMPONENT_COLUMNS_V41:
        rank = frame[column].groupby(frame["date"]).rank(
            method="average", pct=True
        )
        score = score.add(COMPONENT_WEIGHTS_V41[column] * rank)
    return score
