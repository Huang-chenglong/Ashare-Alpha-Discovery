import numpy as np
import pandas as pd

from ashare_alpha.robust_v46 import CANDIDATES_V46, build_robust_candidates_v46


def test_v46_robust_formulas_are_frozen() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-31", "2023-02-28", "2023-03-31"]),
            "asset": ["000001"] * 3,
            "rrsm_rank_v44": [0.2, 0.8, 0.5],
            "tafs_rank_v44": [0.7, 0.4, 0.6],
        }
    )
    result = build_robust_candidates_v46(frame)
    assert tuple(CANDIDATES_V46) == (
        "pmedian3_v46",
        "pfloor3_v46",
        "pstable3_v46",
    )
    path = np.array([0.40, 0.64, 0.54])
    assert np.isclose(result.loc[2, "pmedian3_v46"], np.median(path))
    assert np.isclose(result.loc[2, "pfloor3_v46"], path.min())
    assert np.isclose(
        result.loc[2, "pstable3_v46"],
        path.mean() - 0.5 * (path.max() - path.min()),
    )
    assert result.loc[:1, list(CANDIDATES_V46)].isna().all().all()
