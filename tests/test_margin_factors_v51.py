from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_alpha.margin_factors_v51 import build_persistent_path_candidates_v51


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2023-01-31", periods=4, freq="ME")
    return pd.DataFrame(
        {
            "date": dates,
            "asset": "000001",
            "mpbr_v50": [0.1, 0.2, 0.3, 0.4],
            "financing_leverage_rank": [0.2, 0.3, 0.4, 0.5],
            "financing_inflow_rank": [0.3, 0.4, 0.5, 0.6],
            "negative_momentum_rank": [0.4, 0.5, 0.6, 0.7],
            "path_inflow_breadth_rank": [0.5, 0.6, 0.7, 0.8],
        }
    )


def test_v51_formulas_are_exact() -> None:
    result = build_persistent_path_candidates_v51(_panel())
    row = result.iloc[2]
    assert np.isclose(row["mpbr_ema2_v51"], 0.65 * 0.3 + 0.35 * 0.2)
    ema3 = 0.50 * 0.3 + 0.30 * 0.2 + 0.20 * 0.1
    assert np.isclose(row["mpbr_ema3_v51"], ema3)
    assert np.isclose(row["mpbr_stable3_v51"], ema3 - 0.20 * 0.2)


def test_v51_breaks_lags_across_missing_month() -> None:
    result = build_persistent_path_candidates_v51(
        _panel().drop(index=1).reset_index(drop=True)
    )
    march = result[result["date"].eq(pd.Timestamp("2023-03-31"))].iloc[0]
    assert pd.isna(march["mpbr_ema2_v51"])
    assert pd.isna(march["mpbr_ema3_v51"])
