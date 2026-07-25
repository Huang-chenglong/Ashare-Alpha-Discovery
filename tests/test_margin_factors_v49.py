from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_alpha.margin_factors_v49 import (
    CANDIDATES_V49,
    build_persistent_margin_candidates_v49,
)


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2023-01-31", periods=4, freq="ME")
    return pd.DataFrame(
        {
            "date": dates,
            "asset": "000001",
            "mcar_v48": [0.1, 0.2, 0.3, 0.4],
            "financing_leverage_rank": [0.2, 0.3, 0.4, 0.5],
            "financing_inflow_rank": [0.3, 0.4, 0.5, 0.6],
            "negative_momentum_rank": [0.4, 0.5, 0.6, 0.7],
        }
    )


def test_v49_formulas_are_exact() -> None:
    result = build_persistent_margin_candidates_v49(_panel())
    row = result.iloc[2]
    assert np.isclose(row["mcar_ema2_v49"], 0.65 * 0.3 + 0.35 * 0.2)
    expected_ema3 = 0.50 * 0.3 + 0.30 * 0.2 + 0.20 * 0.1
    assert np.isclose(row["mcar_ema3_v49"], expected_ema3)
    assert np.isclose(row["mcar_stable3_v49"], expected_ema3 - 0.20 * 0.2)
    assert set(CANDIDATES_V49).issubset(result.columns)


def test_v49_requires_consecutive_months() -> None:
    panel = _panel().drop(index=1).reset_index(drop=True)
    result = build_persistent_margin_candidates_v49(panel)
    march = result[result["date"].eq(pd.Timestamp("2023-03-31"))].iloc[0]
    assert pd.isna(march["mcar_ema2_v49"])
    assert pd.isna(march["mcar_ema3_v49"])
