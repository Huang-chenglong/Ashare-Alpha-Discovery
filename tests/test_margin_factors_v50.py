from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_alpha.margin_factors_v50 import _snapshot_path_features


def test_snapshot_path_uses_exact_interval_amount_and_four_observations() -> None:
    dates = pd.bdate_range("2024-01-01", periods=25)
    daily = pd.DataFrame(
        {
            "date": dates,
            "asset": "000001",
            "amount": 100.0,
        }
    )
    snapshot_dates = dates[[4, 9, 14, 19, 24]]
    margin = pd.DataFrame(
        {
            "trade_date": snapshot_dates,
            "asset": "000001",
            "financing_balance": [1000.0, 1050.0, 1100.0, 1150.0, 1200.0],
        }
    )
    result = _snapshot_path_features(daily, margin)
    last = result.iloc[-1]
    assert np.isclose(last["path_inflow_breadth"], 1.0)
    assert np.isclose(last["path_signed_ratio"], 1.0)
    assert np.isclose(last["path_dispersion_ratio"], 0.0)


def test_snapshot_gap_invalidates_four_observation_path() -> None:
    dates = pd.bdate_range("2024-01-01", periods=25)
    daily = pd.DataFrame({"date": dates, "asset": "000001", "amount": 100.0})
    global_dates = dates[[4, 9, 14, 19, 24]]
    margin = pd.DataFrame(
        {
            "trade_date": global_dates.delete(2),
            "asset": "000001",
            "financing_balance": [1000.0, 1050.0, 1150.0, 1200.0],
        }
    )
    filler = pd.DataFrame(
        {
            "trade_date": global_dates,
            "asset": "600000",
            "financing_balance": [1, 2, 3, 4, 5],
        }
    )
    result = _snapshot_path_features(daily, pd.concat([margin, filler]))
    last = result[result["asset"].eq("000001")].iloc[-1]
    assert pd.isna(last["path_inflow_breadth"])
