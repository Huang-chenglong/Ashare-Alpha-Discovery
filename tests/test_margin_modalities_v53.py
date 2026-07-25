from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_alpha.margin_modalities_v53 import _snapshot_flow_features_v53


def test_v53_snapshot_surprises_use_only_prior_four_snapshots() -> None:
    dates = pd.date_range("2021-01-01", periods=6, freq="7D")
    daily = pd.DataFrame(
        {
            "date": dates,
            "asset": ["000001"] * 6,
            "amount": [100.0] * 6,
            "raw_close": [10.0] * 6,
        }
    )
    margin = pd.DataFrame(
        {
            "trade_date": dates,
            "asset": ["000001"] * 6,
            "financing_balance": [1000] * 6,
            "financing_buy": [10, 10, 10, 10, 50, 10],
            "short_balance_quantity": [10, 11, 12, 13, 14, 15],
            "short_sell_quantity": [1, 1, 1, 1, 5, 1],
        }
    )
    features = _snapshot_flow_features_v53(daily, margin)
    assert features.loc[:3, "financing_buy_surprise_4"].isna().all()
    assert np.isclose(features.loc[4, "financing_buy_surprise_4"], 4.0)
    assert np.isclose(features.loc[4, "short_sell_surprise_4"], 4.0)
    assert np.isclose(features.loc[4, "short_inventory_path_4"], 0.4)
