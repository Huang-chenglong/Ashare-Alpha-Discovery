from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_alpha.strategy_v54 import buffered_long_short_v54


def test_v54_long_short_includes_trading_and_borrow_costs() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31"] * 6),
            "asset": [f"{value:06d}" for value in range(6)],
            "score": np.arange(6, dtype=float),
            "label": np.arange(6, dtype=float),
            "future_return_20": [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03],
            "candidate": ["test"] * 6,
        }
    )
    monthly = buffered_long_short_v54(
        panel,
        holdings_each_side=2,
        annual_short_borrow_bps=1200.0,
    )
    row = monthly.iloc[0]
    assert np.isclose(row["gross_spread_return"], 0.05)
    assert np.isclose(row["traded_notional"], 2.0)
    assert np.isclose(row["trading_cost"], 0.004)
    assert np.isclose(row["borrow_cost"], 0.01)
    assert np.isclose(row["net_spread_return"], 0.036)
