import numpy as np
import pandas as pd

from ashare_alpha.baselines import compute_baselines


def test_intermediate_momentum_excludes_recent_twenty_days() -> None:
    days = 140
    close = np.linspace(10.0, 20.0, days)
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=days),
            "asset": "000001",
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 10_000.0,
            "turnover_fraction": 0.01,
            "float_shares": 1_000_000.0,
            "float_market_cap": close * 1_000_000.0,
            "is_member": True,
            "tradestatus": 1,
            "is_st": 0,
        }
    )
    result = compute_baselines(frame).iloc[-1]
    expected = close[-21] / close[-61] - 1.0
    assert abs(result["momentum_60_20"] - expected) < 1e-12
