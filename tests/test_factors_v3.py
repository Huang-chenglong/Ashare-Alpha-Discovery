import numpy as np
import pandas as pd

from ashare_alpha.factors_v3 import compute_candidates_v3


def make_supply_daily(turnover_after_event: float) -> pd.DataFrame:
    days = 80
    float_shares = np.full(days, 1_000_000.0)
    float_shares[40:] = 1_200_000.0
    turnover = np.full(days, 0.01)
    turnover[40:] = turnover_after_event
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=days),
            "asset": "000001",
            "open": 10.0,
            "close": np.linspace(10.0, 11.0, days),
            "volume": turnover * float_shares,
            "turnover_fraction": turnover,
            "float_shares": float_shares,
            "float_market_cap": float_shares * 10.0,
            "is_member": True,
            "tradestatus": 1,
            "is_st": 0,
        }
    )


def test_more_turnover_absorbs_more_float_inventory() -> None:
    slow = compute_candidates_v3(make_supply_daily(0.005)).iloc[-1]
    fast = compute_candidates_v3(make_supply_daily(0.05)).iloc[-1]
    assert fast["remaining_float_inventory_fraction_025"] > slow[
        "remaining_float_inventory_fraction_025"
    ]


def test_inventory_is_nonpositive_and_finite_after_event() -> None:
    result = compute_candidates_v3(make_supply_daily(0.01)).iloc[-1]
    for suffix in ["025", "050", "100"]:
        value = result[f"remaining_float_inventory_days_{suffix}"]
        assert np.isfinite(value)
        assert value <= 0.0
