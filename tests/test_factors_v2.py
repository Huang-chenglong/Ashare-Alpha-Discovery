import numpy as np
import pandas as pd

from ashare_alpha.factors_v2 import compute_candidates_v2


def make_gap_daily(days: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=days)
    frames = []
    for asset, open_price in [("000001", 10.2), ("000002", 10.0), ("000003", 9.9)]:
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "asset": asset,
                    "open": open_price,
                    "high": max(open_price, 10.0) + 0.1,
                    "low": min(open_price, 10.0) - 0.1,
                    "close": 10.0,
                    "volume": 10_000.0,
                    "turnover_fraction": np.linspace(0.01, 0.03, days),
                    "float_shares": 1_000_000.0,
                    "float_market_cap": 10_000_000.0,
                    "is_member": True,
                    "tradestatus": 1,
                    "is_st": 0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_positive_gap_reversal_is_bounded_fraction() -> None:
    result = compute_candidates_v2(make_gap_daily())
    value = float(
        result[result["asset"].eq("000001")].iloc[-1][
            "active_positive_gap_reversal_20"
        ]
    )
    assert 0.0 <= value <= 1.0


def test_burst_close_strength_exists_after_warmup() -> None:
    daily = make_gap_daily()
    target = daily["asset"].eq("000001") & daily["date"].eq(daily["date"].max())
    daily.loc[target, ["close", "turnover_fraction"]] = [10.19, 0.30]
    result = compute_candidates_v2(daily)
    value = result[result["asset"].eq("000001")].iloc[-1]["burst_close_strength_20"]
    assert np.isfinite(value)
