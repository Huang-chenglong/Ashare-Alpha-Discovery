import numpy as np
import pandas as pd

from ashare_alpha.factors import compute_candidates


def make_daily(turnover: list[float]) -> pd.DataFrame:
    count = len(turnover)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=count),
            "asset": "000001",
            "open": 10.0,
            "close": np.linspace(10.0, 11.0, count),
            "volume": np.asarray(turnover) * 1_000_000.0,
            "turnover_fraction": turnover,
            "float_shares": 1_000_000.0,
            "float_market_cap": 10_000_000.0,
            "is_member": True,
            "tradestatus": 1,
            "is_st": 0,
        }
    )


def test_diffuse_turnover_is_zero_for_uniform_participation() -> None:
    result = compute_candidates(make_daily([0.01] * 60))
    assert abs(float(result.iloc[-1]["diffuse_turnover_20"])) < 1e-12
    assert abs(float(result.iloc[-1]["diffuse_turnover_60"])) < 1e-12


def test_diffuse_turnover_penalizes_a_concentrated_burst() -> None:
    result = compute_candidates(make_daily([0.01] * 59 + [0.20]))
    assert float(result.iloc[-1]["diffuse_turnover_20"]) < -1.0
    assert float(result.iloc[-1]["diffuse_turnover_60"]) < 0.0
