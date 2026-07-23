import numpy as np
import pandas as pd

from ashare_alpha.factors_v4 import compute_candidates_v4


def make_lottery_daily(concentrated: bool) -> pd.DataFrame:
    days = 80
    returns = np.full(days, 0.002)
    if concentrated:
        returns[-20:] = 0.0
        returns[-1] = 0.04
    close = 10.0 * np.exp(np.cumsum(returns))
    target = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=days),
            "asset": "000001",
            "open": close,
            "close": close,
            "volume": 10_000.0,
            "turnover_fraction": np.linspace(0.01, 0.02, days),
            "float_shares": 1_000_000.0,
            "float_market_cap": close * 1_000_000.0,
            "is_member": True,
            "tradestatus": 1,
            "is_st": 0,
        }
    )
    controls = []
    for asset in ["000002", "000003"]:
        controls.append(
            target.assign(
                asset=asset,
                open=10.0,
                close=10.0,
                float_market_cap=10_000_000.0,
            )
        )
    return pd.concat([target, *controls], ignore_index=True)


def test_diffuse_upside_variance_penalizes_concentrated_jump() -> None:
    diffuse_frame = compute_candidates_v4(make_lottery_daily(False))
    concentrated_frame = compute_candidates_v4(make_lottery_daily(True))
    diffuse = diffuse_frame[diffuse_frame["asset"].eq("000001")].iloc[-1]
    concentrated = concentrated_frame[
        concentrated_frame["asset"].eq("000001")
    ].iloc[-1]
    assert diffuse["diffuse_upside_variance_20"] > concentrated[
        "diffuse_upside_variance_20"
    ]
