import numpy as np
import pandas as pd

from ashare_alpha.factors_v6 import compute_candidates_v6


def make_panel(days: int) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=days)
    time = np.arange(days, dtype=float)
    frames = []
    for index, asset in enumerate(["000001", "000002", "000003"]):
        daily_return = 0.0002 * (index - 1) + 0.003 * np.sin(time / (7.0 + index))
        close = 10.0 * np.exp(np.cumsum(daily_return))
        open_price = close * (1.0 + 0.001 * np.cos(time / (5.0 + index)))
        float_shares = np.full(days, 1_000_000.0 + index * 100_000.0)
        float_shares[80:] *= 1.1
        turnover = 0.01 + 0.003 * (1.0 + np.sin(time / (9.0 + index)))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "asset": asset,
                    "open": open_price,
                    "high": np.maximum(open_price, close) * 1.01,
                    "low": np.minimum(open_price, close) * 0.99,
                    "close": close,
                    "volume": turnover * float_shares,
                    "turnover_fraction": turnover,
                    "float_shares": float_shares,
                    "float_market_cap": close * float_shares,
                    "is_member": True,
                    "tradestatus": 1,
                    "is_st": 0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_appending_extreme_future_cannot_change_historical_score() -> None:
    history = make_panel(150)
    extended = make_panel(160)
    future = extended["date"].gt(history["date"].max())
    extended.loc[future, ["open", "high", "low", "close"]] *= 20.0
    extended.loc[future, "turnover_fraction"] *= 30.0
    extended.loc[future, "volume"] *= 30.0

    original_scores = compute_candidates_v6(history).set_index(["date", "asset"])[
        "supply_aware_defensive_attention"
    ]
    extended_scores = compute_candidates_v6(extended).set_index(["date", "asset"])[
        "supply_aware_defensive_attention"
    ].reindex(original_scores.index)
    np.testing.assert_allclose(
        original_scores.to_numpy(),
        extended_scores.to_numpy(),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
