from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_alpha.margin_factors_v48 import (
    CANDIDATES_V48,
    CONTROL_COLUMNS_V48,
    build_margin_factor_panel_v48,
)


def _daily() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=90)
    rows = []
    for asset_index, asset in enumerate(["000001", "600000"]):
        for index, date in enumerate(dates):
            raw_close = 10.0 + asset_index + 0.01 * index
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "open": raw_close,
                    "high": raw_close * 1.01,
                    "low": raw_close * 0.99,
                    "close": raw_close,
                    "raw_close": raw_close,
                    "volume": 1_000_000.0,
                    "amount": 10_000_000.0,
                    "float_shares": 100_000_000.0,
                    "float_market_cap": 1_000_000_000.0,
                    "turnover_fraction": 0.01,
                    "is_member": True,
                    "tradestatus": 1,
                    "is_st": 0,
                }
            )
    return pd.DataFrame(rows)


def _margin(daily: pd.DataFrame) -> pd.DataFrame:
    month_dates = (
        daily.groupby(daily["date"].dt.to_period("M"))["date"].max().tolist()
    )
    rows = []
    for month_index, date in enumerate(month_dates):
        for asset_index, asset in enumerate(["000001", "600000"]):
            balance = 100_000_000.0 + month_index * (1 - 2 * asset_index) * 1_000_000
            rows.append(
                {
                    "trade_date": date,
                    "asset": asset,
                    "exchange": "SZSE" if asset.startswith("0") else "SSE",
                    "security_name": asset,
                    "financing_balance": balance,
                    "financing_buy": 1_000_000.0,
                    "financing_repayment": np.nan,
                    "short_balance_quantity": 1000.0,
                    "short_sell_quantity": 100.0,
                    "short_repayment_quantity": np.nan,
                    "short_balance_value": (
                        10_000.0 if asset.startswith("0") else np.nan
                    ),
                    "total_margin_balance": np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_v48_builds_frozen_candidates_and_controls() -> None:
    daily = _daily()
    panel, coverage = build_margin_factor_panel_v48(daily, _margin(daily))
    assert set(CANDIDATES_V48).issubset(panel.columns)
    assert set(CONTROL_COLUMNS_V48).issubset(panel.columns)
    assert coverage["margin_coverage"].eq(1.0).all()
    complete = panel.dropna(subset=["net_financing_flow_to_turnover"])
    assert not complete.empty
    assert complete["mdar_v48"].ge(0.0).all()
    assert complete["mcar_v48"].ge(0.0).all()
    assert complete["mlcf_v48"].le(0.0).all()


def test_v48_month_gap_breaks_balance_change() -> None:
    daily = _daily()
    margin = _margin(daily)
    dates = sorted(margin["trade_date"].unique())
    missing_key = (dates[1], "000001")
    margin = margin[
        ~(
            margin["trade_date"].eq(missing_key[0])
            & margin["asset"].eq(missing_key[1])
        )
    ]
    panel, _ = build_margin_factor_panel_v48(daily, margin)
    asset_rows = panel[panel["asset"].eq("000001")].sort_values("date")
    assert pd.isna(
        asset_rows.loc[
            asset_rows["date"].eq(pd.Timestamp(dates[2])),
            "net_financing_flow_to_turnover",
        ].iloc[0]
    )
