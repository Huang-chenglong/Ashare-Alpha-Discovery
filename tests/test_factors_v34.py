from __future__ import annotations

import pandas as pd

from ashare_alpha.factors_v34 import (
    build_quarterly_financial_features_v34,
    finalize_candidates_v34,
)


def _history() -> pd.DataFrame:
    rows = []
    for asset, scale in [("000001", 1.0), ("000002", 2.0)]:
        for index, report_date in enumerate(
            pd.date_range("2022-12-31", periods=6, freq="QE")
        ):
            cash = (50_000.0 + index * 12_000.0) * scale
            profit = (4.0 + index * 0.8) * scale
            rows.append(
                {
                    "asset": asset,
                    "report_date": report_date,
                    "report_announcement_date": report_date + pd.Timedelta(days=30),
                    "total_assets": 1_000_000.0 * scale,
                    "ttm_operating_cash_flow": cash,
                    "ttm_parent_net_profit_10k": profit,
                }
            )
    return pd.DataFrame(rows)


def test_v34_uses_prior_cash_and_current_profit_improvement() -> None:
    features = build_quarterly_financial_features_v34(_history())
    assert len(features) == 2
    assert features["current_profit_improvement_yoy"].gt(0.0).all()
    assert features["prior_cash_improvement_yoy"].gt(0.0).all()
    assert features["positive_prior_cash_improvement"].gt(0.0).all()


def test_v34_financial_industry_is_excluded_before_ranking() -> None:
    features = build_quarterly_financial_features_v34(_history())
    features = features.rename(columns={"financial_available_date": "date"})
    features["industry_l1"] = ["49", "22"]
    final = finalize_candidates_v34(features)
    assert pd.isna(final.loc[final["asset"].eq("000001"), "clpi_v34"]).all()
    assert final.loc[final["asset"].eq("000002"), "clpi_v34"].gt(0.0).all()
