from __future__ import annotations

import pandas as pd

from ashare_alpha.factors_v33 import (
    attach_point_in_time_financial_v33,
    build_quarterly_financial_features_v33,
    finalize_candidates_v33,
)


def _financial_history() -> pd.DataFrame:
    records = []
    for asset, scale in [("000001", 1.0), ("000002", 2.0)]:
        for quarter, report_date in enumerate(
            pd.to_datetime(
                ["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"]
            )
        ):
            records.append(
                {
                    "asset": asset,
                    "report_date": report_date,
                    "report_announcement_date": report_date + pd.Timedelta(days=30),
                    "total_assets": 1_000_000.0 * scale,
                    "ttm_operating_cash_flow": 50_000.0 * scale,
                    "ttm_parent_net_profit_10k": 4.0 * scale,
                }
            )
        records.append(
            {
                "asset": asset,
                "report_date": pd.Timestamp("2024-03-31"),
                "report_announcement_date": pd.Timestamp("2024-04-25"),
                "total_assets": 1_050_000.0 * scale,
                "ttm_operating_cash_flow": 80_000.0 * scale,
                "ttm_parent_net_profit_10k": 7.0 * scale,
            }
        )
    return pd.DataFrame(records)


def test_v33_uses_year_over_year_cash_and_profit_improvements() -> None:
    features = build_quarterly_financial_features_v33(_financial_history())
    assert len(features) == 2
    assert features["profitability_improvement_yoy"].gt(0.0).all()
    assert features["cash_improvement_yoy"].gt(0.0).all()
    assert features["financial_available_date"].eq(pd.Timestamp("2024-04-25")).all()


def test_v33_alignment_is_point_in_time_and_expires() -> None:
    features = build_quarterly_financial_features_v33(_financial_history())
    formation = pd.DataFrame(
        {
            "asset": ["000001", "000001", "000001"],
            "date": pd.to_datetime(["2024-04-24", "2024-04-30", "2024-11-01"]),
        }
    )
    aligned = attach_point_in_time_financial_v33(formation, features)
    assert pd.isna(aligned.loc[0, "report_date"])
    assert aligned.loc[1, "report_date"] == pd.Timestamp("2024-03-31")
    assert pd.isna(aligned.loc[2, "report_date"])


def test_v33_excludes_financial_industries_before_ranking() -> None:
    features = build_quarterly_financial_features_v33(_financial_history())
    formation = pd.DataFrame(
        {
            "asset": ["000001", "000002"],
            "date": pd.to_datetime(["2024-04-30", "2024-04-30"]),
        }
    )
    aligned = attach_point_in_time_financial_v33(formation, features)
    aligned["industry_l1"] = ["48", "22"]
    final = finalize_candidates_v33(aligned)
    assert pd.isna(final.loc[final["asset"].eq("000001"), "cspi_v33"]).all()
    assert final.loc[final["asset"].eq("000002"), "cspi_v33"].gt(0.0).all()
