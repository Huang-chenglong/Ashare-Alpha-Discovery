from __future__ import annotations

import pandas as pd

from ashare_alpha.factors_v35 import (
    FINANCIAL_MAIN_EFFECTS_V35,
    attach_point_in_time_financial_v35,
    build_quarterly_financial_features_v35,
    finalize_candidates_v35,
    normalize_protocol_periods_v35,
)


def _history() -> pd.DataFrame:
    rows = []
    for asset, scale in [("000001", 1.0), ("000002", 2.0)]:
        for index, report_date in enumerate(
            pd.date_range("2023-03-31", periods=5, freq="QE")
        ):
            rows.append(
                {
                    "asset": asset,
                    "report_date": report_date,
                    "report_announcement_date": report_date + pd.Timedelta(days=30),
                    "inventory": (300_000.0 - index * 25_000.0) * scale,
                    "total_assets": (2_000_000.0 + index * 20_000.0) * scale,
                    "ttm_revenue_10k": (100.0 + index * 10.0) * scale,
                    "receivables_and_notes": (
                        400_000.0 - index * 30_000.0
                    ) * scale,
                    "ttm_operating_cash_flow": (
                        100_000.0 + index * 12_000.0
                    ) * scale,
                    "ttm_parent_net_profit_10k": (5.0 + index * 0.5) * scale,
                }
            )
    return pd.DataFrame(rows)


def test_v35_builds_positive_self_financed_release_components() -> None:
    features = build_quarterly_financial_features_v35(_history())
    assert len(features) == 2
    assert features["working_capital_release_yoy"].gt(0.0).all()
    assert features["revenue_growth_yoy"].gt(0.0).all()
    assert features["cash_margin"].gt(0.0).all()
    assert features["financial_available_date"].eq(
        pd.Timestamp("2024-04-30")
    ).all()


def test_v35_alignment_is_point_in_time_and_expires() -> None:
    features = build_quarterly_financial_features_v35(_history())
    formation = pd.DataFrame(
        {
            "asset": ["000001", "000001", "000001"],
            "date": pd.to_datetime(["2024-04-29", "2024-05-31", "2024-11-01"]),
        }
    )
    aligned = attach_point_in_time_financial_v35(formation, features)
    assert pd.isna(aligned.loc[0, "report_date"])
    assert aligned.loc[1, "report_date"] == pd.Timestamp("2024-03-31")
    assert pd.isna(aligned.loc[2, "report_date"])


def test_v35_excludes_financial_industry_before_ranking() -> None:
    features = build_quarterly_financial_features_v35(_history())
    features = features.rename(columns={"financial_available_date": "date"})
    features["industry_l1"] = ["48", "22"]
    final = finalize_candidates_v35(features)
    assert pd.isna(final.loc[final["asset"].eq("000001"), "sfgr_v35"]).all()
    assert final.loc[final["asset"].eq("000002"), "sfgr_v35"].gt(0.0).all()


def test_v35_neutralizes_every_formula_component_and_level() -> None:
    required = {
        "positive_working_capital_release_rank",
        "positive_revenue_growth_rank",
        "positive_cash_margin_rank",
        "working_capital_release_yoy",
        "receivables_release_yoy",
        "inventory_release_yoy",
        "revenue_growth_yoy",
        "cash_margin",
        "receivables_intensity",
        "inventory_intensity",
        "profitability_roa",
        "cash_flow_roa",
        "cash_accrual_spread",
        "asset_growth_yoy",
    }
    assert set(FINANCIAL_MAIN_EFFECTS_V35) == required


def test_v35_normalizes_yaml_dates_for_pandas_comparison() -> None:
    periods = {
        "discovery": [pd.Timestamp("2020-01-01").date(), pd.Timestamp("2022-12-31").date()],
        "internal_validation": [
            pd.Timestamp("2023-01-01").date(),
            pd.Timestamp("2024-12-31").date(),
        ],
    }
    normalized = normalize_protocol_periods_v35(periods)
    dates = pd.Series(pd.to_datetime(["2019-12-31", "2020-01-01", "2024-12-31"]))
    selected = dates.between(
        normalized["discovery"][0],
        normalized["internal_validation"][1],
    )
    assert selected.tolist() == [False, True, True]
