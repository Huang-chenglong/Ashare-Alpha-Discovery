from __future__ import annotations

import pandas as pd

from ashare_alpha.factors_v32 import (
    attach_point_in_time_financial_v32,
    build_quarterly_financial_features_v32,
)


def _financial_history() -> pd.DataFrame:
    records = []
    for asset, multiplier in [("000001", 1.0), ("000002", 2.0)]:
        records.extend(
            [
                {
                    "asset": asset,
                    "report_date": "2023-12-31",
                    "report_announcement_date": "2024-03-20",
                    "listed_float_a_shares": 1_000.0 * multiplier,
                    "shareholder_count": 100.0,
                    "institution_count": 10.0,
                    "institution_shares": 300.0 * multiplier,
                    "top10_float_a_shares": 500.0 * multiplier,
                    "free_float_shares": 800.0 * multiplier,
                },
                {
                    "asset": asset,
                    "report_date": "2024-03-31",
                    "report_announcement_date": "2024-04-25",
                    "listed_float_a_shares": 1_100.0 * multiplier,
                    "shareholder_count": 105.0,
                    "institution_count": 15.0,
                    "institution_shares": 350.0 * multiplier,
                    "top10_float_a_shares": 480.0 * multiplier,
                    "free_float_shares": 900.0 * multiplier,
                },
            ]
        )
    return pd.DataFrame(records)


def test_quarterly_v32_features_require_consecutive_clean_reports() -> None:
    features = build_quarterly_financial_features_v32(_financial_history())
    assert len(features) == 2
    assert features["float_supply_growth"].gt(0.0).all()
    assert features["institution_breadth_growth"].gt(0.0).all()
    assert features["positive_diffusion"].gt(0.0).all()
    assert features["financial_available_date"].eq(pd.Timestamp("2024-04-25")).all()


def test_point_in_time_v32_alignment_never_uses_future_release() -> None:
    features = build_quarterly_financial_features_v32(_financial_history())
    formation = pd.DataFrame(
        {
            "asset": ["000001", "000001", "000002", "000002"],
            "date": pd.to_datetime(
                ["2024-04-24", "2024-04-30", "2024-04-24", "2024-04-30"]
            ),
        }
    )
    aligned = attach_point_in_time_financial_v32(formation, features)
    before = aligned["date"].eq(pd.Timestamp("2024-04-24"))
    after = aligned["date"].eq(pd.Timestamp("2024-04-30"))
    assert aligned.loc[before, "diba_v32"].isna().all()
    assert aligned.loc[after, "diba_v32"].gt(0.0).all()
    assert aligned.loc[after, "diba_v32"].nunique() == 1


def test_point_in_time_v32_alignment_expires_stale_signal() -> None:
    features = build_quarterly_financial_features_v32(_financial_history())
    formation = pd.DataFrame(
        {"asset": ["000001"], "date": pd.to_datetime(["2024-11-01"])}
    )
    aligned = attach_point_in_time_financial_v32(formation, features)
    assert pd.isna(aligned.loc[0, "diba_v32"])
