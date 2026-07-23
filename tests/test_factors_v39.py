import pandas as pd

from ashare_alpha.factors_v39 import (
    FINANCIAL_MAIN_EFFECTS_V39,
    build_quarterly_financial_features_v39,
)


def test_v39_combines_advance_receipts_and_contract_liabilities_across_reclassification() -> None:
    reports = pd.date_range("2018-03-31", periods=5, freq="QE")
    history = pd.DataFrame(
        {
            "asset": "000001",
            "report_date": reports,
            "report_announcement_date": reports + pd.Timedelta(days=30),
            "advance_receipts": [10.0, 10.0, 10.0, 100.0, 0.0],
            "contract_liabilities_10k": [0.0, 0.0, 0.0, 0.0, 0.02],
            "inventory": [10.0, 10.0, 10.0, 10.0, 8.0],
            "total_assets": [1000.0] * 5,
            "ttm_revenue_10k": [0.01] * 5,
            "receivables_and_notes": [10.0, 10.0, 10.0, 10.0, 8.0],
            "ttm_operating_cash_flow": [10.0] * 5,
            "ttm_parent_net_profit_10k": [0.0005] * 5,
        }
    )
    result = build_quarterly_financial_features_v39(history)
    latest = result.iloc[-1]
    assert latest["advance_receipts_intensity"] == 0.0
    assert latest["contract_liabilities_intensity"] == 2.0
    assert latest["customer_financing_intensity"] == 2.0


def test_v39_controls_both_sides_of_working_capital() -> None:
    assert "customer_financing_growth_yoy" in FINANCIAL_MAIN_EFFECTS_V39
    assert "receivables_release_yoy" in FINANCIAL_MAIN_EFFECTS_V39
    assert "inventory_release_yoy" in FINANCIAL_MAIN_EFFECTS_V39
