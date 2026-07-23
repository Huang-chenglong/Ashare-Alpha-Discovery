import pandas as pd

from ashare_alpha.factors_v38 import (
    FINANCIAL_MAIN_EFFECTS_V38,
    build_quarterly_financial_features_v38,
)


def test_v38_reporting_acceleration_uses_same_quarter_year_over_year() -> None:
    reports = pd.date_range("2018-03-31", periods=6, freq="QE")
    lags = [90, 90, 90, 90, 60, 90]
    history = pd.DataFrame(
        {
            "asset": "000001",
            "report_date": reports,
            "report_announcement_date": [
                date + pd.Timedelta(days=lag)
                for date, lag in zip(reports, lags)
            ],
            "total_assets": [100.0, 101.0, 102.0, 103.0, 110.0, 111.0],
            "ttm_operating_cash_flow": [5.0, 5.0, 5.0, 5.0, 8.0, 6.0],
            "ttm_parent_net_profit_10k": [
                0.0004,
                0.0004,
                0.0004,
                0.0004,
                0.0005,
                0.0005,
            ],
        }
    )
    result = build_quarterly_financial_features_v38(history)
    accelerated = result.loc[result["report_date"].eq(reports[4])].iloc[0]
    assert accelerated["prior_year_announcement_lag_days"] == 90
    assert accelerated["announcement_lag_days"] == 60
    assert accelerated["reporting_acceleration_yoy"] == 30


def test_v38_controls_include_timing_and_accounting_components() -> None:
    assert FINANCIAL_MAIN_EFFECTS_V38[:3] == [
        "positive_reporting_acceleration_rank",
        "positive_cash_improvement_rank",
        "positive_cash_roa_rank",
    ]
    assert "prior_year_announcement_lag_days" in FINANCIAL_MAIN_EFFECTS_V38
    assert "cash_improvement_yoy" in FINANCIAL_MAIN_EFFECTS_V38
