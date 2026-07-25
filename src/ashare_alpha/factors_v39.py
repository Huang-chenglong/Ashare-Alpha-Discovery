from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import BASELINE_COLUMNS, compute_baselines
from .evaluate import month_end_formation
from .factors import Candidate
from .factors_v33 import attach_point_in_time_financial_v33


CANDIDATES_V39 = {
    "cfwa_v39": Candidate(
        "customer_funding_working_capital_alignment",
        (
            "Rising customer prepayment financing, falling receivables-and-"
            "inventory intensity, and positive revenue growth jointly predict "
            "higher returns after all liability-side, asset-side, growth, cash, "
            "known-factor, size, and industry main effects are removed."
        ),
    )
}
CANDIDATE_COLUMNS_V39 = list(CANDIDATES_V39)
FINANCIAL_MAIN_EFFECTS_V39 = [
    "positive_customer_financing_growth_rank",
    "positive_working_capital_release_rank",
    "positive_revenue_growth_rank",
    "customer_financing_growth_yoy",
    "advance_receipts_growth_yoy",
    "contract_liabilities_growth_yoy",
    "customer_financing_intensity",
    "advance_receipts_intensity",
    "contract_liabilities_intensity",
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
]
CONTROL_COLUMNS_BY_CANDIDATE_V39 = {
    "cfwa_v39": [*BASELINE_COLUMNS, *FINANCIAL_MAIN_EFFECTS_V39]
}
EXCLUDED_INDUSTRIES_V39 = ("48", "49")
MAXIMUM_SIGNAL_AGE_DAYS_V39 = 183


def build_quarterly_financial_features_v39(
    financial_history: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "asset",
        "report_date",
        "report_announcement_date",
        "advance_receipts",
        "contract_liabilities_10k",
        "inventory",
        "total_assets",
        "ttm_revenue_10k",
        "receivables_and_notes",
        "ttm_operating_cash_flow",
        "ttm_parent_net_profit_10k",
    }
    missing = required.difference(financial_history.columns)
    if missing:
        raise ValueError(f"V39 financial history is missing {sorted(missing)}")
    frame = financial_history[list(required)].copy()
    frame["asset"] = frame["asset"].astype("string").str.zfill(6)
    frame["report_date"] = pd.to_datetime(frame["report_date"])
    frame["report_announcement_date"] = pd.to_datetime(
        frame["report_announcement_date"]
    )
    numeric = sorted(
        required.difference(
            {"asset", "report_date", "report_announcement_date"}
        )
    )
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_values(["asset", "report_date"]).drop_duplicates(
        ["asset", "report_date"], keep="last"
    )

    revenue = 10_000.0 * frame["ttm_revenue_10k"].where(
        frame["ttm_revenue_10k"].gt(0.0)
    )
    assets = frame["total_assets"].where(frame["total_assets"].gt(0.0))
    advance = frame["advance_receipts"].clip(lower=0.0).fillna(0.0)
    contract = (
        10_000.0
        * frame["contract_liabilities_10k"].clip(lower=0.0).fillna(0.0)
    )
    receivables = frame["receivables_and_notes"].where(
        frame["receivables_and_notes"].ge(0.0)
    )
    inventory = frame["inventory"].where(frame["inventory"].ge(0.0))
    frame["advance_receipts_intensity"] = advance / revenue
    frame["contract_liabilities_intensity"] = contract / revenue
    frame["customer_financing_intensity"] = (
        frame["advance_receipts_intensity"]
        + frame["contract_liabilities_intensity"]
    )
    frame["receivables_intensity"] = receivables / revenue
    frame["inventory_intensity"] = inventory / revenue
    frame["working_capital_intensity"] = (
        frame["receivables_intensity"] + frame["inventory_intensity"]
    )
    frame["cash_margin"] = frame["ttm_operating_cash_flow"] / revenue
    frame["cash_flow_roa"] = frame["ttm_operating_cash_flow"] / assets
    frame["profitability_roa"] = (
        10_000.0 * frame["ttm_parent_net_profit_10k"] / assets
    )
    frame["cash_accrual_spread"] = (
        frame["cash_flow_roa"] - frame["profitability_roa"]
    )

    lag_columns = [
        "total_assets",
        "ttm_revenue_10k",
        "advance_receipts_intensity",
        "contract_liabilities_intensity",
        "customer_financing_intensity",
        "receivables_intensity",
        "inventory_intensity",
        "working_capital_intensity",
    ]
    lagged = frame.groupby("asset", sort=False)[lag_columns].shift(4)
    quarter_number = (
        frame["report_date"].dt.year * 4 + frame["report_date"].dt.quarter
    )
    consecutive_year = quarter_number.groupby(
        frame["asset"], sort=False
    ).diff(4).eq(4)
    announcement_lag = (
        frame["report_announcement_date"] - frame["report_date"]
    ).dt.days
    timely = announcement_lag.between(0, 365)

    frame["customer_financing_growth_yoy"] = (
        frame["customer_financing_intensity"]
        - lagged["customer_financing_intensity"]
    )
    frame["advance_receipts_growth_yoy"] = (
        frame["advance_receipts_intensity"]
        - lagged["advance_receipts_intensity"]
    )
    frame["contract_liabilities_growth_yoy"] = (
        frame["contract_liabilities_intensity"]
        - lagged["contract_liabilities_intensity"]
    )
    frame["working_capital_release_yoy"] = (
        lagged["working_capital_intensity"]
        - frame["working_capital_intensity"]
    )
    frame["receivables_release_yoy"] = (
        lagged["receivables_intensity"] - frame["receivables_intensity"]
    )
    frame["inventory_release_yoy"] = (
        lagged["inventory_intensity"] - frame["inventory_intensity"]
    )
    frame["revenue_growth_yoy"] = np.log(
        frame["ttm_revenue_10k"].where(frame["ttm_revenue_10k"].gt(0.0))
        / lagged["ttm_revenue_10k"].where(lagged["ttm_revenue_10k"].gt(0.0))
    )
    frame["asset_growth_yoy"] = np.log(
        frame["total_assets"].where(frame["total_assets"].gt(0.0))
        / lagged["total_assets"].where(lagged["total_assets"].gt(0.0))
    )
    frame["positive_customer_financing_growth"] = frame[
        "customer_financing_growth_yoy"
    ].clip(lower=0.0)
    frame["positive_working_capital_release"] = frame[
        "working_capital_release_yoy"
    ].clip(lower=0.0)
    frame["positive_revenue_growth"] = frame["revenue_growth_yoy"].clip(
        lower=0.0
    )

    bounds = (
        frame["customer_financing_intensity"].between(0.0, 5.0)
        & frame["receivables_intensity"].between(0.0, 5.0)
        & frame["inventory_intensity"].between(0.0, 10.0)
        & frame["customer_financing_growth_yoy"].between(-5.0, 5.0)
        & frame["working_capital_release_yoy"].between(-10.0, 10.0)
        & frame["revenue_growth_yoy"].between(-3.0, 3.0)
        & frame["cash_margin"].between(-2.0, 2.0)
        & frame["profitability_roa"].between(-2.0, 2.0)
    )
    output_columns = [
        "asset",
        "report_date",
        "report_announcement_date",
        "positive_customer_financing_growth",
        "positive_working_capital_release",
        "positive_revenue_growth",
        *FINANCIAL_MAIN_EFFECTS_V39[3:],
    ]
    finite = (
        frame[output_columns[3:]]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    return (
        frame.loc[consecutive_year & timely & bounds & finite, output_columns]
        .rename(
            columns={
                "report_announcement_date": "financial_available_date"
            }
        )
        .sort_values(["asset", "financial_available_date", "report_date"])
        .reset_index(drop=True)
    )


def _positive_percentile_rank(
    values: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    positive = values.where(values.gt(0.0))
    return positive.groupby(dates).rank(method="average", pct=True).fillna(0.0)


def finalize_candidates_v39(aligned: pd.DataFrame) -> pd.DataFrame:
    frame = aligned.copy()
    excluded = frame["industry_l1"].astype("string").isin(
        EXCLUDED_INDUSTRIES_V39
    )
    raw_financial = [
        "positive_customer_financing_growth",
        "positive_working_capital_release",
        "positive_revenue_growth",
        *FINANCIAL_MAIN_EFFECTS_V39[3:],
    ]
    frame.loc[excluded, raw_financial] = np.nan
    rank_inputs = [
        (
            "positive_customer_financing_growth",
            "positive_customer_financing_growth_rank",
        ),
        (
            "positive_working_capital_release",
            "positive_working_capital_release_rank",
        ),
        ("positive_revenue_growth", "positive_revenue_growth_rank"),
    ]
    for raw_column, rank_column in rank_inputs:
        frame[rank_column] = _positive_percentile_rank(
            frame[raw_column], frame["date"]
        )
    has_financial = frame["report_date"].notna() & ~excluded
    rank_columns = FINANCIAL_MAIN_EFFECTS_V39[:3]
    frame.loc[~has_financial, rank_columns] = np.nan
    frame["cfwa_v39"] = frame[rank_columns].prod(axis=1, min_count=3)
    return frame.sort_values(["date", "asset"]).reset_index(drop=True)


def build_monthly_financial_panel_v39(
    daily: pd.DataFrame,
    financial_history: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "date",
        "asset",
        "float_market_cap",
        "is_member",
        "tradestatus",
        "is_st",
        *BASELINE_COLUMNS,
    ]
    month_end = month_end_formation(compute_baselines(daily))[keys].copy()
    quarterly = build_quarterly_financial_features_v39(financial_history)
    return attach_point_in_time_financial_v33(month_end, quarterly)
