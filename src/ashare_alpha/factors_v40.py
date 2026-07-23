from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import BASELINE_COLUMNS, compute_baselines
from .evaluate import month_end_formation
from .factors import Candidate
from .factors_v33 import attach_point_in_time_financial_v33


CANDIDATES_V40 = {
    "cfma_v40": Candidate(
        "customer_financing_momentum_alignment",
        (
            "Positive customer-financing growth that accelerates from the prior "
            "quarter and coincides with positive revenue growth predicts higher "
            "returns after all component, level, cash-quality, known-factor, size, "
            "and industry main effects are removed."
        ),
    )
}
CANDIDATE_COLUMNS_V40 = list(CANDIDATES_V40)
FINANCIAL_MAIN_EFFECTS_V40 = [
    "positive_customer_financing_growth_rank",
    "positive_customer_financing_acceleration_rank",
    "positive_revenue_growth_rank",
    "customer_financing_growth_yoy",
    "prior_quarter_customer_financing_growth_yoy",
    "customer_financing_acceleration",
    "customer_financing_intensity",
    "advance_receipts_intensity",
    "contract_liabilities_intensity",
    "revenue_growth_yoy",
    "cash_margin",
    "profitability_roa",
    "cash_flow_roa",
    "cash_accrual_spread",
    "asset_growth_yoy",
]
CONTROL_COLUMNS_BY_CANDIDATE_V40 = {
    "cfma_v40": [*BASELINE_COLUMNS, *FINANCIAL_MAIN_EFFECTS_V40]
}
EXCLUDED_INDUSTRIES_V40 = ("48", "49")
MAXIMUM_SIGNAL_AGE_DAYS_V40 = 183


def build_quarterly_financial_features_v40(
    financial_history: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "asset",
        "report_date",
        "report_announcement_date",
        "advance_receipts",
        "contract_liabilities_10k",
        "total_assets",
        "ttm_revenue_10k",
        "ttm_operating_cash_flow",
        "ttm_parent_net_profit_10k",
    }
    missing = required.difference(financial_history.columns)
    if missing:
        raise ValueError(f"V40 financial history is missing {sorted(missing)}")
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
    frame["advance_receipts_intensity"] = (
        frame["advance_receipts"].clip(lower=0.0).fillna(0.0) / revenue
    )
    frame["contract_liabilities_intensity"] = (
        10_000.0
        * frame["contract_liabilities_10k"].clip(lower=0.0).fillna(0.0)
        / revenue
    )
    frame["customer_financing_intensity"] = (
        frame["advance_receipts_intensity"]
        + frame["contract_liabilities_intensity"]
    )
    frame["cash_margin"] = frame["ttm_operating_cash_flow"] / revenue
    frame["cash_flow_roa"] = frame["ttm_operating_cash_flow"] / assets
    frame["profitability_roa"] = (
        10_000.0 * frame["ttm_parent_net_profit_10k"] / assets
    )
    frame["cash_accrual_spread"] = (
        frame["cash_flow_roa"] - frame["profitability_roa"]
    )

    grouped = frame.groupby("asset", sort=False)
    lag1 = grouped[
        ["customer_financing_intensity", "ttm_revenue_10k"]
    ].shift(1)
    lag4 = grouped[
        ["customer_financing_intensity", "ttm_revenue_10k", "total_assets"]
    ].shift(4)
    lag5 = grouped["customer_financing_intensity"].shift(5)
    quarter_number = (
        frame["report_date"].dt.year * 4 + frame["report_date"].dt.quarter
    )
    consecutive_one = quarter_number.groupby(
        frame["asset"], sort=False
    ).diff(1).eq(1)
    consecutive_five = quarter_number.groupby(
        frame["asset"], sort=False
    ).diff(5).eq(5)
    timely = (
        frame["report_announcement_date"] - frame["report_date"]
    ).dt.days.between(0, 365)

    frame["customer_financing_growth_yoy"] = (
        frame["customer_financing_intensity"]
        - lag4["customer_financing_intensity"]
    )
    frame["prior_quarter_customer_financing_growth_yoy"] = (
        lag1["customer_financing_intensity"] - lag5
    )
    frame["customer_financing_acceleration"] = (
        frame["customer_financing_growth_yoy"]
        - frame["prior_quarter_customer_financing_growth_yoy"]
    )
    frame["revenue_growth_yoy"] = np.log(
        frame["ttm_revenue_10k"].where(frame["ttm_revenue_10k"].gt(0.0))
        / lag4["ttm_revenue_10k"].where(lag4["ttm_revenue_10k"].gt(0.0))
    )
    frame["asset_growth_yoy"] = np.log(
        frame["total_assets"].where(frame["total_assets"].gt(0.0))
        / lag4["total_assets"].where(lag4["total_assets"].gt(0.0))
    )
    frame["positive_customer_financing_growth"] = frame[
        "customer_financing_growth_yoy"
    ].clip(lower=0.0)
    frame["positive_customer_financing_acceleration"] = frame[
        "customer_financing_acceleration"
    ].clip(lower=0.0)
    frame["positive_revenue_growth"] = frame["revenue_growth_yoy"].clip(
        lower=0.0
    )
    bounds = (
        frame["customer_financing_intensity"].between(0.0, 5.0)
        & frame["customer_financing_growth_yoy"].between(-5.0, 5.0)
        & frame["prior_quarter_customer_financing_growth_yoy"].between(-5.0, 5.0)
        & frame["customer_financing_acceleration"].between(-5.0, 5.0)
        & frame["revenue_growth_yoy"].between(-3.0, 3.0)
        & frame["cash_margin"].between(-2.0, 2.0)
        & frame["profitability_roa"].between(-2.0, 2.0)
    )
    output_columns = [
        "asset",
        "report_date",
        "report_announcement_date",
        "positive_customer_financing_growth",
        "positive_customer_financing_acceleration",
        "positive_revenue_growth",
        *FINANCIAL_MAIN_EFFECTS_V40[3:],
    ]
    finite = (
        frame[output_columns[3:]]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    return (
        frame.loc[
            consecutive_one & consecutive_five & timely & bounds & finite,
            output_columns,
        ]
        .rename(
            columns={
                "report_announcement_date": "financial_available_date"
            }
        )
        .sort_values(["asset", "financial_available_date", "report_date"])
        .reset_index(drop=True)
    )


def _positive_percentile_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    positive = values.where(values.gt(0.0))
    return positive.groupby(dates).rank(method="average", pct=True).fillna(0.0)


def finalize_candidates_v40(aligned: pd.DataFrame) -> pd.DataFrame:
    frame = aligned.copy()
    excluded = frame["industry_l1"].astype("string").isin(
        EXCLUDED_INDUSTRIES_V40
    )
    raw_financial = [
        "positive_customer_financing_growth",
        "positive_customer_financing_acceleration",
        "positive_revenue_growth",
        *FINANCIAL_MAIN_EFFECTS_V40[3:],
    ]
    frame.loc[excluded, raw_financial] = np.nan
    for raw_column, rank_column in [
        (
            "positive_customer_financing_growth",
            "positive_customer_financing_growth_rank",
        ),
        (
            "positive_customer_financing_acceleration",
            "positive_customer_financing_acceleration_rank",
        ),
        ("positive_revenue_growth", "positive_revenue_growth_rank"),
    ]:
        frame[rank_column] = _positive_percentile_rank(
            frame[raw_column], frame["date"]
        )
    has_financial = frame["report_date"].notna() & ~excluded
    ranks = FINANCIAL_MAIN_EFFECTS_V40[:3]
    frame.loc[~has_financial, ranks] = np.nan
    frame["cfma_v40"] = frame[ranks].prod(axis=1, min_count=3)
    return frame.sort_values(["date", "asset"]).reset_index(drop=True)


def build_monthly_financial_panel_v40(
    daily: pd.DataFrame,
    financial_history: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "date", "asset", "float_market_cap", "is_member", "tradestatus",
        "is_st", *BASELINE_COLUMNS,
    ]
    month_end = month_end_formation(compute_baselines(daily))[keys].copy()
    quarterly = build_quarterly_financial_features_v40(financial_history)
    return attach_point_in_time_financial_v33(month_end, quarterly)
