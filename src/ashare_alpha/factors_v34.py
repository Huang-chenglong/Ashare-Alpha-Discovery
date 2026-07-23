from __future__ import annotations

import pandas as pd

from .baselines import BASELINE_COLUMNS, compute_baselines
from .evaluate import month_end_formation
from .factors import Candidate
from .factors_v33 import (
    EXCLUDED_INDUSTRIES_V33,
    MAXIMUM_ANNOUNCEMENT_LAG_DAYS_V33,
    MAXIMUM_SIGNAL_AGE_DAYS_V33,
    attach_point_in_time_financial_v33,
    build_quarterly_financial_features_v33,
)


CANDIDATES_V34 = {
    "clpi_v34": Candidate(
        "cash_led_profitability_improvement",
        (
            "The interaction of positive prior-quarter cash-ROA improvement and "
            "positive current profit-ROA improvement predicts higher returns "
            "after current and lagged cash/profit changes and levels are removed."
        ),
    )
}
CANDIDATE_COLUMNS_V34 = list(CANDIDATES_V34)
FINANCIAL_MAIN_EFFECTS_V34 = [
    "positive_current_profit_improvement_rank",
    "positive_prior_cash_improvement_rank",
    "current_profit_improvement_yoy",
    "prior_cash_improvement_yoy",
    "current_cash_improvement_yoy",
    "prior_profit_improvement_yoy",
    "profitability_roa",
    "cash_flow_roa",
    "cash_accrual_spread",
    "asset_growth_yoy",
]
CONTROL_COLUMNS_BY_CANDIDATE_V34 = {
    "clpi_v34": [*BASELINE_COLUMNS, *FINANCIAL_MAIN_EFFECTS_V34]
}
EXCLUDED_INDUSTRIES_V34 = EXCLUDED_INDUSTRIES_V33
MAXIMUM_ANNOUNCEMENT_LAG_DAYS_V34 = MAXIMUM_ANNOUNCEMENT_LAG_DAYS_V33
MAXIMUM_SIGNAL_AGE_DAYS_V34 = MAXIMUM_SIGNAL_AGE_DAYS_V33


def build_quarterly_financial_features_v34(
    financial_history: pd.DataFrame,
) -> pd.DataFrame:
    base = build_quarterly_financial_features_v33(financial_history).sort_values(
        ["asset", "report_date"]
    )
    grouped = base.groupby("asset", sort=False)
    base["prior_cash_improvement_yoy"] = grouped[
        "cash_improvement_yoy"
    ].shift(1)
    base["prior_profit_improvement_yoy"] = grouped[
        "profitability_improvement_yoy"
    ].shift(1)
    quarter_number = base["report_date"].dt.year * 4 + base["report_date"].dt.quarter
    consecutive = quarter_number.groupby(base["asset"], sort=False).diff().eq(1)
    base["positive_current_profit_improvement"] = base[
        "profitability_improvement_yoy"
    ].clip(lower=0.0)
    base["positive_prior_cash_improvement"] = base[
        "prior_cash_improvement_yoy"
    ].clip(lower=0.0)
    output = base.rename(
        columns={
            "profitability_improvement_yoy": "current_profit_improvement_yoy",
            "cash_improvement_yoy": "current_cash_improvement_yoy",
        }
    )
    columns = [
        "asset",
        "report_date",
        "financial_available_date",
        "positive_current_profit_improvement",
        "positive_prior_cash_improvement",
        *FINANCIAL_MAIN_EFFECTS_V34[2:],
    ]
    valid = consecutive & output[columns[3:]].notna().all(axis=1)
    return output.loc[valid, columns].reset_index(drop=True)


def attach_point_in_time_financial_v34(
    month_end: pd.DataFrame,
    quarterly_features: pd.DataFrame,
) -> pd.DataFrame:
    return attach_point_in_time_financial_v33(month_end, quarterly_features)


def _positive_percentile_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    positive = values.where(values.gt(0.0))
    return positive.groupby(dates).rank(method="average", pct=True).fillna(0.0)


def finalize_candidates_v34(aligned: pd.DataFrame) -> pd.DataFrame:
    frame = aligned.copy()
    excluded = frame["industry_l1"].astype("string").isin(EXCLUDED_INDUSTRIES_V34)
    raw_financial = [
        "positive_current_profit_improvement",
        "positive_prior_cash_improvement",
        *FINANCIAL_MAIN_EFFECTS_V34[2:],
    ]
    frame.loc[excluded, raw_financial] = pd.NA
    frame["positive_current_profit_improvement_rank"] = _positive_percentile_rank(
        frame["positive_current_profit_improvement"], frame["date"]
    )
    frame["positive_prior_cash_improvement_rank"] = _positive_percentile_rank(
        frame["positive_prior_cash_improvement"], frame["date"]
    )
    has_financial = frame["report_date"].notna() & ~excluded
    rank_columns = FINANCIAL_MAIN_EFFECTS_V34[:2]
    frame.loc[~has_financial, rank_columns] = pd.NA
    frame["clpi_v34"] = frame[rank_columns].prod(axis=1, min_count=2)
    return frame.sort_values(["date", "asset"]).reset_index(drop=True)


def build_monthly_financial_panel_v34(
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
    quarterly = build_quarterly_financial_features_v34(financial_history)
    return attach_point_in_time_financial_v34(month_end, quarterly)
