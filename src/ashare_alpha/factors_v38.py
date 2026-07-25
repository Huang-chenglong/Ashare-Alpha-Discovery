from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import BASELINE_COLUMNS, compute_baselines
from .evaluate import month_end_formation
from .factors import Candidate
from .factors_v33 import (
    EXCLUDED_INDUSTRIES_V33,
    attach_point_in_time_financial_v33,
    build_quarterly_financial_features_v33,
)


CANDIDATES_V38 = {
    "cora_v38": Candidate(
        "cash_confirmed_reporting_acceleration",
        (
            "A year-over-year acceleration in same-quarter report publication, "
            "positive operating-cash improvement, and positive current cash ROA "
            "jointly predict higher returns after all timing and accounting main "
            "effects are removed."
        ),
    )
}
CANDIDATE_COLUMNS_V38 = list(CANDIDATES_V38)
FINANCIAL_MAIN_EFFECTS_V38 = [
    "positive_reporting_acceleration_rank",
    "positive_cash_improvement_rank",
    "positive_cash_roa_rank",
    "reporting_acceleration_yoy",
    "announcement_lag_days",
    "prior_year_announcement_lag_days",
    "cash_improvement_yoy",
    "profitability_improvement_yoy",
    "cash_flow_roa",
    "profitability_roa",
    "cash_accrual_spread",
    "asset_growth_yoy",
]
CONTROL_COLUMNS_BY_CANDIDATE_V38 = {
    "cora_v38": [*BASELINE_COLUMNS, *FINANCIAL_MAIN_EFFECTS_V38]
}
MAXIMUM_SIGNAL_AGE_DAYS_V38 = 183


def build_quarterly_financial_features_v38(
    financial_history: pd.DataFrame,
) -> pd.DataFrame:
    quality = build_quarterly_financial_features_v33(financial_history)
    timing = financial_history[
        ["asset", "report_date", "report_announcement_date"]
    ].copy()
    timing["asset"] = timing["asset"].astype("string").str.zfill(6)
    timing["report_date"] = pd.to_datetime(timing["report_date"])
    timing["report_announcement_date"] = pd.to_datetime(
        timing["report_announcement_date"]
    )
    timing = timing.sort_values(["asset", "report_date"]).drop_duplicates(
        ["asset", "report_date"], keep="last"
    )
    timing["announcement_lag_days"] = (
        timing["report_announcement_date"] - timing["report_date"]
    ).dt.days.astype(float)
    timing["prior_year_announcement_lag_days"] = timing.groupby(
        "asset", sort=False
    )["announcement_lag_days"].shift(4)
    quarter_number = timing["report_date"].dt.year * 4 + timing[
        "report_date"
    ].dt.quarter
    consecutive_year = quarter_number.groupby(
        timing["asset"], sort=False
    ).diff(4).eq(4)
    timing["reporting_acceleration_yoy"] = (
        timing["prior_year_announcement_lag_days"]
        - timing["announcement_lag_days"]
    )
    timing = timing.loc[
        consecutive_year
        & timing["announcement_lag_days"].between(0.0, 365.0)
        & timing["prior_year_announcement_lag_days"].between(0.0, 365.0)
        & timing["reporting_acceleration_yoy"].between(-180.0, 180.0)
    ].copy()
    timing["financial_available_date"] = timing[
        "report_announcement_date"
    ]
    keys = ["asset", "report_date", "financial_available_date"]
    timing_columns = [
        *keys,
        "announcement_lag_days",
        "prior_year_announcement_lag_days",
        "reporting_acceleration_yoy",
    ]
    frame = quality.merge(
        timing[timing_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    frame["positive_reporting_acceleration"] = frame[
        "reporting_acceleration_yoy"
    ].clip(lower=0.0)
    frame["positive_cash_roa"] = frame["cash_flow_roa"].clip(lower=0.0)
    return frame.sort_values(
        ["asset", "financial_available_date", "report_date"]
    ).reset_index(drop=True)


def attach_point_in_time_financial_v38(
    month_end: pd.DataFrame,
    quarterly_features: pd.DataFrame,
) -> pd.DataFrame:
    return attach_point_in_time_financial_v33(month_end, quarterly_features)


def _positive_percentile_rank(
    values: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    positive = values.where(values.gt(0.0))
    return positive.groupby(dates).rank(method="average", pct=True).fillna(0.0)


def finalize_candidates_v38(aligned: pd.DataFrame) -> pd.DataFrame:
    frame = aligned.copy()
    excluded = frame["industry_l1"].astype("string").isin(
        EXCLUDED_INDUSTRIES_V33
    )
    raw_financial = [
        "positive_reporting_acceleration",
        "positive_cash_improvement",
        "positive_cash_roa",
        *FINANCIAL_MAIN_EFFECTS_V38[3:],
    ]
    frame.loc[excluded, raw_financial] = np.nan
    rank_inputs = [
        (
            "positive_reporting_acceleration",
            "positive_reporting_acceleration_rank",
        ),
        ("positive_cash_improvement", "positive_cash_improvement_rank"),
        ("positive_cash_roa", "positive_cash_roa_rank"),
    ]
    for raw_column, rank_column in rank_inputs:
        frame[rank_column] = _positive_percentile_rank(
            frame[raw_column], frame["date"]
        )
    has_financial = frame["report_date"].notna() & ~excluded
    rank_columns = FINANCIAL_MAIN_EFFECTS_V38[:3]
    frame.loc[~has_financial, rank_columns] = np.nan
    frame["cora_v38"] = frame[rank_columns].prod(axis=1, min_count=3)
    return frame.sort_values(["date", "asset"]).reset_index(drop=True)


def build_monthly_financial_panel_v38(
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
    quarterly = build_quarterly_financial_features_v38(financial_history)
    return attach_point_in_time_financial_v38(month_end, quarterly)
