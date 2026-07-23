from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import BASELINE_COLUMNS, compute_baselines
from .evaluate import month_end_formation
from .factors import Candidate


CANDIDATES_V33 = {
    "cspi_v33": Candidate(
        "cash_synchronized_profitability_improvement",
        (
            "The interaction of positive year-over-year improvements in TTM "
            "profit-to-assets and TTM operating-cash-flow-to-assets predicts "
            "higher returns after both improvements and their levels are removed."
        ),
    )
}
CANDIDATE_COLUMNS_V33 = list(CANDIDATES_V33)
FINANCIAL_MAIN_EFFECTS_V33 = [
    "positive_profitability_improvement_rank",
    "positive_cash_improvement_rank",
    "profitability_improvement_yoy",
    "cash_improvement_yoy",
    "profitability_roa",
    "cash_flow_roa",
    "cash_accrual_spread",
    "asset_growth_yoy",
]
CONTROL_COLUMNS_BY_CANDIDATE_V33 = {
    "cspi_v33": [*BASELINE_COLUMNS, *FINANCIAL_MAIN_EFFECTS_V33]
}
EXCLUDED_INDUSTRIES_V33 = ("48", "49")
MAXIMUM_ANNOUNCEMENT_LAG_DAYS_V33 = 365
MAXIMUM_SIGNAL_AGE_DAYS_V33 = 183


def build_quarterly_financial_features_v33(
    financial_history: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "asset",
        "report_date",
        "report_announcement_date",
        "total_assets",
        "ttm_operating_cash_flow",
        "ttm_parent_net_profit_10k",
    }
    missing = required.difference(financial_history.columns)
    if missing:
        raise ValueError(f"V33 financial history is missing {sorted(missing)}")
    frame = financial_history[list(required)].copy()
    frame["asset"] = frame["asset"].astype("string").str.zfill(6)
    frame["report_date"] = pd.to_datetime(frame["report_date"])
    frame["report_announcement_date"] = pd.to_datetime(
        frame["report_announcement_date"]
    )
    numeric = [
        "total_assets",
        "ttm_operating_cash_flow",
        "ttm_parent_net_profit_10k",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_values(["asset", "report_date"]).drop_duplicates(
        ["asset", "report_date"], keep="last"
    )
    assets = frame["total_assets"].where(frame["total_assets"].gt(0.0))
    frame["cash_flow_roa"] = frame["ttm_operating_cash_flow"] / assets
    frame["profitability_roa"] = (
        10_000.0 * frame["ttm_parent_net_profit_10k"] / assets
    )

    grouped = frame.groupby("asset", sort=False)
    lagged = grouped[
        ["total_assets", "cash_flow_roa", "profitability_roa"]
    ].shift(4)
    quarter_number = frame["report_date"].dt.year * 4 + frame["report_date"].dt.quarter
    consecutive_year = quarter_number.groupby(
        frame["asset"], sort=False
    ).diff(4).eq(4)
    announcement_lag = (
        frame["report_announcement_date"] - frame["report_date"]
    ).dt.days
    timely = announcement_lag.between(0, MAXIMUM_ANNOUNCEMENT_LAG_DAYS_V33)

    frame["profitability_improvement_yoy"] = (
        frame["profitability_roa"] - lagged["profitability_roa"]
    )
    frame["cash_improvement_yoy"] = (
        frame["cash_flow_roa"] - lagged["cash_flow_roa"]
    )
    frame["cash_accrual_spread"] = (
        frame["cash_flow_roa"] - frame["profitability_roa"]
    )
    frame["asset_growth_yoy"] = np.log(
        frame["total_assets"].where(frame["total_assets"].gt(0.0))
        / lagged["total_assets"].where(lagged["total_assets"].gt(0.0))
    )
    frame["positive_profitability_improvement"] = frame[
        "profitability_improvement_yoy"
    ].clip(lower=0.0)
    frame["positive_cash_improvement"] = frame["cash_improvement_yoy"].clip(
        lower=0.0
    )
    output_columns = [
        "asset",
        "report_date",
        "report_announcement_date",
        "positive_profitability_improvement",
        "positive_cash_improvement",
        *FINANCIAL_MAIN_EFFECTS_V33[2:],
    ]
    finite = frame[output_columns[3:]].replace(
        [np.inf, -np.inf], np.nan
    ).notna().all(axis=1)
    valid = consecutive_year & timely & finite
    return (
        frame.loc[valid, output_columns]
        .rename(columns={"report_announcement_date": "financial_available_date"})
        .sort_values(["asset", "financial_available_date", "report_date"])
        .reset_index(drop=True)
    )


def attach_point_in_time_financial_v33(
    month_end: pd.DataFrame,
    quarterly_features: pd.DataFrame,
) -> pd.DataFrame:
    formation = month_end.copy()
    formation["asset"] = formation["asset"].astype("string").str.zfill(6)
    formation["date"] = pd.to_datetime(formation["date"])
    financial = quarterly_features.copy()
    financial["asset"] = financial["asset"].astype("string").str.zfill(6)
    financial["financial_available_date"] = pd.to_datetime(
        financial["financial_available_date"]
    )
    financial["report_date"] = pd.to_datetime(financial["report_date"])

    output: list[pd.DataFrame] = []
    financial_assets = set(financial["asset"])
    for asset, rows in formation.groupby("asset", sort=True):
        rows = rows.sort_values("date").copy()
        if asset not in financial_assets:
            output.append(rows)
            continue
        events = financial.loc[financial["asset"].eq(asset)].sort_values(
            ["financial_available_date", "report_date"]
        )
        events = events.drop_duplicates("financial_available_date", keep="last")
        events = events.loc[
            events["report_date"].eq(events["report_date"].cummax())
        ].drop(columns="asset")
        output.append(
            pd.merge_asof(
                rows,
                events,
                left_on="date",
                right_on="financial_available_date",
                direction="backward",
                allow_exact_matches=True,
            )
        )
    aligned = pd.concat(output, ignore_index=True)
    age = (aligned["date"] - aligned["financial_available_date"]).dt.days
    financial_columns = quarterly_features.columns.difference(["asset"])
    stale = ~age.between(0, MAXIMUM_SIGNAL_AGE_DAYS_V33)
    aligned.loc[stale, financial_columns] = np.nan
    return aligned.sort_values(["date", "asset"]).reset_index(drop=True)


def _positive_percentile_rank(values: pd.Series, dates: pd.Series) -> pd.Series:
    positive = values.where(values.gt(0.0))
    return positive.groupby(dates).rank(method="average", pct=True).fillna(0.0)


def finalize_candidates_v33(aligned: pd.DataFrame) -> pd.DataFrame:
    frame = aligned.copy()
    excluded = frame["industry_l1"].astype("string").isin(EXCLUDED_INDUSTRIES_V33)
    raw_financial = [
        "positive_profitability_improvement",
        "positive_cash_improvement",
        *FINANCIAL_MAIN_EFFECTS_V33[2:],
    ]
    frame.loc[excluded, raw_financial] = np.nan
    frame["positive_profitability_improvement_rank"] = _positive_percentile_rank(
        frame["positive_profitability_improvement"], frame["date"]
    )
    frame["positive_cash_improvement_rank"] = _positive_percentile_rank(
        frame["positive_cash_improvement"], frame["date"]
    )
    has_financial = frame["report_date"].notna() & ~excluded
    rank_columns = FINANCIAL_MAIN_EFFECTS_V33[:2]
    frame.loc[~has_financial, rank_columns] = np.nan
    frame["cspi_v33"] = frame[rank_columns].prod(axis=1, min_count=2)
    return frame.sort_values(["date", "asset"]).reset_index(drop=True)


def build_monthly_financial_panel_v33(
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
    quarterly = build_quarterly_financial_features_v33(financial_history)
    return attach_point_in_time_financial_v33(month_end, quarterly)
