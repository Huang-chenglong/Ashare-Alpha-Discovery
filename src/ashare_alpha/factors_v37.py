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


CANDIDATES_V37 = {
    "clua_v37": Candidate(
        "cash_lead_underattention",
        (
            "Positive operating-cash improvement relative to profit improvement, "
            "positive cash ROA, and abnormally low turnover attention during the "
            "first five completed post-announcement trading sessions jointly "
            "predict delayed price incorporation after all component main effects "
            "are removed."
        ),
    )
}
CANDIDATE_COLUMNS_V37 = list(CANDIDATES_V37)
FINANCIAL_MAIN_EFFECTS_V37 = [
    "positive_cash_lead_rank",
    "positive_cash_roa_rank",
    "low_announcement_attention_rank",
    "cash_lead_improvement_yoy",
    "cash_improvement_yoy",
    "profitability_improvement_yoy",
    "cash_flow_roa",
    "profitability_roa",
    "cash_accrual_spread",
    "asset_growth_yoy",
    "announcement_attention_5",
]
CONTROL_COLUMNS_BY_CANDIDATE_V37 = {
    "clua_v37": [*BASELINE_COLUMNS, *FINANCIAL_MAIN_EFFECTS_V37]
}
PRIOR_TURNOVER_SESSIONS_V37 = 60
MINIMUM_PRIOR_TURNOVER_OBSERVATIONS_V37 = 40
ANNOUNCEMENT_WINDOW_SESSIONS_V37 = 5
MAXIMUM_FIRST_SESSION_DELAY_DAYS_V37 = 10
MAXIMUM_COMPLETION_DELAY_DAYS_V37 = 20
MAXIMUM_SIGNAL_AGE_DAYS_V37 = 90


def _announcement_attention_for_asset_v37(
    events: pd.DataFrame,
    trading: pd.DataFrame,
) -> pd.DataFrame:
    output = events.copy()
    output["event_window_completion_date"] = pd.NaT
    output["announcement_attention_5"] = np.nan
    if trading.empty:
        return output
    sessions = trading.loc[
        trading["tradestatus"].fillna(0).eq(1)
        & trading["turnover_fraction"].gt(0.0),
        ["date", "turnover_fraction"],
    ].sort_values("date")
    dates = pd.to_datetime(sessions["date"]).to_numpy(dtype="datetime64[ns]")
    turnover = pd.to_numeric(
        sessions["turnover_fraction"], errors="coerce"
    ).to_numpy(dtype=float)
    if len(dates) == 0:
        return output

    for row_index, announcement in output[
        "financial_available_date"
    ].items():
        announcement_date = np.datetime64(pd.Timestamp(announcement), "ns")
        start = int(np.searchsorted(dates, announcement_date, side="left"))
        end = start + ANNOUNCEMENT_WINDOW_SESSIONS_V37
        if start < MINIMUM_PRIOR_TURNOVER_OBSERVATIONS_V37 or end > len(dates):
            continue
        prior = turnover[
            max(0, start - PRIOR_TURNOVER_SESSIONS_V37):start
        ]
        event = turnover[start:end]
        prior = prior[np.isfinite(prior) & (prior > 0.0)]
        if (
            len(prior) < MINIMUM_PRIOR_TURNOVER_OBSERVATIONS_V37
            or len(event) != ANNOUNCEMENT_WINDOW_SESSIONS_V37
            or not np.isfinite(event).all()
            or not (event > 0.0).all()
        ):
            continue
        first_delay = int(
            (pd.Timestamp(dates[start]) - pd.Timestamp(announcement)).days
        )
        completion_delay = int(
            (pd.Timestamp(dates[end - 1]) - pd.Timestamp(announcement)).days
        )
        if not 0 <= first_delay <= MAXIMUM_FIRST_SESSION_DELAY_DAYS_V37:
            continue
        if not 0 <= completion_delay <= MAXIMUM_COMPLETION_DELAY_DAYS_V37:
            continue
        baseline = float(np.median(prior))
        event_mean = float(np.mean(event))
        output.loc[row_index, "event_window_completion_date"] = pd.Timestamp(
            dates[end - 1]
        )
        output.loc[row_index, "announcement_attention_5"] = np.clip(
            np.log(event_mean / baseline), -3.0, 3.0
        )
    return output


def build_quarterly_financial_features_v37(
    financial_history: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    quarterly = build_quarterly_financial_features_v33(financial_history)
    quarterly["cash_lead_improvement_yoy"] = (
        quarterly["cash_improvement_yoy"]
        - quarterly["profitability_improvement_yoy"]
    )
    quarterly["positive_cash_lead"] = quarterly[
        "cash_lead_improvement_yoy"
    ].clip(lower=0.0)
    quarterly["positive_cash_roa"] = quarterly["cash_flow_roa"].clip(lower=0.0)
    daily_frame = daily[
        ["date", "asset", "tradestatus", "turnover_fraction"]
    ].copy()
    daily_frame["asset"] = daily_frame["asset"].astype("string").str.zfill(6)
    daily_frame["date"] = pd.to_datetime(daily_frame["date"])
    output: list[pd.DataFrame] = []
    daily_assets = {
        asset: rows
        for asset, rows in daily_frame.groupby("asset", sort=False)
    }
    for asset, events in quarterly.groupby("asset", sort=True):
        enriched = _announcement_attention_for_asset_v37(
            events, daily_assets.get(asset, daily_frame.iloc[0:0])
        )
        output.append(enriched)
    if not output:
        return quarterly.iloc[0:0].copy()
    frame = pd.concat(output, ignore_index=True)
    frame["report_announcement_date"] = frame["financial_available_date"]
    frame["financial_available_date"] = frame["event_window_completion_date"]
    required = [
        "positive_cash_lead",
        "positive_cash_roa",
        "cash_lead_improvement_yoy",
        "announcement_attention_5",
    ]
    complete = frame[required].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    return (
        frame.loc[complete]
        .drop(columns="event_window_completion_date")
        .sort_values(["asset", "financial_available_date", "report_date"])
        .reset_index(drop=True)
    )


def attach_point_in_time_financial_v37(
    month_end: pd.DataFrame,
    quarterly_features: pd.DataFrame,
) -> pd.DataFrame:
    aligned = attach_point_in_time_financial_v33(month_end, quarterly_features)
    age = (aligned["date"] - aligned["financial_available_date"]).dt.days
    financial_columns = quarterly_features.columns.difference(["asset"])
    aligned.loc[
        ~age.between(0, MAXIMUM_SIGNAL_AGE_DAYS_V37), financial_columns
    ] = np.nan
    return aligned


def _positive_percentile_rank(
    values: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    positive = values.where(values.gt(0.0))
    return positive.groupby(dates).rank(method="average", pct=True).fillna(0.0)


def finalize_candidates_v37(aligned: pd.DataFrame) -> pd.DataFrame:
    frame = aligned.copy()
    excluded = frame["industry_l1"].astype("string").isin(
        EXCLUDED_INDUSTRIES_V33
    )
    raw_financial = [
        "positive_cash_lead",
        "positive_cash_roa",
        *FINANCIAL_MAIN_EFFECTS_V37[3:],
    ]
    frame.loc[excluded, raw_financial] = np.nan
    frame["positive_cash_lead_rank"] = _positive_percentile_rank(
        frame["positive_cash_lead"], frame["date"]
    )
    frame["positive_cash_roa_rank"] = _positive_percentile_rank(
        frame["positive_cash_roa"], frame["date"]
    )
    frame["low_announcement_attention_rank"] = (
        -frame["announcement_attention_5"]
    ).groupby(frame["date"]).rank(method="average", pct=True)
    has_financial = frame["report_date"].notna() & ~excluded
    rank_columns = FINANCIAL_MAIN_EFFECTS_V37[:3]
    frame.loc[~has_financial, rank_columns] = np.nan
    frame["clua_v37"] = frame[rank_columns].prod(axis=1, min_count=3)
    return frame.sort_values(["date", "asset"]).reset_index(drop=True)


def build_monthly_financial_panel_v37(
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
    quarterly = build_quarterly_financial_features_v37(
        financial_history, daily
    )
    return attach_point_in_time_financial_v37(month_end, quarterly)
