from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import BASELINE_COLUMNS, compute_baselines
from .evaluate import month_end_formation
from .factors import Candidate


CANDIDATES_V32 = {
    "diba_v32": Candidate(
        "distributed_institutional_breadth_absorption",
        (
            "The interaction of expanding free float, expanding institutional "
            "breadth, and falling top-ten float-holder concentration predicts "
            "higher returns after all component main effects are removed."
        ),
    )
}
CANDIDATE_COLUMNS_V32 = list(CANDIDATES_V32)
FINANCIAL_MAIN_EFFECTS_V32 = [
    "positive_float_expansion_rank",
    "positive_breadth_growth_rank",
    "positive_diffusion_rank",
    "float_supply_growth",
    "listed_float_growth",
    "institution_breadth_growth",
    "institution_holding_growth",
    "top10_holding_growth",
    "top10_concentration_change",
    "institution_ownership_change",
    "shareholder_growth",
]
CONTROL_COLUMNS_BY_CANDIDATE_V32 = {
    "diba_v32": [*BASELINE_COLUMNS, *FINANCIAL_MAIN_EFFECTS_V32]
}
MAXIMUM_ANNOUNCEMENT_LAG_DAYS_V32 = 365
MAXIMUM_SIGNAL_AGE_DAYS_V32 = 183


def _log_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    return np.log(current.where(current.gt(0.0))) - np.log(
        previous.where(previous.gt(0.0))
    )


def build_quarterly_financial_features_v32(
    financial_history: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "asset",
        "report_date",
        "report_announcement_date",
        "listed_float_a_shares",
        "shareholder_count",
        "institution_count",
        "institution_shares",
        "top10_float_a_shares",
        "free_float_shares",
    }
    missing = required.difference(financial_history.columns)
    if missing:
        raise ValueError(f"V32 financial history is missing {sorted(missing)}")

    frame = financial_history[list(required)].copy()
    frame["asset"] = frame["asset"].astype("string").str.zfill(6)
    frame["report_date"] = pd.to_datetime(frame["report_date"])
    frame["report_announcement_date"] = pd.to_datetime(
        frame["report_announcement_date"]
    )
    frame = frame.sort_values(["asset", "report_date"]).drop_duplicates(
        ["asset", "report_date"], keep="last"
    )
    numeric = sorted(
        required.difference({"asset", "report_date", "report_announcement_date"})
    )
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame[numeric] = frame[numeric].where(frame[numeric].gt(0.0))

    grouped = frame.groupby("asset", sort=False)
    previous = grouped[numeric].shift(1)
    quarter_number = frame["report_date"].dt.year * 4 + frame["report_date"].dt.quarter
    consecutive = quarter_number.groupby(frame["asset"], sort=False).diff().eq(1)
    announcement_lag = (
        frame["report_announcement_date"] - frame["report_date"]
    ).dt.days
    timely = announcement_lag.between(0, MAXIMUM_ANNOUNCEMENT_LAG_DAYS_V32)

    listed = frame["listed_float_a_shares"]
    previous_listed = previous["listed_float_a_shares"]
    top10_concentration = frame["top10_float_a_shares"] / listed
    previous_top10_concentration = (
        previous["top10_float_a_shares"] / previous_listed
    )
    institution_ownership = frame["institution_shares"] / listed
    previous_institution_ownership = (
        previous["institution_shares"] / previous_listed
    )
    ratio_sanity = (
        top10_concentration.between(0.0, 1.05)
        & previous_top10_concentration.between(0.0, 1.05)
        & institution_ownership.between(0.0, 1.05)
        & previous_institution_ownership.between(0.0, 1.05)
        & frame["free_float_shares"].le(1.05 * listed)
        & previous["free_float_shares"].le(1.05 * previous_listed)
    )

    frame["float_supply_growth"] = _log_change(
        frame["free_float_shares"], previous["free_float_shares"]
    )
    frame["listed_float_growth"] = _log_change(listed, previous_listed)
    frame["institution_breadth_growth"] = (
        np.log1p(frame["institution_count"])
        - np.log1p(previous["institution_count"])
    )
    frame["institution_holding_growth"] = _log_change(
        frame["institution_shares"], previous["institution_shares"]
    )
    frame["top10_holding_growth"] = _log_change(
        frame["top10_float_a_shares"], previous["top10_float_a_shares"]
    )
    frame["top10_concentration_change"] = (
        top10_concentration - previous_top10_concentration
    )
    frame["institution_ownership_change"] = (
        institution_ownership - previous_institution_ownership
    )
    frame["shareholder_growth"] = _log_change(
        frame["shareholder_count"], previous["shareholder_count"]
    )
    frame["positive_float_expansion"] = frame["float_supply_growth"].clip(lower=0.0)
    frame["positive_breadth_growth"] = frame["institution_breadth_growth"].clip(
        lower=0.0
    )
    frame["positive_diffusion"] = (-frame["top10_concentration_change"]).clip(
        lower=0.0
    )

    output_columns = [
        "asset",
        "report_date",
        "report_announcement_date",
        "positive_float_expansion",
        "positive_breadth_growth",
        "positive_diffusion",
        *FINANCIAL_MAIN_EFFECTS_V32[3:],
    ]
    valid = consecutive & timely & ratio_sanity
    valid &= frame[output_columns[3:]].replace(
        [np.inf, -np.inf], np.nan
    ).notna().all(axis=1)
    return (
        frame.loc[valid, output_columns]
        .rename(columns={"report_announcement_date": "financial_available_date"})
        .sort_values(["asset", "financial_available_date", "report_date"])
        .reset_index(drop=True)
    )


def _positive_percentile_rank(
    values: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    positive = values.where(values.gt(0.0))
    return (
        positive.groupby(dates).rank(method="average", pct=True).fillna(0.0)
    )


def attach_point_in_time_financial_v32(
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
        latest_report = events["report_date"].cummax()
        events = events.loc[events["report_date"].eq(latest_report)].drop(
            columns="asset"
        )
        aligned = pd.merge_asof(
            rows,
            events,
            left_on="date",
            right_on="financial_available_date",
            direction="backward",
            allow_exact_matches=True,
        )
        output.append(aligned)
    aligned = pd.concat(output, ignore_index=True)
    age = (aligned["date"] - aligned["financial_available_date"]).dt.days
    financial_columns = quarterly_features.columns.difference(["asset"])
    stale = ~age.between(0, MAXIMUM_SIGNAL_AGE_DAYS_V32)
    aligned.loc[stale, financial_columns] = np.nan

    aligned["positive_float_expansion_rank"] = _positive_percentile_rank(
        aligned["positive_float_expansion"], aligned["date"]
    )
    aligned["positive_breadth_growth_rank"] = _positive_percentile_rank(
        aligned["positive_breadth_growth"], aligned["date"]
    )
    aligned["positive_diffusion_rank"] = _positive_percentile_rank(
        aligned["positive_diffusion"], aligned["date"]
    )
    has_financial = aligned["report_date"].notna()
    rank_columns = FINANCIAL_MAIN_EFFECTS_V32[:3]
    aligned.loc[~has_financial, rank_columns] = np.nan
    aligned["diba_v32"] = aligned[rank_columns].prod(axis=1, min_count=3)
    return aligned.sort_values(["date", "asset"]).reset_index(drop=True)


def compute_candidates_v32(
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
    quarterly = build_quarterly_financial_features_v32(financial_history)
    return attach_point_in_time_financial_v32(month_end, quarterly)
