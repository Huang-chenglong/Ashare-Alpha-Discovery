from __future__ import annotations

import numpy as np
import pandas as pd

from .margin_factors_v48 import (
    CONTROL_COLUMNS_V48,
    build_margin_factor_panel_v48,
)
from .statistics import benjamini_hochberg


CANDIDATES_V50 = (
    "mpbr_v50",
    "mpsr_v50",
    "mpcs_v50",
)

PATH_MAIN_EFFECTS_V50 = (
    "path_inflow_breadth_rank",
    "path_inflow_breadth_rank_squared",
    "path_signed_ratio_rank",
    "path_signed_ratio_rank_squared",
    "path_stability_rank",
    "path_stability_rank_squared",
)

CONTROL_COLUMNS_V50 = tuple(
    dict.fromkeys([*CONTROL_COLUMNS_V48, *PATH_MAIN_EFFECTS_V50])
)


def _positive_rank(values: pd.Series) -> pd.Series:
    return values.where(values.gt(0.0)).rank(
        method="average", pct=True
    ).fillna(0.0)


def _snapshot_path_features(
    daily: pd.DataFrame,
    dense_margin: pd.DataFrame,
) -> pd.DataFrame:
    frame = daily.sort_values(["asset", "date"]).copy()
    frame["cumulative_amount"] = frame.groupby("asset", sort=False)["amount"].cumsum()
    margin = dense_margin.copy()
    margin["trade_date"] = pd.to_datetime(margin["trade_date"])
    margin["asset"] = margin["asset"].astype("string").str.zfill(6)
    dates = pd.DatetimeIndex(margin["trade_date"].unique()).sort_values()
    order = pd.Series(range(len(dates)), index=dates)
    margin["snapshot_order"] = margin["trade_date"].map(order).astype(int)
    snapshots = frame[
        ["date", "asset", "cumulative_amount"]
    ].merge(
        margin[["trade_date", "asset", "snapshot_order", "financing_balance"]],
        left_on=["date", "asset"],
        right_on=["trade_date", "asset"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["asset", "date"])
    grouped = snapshots.groupby("asset", sort=False)
    previous_order = grouped["snapshot_order"].shift(1)
    consecutive = snapshots["snapshot_order"].sub(previous_order).eq(1)
    balance_change = grouped["financing_balance"].diff().where(consecutive)
    interval_amount = grouped["cumulative_amount"].diff().where(consecutive)
    snapshots["snapshot_flow"] = balance_change / interval_amount.replace(0.0, np.nan)
    snapshots["positive_flow"] = snapshots["snapshot_flow"].gt(0.0).astype(float)
    snapshots.loc[snapshots["snapshot_flow"].isna(), "positive_flow"] = np.nan
    snapshots["absolute_flow"] = snapshots["snapshot_flow"].abs()

    grouped = snapshots.groupby("asset", sort=False)
    last_order = grouped["snapshot_order"].shift(4)
    complete_path = snapshots["snapshot_order"].sub(last_order).eq(4)
    rolling = grouped["snapshot_flow"].rolling(4, min_periods=4)
    flow_sum = rolling.sum().reset_index(level=0, drop=True)
    flow_abs_sum = (
        grouped["absolute_flow"]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )
    breadth = (
        grouped["positive_flow"]
        .rolling(4, min_periods=4)
        .mean()
        .reset_index(level=0, drop=True)
    )
    dispersion = rolling.std().reset_index(level=0, drop=True)
    mean_absolute = (
        grouped["absolute_flow"]
        .rolling(4, min_periods=4)
        .mean()
        .reset_index(level=0, drop=True)
    )
    snapshots["path_inflow_breadth"] = breadth.where(complete_path)
    snapshots["path_signed_ratio"] = (
        flow_sum / flow_abs_sum.replace(0.0, np.nan)
    ).where(complete_path)
    snapshots["path_dispersion_ratio"] = (
        dispersion / mean_absolute.replace(0.0, np.nan)
    ).where(complete_path)
    return snapshots[
        [
            "date",
            "asset",
            "path_inflow_breadth",
            "path_signed_ratio",
            "path_dispersion_ratio",
        ]
    ]


def build_margin_path_candidates_v50(
    daily: pd.DataFrame,
    dense_margin: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base, coverage = build_margin_factor_panel_v48(daily, dense_margin)
    path = _snapshot_path_features(daily, dense_margin)
    panel = base.merge(path, on=["date", "asset"], how="left", validate="one_to_one")
    parts: list[pd.DataFrame] = []
    for _, rows in panel.groupby("date", sort=True):
        rows = rows.copy()
        rows["path_inflow_breadth_rank"] = rows["path_inflow_breadth"].rank(
            method="average", pct=True
        )
        rows["path_signed_ratio_rank"] = _positive_rank(rows["path_signed_ratio"])
        rows["path_stability_rank"] = (
            -rows["path_dispersion_ratio"]
        ).rank(method="average", pct=True)
        for column in [
            "path_inflow_breadth_rank",
            "path_signed_ratio_rank",
            "path_stability_rank",
        ]:
            rows[f"{column}_squared"] = rows[column].pow(2)
        parts.append(rows)
    panel = pd.concat(parts, ignore_index=True)
    panel["mpbr_v50"] = panel["mcar_v48"] * panel["path_inflow_breadth"]
    panel["mpsr_v50"] = panel["mcar_v48"] * panel["path_signed_ratio_rank"]
    panel["mpcs_v50"] = (
        panel["mcar_v48"]
        * panel["path_inflow_breadth"]
        * panel["path_stability_rank"]
    )
    return panel.sort_values(["date", "asset"]).reset_index(drop=True), coverage


def apply_construction_gate_v50(
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    *,
    required_core_years_positive: list[int] | tuple[int, ...],
    family_bh_q_max: float,
) -> pd.DataFrame:
    output = summary.copy()
    output["family_q_bh"] = benjamini_hochberg(
        output["hac_p_one_sided"].astype(float)
    ).to_numpy()
    required_years = set(int(value) for value in required_core_years_positive)
    core_pass: dict[str, bool] = {}
    for candidate in CANDIDATES_V50:
        years = yearly[yearly["candidate"].eq(candidate)].set_index("year")
        core_pass[candidate] = bool(
            required_years.issubset(set(years.index))
            and years.reindex(sorted(required_years))["mean_rank_ic"].gt(0.0).all()
        )
    output["required_core_years_positive"] = output["candidate"].map(core_pass)
    output["passes_construction_gate_before_family_bh"] = (
        output["passes_confirmation_gate"]
        & output["required_core_years_positive"]
    )
    output["passes_construction_gate"] = (
        output["passes_construction_gate_before_family_bh"]
        & output["family_q_bh"].le(float(family_bh_q_max))
    )
    return output.drop(columns="passes_confirmation_gate")


def select_candidate_v50(summary: pd.DataFrame) -> str | None:
    passed = summary[summary["passes_construction_gate"]].copy()
    if passed.empty:
        return None
    order = {candidate: index for index, candidate in enumerate(CANDIDATES_V50)}
    passed["registry_order"] = passed["candidate"].map(order)
    passed = passed.sort_values(
        ["mean_net_active_return", "registry_order"],
        ascending=[False, True],
        kind="mergesort",
    )
    return str(passed.iloc[0]["candidate"])


__all__ = [
    "CANDIDATES_V50",
    "CONTROL_COLUMNS_V50",
    "PATH_MAIN_EFFECTS_V50",
    "apply_construction_gate_v50",
    "build_margin_path_candidates_v50",
    "select_candidate_v50",
]
