from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import month_end_formation
from .factors import CANDIDATE_COLUMNS, CONTROL_COLUMNS, compute_candidates
from .statistics import benjamini_hochberg


CANDIDATES_V48 = (
    "mdar_v48",
    "mcar_v48",
    "mlcf_v48",
)

MARGIN_MAIN_EFFECTS_V48 = (
    "financing_leverage_rank",
    "financing_leverage_rank_squared",
    "financing_inflow_rank",
    "financing_inflow_rank_squared",
    "financing_outflow_rank",
    "financing_outflow_rank_squared",
    "positive_momentum_rank",
    "positive_momentum_rank_squared",
    "negative_momentum_rank",
    "negative_momentum_rank_squared",
    "short_leverage_rank",
    "short_sell_intensity_rank",
)

CONTROL_COLUMNS_V48 = tuple(
    dict.fromkeys([*CANDIDATE_COLUMNS, *CONTROL_COLUMNS, *MARGIN_MAIN_EFFECTS_V48])
)


def _rolling_sum(frame: pd.DataFrame, column: str, window: int) -> pd.Series:
    return (
        frame.groupby("asset", sort=False)[column]
        .rolling(window, min_periods=window)
        .sum()
        .reset_index(level=0, drop=True)
    )


def _rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def _positive_rank(values: pd.Series) -> pd.Series:
    positive = values.where(values.gt(0.0))
    return positive.rank(method="average", pct=True).fillna(0.0)


def _cross_sectional_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    for _, rows in frame.groupby("date", sort=True):
        rows = rows.copy()
        rows["financing_leverage_rank"] = _rank(rows["financing_leverage"])
        rows["financing_inflow_rank"] = _positive_rank(
            rows["net_financing_flow_to_turnover"]
        )
        rows["financing_outflow_rank"] = _positive_rank(
            -rows["net_financing_flow_to_turnover"]
        )
        rows["positive_momentum_rank"] = _positive_rank(rows["momentum_20"])
        rows["negative_momentum_rank"] = _positive_rank(-rows["momentum_20"])
        rows["short_leverage_rank"] = _rank(rows["short_leverage"])
        rows["short_sell_intensity_rank"] = _rank(rows["short_sell_intensity"])
        for column in [
            "financing_leverage_rank",
            "financing_inflow_rank",
            "financing_outflow_rank",
            "positive_momentum_rank",
            "negative_momentum_rank",
        ]:
            rows[f"{column}_squared"] = rows[column].pow(2)
        output.append(rows)
    return pd.concat(output, ignore_index=True)


def build_margin_factor_panel_v48(
    daily: pd.DataFrame,
    margin: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_daily = {
        "date",
        "asset",
        "close",
        "raw_close",
        "volume",
        "amount",
        "float_market_cap",
        "turnover_fraction",
        "is_member",
        "tradestatus",
        "is_st",
    }
    missing_daily = required_daily.difference(daily.columns)
    if missing_daily:
        raise ValueError(f"V48 daily data is missing {sorted(missing_daily)}")
    required_margin = {
        "trade_date",
        "asset",
        "exchange",
        "financing_balance",
        "financing_buy",
        "short_balance_quantity",
        "short_sell_quantity",
        "short_balance_value",
    }
    missing_margin = required_margin.difference(margin.columns)
    if missing_margin:
        raise ValueError(f"V48 margin data is missing {sorted(missing_margin)}")

    enriched = compute_candidates(daily)
    enriched = enriched.sort_values(["asset", "date"]).copy()
    enriched["amount_20"] = _rolling_sum(enriched, "amount", 20)
    enriched["momentum_20"] = enriched.groupby("asset", sort=False)[
        "close"
    ].pct_change(20, fill_method=None)
    formation = month_end_formation(enriched)

    margin_frame = margin.copy()
    margin_frame["trade_date"] = pd.to_datetime(margin_frame["trade_date"])
    margin_frame["asset"] = margin_frame["asset"].astype("string").str.zfill(6)
    if margin_frame.duplicated(["trade_date", "asset"]).any():
        duplicates = margin_frame.loc[
            margin_frame.duplicated(["trade_date", "asset"], keep=False),
            ["trade_date", "asset", "exchange"],
        ]
        raise ValueError(f"V48 margin keys are duplicated: {duplicates.head().to_dict('records')}")
    panel = formation.merge(
        margin_frame,
        left_on=["date", "asset"],
        right_on=["trade_date", "asset"],
        how="left",
        validate="one_to_one",
    ).drop(columns="trade_date")

    month_number = panel["date"].dt.year * 12 + panel["date"].dt.month
    grouped_balance = panel.groupby("asset", sort=False)["financing_balance"]
    grouped_month = month_number.groupby(panel["asset"], sort=False)
    previous_balance = grouped_balance.shift(1).where(
        month_number.sub(grouped_month.shift(1)).eq(1)
    )
    panel["financing_leverage"] = (
        panel["financing_balance"] / panel["float_market_cap"]
    )
    panel["net_financing_flow_to_turnover"] = (
        panel["financing_balance"] - previous_balance
    ) / panel["amount_20"].replace(0.0, np.nan)
    computed_short_value = panel["short_balance_quantity"] * panel["raw_close"]
    panel["unified_short_balance_value"] = panel["short_balance_value"].fillna(
        computed_short_value
    )
    panel["short_leverage"] = (
        panel["unified_short_balance_value"] / panel["float_market_cap"]
    )
    panel["short_sell_intensity"] = (
        panel["short_sell_quantity"] / panel["volume"].replace(0.0, np.nan)
    )
    panel = _cross_sectional_ranks(panel)
    panel["mdar_v48"] = (
        panel["financing_outflow_rank"]
        * panel["positive_momentum_rank"]
        * panel["financing_leverage_rank"]
    )
    panel["mcar_v48"] = (
        panel["financing_inflow_rank"]
        * panel["negative_momentum_rank"]
        * panel["financing_leverage_rank"]
    )
    panel["mlcf_v48"] = -(
        panel["financing_inflow_rank"]
        * panel["positive_momentum_rank"]
        * panel["financing_leverage_rank"]
    )
    coverage = (
        panel.assign(
            has_margin=panel["financing_balance"].notna(),
            has_consecutive_margin=panel["net_financing_flow_to_turnover"].notna(),
            complete_controls=panel[list(CONTROL_COLUMNS_V48)].notna().all(axis=1),
        )
        .groupby("date")
        .agg(
            research_universe=("asset", "size"),
            margin_rows=("has_margin", "sum"),
            consecutive_margin_rows=("has_consecutive_margin", "sum"),
            complete_control_rows=("complete_controls", "sum"),
        )
        .reset_index()
    )
    coverage["margin_coverage"] = (
        coverage["margin_rows"] / coverage["research_universe"]
    )
    coverage["consecutive_margin_coverage"] = (
        coverage["consecutive_margin_rows"] / coverage["research_universe"]
    )
    return panel.sort_values(["date", "asset"]).reset_index(drop=True), coverage


def apply_construction_gate_v48(
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
    for candidate in CANDIDATES_V48:
        candidate_years = yearly[yearly["candidate"].eq(candidate)].set_index("year")
        core_pass[candidate] = bool(
            required_years.issubset(set(candidate_years.index))
            and candidate_years.reindex(sorted(required_years))["mean_rank_ic"]
            .gt(0.0)
            .all()
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


def select_candidate_v48(summary: pd.DataFrame) -> str | None:
    passed = summary[summary["passes_construction_gate"]].copy()
    if passed.empty:
        return None
    order = {candidate: index for index, candidate in enumerate(CANDIDATES_V48)}
    passed["registry_order"] = passed["candidate"].map(order)
    passed = passed.sort_values(
        ["mean_net_active_return", "registry_order"],
        ascending=[False, True],
        kind="mergesort",
    )
    return str(passed.iloc[0]["candidate"])


__all__ = [
    "CANDIDATES_V48",
    "CONTROL_COLUMNS_V48",
    "MARGIN_MAIN_EFFECTS_V48",
    "apply_construction_gate_v48",
    "build_margin_factor_panel_v48",
    "select_candidate_v48",
]
