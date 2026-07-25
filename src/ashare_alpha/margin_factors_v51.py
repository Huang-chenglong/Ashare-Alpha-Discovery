from __future__ import annotations

import pandas as pd

from .margin_factors_v50 import CONTROL_COLUMNS_V50
from .statistics import benjamini_hochberg


CANDIDATES_V51 = (
    "mpbr_ema2_v51",
    "mpbr_ema3_v51",
    "mpbr_stable3_v51",
)

LAGGED_PATH_MAIN_EFFECTS_V51 = (
    "financing_leverage_rank_lag1",
    "financing_leverage_rank_lag2",
    "financing_inflow_rank_lag1",
    "financing_inflow_rank_lag2",
    "negative_momentum_rank_lag1",
    "negative_momentum_rank_lag2",
    "path_inflow_breadth_rank_lag1",
    "path_inflow_breadth_rank_lag2",
)

CONTROL_COLUMNS_V51 = tuple(
    dict.fromkeys([*CONTROL_COLUMNS_V50, *LAGGED_PATH_MAIN_EFFECTS_V51])
)


def _lag(
    frame: pd.DataFrame,
    column: str,
    periods: int,
    month_number: pd.Series,
) -> pd.Series:
    values = frame.groupby("asset", sort=False)[column].shift(periods)
    lag_month = month_number.groupby(frame["asset"], sort=False).shift(periods)
    return values.where(month_number.sub(lag_month).eq(periods))


def build_persistent_path_candidates_v51(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "asset",
        "mpbr_v50",
        "financing_leverage_rank",
        "financing_inflow_rank",
        "negative_momentum_rank",
        "path_inflow_breadth_rank",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"V51 input is missing {sorted(missing)}")
    output = frame.sort_values(["asset", "date"]).copy()
    month_number = output["date"].dt.year * 12 + output["date"].dt.month
    for column in [
        "mpbr_v50",
        "financing_leverage_rank",
        "financing_inflow_rank",
        "negative_momentum_rank",
        "path_inflow_breadth_rank",
    ]:
        output[f"{column}_lag1"] = _lag(output, column, 1, month_number)
        output[f"{column}_lag2"] = _lag(output, column, 2, month_number)
    current = output["mpbr_v50"]
    lag1 = output["mpbr_v50_lag1"]
    lag2 = output["mpbr_v50_lag2"]
    output["mpbr_ema2_v51"] = (0.65 * current + 0.35 * lag1).where(
        current.notna() & lag1.notna()
    )
    ema3 = 0.50 * current + 0.30 * lag1 + 0.20 * lag2
    complete3 = current.notna() & lag1.notna() & lag2.notna()
    output["mpbr_ema3_v51"] = ema3.where(complete3)
    path = pd.concat(
        [current, lag1, lag2],
        axis=1,
        keys=["current", "lag1", "lag2"],
    )
    output["mpbr_stable3_v51"] = (
        ema3 - 0.20 * (path.max(axis=1) - path.min(axis=1))
    ).where(complete3)
    return output.sort_values(["date", "asset"]).reset_index(drop=True)


def apply_construction_gate_v51(
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
    for candidate in CANDIDATES_V51:
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


def select_candidate_v51(summary: pd.DataFrame) -> str | None:
    passed = summary[summary["passes_construction_gate"]].copy()
    if passed.empty:
        return None
    order = {candidate: index for index, candidate in enumerate(CANDIDATES_V51)}
    passed["registry_order"] = passed["candidate"].map(order)
    passed = passed.sort_values(
        ["mean_net_active_return", "registry_order"],
        ascending=[False, True],
        kind="mergesort",
    )
    return str(passed.iloc[0]["candidate"])


__all__ = [
    "CANDIDATES_V51",
    "CONTROL_COLUMNS_V51",
    "LAGGED_PATH_MAIN_EFFECTS_V51",
    "apply_construction_gate_v51",
    "build_persistent_path_candidates_v51",
    "select_candidate_v51",
]
