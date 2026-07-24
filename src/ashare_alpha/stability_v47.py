from __future__ import annotations

import pandas as pd

from .statistics import benjamini_hochberg


CANDIDATES_V47 = ("wpen10_v47", "wpen20_v47", "wgate20_v47")


def build_stability_candidates_v47(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "asset", "rrsm_rank_v44", "tafs_rank_v44"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"V47 cache is missing {sorted(missing)}")
    output = frame.sort_values(["asset", "date"]).copy()
    current = 0.60 * output["rrsm_rank_v44"] + 0.40 * output["tafs_rank_v44"]
    month_number = output["date"].dt.year * 12 + output["date"].dt.month
    grouped_value = current.groupby(output["asset"], sort=False)
    grouped_month = month_number.groupby(output["asset"], sort=False)
    lag1 = grouped_value.shift(1).where(
        month_number.sub(grouped_month.shift(1)).eq(1)
    )
    lag2 = grouped_value.shift(2).where(
        month_number.sub(grouped_month.shift(2)).eq(2)
    )
    path = pd.concat(
        [current, lag1, lag2],
        axis=1,
        keys=["current", "lag1", "lag2"],
    )
    complete = path.notna().all(axis=1)
    weighted = 0.60 * current + 0.25 * lag1 + 0.15 * lag2
    path_range = path.max(axis=1) - path.min(axis=1)
    path_floor = path.min(axis=1)
    output["wpen10_v47"] = (weighted - 0.10 * path_range).where(complete)
    output["wpen20_v47"] = (weighted - 0.20 * path_range).where(complete)
    output["wgate20_v47"] = (
        weighted * (0.80 + 0.20 * path_floor)
    ).where(complete)
    return output.sort_values(["date", "asset"]).reset_index(drop=True)


def apply_family_gate_v47(
    summary: pd.DataFrame, *, q_maximum: float
) -> pd.DataFrame:
    if tuple(summary["candidate"]) != CANDIDATES_V47:
        raise ValueError("V47 summary order differs from the frozen registry")
    output = summary.copy()
    output["family_q_bh"] = benjamini_hochberg(
        output["hac_p_one_sided"].astype(float)
    ).to_numpy()
    output["passes_construction_gate_before_family_bh"] = output[
        "passes_confirmation_gate"
    ].astype(bool)
    output["passes_construction_gate"] = (
        output["passes_construction_gate_before_family_bh"]
        & output["family_q_bh"].le(q_maximum)
    )
    return output.drop(columns=["passes_confirmation_gate"])


def select_candidate_v47(summary: pd.DataFrame) -> str | None:
    passed = summary.loc[summary["passes_construction_gate"]].copy()
    if passed.empty:
        return None
    order = {candidate: index for index, candidate in enumerate(CANDIDATES_V47)}
    passed["registry_order"] = passed["candidate"].map(order)
    passed = passed.sort_values(
        ["mean_net_active_return", "registry_order"],
        ascending=[False, True],
        kind="mergesort",
    )
    return str(passed.iloc[0]["candidate"])


__all__ = [
    "CANDIDATES_V47",
    "apply_family_gate_v47",
    "build_stability_candidates_v47",
    "select_candidate_v47",
]
