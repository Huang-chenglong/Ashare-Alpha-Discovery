from __future__ import annotations

import pandas as pd

from .statistics import benjamini_hochberg


CANDIDATES_V45 = (
    "pcs70_30_v45",
    "pcs60_25_15_v45",
    "pcg_v45",
)
BASE_COMPONENT_V45 = "consensus60_40_v45"


def build_persistence_candidates_v45(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "asset", "rrsm_rank_v44", "tafs_rank_v44"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"V45 cache is missing {sorted(missing)}")
    output = frame.sort_values(["asset", "date"]).copy()
    output[BASE_COMPONENT_V45] = (
        0.60 * output["rrsm_rank_v44"] + 0.40 * output["tafs_rank_v44"]
    )
    month_number = (
        output["date"].dt.year * 12 + output["date"].dt.month
    )
    grouped_value = output.groupby("asset", sort=False)[BASE_COMPONENT_V45]
    grouped_month = month_number.groupby(output["asset"], sort=False)
    lag1 = grouped_value.shift(1).where(
        month_number.sub(grouped_month.shift(1)).eq(1)
    )
    lag2 = grouped_value.shift(2).where(
        month_number.sub(grouped_month.shift(2)).eq(2)
    )
    output["pcs70_30_v45"] = (
        0.70 * output[BASE_COMPONENT_V45] + 0.30 * lag1
    )
    output["pcs60_25_15_v45"] = (
        0.60 * output[BASE_COMPONENT_V45] + 0.25 * lag1 + 0.15 * lag2
    )
    output["pcg_v45"] = output[BASE_COMPONENT_V45] * (
        0.50 + 0.50 * lag1
    )
    return output.sort_values(["date", "asset"]).reset_index(drop=True)


def apply_family_gate_v45(
    summary: pd.DataFrame,
    *,
    q_maximum: float,
) -> pd.DataFrame:
    if tuple(summary["candidate"]) != CANDIDATES_V45:
        raise ValueError("V45 summary order differs from the frozen registry")
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


def select_candidate_v45(summary: pd.DataFrame) -> str | None:
    passed = summary.loc[summary["passes_construction_gate"]].copy()
    if passed.empty:
        return None
    registry_order = {
        candidate: index for index, candidate in enumerate(CANDIDATES_V45)
    }
    passed["registry_order"] = passed["candidate"].map(registry_order)
    passed = passed.sort_values(
        ["mean_net_active_return", "registry_order"],
        ascending=[False, True],
        kind="mergesort",
    )
    return str(passed.iloc[0]["candidate"])


__all__ = [
    "BASE_COMPONENT_V45",
    "CANDIDATES_V45",
    "apply_family_gate_v45",
    "build_persistence_candidates_v45",
    "select_candidate_v45",
]
