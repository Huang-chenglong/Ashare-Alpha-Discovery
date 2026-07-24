from __future__ import annotations

import numpy as np
import pandas as pd

from .statistics import benjamini_hochberg


CANDIDATES_V46 = (
    "pmedian3_v46",
    "pfloor3_v46",
    "pstable3_v46",
)
BASE_COMPONENT_V46 = "consensus60_40_v46"


def build_robust_candidates_v46(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "asset", "rrsm_rank_v44", "tafs_rank_v44"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"V46 cache is missing {sorted(missing)}")
    output = frame.sort_values(["asset", "date"]).copy()
    output[BASE_COMPONENT_V46] = (
        0.60 * output["rrsm_rank_v44"] + 0.40 * output["tafs_rank_v44"]
    )
    month_number = output["date"].dt.year * 12 + output["date"].dt.month
    grouped_value = output.groupby("asset", sort=False)[BASE_COMPONENT_V46]
    grouped_month = month_number.groupby(output["asset"], sort=False)
    lag1 = grouped_value.shift(1).where(
        month_number.sub(grouped_month.shift(1)).eq(1)
    )
    lag2 = grouped_value.shift(2).where(
        month_number.sub(grouped_month.shift(2)).eq(2)
    )
    path = pd.concat(
        [output[BASE_COMPONENT_V46], lag1, lag2],
        axis=1,
        keys=["current", "lag1", "lag2"],
    )
    complete = path.notna().all(axis=1)
    output["pmedian3_v46"] = path.median(axis=1).where(complete)
    output["pfloor3_v46"] = path.min(axis=1).where(complete)
    output["pstable3_v46"] = (
        path.mean(axis=1) - 0.50 * (path.max(axis=1) - path.min(axis=1))
    ).where(complete)
    return output.sort_values(["date", "asset"]).reset_index(drop=True)


def apply_family_gate_v46(
    summary: pd.DataFrame,
    *,
    q_maximum: float,
) -> pd.DataFrame:
    if tuple(summary["candidate"]) != CANDIDATES_V46:
        raise ValueError("V46 summary order differs from the frozen registry")
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


def select_candidate_v46(summary: pd.DataFrame) -> str | None:
    passed = summary.loc[summary["passes_construction_gate"]].copy()
    if passed.empty:
        return None
    registry_order = {
        candidate: index for index, candidate in enumerate(CANDIDATES_V46)
    }
    passed["registry_order"] = passed["candidate"].map(registry_order)
    passed = passed.sort_values(
        ["mean_net_active_return", "registry_order"],
        ascending=[False, True],
        kind="mergesort",
    )
    return str(passed.iloc[0]["candidate"])


__all__ = [
    "BASE_COMPONENT_V46",
    "CANDIDATES_V46",
    "apply_family_gate_v46",
    "build_robust_candidates_v46",
    "select_candidate_v46",
]
