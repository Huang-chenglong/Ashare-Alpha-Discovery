from __future__ import annotations

import numpy as np
import pandas as pd

from .statistics import benjamini_hochberg


CANDIDATES_V44 = (
    "rrsm75_tafs25_v44",
    "rrsm60_tafs40_v44",
    "rrsm_tafs_gate_v44",
)
COMPONENT_COLUMNS_V44 = ("rrsm_rank_v44", "tafs_rank_v44")


def combine_component_ranks_v44(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(COMPONENT_COLUMNS_V44).difference(frame.columns)
    if missing:
        raise ValueError(f"V44 component ranks are missing {sorted(missing)}")
    output = frame.copy()
    rrsm = output["rrsm_rank_v44"]
    tafs = output["tafs_rank_v44"]
    output["rrsm75_tafs25_v44"] = 0.75 * rrsm + 0.25 * tafs
    output["rrsm60_tafs40_v44"] = 0.60 * rrsm + 0.40 * tafs
    output["rrsm_tafs_gate_v44"] = rrsm * (0.50 + 0.50 * tafs)
    return output


def apply_family_gate_v44(
    summary: pd.DataFrame,
    *,
    q_maximum: float,
) -> pd.DataFrame:
    if tuple(summary["candidate"]) != CANDIDATES_V44:
        raise ValueError("V44 summary order differs from the frozen registry")
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


def select_candidate_v44(summary: pd.DataFrame) -> str | None:
    passed = summary.loc[summary["passes_construction_gate"]].copy()
    if passed.empty:
        return None
    registry_order = {candidate: index for index, candidate in enumerate(CANDIDATES_V44)}
    passed["registry_order"] = passed["candidate"].map(registry_order)
    passed = passed.sort_values(
        ["mean_net_active_return", "registry_order"],
        ascending=[False, True],
        kind="mergesort",
    )
    return str(passed.iloc[0]["candidate"])


def rank_components_v44(
    frame: pd.DataFrame,
    *,
    rrsm_score_column: str,
    tafs_score_column: str,
) -> pd.DataFrame:
    output = frame.copy()
    output["rrsm_rank_v44"] = output.groupby("date")[rrsm_score_column].rank(
        method="average", pct=True
    )
    output["tafs_rank_v44"] = output.groupby("date")[tafs_score_column].rank(
        method="average", pct=True
    )
    finite = output[list(COMPONENT_COLUMNS_V44)].replace(
        [np.inf, -np.inf], np.nan
    )
    return output.loc[finite.notna().all(axis=1)].copy()


__all__ = [
    "CANDIDATES_V44",
    "COMPONENT_COLUMNS_V44",
    "apply_family_gate_v44",
    "combine_component_ranks_v44",
    "rank_components_v44",
    "select_candidate_v44",
]
