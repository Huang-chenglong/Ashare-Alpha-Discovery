from __future__ import annotations

import pandas as pd

from ashare_alpha.margin_factors_v48 import (
    CANDIDATES_V48,
    apply_construction_gate_v48,
    select_candidate_v48,
)


def test_v48_family_gate_and_selection() -> None:
    summary = pd.DataFrame(
        {
            "candidate": list(CANDIDATES_V48),
            "hac_p_one_sided": [0.001, 0.20, 0.30],
            "mean_net_active_return": [0.002, 0.003, 0.004],
            "passes_confirmation_gate": [True, False, False],
        }
    )
    yearly = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "year": year,
                "mean_rank_ic": 0.02 if candidate == "mdar_v48" else -0.01,
            }
            for candidate in CANDIDATES_V48
            for year in [2021, 2022, 2023]
        ]
    )
    gated = apply_construction_gate_v48(
        summary,
        yearly,
        required_core_years_positive=[2021, 2022, 2023],
        family_bh_q_max=0.10,
    )
    assert gated.loc[
        gated["candidate"].eq("mdar_v48"), "passes_construction_gate"
    ].item()
    assert select_candidate_v48(gated) == "mdar_v48"


def test_v48_selection_returns_none_without_full_pass() -> None:
    summary = pd.DataFrame(
        {
            "candidate": list(CANDIDATES_V48),
            "passes_construction_gate": [False, False, False],
            "mean_net_active_return": [0.1, 0.2, 0.3],
        }
    )
    assert select_candidate_v48(summary) is None
