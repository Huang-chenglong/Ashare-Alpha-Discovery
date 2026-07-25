import numpy as np
import pandas as pd

from ashare_alpha.composite_v44 import (
    CANDIDATES_V44,
    apply_family_gate_v44,
    combine_component_ranks_v44,
    select_candidate_v44,
)


def test_v44_formulas_are_frozen() -> None:
    frame = pd.DataFrame(
        {"rrsm_rank_v44": [0.2, 0.8], "tafs_rank_v44": [0.6, 0.4]}
    )
    combined = combine_component_ranks_v44(frame)
    assert tuple(CANDIDATES_V44) == (
        "rrsm75_tafs25_v44",
        "rrsm60_tafs40_v44",
        "rrsm_tafs_gate_v44",
    )
    assert np.allclose(combined["rrsm75_tafs25_v44"], [0.30, 0.70])
    assert np.allclose(combined["rrsm60_tafs40_v44"], [0.36, 0.64])
    assert np.allclose(combined["rrsm_tafs_gate_v44"], [0.16, 0.56])


def test_family_bh_is_conjunctive_and_selection_prefers_net_return() -> None:
    summary = pd.DataFrame(
        {
            "candidate": CANDIDATES_V44,
            "hac_p_one_sided": [0.01, 0.02, 0.20],
            "mean_net_active_return": [0.003, 0.005, 0.010],
            "passes_confirmation_gate": [True, True, False],
        }
    )
    gated = apply_family_gate_v44(summary, q_maximum=0.10)
    assert gated["passes_construction_gate"].tolist() == [True, True, False]
    assert select_candidate_v44(gated) == "rrsm60_tafs40_v44"


def test_no_base_gate_pass_keeps_confirmation_sealed() -> None:
    summary = pd.DataFrame(
        {
            "candidate": CANDIDATES_V44,
            "hac_p_one_sided": [0.01, 0.02, 0.03],
            "mean_net_active_return": [0.003, 0.005, 0.010],
            "passes_confirmation_gate": [False, False, False],
        }
    )
    gated = apply_family_gate_v44(summary, q_maximum=0.10)
    assert select_candidate_v44(gated) is None
