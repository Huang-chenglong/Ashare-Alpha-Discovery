import numpy as np
import pandas as pd
import yaml

from ashare_alpha.stability_v47 import (
    CANDIDATES_V47,
    build_stability_candidates_v47,
)


def test_v47_mild_stability_formulas_are_frozen() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-31", "2023-02-28", "2023-03-31"]),
            "asset": ["000001"] * 3,
            "rrsm_rank_v44": [0.2, 0.8, 0.5],
            "tafs_rank_v44": [0.7, 0.4, 0.6],
        }
    )
    result = build_stability_candidates_v47(frame)
    assert tuple(CANDIDATES_V47) == ("wpen10_v47", "wpen20_v47", "wgate20_v47")
    path = np.array([0.40, 0.64, 0.54])
    weighted = 0.60 * path[2] + 0.25 * path[1] + 0.15 * path[0]
    spread = path.max() - path.min()
    assert np.isclose(result.loc[2, "wpen10_v47"], weighted - 0.10 * spread)
    assert np.isclose(result.loc[2, "wpen20_v47"], weighted - 0.20 * spread)
    assert np.isclose(
        result.loc[2, "wgate20_v47"], weighted * (0.80 + 0.20 * path.min())
    )


def test_v47_confirmation_freezes_one_formula_and_core_years() -> None:
    with open("configs/confirmation_v47.yaml", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    assert protocol["candidate"] == "wpen10_v47"
    assert protocol["alternative_formulas_permitted"] is False
    assert protocol["refitting_on_h_permitted"] is False
    assert protocol["gate"]["required_core_years_positive"] == [2023, 2024]
