import numpy as np
import pandas as pd
import yaml

from ashare_alpha.persistence_v45 import (
    CANDIDATES_V45,
    build_persistence_candidates_v45,
)


def test_v45_formulas_require_consecutive_months() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-01-31", "2023-02-28", "2023-04-30", "2023-05-31"]
            ),
            "asset": ["000001"] * 4,
            "rrsm_rank_v44": [0.2, 0.4, 0.6, 0.8],
            "tafs_rank_v44": [0.7, 0.5, 0.3, 0.1],
        }
    )
    result = build_persistence_candidates_v45(frame)
    assert tuple(CANDIDATES_V45) == (
        "pcs70_30_v45",
        "pcs60_25_15_v45",
        "pcg_v45",
    )
    base = [0.40, 0.44, 0.48, 0.52]
    assert np.isclose(result.loc[1, "pcs70_30_v45"], 0.7 * base[1] + 0.3 * base[0])
    assert result.loc[2, "pcs70_30_v45"] != result.loc[2, "pcs70_30_v45"]
    assert result["pcs60_25_15_v45"].isna().all()
    assert np.isclose(result.loc[1, "pcg_v45"], base[1] * (0.5 + 0.5 * base[0]))


def test_confirmation_freezes_only_selected_candidate() -> None:
    with open("configs/confirmation_v45.yaml", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    assert protocol["candidate"] == "pcs60_25_15_v45"
    assert protocol["alternative_formulas_permitted"] is False
    assert protocol["refitting_on_g_permitted"] is False
    assert protocol["threshold_tuning_on_g_permitted"] is False
