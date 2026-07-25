from __future__ import annotations

import pandas as pd

from ashare_alpha.model_v54 import CANDIDATES_V54, make_model_v54, signed_tail_target_v54


def test_v54_model_registry_is_complete() -> None:
    assert all(make_model_v54(candidate) is not None for candidate in CANDIDATES_V54)


def test_v54_signed_tail_target_has_both_deciles() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-31"] * 10),
            "label": list(range(10)),
        }
    )
    assert signed_tail_target_v54(frame).value_counts().to_dict() == {
        0.0: 8,
        -1.0: 1,
        1.0: 1,
    }
