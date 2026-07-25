from __future__ import annotations

import pandas as pd

from ashare_alpha.model_v55 import make_model_v55, top_decile_target_v55


def test_v55_model_and_target() -> None:
    assert make_model_v55() is not None
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-31"] * 10),
            "label": list(range(10)),
        }
    )
    assert top_decile_target_v55(frame).sum() == 1
