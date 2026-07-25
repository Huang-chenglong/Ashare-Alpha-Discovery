from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_alpha.model_v52 import (
    CANDIDATES_V52,
    make_model_v52,
    walk_forward_predictions_v52,
)


def test_v52_model_registry_is_complete() -> None:
    for candidate in CANDIDATES_V52:
        assert make_model_v52(candidate) is not None


def test_v52_walk_forward_does_not_predict_without_prior_exits() -> None:
    dates = pd.to_datetime(
        ["2021-01-31"] * 3 + ["2021-02-28"] * 3 + ["2021-03-31"] * 3
    )
    frame = pd.DataFrame(
        {
            "date": dates,
            "asset": [f"{value:06d}" for value in range(9)],
            "label": np.linspace(-0.04, 0.04, 9),
            "exit_date": pd.to_datetime(
                ["2021-03-15"] * 3 + ["2021-03-20"] * 3 + ["2021-04-20"] * 3
            ),
        }
    )
    matrix = pd.DataFrame(
        {"feature": np.linspace(0.1, 0.9, 9)},
        index=frame.index,
    )
    predictions = walk_forward_predictions_v52(
        frame,
        matrix,
        candidate="ridge10_v52",
        prediction_start="2021-01-01",
        prediction_end="2021-03-31",
        minimum_training_rows=1,
    )
    assert predictions.iloc[:6].isna().all()
    assert predictions.iloc[6:].notna().all()
