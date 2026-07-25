from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_alpha.model_v53 import (
    CANDIDATES_V53,
    make_model_v53,
    top_decile_target_v53,
    walk_forward_predictions_v53,
)


def test_v53_model_registry_is_complete() -> None:
    for candidate in CANDIDATES_V53:
        assert make_model_v53(candidate) is not None


def test_v53_top_decile_target_is_cross_sectional() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-31"] * 10 + ["2021-02-28"] * 10),
            "label": list(range(10)) + list(reversed(range(10))),
        }
    )
    target = top_decile_target_v53(frame)
    assert target.groupby(frame["date"]).sum().tolist() == [1.0, 1.0]


def test_v53_walk_forward_waits_for_realized_exits() -> None:
    dates = pd.to_datetime(
        ["2021-01-31"] * 20
        + ["2021-02-28"] * 20
        + ["2021-03-31"] * 20
    )
    frame = pd.DataFrame(
        {
            "date": dates,
            "asset": [f"{value:06d}" for value in range(60)],
            "label": np.tile(np.linspace(-0.04, 0.04, 20), 3),
            "exit_date": pd.to_datetime(
                ["2021-03-15"] * 20
                + ["2021-03-20"] * 20
                + ["2021-04-20"] * 20
            ),
        }
    )
    matrix = pd.DataFrame(
        {"feature": np.tile(np.linspace(0.1, 0.9, 20), 3)},
        index=frame.index,
    )
    predictions = walk_forward_predictions_v53(
        frame,
        matrix,
        candidate="logit_tail_v53",
        prediction_start="2021-01-01",
        prediction_end="2021-03-31",
        minimum_training_rows=1,
    )
    assert predictions.iloc[:40].isna().all()
    assert predictions.iloc[40:].notna().all()
