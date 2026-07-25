from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_alpha.v52_pipeline import neutralize_model_predictions_v52


def test_v52_prediction_neutralization_rejects_misaligned_index() -> None:
    panel = pd.DataFrame({"date": pd.to_datetime(["2021-01-31"])})
    predictions = pd.Series([0.1], index=[1])
    with np.testing.assert_raises_regex(ValueError, "do not align"):
        neutralize_model_predictions_v52(
            panel,
            predictions,
            candidate="ridge10_v52",
        )
