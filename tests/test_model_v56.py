from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_alpha.model_v53 import FEATURE_COLUMNS_V53
from ashare_alpha.model_v56 import make_model_v56


def test_v56_hgb_accepts_optional_missing_features() -> None:
    matrix = pd.DataFrame(
        np.tile(np.linspace(0.1, 0.9, len(FEATURE_COLUMNS_V53)), (30, 1)),
        columns=FEATURE_COLUMNS_V53,
    )
    matrix.iloc[::2, -3:] = np.nan
    target = np.array([0] * 20 + [1] * 10)
    model = make_model_v56()
    model.fit(matrix, target)
    assert model.predict_proba(matrix).shape == (30, 2)
