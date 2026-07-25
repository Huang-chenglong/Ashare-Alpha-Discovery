import numpy as np
import pandas as pd

from ashare_alpha.factors_v9 import _rolling_sum


def test_rolling_sum_is_asset_local() -> None:
    frame = pd.DataFrame(
        {
            "asset": ["A"] * 3 + ["B"] * 3,
            "value": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    result = _rolling_sum(frame, "value", 2)
    np.testing.assert_allclose(result.to_numpy(), [np.nan, 3.0, 5.0, np.nan, 30.0, 50.0], equal_nan=True)
