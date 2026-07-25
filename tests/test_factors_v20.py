import pandas as pd

from ashare_alpha.factors_v20 import _rolling


def test_v20_rolling_does_not_use_partial_windows() -> None:
    frame = pd.DataFrame({"asset": ["A"] * 3, "value": [1.0, 4.0, 7.0]})
    result = _rolling(frame, "value", 3, "mean")
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == 4.0
