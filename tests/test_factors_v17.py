import pandas as pd

from ashare_alpha.factors_v17 import _rolling


def test_v17_rolling_requires_full_window() -> None:
    frame = pd.DataFrame({"asset": ["A"] * 3, "value": [1.0, 2.0, 6.0]})
    result = _rolling(frame, "value", 3, "mean")
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == 3.0
