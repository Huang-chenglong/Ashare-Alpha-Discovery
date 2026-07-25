import pandas as pd

from ashare_alpha.factors_v12 import _rolling


def test_v12_rolling_window_requires_full_history() -> None:
    frame = pd.DataFrame({"asset": ["A"] * 3, "value": [1.0, 2.0, 3.0]})
    result = _rolling(frame, "value", 3, "mean")
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == 2.0
