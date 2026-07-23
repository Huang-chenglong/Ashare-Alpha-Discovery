import pandas as pd

from ashare_alpha.factors_v16 import _rolling


def test_v16_rolling_does_not_mix_assets() -> None:
    frame = pd.DataFrame(
        {"asset": ["A", "A", "B", "B"], "value": [1.0, 3.0, 10.0, 20.0]}
    )
    result = _rolling(frame, "value", 2, "mean")
    assert result.iloc[[0, 2]].isna().all()
    assert result.iloc[1] == 2.0
    assert result.iloc[3] == 15.0
