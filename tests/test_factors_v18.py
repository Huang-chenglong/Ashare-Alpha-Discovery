import pandas as pd

from ashare_alpha.factors_v18 import _eligible_rank


def test_eligible_rank_excludes_nonmembers() -> None:
    values = pd.Series([1.0, 2.0, 99.0])
    dates = pd.to_datetime(pd.Series(["2024-01-31"] * 3))
    result = _eligible_rank(values, dates, pd.Series([True, True, False]))
    assert result.iloc[:2].tolist() == [0.5, 1.0]
    assert pd.isna(result.iloc[2])
