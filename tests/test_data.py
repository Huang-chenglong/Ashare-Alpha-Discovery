import pandas as pd

from ashare_alpha.data import attach_execution_returns


def test_execution_uses_delayed_open_on_shared_calendar() -> None:
    calendar = pd.bdate_range("2020-01-01", periods=30)
    formation = pd.DataFrame({"date": [calendar[0]], "asset": ["000001"]})
    prices = pd.DataFrame(
        {
            "date": calendar[2:],
            "asset": "000001",
            "open": 10.0,
            "close": 10.0,
        }
    )
    prices.loc[prices["date"].eq(calendar[21]), "open"] = 11.0
    output, manifest = attach_execution_returns(formation, prices, calendar)
    assert output.loc[0, "entry_date"] == calendar[2]
    assert output.loc[0, "entry_delay"] == 1
    assert output.loc[0, "exit_date"] == calendar[21]
    assert abs(output.loc[0, "future_return_20"] - 0.10) < 1e-12
    assert manifest["return_coverage"] == 1.0
