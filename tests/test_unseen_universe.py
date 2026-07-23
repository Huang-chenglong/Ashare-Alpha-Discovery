import pandas as pd

from ashare_alpha.unseen_universe import attach_adjusted_tdx_vwap, stable_asset_bucket


def test_stable_asset_bucket_is_deterministic_and_code_normalized() -> None:
    assert stable_asset_bucket("1") == stable_asset_bucket("000001")
    assert stable_asset_bucket("000001") in {0, 1}
    assert [stable_asset_bucket(code) for code in ["000001", "600000", "300750"]] == [
        stable_asset_bucket(code) for code in ["000001", "600000", "300750"]
    ]


def test_adjusted_tdx_vwap_uses_same_day_adjustment(tmp_path) -> None:
    from unittest.mock import patch

    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "asset": ["600000"],
            "open": [20.0],
            "high": [22.0],
            "low": [18.0],
            "close": [21.0],
            "amount": [1_000.0],
            "volume": [100.0],
        }
    )
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "asset": ["600000"],
            "open": [9.5],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [100.0],
            "amount": [1_000.0],
        }
    )
    day = tmp_path / "sh" / "lday" / "sh600000.day"
    day.parent.mkdir(parents=True)
    day.touch()
    with patch("ashare_alpha.unseen_universe.read_tdx_day_file", return_value=raw):
        enriched, audit = attach_adjusted_tdx_vwap(panel, tmp_path)
    assert enriched.loc[0, "adjusted_vwap"] == 20.0
    assert audit["inside_daily_range_fraction_with_tolerance"] == 1.0
