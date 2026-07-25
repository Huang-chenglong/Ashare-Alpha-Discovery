import pandas as pd

from ashare_alpha.unseen_universe import (
    attach_adjusted_tdx_vwap,
    build_bucket_panel,
    stable_asset_bucket,
)


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


def test_bucket_panel_retains_unadjusted_close(tmp_path) -> None:
    from unittest.mock import patch

    membership = pd.DataFrame(
        {
            "membership_date": pd.to_datetime(["2024-01-31"]),
            "source_update_date": pd.to_datetime(["2024-01-31"]),
            "asset": ["600000"],
            "asset_name": ["TDX-600000"],
            "liquidity_60": [1_000_000.0],
        }
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31"]),
            "asset": ["600000"],
            "open": [20.0],
            "high": [22.0],
            "low": [18.0],
            "close": [21.0],
            "volume": [100.0],
            "amount": [2_000.0],
        }
    ).to_csv(cache / "600000.csv", index=False)
    day = tmp_path / "tdx" / "sh" / "lday" / "sh600000.day"
    day.parent.mkdir(parents=True)
    day.touch()
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31"]),
            "asset": ["600000"],
            "open": [9.5],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [100.0],
            "amount": [1_000.0],
        }
    )
    events = pd.DataFrame(
        {
            "asset": ["600000"],
            "event_date": pd.to_datetime(["2020-01-01"]),
            "float_shares": [1_000.0],
        }
    )
    with (
        patch("ashare_alpha.unseen_universe.read_tdx_day_file", return_value=raw),
        patch("ashare_alpha.unseen_universe._normalize_gbbq", return_value=events),
    ):
        panel, _ = build_bucket_panel(
            membership,
            cache,
            tmp_path / "tdx",
            tmp_path / "gbbq",
        )
    assert panel.loc[0, "raw_close"] == 10.5
    assert panel.loc[0, "float_market_cap"] == 10_500.0
