import pandas as pd
import pytest

from ashare_alpha.sealed_bucket import make_asset_disjoint_bucket


def sample_panel() -> pd.DataFrame:
    rows = []
    for date in pd.to_datetime(["2024-01-31", "2024-02-29"]):
        for asset in ("000001", "000002", "000003"):
            rows.append({"date": date, "asset": asset, "is_member": 1})
    return pd.DataFrame(rows)


def test_make_asset_disjoint_bucket_removes_global_exclusions() -> None:
    output, manifest = make_asset_disjoint_bucket(
        sample_panel(),
        [{"000001"}],
        minimum_assets=2,
        minimum_monthly_members=2,
    )
    assert set(output["asset"]) == {"000002", "000003"}
    assert manifest["asset_overlap_with_exclusions"] == 0
    assert manifest["minimum_monthly_active_members"] == 2


def test_make_asset_disjoint_bucket_rejects_insufficient_monthly_members() -> None:
    with pytest.raises(ValueError, match="Minimum monthly active membership"):
        make_asset_disjoint_bucket(
            sample_panel(),
            [{"000001"}],
            minimum_assets=2,
            minimum_monthly_members=3,
        )
