from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def normalize_assets(values: pd.Series) -> pd.Series:
    return values.astype(str).str.zfill(6)


def make_asset_disjoint_bucket(
    candidate: pd.DataFrame,
    excluded_asset_sets: Iterable[set[str]],
    *,
    minimum_assets: int,
    minimum_monthly_members: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {"date", "asset", "is_member"}
    missing = required.difference(candidate.columns)
    if missing:
        raise ValueError(f"Candidate panel is missing columns: {sorted(missing)}")

    panel = candidate.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["asset"] = normalize_assets(panel["asset"])
    if panel.duplicated(["date", "asset"]).any():
        raise ValueError("Candidate panel contains duplicate date/asset rows")

    excluded: set[str] = set()
    for assets in excluded_asset_sets:
        excluded.update(str(asset).zfill(6) for asset in assets)
    source_assets = set(panel["asset"])
    selected_assets = source_assets.difference(excluded)
    if len(selected_assets) < minimum_assets:
        raise ValueError(
            f"Only {len(selected_assets)} asset-disjoint candidates; "
            f"minimum is {minimum_assets}"
        )

    output = panel[panel["asset"].isin(selected_assets)].copy()
    observed_overlap = set(output["asset"]).intersection(excluded)
    if observed_overlap:
        raise ValueError(f"Asset-disjoint filter failed: {sorted(observed_overlap)[:10]}")

    active = output[output["is_member"].eq(1)].copy()
    active["month"] = active["date"].dt.to_period("M")
    monthly_members = active.groupby("month")["asset"].nunique()
    if monthly_members.empty or monthly_members.min() < minimum_monthly_members:
        observed = int(monthly_members.min()) if not monthly_members.empty else 0
        raise ValueError(
            f"Minimum monthly active membership is {observed}; "
            f"required {minimum_monthly_members}"
        )

    manifest = {
        "source_assets": len(source_assets),
        "excluded_assets": len(excluded),
        "selected_assets": len(selected_assets),
        "asset_overlap_with_exclusions": len(observed_overlap),
        "rows": len(output),
        "date_min": output["date"].min().date().isoformat(),
        "date_max": output["date"].max().date().isoformat(),
        "membership_months": int(len(monthly_members)),
        "minimum_monthly_active_members": int(monthly_members.min()),
        "median_monthly_active_members": float(monthly_members.median()),
        "maximum_monthly_active_members": int(monthly_members.max()),
    }
    return output.sort_values(["date", "asset"]).reset_index(drop=True), manifest
