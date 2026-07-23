from __future__ import annotations

import hashlib
import json
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


DAY_RECORD = struct.Struct("<5If2I")
PRICE_COLUMNS = ["date", "asset", "open", "high", "low", "close", "volume", "amount"]
MAINLAND_PREFIXES = {
    "sh": ("600", "601", "603", "605", "688", "689"),
    "sz": ("000", "001", "002", "003", "300", "301"),
}


def stable_asset_bucket(asset: str, bucket_count: int = 2) -> int:
    digest = hashlib.sha256(str(asset).zfill(6).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % bucket_count


def read_tdx_day_file(path: str | Path, asset: str) -> pd.DataFrame:
    source = Path(path)
    payload = source.read_bytes()
    if len(payload) % DAY_RECORD.size:
        raise ValueError(f"TDX file size is not divisible by 32 bytes: {source}")
    records = list(DAY_RECORD.iter_unpack(payload))
    frame = pd.DataFrame(
        records,
        columns=[
            "date_integer", "open", "high", "low", "close", "amount", "volume", "reserved"
        ],
    )
    if frame.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    frame["date"] = pd.to_datetime(
        frame.pop("date_integer").astype(str), format="%Y%m%d", errors="raise"
    )
    for column in ["open", "high", "low", "close"]:
        frame[column] = frame[column] / 100.0
    frame["asset"] = str(asset).zfill(6)
    return frame[PRICE_COLUMNS]


def tdx_day_path(root: str | Path, asset: str) -> Path:
    code = str(asset).zfill(6)
    market = "sh" if code.startswith(("5", "6", "9")) else "sz"
    return Path(root) / market / "lday" / f"{market}{code}.day"


def eligible_tdx_files(root: str | Path) -> list[tuple[str, Path]]:
    base = Path(root)
    output: list[tuple[str, Path]] = []
    for market in ("sh", "sz"):
        directory = base / market / "lday"
        for path in sorted(directory.glob(f"{market}*.day")):
            asset = path.stem[2:8]
            if asset.startswith(MAINLAND_PREFIXES[market]):
                output.append((asset, path))
    if not output:
        raise FileNotFoundError(f"No mainland A-share .day files found under {base}")
    return output


def collect_seen_assets(paths: Iterable[str | Path]) -> tuple[set[str], list[dict[str, object]]]:
    assets: set[str] = set()
    sources: list[dict[str, object]] = []
    expanded: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*membership.csv")))
            expanded.extend(sorted(path.glob("*.parquet")))
        else:
            expanded.append(path)
    for path in dict.fromkeys(expanded):
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            if path.suffix.lower() == ".parquet":
                values = pd.read_parquet(path, columns=["asset"])["asset"]
            elif path.suffix.lower() == ".csv":
                values = pd.read_csv(path, dtype={"asset": str}, usecols=["asset"])["asset"]
            else:
                continue
        except (KeyError, ValueError):
            continue
        source_assets = set(values.astype("string").str.zfill(6).dropna())
        assets.update(source_assets)
        sources.append(
            {
                "path": str(path.resolve()),
                "assets": len(source_assets),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if not assets:
        raise ValueError("Seen-asset inputs did not contain any asset identifiers")
    return assets, sources


def build_unseen_memberships(
    tdx_root: str | Path,
    seen_assets: set[str],
    *,
    start: str = "2019-01-01",
    end: str = "2026-05-31",
    minimum_members: int = 80,
) -> tuple[dict[int, pd.DataFrame], dict[str, object]]:
    rows: list[pd.DataFrame] = []
    eligible = eligible_tdx_files(tdx_root)
    unseen_files = [(asset, path) for asset, path in eligible if asset not in seen_assets]
    for asset, path in unseen_files:
        bars = read_tdx_day_file(path, asset)[["date", "asset", "amount"]]
        bars["liquidity_60"] = bars["amount"].rolling(60, min_periods=30).mean()
        bars["month"] = bars["date"].dt.to_period("M")
        monthly = bars.groupby("month", as_index=False).tail(1)
        monthly = monthly[monthly["date"].between(start, end)].copy()
        monthly["bucket"] = stable_asset_bucket(asset)
        rows.append(monthly)
    if not rows:
        raise ValueError("No unseen assets remained after applying the exclusion union")
    candidates = pd.concat(rows, ignore_index=True)
    common_last_date = candidates.groupby("month")["date"].transform("max")
    candidates = candidates[
        candidates["date"].eq(common_last_date) & candidates["liquidity_60"].gt(0.0)
    ].copy()
    memberships: dict[int, pd.DataFrame] = {}
    monthly_ranges: dict[str, dict[str, int]] = {}
    for bucket in (0, 1):
        selected = candidates[candidates["bucket"].eq(bucket)].copy()
        counts = selected.groupby("month")["asset"].nunique()
        if counts.empty or int(counts.min()) < minimum_members:
            raise RuntimeError(
                f"Bucket {bucket} has only {int(counts.min()) if not counts.empty else 0} "
                f"members in at least one month; required {minimum_members}"
            )
        selected["membership_date"] = selected["month"].dt.to_timestamp("M")
        selected["source_update_date"] = selected["date"]
        selected["asset_name"] = "TDX-" + selected["asset"]
        membership = selected[
            ["membership_date", "source_update_date", "asset", "asset_name", "liquidity_60"]
        ].sort_values(["membership_date", "asset"])
        memberships[bucket] = membership.reset_index(drop=True)
        monthly_ranges[str(bucket)] = {
            "minimum": int(counts.min()),
            "median": int(counts.median()),
            "maximum": int(counts.max()),
            "months": int(len(counts)),
            "asset_union": int(membership["asset"].nunique()),
        }
    intersection = set(memberships[0]["asset"]).intersection(memberships[1]["asset"])
    manifest = {
        "selection": "all positive-liquidity unseen assets at each common TDX month-end",
        "unseen_definition": "eligible TDX asset code absent from the union of every supplied prior panel",
        "bucket_rule": "sha256(asset) first 8 bytes modulo 2",
        "eligible_tdx_assets": len(eligible),
        "seen_asset_union": len(seen_assets),
        "unseen_tdx_assets": len(unseen_files),
        "minimum_members": minimum_members,
        "monthly_ranges": monthly_ranges,
        "asset_intersection_count": len(intersection),
        "start": start,
        "end": end,
    }
    return memberships, manifest


def _market_symbol(asset: str) -> str:
    return ("sh" if str(asset).startswith(("5", "6", "9")) else "sz") + str(asset).zfill(6)


def _fetch_tencent_hfq(symbol: str, start: str, end: str, timeout: float = 30.0) -> pd.DataFrame:
    import requests

    start_date = pd.Timestamp(start).normalize()
    cursor = pd.Timestamp(end).normalize()
    pages: list[pd.DataFrame] = []
    while cursor >= start_date:
        params = {
            "_var": f"kline_dayhfq{cursor.year}",
            "param": f"{symbol},day,{start_date.date()},{cursor.date()},640,hfq",
            "r": "0.8205512681390605",
        }
        response = requests.get(
            "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        separator = response.text.find("={")
        if separator < 0:
            raise RuntimeError(f"Tencent returned malformed JSONP for {symbol}")
        payload = json.loads(response.text[separator + 1 :])
        data = payload.get("data", {}).get(symbol, {})
        raw_rows = data.get("hfqday") or data.get("day") or []
        if not raw_rows:
            break
        page = pd.DataFrame(raw_rows).iloc[:, :6]
        page.columns = ["date", "open", "close", "high", "low", "lots"]
        page["date"] = pd.to_datetime(page["date"], errors="coerce")
        page = page.dropna(subset=["date"])
        if page.empty:
            break
        pages.append(page)
        earliest = page["date"].min().normalize()
        if earliest <= start_date:
            break
        next_cursor = earliest - pd.Timedelta(days=1)
        if next_cursor >= cursor:
            raise RuntimeError(f"Tencent pagination did not advance for {symbol}")
        cursor = next_cursor
    if not pages:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    frame = pd.concat(pages, ignore_index=True).drop_duplicates("date", keep="last")
    frame = frame[frame["date"].between(start_date, pd.Timestamp(end))].copy()
    numeric = ["open", "high", "low", "close", "lots"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame["volume"] = frame.pop("lots") * 100.0
    frame["amount"] = frame["volume"] * frame["close"]
    frame["asset"] = symbol[2:]
    return frame[PRICE_COLUMNS].sort_values("date")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def download_tencent_hfq(
    assets: Iterable[str],
    cache_directory: str | Path,
    start: str,
    end: str,
    *,
    workers: int = 8,
    progress: Callable[[str], None] = print,
) -> dict[str, str]:
    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)
    requested = sorted({str(asset).zfill(6) for asset in assets})

    def one(asset: str) -> tuple[str, str, int]:
        path = directory / f"{asset}.csv"
        if path.exists():
            cached = pd.read_csv(path, nrows=2)
            if set(PRICE_COLUMNS).issubset(cached.columns):
                return asset, "cached", sum(1 for _ in path.open(encoding="utf-8")) - 1
        frame = _fetch_tencent_hfq(_market_symbol(asset), start, end)
        if frame.empty:
            raise RuntimeError(f"Tencent returned no HFQ rows for {asset}")
        _atomic_csv(frame, path)
        return asset, "downloaded", len(frame)

    status: dict[str, str] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(one, asset): asset for asset in requested}
        for number, future in enumerate(as_completed(futures), 1):
            asset = futures[future]
            try:
                _, source, rows = future.result()
                status[asset] = source
                progress(f"price {number}/{len(futures)}: {asset} ({source}, {rows} rows)")
            except Exception as error:  # preserve provider failures for audit
                errors[asset] = str(error)
                progress(f"price {number}/{len(futures)}: {asset} FAILED: {error}")
    error_path = directory / "download_errors.json"
    if errors:
        error_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"HFQ download failed for {len(errors)} assets; see {error_path}")
    error_path.unlink(missing_ok=True)
    return status


def _normalize_gbbq(path: str | Path) -> pd.DataFrame:
    from pytdx.reader import GbbqReader

    raw = GbbqReader().get_df(str(path))
    events = raw.loc[raw["category"].isin([5, 9])].copy()
    events["asset"] = events["code"].astype(str).str.zfill(6)
    events["event_date"] = pd.to_datetime(
        events["datetime"].astype(str), format="%Y%m%d", errors="coerce"
    )
    events["float_shares"] = (
        pd.to_numeric(events["songgu_qianzongguben"], errors="coerce") * 10_000.0
    )
    events = events.dropna(subset=["event_date", "float_shares"])
    events = events[events["float_shares"].gt(0.0)]
    return (
        events.sort_values(["asset", "event_date", "category"])
        .drop_duplicates(["asset", "event_date"], keep="last")
        [["asset", "event_date", "float_shares"]]
    )


def build_bucket_panel(
    membership: pd.DataFrame,
    hfq_cache: str | Path,
    tdx_root: str | Path,
    gbbq_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    member = membership.copy()
    member["asset"] = member["asset"].astype("string").str.zfill(6)
    member["month"] = pd.to_datetime(member["membership_date"]).dt.to_period("M")
    assets = sorted(member["asset"].unique())
    prices = pd.concat(
        [
            pd.read_csv(
                Path(hfq_cache) / f"{asset}.csv",
                dtype={"asset": "string"},
                parse_dates=["date"],
            )
            for asset in assets
        ],
        ignore_index=True,
    )
    prices["asset"] = prices["asset"].astype("string").str.zfill(6)
    prices["month"] = prices["date"].dt.to_period("M")
    keys = member[["month", "asset"]].drop_duplicates().assign(is_member=True)
    panel = prices.merge(keys, on=["month", "asset"], how="left")
    panel["is_member"] = panel["is_member"].eq(True)
    panel["tradestatus"] = panel[["open", "high", "low", "close"]].gt(0.0).all(axis=1).astype(int)
    panel["is_st"] = 0

    actual_frames: list[pd.DataFrame] = []
    missing_day: list[str] = []
    for asset in assets:
        path = tdx_day_path(tdx_root, asset)
        if not path.exists():
            missing_day.append(asset)
            continue
        actual = read_tdx_day_file(path, asset)[["date", "asset", "amount", "volume", "close"]]
        actual = actual.rename(
            columns={"amount": "tdx_amount", "volume": "tdx_volume", "close": "raw_close"}
        )
        actual_frames.append(actual)
    actuals = pd.concat(actual_frames, ignore_index=True)
    panel = panel.merge(actuals, on=["date", "asset"], how="left", validate="one_to_one")
    actual_match = panel["tdx_amount"].notna() & panel["tdx_volume"].notna()
    panel.loc[actual_match, "amount"] = panel.loc[actual_match, "tdx_amount"]
    panel.loc[actual_match, "volume"] = panel.loc[actual_match, "tdx_volume"]

    events = _normalize_gbbq(gbbq_path)
    structure_frames: list[pd.DataFrame] = []
    missing_events: list[str] = []
    for asset, rows in panel.groupby("asset", sort=True):
        asset_events = events[events["asset"].eq(asset)][["event_date", "float_shares"]]
        if asset_events.empty:
            missing_events.append(str(asset))
            continue
        structure = pd.merge_asof(
            rows[["date", "asset", "raw_close", "tdx_volume"]].sort_values("date"),
            asset_events.sort_values("event_date"),
            left_on="date",
            right_on="event_date",
            direction="backward",
            allow_exact_matches=True,
        )
        structure_frames.append(structure)
    structure = pd.concat(structure_frames, ignore_index=True)
    structure["float_market_cap"] = structure["raw_close"] * structure["float_shares"]
    structure["turnover_fraction"] = structure["tdx_volume"] / structure["float_shares"]
    panel = panel.drop(columns=["raw_close"], errors="ignore").merge(
        structure[["date", "asset", "float_shares", "float_market_cap", "turnover_fraction"]],
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.drop(columns=["month", "tdx_amount", "tdx_volume"], errors="ignore")
    panel = panel.replace([np.inf, -np.inf], np.nan).sort_values(["asset", "date"])
    manifest = {
        "rows": int(len(panel)),
        "assets": int(panel["asset"].nunique()),
        "date_min": panel["date"].min().date().isoformat(),
        "date_max": panel["date"].max().date().isoformat(),
        "member_months": int(member["month"].nunique()),
        "member_assets": int(member["asset"].nunique()),
        "missing_tdx_day_assets": missing_day,
        "missing_gbbq_event_assets": missing_events,
        "tdx_actual_row_coverage": float(actual_match.mean()),
        "float_market_cap_row_coverage": float(panel["float_market_cap"].notna().mean()),
        "turnover_row_coverage": float(panel["turnover_fraction"].notna().mean()),
        "adjusted_price_source": "Tencent HFQ",
        "amount_volume_source": "local TongdaXin daily files where matched",
        "share_source": "local TongdaXin gbbq category 5/9, backward as-of effective date",
        "is_st_limitation": "No point-in-time ST feed; field is zero and limitation is disclosed",
    }
    return panel.reset_index(drop=True), manifest


def attach_adjusted_tdx_vwap(
    panel: pd.DataFrame,
    tdx_root: str | Path,
    *,
    tolerance: float = 0.005,
    minimum_inside_fraction: float = 0.9999,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach an HFQ-consistent VWAP derived from actual TDX amount and volume."""
    required = {"date", "asset", "open", "high", "low", "close", "amount", "volume"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Panel is missing VWAP inputs: {sorted(missing)}")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["asset"] = frame["asset"].astype("string").str.zfill(6)
    raw_frames: list[pd.DataFrame] = []
    missing_assets: list[str] = []
    for asset in sorted(frame["asset"].dropna().unique()):
        path = tdx_day_path(tdx_root, str(asset))
        if not path.exists():
            missing_assets.append(str(asset))
            continue
        raw = read_tdx_day_file(path, str(asset))[["date", "asset", "close"]]
        raw_frames.append(raw.rename(columns={"close": "tdx_raw_close"}))
    if not raw_frames:
        raise FileNotFoundError("No matching TDX daily files were available for VWAP enrichment")
    raw_closes = pd.concat(raw_frames, ignore_index=True)
    frame = frame.merge(raw_closes, on=["date", "asset"], how="left", validate="one_to_one")
    valid = (
        frame["amount"].gt(0.0)
        & frame["volume"].gt(0.0)
        & frame["tdx_raw_close"].gt(0.0)
        & frame["close"].gt(0.0)
    )
    frame["tdx_adjustment_factor"] = np.nan
    frame.loc[valid, "tdx_adjustment_factor"] = (
        frame.loc[valid, "close"] / frame.loc[valid, "tdx_raw_close"]
    )
    frame["adjusted_vwap"] = np.nan
    frame.loc[valid, "adjusted_vwap"] = (
        frame.loc[valid, "amount"]
        / frame.loc[valid, "volume"]
        * frame.loc[valid, "tdx_adjustment_factor"]
    )
    physical = valid & frame["low"].gt(0.0) & frame["high"].ge(frame["low"])
    inside = (
        frame.loc[physical, "adjusted_vwap"].ge(frame.loc[physical, "low"] * (1.0 - tolerance))
        & frame.loc[physical, "adjusted_vwap"].le(frame.loc[physical, "high"] * (1.0 + tolerance))
    )
    inside_fraction = float(inside.mean()) if len(inside) else 0.0
    if inside_fraction < minimum_inside_fraction:
        raise ValueError(
            f"Adjusted VWAP physical-bound coverage {inside_fraction:.8f} is below "
            f"the frozen minimum {minimum_inside_fraction:.8f}"
        )
    manifest = {
        "rows": int(len(frame)),
        "assets": int(frame["asset"].nunique()),
        "valid_vwap_rows": int(valid.sum()),
        "valid_vwap_fraction": float(valid.mean()),
        "physical_bound_rows": int(physical.sum()),
        "inside_daily_range_fraction_with_tolerance": inside_fraction,
        "range_tolerance": tolerance,
        "minimum_inside_fraction": minimum_inside_fraction,
        "missing_tdx_day_assets": missing_assets,
        "raw_vwap_formula": "TDX amount / TDX volume",
        "adjustment_formula": "Tencent HFQ close / TDX raw close",
    }
    return frame.replace([np.inf, -np.inf], np.nan), manifest
