from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_hfq_prices(cache_directory: str | Path, assets: list[str]) -> pd.DataFrame:
    directory = Path(cache_directory)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for asset in sorted({str(value).zfill(6) for value in assets}):
        path = directory / f"{asset}.csv"
        if not path.exists():
            missing.append(asset)
            continue
        frame = pd.read_csv(
            path,
            usecols=["date", "asset", "open", "high", "low", "close"],
            dtype={"asset": "string"},
            parse_dates=["date"],
        )
        frame["asset"] = frame["asset"].astype("string").str.zfill(6)
        frame[["open", "high", "low", "close"]] = frame[
            ["open", "high", "low", "close"]
        ].apply(
            pd.to_numeric, errors="coerce"
        )
        frames.append(frame)
    if missing:
        raise FileNotFoundError(
            f"HFQ cache is missing {len(missing)} requested assets; first={missing[:10]}"
        )
    if not frames:
        raise FileNotFoundError(f"No HFQ price files found in {directory}")
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["date", "asset"], keep="last")
        .sort_values(["asset", "date"])
    )


def load_formation_daily(
    structural_path: str | Path,
    hfq_cache_directory: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    structural = pd.read_parquet(structural_path)
    structural["date"] = pd.to_datetime(structural["date"])
    structural["asset"] = structural["asset"].astype("string").str.zfill(6)
    prices = load_hfq_prices(
        hfq_cache_directory, structural["asset"].drop_duplicates().tolist()
    )
    adjusted = prices.rename(
        columns={column: f"hfq_{column}" for column in ["open", "high", "low", "close"]}
    )
    daily = structural.drop(
        columns=["open", "high", "low", "close"], errors="ignore"
    ).merge(
        adjusted,
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    daily = daily.rename(
        columns={f"hfq_{column}": column for column in ["open", "high", "low", "close"]}
    )
    return daily.sort_values(["asset", "date"]), prices


def align_industry_history(
    panel: pd.DataFrame,
    industry_history: pd.DataFrame,
) -> pd.DataFrame:
    history = industry_history.copy()
    history["asset"] = history["asset"].astype("string").str.zfill(6)
    history["start_date"] = pd.to_datetime(history["start_date"])
    output: list[pd.DataFrame] = []
    history_assets = set(history["asset"])
    for asset, rows in panel.groupby("asset", sort=True):
        rows = rows.sort_values("date").copy()
        if asset not in history_assets:
            rows["industry_l1"] = pd.NA
            output.append(rows)
            continue
        asset_history = history.loc[
            history["asset"].eq(asset), ["start_date", "industry_l1"]
        ].sort_values("start_date")
        aligned = pd.merge_asof(
            rows,
            asset_history,
            left_on="date",
            right_on="start_date",
            direction="backward",
            allow_exact_matches=True,
        ).drop(columns="start_date")
        output.append(aligned)
    return pd.concat(output, ignore_index=True).sort_values(["date", "asset"])


def attach_execution_returns(
    formation: pd.DataFrame,
    prices: pd.DataFrame,
    market_dates: pd.Series | pd.Index | np.ndarray,
    *,
    entry_offset: int = 1,
    exit_offset: int = 21,
    maximum_delay: int = 5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not 0 < entry_offset < exit_offset:
        raise ValueError("Require 0 < entry_offset < exit_offset")
    calendar = pd.DatetimeIndex(pd.to_datetime(market_dates).dropna().unique()).sort_values()
    positions = {date: index for index, date in enumerate(calendar)}
    clean_prices = prices.copy()
    clean_prices["date"] = pd.to_datetime(clean_prices["date"])
    clean_prices["asset"] = clean_prices["asset"].astype("string").str.zfill(6)
    clean_prices = clean_prices[
        clean_prices["open"].gt(0.0) & clean_prices["close"].gt(0.0)
    ]
    indexed = clean_prices.set_index(["asset", "date"])[["open", "close"]].sort_index()
    available_assets = set(indexed.index.get_level_values("asset"))
    records: list[dict[str, object]] = []

    for asset, asset_rows in formation.groupby("asset", sort=True):
        asset_prices = indexed.loc[asset] if asset in available_assets else pd.DataFrame()
        for formation_date in asset_rows["date"]:
            position = positions.get(formation_date)
            entry_date = pd.NaT
            exit_date = pd.NaT
            entry_open = np.nan
            exit_open = np.nan
            entry_delay = np.nan
            exit_delay = np.nan
            if position is not None:
                for delay in range(maximum_delay + 1):
                    target = position + entry_offset + delay
                    if target < len(calendar) and calendar[target] in asset_prices.index:
                        entry_date = calendar[target]
                        entry_open = float(asset_prices.loc[entry_date, "open"])
                        entry_delay = delay
                        break
                for delay in range(maximum_delay + 1):
                    target = position + exit_offset + delay
                    if target < len(calendar) and calendar[target] in asset_prices.index:
                        exit_date = calendar[target]
                        exit_open = float(asset_prices.loc[exit_date, "open"])
                        exit_delay = delay
                        break

            full_window = (
                position is not None
                and position + exit_offset + maximum_delay < len(calendar)
            )
            status = "observed_open_to_open"
            future_return = np.nan
            if np.isfinite(entry_open) and np.isfinite(exit_open):
                future_return = exit_open / entry_open - 1.0
            elif full_window and not np.isfinite(entry_open):
                future_return = 0.0
                status = "entry_unfilled_cash"
            elif full_window and np.isfinite(entry_open):
                cutoff = calendar[position + exit_offset + maximum_delay]
                marks = asset_prices.loc[entry_date:cutoff, "close"]
                if not marks.empty:
                    exit_date = marks.index[-1]
                    future_return = float(marks.iloc[-1]) / entry_open - 1.0
                    status = "exit_stale_close_mark"
                else:
                    status = "unresolved_after_entry"
            else:
                status = "calendar_truncated"
            records.append(
                {
                    "date": formation_date,
                    "asset": asset,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_delay": entry_delay,
                    "exit_delay": exit_delay,
                    "return_status": status,
                    "future_return_20": future_return,
                }
            )

    returns = pd.DataFrame(records)
    output = formation.merge(
        returns, on=["date", "asset"], how="left", validate="one_to_one"
    )
    benchmark = output.groupby("date")["future_return_20"].transform("mean")
    output["label"] = output["future_return_20"] - benchmark
    manifest = {
        "rows": int(len(output)),
        "return_rows": int(output["future_return_20"].notna().sum()),
        "return_coverage": float(output["future_return_20"].notna().mean()),
        "observed_open_to_open": int(output["return_status"].eq("observed_open_to_open").sum()),
        "entry_unfilled_cash": int(output["return_status"].eq("entry_unfilled_cash").sum()),
        "exit_stale_close_mark": int(output["return_status"].eq("exit_stale_close_mark").sum()),
        "unresolved_or_truncated": int(
            output["return_status"].isin(["unresolved_after_entry", "calendar_truncated"]).sum()
        ),
        "maximum_execution_delay_sessions": maximum_delay,
    }
    return output.sort_values(["date", "asset"]), manifest
