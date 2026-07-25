from __future__ import annotations

import argparse
import json

import pandas as pd

from ashare_alpha.evaluate import month_end_formation
from ashare_alpha.margin_data import download_official_margin_snapshots


def dense_snapshot_dates(
    structural_paths: list[str],
    *,
    start_date: str,
    end_date: str,
) -> pd.DatetimeIndex:
    calendars: list[pd.Series] = []
    month_ends: list[pd.Timestamp] = []
    for path in structural_paths:
        daily = pd.read_parquet(
            path,
            columns=["date", "is_member", "tradestatus", "is_st"],
        )
        daily["date"] = pd.to_datetime(daily["date"])
        calendars.append(daily["date"].drop_duplicates())
        formation = month_end_formation(daily)
        month_ends.extend(
            formation.loc[
                formation["date"].between(start_date, end_date), "date"
            ].tolist()
        )
    calendar = pd.DatetimeIndex(pd.concat(calendars).unique()).sort_values()
    calendar = calendar[calendar.to_series().between(start_date, end_date).to_numpy()]
    calendar_frame = pd.DataFrame({"date": calendar})
    weekly = calendar_frame.groupby(
        calendar_frame["date"].dt.to_period("W-FRI")
    )["date"].max()
    return (
        pd.DatetimeIndex([*weekly.tolist(), *month_ends])
        .normalize()
        .unique()
        .sort_values()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download weekly-plus-month-end official margin paths."
    )
    parser.add_argument("--structural-data", action="append", required=True)
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2026-05-31")
    parser.add_argument("--raw-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pause-seconds", type=float, default=0.10)
    args = parser.parse_args()
    dates = dense_snapshot_dates(
        args.structural_data,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if len(dates) < 300:
        raise ValueError(f"Expected at least 300 dense snapshot dates; found {len(dates)}")
    _, manifest = download_official_margin_snapshots(
        dates,
        raw_directory=args.raw_directory,
        output_path=args.output,
        manifest_path=args.manifest,
        pause_seconds=args.pause_seconds,
    )
    print(
        json.dumps(
            {
                "snapshot_dates": len(dates),
                "date_min": dates.min().strftime("%Y-%m-%d"),
                "date_max": dates.max().strftime("%Y-%m-%d"),
                "output_sha256": manifest["output_sha256"],
                "audit": manifest["audit"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
