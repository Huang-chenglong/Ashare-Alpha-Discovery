from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


SSE_DETAIL_URL = "https://query.sse.com.cn/marketdata/tradedata/queryMargin.do"
SZSE_DETAIL_URL = "https://www.szse.cn/api/report/ShowReport"

MARGIN_COLUMNS = [
    "trade_date",
    "asset",
    "exchange",
    "security_name",
    "financing_balance",
    "financing_buy",
    "financing_repayment",
    "short_balance_quantity",
    "short_sell_quantity",
    "short_repayment_quantity",
    "short_balance_value",
    "total_margin_balance",
]

NONNEGATIVE_COLUMNS = [
    "financing_balance",
    "financing_buy",
    "financing_repayment",
    "short_balance_quantity",
    "short_sell_quantity",
    "short_repayment_quantity",
    "short_balance_value",
    "total_margin_balance",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(
        values.astype("string").str.replace(",", "", regex=False),
        errors="coerce",
    )


def normalize_sse_margin_payload(
    payload: dict[str, object], trade_date: str | pd.Timestamp
) -> pd.DataFrame:
    rows = payload.get("result")
    if not isinstance(rows, list):
        raise ValueError("SSE response does not contain a result list")
    frame = pd.DataFrame(rows)
    required = {
        "opDate",
        "stockCode",
        "securityAbbr",
        "rzye",
        "rzmre",
        "rzche",
        "rqyl",
        "rqmcl",
        "rqchl",
    }
    missing = required.difference(frame.columns)
    if missing and not frame.empty:
        raise ValueError(f"SSE response is missing {sorted(missing)}")
    if frame.empty:
        return pd.DataFrame(columns=MARGIN_COLUMNS)

    expected_date = pd.Timestamp(trade_date).normalize()
    output = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(frame["opDate"], format="%Y%m%d"),
            "asset": frame["stockCode"].astype("string").str.zfill(6),
            "exchange": "SSE",
            "security_name": frame["securityAbbr"].astype("string"),
            "financing_balance": _numeric(frame["rzye"]),
            "financing_buy": _numeric(frame["rzmre"]),
            "financing_repayment": _numeric(frame["rzche"]),
            "short_balance_quantity": _numeric(frame["rqyl"]),
            "short_sell_quantity": _numeric(frame["rqmcl"]),
            "short_repayment_quantity": _numeric(frame["rqchl"]),
            "short_balance_value": pd.Series(pd.NA, index=frame.index, dtype="Float64"),
            "total_margin_balance": pd.Series(pd.NA, index=frame.index, dtype="Float64"),
        }
    )
    if not output["trade_date"].eq(expected_date).all():
        observed = sorted(output["trade_date"].dt.strftime("%Y-%m-%d").unique())
        raise ValueError(
            f"SSE response dates {observed[:5]} differ from {expected_date.date()}"
        )
    return output[MARGIN_COLUMNS]


def _find_szse_column(columns: Iterable[object], prefix: str) -> object:
    matches = [column for column in columns if str(column).strip().startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"SZSE sheet requires one {prefix!r} column; found {matches}")
    return matches[0]


def normalize_szse_margin_table(
    table: pd.DataFrame, trade_date: str | pd.Timestamp
) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame(columns=MARGIN_COLUMNS)
    columns = table.columns
    asset_column = _find_szse_column(columns, "证券代码")
    name_column = _find_szse_column(columns, "证券简称")
    financing_buy_column = _find_szse_column(columns, "融资买入额")
    financing_balance_column = _find_szse_column(columns, "融资余额")
    short_sell_column = _find_szse_column(columns, "融券卖出量")
    short_quantity_column = _find_szse_column(columns, "融券余量")
    short_value_column = _find_szse_column(columns, "融券余额")
    total_column = _find_szse_column(columns, "融资融券余额")

    output = pd.DataFrame(
        {
            "trade_date": pd.Timestamp(trade_date).normalize(),
            "asset": table[asset_column].astype("string").str.zfill(6),
            "exchange": "SZSE",
            "security_name": (
                table[name_column]
                .astype("string")
                .str.replace("&nbsp;", "", regex=False)
                .str.strip()
            ),
            "financing_balance": _numeric(table[financing_balance_column]),
            "financing_buy": _numeric(table[financing_buy_column]),
            "financing_repayment": pd.Series(
                pd.NA, index=table.index, dtype="Float64"
            ),
            "short_balance_quantity": _numeric(table[short_quantity_column]),
            "short_sell_quantity": _numeric(table[short_sell_column]),
            "short_repayment_quantity": pd.Series(
                pd.NA, index=table.index, dtype="Float64"
            ),
            "short_balance_value": _numeric(table[short_value_column]),
            "total_margin_balance": _numeric(table[total_column]),
        }
    )
    return output[MARGIN_COLUMNS]


def _request_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, str],
    headers: dict[str, str],
    attempts: int,
    timeout_seconds: float,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            return response
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Request failed after {attempts} attempts: {url}") from last_error


def fetch_sse_margin_detail(
    trade_date: str | pd.Timestamp,
    *,
    session: requests.Session,
    attempts: int = 4,
    timeout_seconds: float = 30.0,
) -> tuple[pd.DataFrame, bytes]:
    date_text = pd.Timestamp(trade_date).strftime("%Y%m%d")
    response = _request_with_retries(
        session,
        SSE_DETAIL_URL,
        params={
            "isPagination": "true",
            "tabType": "mxtype",
            "detailsDate": date_text,
            "stockCode": "",
            "beginDate": "",
            "endDate": "",
            "pageHelp.pageSize": "5000",
            "pageHelp.pageNo": "1",
            "pageHelp.beginPage": "1",
            "pageHelp.cacheSize": "1",
            "pageHelp.endPage": "21",
        },
        headers={
            "Referer": "https://www.sse.com.cn/",
            "User-Agent": "Mozilla/5.0 (Ashare-Alpha-Discovery research client)",
        },
        attempts=attempts,
        timeout_seconds=timeout_seconds,
    )
    payload = response.json()
    return normalize_sse_margin_payload(payload, date_text), response.content


def fetch_szse_margin_detail(
    trade_date: str | pd.Timestamp,
    *,
    session: requests.Session,
    attempts: int = 4,
    timeout_seconds: float = 30.0,
) -> tuple[pd.DataFrame, bytes]:
    date_text = pd.Timestamp(trade_date).strftime("%Y-%m-%d")
    response = _request_with_retries(
        session,
        SZSE_DETAIL_URL,
        params={
            "SHOWTYPE": "xlsx",
            "CATALOGID": "1837_xxpl",
            "txtDate": date_text,
            "tab2PAGENO": "1",
            "random": "0.24279342734085696",
            "TABKEY": "tab2",
        },
        headers={
            "Referer": "https://www.szse.cn/disclosure/margin/margin/index.html",
            "User-Agent": "Mozilla/5.0 (Ashare-Alpha-Discovery research client)",
        },
        attempts=attempts,
        timeout_seconds=timeout_seconds,
    )
    table = pd.read_excel(
        io.BytesIO(response.content),
        engine="openpyxl",
        dtype="string",
    )
    return normalize_szse_margin_table(table, date_text), response.content


def _read_cached_sse(path: Path, trade_date: pd.Timestamp) -> pd.DataFrame:
    return normalize_sse_margin_payload(
        json.loads(path.read_text(encoding="utf-8")), trade_date
    )


def _read_cached_szse(path: Path, trade_date: pd.Timestamp) -> pd.DataFrame:
    table = pd.read_excel(path, engine="openpyxl", dtype="string")
    return normalize_szse_margin_table(table, trade_date)


def audit_margin_history(
    margin: pd.DataFrame,
    expected_dates: Iterable[str | pd.Timestamp],
) -> tuple[dict[str, object], pd.DataFrame]:
    expected = pd.DatetimeIndex(pd.to_datetime(list(expected_dates))).normalize().unique()
    frame = margin.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame["asset"] = frame["asset"].astype("string").str.zfill(6)
    duplicate_rows = int(frame.duplicated(["trade_date", "asset", "exchange"]).sum())
    invalid_asset_rows = int(
        (~frame["asset"].str.fullmatch(r"\d{6}", na=False)).sum()
    )
    negative_by_column = {
        column: int(frame[column].dropna().lt(0.0).sum())
        for column in NONNEGATIVE_COLUMNS
    }
    required_null_rate = {
        column: float(frame[column].isna().mean())
        for column in [
            "trade_date",
            "asset",
            "exchange",
            "financing_balance",
            "financing_buy",
            "short_balance_quantity",
            "short_sell_quantity",
        ]
    }
    relation_rows = frame[
        frame["total_margin_balance"].notna()
        & frame["short_balance_value"].notna()
        & frame["financing_balance"].notna()
    ]
    relation_violations = int(
        (
            relation_rows["total_margin_balance"]
            + 1.0
            < relation_rows["financing_balance"]
            + relation_rows["short_balance_value"]
        ).sum()
    )

    date_exchange = (
        frame.groupby(["trade_date", "exchange"])
        .agg(rows=("asset", "size"), assets=("asset", "nunique"))
        .reset_index()
    )
    expected_grid = pd.MultiIndex.from_product(
        [expected, ["SSE", "SZSE"]], names=["trade_date", "exchange"]
    ).to_frame(index=False)
    date_exchange = expected_grid.merge(
        date_exchange,
        on=["trade_date", "exchange"],
        how="left",
        validate="one_to_one",
    )
    date_exchange[["rows", "assets"]] = date_exchange[["rows", "assets"]].fillna(0)
    missing_source_dates = date_exchange.loc[
        date_exchange["rows"].eq(0), ["trade_date", "exchange"]
    ]
    ready = bool(
        duplicate_rows == 0
        and invalid_asset_rows == 0
        and not any(negative_by_column.values())
        and relation_violations == 0
        and missing_source_dates.empty
        and required_null_rate["financing_balance"] == 0.0
        and required_null_rate["financing_buy"] == 0.0
    )
    audit = {
        "status": "admitted" if ready else "rejected",
        "grain": "one row per exchange, trade_date, and security",
        "rows": int(len(frame)),
        "assets": int(frame["asset"].nunique()),
        "date_min": frame["trade_date"].min().strftime("%Y-%m-%d"),
        "date_max": frame["trade_date"].max().strftime("%Y-%m-%d"),
        "expected_snapshot_dates": int(len(expected)),
        "observed_snapshot_dates": int(frame["trade_date"].nunique()),
        "duplicate_primary_keys": duplicate_rows,
        "invalid_asset_rows": invalid_asset_rows,
        "negative_values": negative_by_column,
        "required_null_rate": required_null_rate,
        "szse_balance_identity_violations": relation_violations,
        "missing_source_dates": [
            {
                "trade_date": row.trade_date.strftime("%Y-%m-%d"),
                "exchange": row.exchange,
            }
            for row in missing_source_dates.itertuples(index=False)
        ],
    }
    return audit, date_exchange


def download_official_margin_snapshots(
    dates: Iterable[str | pd.Timestamp],
    *,
    raw_directory: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    pause_seconds: float = 0.10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    snapshot_dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().unique()
    snapshot_dates = snapshot_dates.sort_values()
    raw_root = Path(raw_directory)
    output = Path(output_path)
    manifest_output = Path(manifest_path)
    raw_root.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    raw_files: list[dict[str, object]] = []
    with requests.Session() as session:
        for trade_date in snapshot_dates:
            date_text = trade_date.strftime("%Y%m%d")
            sse_path = raw_root / "sse" / f"{date_text}.json"
            szse_path = raw_root / "szse" / f"{date_text}.xlsx"
            sse_path.parent.mkdir(parents=True, exist_ok=True)
            szse_path.parent.mkdir(parents=True, exist_ok=True)

            if sse_path.exists():
                sse = _read_cached_sse(sse_path, trade_date)
            else:
                sse, raw = fetch_sse_margin_detail(trade_date, session=session)
                sse_path.write_bytes(raw)
                time.sleep(pause_seconds)
            if szse_path.exists():
                szse = _read_cached_szse(szse_path, trade_date)
            else:
                szse, raw = fetch_szse_margin_detail(trade_date, session=session)
                szse_path.write_bytes(raw)
                time.sleep(pause_seconds)
            if sse.empty or szse.empty:
                raise ValueError(f"Empty exchange response for {trade_date.date()}")
            frames.extend([sse, szse])
            raw_files.extend(
                [
                    {
                        "exchange": "SSE",
                        "trade_date": trade_date.strftime("%Y-%m-%d"),
                        "path": sse_path.as_posix(),
                        "bytes": sse_path.stat().st_size,
                        "sha256": sha256_file(sse_path),
                    },
                    {
                        "exchange": "SZSE",
                        "trade_date": trade_date.strftime("%Y-%m-%d"),
                        "path": szse_path.as_posix(),
                        "bytes": szse_path.stat().st_size,
                        "sha256": sha256_file(szse_path),
                    },
                ]
            )

    margin = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["trade_date", "asset", "exchange"], keep="last")
        .sort_values(["trade_date", "exchange", "asset"])
        .reset_index(drop=True)
    )
    audit, date_exchange = audit_margin_history(margin, snapshot_dates)
    if audit["status"] != "admitted":
        raise ValueError(f"Margin data admission failed: {audit}")
    margin.to_parquet(output, index=False)
    date_audit_path = manifest_output.with_name(
        f"{manifest_output.stem}_date_coverage.csv"
    )
    date_exchange.to_csv(date_audit_path, index=False, encoding="utf-8-sig")
    manifest = {
        "dataset": "official_sse_szse_month_end_margin_snapshots",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_urls": {
            "SSE": SSE_DETAIL_URL,
            "SZSE": SZSE_DETAIL_URL,
        },
        "output_path": output.as_posix(),
        "output_sha256": sha256_file(output),
        "date_coverage_path": date_audit_path.as_posix(),
        "date_coverage_sha256": sha256_file(date_audit_path),
        "audit": audit,
        "raw_files": raw_files,
    }
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return margin, manifest


__all__ = [
    "MARGIN_COLUMNS",
    "SSE_DETAIL_URL",
    "SZSE_DETAIL_URL",
    "audit_margin_history",
    "download_official_margin_snapshots",
    "fetch_sse_margin_detail",
    "fetch_szse_margin_detail",
    "normalize_sse_margin_payload",
    "normalize_szse_margin_table",
    "sha256_file",
]
