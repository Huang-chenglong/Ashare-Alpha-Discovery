from __future__ import annotations

import hashlib
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REPORT_HEADER = struct.Struct("<hIH3I")
STOCK_HEADER = struct.Struct("<6scI")
REPORT_FILENAME = re.compile(r"^gpcw(?P<date>\d{8})\.zip$")

# TongdaXin GPCW column numbers. The audited names are taken from the maintained
# mootdx field map. A GPCW float array is zero-based, while the public field
# numbers are one-based.
FIELDS_V32 = {
    239: "listed_float_a_shares",
    242: "shareholder_count",
    246: "institution_count",
    247: "institution_shares",
    264: "top10_float_a_shares",
    266: "free_float_shares",
    314: "report_announcement_date",
}
FIELDS_V33 = {
    40: "total_assets",
    307: "ttm_operating_cash_flow",
    308: "ttm_parent_net_profit_10k",
    314: "report_announcement_date",
}
FIELDS_V35 = {
    17: "inventory",
    40: "total_assets",
    283: "ttm_revenue_10k",
    296: "receivables_and_notes",
    307: "ttm_operating_cash_flow",
    308: "ttm_parent_net_profit_10k",
    314: "report_announcement_date",
}


@dataclass(frozen=True)
class FinancialPackage:
    filename: str
    md5: str
    filesize: int
    report_date: pd.Timestamp


def parse_financial_manifest(
    content: bytes | bytearray | memoryview | str,
) -> list[FinancialPackage]:
    text = (
        bytes(content).decode("utf-8")
        if isinstance(content, (bytes, bytearray, memoryview))
        else content
    )
    output: list[FinancialPackage] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 3:
            raise ValueError(f"Malformed financial manifest line {line_number}: {line!r}")
        filename, digest, size_text = parts
        match = REPORT_FILENAME.fullmatch(filename)
        if match is None:
            raise ValueError(f"Unexpected financial package name: {filename!r}")
        if not re.fullmatch(r"[0-9a-f]{32}", digest):
            raise ValueError(f"Invalid MD5 for {filename}: {digest!r}")
        filesize = int(size_text)
        if filesize <= 0:
            raise ValueError(f"Invalid file size for {filename}: {filesize}")
        output.append(
            FinancialPackage(
                filename=filename,
                md5=digest,
                filesize=filesize,
                report_date=pd.to_datetime(match.group("date"), format="%Y%m%d"),
            )
        )
    if not output:
        raise ValueError("TongdaXin financial manifest is empty")
    if len({row.filename for row in output}) != len(output):
        raise ValueError("TongdaXin financial manifest contains duplicate filenames")
    return output


def select_financial_packages(
    packages: Iterable[FinancialPackage],
    *,
    start: str,
    end: str,
) -> list[FinancialPackage]:
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    if start_date > end_date:
        raise ValueError("Financial package start must not exceed end")
    return sorted(
        [
            package
            for package in packages
            if start_date <= package.report_date <= end_date
        ],
        key=lambda package: package.report_date,
    )


def md5_file(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_financial_package(path: str | Path, package: FinancialPackage) -> None:
    source = Path(path)
    if source.stat().st_size != package.filesize:
        raise ValueError(
            f"{package.filename} has {source.stat().st_size} bytes; "
            f"expected {package.filesize}"
        )
    digest = md5_file(source)
    if digest != package.md5:
        raise ValueError(
            f"{package.filename} MD5 mismatch: observed {digest}, expected {package.md5}"
        )
    with zipfile.ZipFile(source) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"{package.filename} contains corrupt member {bad_member}")
        dat_members = [
            member
            for member in archive.infolist()
            if not member.is_dir() and Path(member.filename).suffix.lower() == ".dat"
        ]
        if len(dat_members) != 1:
            raise ValueError(
                f"{package.filename} must contain exactly one DAT file; "
                f"found {len(dat_members)}"
            )


def _decode_report_date(value: float | int) -> pd.Timestamp:
    if not pd.notna(value):
        return pd.NaT
    integer = int(value)
    if integer <= 0:
        return pd.NaT
    digits = str(integer)
    if len(digits) == 6:
        date_format = "%y%m%d"
    elif len(digits) == 8:
        date_format = "%Y%m%d"
    else:
        return pd.NaT
    try:
        return pd.to_datetime(digits, format=date_format, errors="raise")
    except (ValueError, OverflowError):
        return pd.NaT


def parse_financial_package(
    path: str | Path,
    *,
    field_map: dict[int, str] = FIELDS_V32,
) -> pd.DataFrame:
    source = Path(path)
    with zipfile.ZipFile(source) as archive:
        dat_members = [
            member
            for member in archive.infolist()
            if not member.is_dir() and Path(member.filename).suffix.lower() == ".dat"
        ]
        if len(dat_members) != 1:
            raise ValueError(
                f"{source.name} must contain exactly one DAT file; found {len(dat_members)}"
            )
        payload = archive.read(dat_members[0])

    if len(payload) < REPORT_HEADER.size:
        raise ValueError(f"{source.name} has a truncated report header")
    _, report_date_integer, stock_count, _, report_size, _ = REPORT_HEADER.unpack_from(payload)
    if report_size <= 0 or report_size % 4:
        raise ValueError(f"{source.name} has invalid report size {report_size}")
    field_count = report_size // 4
    requested = sorted(field_map)
    if not requested or requested[0] < 1 or requested[-1] > field_count:
        raise ValueError(
            f"{source.name} has {field_count} fields but requested {requested}"
        )
    directory_end = REPORT_HEADER.size + stock_count * STOCK_HEADER.size
    if directory_end > len(payload):
        raise ValueError(f"{source.name} has a truncated stock directory")

    rows: list[dict[str, object]] = []
    for stock_index in range(stock_count):
        offset = REPORT_HEADER.size + stock_index * STOCK_HEADER.size
        raw_code, _, record_offset = STOCK_HEADER.unpack_from(payload, offset)
        record_end = record_offset + report_size
        if record_offset < directory_end or record_end > len(payload):
            raise ValueError(
                f"{source.name} has invalid record offset for stock {stock_index}"
            )
        code = raw_code.decode("ascii")
        row: dict[str, object] = {
            "asset": code.zfill(6),
            "report_date": pd.to_datetime(
                str(report_date_integer), format="%Y%m%d", errors="raise"
            ),
        }
        for field_number, name in field_map.items():
            raw_value = struct.unpack_from(
                "<f", payload, record_offset + (field_number - 1) * 4
            )[0]
            row[name] = raw_value
        if "report_announcement_date" in row:
            row["report_announcement_date"] = _decode_report_date(
                row["report_announcement_date"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def load_financial_history(
    directory: str | Path,
    *,
    field_map: dict[int, str] = FIELDS_V32,
) -> pd.DataFrame:
    root = Path(directory)
    paths = sorted(root.glob("gpcw????????.zip"))
    if not paths:
        raise FileNotFoundError(f"No GPCW packages found in {root}")
    frames = [parse_financial_package(path, field_map=field_map) for path in paths]
    history = pd.concat(frames, ignore_index=True)
    history["asset"] = history["asset"].astype("string").str.zfill(6)
    return (
        history.drop_duplicates(["asset", "report_date"], keep="last")
        .sort_values(["asset", "report_date"])
        .reset_index(drop=True)
    )
