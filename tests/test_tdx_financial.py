from __future__ import annotations

import hashlib
import struct
import zipfile

import pandas as pd
import pytest

from ashare_alpha.tdx_financial import (
    FIELDS_V32,
    FinancialPackage,
    parse_financial_manifest,
    parse_financial_package,
    select_financial_packages,
    validate_financial_package,
)


def _fake_package(tmp_path):
    field_count = 320
    report_size = field_count * 4
    header_size = struct.calcsize("<hIH3I")
    stock_size = struct.calcsize("<6scI")
    record_offset = header_size + stock_size
    values = [0.0] * field_count
    expected = {
        239: 25_000_000.0,
        242: 12_345.0,
        246: 78.0,
        247: 9_000_000.0,
        264: 4_000_000.0,
        266: 20_000_000.0,
        # TongdaXin stores announcement dates as YYMMDD floats in real GPCW
        # files, even though some third-party descriptions show YYYYMMDD.
        314: 260428.0,
    }
    for field_number, value in expected.items():
        values[field_number - 1] = value
    payload = (
        struct.pack("<hIH3I", 1, 20260331, 1, 0, report_size, 0)
        + struct.pack("<6scI", b"600000", b"\x00", record_offset)
        + struct.pack(f"<{field_count}f", *values)
    )
    path = tmp_path / "gpcw20260331.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("gpcw20260331.dat", payload)
    return path


def test_manifest_parse_and_period_selection() -> None:
    content = "\n".join(
        [
            "gpcw20260331.zip,0123456789abcdef0123456789abcdef,123",
            "gpcw20251231.zip,abcdef0123456789abcdef0123456789,456",
        ]
    )
    parsed = parse_financial_manifest(bytearray(content.encode("utf-8")))
    assert [row.filename for row in parsed] == [
        "gpcw20260331.zip",
        "gpcw20251231.zip",
    ]
    selected = select_financial_packages(
        parsed, start="2025-01-01", end="2025-12-31"
    )
    assert [row.filename for row in selected] == ["gpcw20251231.zip"]


def test_manifest_rejects_bad_digest() -> None:
    with pytest.raises(ValueError, match="Invalid MD5"):
        parse_financial_manifest("gpcw20260331.zip,not-an-md5,123")


def test_parse_financial_package_reads_registered_fields(tmp_path) -> None:
    path = _fake_package(tmp_path)
    frame = parse_financial_package(path)
    assert list(frame["asset"]) == ["600000"]
    assert frame.loc[0, "report_date"] == pd.Timestamp("2026-03-31")
    assert frame.loc[0, "report_announcement_date"] == pd.Timestamp("2026-04-28")
    for field_number, column in FIELDS_V32.items():
        if field_number != 314:
            assert pd.notna(frame.loc[0, column])


def test_parse_financial_package_accepts_a_generic_field_map(tmp_path) -> None:
    path = _fake_package(tmp_path)
    frame = parse_financial_package(path, field_map={242: "holders"})
    assert frame.loc[0, "holders"] == 12_345.0


def test_validate_financial_package_checks_md5_and_zip(tmp_path) -> None:
    path = _fake_package(tmp_path)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    package = FinancialPackage(
        filename=path.name,
        md5=digest,
        filesize=path.stat().st_size,
        report_date=pd.Timestamp("2026-03-31"),
    )
    validate_financial_package(path, package)
    wrong = FinancialPackage(
        filename=path.name,
        md5="0" * 32,
        filesize=path.stat().st_size,
        report_date=pd.Timestamp("2026-03-31"),
    )
    with pytest.raises(ValueError, match="MD5 mismatch"):
        validate_financial_package(path, wrong)
