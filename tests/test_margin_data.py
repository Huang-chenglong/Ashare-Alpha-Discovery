from __future__ import annotations

import pandas as pd

from ashare_alpha.margin_data import (
    audit_margin_history,
    normalize_sse_margin_payload,
    normalize_szse_margin_table,
)


def test_normalize_sse_margin_payload() -> None:
    payload = {
        "result": [
            {
                "opDate": "20240102",
                "stockCode": "600000",
                "securityAbbr": "浦发银行",
                "rzye": "1000",
                "rzmre": "100",
                "rzche": "80",
                "rqyl": "20",
                "rqmcl": "5",
                "rqchl": "3",
            }
        ]
    }
    result = normalize_sse_margin_payload(payload, "20240102")
    assert result.loc[0, "asset"] == "600000"
    assert result.loc[0, "exchange"] == "SSE"
    assert result.loc[0, "financing_balance"] == 1000
    assert pd.isna(result.loc[0, "short_balance_value"])


def test_normalize_szse_margin_table() -> None:
    source = pd.DataFrame(
        {
            "证券代码": ["000001"],
            "证券简称": ["平安银行"],
            "融资买入额(元)": ["1,000"],
            "融资余额(元)": ["9,000"],
            "融券卖出量(股/份)": ["10"],
            "融券余量(股/份)": ["20"],
            "融券余额(元)": ["100"],
            "融资融券余额(元)": ["9,100"],
        }
    )
    result = normalize_szse_margin_table(source, "2024-01-02")
    assert result.loc[0, "asset"] == "000001"
    assert result.loc[0, "financing_buy"] == 1000
    assert result.loc[0, "total_margin_balance"] == 9100


def test_audit_margin_history_rejects_duplicate_keys() -> None:
    row = {
        "trade_date": pd.Timestamp("2024-01-02"),
        "asset": "600000",
        "exchange": "SSE",
        "security_name": "浦发银行",
        "financing_balance": 1000.0,
        "financing_buy": 100.0,
        "financing_repayment": 80.0,
        "short_balance_quantity": 20.0,
        "short_sell_quantity": 5.0,
        "short_repayment_quantity": 3.0,
        "short_balance_value": pd.NA,
        "total_margin_balance": pd.NA,
    }
    other = dict(row, asset="000001", exchange="SZSE", short_balance_value=10.0)
    frame = pd.DataFrame([row, row, other])
    audit, _ = audit_margin_history(frame, ["2024-01-02"])
    assert audit["status"] == "rejected"
    assert audit["duplicate_primary_keys"] == 1
    assert audit["invalid_asset_rows"] == 0
