# Data contract

Raw market data is not redistributed. Discovery and confirmation expect a structural
Parquet plus one Tencent-hfq CSV per stock and a Shenwan history Parquet.

## Structural daily Parquet

Required columns:

| Column | Meaning |
|---|---|
| `date`, `asset` | Trading date and six-digit stock code |
| `volume`, `amount` | TongdaXin daily volume and amount |
| `float_shares`, `float_market_cap` | Point-in-time float shares and float capitalization |
| `turnover_fraction` | `volume / float_shares` |
| `is_member` | Point-in-time membership in the monthly research bucket |
| `tradestatus`, `is_st` | Point-in-time tradeability and ST filters where available |

The committed experiments used stable asset-hash buckets A and B, 500 members per month,
excluding same-month CSI 300 and CSI 500 constituents. The asset unions must be disjoint.

## Tencent backward-adjusted CSVs

Each `{asset}.csv` requires `date,asset,open,high,low,close`. Files must cover the rolling
formation history and the delayed execution window. `load_hfq_prices` fails closed if any
requested stock file is absent.

## Shenwan history

Required columns are `asset,start_date,industry_l1`. Alignment uses the most recent
`start_date <= formation_date`; forward industry assignment is forbidden.

## Missing execution observations

- Search up to five market sessions for entry and exit opens.
- If entry is unavailable after the registered delay and the calendar is complete, hold cash.
- If entry occurred but exit open is unavailable, use the last available close through the
  registered exit-delay cutoff.
- Calendar-truncated observations remain missing and cannot define the formation sample.

The input hashes used for the published result are in
[`results/material_passport.yaml`](../results/material_passport.yaml).

## Official margin-financing history

V48-V56 use a locally cached, non-redistributed Parquet assembled directly from the
SSE and SZSE published security-level margin tables. Its grain is one exchange,
trading date, and six-digit security code. Required fields are financing balance,
financing purchases, short balance quantity, short-sale quantity, and the source
exchange. Balance and purchase fields must be non-negative. Repayment fields remain
signed because exchange adjustments can make them negative; no registered candidate
uses a repayment field.

The admitted weekly-plus-month-end snapshot contains 1,084,719 rows, 4,483 assets,
and all 360 expected dates from 2020-01-03 through 2026-05-29. Its SHA-256 is
`0647e540f1598853ad9919883e4b58b01d9ee8555f6ff1042e677ce8bf4b360e`.

## Unseen C/D asset buckets

The V54 continuation excludes the asset union of all thirteen earlier real panels
before selecting any stock. The 479 remaining local TDX securities are divided by
`sha256(asset) mod 2`:

- C: 255 assets, panel SHA-256 `dec98133...eba6c39`;
- D: 204 assets, panel SHA-256 `e612e986...69c29a`;
- C/D overlap: zero; overlap with all 4,755 previously used assets: zero.

Tencent HFQ prices, local TDX amount/volume, and local TDX `gbbq` float-share
histories have 100% source coverage in both panels. C and D outcomes were not opened
because V54-V56 did not earn validation permission.
