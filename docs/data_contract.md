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
