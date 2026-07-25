# Margin and order-book modality audit: V48–V56

## Bottom line

No new factor has passed every frozen statistical and economic gate. The strongest
result is `hgb_tail_v53`: it passed adaptive A construction and independently
replicated its rank relationship in asset- and future-time-disjoint B, but its
frozen long-only implementation was slightly negative after costs. The repository
therefore does not label it an effective factor.

## Data admission

The official SSE/SZSE weekly-plus-month-end margin file has:

- 1,084,719 security-date rows and 4,483 assets;
- 360/360 expected snapshot dates, 2020-01-03 to 2026-05-29;
- zero duplicate primary keys, missing dates, invalid codes, core negative fields,
  required nulls, or SZSE balance-identity violations;
- one signed negative SSE short-repayment adjustment, retained for audit but unused
  by every registered factor.

Local TongdaXin contains 4,788 Shanghai, 4,278 Shenzhen, and 333 Beijing daily files,
but zero historical `minline` or `fzline` files. A Level-2 plugin configuration is
not a historical order-book archive. The project therefore does not relabel OHLCV
or current quote snapshots as order-book data.

## Registered results

| Version | New information | Best construction result | Decision |
|---|---|---|---|
| V48 | Month-end financing balance/flow interactions | `mcar_v48` IC 0.01629, but 2021 and net return failed | A failed; B unread |
| V49 | Persistent monthly financing interaction | `mcar_stable3_v49` IC 0.01336, IR 0.050 | A failed; B unread |
| V50 | Four-snapshot financing-balance path | `mpbr_v50` IC 0.01872, net −0.1552%/month | A failed; B unread |
| V51 | Persistent weekly path | best IC 0.01471, net −0.3836%/month | A failed; B unread |
| V52 | Regularized nonlinear regression | best IC 0.01369, all net returns negative | A failed; B unread |
| V53 | Direct financing-buy/short-path tail classifier | A IC 0.02186, q 0.0431, net +0.1506%/month, IR 0.320 | A passed |
| V53-B | Frozen `hgb_tail_v53`, 2,124 zero-overlap assets, 2024–2026 | IC 0.02627, p 0.00957, all years positive; net −0.0497%/month, IR −0.094 | Independent confirmation failed |
| V54 | Financing-only signed-tail spread with 8% borrow fee | best IC 0.02520; every net spread negative | A failed; C/D unread |
| V55 | Financing-only top-decile long portfolio | IC 0.01213, net −0.6827%/month | A failed; C/D unread |
| V56 | Native missing-aware margin classifier | IC 0.01064, net −0.3625%/month | A failed; C/D unread |

All scores remove log float size, centered squared size, point-in-time Shenwan L1
industry, and registered price/volume and modality main effects. V53 maximum absolute
post-neutralization exposure was approximately `1.0e-12`.

## Why V53 is not called effective

The independent B evidence supports a predictive ranking relationship: the mean IC
is above the 0.015 floor, statistically significant, and positive in 2024, 2025,
and 2026. The same frozen gate also required positive cost-adjusted return and an
information ratio of at least 0.30. Both failed. Changing holdings, direction, or
costs after seeing B would convert confirmation into tuning, so B is closed.

## Remaining clean evidence

The data builder identified 479 TDX securities absent from every earlier real panel
and split them into C (255) and D (204), with zero overlap. Their adjusted price,
turnover, and float-share panels are complete. V54-V56 failed before authorization,
so no C or D return was read. These buckets should remain sealed for a genuinely new,
mechanism-led candidate rather than another hyperparameter variation.
