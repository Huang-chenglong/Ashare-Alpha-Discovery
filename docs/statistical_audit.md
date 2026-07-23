# Statistical validation and fallacy audit

## Verdict

The current research verdict is **NO NEW CONFIRMED FACTOR**.

Ninety-two directional candidate tests are retained in the cumulative ledger. No V20–V39 candidate passed every registered discovery, temporal-validation, yearly-consistency, cost, and information-ratio gate. Formula-new confirmation bucket F therefore remains sealed.

V19 is also invalid: it passed research bucket C but failed the one-shot asset-disjoint D confirmation with mean rank IC `-0.00093`, mean net active return `-0.1841%` per month, and IR `-0.280`.

## Multiple testing

- Every evaluated direction is retained; rejected-before-outcome ideas consume zero tests.
- Benjamini–Hochberg is recomputed over the entire cumulative family.
- V30 and V31 are learned models. Their in-training discovery statistics are optimistic by construction, so both additionally require a one-sided HAC test on the untouched-by-training 2023–2024 period.
- V32 was a frozen accounting/ownership interaction. It failed with discovery IC `0.00523`, cumulative BH q=`0.5359`, validation HAC p=`0.1781`, and negative cost-adjusted return.
- V33 passed cumulative BH but missed the discovery IC threshold and then reversed in validation: IC `-0.00715`, HAC p=`0.8125`, and negative cost-adjusted return.
- V34 also failed: discovery IC `0.01114`, cumulative BH q=`0.1985`, validation IC `-0.00127`, HAC p=`0.5698`, and IR `0.216`.
- V35 also failed: discovery IC `-0.00383`, cumulative BH q=`0.8961`, validation IC `0.01504`, HAC p=`0.1255`, negative cost-adjusted return, and IR `-0.690`.
- V36 used deterministic five-fold asset cross-fitting in discovery and still failed: discovery IC `-0.00172`, cumulative BH q=`0.8182`, validation IC `0.01011`, HAC p=`0.1556`, a negative 2023 IC, and negative cost-adjusted return.
- V37 failed coverage and direction gates: only 25 discovery and 18 validation months met the complete-sample requirement; validation IC was `-0.01976`, HAC p=`0.9602`, and IR `-1.90`.
- V38 had full month coverage but failed magnitude, yearly, HAC, cost, and IR gates: discovery IC `0.00666`, validation IC `0.00304`, HAC p=`0.2726`, and IR `-1.47`.
- V39 was positive but insufficient: discovery IC `0.01098`, cumulative BH q=`0.1875`, validation IC `0.00512`, HAC p=`0.2796`, and negative cost-adjusted return.
- V30 missed that additional gate: validation HAC p=`0.05352` versus the frozen `0.05` limit.
- No sign reversal, weight optimization, alpha search, or threshold relaxation was performed after seeing a result.

## Exposure controls

All current candidate scores are cross-sectionally residualized against point-in-time log float market capitalization, its centered square, and point-in-time Shenwan L1 industry. V20–V32 additionally remove eleven or more registered liquidity, reversal, volatility, lottery, momentum, overnight/intraday, accounting, ownership, or mechanism-specific controls.

The learned V30/V31 target is residualized against fourteen known factors before fitting. Model predictions are residualized against the same fourteen factors again before IC and portfolio evaluation. Their reported validation IC therefore does not come directly from those known main effects.

## Fallacy scan

Coverage: **11/11 checked**.

| Fallacy | Assessment |
|---|---|
| Simpson's paradox | Year-level results are required and reported; pooled means cannot override a negative required year. |
| Ecological fallacy | Inference is limited to stock-month ranking, not investors or firms' causal behavior. |
| Berkson's paradox | CAUTION: the liquid ex-index universe is selected on trading availability and liquidity. |
| Collider bias | CAUTION: eligibility and complete-control filtering can depend on variables related to signals and returns. |
| Base-rate neglect | Not applicable; this is not a diagnostic classifier. |
| Regression to the mean | Addressed with temporal and asset-disjoint stages, but adaptive reuse of E remains a limitation. |
| Survivorship bias | Historical monthly membership and available delisted histories are used; vendor and ST-history gaps remain. |
| Look-elsewhere effect | Addressed with a cumulative 92-test family and retained failures, not erased. |
| Garden of forking paths | Each version is committed before outcomes; adaptive motivation is disclosed. |
| Correlation versus causation | No causal claim is made. |
| Reverse causality | Formation variables precede labels, but latent state can drive both and remains a limitation. |

## Data and execution limitations

- E/F are asset-disjoint but were used by an earlier project for a different formula.
- Historical `is_st` is incomplete; the reconstructed panel currently uses `is_st=0`.
- Flat 20bp one-way cost omits nonlinear impact, limit queues, capacity, and the full fee/tax schedule.
- The equal-weight eligible-universe benchmark is not an official index.
- Tencent HFQ prices and reconstructed TongdaXin float history are vendor-derived rather than exchange-certified point-in-time databases.
- GPCW quarterly packages were downloaded in 2026. Announcement-date gating is enforced, but archived vintage values are unavailable, so later corrections/restatements cannot be excluded.
- The repeated 2023–2024 inspections mean it is now adaptive research data, not a pristine holdout for new manual variants.

## Why research stops here

Continuing to alter formulas against the same E validation interval would convert validation into training and invalidate nominal p-values. A credible next stage requires at least one of:

1. new asset- and formula-unseen history;
2. point-in-time fundamentals/shareholder data with a new frozen hypothesis;
3. intraday or order-book data, which are absent from the current TongdaXin installation;
4. a prospective paper-trading period fixed before observations arrive.
