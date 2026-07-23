# Statistical validation and fallacy audit

## Verdict

The registered SADA implementation passes both the adaptive A research gate and the
asset-disjoint B historical replication gate. The numerical result is reproducible and
economically positive after the registered 20 bp one-way cost. Overall deployment
confidence remains **CAUTION**, not SOLID, because the B universe was used in an earlier
project, the sample is conditioned on liquidity/ex-index status, and January-May 2026 is
negative.

## Numerical checks

- 45 directional tests were included in cumulative BH-FDR; SADA discovery q=`6.17e-14`.
- B mean rank IC=`0.12220`; Bartlett-HAC(3) t=`9.563`; one-sided p=`5.72e-22`.
- B top-100 mean net active return=`0.5373%`/month; annualized monthly IR=`0.872`.
- Return coverage and selected-return coverage are both 100%.
- Mean traded notional is `1.1151` per month (approximately 55.8% conventional one-way
  turnover when buys and sells are halved).
- At 50 bp one-way cost, mean net active return remains `0.2028%`/month and IR=`0.329`.
- Maximum B absolute size/size-squared/industry residual is `2.14e-14`.
- A and B output bundles each reproduced byte-for-byte in a full rerun (10/10 files).

## Fallacy scan

Coverage: **11/11 checked**.

| Fallacy | Assessment |
|---|---|
| Simpson's paradox | CAUTION. Six years are positive, but the pooled result hides a negative partial 2026; yearly results are reported explicitly. |
| Ecological fallacy | Clear. Unit of analysis and inference is the stock-month; no investor-level inference is made. |
| Berkson's paradox | CAUTION. Both buckets condition on top trailing liquidity and ex-index status. |
| Collider bias | CAUTION. Eligibility can depend on liquidity and trading state related to both signals and returns; no causal interpretation is made. |
| Base-rate neglect | Not applicable; this is not a classification or diagnostic-accuracy study. |
| Regression to the mean | CAUTION. Asset-disjoint replication helps, but adaptive A research and a reused B universe cannot eliminate favorable-period selection. |
| Survivorship bias | CAUTION. Monthly historical universes and available delisted histories are retained, but historical ST coverage and vendor availability are imperfect. |
| Look-elsewhere effect | Addressed but not erased. All 45 viewed directional tests enter cumulative BH; every failed round is retained; B evaluates one selected formula only. |
| Garden of forking paths | CAUTION. V1-V6 are adaptive, but each round has a time-ordered frozen protocol. Formula and gates did not change after SADA saw B. |
| Correlation versus causation | Clear boundary. Results are described as historical prediction/association, not a causal effect. |
| Reverse causality | CAUTION. Formation inputs precede returns, but latent risk, attention, and market regimes can affect both. |

## Holdout-purity amendment

The frozen confirmation configuration used the phrase “untouched confirmation.” That was
true only with respect to evaluating the frozen SADA formula in this repository. The same
bucket-B universe had been inspected for other factors in an earlier project. The frozen
file and its original SHA are preserved for provenance; this audit supersedes the overly
broad wording. The correct evidence label is **asset-disjoint historical replication of a
pre-specified formula**, not a globally pristine external holdout.

## Remaining implementation limits

- The benchmark is an equal-weight eligible-universe benchmark, not an official index.
- Flat transaction cost does not model nonlinear market impact, price-limit queues,
  capacity, or a full Chinese fee/tax schedule.
- A stale-close exit rule is explicit but simplified; 16 of 38,466 B outcomes used it.
- 2025 IC remained positive but that year's net active return was slightly negative.
- January-May 2026 had negative IC and sharply negative net active return.
- Exact-formula novelty was not established by an exhaustive global literature search.
