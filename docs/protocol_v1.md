# Temporal participation factor discovery protocol v1

## Status and scope

This protocol was written before any candidate in this registry was evaluated.
The project is an exploratory factor-discovery study followed by a one-shot,
asset-disjoint confirmation. It does not claim that a formula is globally novel merely
because an exact web-search match was not found.

The economic object is not average turnover. It is how a fixed amount of trading is
distributed through time and whether unusually active days carry unusually positive or
negative returns. Average turnover, turnover volatility, reversal, return volatility,
size, and industry are removed before evaluation.

## Point-in-time inputs

For stock `i` and trading day `d`:

- `tau(i,d) = TDX volume(i,d) / point-in-time float shares(i,d)`.
- `x(i,d)` is the Tencent-hfq close-to-close log return minus the same-day median
  log return of the eligible research universe.
- All rolling windows end on the formation date. No forward return enters a factor,
  neutralization regression, sample filter, or portfolio target.

## Frozen candidates

For a window of `h` sessions, define `p(i,d) = tau(i,d) / sum(tau(i,s))` inside the
window.

1. `diffuse_turnover_h = -log(h * sum(p(i,d)^2))`, for `h in {20,60}`.
   It equals zero for perfectly even participation and becomes more negative as trading
   concentrates in fewer sessions. The hypothesis is that diffuse participation is less
   exposed to episodic attention and earns higher subsequent returns.

2. `turnover_burst_reversal_h`, for `h in {20,60}`. Let
   `q(i,d) = tau(i,d)^2 / sum(tau(i,s)^2)`. Then
   `turnover_burst_reversal_h = -(sum(q(i,d)*x(i,d)) - mean(x(i,d)))`.
   Squared-turnover weights isolate the return carried by participation bursts; subtracting
   the ordinary mean return removes plain reversal. The hypothesis is that burst-driven
   buying reflects transient attention while burst-driven selling reflects temporary
   pressure, so the signed burst component reverses.

3. `attention_normalization_20_60 = log(60*HHI_60) - log(20*HHI_20)`.
   A higher score means participation has recently become more diffuse relative to its
   quarterly pattern. The hypothesis is that normalization after an attention burst is
   followed by better returns.

4. `float_supply_days_120 = -max(float_shares_t - float_shares_{t-120}, 0) /
   mean(volume_{t-19:t})`. This measures newly tradable supply in days of normal volume.
   The negative orientation represents an ex-ante supply-overhang hypothesis. Because
   published evidence on free-float increases is mixed, this candidate receives no
   preferred status.

Six candidates constitute one multiple-testing family. Signs and horizons may not be
changed after results are observed.

## Common-sample neutralization

At each month end, every candidate is 3-MAD winsorized and residualized by equal-weight
OLS against:

- z-scored log point-in-time float market capitalization and its centered square;
- Shenwan level-1 industry dummies;
- 20-day low turnover;
- 60-day negative log-turnover volatility;
- 20-day return reversal; and
- 60-day low volatility.

The residual is z-scored. Each candidate uses the rows where that candidate and every
control are observed. Forward outcomes are attached only after the formation panel and
its row set are frozen.

## Samples and gates

Liquid bucket A is the research sample. 2020-2022 is discovery and 2023-2024 is internal
validation. Discovery p-values use a one-sided Bartlett-HAC mean test with maximum lag
three and Benjamini-Hochberg correction across all six registered candidates.

A candidate must pass every gate in `configs/discovery_v1.yaml`. At most one candidate is
selected by the frozen tie-break. Only that selected candidate may be evaluated on
asset-disjoint liquid bucket B through May 2026. If no candidate passes, bucket B remains
unopened. If the selected candidate fails confirmation, this experiment is a failed search;
another formula may not be substituted into the opened holdout.

The economic diagnostic is a monthly top-100 equal-weight long-only portfolio. Existing
holdings remain while their factor percentile is at least 60%. The benchmark is the
equal-weight common formation universe, and intended turnover pays 20 basis points per
one-way notional.

## Known prior art and novelty boundary

Turnover, abnormal turnover, volume-conditioned reversal, turnover volatility, entropy,
and free-float changes all have prior literature. The potentially differentiating object
here is the exact within-stock temporal-concentration residual after simultaneous removal
of the listed baseline factors. A successful result would support incremental predictive
information in this implementation, not universal originality or causality.
