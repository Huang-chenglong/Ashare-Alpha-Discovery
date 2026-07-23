# Supply-Aware Defensive Attention (SADA) protocol v6

## What is and is not new

SADA is a new deterministic engineering composite in this repository, not a claim that
its economic primitives are unknown to finance. Anti-MAX, low turnover, negative intraday
return, low volatility and short-term reversal are established signals. V3 supplied the
new recursive float-inventory state, which by itself predicted ranks but did not support a
profitable long-only top tail. V6 uses that state only as an ex-ante risk penalty.

This protocol was frozen after 44 prior tests and before V6 outcomes. Its discovery p-value
joins all prior tests in a 45-hypothesis BH family. The formula has no fitted return
coefficient and no hyperparameter selected from V6 outcomes.

## Formula

At each eligible daily cross-section, 3-MAD winsorize, z-score and clip to `[-5,5]`:

1. negative 20-day maximum market-adjusted daily return;
2. 20-day low turnover;
3. negative mean 20-day market-adjusted intraday return;
4. 60-day low return volatility; and
5. 20-day return reversal.

Their equal-weight mean is the defensive-attention core. Independently, construct the
V3 recursive supply inventory at absorption scale 0.25 and express it in 20-day normal
turnover days. After cross-sectional standardization, only negative supply scores incur a
penalty:

`SADA = mean(z_antiMAX, z_lowTurn, z_negIntra, z_lowVol, z_reversal)
        - 0.25 * max(-z_supply, 0)`.

At month end SADA is residualized against log point-in-time float market capitalization,
its centered square, and Shenwan L1 industries, then z-scored. Constituent signals remain
by design; this is a composite, not an incremental-anomaly test.

## Selection and confirmation

The A-sample discovery/validation gates, open-to-open execution, fill delays, top-100
buffer, and 20 bp one-way costs are unchanged. V6 contains exactly one candidate. It may
open asset-disjoint bucket B only if every A gate passes. Bucket B is then evaluated once
through May 2026, with no formula, sign, weight, universe, or threshold changes.
