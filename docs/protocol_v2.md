# Active gap absorption discovery protocol v2

## Research status

V2 is an adaptive research round written after V1 failed its full research gate and
before any V2 outcome was computed. Bucket A is therefore no longer an untouched
holdout. Bucket B remains asset-disjoint and unopened. V2 p-values are corrected in one
Benjamini-Hochberg family together with all six V1 discovery p-values.

Overnight-to-intraday reversal is established prior art. V2 does not claim invention of
that economic idea. It tests whether a precise, turnover-surprise-weighted absorption
measure contains incremental information after controls that include the principal V1
temporal-participation factors. Relevant prior art includes Lou, Polk and Skouras,
“A tug of war: Overnight versus intraday expected returns” (JFE, 2019), and Berkman et
al., “Overnight returns, daytime reversals, and future stock returns” (JFE, 2022,
doi:10.1016/j.jfineco.2021.09.019).

## Point-in-time primitives

For stock `i` on day `d`, all prices are Tencent backward-adjusted OHLC:

- `g(i,d) = log(open(i,d) / close(i,d-1)) - median_i(g(i,d))`;
- `u(i,d) = log(close(i,d) / open(i,d)) - median_i(u(i,d))`;
- `tau(i,d)` is TongdaXin volume divided by point-in-time float shares;
- `a(i,d) = max(log(tau(i,d) / median(tau(i,d-59:d))), 0)`.

Same-day cross-sectional medians use only the eligible bucket-A formation universe.
Every rolling window ends on the formation date.

## Frozen candidates

For `h` in `{20, 60}`:

1. `active_positive_gap_reversal_h` is
   `sum(a * min(max(g,0), max(-u,0))) / sum(a * max(g,0))` over `h` days.
   A higher value means more positive overnight gaps were actively faded intraday. The
   positive orientation follows the disagreement/reversal prediction in prior evidence.

2. `active_negative_gap_recovery_h` is
   `sum(a * min(max(-g,0), max(u,0))) / sum(a * max(-g,0))` over `h` days.
   A higher value means more negative overnight gaps were actively recovered. The
   hypothesis is persistent latent demand in the A-share setting; international evidence
   is mixed, so the sign is fixed but not privileged.

3. Define daily close location `c = (close-low)/(high-low)-0.5`. Then
   `burst_close_strength_h = sum(tau^2*c)/sum(tau^2) - mean(c)`.
   It isolates whether unusually active days close relatively high while subtracting
   ordinary close-location strength. The positive orientation represents persistent
   informed demand rather than transient price pressure.

The six formulas and signs may not be changed after V2 outcomes are observed.

## Incremental neutralization and gates

Each monthly candidate is 3-MAD winsorized and residualized by equal-weight OLS against
log float size and its square, Shenwan L1 dummies, low turnover, turnover volatility,
20-day reversal, 60-day volatility, and V1's 20/60-day turnover diffuseness and
turnover-burst return components. Exact transformed regressors are retained so numerical
orthogonality can be audited.

Discovery is 2020-2022 and internal validation is 2023-2024. All gates and the top-100
buffered portfolio with 20 bp one-way costs remain unchanged from V1. A V2 candidate can
be selected only if it passes every gate after cumulative 12-candidate BH correction.
Until then bucket B cannot be loaded by the discovery script.
