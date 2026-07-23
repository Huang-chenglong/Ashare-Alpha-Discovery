# Dynamic float-supply inventory protocol v3

## Status and novelty boundary

V3 was frozen after V1 and V2 results were known but before any V3 outcome was computed.
It is adaptive research on bucket A; only bucket B can provide confirmation. All 18
directional tests from V1 through V3 share one Benjamini-Hochberg family.

Lockup expiration, tradable-share supply, and liquidity effects have substantial prior
literature. For example, Fang et al. study staggered changes in tradable supply during
China's split-share reform (Journal of Banking & Finance, 2017,
doi:10.1016/j.jbankfin.2017.08.012). The potentially differentiating element here is the
exact recursive inventory proxy driven by observed point-in-time float changes and
stock-specific realized turnover. Even if confirmed, the result would establish an
incremental implementation, not universal conceptual originality or causality.

## Frozen state equation

Let `F(i,d)` be point-in-time float shares and `tau(i,d)` daily turnover fraction. Define
new supply as a fraction of current float:

`s(i,d) = max(F(i,d)-F(i,d-1), 0) / F(i,d)`.

For absorption scale `k` in `{0.25, 0.50, 1.00}`, initialize inventory at zero at the
beginning of each stock's available history and update after each trading day:

`I_k(i,d) = (I_k(i,d-1) + s(i,d)) * exp(-tau(i,d)/k)`.

Thus a float increase adds supply, and subsequent turnover decays it. The scales mean
that one cumulative turn of `k` leaves `exp(-1)` of the inventory, absent another event.
The update is point-in-time, deterministic, and contains no estimated return coefficient.

The six candidates are:

- `remaining_float_inventory_fraction_k = -log(1 + I_k)`;
- `remaining_float_inventory_days_k = -log(1 + I_k / mean_20(tau))`.

Higher values always mean less unabsorbed supply. The days variant conditions the same
inventory on the stock's current normal absorption capacity. All signs and scales are
fixed before outcome evaluation.

## Incremental test

At each month end a candidate is 3-MAD winsorized and residualized against log float size
and its centered square, Shenwan L1 industries, 20-day low turnover, 60-day turnover
volatility, 20-day reversal, 60-day return volatility, and V1's static
`float_supply_days_120`. Exact transformed regressors are retained for orthogonality QA.

The 2020-2022 discovery, 2023-2024 validation, open-to-open label, execution delays,
top-100 buffered portfolio, 20 bp one-way cost, gates, and tie-break are unchanged. The
discovery runner accepts no confirmation data path.
