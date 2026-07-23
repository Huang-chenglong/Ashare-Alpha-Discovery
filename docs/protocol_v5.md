# Defensive bottleneck residual protocol v5

## Claim boundary

V5 follows four failed novel-structure rounds and one explicit known-baseline diagnostic.
It was frozen before V5 outcomes. All 44 directional tests seen so far share one BH family.
Bucket B remains unopened.

MAX, low turnover, low volatility, short-term reversal, and overnight/intraday return
decomposition are prior art. V5 makes no novelty claim for those constituents. Its tested
object is the exact nonlinear *joint bottleneck residual* after every constituent main
effect, float size, nonlinear size and industry have been removed. A web/literature search
cannot prove global novelty; confirmation can establish predictive validity only for this
implementation and sample design.

## Frozen construction

At every eligible daily cross-section, each constituent is 3-MAD winsorized, z-scored,
and clipped to `[-5,5]`. Given a registered set of `m` constituent z-scores, define

`DBR = -log(mean_j(exp(-z_j)))`.

This smooth minimum is high only when no constituent is weak; one weak leg sharply lowers
the joint score. Six predeclared two-, three-, and four-way sets test whether this joint
state is more than an additive combination.

At month end, each DBR candidate is again 3-MAD winsorized and residualized against all
five constituent raw main effects simultaneously, plus log float size, centered squared
size and Shenwan L1 dummies. Therefore a passing result cannot be explained by linear
exposure to anti-MAX, low turnover, negative intraday return, low volatility, or reversal.

Periods, forward execution returns, delayed fills, buffered top-100 holdings, 20 bp
one-way cost, and research gates are unchanged. At most one V5 candidate can be selected
by the frozen tie-break, and only then may a separate confirmation script load bucket B.
