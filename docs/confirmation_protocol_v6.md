# Frozen one-shot confirmation protocol

SADA was the only V6 candidate and passed every registered A-sample research gate. This
document and `configs/confirmation_v6.yaml` were written before bucket B was loaded by a
factor evaluation. Bucket B uses a stable asset-hash partition disjoint from A, the same
500-stock monthly construction and the same exclusion of same-month CSI 300/500 members.

The confirmation script first asserts that the union of A asset identifiers and the union
of B asset identifiers have zero intersection. It then independently recomputes every
input, SADA component, supply state, monthly size/industry residual and execution return
inside B. No score, coefficient, cross-sectional median, industry regression, benchmark,
or holding state is carried from A.

The only evaluated candidate is `supply_aware_defensive_attention`, unchanged from V6.
The period is January 2020 through May 2026. Passing requires at least 75 monthly ICs,
mean rank IC at least 0.015, one-sided HAC p at most 0.05, at least five positive calendar
years including 2025, positive mean net active return, and annualized monthly net IR at
least 0.30. The same top-100 buffer and 20 bp one-way cost apply.

If any condition fails, the candidate is rejected and B cannot be used to choose a
replacement. If all pass, the correct claim is “confirmed on one asset-disjoint historical
sample under the registered implementation,” not live profitability, causality, or a
globally new academic anomaly.
