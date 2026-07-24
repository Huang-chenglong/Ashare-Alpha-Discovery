# Research ledger: V7–V41

## Status codes

- **REJECTED PRE-OUTCOME**: prior art conflict found before looking at returns; zero tests consumed.
- **FAILED RESEARCH**: evaluated on the active research bucket but missed at least one frozen gate.
- **FAILED CONFIRMATION**: passed research but failed the one-shot asset-disjoint confirmation.
- **SEALED**: data not opened because no candidate earned confirmation access.

## Decision ledger

| Version | Mechanism | Tests | Decision | Key evidence |
|---|---|---:|---|---|
| V7 | Directional impact survival | 4 | FAILED RESEARCH | No candidate passed C gates. |
| V8 | Participation-distribution dominance | 3 | FAILED RESEARCH | `pds_120` was positive but cost-negative and below gates. |
| V9 | Directional participation dispersion | 2 | FAILED RESEARCH | Both candidates failed. |
| V10 | Joint return/participation path reversibility | 2 | FAILED RESEARCH | Both frozen horizons failed. |
| V11 | Pre-frozen 90-session midpoint | 1 | FAILED RESEARCH | Adaptive midpoint failed. |
| V12 | Directional range conversion | 2 | FAILED RESEARCH | Both candidates failed. |
| V13 | Close–VWAP tail asymmetry | 0 | REJECTED PRE-OUTCOME | WorldQuant Alpha#42 directly uses VWAP minus close. |
| V14 | VWAP state transitions | 2 | FAILED RESEARCH | Both candidates failed. |
| V15 | VWAP reclaim with volume | 0 | REJECTED PRE-OUTCOME | Exact mechanism is established trading prior art. |
| V16 | Directional volume clock | 2 | FAILED RESEARCH | Both candidates failed. |
| V17 | Consecutive range acceptance | 2 | FAILED RESEARCH | Both candidates failed. |
| V18 | PDS/path-reversibility equal consensus | 1 | FAILED RESEARCH | Positive validation but cumulative q failed. |
| V19 | Multiscale participation consensus | 1 | FAILED CONFIRMATION | C passed; D IC `-0.00093`, net `-0.1841%`/month. |
| V20 | Signed tail response quality | 2 | FAILED RESEARCH | Discovery and validation IC were negative. |
| V21 | Directional range relaxation | 2 | FAILED RESEARCH | `drra_90` validation IC `0.01581`, but discovery and IR gates failed. |
| V22 | Participation-weighted range relaxation | 2 | FAILED RESEARCH | Both discovery ICs were negative. |
| V23 | Size×turnover cohort breadth pressure | 0 | REJECTED PRE-OUTCOME | Peer Return Gap and Net Peer Momentum are direct mechanism conflicts. |
| V24 | Matched-excursion participation imbalance | 2 | FAILED RESEARCH | Validation direction reversed strongly. |
| V25 | Active stress-resilience tilt | 2 | FAILED RESEARCH | Positive IC, but q/coverage/cost gates failed. |
| V26 | Float-supply impact decay | 2 | FAILED RESEARCH | Best economic result; neither horizon passed every gate. |
| V27 | Multiscale FSID rank consensus | 1 | FAILED RESEARCH | Discovery IC `0.03615`; validation IC `-0.00235`. |
| V28 | Directional price-volume lead/lag | 0 | REJECTED PRE-OUTCOME | Direct Granger/transfer-entropy prior art. |
| V29 | Tail-shadow rejection asymmetry | 2 | FAILED RESEARCH | Discovery IC near zero/negative; costs negative. |
| V30 | Shallow nonlinear structural model | 1 | FAILED RESEARCH | Validation IC `0.01647`, net `+0.2043%`, but HAC p=`0.05352`. |
| V31 | Ridge pairwise interaction model | 1 | FAILED RESEARCH | Validation IC `0.00283`, net `-0.3421%`. |
| V32 | Distributed institutional breadth absorption | 1 | FAILED RESEARCH | Discovery IC `0.00523`, BH q `0.5359`; validation net `-0.1388%`. |
| V33 | Cash-synchronized profitability improvement | 1 | FAILED RESEARCH | Discovery IC `0.01487`; validation IC reversed to `-0.00715`. |
| V34 | Cash-led profitability improvement | 1 | FAILED RESEARCH | Discovery IC `0.01114`; validation IC `-0.00127`, IR `0.216`. |
| V35 | Self-financed growth release | 1 | FAILED RESEARCH | Discovery IC `-0.00383`; validation IC `0.01504`, but net `-0.2073%`/month and HAC p=`0.1255`. |
| V36 | Cross-fitted fundamental-structural model | 1 | FAILED RESEARCH | Asset-fold discovery IC `-0.00172`; validation IC `0.01011`, but 2023 was negative, net `-0.0872%`/month, and HAC p=`0.1556`. |
| V37 | Cash-lead underattention | 1 | FAILED RESEARCH | Only 25/18 eligible months; validation IC `-0.01976`, net `-0.5024%`/month, and IR `-1.90`. |
| V38 | Cash-confirmed reporting acceleration | 1 | FAILED RESEARCH | Discovery IC `0.00666`; validation IC `0.00304`, net `-0.4366%`/month, and IR `-1.47`. |
| V39 | Customer-funding/working-capital alignment | 1 | FAILED RESEARCH | Discovery IC `0.01098`, q `0.1875`; validation IC `0.00512`, net `-0.2160%`/month. |
| V40 | Customer-financing momentum alignment | 1 | FAILED RESEARCH | Discovery IC `0.01616`, q `0.1018`; validation IC `0.00277`, 2023 negative, HAC p=`0.4106`. |
| V41 | Structural/customer-financing consensus | 1 | FAILED RESEARCH AND CONFIRMATION | Adaptive E IC `0.00903`, HAC p=`0.1941`, net `-0.1421%`; one-shot F IC `0.01028`, net `-0.2622%`, IR `-0.624`. |
| F | Formula-new asset-disjoint confirmation | — | OPENED ONCE, THEN CLOSED | V41 used F exactly once under commit-frozen code and gates; F failed and is prohibited from V41 tuning. |

Together with the fourteen registered baselines and V1–V6 research candidates, the cumulative ledger contains **94 unique directional tests**. V41 is explicitly adaptive on E and therefore has no fabricated BH q-value.

## Original precise constructs introduced in this branch

### MEPI — Matched-Excursion Participation Imbalance

For adjacent opposite-signed market-residual returns whose net displacement is no more than 25% of total absolute displacement, MEPI compares turnover on the positive and negative legs. The event score is weighted by cancellation quality and averaged over 60/90 sessions. V24 failed temporal validation.

### ASRT — Active Stress-Resilience Tilt

Market stress is identified only from the prior 60-session distribution. ASRT measures whether a stock's positive abnormal participation concentrates on its relatively resilient stress observations, after removing ordinary stress resilience, activity, downside beta, size, industry, and known factors. V25 did not pass the complete gate.

### FSID — Float-Supply Impact Decay

After a float-share increase completes a 20-market-session observation path:

```text
early impact = sum(residual return, days 1–5) / sum(turnover, days 1–5)
late impact  = sum(residual return, days 6–20) / sum(turnover, days 6–20)
event score  = clip(late impact - early impact, -5, 5)
```

Completed event scores are supply-fraction weighted over 100/120 sessions. Controls include remaining float inventory, event size, total residual return, total turnover, raw return slope, event age, size, industry, and eleven known factors. V26 showed positive time-out behavior but did not pass every statistical gate.

### TSRA — Tail-Shadow Rejection Asymmetry

TSRA compares next-session lower-shadow recovery after bottom residual tails with next-session upper-shadow rejection after top residual tails. It controls unconditional wick geometry and V21 range relaxation. V29 failed discovery.

### NSIM/RPSI learned factors

V30 and V31 fit only 2020–2022 residual labels and evaluate 2023–2024 out of training. Known factors are removed from both the training target and final score. V30 used a fixed shallow boosted model; V31 used all fixed pairwise rank interactions with Ridge. Neither earned access to F.

### DIBA — Distributed Institutional Breadth Absorption

V32 multiplied positive cross-sectional ranks of free-float expansion, institutional-breadth growth, and falling top-ten float-holder concentration. It removed all three ranked main effects, eight additional ownership changes, fourteen known price/volume factors, size, squared size, and industry. The score was exposure-clean but failed the discovery, cumulative-FDR, validation-HAC, net-return, and IR gates. Current-vintage GPCW data were gated by report announcement date; possible later corrections remain a disclosed limitation.

### SFGR — Self-Financed Growth Release

V35 multiplied the positive ranks of revenue growth, year-over-year release of receivables-plus-inventory intensity, and operating-cash margin. It removed every component and related accounting level, fourteen registered price/volume factors, size, squared size, and industry. The validation IC was positive but the discovery direction, validation HAC test, yearly consistency, cost-adjusted return, and IR all failed. This is retained as the eighty-eighth negative test; bucket F was not read.

### FSIM — Fundamental-Structural Interaction Model

V36 used five deterministic asset folds so every 2020–2022 discovery prediction came from a model trained without that asset's labels. A final 2020–2022 model predicted 2023–2024. The score then removed linear and quadratic ranks of all 52 model inputs, 38 missingness effects, size, squared size, and industry. The asset-fold discovery IC was negative and the validation portfolio was cost-negative; the clean exposure diagnostics do not rescue the failed economic and statistical gates.

### SCFC — Structural/Customer-Financing Consensus

V41 combined the frozen V30 structural model score and V40
customer-financing-momentum interaction using fixed equal weights. Each
component was neutralized against its frozen controls and converted to a
within-month percentile rank before combination. The composite was then
neutralized against fourteen known price/volume factors, fifteen financial main
effects, log float size, squared size, and point-in-time Shenwan L1 industry.

E was not treated as out-of-sample: V30 was fitted on E during 2020–2022 and
V20–V40 outcomes had already informed the project. The frozen E gate therefore
used only 2023–2025 and was labelled an adaptive post-fit diagnostic. It failed
with 36-month IC `0.00903`, HAC p=`0.1941`, 2025 IC `-0.01129`, net
`-0.1421%`/month, and IR `-0.390`.

F contained 2,124 assets with zero code overlap with E's 2,149 assets. Its
one-shot confirmation produced 70-month IC `0.01028` and HAC p=`0.0299`, but
missed the frozen IC floor `0.015`; 2024 IC was negative, net active return was
`-0.2622%`/month, and IR was `-0.624`. The registered joint rule was
`adaptive E pass AND independent F pass`; both operands were false. Maximum
absolute post-neutralization exposure was `1.13e-14` on E and `2.71e-14` on F,
so exposure removal succeeded but did not rescue the signal.

## Prior-art boundary

Exact-formula novelty is not globally provable. The repository therefore uses bounded language: “exact construct not found in the documented pre-outcome search.” Related mechanisms include [Style Investing](https://doi.org/10.3386/w8039), [Comovement](https://doi.org/10.3386/w8895), [Momentum, Reversals, and Investor Clientele](https://doi.org/10.3386/w29453), [Trading Volume and Time Varying Betas](https://doi.org/10.1093/rof/rfab014), and [Peer Return Gap in China](https://doi.org/10.1016/j.finr.2025.100088).
