# TongdaXin financial-data extension: V32–V41

## Data contract

The project downloads 33 official TongdaXin GPCW quarterly packages from
2018-03-31 through 2026-03-31. Every package is checked against the official
manifest byte size and MD5, its ZIP CRC is tested, and exactly one DAT member is
required. The parser is binary and field-number explicit.

The actual DAT date representation is six-digit `YYMMDD`. Reports become visible
on field 314. At each formation date, the latest announced fiscal report is
selected, an older report announced later cannot roll state backward, and
signals expire after 183 days.

This is announcement-gated but not archived-vintage data. The packages were
downloaded on 2026-07-23, so later vendor corrections or restatements cannot be
excluded. This limitation prevents a claim of perfectly point-in-time
fundamental data.

## Frozen candidates

| Version | Exact interaction | Discovery | Validation | Decision |
|---|---|---|---|---|
| V32 `diba_v32` | positive free-float expansion × institutional-breadth growth × falling top-ten concentration | IC 0.00523; BH q 0.5359 | IC 0.01133; net −0.1388%/month; HAC p 0.178 | Failed |
| V33 `cspi_v33` | positive profit-ROA YoY improvement × positive operating-cash-ROA YoY improvement | IC 0.01487; BH q 0.0623 | IC −0.00715; net −0.1677%/month; HAC p 0.813 | Failed |
| V34 `clpi_v34` | positive prior-quarter cash-ROA improvement × positive current profit-ROA improvement | IC 0.01114; BH q 0.1985 | IC −0.00127; net +0.0564%/month; IR 0.216 | Failed |
| V35 `sfgr_v35` | positive revenue growth × working-capital-intensity release × positive cash margin | IC −0.00383; BH q 0.8961 | IC 0.01504; net −0.2073%/month; HAC p 0.125 | Failed |
| V36 `fsim_v36` | fixed cross-fitted nonlinear map of 19 financial and 33 price/structure states | Asset-fold IC −0.00172; BH q 0.8182 | IC 0.01011; net −0.0872%/month; HAC p 0.156 | Failed |
| V37 `clua_v37` | cash improvement leading profit × positive cash ROA × low five-session announcement attention | IC 0.00730; only 25 months | IC −0.01976; net −0.5024%/month; IR −1.90 | Failed |
| V38 `cora_v38` | faster same-quarter reporting × positive cash improvement × positive cash ROA | IC 0.00666; BH q 0.3531 | IC 0.00304; net −0.4366%/month; IR −1.47 | Failed |
| V39 `cfwa_v39` | customer-financing growth × receivables/inventory release × revenue growth | IC 0.01098; BH q 0.1875 | IC 0.00512; net −0.2160%/month; HAC p 0.280 | Failed |
| V40 `cfma_v40` | customer-financing growth × quarterly acceleration × revenue growth | IC 0.01616; BH q 0.1018 | IC 0.00277; net +0.1732%/month; HAC p 0.411 | Failed |
| V41 `scfc_v41` | 50% frozen structural-model rank × 50% customer-financing-momentum rank | Adaptive E IC 0.00903; HAC p 0.194; net −0.1421%/month | One-shot F IC 0.01028; HAC p 0.0299; net −0.2622%/month; IR −0.624 | Failed both gates |

All V32–V41 scores remove their ranked components, raw component changes, related
levels and ratios, fourteen registered price/volume factors, log float market
capitalization, centered squared size, and point-in-time Shenwan L1 industry.
V33–V35 exclude financial-industry codes 48 and 49 before ranking.

## Prior-art boundary

Primitive mechanisms are prior art: ownership breadth, issuance/supply,
cash-based profitability, accrual quality, cash-flow surprises, and the
predictive comparison of earnings and cash flow. Only the exact registered
interaction formulas were not found in the bounded pre-outcome search. The
repository does not claim universal novelty.

Relevant primary sources:

- <https://doi.org/10.1016/S0304-405X(02)00223-4>
- <https://doi.org/10.1093/rof/rfs026>
- <https://doi.org/10.1016/j.jfineco.2016.03.002>
- <https://doi.org/10.1016/j.pacfin.2020.101336>
- <https://doi.org/10.1111/j.1468-5957.2006.01425.x>
- <https://doi.org/10.1016/j.jacceco.2021.101430>

## Decision

None of V32–V41 passed every frozen gate. V41 opened asset-disjoint bucket F
exactly once after its model, formula, hashes, portfolio, costs, and thresholds
were committed. Both its adaptive E diagnostic and F confirmation failed. F is
therefore closed to further V41 tuning. A statistically credible continuation
needs archived-vintage fundamentals, a new prospective period, or genuinely new
intraday/order-book data.
