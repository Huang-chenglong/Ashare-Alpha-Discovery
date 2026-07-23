# Upside-tail concentration protocol v4

V4 was frozen after V1-V3 results and before V4 outcome evaluation. Bucket A is adaptive
research; bucket B remains unopened. Benjamini-Hochberg correction covers all 24 candidates
tested through V4.

The negative MAX/lottery effect is established in China. Nartea, Kong and Wu report that
stocks with high prior-month maximum daily returns subsequently underperform (Journal of
Banking & Finance, 2017, doi:10.1016/j.jbankfin.2016.12.008). Yao et al. show that price
limits complicate MAX measurement (International Review of Economics & Finance, 2021,
doi:10.1016/j.iref.2021.01.014). V4 therefore does not test or claim MAX as new. It asks
whether the *temporal concentration* and abnormal-turnover alignment of the entire upside
variance distribution add information after MAX, skewness, volatility and turnover controls.

For daily market-median-adjusted log return `x`, define `x+ = max(x,0)`, turnover `tau`,
and positive activity surprise `a=max(log(tau/median_60(tau)),0)`. For `h` in `{20,60}`:

- `diffuse_upside_variance_h = -log(h * sum(x+^4) / sum(x+^2)^2)`. Higher means upside
  variance is spread through time rather than concentrated in lottery-like jumps.
- `low_active_upside_variance_share_h = -sum(a*x+^2)/sum(a*x^2)`. Higher means unusually
  active days are less dominated by upside jumps.
- `low_turnover_upside_amplification_h` is the negative difference between the
  `tau^2`-weighted and equal-time means of `x+^2`, divided by equal-time mean `x^2`.

All signs and horizons are fixed. Monthly OLS removes size and size squared, Shenwan L1,
low turnover, turnover volatility, reversal, return volatility, 20/60-day MAX,
20/60-day realized skewness, and 20/60-day turnover diffuseness. Forward returns, costs,
research gates, and the no-access rule for bucket B are unchanged.
