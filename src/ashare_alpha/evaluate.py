from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import CANDIDATES, CANDIDATE_COLUMNS, CONTROL_COLUMNS
from .statistics import benjamini_hochberg, hac_mean_test, mad_winsorize, zscore


def month_end_formation(daily: pd.DataFrame) -> pd.DataFrame:
    eligible = (
        daily["is_member"].fillna(False).astype(bool)
        & daily["tradestatus"].fillna(0).eq(1)
        & daily["is_st"].fillna(1).eq(0)
    )
    frame = daily.loc[eligible].copy()
    last_dates = frame.groupby(frame["date"].dt.to_period("M"))["date"].max()
    return frame[frame["date"].isin(last_dates.to_numpy())].copy()


def _neutralize_cross_section(
    cross_section: pd.DataFrame,
    candidate: str,
    control_columns: list[str] | tuple[str, ...] = CONTROL_COLUMNS,
) -> pd.DataFrame:
    required = [candidate, "float_market_cap", "industry_l1", *control_columns]
    finite = cross_section[required].replace([np.inf, -np.inf], np.nan)
    valid = finite.notna().all(axis=1) & cross_section["float_market_cap"].gt(0.0)
    rows = cross_section.loc[valid].copy()
    if len(rows) < 100:
        return rows.iloc[0:0]

    candidate_values = mad_winsorize(rows[candidate])
    log_size = np.log(rows["float_market_cap"])
    size_z = zscore(log_size)
    size_squared = size_z.pow(2) - size_z.pow(2).mean()
    controls = {
        column: zscore(mad_winsorize(rows[column])) for column in control_columns
    }
    industries = pd.get_dummies(rows["industry_l1"], drop_first=True, dtype=float)
    design = np.column_stack(
        [
            np.ones(len(rows)),
            size_z.to_numpy(),
            size_squared.to_numpy(),
            *(values.to_numpy() for values in controls.values()),
            *industries.to_numpy().T,
        ]
    )
    coefficients = np.linalg.lstsq(design, candidate_values.to_numpy(), rcond=None)[0]
    residual = pd.Series(
        candidate_values.to_numpy() - design @ coefficients,
        index=rows.index,
    )
    rows["score"] = zscore(residual)
    rows["candidate"] = candidate
    rows["size_z"] = size_z
    rows["size_squared"] = size_squared
    for column, values in controls.items():
        rows[f"neutralization_control_{column}"] = values
    return rows


def neutralized_candidate_panels(
    month_end: pd.DataFrame,
    *,
    candidate_columns: list[str] | tuple[str, ...] = CANDIDATE_COLUMNS,
    control_columns: list[str] | tuple[str, ...] = CONTROL_COLUMNS,
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    for candidate in candidate_columns:
        for _, cross_section in month_end.groupby("date", sort=True):
            neutralized = _neutralize_cross_section(
                cross_section, candidate, control_columns
            )
            if not neutralized.empty:
                output.append(neutralized)
    if not output:
        raise ValueError("No candidate/date cross-section has enough complete rows")
    return pd.concat(output, ignore_index=True).sort_values(
        ["candidate", "date", "asset"]
    )


def buffered_long_only(
    panel: pd.DataFrame,
    *,
    holdings: int = 100,
    retention_percentile: float = 0.60,
    cost_bps_one_way: float = 20.0,
) -> pd.DataFrame:
    previous_weights = pd.Series(dtype=float)
    previous_selected: tuple[str, ...] = ()
    records: list[dict[str, object]] = []
    for date, cross_section in panel.groupby("date", sort=True):
        cross_section = cross_section.sort_values("asset").copy()
        indexed = cross_section.set_index("asset", drop=False)
        ranks = indexed["score"].rank(method="first", pct=True)
        retained = [
            asset
            for asset in previous_selected
            if asset in indexed.index and ranks.loc[asset] >= retention_percentile
        ]
        fill = (
            indexed.loc[~indexed.index.isin(retained)]
            .sort_values("score", ascending=False, kind="mergesort")
            .head(holdings - len(retained))["asset"]
            .tolist()
        )
        selected = tuple(retained + fill)
        if len(selected) != holdings:
            continue
        weights = pd.Series(1.0 / holdings, index=selected)
        union = previous_weights.index.union(weights.index)
        traded_notional = float(
            weights.reindex(union, fill_value=0.0)
            .sub(previous_weights.reindex(union, fill_value=0.0))
            .abs()
            .sum()
        )
        selected_returns = indexed["future_return_20"].reindex(selected)
        portfolio_return = (
            float(weights.mul(selected_returns).sum())
            if selected_returns.notna().all()
            else np.nan
        )
        benchmark_return = float(cross_section["future_return_20"].mean())
        cost = traded_notional * cost_bps_one_way / 10_000.0
        ic_rows = cross_section[["score", "label"]].dropna()
        records.append(
            {
                "date": date,
                "candidate": str(cross_section["candidate"].iloc[0]),
                "assets": len(cross_section),
                "rank_ic": ic_rows["score"].corr(ic_rows["label"], method="spearman"),
                "return_coverage": float(cross_section["future_return_20"].notna().mean()),
                "selected_return_coverage": float(selected_returns.notna().mean()),
                "gross_active_return": portfolio_return - benchmark_return,
                "traded_notional": traded_notional,
                "cost": cost,
                "net_active_return": portfolio_return - benchmark_return - cost,
                "selected_assets": "|".join(selected),
            }
        )
        previous_selected = selected
        previous_weights = weights
    return pd.DataFrame(records)


def evaluate_research(
    panels_with_returns: pd.DataFrame,
    *,
    candidate_columns: list[str] | tuple[str, ...] = CANDIDATE_COLUMNS,
    candidate_definitions: dict = CANDIDATES,
    prior_discovery_p_values: pd.Series | None = None,
    discovery_start: str = "2020-01-01",
    discovery_end: str = "2022-12-31",
    validation_start: str = "2023-01-01",
    validation_end: str = "2024-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = pd.concat(
        [
            buffered_long_only(
                panels_with_returns[panels_with_returns["candidate"].eq(candidate)]
            )
            for candidate in candidate_columns
        ],
        ignore_index=True,
    )
    summary_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    for candidate in candidate_columns:
        candidate_monthly = monthly[monthly["candidate"].eq(candidate)].copy()
        discovery = candidate_monthly[candidate_monthly["date"].between(
            discovery_start, discovery_end
        )]
        validation = candidate_monthly[candidate_monthly["date"].between(
            validation_start, validation_end
        )]
        statistic, p_value = hac_mean_test(discovery["rank_ic"])
        yearly = (
            candidate_monthly.assign(year=candidate_monthly["date"].dt.year)
            .groupby("year")
            .agg(
                mean_rank_ic=("rank_ic", "mean"),
                mean_net_active_return=("net_active_return", "mean"),
                months=("rank_ic", "count"),
            )
        )
        for year, values in yearly.iterrows():
            yearly_rows.append({"candidate": candidate, "year": year, **values.to_dict()})
        validation_mean = float(validation["net_active_return"].mean())
        validation_std = float(validation["net_active_return"].std(ddof=1))
        validation_ir = (
            validation_mean / validation_std * np.sqrt(12.0)
            if validation_std > 0.0
            else np.nan
        )
        summary_rows.append(
            {
                "candidate": candidate,
                "family": candidate_definitions[candidate].family,
                "hypothesis": candidate_definitions[candidate].hypothesis,
                "discovery_months": len(discovery),
                "discovery_mean_ic": float(discovery["rank_ic"].mean()),
                "discovery_hac_t": statistic,
                "discovery_p_one_sided": p_value,
                "validation_months": len(validation),
                "validation_mean_ic": float(validation["rank_ic"].mean()),
                "validation_mean_net_active_return": validation_mean,
                "validation_net_information_ratio": validation_ir,
                "validation_2023_2024_positive": bool(
                    (yearly.reindex([2023, 2024])["mean_rank_ic"] > 0.0).all()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    current_p = summary["discovery_p_one_sided"].reset_index(drop=True)
    if prior_discovery_p_values is None:
        all_p = current_p
        offset = 0
    else:
        prior = pd.Series(prior_discovery_p_values, dtype=float).reset_index(drop=True)
        all_p = pd.concat([prior, current_p], ignore_index=True)
        offset = len(prior)
    all_q = benjamini_hochberg(all_p)
    summary["discovery_q_bh"] = all_q.iloc[offset:].to_numpy()
    summary["multiplicity_family_size"] = len(all_p)
    summary["passes_research_gate"] = (
        summary["discovery_months"].ge(34)
        & summary["discovery_mean_ic"].ge(0.015)
        & summary["discovery_q_bh"].le(0.10)
        & summary["validation_months"].ge(23)
        & summary["validation_mean_ic"].ge(0.010)
        & summary["validation_2023_2024_positive"]
        & summary["validation_mean_net_active_return"].gt(0.0)
        & summary["validation_net_information_ratio"].ge(0.30)
    )
    summary = summary.sort_values(
        [
            "passes_research_gate",
            "validation_net_information_ratio",
            "validation_mean_ic",
            "candidate",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return summary, pd.DataFrame(yearly_rows), monthly


def exposure_diagnostics(
    panels: pd.DataFrame,
    *,
    control_columns: list[str] | tuple[str, ...] = CONTROL_COLUMNS,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (candidate, date), rows in panels.groupby(["candidate", "date"], sort=True):
        industry_means = rows.groupby("industry_l1")["score"].mean()
        records.append(
            {
                "candidate": candidate,
                "date": date,
                "assets": len(rows),
                "linear_size_exposure": rows["score"].corr(rows["size_z"]),
                "quadratic_size_exposure": rows["score"].corr(rows["size_squared"]),
                "max_abs_industry_mean": float(industry_means.abs().max()),
                **{
                    f"exposure_{control}": rows["score"].corr(
                        rows[f"neutralization_control_{control}"]
                    )
                    for control in control_columns
                },
            }
        )
    return pd.DataFrame(records)
