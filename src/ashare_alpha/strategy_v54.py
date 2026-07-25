from __future__ import annotations

import numpy as np
import pandas as pd

from .statistics import hac_mean_test


def buffered_long_short_v54(
    panel: pd.DataFrame,
    *,
    holdings_each_side: int = 10,
    long_retention_percentile: float = 0.70,
    short_retention_percentile: float = 0.30,
    cost_bps_one_way: float = 20.0,
    annual_short_borrow_bps: float = 800.0,
) -> pd.DataFrame:
    previous_weights = pd.Series(dtype=float)
    previous_long: tuple[str, ...] = ()
    previous_short: tuple[str, ...] = ()
    records: list[dict[str, object]] = []
    for date, rows in panel.groupby("date", sort=True):
        indexed = rows.sort_values("asset").set_index("asset", drop=False)
        ranks = indexed["score"].rank(method="first", pct=True)
        retained_long = [
            asset
            for asset in previous_long
            if asset in indexed.index
            and ranks.loc[asset] >= long_retention_percentile
        ][:holdings_each_side]
        long_fill = (
            indexed.loc[~indexed.index.isin(retained_long)]
            .sort_values("score", ascending=False, kind="mergesort")
            .head(holdings_each_side - len(retained_long))["asset"]
            .tolist()
        )
        selected_long = tuple(retained_long + long_fill)
        retained_short = [
            asset
            for asset in previous_short
            if asset in indexed.index
            and asset not in selected_long
            and ranks.loc[asset] <= short_retention_percentile
        ][:holdings_each_side]
        short_fill = (
            indexed.loc[
                ~indexed.index.isin([*selected_long, *retained_short])
            ]
            .sort_values("score", ascending=True, kind="mergesort")
            .head(holdings_each_side - len(retained_short))["asset"]
            .tolist()
        )
        selected_short = tuple(retained_short + short_fill)
        if (
            len(selected_long) != holdings_each_side
            or len(selected_short) != holdings_each_side
        ):
            continue
        weights = pd.Series(
            [
                *([1.0 / holdings_each_side] * holdings_each_side),
                *([-1.0 / holdings_each_side] * holdings_each_side),
            ],
            index=[*selected_long, *selected_short],
        )
        union = previous_weights.index.union(weights.index)
        traded_notional = float(
            weights.reindex(union, fill_value=0.0)
            .sub(previous_weights.reindex(union, fill_value=0.0))
            .abs()
            .sum()
        )
        selected_returns = indexed["future_return_20"].reindex(weights.index)
        gross = (
            float(weights.mul(selected_returns).sum())
            if selected_returns.notna().all()
            else np.nan
        )
        trading_cost = traded_notional * cost_bps_one_way / 10_000.0
        borrow_cost = annual_short_borrow_bps / 10_000.0 / 12.0
        ic_rows = indexed[["score", "label"]].dropna()
        records.append(
            {
                "date": date,
                "candidate": str(indexed["candidate"].iloc[0]),
                "assets": len(indexed),
                "rank_ic": ic_rows["score"].corr(
                    ic_rows["label"],
                    method="spearman",
                ),
                "return_coverage": float(
                    indexed["future_return_20"].notna().mean()
                ),
                "selected_return_coverage": float(
                    selected_returns.notna().mean()
                ),
                "gross_spread_return": gross,
                "traded_notional": traded_notional,
                "trading_cost": trading_cost,
                "borrow_cost": borrow_cost,
                "net_spread_return": gross - trading_cost - borrow_cost,
                "long_assets": "|".join(selected_long),
                "short_assets": "|".join(selected_short),
            }
        )
        previous_weights = weights
        previous_long = selected_long
        previous_short = selected_short
    return pd.DataFrame(records)


def evaluate_long_short_v54(
    panel: pd.DataFrame,
    *,
    minimum_months: int,
    mean_ic_minimum: float,
    p_value_maximum: float,
    positive_years_minimum: int,
    required_positive_year: int,
    mean_net_return_minimum: float,
    net_information_ratio_minimum: float,
    monthly_return_coverage_minimum: float,
    selected_return_coverage_minimum: float,
    holdings_each_side: int,
    long_retention_percentile: float,
    short_retention_percentile: float,
    cost_bps_one_way: float,
    annual_short_borrow_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = buffered_long_short_v54(
        panel,
        holdings_each_side=holdings_each_side,
        long_retention_percentile=long_retention_percentile,
        short_retention_percentile=short_retention_percentile,
        cost_bps_one_way=cost_bps_one_way,
        annual_short_borrow_bps=annual_short_borrow_bps,
    )
    statistic, p_value = hac_mean_test(monthly["rank_ic"])
    mean_net = float(monthly["net_spread_return"].mean())
    standard_deviation = float(monthly["net_spread_return"].std(ddof=1))
    information_ratio = (
        mean_net / standard_deviation * np.sqrt(12.0)
        if standard_deviation > 0
        else np.nan
    )
    yearly = (
        monthly.assign(year=monthly["date"].dt.year)
        .groupby("year")
        .agg(
            mean_rank_ic=("rank_ic", "mean"),
            mean_net_spread_return=("net_spread_return", "mean"),
            months=("rank_ic", "count"),
        )
        .reset_index()
    )
    required = yearly.loc[
        yearly["year"].eq(required_positive_year),
        "mean_rank_ic",
    ]
    values = {
        "candidate": str(panel["candidate"].iloc[0]),
        "months": int(monthly["rank_ic"].notna().sum()),
        "mean_rank_ic": float(monthly["rank_ic"].mean()),
        "hac_t": statistic,
        "hac_p_one_sided": p_value,
        "positive_years": int(yearly["mean_rank_ic"].gt(0).sum()),
        "required_positive_year": required_positive_year,
        "required_year_positive": bool(
            len(required) == 1 and required.iloc[0] > 0
        ),
        "mean_net_spread_return": mean_net,
        "net_information_ratio": information_ratio,
        "mean_monthly_traded_notional": float(
            monthly["traded_notional"].mean()
        ),
        "minimum_monthly_return_coverage": float(
            monthly["return_coverage"].min()
        ),
        "minimum_selected_return_coverage": float(
            monthly["selected_return_coverage"].min()
        ),
    }
    values["passes_confirmation_gate"] = bool(
        values["months"] >= minimum_months
        and values["mean_rank_ic"] >= mean_ic_minimum
        and values["hac_p_one_sided"] <= p_value_maximum
        and values["positive_years"] >= positive_years_minimum
        and values["required_year_positive"]
        and values["mean_net_spread_return"] > mean_net_return_minimum
        and values["net_information_ratio"] >= net_information_ratio_minimum
        and values["minimum_monthly_return_coverage"]
        >= monthly_return_coverage_minimum
        and values["minimum_selected_return_coverage"]
        >= selected_return_coverage_minimum
    )
    return pd.DataFrame([values]), yearly, monthly


__all__ = ["buffered_long_short_v54", "evaluate_long_short_v54"]
