from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import buffered_long_only
from .statistics import hac_mean_test


def evaluate_confirmation(
    panel: pd.DataFrame,
    *,
    minimum_months: int = 75,
    mean_ic_minimum: float = 0.015,
    p_value_maximum: float = 0.05,
    positive_years_minimum: int = 5,
    required_positive_year: int = 2025,
    mean_net_return_minimum: float = 0.0,
    net_information_ratio_minimum: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = buffered_long_only(panel)
    statistic, p_value = hac_mean_test(monthly["rank_ic"])
    mean_net = float(monthly["net_active_return"].mean())
    net_standard_deviation = float(monthly["net_active_return"].std(ddof=1))
    net_information_ratio = (
        mean_net / net_standard_deviation * np.sqrt(12.0)
        if net_standard_deviation > 0.0
        else np.nan
    )
    yearly = (
        monthly.assign(year=monthly["date"].dt.year)
        .groupby("year")
        .agg(
            mean_rank_ic=("rank_ic", "mean"),
            mean_net_active_return=("net_active_return", "mean"),
            months=("rank_ic", "count"),
        )
        .reset_index()
    )
    positive_years = int(yearly["mean_rank_ic"].gt(0.0).sum())
    required = yearly.loc[yearly["year"].eq(required_positive_year), "mean_rank_ic"]
    required_year_positive = bool(len(required) == 1 and required.iloc[0] > 0.0)
    values = {
        "candidate": str(panel["candidate"].iloc[0]),
        "months": int(monthly["rank_ic"].notna().sum()),
        "mean_rank_ic": float(monthly["rank_ic"].mean()),
        "hac_t": statistic,
        "hac_p_one_sided": p_value,
        "positive_years": positive_years,
        "required_positive_year": required_positive_year,
        "required_year_positive": required_year_positive,
        "mean_net_active_return": mean_net,
        "net_information_ratio": net_information_ratio,
        "mean_monthly_traded_notional": float(monthly["traded_notional"].mean()),
        "minimum_monthly_return_coverage": float(monthly["return_coverage"].min()),
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
        and values["mean_net_active_return"] > mean_net_return_minimum
        and values["net_information_ratio"] >= net_information_ratio_minimum
    )
    return pd.DataFrame([values]), yearly, monthly
