from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import compute_baselines
from .factors import Candidate


HORIZONS_V25 = (60, 90)
MARKET_STATE_LOOKBACK_V25 = 60
MARKET_STRESS_QUANTILE_V25 = 0.20
MINIMUM_STRESS_OBSERVATIONS_V25 = 8
MINIMUM_ACTIVE_STRESS_OBSERVATIONS_V25 = 4
CANDIDATES_V25 = {
    f"asrt_{horizon}": Candidate(
        "active_stress_resilience_tilt",
        (
            "Positive abnormal participation tilting toward a stock's resilient market-stress "
            f"observations over {horizon} sessions predicts higher returns."
        ),
    )
    for horizon in HORIZONS_V25
}
CANDIDATE_COLUMNS_V25 = list(CANDIDATES_V25)
COMMON_CONTROLS_V25 = [
    "low_turnover_20", "low_turnover_vol_60", "reversal_20", "low_volatility_60",
    "anti_max_return_20", "anti_skewness_60", "momentum_60_20", "momentum_120_20",
    "overnight_mean_20", "negative_intraday_mean_20", "diffuse_turnover_60",
]
CONTROL_COLUMNS_BY_CANDIDATE_V25 = {
    f"asrt_{horizon}": [
        *COMMON_CONTROLS_V25,
        f"mean_stress_resilience_{horizon}",
        f"mean_stress_activity_{horizon}",
        f"active_stress_fraction_{horizon}",
        f"unconditional_resilience_activity_correlation_{horizon}",
        f"downside_market_beta_{horizon}",
    ]
    for horizon in HORIZONS_V25
}


def _rolling(frame: pd.DataFrame, column: str, horizon: int, method: str) -> pd.Series:
    roller = frame.groupby("asset", sort=False)[column].rolling(
        horizon, min_periods=horizon
    )
    return getattr(roller, method)().reset_index(level=0, drop=True)


def compute_candidates_v25(daily: pd.DataFrame) -> pd.DataFrame:
    frame = compute_baselines(daily).sort_values(["asset", "date"]).reset_index(drop=True)
    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    close = frame["close"].where(frame["close"].gt(0.0))
    log_return = np.log(close).groupby(frame["asset"], sort=False).diff()
    market_by_date = (
        log_return.where(eligible).groupby(frame["date"]).median().sort_index()
    )
    stress_threshold = market_by_date.rolling(
        MARKET_STATE_LOOKBACK_V25, min_periods=MARKET_STATE_LOOKBACK_V25
    ).quantile(MARKET_STRESS_QUANTILE_V25).shift(1)
    market = frame["date"].map(market_by_date)
    threshold = frame["date"].map(stress_threshold)
    stress = market.le(threshold) & threshold.notna()
    residual = log_return - market
    cross_section_mad = residual.where(eligible).abs().groupby(frame["date"]).transform(
        "median"
    )
    resilience = (residual / (1.4826 * cross_section_mad).replace(0.0, np.nan)).clip(
        -5.0, 5.0
    )

    turnover = frame["turnover_fraction"].where(frame["turnover_fraction"].gt(0.0))
    log_turnover = np.log(turnover)
    grouped_turnover = log_turnover.groupby(frame["asset"], sort=False)
    trailing_mean = grouped_turnover.transform(
        lambda values: values.rolling(60, min_periods=60).mean().shift(1)
    )
    trailing_std = grouped_turnover.transform(
        lambda values: values.rolling(60, min_periods=60).std(ddof=0).shift(1)
    )
    activity = ((log_turnover - trailing_mean) / trailing_std.replace(0.0, np.nan)).clip(
        -3.0, 3.0
    )
    observation = stress & eligible & resilience.notna() & activity.notna()
    active = observation & activity.gt(0.0)
    active_weight = activity.clip(lower=0.0)

    frame["asrt_stress_count"] = observation.astype(float)
    frame["asrt_active_count"] = active.astype(float)
    frame["asrt_active_weight"] = active_weight.where(observation, 0.0)
    frame["asrt_active_resilience"] = (
        active_weight * resilience
    ).where(observation, 0.0)
    frame["asrt_stress_resilience"] = resilience.where(observation, 0.0)
    frame["asrt_stress_activity"] = activity.where(observation, 0.0)
    frame["asrt_resilience"] = resilience.where(eligible & activity.notna())
    frame["asrt_activity"] = activity.where(eligible & resilience.notna())
    frame["asrt_resilience_activity"] = frame["asrt_resilience"] * frame["asrt_activity"]
    frame["asrt_stock_return"] = log_return.where(observation, 0.0)
    frame["asrt_market_return"] = market.where(observation, 0.0)
    frame["asrt_return_market_product"] = (
        frame["asrt_stock_return"] * frame["asrt_market_return"]
    )
    frame["asrt_market_return_squared"] = frame["asrt_market_return"].pow(2)

    for horizon in HORIZONS_V25:
        stress_count = _rolling(frame, "asrt_stress_count", horizon, "sum")
        active_count = _rolling(frame, "asrt_active_count", horizon, "sum")
        active_weight_sum = _rolling(frame, "asrt_active_weight", horizon, "sum")
        enough = (
            stress_count.ge(MINIMUM_STRESS_OBSERVATIONS_V25)
            & active_count.ge(MINIMUM_ACTIVE_STRESS_OBSERVATIONS_V25)
            & active_weight_sum.gt(0.0)
        )
        mean_resilience = (
            _rolling(frame, "asrt_stress_resilience", horizon, "sum")
            / stress_count.replace(0.0, np.nan)
        )
        active_resilience = (
            _rolling(frame, "asrt_active_resilience", horizon, "sum")
            / active_weight_sum.replace(0.0, np.nan)
        )
        frame[f"asrt_{horizon}"] = (active_resilience - mean_resilience).where(enough)
        frame[f"mean_stress_resilience_{horizon}"] = mean_resilience.where(enough)
        frame[f"mean_stress_activity_{horizon}"] = (
            _rolling(frame, "asrt_stress_activity", horizon, "sum")
            / stress_count.replace(0.0, np.nan)
        ).where(enough)
        frame[f"active_stress_fraction_{horizon}"] = (
            active_count / stress_count.replace(0.0, np.nan)
        ).where(enough)

        covariance = _rolling(frame, "asrt_resilience_activity", horizon, "mean") - (
            _rolling(frame, "asrt_resilience", horizon, "mean")
            * _rolling(frame, "asrt_activity", horizon, "mean")
        )
        frame[f"unconditional_resilience_activity_correlation_{horizon}"] = covariance / (
            _rolling(frame, "asrt_resilience", horizon, "std")
            * _rolling(frame, "asrt_activity", horizon, "std")
        ).replace(0.0, np.nan)

        mean_stock = _rolling(frame, "asrt_stock_return", horizon, "sum") / stress_count.replace(
            0.0, np.nan
        )
        mean_market = _rolling(frame, "asrt_market_return", horizon, "sum") / stress_count.replace(
            0.0, np.nan
        )
        downside_covariance = (
            _rolling(frame, "asrt_return_market_product", horizon, "sum")
            / stress_count.replace(0.0, np.nan)
            - mean_stock * mean_market
        )
        downside_variance = (
            _rolling(frame, "asrt_market_return_squared", horizon, "sum")
            / stress_count.replace(0.0, np.nan)
            - mean_market.pow(2)
        )
        frame[f"downside_market_beta_{horizon}"] = (
            downside_covariance / downside_variance.replace(0.0, np.nan)
        ).where(enough)

    return frame.drop(columns=[
        "asrt_stress_count", "asrt_active_count", "asrt_active_weight",
        "asrt_active_resilience", "asrt_stress_resilience", "asrt_stress_activity",
        "asrt_resilience", "asrt_activity", "asrt_resilience_activity",
        "asrt_stock_return", "asrt_market_return", "asrt_return_market_product",
        "asrt_market_return_squared",
    ])

