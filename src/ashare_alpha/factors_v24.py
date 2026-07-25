from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import compute_baselines
from .factors import Candidate


HORIZONS_V24 = (60, 90)
CANCELLATION_RATIO_MAX_V24 = 0.25
MINIMUM_EVENTS_V24 = 8
CANDIDATES_V24 = {
    f"mepi_{horizon}": Candidate(
        "matched_excursion_participation_imbalance",
        (
            "Higher turnover on the positive leg of nearly price-cancelling "
            f"two-session excursions over {horizon} sessions predicts higher returns."
        ),
    )
    for horizon in HORIZONS_V24
}
CANDIDATE_COLUMNS_V24 = list(CANDIDATES_V24)
COMMON_CONTROLS_V24 = [
    "low_turnover_20", "low_turnover_vol_60", "reversal_20", "low_volatility_60",
    "anti_max_return_20", "anti_skewness_60", "momentum_60_20", "momentum_120_20",
    "overnight_mean_20", "negative_intraday_mean_20", "diffuse_turnover_60",
]
CONTROL_COLUMNS_BY_CANDIDATE_V24 = {
    f"mepi_{horizon}": [
        *COMMON_CONTROLS_V24,
        f"matched_event_frequency_{horizon}",
        f"mean_match_quality_{horizon}",
        f"positive_leg_second_balance_{horizon}",
        f"mean_matched_excursion_magnitude_{horizon}",
        f"unconditional_signed_turnover_change_{horizon}",
        f"residual_turnover_change_correlation_{horizon}",
    ]
    for horizon in HORIZONS_V24
}


def _rolling(frame: pd.DataFrame, column: str, horizon: int, method: str) -> pd.Series:
    roller = frame.groupby("asset", sort=False)[column].rolling(
        horizon, min_periods=horizon
    )
    return getattr(roller, method)().reset_index(level=0, drop=True)


def compute_candidates_v24(daily: pd.DataFrame) -> pd.DataFrame:
    frame = compute_baselines(daily).sort_values(["asset", "date"]).reset_index(drop=True)
    eligible = (
        frame["is_member"].fillna(False).astype(bool)
        & frame["tradestatus"].fillna(0).eq(1)
        & frame["is_st"].fillna(1).eq(0)
    )
    previous_eligible = eligible.groupby(frame["asset"], sort=False).shift(
        1, fill_value=False
    )
    close = frame["close"].where(frame["close"].gt(0.0))
    log_return = np.log(close).groupby(frame["asset"], sort=False).diff()
    market = log_return.where(eligible).groupby(frame["date"]).transform("median")
    residual = log_return - market
    previous_residual = residual.groupby(frame["asset"], sort=False).shift(1)

    turnover = frame["turnover_fraction"].where(frame["turnover_fraction"].gt(0.0))
    previous_turnover = turnover.groupby(frame["asset"], sort=False).shift(1)
    turnover_change = (turnover - previous_turnover) / (
        turnover + previous_turnover
    ).replace(0.0, np.nan)
    excursion_magnitude = residual.abs() + previous_residual.abs()
    cancellation_ratio = (residual + previous_residual).abs() / excursion_magnitude.replace(
        0.0, np.nan
    )
    matched = (
        eligible
        & previous_eligible
        & residual.mul(previous_residual).lt(0.0)
        & cancellation_ratio.le(CANCELLATION_RATIO_MAX_V24)
        & turnover_change.notna()
    )
    match_quality = 1.0 - cancellation_ratio
    participation_imbalance = np.sign(residual) * turnover_change

    frame["mepi_event"] = matched.astype(float)
    frame["mepi_quality"] = match_quality.where(matched, 0.0)
    frame["mepi_weighted_imbalance"] = (
        match_quality * participation_imbalance
    ).where(matched, 0.0)
    frame["mepi_positive_leg_second"] = np.sign(residual).where(matched, 0.0)
    frame["mepi_excursion_magnitude"] = excursion_magnitude.where(matched, 0.0)
    frame["mepi_signed_turnover_change"] = (
        np.sign(residual) * turnover_change
    ).where(eligible & previous_eligible)
    frame["mepi_residual"] = residual.where(eligible & previous_eligible)
    frame["mepi_turnover_change"] = turnover_change.where(eligible & previous_eligible)
    frame["mepi_residual_turnover_product"] = (
        frame["mepi_residual"] * frame["mepi_turnover_change"]
    )

    for horizon in HORIZONS_V24:
        event_count = _rolling(frame, "mepi_event", horizon, "sum")
        quality_sum = _rolling(frame, "mepi_quality", horizon, "sum")
        enough = event_count.ge(MINIMUM_EVENTS_V24) & quality_sum.gt(0.0)
        frame[f"mepi_{horizon}"] = (
            _rolling(frame, "mepi_weighted_imbalance", horizon, "sum")
            / quality_sum.replace(0.0, np.nan)
        ).where(enough)
        frame[f"matched_event_frequency_{horizon}"] = (event_count / horizon).where(enough)
        frame[f"mean_match_quality_{horizon}"] = (
            quality_sum / event_count.replace(0.0, np.nan)
        ).where(enough)
        frame[f"positive_leg_second_balance_{horizon}"] = (
            _rolling(frame, "mepi_positive_leg_second", horizon, "sum")
            / event_count.replace(0.0, np.nan)
        ).where(enough)
        frame[f"mean_matched_excursion_magnitude_{horizon}"] = (
            _rolling(frame, "mepi_excursion_magnitude", horizon, "sum")
            / event_count.replace(0.0, np.nan)
        ).where(enough)
        frame[f"unconditional_signed_turnover_change_{horizon}"] = _rolling(
            frame, "mepi_signed_turnover_change", horizon, "mean"
        )
        covariance = _rolling(
            frame, "mepi_residual_turnover_product", horizon, "mean"
        ) - (
            _rolling(frame, "mepi_residual", horizon, "mean")
            * _rolling(frame, "mepi_turnover_change", horizon, "mean")
        )
        frame[f"residual_turnover_change_correlation_{horizon}"] = covariance / (
            _rolling(frame, "mepi_residual", horizon, "std")
            * _rolling(frame, "mepi_turnover_change", horizon, "std")
        ).replace(0.0, np.nan)

    return frame.drop(columns=[
        "mepi_event", "mepi_quality", "mepi_weighted_imbalance",
        "mepi_positive_leg_second", "mepi_excursion_magnitude",
        "mepi_signed_turnover_change", "mepi_residual", "mepi_turnover_change",
        "mepi_residual_turnover_product",
    ])

