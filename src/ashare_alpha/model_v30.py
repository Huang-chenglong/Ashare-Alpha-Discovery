from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .baselines import BASELINE_COLUMNS
from .evaluate import month_end_formation
from .factors_v24 import compute_candidates_v24
from .factors_v25 import compute_candidates_v25
from .factors_v26 import compute_candidates_v26
from .factors_v29 import compute_candidates_v29
from .statistics import mad_winsorize, zscore


KNOWN_FEATURES_V30 = list(BASELINE_COLUMNS)
NOVEL_FEATURES_V30 = [
    "mepi_60", "mepi_90", "matched_event_frequency_60",
    "matched_event_frequency_90", "asrt_60", "asrt_90",
    "mean_stress_resilience_60", "mean_stress_resilience_90",
    "fsid_100", "fsid_120", "mean_completed_residual_return_120",
    "mean_raw_absorption_slope_120", "remaining_float_inventory_days_025_control",
    "drra_60", "drra_90", "tsra_60", "tsra_90",
    "mean_lower_shadow_share_60", "mean_upper_shadow_share_60",
]
FEATURES_V30 = [*KNOWN_FEATURES_V30, *NOVEL_FEATURES_V30]
MODEL_PARAMETERS_V30 = {
    "loss": "squared_error",
    "learning_rate": 0.03,
    "max_iter": 100,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 100,
    "l2_regularization": 10.0,
    "max_bins": 63,
    "early_stopping": False,
    "random_state": 0,
}


def _monthly_subset(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    keys = [
        "date", "asset", "float_market_cap", "is_member", "tradestatus", "is_st",
    ]
    available = list(dict.fromkeys([*keys, *columns]))
    return month_end_formation(frame)[available].copy()


def build_monthly_features_v30(daily: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        *KNOWN_FEATURES_V30,
        "drra_60", "drra_90", "tsra_60", "tsra_90",
        "mean_lower_shadow_share_60", "mean_upper_shadow_share_60",
    ]
    base = _monthly_subset(compute_candidates_v29(daily), base_columns)
    gc.collect()

    additions = [
        (
            compute_candidates_v24,
            ["mepi_60", "mepi_90", "matched_event_frequency_60", "matched_event_frequency_90"],
        ),
        (
            compute_candidates_v25,
            ["asrt_60", "asrt_90", "mean_stress_resilience_60", "mean_stress_resilience_90"],
        ),
        (
            compute_candidates_v26,
            [
                "fsid_100", "fsid_120", "mean_completed_residual_return_120",
                "mean_raw_absorption_slope_120", "remaining_float_inventory_days_025_control",
            ],
        ),
    ]
    for compute, columns in additions:
        addition = _monthly_subset(compute(daily), columns)[["date", "asset", *columns]]
        base = base.merge(addition, on=["date", "asset"], how="left", validate="one_to_one")
        del addition
        gc.collect()
    return base.sort_values(["date", "asset"]).reset_index(drop=True)


def rank_feature_matrix_v30(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    matrix = pd.DataFrame(index=panel.index)
    columns: list[str] = []
    for feature in FEATURES_V30:
        ranked = panel[feature].groupby(panel["date"]).rank(method="average", pct=True)
        rank_column = f"rank__{feature}"
        matrix[rank_column] = ranked.fillna(0.5)
        columns.append(rank_column)
        if feature in NOVEL_FEATURES_V30:
            missing_column = f"missing__{feature}"
            matrix[missing_column] = ranked.isna().astype(float)
            columns.append(missing_column)
    return matrix, columns


def residualize_training_target_v30(panel: pd.DataFrame) -> pd.Series:
    output = pd.Series(np.nan, index=panel.index, dtype=float)
    for _, rows in panel.groupby("date", sort=True):
        required = ["label", "float_market_cap", "industry_l1", *KNOWN_FEATURES_V30]
        finite = rows[required].replace([np.inf, -np.inf], np.nan)
        valid = finite.notna().all(axis=1) & rows["float_market_cap"].gt(0.0)
        sample = rows.loc[valid]
        if len(sample) < 100:
            continue
        target = mad_winsorize(sample["label"])
        size_z = zscore(np.log(sample["float_market_cap"]))
        size_squared = size_z.pow(2) - size_z.pow(2).mean()
        controls = [
            zscore(mad_winsorize(sample[column])).to_numpy()
            for column in KNOWN_FEATURES_V30
        ]
        industries = pd.get_dummies(sample["industry_l1"], drop_first=True, dtype=float)
        design = np.column_stack([
            np.ones(len(sample)), size_z.to_numpy(), size_squared.to_numpy(),
            *controls, *industries.to_numpy().T,
        ])
        coefficients = np.linalg.lstsq(design, target.to_numpy(), rcond=None)[0]
        output.loc[sample.index] = target.to_numpy() - design @ coefficients
    return output


def fit_model_v30(matrix: pd.DataFrame, target: pd.Series, dates: pd.Series) -> HistGradientBoostingRegressor:
    training = dates.between("2020-01-01", "2022-12-31") & target.notna()
    if int(training.sum()) < 5_000:
        raise ValueError("V30 has too few complete training rows")
    model = HistGradientBoostingRegressor(**MODEL_PARAMETERS_V30)
    model.fit(matrix.loc[training], target.loc[training])
    return model

