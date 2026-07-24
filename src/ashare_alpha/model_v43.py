from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .model_v30 import (
    FEATURES_V30,
    KNOWN_FEATURES_V30,
    build_monthly_features_v30,
    rank_feature_matrix_v30,
    residualize_training_target_v30,
)


CANDIDATE_V43 = "rrsm_v43"
FEATURES_V43 = list(FEATURES_V30)
PREDICTION_YEARS_V43 = (2023, 2024, 2025)
MODEL_PARAMETERS_V43 = {
    "loss": "squared_error",
    "learning_rate": 0.03,
    "max_iter": 150,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 200,
    "l2_regularization": 20.0,
    "max_bins": 63,
    "early_stopping": False,
    "random_state": 43,
}


def make_rank_target_v43(
    dates: pd.Series,
    residual_target: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    target = pd.Series(np.nan, index=residual_target.index, dtype=float)
    weights = pd.Series(np.nan, index=residual_target.index, dtype=float)
    valid = residual_target.notna()
    valid_dates = dates.loc[valid]
    for _, index in valid_dates.groupby(valid_dates).groups.items():
        values = residual_target.loc[index]
        target.loc[index] = values.rank(method="average", pct=True) - 0.5
        weights.loc[index] = 1.0 / len(index)
    finite_weights = weights.dropna()
    if not finite_weights.empty:
        weights.loc[finite_weights.index] *= len(finite_weights) / finite_weights.sum()
    return target, weights


def complete_prediction_rows_v43(panel: pd.DataFrame) -> pd.Series:
    finite = panel[KNOWN_FEATURES_V30].replace([np.inf, -np.inf], np.nan)
    return (
        finite.notna().all(axis=1)
        & panel["float_market_cap"].gt(0.0)
        & panel["industry_l1"].notna()
    )


def _fit_model_v43(
    matrix: pd.DataFrame,
    target: pd.Series,
    weights: pd.Series,
) -> HistGradientBoostingRegressor:
    valid = target.notna() & weights.notna()
    if int(valid.sum()) < 10_000:
        raise ValueError("V43 has too few complete training rows")
    model = HistGradientBoostingRegressor(**MODEL_PARAMETERS_V43)
    model.fit(
        matrix.loc[valid],
        target.loc[valid],
        sample_weight=weights.loc[valid],
    )
    return model


def fit_walk_forward_models_v43(
    panel: pd.DataFrame,
    matrix: pd.DataFrame,
) -> tuple[pd.Series, dict[int, HistGradientBoostingRegressor], pd.DataFrame]:
    dates = pd.to_datetime(panel["date"])
    eligible = complete_prediction_rows_v43(panel)
    predictions = pd.Series(np.nan, index=panel.index, dtype=float)
    models: dict[int, HistGradientBoostingRegressor] = {}
    audit: list[dict[str, object]] = []

    for year in PREDICTION_YEARS_V43:
        training = dates.between("2020-01-01", f"{year - 1}-12-31")
        residual = residualize_training_target_v30(panel.loc[training])
        target, weights = make_rank_target_v43(dates.loc[training], residual)
        model = _fit_model_v43(matrix.loc[training], target, weights)
        prediction = dates.dt.year.eq(year) & eligible
        predictions.loc[prediction] = model.predict(matrix.loc[prediction])
        models[year] = model
        training_index = target.index[target.notna()]
        audit.append(
            {
                "prediction_year": year,
                "training_end": f"{year - 1}-12-31",
                "training_rows": int(target.notna().sum()),
                "training_assets": int(
                    panel.loc[training_index, "asset"].nunique()
                ),
                "prediction_rows": int(prediction.sum()),
                "prediction_assets": int(
                    panel.loc[prediction, "asset"].nunique()
                ),
            }
        )
    return predictions, models, pd.DataFrame(audit)


def predict_walk_forward_v43(
    panel: pd.DataFrame,
    matrix: pd.DataFrame,
    models: dict[int, HistGradientBoostingRegressor],
) -> pd.Series:
    dates = pd.to_datetime(panel["date"])
    eligible = complete_prediction_rows_v43(panel)
    predictions = pd.Series(np.nan, index=panel.index, dtype=float)
    for year in PREDICTION_YEARS_V43:
        if year not in models:
            raise ValueError(f"V43 model bundle is missing prediction year {year}")
        selected = dates.dt.year.eq(year) & eligible
        predictions.loc[selected] = models[year].predict(matrix.loc[selected])
    return predictions


__all__ = [
    "CANDIDATE_V43",
    "FEATURES_V43",
    "KNOWN_FEATURES_V30",
    "MODEL_PARAMETERS_V43",
    "PREDICTION_YEARS_V43",
    "build_monthly_features_v30",
    "fit_walk_forward_models_v43",
    "make_rank_target_v43",
    "predict_walk_forward_v43",
    "rank_feature_matrix_v30",
]
