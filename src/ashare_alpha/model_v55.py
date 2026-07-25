from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .model_v54 import FEATURE_COLUMNS_V54


CANDIDATE_V55 = "hgb_financing_long_v55"


def make_model_v55() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=7,
        min_samples_leaf=40,
        l2_regularization=10.0,
        early_stopping=False,
        class_weight="balanced",
        random_state=20260725,
    )


def top_decile_target_v55(frame: pd.DataFrame) -> pd.Series:
    ranks = frame.groupby("date", sort=False)["label"].rank(
        method="average",
        pct=True,
    )
    target = ranks.gt(0.90).astype(float)
    target.loc[frame["label"].isna()] = np.nan
    return target


def _probability(model, matrix: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError("V55 target is missing its positive class")
    return model.predict_proba(matrix)[:, int(positive[0])]


def walk_forward_predictions_v55(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    prediction_start: str,
    prediction_end: str,
    minimum_training_rows: int = 2500,
) -> pd.Series:
    target = top_decile_target_v55(frame)
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    dates = sorted(
        frame.loc[
            frame["date"].between(prediction_start, prediction_end),
            "date",
        ].unique()
    )
    complete_features = matrix.notna().all(axis=1)
    for prediction_date in dates:
        train = (
            complete_features
            & target.notna()
            & frame["exit_date"].notna()
            & frame["exit_date"].lt(prediction_date)
        )
        predict = complete_features & frame["date"].eq(prediction_date)
        if int(train.sum()) < minimum_training_rows or not predict.any():
            continue
        model = make_model_v55()
        model.fit(matrix.loc[train], target.loc[train].astype(int))
        output.loc[predict] = _probability(model, matrix.loc[predict])
    return output


def fit_final_model_v55(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    label_available_before: str,
    minimum_training_rows: int = 2500,
) -> dict[str, object]:
    target = top_decile_target_v55(frame)
    complete = (
        matrix.notna().all(axis=1)
        & target.notna()
        & frame["exit_date"].notna()
        & frame["exit_date"].lt(label_available_before)
    )
    if int(complete.sum()) < minimum_training_rows:
        raise ValueError("V55 final model has too few point-in-time rows")
    model = make_model_v55()
    model.fit(matrix.loc[complete], target.loc[complete].astype(int))
    return {
        "candidate": CANDIDATE_V55,
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS_V54),
        "training_rows": int(complete.sum()),
        "maximum_training_exit_date": frame.loc[complete, "exit_date"]
        .max()
        .strftime("%Y-%m-%d"),
        "label_available_before": label_available_before,
    }


def predict_frozen_model_v55(
    bundle: dict[str, object],
    matrix: pd.DataFrame,
) -> pd.Series:
    if list(matrix.columns) != bundle["feature_columns"]:
        raise ValueError("V55 frozen feature columns differ")
    complete = matrix.notna().all(axis=1)
    output = pd.Series(np.nan, index=matrix.index, dtype=float)
    output.loc[complete] = _probability(
        bundle["model"],
        matrix.loc[complete],
    )
    return output


__all__ = [
    "CANDIDATE_V55",
    "fit_final_model_v55",
    "make_model_v55",
    "predict_frozen_model_v55",
    "top_decile_target_v55",
    "walk_forward_predictions_v55",
]
