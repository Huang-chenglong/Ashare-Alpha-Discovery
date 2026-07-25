from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .model_v53 import FEATURE_COLUMNS_V53, rank_feature_matrix_v53
from .model_v54 import FEATURE_COLUMNS_V54


CANDIDATE_V56 = "hgb_missing_margin_v56"


def make_model_v56() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=7,
        min_samples_leaf=80,
        l2_regularization=10.0,
        early_stopping=False,
        class_weight="balanced",
        random_state=20260725,
    )


def top_decile_target_v56(frame: pd.DataFrame) -> pd.Series:
    ranks = frame.groupby("date", sort=False)["label"].rank(
        method="average",
        pct=True,
    )
    target = ranks.gt(0.90).astype(float)
    target.loc[frame["label"].isna()] = np.nan
    return target


def _eligible(matrix: pd.DataFrame) -> pd.Series:
    return matrix[list(FEATURE_COLUMNS_V54)].notna().all(axis=1)


def _probability(model, matrix: pd.DataFrame) -> np.ndarray:
    positive = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positive) != 1:
        raise ValueError("V56 target is missing its positive class")
    return model.predict_proba(matrix)[:, int(positive[0])]


def walk_forward_predictions_v56(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    prediction_start: str,
    prediction_end: str,
    minimum_training_rows: int = 2500,
) -> pd.Series:
    if list(matrix.columns) != list(FEATURE_COLUMNS_V53):
        raise ValueError("V56 matrix columns differ from the frozen registry")
    target = top_decile_target_v56(frame)
    eligible = _eligible(matrix)
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    dates = sorted(
        frame.loc[
            frame["date"].between(prediction_start, prediction_end),
            "date",
        ].unique()
    )
    for prediction_date in dates:
        train = (
            eligible
            & target.notna()
            & frame["exit_date"].notna()
            & frame["exit_date"].lt(prediction_date)
        )
        predict = eligible & frame["date"].eq(prediction_date)
        if int(train.sum()) < minimum_training_rows or not predict.any():
            continue
        model = make_model_v56()
        model.fit(matrix.loc[train], target.loc[train].astype(int))
        output.loc[predict] = _probability(model, matrix.loc[predict])
    return output


def fit_final_model_v56(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    label_available_before: str,
    minimum_training_rows: int = 2500,
) -> dict[str, object]:
    target = top_decile_target_v56(frame)
    complete = (
        _eligible(matrix)
        & target.notna()
        & frame["exit_date"].notna()
        & frame["exit_date"].lt(label_available_before)
    )
    if int(complete.sum()) < minimum_training_rows:
        raise ValueError("V56 final model has too few point-in-time rows")
    model = make_model_v56()
    model.fit(matrix.loc[complete], target.loc[complete].astype(int))
    return {
        "candidate": CANDIDATE_V56,
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS_V53),
        "mandatory_feature_columns": list(FEATURE_COLUMNS_V54),
        "training_rows": int(complete.sum()),
        "maximum_training_exit_date": frame.loc[complete, "exit_date"]
        .max()
        .strftime("%Y-%m-%d"),
        "label_available_before": label_available_before,
    }


def predict_frozen_model_v56(
    bundle: dict[str, object],
    matrix: pd.DataFrame,
) -> pd.Series:
    if list(matrix.columns) != bundle["feature_columns"]:
        raise ValueError("V56 frozen feature columns differ")
    eligible = matrix[bundle["mandatory_feature_columns"]].notna().all(axis=1)
    output = pd.Series(np.nan, index=matrix.index, dtype=float)
    output.loc[eligible] = _probability(
        bundle["model"],
        matrix.loc[eligible],
    )
    return output


__all__ = [
    "CANDIDATE_V56",
    "fit_final_model_v56",
    "make_model_v56",
    "predict_frozen_model_v56",
    "rank_feature_matrix_v53",
    "top_decile_target_v56",
    "walk_forward_predictions_v56",
]
