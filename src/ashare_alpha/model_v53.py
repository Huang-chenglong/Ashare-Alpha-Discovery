from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .margin_modalities_v53 import NEW_MODALITY_RAW_FEATURES_V53
from .model_v52 import FEATURE_COLUMNS_V52
from .statistics import benjamini_hochberg


CANDIDATES_V53 = (
    "logit_tail_v53",
    "hgb_tail_v53",
    "extra_tail_v53",
)

FEATURE_COLUMNS_V53 = tuple(
    dict.fromkeys([*FEATURE_COLUMNS_V52, *NEW_MODALITY_RAW_FEATURES_V53])
)


@dataclass(frozen=True)
class ModelSpecV53:
    candidate: str
    description: str


MODEL_SPECS_V53 = {
    "logit_tail_v53": ModelSpecV53(
        "logit_tail_v53",
        "balanced L2 logistic classifier with C 0.1",
    ),
    "hgb_tail_v53": ModelSpecV53(
        "hgb_tail_v53",
        "balanced shallow histogram gradient-boosting classifier",
    ),
    "extra_tail_v53": ModelSpecV53(
        "extra_tail_v53",
        "balanced depth-five extremely randomized tree classifier",
    ),
}


def make_model_v53(candidate: str):
    if candidate == "logit_tail_v53":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                penalty="l2",
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                random_state=20260724,
            ),
        )
    if candidate == "hgb_tail_v53":
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=7,
            min_samples_leaf=80,
            l2_regularization=10.0,
            early_stopping=False,
            class_weight="balanced",
            random_state=20260724,
        )
    if candidate == "extra_tail_v53":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=50,
            max_features=0.75,
            bootstrap=False,
            class_weight="balanced",
            n_jobs=-1,
            random_state=20260724,
        )
    raise ValueError(f"Unknown V53 model {candidate}")


def rank_feature_matrix_v53(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURE_COLUMNS_V53).difference(frame.columns)
    if missing:
        raise ValueError(f"V53 feature panel is missing {sorted(missing)}")
    ranked = pd.DataFrame(index=frame.index)
    for column in FEATURE_COLUMNS_V53:
        ranked[column] = frame.groupby("date", sort=False)[column].rank(
            method="average",
            pct=True,
        )
    return ranked.replace([np.inf, -np.inf], np.nan)


def top_decile_target_v53(frame: pd.DataFrame) -> pd.Series:
    ranks = frame.groupby("date", sort=False)["label"].rank(
        method="average",
        pct=True,
    )
    target = ranks.gt(0.90).astype(float)
    target.loc[frame["label"].isna()] = np.nan
    return target


def _positive_probability(model, matrix: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise ValueError("V53 training target does not contain a positive class")
    return model.predict_proba(matrix)[:, int(positive[0])]


def walk_forward_predictions_v53(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    candidate: str,
    prediction_start: str,
    prediction_end: str,
    minimum_training_rows: int = 2500,
) -> pd.Series:
    if candidate not in CANDIDATES_V53:
        raise ValueError(f"Unknown V53 candidate {candidate}")
    target = top_decile_target_v53(frame)
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    dates = sorted(
        frame.loc[
            frame["date"].between(prediction_start, prediction_end),
            "date",
        ].unique()
    )
    complete_features = matrix.notna().all(axis=1)
    complete_target = target.notna() & frame["exit_date"].notna()
    for prediction_date in dates:
        train = (
            complete_features
            & complete_target
            & frame["exit_date"].lt(prediction_date)
        )
        predict = complete_features & frame["date"].eq(prediction_date)
        if int(train.sum()) < minimum_training_rows or not predict.any():
            continue
        model = make_model_v53(candidate)
        model.fit(matrix.loc[train], target.loc[train].astype(int))
        output.loc[predict] = _positive_probability(
            model,
            matrix.loc[predict],
        )
    return output


def fit_final_model_v53(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    candidate: str,
    label_available_before: str,
    minimum_training_rows: int = 2500,
) -> dict[str, object]:
    target = top_decile_target_v53(frame)
    complete = (
        matrix.notna().all(axis=1)
        & target.notna()
        & frame["exit_date"].notna()
        & frame["exit_date"].lt(label_available_before)
    )
    if int(complete.sum()) < minimum_training_rows:
        raise ValueError("V53 final model has too few point-in-time training rows")
    model = make_model_v53(candidate)
    model.fit(matrix.loc[complete], target.loc[complete].astype(int))
    return {
        "candidate": candidate,
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS_V53),
        "target": "within_month_label_percentile_strictly_above_0.90",
        "training_rows": int(complete.sum()),
        "training_assets": int(frame.loc[complete, "asset"].nunique()),
        "training_date_min": frame.loc[complete, "date"].min().strftime("%Y-%m-%d"),
        "training_date_max": frame.loc[complete, "date"].max().strftime("%Y-%m-%d"),
        "maximum_training_exit_date": frame.loc[complete, "exit_date"]
        .max()
        .strftime("%Y-%m-%d"),
        "label_available_before": label_available_before,
    }


def predict_frozen_model_v53(
    bundle: dict[str, object],
    matrix: pd.DataFrame,
) -> pd.Series:
    if list(matrix.columns) != bundle["feature_columns"]:
        raise ValueError("V53 frozen model feature columns differ")
    complete = matrix.notna().all(axis=1)
    output = pd.Series(np.nan, index=matrix.index, dtype=float)
    output.loc[complete] = _positive_probability(
        bundle["model"],
        matrix.loc[complete],
    )
    return output


def apply_construction_gate_v53(
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    *,
    required_core_years_positive: list[int] | tuple[int, ...],
    family_bh_q_max: float,
) -> pd.DataFrame:
    output = summary.copy()
    output["family_q_bh"] = benjamini_hochberg(
        output["hac_p_one_sided"].astype(float)
    ).to_numpy()
    required_years = set(int(value) for value in required_core_years_positive)
    core_pass: dict[str, bool] = {}
    for candidate in CANDIDATES_V53:
        candidate_years = yearly[
            yearly["candidate"].eq(candidate)
        ].set_index("year")
        core_pass[candidate] = bool(
            required_years.issubset(set(candidate_years.index))
            and candidate_years.reindex(sorted(required_years))["mean_rank_ic"]
            .gt(0.0)
            .all()
        )
    output["required_core_years_positive"] = output["candidate"].map(core_pass)
    output["passes_construction_gate_before_family_bh"] = (
        output["passes_confirmation_gate"]
        & output["required_core_years_positive"]
    )
    output["passes_construction_gate"] = (
        output["passes_construction_gate_before_family_bh"]
        & output["family_q_bh"].le(float(family_bh_q_max))
    )
    return output.drop(columns="passes_confirmation_gate")


def select_candidate_v53(summary: pd.DataFrame) -> str | None:
    passed = summary[summary["passes_construction_gate"]].copy()
    if passed.empty:
        return None
    registry_order = {
        candidate: index for index, candidate in enumerate(CANDIDATES_V53)
    }
    passed["registry_order"] = passed["candidate"].map(registry_order)
    passed = passed.sort_values(
        ["mean_net_active_return", "registry_order"],
        ascending=[False, True],
        kind="mergesort",
    )
    return str(passed.iloc[0]["candidate"])


__all__ = [
    "CANDIDATES_V53",
    "FEATURE_COLUMNS_V53",
    "MODEL_SPECS_V53",
    "apply_construction_gate_v53",
    "fit_final_model_v53",
    "make_model_v53",
    "predict_frozen_model_v53",
    "rank_feature_matrix_v53",
    "select_candidate_v53",
    "top_decile_target_v53",
    "walk_forward_predictions_v53",
]
