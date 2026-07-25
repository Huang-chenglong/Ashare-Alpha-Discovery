from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .factors import CANDIDATE_COLUMNS, CONTROL_COLUMNS
from .model_v52 import FEATURE_COLUMNS_V52
from .statistics import benjamini_hochberg


CANDIDATES_V54 = (
    "logit_spread_v54",
    "hgb_spread_v54",
    "extra_spread_v54",
)

FINANCING_FEATURES_V54 = (
    "financing_buy_to_amount",
    "financing_buy_to_balance",
    "financing_buy_surprise_4",
)

FEATURE_COLUMNS_V54 = tuple(
    dict.fromkeys(
        [
            *(
                column
                for column in FEATURE_COLUMNS_V52
                if column not in {"short_leverage", "short_sell_intensity"}
            ),
            *FINANCING_FEATURES_V54,
        ]
    )
)

CONTROL_COLUMNS_V54 = tuple(
    dict.fromkeys(
        [
            *CANDIDATE_COLUMNS,
            *CONTROL_COLUMNS,
            "financing_leverage_rank",
            "financing_inflow_rank",
            "financing_outflow_rank",
            "positive_momentum_rank",
            "negative_momentum_rank",
            "path_inflow_breadth_rank",
            "path_signed_ratio_rank",
            "path_stability_rank",
            *(f"{column}_rank" for column in FINANCING_FEATURES_V54),
        ]
    )
)


def make_model_v54(candidate: str):
    if candidate == "logit_spread_v54":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                random_state=20260725,
            ),
        )
    if candidate == "hgb_spread_v54":
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
    if candidate == "extra_spread_v54":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=30,
            max_features=0.75,
            bootstrap=False,
            class_weight="balanced",
            n_jobs=-1,
            random_state=20260725,
        )
    raise ValueError(f"Unknown V54 model {candidate}")


def rank_feature_matrix_v54(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURE_COLUMNS_V54).difference(frame.columns)
    if missing:
        raise ValueError(f"V54 feature panel is missing {sorted(missing)}")
    ranked = pd.DataFrame(index=frame.index)
    for column in FEATURE_COLUMNS_V54:
        ranked[column] = frame.groupby("date", sort=False)[column].rank(
            method="average",
            pct=True,
        )
    return ranked.replace([np.inf, -np.inf], np.nan)


def signed_tail_target_v54(frame: pd.DataFrame) -> pd.Series:
    ranks = frame.groupby("date", sort=False)["label"].rank(
        method="average",
        pct=True,
    )
    target = pd.Series(0.0, index=frame.index)
    target.loc[ranks.le(0.10)] = -1.0
    target.loc[ranks.gt(0.90)] = 1.0
    target.loc[frame["label"].isna()] = np.nan
    return target


def _tail_spread_probability(model, matrix: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    negative = np.flatnonzero(classes == -1)
    if len(positive) != 1 or len(negative) != 1:
        raise ValueError("V54 training target is missing a tail class")
    probabilities = model.predict_proba(matrix)
    return (
        probabilities[:, int(positive[0])]
        - probabilities[:, int(negative[0])]
    )


def walk_forward_predictions_v54(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    candidate: str,
    prediction_start: str,
    prediction_end: str,
    minimum_training_rows: int = 2500,
) -> pd.Series:
    if candidate not in CANDIDATES_V54:
        raise ValueError(f"Unknown V54 candidate {candidate}")
    target = signed_tail_target_v54(frame)
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
        model = make_model_v54(candidate)
        model.fit(matrix.loc[train], target.loc[train].astype(int))
        output.loc[predict] = _tail_spread_probability(
            model,
            matrix.loc[predict],
        )
    return output


def fit_final_model_v54(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    candidate: str,
    label_available_before: str,
    minimum_training_rows: int = 2500,
) -> dict[str, object]:
    target = signed_tail_target_v54(frame)
    complete = (
        matrix.notna().all(axis=1)
        & target.notna()
        & frame["exit_date"].notna()
        & frame["exit_date"].lt(label_available_before)
    )
    if int(complete.sum()) < minimum_training_rows:
        raise ValueError("V54 final model has too few point-in-time training rows")
    model = make_model_v54(candidate)
    model.fit(matrix.loc[complete], target.loc[complete].astype(int))
    return {
        "candidate": candidate,
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS_V54),
        "target": "bottom_decile_minus_one_middle_zero_top_decile_plus_one",
        "training_rows": int(complete.sum()),
        "training_assets": int(frame.loc[complete, "asset"].nunique()),
        "maximum_training_exit_date": frame.loc[complete, "exit_date"]
        .max()
        .strftime("%Y-%m-%d"),
        "label_available_before": label_available_before,
    }


def predict_frozen_model_v54(
    bundle: dict[str, object],
    matrix: pd.DataFrame,
) -> pd.Series:
    if list(matrix.columns) != bundle["feature_columns"]:
        raise ValueError("V54 frozen model feature columns differ")
    complete = matrix.notna().all(axis=1)
    output = pd.Series(np.nan, index=matrix.index, dtype=float)
    output.loc[complete] = _tail_spread_probability(
        bundle["model"],
        matrix.loc[complete],
    )
    return output


def apply_construction_gate_v54(
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
    required_years = set(map(int, required_core_years_positive))
    core: dict[str, bool] = {}
    for candidate in CANDIDATES_V54:
        values = yearly[yearly["candidate"].eq(candidate)].set_index("year")
        core[candidate] = bool(
            required_years.issubset(set(values.index))
            and values.reindex(sorted(required_years))["mean_rank_ic"].gt(0).all()
        )
    output["required_core_years_positive"] = output["candidate"].map(core)
    output["passes_construction_gate_before_family_bh"] = (
        output["passes_confirmation_gate"]
        & output["required_core_years_positive"]
    )
    output["passes_construction_gate"] = (
        output["passes_construction_gate_before_family_bh"]
        & output["family_q_bh"].le(float(family_bh_q_max))
    )
    return output.drop(columns="passes_confirmation_gate")


def select_candidate_v54(summary: pd.DataFrame) -> str | None:
    passed = summary[summary["passes_construction_gate"]].copy()
    if passed.empty:
        return None
    order = {value: index for index, value in enumerate(CANDIDATES_V54)}
    passed["registry_order"] = passed["candidate"].map(order)
    passed = passed.sort_values(
        ["mean_net_spread_return", "registry_order"],
        ascending=[False, True],
        kind="mergesort",
    )
    return str(passed.iloc[0]["candidate"])


__all__ = [
    "CANDIDATES_V54",
    "CONTROL_COLUMNS_V54",
    "FEATURE_COLUMNS_V54",
    "apply_construction_gate_v54",
    "fit_final_model_v54",
    "make_model_v54",
    "predict_frozen_model_v54",
    "rank_feature_matrix_v54",
    "select_candidate_v54",
    "signed_tail_target_v54",
    "walk_forward_predictions_v54",
]
