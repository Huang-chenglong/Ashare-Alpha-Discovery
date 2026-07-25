from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .factors import CANDIDATE_COLUMNS, CONTROL_COLUMNS
from .statistics import benjamini_hochberg


CANDIDATES_V52 = (
    "ridge10_v52",
    "hgb7_v52",
    "extra4_v52",
)

FEATURE_COLUMNS_V52 = tuple(
    dict.fromkeys(
        [
            "mdar_v48",
            "mcar_v48",
            "mlcf_v48",
            "mpbr_v50",
            "mpsr_v50",
            "mpcs_v50",
            "financing_leverage",
            "net_financing_flow_to_turnover",
            "momentum_20",
            "short_leverage",
            "short_sell_intensity",
            "path_inflow_breadth",
            "path_signed_ratio",
            "path_dispersion_ratio",
            *CANDIDATE_COLUMNS,
            *CONTROL_COLUMNS,
        ]
    )
)


@dataclass(frozen=True)
class ModelSpecV52:
    candidate: str
    description: str


MODEL_SPECS_V52 = {
    "ridge10_v52": ModelSpecV52(
        "ridge10_v52",
        "standardized ridge with alpha 10",
    ),
    "hgb7_v52": ModelSpecV52(
        "hgb7_v52",
        "depth-limited histogram gradient boosting with at most seven leaves",
    ),
    "extra4_v52": ModelSpecV52(
        "extra4_v52",
        "depth-four extremely randomized trees with large terminal leaves",
    ),
}


def make_model_v52(candidate: str):
    if candidate == "ridge10_v52":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if candidate == "hgb7_v52":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=7,
            max_depth=None,
            min_samples_leaf=80,
            l2_regularization=10.0,
            early_stopping=False,
            random_state=20260724,
        )
    if candidate == "extra4_v52":
        return ExtraTreesRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=50,
            max_features=0.75,
            bootstrap=False,
            n_jobs=-1,
            random_state=20260724,
        )
    raise ValueError(f"Unknown V52 model {candidate}")


def rank_feature_matrix_v52(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURE_COLUMNS_V52).difference(frame.columns)
    if missing:
        raise ValueError(f"V52 feature panel is missing {sorted(missing)}")
    ranked = pd.DataFrame(index=frame.index)
    for column in FEATURE_COLUMNS_V52:
        ranked[column] = frame.groupby("date", sort=False)[column].rank(
            method="average",
            pct=True,
        )
    return ranked.replace([np.inf, -np.inf], np.nan)


def walk_forward_predictions_v52(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    candidate: str,
    prediction_start: str,
    prediction_end: str,
    minimum_training_rows: int = 2500,
) -> pd.Series:
    if candidate not in CANDIDATES_V52:
        raise ValueError(f"Unknown V52 candidate {candidate}")
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    dates = sorted(
        frame.loc[
            frame["date"].between(prediction_start, prediction_end), "date"
        ].unique()
    )
    complete_features = matrix.notna().all(axis=1)
    complete_label = frame["label"].notna() & frame["exit_date"].notna()
    for prediction_date in dates:
        train = (
            complete_features
            & complete_label
            & frame["exit_date"].lt(prediction_date)
        )
        predict = complete_features & frame["date"].eq(prediction_date)
        if int(train.sum()) < minimum_training_rows or not predict.any():
            continue
        model = make_model_v52(candidate)
        model.fit(matrix.loc[train], frame.loc[train, "label"])
        output.loc[predict] = model.predict(matrix.loc[predict])
    return output


def fit_final_model_v52(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    candidate: str,
    label_available_before: str,
    minimum_training_rows: int = 2500,
) -> dict[str, object]:
    complete = (
        matrix.notna().all(axis=1)
        & frame["label"].notna()
        & frame["exit_date"].notna()
        & frame["exit_date"].lt(label_available_before)
    )
    if int(complete.sum()) < minimum_training_rows:
        raise ValueError("V52 final model has too few point-in-time training rows")
    model = make_model_v52(candidate)
    model.fit(matrix.loc[complete], frame.loc[complete, "label"])
    return {
        "candidate": candidate,
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS_V52),
        "training_rows": int(complete.sum()),
        "training_assets": int(frame.loc[complete, "asset"].nunique()),
        "training_date_min": frame.loc[complete, "date"].min().strftime("%Y-%m-%d"),
        "training_date_max": frame.loc[complete, "date"].max().strftime("%Y-%m-%d"),
        "maximum_training_exit_date": frame.loc[complete, "exit_date"]
        .max()
        .strftime("%Y-%m-%d"),
        "label_available_before": label_available_before,
    }


def predict_frozen_model_v52(
    bundle: dict[str, object],
    matrix: pd.DataFrame,
) -> pd.Series:
    if list(matrix.columns) != bundle["feature_columns"]:
        raise ValueError("V52 frozen model feature columns differ")
    complete = matrix.notna().all(axis=1)
    output = pd.Series(np.nan, index=matrix.index, dtype=float)
    output.loc[complete] = bundle["model"].predict(matrix.loc[complete])
    return output


def apply_construction_gate_v52(
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
    for candidate in CANDIDATES_V52:
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


def select_candidate_v52(summary: pd.DataFrame) -> str | None:
    passed = summary[summary["passes_construction_gate"]].copy()
    if passed.empty:
        return None
    registry_order = {
        candidate: index for index, candidate in enumerate(CANDIDATES_V52)
    }
    passed["registry_order"] = passed["candidate"].map(registry_order)
    passed = passed.sort_values(
        ["mean_net_active_return", "registry_order"],
        ascending=[False, True],
        kind="mergesort",
    )
    return str(passed.iloc[0]["candidate"])


__all__ = [
    "CANDIDATES_V52",
    "FEATURE_COLUMNS_V52",
    "MODEL_SPECS_V52",
    "apply_construction_gate_v52",
    "fit_final_model_v52",
    "make_model_v52",
    "predict_frozen_model_v52",
    "rank_feature_matrix_v52",
    "select_candidate_v52",
    "walk_forward_predictions_v52",
]
