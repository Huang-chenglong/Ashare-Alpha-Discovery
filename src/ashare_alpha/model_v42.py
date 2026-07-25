from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .factors_v40 import (
    FINANCIAL_MAIN_EFFECTS_V40,
    build_monthly_financial_panel_v40,
    finalize_candidates_v40,
)
from .model_v30 import KNOWN_FEATURES_V30
from .statistics import mad_winsorize, zscore


CANDIDATE_V42 = "tafs_v42"
FEATURES_V42 = [*KNOWN_FEATURES_V30, *FINANCIAL_MAIN_EFFECTS_V40]
PREDICTION_YEARS_V42 = (2023, 2024, 2025)
TOP_TAIL_QUANTILE_V42 = 0.80
MODEL_PARAMETERS_V42 = {
    "loss": "log_loss",
    "learning_rate": 0.03,
    "max_iter": 150,
    "max_leaf_nodes": 7,
    "min_samples_leaf": 200,
    "l2_regularization": 20.0,
    "max_bins": 63,
    "early_stopping": False,
    "random_state": 42,
}


def build_monthly_features_v42(
    daily: pd.DataFrame,
    financial_history: pd.DataFrame,
) -> pd.DataFrame:
    return build_monthly_financial_panel_v40(daily, financial_history)


def rank_feature_matrix_v42(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    values: dict[str, pd.Series] = {}
    columns: list[str] = []
    for feature in FEATURES_V42:
        finite = panel[feature].replace([np.inf, -np.inf], np.nan)
        ranked = finite.groupby(panel["date"]).rank(method="average", pct=True)
        column = f"rank__{feature}"
        values[column] = ranked.fillna(0.5)
        columns.append(column)
    return pd.DataFrame(values, index=panel.index), columns


def main_effect_controls_v42(
    matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    controls: dict[str, pd.Series] = {}
    for column in matrix.columns:
        centered = matrix[column] - 0.5
        controls[f"linear__{column}"] = centered
        controls[f"quadratic__{column}"] = centered.pow(2)
    frame = pd.DataFrame(controls, index=matrix.index)
    return frame, list(frame.columns)


def complete_feature_rows_v42(panel: pd.DataFrame) -> pd.Series:
    finite = panel[FEATURES_V42].replace([np.inf, -np.inf], np.nan)
    return (
        finite.notna().all(axis=1)
        & panel["float_market_cap"].gt(0.0)
        & panel["industry_l1"].notna()
    )


def residualize_training_target_v42(panel: pd.DataFrame) -> pd.Series:
    output = pd.Series(np.nan, index=panel.index, dtype=float)
    required = ["label", "float_market_cap", "industry_l1", *FEATURES_V42]
    for _, rows in panel.groupby("date", sort=True):
        finite = rows[required].replace([np.inf, -np.inf], np.nan)
        valid = finite.notna().all(axis=1) & rows["float_market_cap"].gt(0.0)
        sample = rows.loc[valid]
        if len(sample) < 300:
            continue
        target = mad_winsorize(sample["label"])
        size_z = zscore(np.log(sample["float_market_cap"]))
        size_squared = size_z.pow(2) - size_z.pow(2).mean()
        controls = [
            zscore(mad_winsorize(sample[column])).to_numpy()
            for column in FEATURES_V42
        ]
        industries = pd.get_dummies(
            sample["industry_l1"], drop_first=True, dtype=float
        )
        design = np.column_stack(
            [
                np.ones(len(sample)),
                size_z.to_numpy(),
                size_squared.to_numpy(),
                *controls,
                *industries.to_numpy().T,
            ]
        )
        coefficients = np.linalg.lstsq(
            design, target.to_numpy(), rcond=None
        )[0]
        output.loc[sample.index] = target.to_numpy() - design @ coefficients
    return output


def make_top_tail_labels_v42(
    dates: pd.Series,
    residual_target: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    labels = pd.Series(np.nan, index=residual_target.index, dtype=float)
    weights = pd.Series(np.nan, index=residual_target.index, dtype=float)
    for _, index in dates.loc[residual_target.notna()].groupby(
        dates.loc[residual_target.notna()]
    ).groups.items():
        target = residual_target.loc[index]
        threshold = float(target.quantile(TOP_TAIL_QUANTILE_V42))
        classified = target.ge(threshold).astype(int)
        positive = int(classified.sum())
        negative = int(len(classified) - positive)
        if positive == 0 or negative == 0:
            continue
        labels.loc[index] = classified
        weights.loc[index] = np.where(
            classified.eq(1),
            0.5 * len(classified) / positive,
            0.5 * len(classified) / negative,
        )
    return labels, weights


def _fit_model_v42(
    matrix: pd.DataFrame,
    labels: pd.Series,
    weights: pd.Series,
) -> HistGradientBoostingClassifier:
    valid = labels.notna() & weights.notna()
    if int(valid.sum()) < 10_000:
        raise ValueError("V42 has too few complete training rows")
    model = HistGradientBoostingClassifier(**MODEL_PARAMETERS_V42)
    model.fit(
        matrix.loc[valid],
        labels.loc[valid].astype(int),
        sample_weight=weights.loc[valid],
    )
    return model


def fit_walk_forward_models_v42(
    panel: pd.DataFrame,
    matrix: pd.DataFrame,
) -> tuple[
    pd.Series,
    dict[int, HistGradientBoostingClassifier],
    pd.DataFrame,
]:
    dates = pd.to_datetime(panel["date"])
    predictions = pd.Series(np.nan, index=panel.index, dtype=float)
    models: dict[int, HistGradientBoostingClassifier] = {}
    audit: list[dict[str, object]] = []
    eligible = complete_feature_rows_v42(panel)

    for year in PREDICTION_YEARS_V42:
        training = dates.between("2020-01-01", f"{year - 1}-12-31")
        target = residualize_training_target_v42(panel.loc[training])
        labels, weights = make_top_tail_labels_v42(
            dates.loc[training], target
        )
        model = _fit_model_v42(matrix.loc[training], labels, weights)
        prediction = dates.dt.year.eq(year) & eligible
        predictions.loc[prediction] = model.predict_proba(
            matrix.loc[prediction]
        )[:, 1]
        models[year] = model
        training_label_index = labels.index[labels.notna()]
        audit.append(
            {
                "prediction_year": year,
                "training_end": f"{year - 1}-12-31",
                "training_rows": int(labels.notna().sum()),
                "training_assets": int(
                    panel.loc[training_label_index, "asset"].nunique()
                ),
                "prediction_rows": int(prediction.sum()),
                "prediction_assets": int(
                    panel.loc[prediction, "asset"].nunique()
                ),
            }
        )
    return predictions, models, pd.DataFrame(audit)


def predict_walk_forward_v42(
    panel: pd.DataFrame,
    matrix: pd.DataFrame,
    models: dict[int, HistGradientBoostingClassifier],
) -> pd.Series:
    dates = pd.to_datetime(panel["date"])
    eligible = complete_feature_rows_v42(panel)
    predictions = pd.Series(np.nan, index=panel.index, dtype=float)
    for year in PREDICTION_YEARS_V42:
        if year not in models:
            raise ValueError(f"V42 model bundle is missing prediction year {year}")
        selected = dates.dt.year.eq(year) & eligible
        if not selected.any():
            continue
        predictions.loc[selected] = models[year].predict_proba(
            matrix.loc[selected]
        )[:, 1]
    return predictions


def finalize_features_v42(panel: pd.DataFrame) -> pd.DataFrame:
    return finalize_candidates_v40(panel)
