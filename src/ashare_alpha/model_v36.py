from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .factors_v32 import (
    FINANCIAL_MAIN_EFFECTS_V32,
    build_quarterly_financial_features_v32,
)
from .factors_v33 import attach_point_in_time_financial_v33
from .factors_v35 import (
    EXCLUDED_INDUSTRIES_V35,
    FINANCIAL_MAIN_EFFECTS_V35,
    build_quarterly_financial_features_v35,
)
from .model_v30 import (
    KNOWN_FEATURES_V30,
    MODEL_PARAMETERS_V30,
    NOVEL_FEATURES_V30,
    build_monthly_features_v30,
)
from .statistics import mad_winsorize, zscore


CANDIDATE_V36 = "fsim_v36"
FOLD_COUNT_V36 = 5
FINANCIAL_FEATURES_V36 = [
    *FINANCIAL_MAIN_EFFECTS_V32[3:],
    *FINANCIAL_MAIN_EFFECTS_V35[3:],
]
FEATURES_V36 = [
    *KNOWN_FEATURES_V30,
    *NOVEL_FEATURES_V30,
    *FINANCIAL_FEATURES_V36,
]
MISSING_AWARE_FEATURES_V36 = [
    *NOVEL_FEATURES_V30,
    *FINANCIAL_FEATURES_V36,
]
MODEL_PARAMETERS_V36 = dict(MODEL_PARAMETERS_V30)
DISCOVERY_START_V36 = pd.Timestamp("2020-01-01")
DISCOVERY_END_V36 = pd.Timestamp("2022-12-31")
VALIDATION_START_V36 = pd.Timestamp("2023-01-01")
VALIDATION_END_V36 = pd.Timestamp("2024-12-31")


def stable_asset_fold_v36(asset: object) -> int:
    digest = hashlib.sha256(str(asset).zfill(6).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % FOLD_COUNT_V36


def build_quarterly_financial_features_v36(
    financial_history: pd.DataFrame,
) -> pd.DataFrame:
    ownership = build_quarterly_financial_features_v32(financial_history)
    quality = build_quarterly_financial_features_v35(financial_history)
    keys = ["asset", "report_date", "financial_available_date"]
    ownership_columns = [*keys, *FINANCIAL_MAIN_EFFECTS_V32[3:]]
    quality_columns = [*keys, *FINANCIAL_MAIN_EFFECTS_V35[3:]]
    return (
        ownership[ownership_columns]
        .merge(
            quality[quality_columns],
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        .sort_values(["asset", "financial_available_date", "report_date"])
        .reset_index(drop=True)
    )


def build_monthly_features_v36(
    daily: pd.DataFrame,
    financial_history: pd.DataFrame,
) -> pd.DataFrame:
    structural = build_monthly_features_v30(daily)
    quarterly = build_quarterly_financial_features_v36(financial_history)
    return attach_point_in_time_financial_v33(structural, quarterly)


def exclude_financial_industries_v36(panel: pd.DataFrame) -> pd.DataFrame:
    excluded = panel["industry_l1"].astype("string").isin(EXCLUDED_INDUSTRIES_V35)
    return panel.loc[~excluded].copy()


def rank_feature_matrix_v36(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    values: dict[str, pd.Series] = {}
    columns: list[str] = []
    for feature in FEATURES_V36:
        finite = panel[feature].replace([np.inf, -np.inf], np.nan)
        ranked = finite.groupby(panel["date"]).rank(method="average", pct=True)
        rank_column = f"rank__{feature}"
        values[rank_column] = ranked.fillna(0.5)
        columns.append(rank_column)
        if feature in MISSING_AWARE_FEATURES_V36:
            missing_column = f"missing__{feature}"
            values[missing_column] = ranked.isna().astype(float)
            columns.append(missing_column)
    return pd.DataFrame(values, index=panel.index), columns


def main_effect_controls_v36(
    matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    controls: dict[str, pd.Series] = {}
    for column in matrix.columns:
        if column.startswith("rank__"):
            centered = matrix[column] - 0.5
            controls[f"linear__{column}"] = centered
            controls[f"quadratic__{column}"] = centered.pow(2)
        elif column.startswith("missing__"):
            controls[column] = matrix[column]
    frame = pd.DataFrame(controls, index=matrix.index)
    return frame, list(frame.columns)


def residualize_training_target_v36(panel: pd.DataFrame) -> pd.Series:
    output = pd.Series(np.nan, index=panel.index, dtype=float)
    required = ["label", "float_market_cap", "industry_l1", *KNOWN_FEATURES_V30]
    for _, rows in panel.groupby("date", sort=True):
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


def _fit_model_v36(
    matrix: pd.DataFrame,
    target: pd.Series,
) -> HistGradientBoostingRegressor:
    valid = target.notna()
    if int(valid.sum()) < 5_000:
        raise ValueError("V36 has too few complete training rows")
    model = HistGradientBoostingRegressor(**MODEL_PARAMETERS_V36)
    model.fit(matrix.loc[valid], target.loc[valid])
    return model


def fit_cross_fitted_model_v36(
    panel: pd.DataFrame,
    matrix: pd.DataFrame,
) -> tuple[pd.Series, list[HistGradientBoostingRegressor], HistGradientBoostingRegressor, pd.DataFrame]:
    dates = pd.to_datetime(panel["date"])
    discovery = dates.between(DISCOVERY_START_V36, DISCOVERY_END_V36)
    validation = dates.between(VALIDATION_START_V36, VALIDATION_END_V36)
    folds = panel["asset"].map(stable_asset_fold_v36)
    predictions = pd.Series(np.nan, index=panel.index, dtype=float)
    fold_models: list[HistGradientBoostingRegressor] = []
    audit_rows: list[dict[str, int]] = []

    for fold in range(FOLD_COUNT_V36):
        training = discovery & folds.ne(fold)
        held_out = discovery & folds.eq(fold)
        training_target = residualize_training_target_v36(panel.loc[training])
        model = _fit_model_v36(matrix.loc[training], training_target)
        predictions.loc[held_out] = model.predict(matrix.loc[held_out])
        fold_models.append(model)
        audit_rows.append(
            {
                "fold": fold,
                "training_rows": int(training_target.notna().sum()),
                "held_out_rows": int(held_out.sum()),
                "training_assets": int(panel.loc[training, "asset"].nunique()),
                "held_out_assets": int(panel.loc[held_out, "asset"].nunique()),
            }
        )

    final_target = residualize_training_target_v36(panel.loc[discovery])
    final_model = _fit_model_v36(matrix.loc[discovery], final_target)
    predictions.loc[validation] = final_model.predict(matrix.loc[validation])
    return predictions, fold_models, final_model, pd.DataFrame(audit_rows)
