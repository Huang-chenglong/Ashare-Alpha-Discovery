from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .model_v30 import FEATURES_V30, NOVEL_FEATURES_V30


RIDGE_PARAMETERS_V31 = {
    "alpha": 100.0,
    "solver": "lsqr",
    "tol": 0.000001,
    "max_iter": 5000,
    "fit_intercept": True,
}


def interaction_matrix_v31(
    panel: pd.DataFrame,
) -> tuple[np.ndarray, list[str], PolynomialFeatures]:
    ranked_columns = []
    ranked_values = []
    missing_values = []
    missing_names = []
    for feature in FEATURES_V30:
        rank = panel[feature].groupby(panel["date"]).rank(method="average", pct=True)
        ranked_columns.append(feature)
        ranked_values.append(rank.fillna(0.5).to_numpy(dtype=float) - 0.5)
        if feature in NOVEL_FEATURES_V30:
            missing_values.append(rank.isna().to_numpy(dtype=float))
            missing_names.append(f"missing__{feature}")
    base = np.column_stack(ranked_values)
    polynomial = PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=False,
    )
    interactions = polynomial.fit_transform(base)
    interaction_names = polynomial.get_feature_names_out(ranked_columns).tolist()
    matrix = np.column_stack([interactions, *missing_values])
    return matrix, [*interaction_names, *missing_names], polynomial


def fit_model_v31(
    matrix: np.ndarray,
    target: pd.Series,
    dates: pd.Series,
) -> Pipeline:
    training = dates.between("2020-01-01", "2022-12-31") & target.notna()
    if int(training.sum()) < 5_000:
        raise ValueError("V31 has too few complete training rows")
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(**RIDGE_PARAMETERS_V31)),
    ])
    model.fit(matrix[training.to_numpy()], target.loc[training].to_numpy())
    return model

