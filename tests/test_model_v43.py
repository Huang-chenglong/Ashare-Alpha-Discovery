import numpy as np
import pandas as pd

from ashare_alpha.model_v43 import (
    FEATURES_V43,
    MODEL_PARAMETERS_V43,
    PREDICTION_YEARS_V43,
    make_rank_target_v43,
    predict_walk_forward_v43,
)


def test_v43_registration_is_frozen() -> None:
    assert len(FEATURES_V43) == 33
    assert PREDICTION_YEARS_V43 == (2023, 2024, 2025)
    assert MODEL_PARAMETERS_V43["random_state"] == 43
    assert MODEL_PARAMETERS_V43["early_stopping"] is False


def test_rank_target_is_centered_and_month_weighted() -> None:
    dates = pd.Series(
        pd.to_datetime(["2022-01-31"] * 4 + ["2022-02-28"] * 8)
    )
    residual = pd.Series(np.arange(12, dtype=float))
    target, weights = make_rank_target_v43(dates, residual)
    for date, rows in pd.DataFrame(
        {"date": dates, "target": target, "weight": weights}
    ).groupby("date"):
        assert np.isclose(rows["target"].mean(), 0.5 / len(rows))
        assert np.isclose(rows["weight"].sum(), 6.0)


def test_v43_prediction_skips_year_with_no_eligible_rows() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-31"]),
            "float_market_cap": [1.0],
            "industry_l1": ["10"],
            **{feature: [np.nan] for feature in FEATURES_V43},
        }
    )
    predictions = predict_walk_forward_v43(
        panel,
        pd.DataFrame({"x": [0.0]}),
        {year: object() for year in PREDICTION_YEARS_V43},
    )
    assert predictions.isna().all()
