import numpy as np
import pandas as pd

from ashare_alpha.model_v42 import (
    FEATURES_V42,
    MODEL_PARAMETERS_V42,
    PREDICTION_YEARS_V42,
    TOP_TAIL_QUANTILE_V42,
    main_effect_controls_v42,
    make_top_tail_labels_v42,
)


def test_v42_registration_is_frozen() -> None:
    assert len(FEATURES_V42) == 29
    assert PREDICTION_YEARS_V42 == (2023, 2024, 2025)
    assert TOP_TAIL_QUANTILE_V42 == 0.80
    assert MODEL_PARAMETERS_V42["random_state"] == 42
    assert MODEL_PARAMETERS_V42["early_stopping"] is False


def test_top_tail_labels_are_balanced_by_month() -> None:
    dates = pd.Series(pd.to_datetime(["2022-01-31"] * 10 + ["2022-02-28"] * 10))
    target = pd.Series(np.tile(np.arange(10, dtype=float), 2))
    labels, weights = make_top_tail_labels_v42(dates, target)
    assert labels.groupby(dates).sum().eq(2).all()
    weighted = pd.DataFrame({"date": dates, "label": labels, "weight": weights})
    for _, rows in weighted.groupby("date"):
        positive = rows.loc[rows["label"].eq(1), "weight"].sum()
        negative = rows.loc[rows["label"].eq(0), "weight"].sum()
        assert np.isclose(positive, negative)


def test_v42_controls_remove_linear_and_quadratic_main_effects() -> None:
    matrix = pd.DataFrame({"rank__a": [0.1, 0.5, 0.9]})
    controls, columns = main_effect_controls_v42(matrix)
    assert columns == ["linear__rank__a", "quadratic__rank__a"]
    assert np.allclose(controls["linear__rank__a"], [-0.4, 0.0, 0.4])
    assert np.allclose(controls["quadratic__rank__a"], [0.16, 0.0, 0.16])
