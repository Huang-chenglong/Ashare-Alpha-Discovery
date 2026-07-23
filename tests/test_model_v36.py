import pandas as pd

from ashare_alpha.model_v30 import KNOWN_FEATURES_V30, NOVEL_FEATURES_V30
from ashare_alpha.model_v36 import (
    FEATURES_V36,
    FINANCIAL_FEATURES_V36,
    FOLD_COUNT_V36,
    MODEL_PARAMETERS_V36,
    main_effect_controls_v36,
    rank_feature_matrix_v36,
    stable_asset_fold_v36,
)


def test_v36_registry_and_estimator_are_frozen() -> None:
    assert len(KNOWN_FEATURES_V30) == 14
    assert len(NOVEL_FEATURES_V30) == 19
    assert len(FINANCIAL_FEATURES_V36) == 19
    assert len(FEATURES_V36) == 52
    assert MODEL_PARAMETERS_V36["max_leaf_nodes"] == 7
    assert MODEL_PARAMETERS_V36["early_stopping"] is False
    assert FOLD_COUNT_V36 == 5


def test_v36_stable_asset_fold_is_deterministic() -> None:
    first = [stable_asset_fold_v36(asset) for asset in ("000001", "600000", "300750")]
    second = [stable_asset_fold_v36(asset) for asset in ("000001", "600000", "300750")]
    assert first == second
    assert all(0 <= fold < FOLD_COUNT_V36 for fold in first)


def test_v36_rank_matrix_and_main_effect_controls_are_complete() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-01-31"]),
            **{
                feature: [1.0, float("nan")]
                for feature in FEATURES_V36
            },
        }
    )
    matrix, columns = rank_feature_matrix_v36(frame)
    controls, control_columns = main_effect_controls_v36(matrix[columns])
    assert matrix.notna().all().all()
    assert controls.notna().all().all()
    expected_missing = len(NOVEL_FEATURES_V30) + len(FINANCIAL_FEATURES_V36)
    assert len(columns) == len(FEATURES_V36) + expected_missing
    assert len(control_columns) == 2 * len(FEATURES_V36) + expected_missing
