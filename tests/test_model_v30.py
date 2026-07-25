from ashare_alpha.model_v30 import (
    FEATURES_V30,
    KNOWN_FEATURES_V30,
    MODEL_PARAMETERS_V30,
    NOVEL_FEATURES_V30,
)


def test_v30_feature_registry_and_model_parameters_are_frozen() -> None:
    assert len(KNOWN_FEATURES_V30) == 14
    assert len(NOVEL_FEATURES_V30) == 19
    assert FEATURES_V30 == [*KNOWN_FEATURES_V30, *NOVEL_FEATURES_V30]
    assert MODEL_PARAMETERS_V30["max_leaf_nodes"] == 7
    assert MODEL_PARAMETERS_V30["max_iter"] == 100
    assert MODEL_PARAMETERS_V30["early_stopping"] is False

