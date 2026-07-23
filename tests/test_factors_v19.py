from ashare_alpha.factors_v19 import CANDIDATE_COLUMNS_V19


def test_v19_has_one_frozen_equal_weight_candidate() -> None:
    assert CANDIDATE_COLUMNS_V19 == ["mspc_equal_weight"]
