from ashare_alpha.factors_v21 import CANDIDATE_COLUMNS_V21


def test_v21_registry_is_frozen() -> None:
    assert CANDIDATE_COLUMNS_V21 == ["drra_60", "drra_90"]
