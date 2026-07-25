from ashare_alpha.factors_v22 import CANDIDATE_COLUMNS_V22


def test_v22_registry_is_frozen() -> None:
    assert CANDIDATE_COLUMNS_V22 == ["pwrr_60", "pwrr_90"]
