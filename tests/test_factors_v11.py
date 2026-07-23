from ashare_alpha.factors_v11 import CALIBRATED_HORIZON_V11, CANDIDATE_COLUMNS_V11


def test_v11_is_the_single_deterministic_midpoint() -> None:
    assert CALIBRATED_HORIZON_V11 == (60 + 120) // 2
    assert CANDIDATE_COLUMNS_V11 == ["joint_path_reversibility_90"]
