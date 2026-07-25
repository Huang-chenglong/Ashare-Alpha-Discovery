from ashare_alpha.factors_v26 import (
    CANDIDATE_COLUMNS_V26,
    EARLY_WINDOW_V26,
    HORIZONS_V26,
    LATE_WINDOW_V26,
    MINIMUM_EARLY_OBSERVATIONS_V26,
    MINIMUM_LATE_OBSERVATIONS_V26,
    SUPPLY_EVENT_THRESHOLD_V26,
)


def test_v26_protocol_constants_are_frozen() -> None:
    assert HORIZONS_V26 == (100, 120)
    assert CANDIDATE_COLUMNS_V26 == ["fsid_100", "fsid_120"]
    assert SUPPLY_EVENT_THRESHOLD_V26 == 0.001
    assert EARLY_WINDOW_V26 == (1, 5)
    assert LATE_WINDOW_V26 == (6, 20)
    assert MINIMUM_EARLY_OBSERVATIONS_V26 == 4
    assert MINIMUM_LATE_OBSERVATIONS_V26 == 12
