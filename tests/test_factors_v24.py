from ashare_alpha.factors_v24 import (
    CANCELLATION_RATIO_MAX_V24,
    CANDIDATE_COLUMNS_V24,
    HORIZONS_V24,
    MINIMUM_EVENTS_V24,
)


def test_v24_protocol_constants_are_frozen() -> None:
    assert HORIZONS_V24 == (60, 90)
    assert CANDIDATE_COLUMNS_V24 == ["mepi_60", "mepi_90"]
    assert CANCELLATION_RATIO_MAX_V24 == 0.25
    assert MINIMUM_EVENTS_V24 == 8
