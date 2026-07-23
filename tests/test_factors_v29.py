from ashare_alpha.factors_v29 import (
    CANDIDATE_COLUMNS_V29,
    HORIZONS_V29,
    MINIMUM_EVENTS_PER_TAIL_V29,
)


def test_v29_protocol_constants_are_frozen() -> None:
    assert HORIZONS_V29 == (60, 90)
    assert CANDIDATE_COLUMNS_V29 == ["tsra_60", "tsra_90"]
    assert MINIMUM_EVENTS_PER_TAIL_V29 == 8
