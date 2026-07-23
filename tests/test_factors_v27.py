from ashare_alpha.factors_v27 import CANDIDATE_COLUMNS_V27, EVENT_CONTROL_STEMS_V27


def test_v27_registry_and_component_controls_are_frozen() -> None:
    assert CANDIDATE_COLUMNS_V27 == ["mfsic_equal_weight"]
    assert EVENT_CONTROL_STEMS_V27 == [
        "mean_completed_supply_fraction",
        "mean_completed_residual_return",
        "mean_completed_turnover",
        "mean_raw_absorption_slope",
        "mean_completed_event_age",
    ]
