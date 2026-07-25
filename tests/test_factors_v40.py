from ashare_alpha.factors_v40 import FINANCIAL_MAIN_EFFECTS_V40


def test_v40_controls_current_and_prior_customer_financing_growth() -> None:
    assert FINANCIAL_MAIN_EFFECTS_V40[:3] == [
        "positive_customer_financing_growth_rank",
        "positive_customer_financing_acceleration_rank",
        "positive_revenue_growth_rank",
    ]
    assert "customer_financing_growth_yoy" in FINANCIAL_MAIN_EFFECTS_V40
    assert (
        "prior_quarter_customer_financing_growth_yoy"
        in FINANCIAL_MAIN_EFFECTS_V40
    )
    assert "customer_financing_acceleration" in FINANCIAL_MAIN_EFFECTS_V40
