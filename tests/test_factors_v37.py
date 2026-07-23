import numpy as np
import pandas as pd

from ashare_alpha.factors_v37 import (
    ANNOUNCEMENT_WINDOW_SESSIONS_V37,
    FINANCIAL_MAIN_EFFECTS_V37,
    _announcement_attention_for_asset_v37,
)


def _daily_turnover() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=55)
    turnover = np.full(len(dates), 0.01)
    turnover[45:50] = 0.02
    return pd.DataFrame(
        {
            "date": dates,
            "asset": "000001",
            "tradestatus": 1,
            "turnover_fraction": turnover,
        }
    )


def test_v37_announcement_attention_uses_completed_five_session_window() -> None:
    daily = _daily_turnover()
    events = pd.DataFrame(
        {
            "asset": ["000001"],
            "financial_available_date": [daily.loc[45, "date"]],
        }
    )
    result = _announcement_attention_for_asset_v37(events, daily)
    assert ANNOUNCEMENT_WINDOW_SESSIONS_V37 == 5
    assert result.loc[0, "event_window_completion_date"] == daily.loc[49, "date"]
    assert np.isclose(result.loc[0, "announcement_attention_5"], np.log(2.0))


def test_v37_attention_is_invariant_to_turnover_after_completion() -> None:
    daily = _daily_turnover()
    events = pd.DataFrame(
        {
            "asset": ["000001"],
            "financial_available_date": [daily.loc[45, "date"]],
        }
    )
    original = _announcement_attention_for_asset_v37(events, daily)
    changed = daily.copy()
    changed.loc[50:, "turnover_fraction"] = 100.0
    perturbed = _announcement_attention_for_asset_v37(events, changed)
    assert original.loc[0, "announcement_attention_5"] == perturbed.loc[
        0, "announcement_attention_5"
    ]


def test_v37_financial_main_effects_are_explicit() -> None:
    assert FINANCIAL_MAIN_EFFECTS_V37[:3] == [
        "positive_cash_lead_rank",
        "positive_cash_roa_rank",
        "low_announcement_attention_rank",
    ]
    assert "announcement_attention_5" in FINANCIAL_MAIN_EFFECTS_V37
