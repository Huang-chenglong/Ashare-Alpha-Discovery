import numpy as np
import pandas as pd

from ashare_alpha.confirmation import evaluate_confirmation


def test_confirmation_gate_requires_every_condition(monkeypatch) -> None:
    dates = pd.date_range("2020-01-31", periods=75, freq="ME")
    monthly = pd.DataFrame(
        {
            "date": dates,
            "candidate": "test",
            "rank_ic": 0.02,
            "net_active_return": 0.01 + 0.005 * np.sin(np.arange(75)),
            "traded_notional": 0.5,
            "return_coverage": 1.0,
            "selected_return_coverage": 1.0,
        }
    )
    monkeypatch.setattr("ashare_alpha.confirmation.buffered_long_only", lambda _: monthly)
    monkeypatch.setattr("ashare_alpha.confirmation.hac_mean_test", lambda _: (3.0, 0.001))
    panel = pd.DataFrame({"candidate": ["test"]})
    summary, _, _ = evaluate_confirmation(panel)
    assert bool(summary.iloc[0]["passes_confirmation_gate"])
    failed, _, _ = evaluate_confirmation(panel, mean_ic_minimum=0.03)
    assert not bool(failed.iloc[0]["passes_confirmation_gate"])

    monthly_failed, _, _ = evaluate_confirmation(
        panel, monthly_return_coverage_minimum=1.01
    )
    selected_failed, _, _ = evaluate_confirmation(
        panel, selected_return_coverage_minimum=1.01
    )
    assert not bool(monthly_failed.iloc[0]["passes_confirmation_gate"])
    assert not bool(selected_failed.iloc[0]["passes_confirmation_gate"])
