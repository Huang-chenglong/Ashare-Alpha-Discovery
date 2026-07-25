import numpy as np
import pandas as pd

from ashare_alpha.factors_v14 import _vwap_state


def test_vwap_state_maps_all_four_orderings() -> None:
    frame = pd.DataFrame(
        {
            "adjusted_vwap": [10.0] * 4,
            "open": [9.0, 9.0, 11.0, 11.0],
            "close": [9.5, 11.0, 10.5, 9.0],
            "tradestatus": [1] * 4,
        }
    )
    assert _vwap_state(frame).tolist() == [0, 1, 2, 3]


def test_vwap_state_rejects_flat_equality_and_bad_vwap() -> None:
    frame = pd.DataFrame(
        {
            "adjusted_vwap": [10.0, np.nan],
            "open": [10.0, 9.0],
            "close": [10.0, 11.0],
            "tradestatus": [1, 1],
        }
    )
    assert _vwap_state(frame).isna().all()
