import pandas as pd

from ashare_alpha.composite_v41 import (
    COMPONENT_WEIGHTS_V41,
    combine_component_scores_v41,
)


def test_v41_component_weights_are_fixed_equal_and_sum_to_one() -> None:
    assert COMPONENT_WEIGHTS_V41 == {
        "nsim_v30_score": 0.5,
        "cfma_v40_score": 0.5,
    }
    assert sum(COMPONENT_WEIGHTS_V41.values()) == 1.0


def test_v41_combines_date_ranks_without_outcome_input() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-31", "2024-01-31", "2024-01-31"]
            ),
            "nsim_v30_score": [3.0, 2.0, 1.0],
            "cfma_v40_score": [1.0, 2.0, 3.0],
        }
    )
    combined = combine_component_scores_v41(frame)
    assert combined.nunique() == 1
    assert combined.iloc[0] == combined.iloc[1] == combined.iloc[2]
