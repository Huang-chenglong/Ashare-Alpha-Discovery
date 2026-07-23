from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import Candidate, CONTROL_COLUMNS, _rolling, compute_candidates


ABSORPTION_SCALES = (0.25, 0.50, 1.00)
CANDIDATES_V3 = {
    **{
        f"remaining_float_inventory_fraction_{int(scale * 100):03d}": Candidate(
            "dynamic_float_supply",
            f"Less remaining float supply after turnover-scaled absorption ({scale:.2f}) predicts higher returns.",
        )
        for scale in ABSORPTION_SCALES
    },
    **{
        f"remaining_float_inventory_days_{int(scale * 100):03d}": Candidate(
            "dynamic_float_supply",
            f"Fewer normal-volume days of remaining float supply ({scale:.2f}) predicts higher returns.",
        )
        for scale in ABSORPTION_SCALES
    },
}
CANDIDATE_COLUMNS_V3 = list(CANDIDATES_V3)
CONTROL_COLUMNS_V3 = [*CONTROL_COLUMNS, "float_supply_days_120"]


def _remaining_inventory(
    frame: pd.DataFrame,
    supply_fraction: pd.Series,
    turnover_fraction: pd.Series,
    absorption_scale: float,
) -> pd.Series:
    inventory = np.full(len(frame), np.nan, dtype=float)
    supply = supply_fraction.fillna(0.0).to_numpy(dtype=float)
    turnover = turnover_fraction.fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    for positions in frame.groupby("asset", sort=False).indices.values():
        state = 0.0
        for position in positions:
            state = (state + supply[position]) * np.exp(
                -turnover[position] / absorption_scale
            )
            inventory[position] = state
    return pd.Series(inventory, index=frame.index)


def compute_candidates_v3(daily: pd.DataFrame) -> pd.DataFrame:
    frame = compute_candidates(daily).reset_index(drop=True)
    previous_float = frame.groupby("asset", sort=False)["float_shares"].shift(1)
    new_float = (frame["float_shares"] - previous_float).clip(lower=0.0)
    supply_fraction = new_float / frame["float_shares"].where(
        frame["float_shares"].gt(0.0)
    )
    usable_turnover = frame["turnover_fraction"].where(
        frame["turnover_fraction"].gt(0.0)
    )
    frame["usable_turnover_v3"] = usable_turnover
    normal_turnover = _rolling(frame, "usable_turnover_v3", 20, "mean")

    for scale in ABSORPTION_SCALES:
        suffix = f"{int(scale * 100):03d}"
        inventory = _remaining_inventory(
            frame,
            supply_fraction,
            usable_turnover,
            scale,
        )
        frame[f"remaining_float_inventory_fraction_{suffix}"] = -np.log1p(
            inventory
        )
        inventory_days = inventory / normal_turnover.replace(0.0, np.nan)
        frame[f"remaining_float_inventory_days_{suffix}"] = -np.log1p(
            inventory_days
        )

    return frame.drop(columns=["usable_turnover_v3"])
