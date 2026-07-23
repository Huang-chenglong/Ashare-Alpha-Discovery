import numpy as np
import pandas as pd

from ashare_alpha.evaluate import _neutralize_cross_section
from ashare_alpha.factors import CONTROL_COLUMNS


def test_neutralization_removes_linear_size_industry_and_baselines() -> None:
    rng = np.random.default_rng(7)
    count = 240
    size = rng.normal(size=count)
    industry = np.asarray([f"I{index % 6}" for index in range(count)])
    industry_effect = np.asarray([int(value[1:]) * 0.3 for value in industry])
    controls = {column: rng.normal(size=count) for column in CONTROL_COLUMNS}
    candidate = 1.2 * size + 0.4 * size**2 + industry_effect
    for values in controls.values():
        candidate = candidate + 0.5 * values
    candidate = candidate + rng.normal(scale=0.2, size=count)
    frame = pd.DataFrame(
        {
            "date": pd.Timestamp("2024-01-31"),
            "asset": [f"{index:06d}" for index in range(count)],
            "diffuse_turnover_20": candidate,
            "float_market_cap": np.exp(size + 20.0),
            "industry_l1": industry,
            **controls,
        }
    )
    result = _neutralize_cross_section(frame, "diffuse_turnover_20")
    assert abs(result["score"].corr(result["size_z"])) < 1e-10
    assert abs(result["score"].corr(result["size_squared"])) < 1e-10
    assert result.groupby("industry_l1")["score"].mean().abs().max() < 1e-10
    for column in CONTROL_COLUMNS:
        transformed = result[f"neutralization_control_{column}"]
        assert abs(result["score"].corr(transformed)) < 1e-10
