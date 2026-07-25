from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import align_industry_history, attach_execution_returns, load_formation_daily
from .evaluate import _neutralize_cross_section, exposure_diagnostics
from .margin_modalities_v53 import (
    CONTROL_COLUMNS_V53,
    build_margin_modalities_v53,
)
from .model_v53 import rank_feature_matrix_v53


def build_model_source_panel_v53(
    *,
    structural_path: str | Path,
    hfq_cache_directory: str | Path,
    industry_path: str | Path,
    dense_margin_path: str | Path,
    start_date: str,
    end_date: str,
    entry_offset: int = 2,
    exit_offset: int = 22,
    maximum_delay: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    daily, prices = load_formation_daily(structural_path, hfq_cache_directory)
    features, coverage = build_margin_modalities_v53(
        daily,
        pd.read_parquet(dense_margin_path),
    )
    features = align_industry_history(features, pd.read_parquet(industry_path))
    features = features[features["date"].between(start_date, end_date)].copy()
    returns, return_manifest = attach_execution_returns(
        features[["date", "asset"]].drop_duplicates(),
        prices,
        daily["date"],
        entry_offset=entry_offset,
        exit_offset=exit_offset,
        maximum_delay=maximum_delay,
    )
    outcomes = [
        "date",
        "asset",
        "entry_date",
        "exit_date",
        "entry_delay",
        "exit_delay",
        "return_status",
        "future_return_20",
        "label",
    ]
    panel = features.merge(
        returns[outcomes],
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    matrix = rank_feature_matrix_v53(panel)
    return (
        panel,
        matrix,
        coverage[coverage["date"].between(start_date, end_date)].copy(),
        return_manifest,
    )


def neutralize_model_predictions_v53(
    panel: pd.DataFrame,
    predictions: pd.Series,
    *,
    candidate: str,
    minimum_rows: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not predictions.index.equals(panel.index):
        raise ValueError("V53 predictions do not align with the source panel")
    source = panel.copy()
    source[candidate] = predictions
    scored_parts: list[pd.DataFrame] = []
    for _, rows in source.groupby("date", sort=True):
        scored = _neutralize_cross_section(
            rows,
            candidate,
            CONTROL_COLUMNS_V53,
            minimum_rows=minimum_rows,
        )
        if not scored.empty:
            scored_parts.append(scored)
    if not scored_parts:
        raise ValueError("V53 produced no complete neutralized cross-section")
    scored = pd.concat(scored_parts, ignore_index=True).sort_values(
        ["date", "asset"]
    )
    return scored, exposure_diagnostics(
        scored,
        control_columns=CONTROL_COLUMNS_V53,
    )


__all__ = [
    "build_model_source_panel_v53",
    "neutralize_model_predictions_v53",
]
