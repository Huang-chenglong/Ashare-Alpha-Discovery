from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import align_industry_history, attach_execution_returns, load_formation_daily
from .evaluate import _neutralize_cross_section, exposure_diagnostics
from .margin_factors_v50 import (
    CANDIDATES_V50,
    CONTROL_COLUMNS_V50,
    build_margin_path_candidates_v50,
)


def build_evaluation_panel_v50(
    *,
    structural_path: str | Path,
    hfq_cache_directory: str | Path,
    industry_path: str | Path,
    dense_margin_path: str | Path,
    start_date: str,
    end_date: str,
    candidates: tuple[str, ...] = CANDIDATES_V50,
    minimum_rows: int = 150,
    entry_offset: int = 2,
    exit_offset: int = 22,
    maximum_delay: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    unknown = set(candidates).difference(CANDIDATES_V50)
    if unknown:
        raise ValueError(f"Unknown V50 candidates: {sorted(unknown)}")
    daily, prices = load_formation_daily(structural_path, hfq_cache_directory)
    features, coverage = build_margin_path_candidates_v50(
        daily, pd.read_parquet(dense_margin_path)
    )
    features = align_industry_history(features, pd.read_parquet(industry_path))
    features = features[features["date"].between(start_date, end_date)].copy()
    scored_parts: list[pd.DataFrame] = []
    for candidate in candidates:
        for _, rows in features.groupby("date", sort=True):
            scored = _neutralize_cross_section(
                rows,
                candidate,
                CONTROL_COLUMNS_V50,
                minimum_rows=minimum_rows,
            )
            if not scored.empty:
                scored_parts.append(scored)
    if not scored_parts:
        raise ValueError("V50 produced no complete neutralized cross-section")
    scored = pd.concat(scored_parts, ignore_index=True)
    returns, return_manifest = attach_execution_returns(
        scored[["date", "asset"]].drop_duplicates(),
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
    evaluated = scored.merge(
        returns[outcomes],
        on=["date", "asset"],
        how="left",
        validate="many_to_one",
    )
    exposures = exposure_diagnostics(scored, control_columns=CONTROL_COLUMNS_V50)
    return (
        evaluated,
        exposures,
        coverage[coverage["date"].between(start_date, end_date)].copy(),
        return_manifest,
    )


__all__ = ["build_evaluation_panel_v50"]
