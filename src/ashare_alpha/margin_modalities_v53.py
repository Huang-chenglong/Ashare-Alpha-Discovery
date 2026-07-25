from __future__ import annotations

import numpy as np
import pandas as pd

from .margin_factors_v50 import (
    CONTROL_COLUMNS_V50,
    build_margin_path_candidates_v50,
)


NEW_MODALITY_RAW_FEATURES_V53 = (
    "financing_buy_to_amount",
    "financing_buy_to_balance",
    "financing_buy_surprise_4",
    "short_sell_value_to_amount",
    "short_sell_surprise_4",
    "short_inventory_change_to_amount",
    "short_inventory_path_4",
)

NEW_MODALITY_MAIN_EFFECTS_V53 = tuple(
    item
    for feature in NEW_MODALITY_RAW_FEATURES_V53
    for item in (f"{feature}_rank", f"{feature}_rank_squared")
)

CONTROL_COLUMNS_V53 = tuple(
    dict.fromkeys([*CONTROL_COLUMNS_V50, *NEW_MODALITY_MAIN_EFFECTS_V53])
)


def _snapshot_flow_features_v53(
    daily: pd.DataFrame,
    dense_margin: pd.DataFrame,
) -> pd.DataFrame:
    frame = daily.sort_values(["asset", "date"]).copy()
    frame["cumulative_amount"] = frame.groupby("asset", sort=False)[
        "amount"
    ].cumsum()
    margin = dense_margin.copy()
    margin["trade_date"] = pd.to_datetime(margin["trade_date"])
    margin["asset"] = margin["asset"].astype("string").str.zfill(6)
    snapshot_dates = pd.DatetimeIndex(
        margin["trade_date"].unique()
    ).sort_values()
    snapshot_order = pd.Series(range(len(snapshot_dates)), index=snapshot_dates)
    margin["snapshot_order"] = margin["trade_date"].map(snapshot_order).astype(int)
    snapshots = frame[
        ["date", "asset", "amount", "raw_close", "cumulative_amount"]
    ].merge(
        margin[
            [
                "trade_date",
                "asset",
                "snapshot_order",
                "financing_balance",
                "financing_buy",
                "short_balance_quantity",
                "short_sell_quantity",
            ]
        ],
        left_on=["date", "asset"],
        right_on=["trade_date", "asset"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["asset", "date"])
    snapshots["financing_buy_to_amount"] = (
        snapshots["financing_buy"] / snapshots["amount"].replace(0.0, np.nan)
    )
    snapshots["financing_buy_to_balance"] = (
        snapshots["financing_buy"]
        / snapshots["financing_balance"].replace(0.0, np.nan)
    )
    snapshots["short_sell_value_to_amount"] = (
        snapshots["short_sell_quantity"]
        * snapshots["raw_close"]
        / snapshots["amount"].replace(0.0, np.nan)
    )

    grouped = snapshots.groupby("asset", sort=False)
    previous_order = grouped["snapshot_order"].shift(1)
    consecutive = snapshots["snapshot_order"].sub(previous_order).eq(1)
    interval_amount = grouped["cumulative_amount"].diff().where(consecutive)
    short_quantity_change = grouped["short_balance_quantity"].diff().where(
        consecutive
    )
    snapshots["short_inventory_change_to_amount"] = (
        short_quantity_change
        * snapshots["raw_close"]
        / interval_amount.replace(0.0, np.nan)
    )

    fourth_previous_order = grouped["snapshot_order"].shift(4)
    complete_past_four = snapshots["snapshot_order"].sub(
        fourth_previous_order
    ).eq(4)
    surprise_names = {
        "financing_buy_to_amount": "financing_buy_surprise_4",
        "short_sell_value_to_amount": "short_sell_surprise_4",
    }
    for feature, surprise_name in surprise_names.items():
        past_mean = grouped[feature].transform(
            lambda values: values.shift(1).rolling(4, min_periods=4).mean()
        )
        surprise = snapshots[feature] / past_mean.replace(0.0, np.nan) - 1.0
        snapshots[surprise_name] = surprise.where(complete_past_four)
    short_path = grouped["short_inventory_change_to_amount"].transform(
        lambda values: values.rolling(4, min_periods=4).sum()
    )
    snapshots["short_inventory_path_4"] = short_path.where(
        complete_past_four
    )
    return snapshots[
        ["date", "asset", *NEW_MODALITY_RAW_FEATURES_V53]
    ]


def build_margin_modalities_v53(
    daily: pd.DataFrame,
    dense_margin: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base, coverage = build_margin_path_candidates_v50(daily, dense_margin)
    modalities = _snapshot_flow_features_v53(daily, dense_margin)
    panel = base.merge(
        modalities,
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    parts: list[pd.DataFrame] = []
    for _, rows in panel.groupby("date", sort=True):
        rows = rows.copy()
        for feature in NEW_MODALITY_RAW_FEATURES_V53:
            ranked = rows[feature].replace([np.inf, -np.inf], np.nan).rank(
                method="average",
                pct=True,
            )
            rows[f"{feature}_rank"] = ranked
            rows[f"{feature}_rank_squared"] = ranked.pow(2)
        parts.append(rows)
    panel = pd.concat(parts, ignore_index=True)
    complete = panel[list(NEW_MODALITY_MAIN_EFFECTS_V53)].notna().all(axis=1)
    modality_coverage = (
        panel.assign(complete_new_modalities=complete)
        .groupby("date")["complete_new_modalities"]
        .agg(["sum", "mean"])
        .reset_index()
        .rename(
            columns={
                "sum": "complete_new_modality_rows",
                "mean": "new_modality_coverage",
            }
        )
    )
    coverage = coverage.merge(
        modality_coverage,
        on="date",
        how="left",
        validate="one_to_one",
    )
    return panel.sort_values(["date", "asset"]).reset_index(drop=True), coverage


__all__ = [
    "CONTROL_COLUMNS_V53",
    "NEW_MODALITY_MAIN_EFFECTS_V53",
    "NEW_MODALITY_RAW_FEATURES_V53",
    "build_margin_modalities_v53",
]
