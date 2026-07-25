from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml

from ashare_alpha.confirmation import evaluate_confirmation
from ashare_alpha.data import (
    align_industry_history,
    attach_execution_returns,
    load_formation_daily,
)
from ashare_alpha.evaluate import _neutralize_cross_section, exposure_diagnostics
from ashare_alpha.model_v43 import (
    CANDIDATE_V43,
    FEATURES_V43,
    KNOWN_FEATURES_V30,
    MODEL_PARAMETERS_V43,
    PREDICTION_YEARS_V43,
    build_monthly_features_v30,
    fit_walk_forward_models_v43,
    rank_feature_matrix_v30,
)
from run_confirmation_v41 import assert_hash, sha256


def build_construction_panel(
    data_paths: list[str],
    cache_paths: list[str],
    industry: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if len(data_paths) != len(cache_paths):
        raise ValueError("Every construction dataset requires one qfq cache")
    panels: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []
    asset_sets: list[set[str]] = []
    for data_path, cache_path in zip(data_paths, cache_paths, strict=True):
        daily, prices = load_formation_daily(data_path, cache_path)
        assets = set(daily["asset"].astype(str).str.zfill(6))
        asset_sets.append(assets)
        features = align_industry_history(
            build_monthly_features_v30(daily), industry
        )
        returns, return_manifest = attach_execution_returns(
            features[["date", "asset"]], prices, daily["date"]
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
        panels.append(
            features.merge(
                returns[outcomes],
                on=["date", "asset"],
                how="left",
                validate="one_to_one",
            )
        )
        manifests.append(return_manifest)
    for left in range(len(asset_sets)):
        for right in range(left + 1, len(asset_sets)):
            if asset_sets[left].intersection(asset_sets[right]):
                raise ValueError("V43 construction datasets have overlapping assets")
    panel = pd.concat(panels, ignore_index=True)
    if panel.duplicated(["date", "asset"]).any():
        raise ValueError("V43 combined panel has duplicate date/asset rows")
    return panel, manifests


def neutralize_predictions_v43(
    panel: pd.DataFrame,
    minimum_rows: int,
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    for _, rows in panel.groupby("date", sort=True):
        scored = _neutralize_cross_section(
            rows,
            CANDIDATE_V43,
            KNOWN_FEATURES_V30,
            minimum_rows=minimum_rows,
        )
        if not scored.empty:
            output.append(scored)
    if not output:
        raise ValueError("V43 produced no complete neutralized month")
    return pd.concat(output, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive E+F construction of frozen rank-target V43."
    )
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--hfq-cache", action="append", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["candidate"] != CANDIDATE_V43:
        raise ValueError("V43 candidate registry mismatch")
    if protocol["features"] != FEATURES_V43:
        raise ValueError("V43 feature registry mismatch")
    if protocol["training"]["fixed_hyperparameters"] != MODEL_PARAMETERS_V43:
        raise ValueError("V43 model parameters differ from protocol")
    if tuple(protocol["training"]["prediction_years"]) != PREDICTION_YEARS_V43:
        raise ValueError("V43 prediction years differ from protocol")
    for path, expected in zip(
        args.data, protocol["freeze"]["construction_data_sha256"], strict=True
    ):
        assert_hash(Path(path), expected, "V43 construction data")
    assert_hash(
        Path(args.industry),
        protocol["freeze"]["industry_sha256"],
        "industry history",
    )

    industry = pd.read_parquet(args.industry)
    panel, return_manifests = build_construction_panel(
        args.data, args.hfq_cache, industry
    )
    panel = panel[panel["date"].between("2020-01-01", "2025-12-31")].copy()
    matrix, model_columns = rank_feature_matrix_v30(panel)
    predictions, models, fit_audit = fit_walk_forward_models_v43(
        panel, matrix[model_columns]
    )
    panel[CANDIDATE_V43] = predictions
    scored = neutralize_predictions_v43(
        panel, minimum_rows=int(protocol["neutralization"]["minimum_rows"])
    )
    scored = scored[
        scored["date"].between("2023-01-01", "2025-12-31")
    ].copy()

    gate, portfolio = protocol["gate"], protocol["portfolio"]
    summary, yearly, monthly = evaluate_confirmation(
        scored,
        minimum_months=int(gate["minimum_months"]),
        mean_ic_minimum=float(gate["mean_rank_ic_min"]),
        p_value_maximum=float(gate["hac_p_one_sided_max"]),
        positive_years_minimum=int(gate["positive_years_min"]),
        required_positive_year=int(gate["required_positive_year"]),
        mean_net_return_minimum=float(gate["mean_net_active_return_min"]),
        net_information_ratio_minimum=float(gate["net_information_ratio_min"]),
        monthly_return_coverage_minimum=float(
            gate["minimum_monthly_return_coverage"]
        ),
        selected_return_coverage_minimum=float(
            gate["minimum_selected_return_coverage"]
        ),
        portfolio_holdings=int(portfolio["holdings"]),
        retention_percentile=float(portfolio["retention_percentile"]),
        cost_bps_one_way=float(portfolio["cost_bps_one_way"]),
    )
    summary["passes_construction_gate"] = summary.pop(
        "passes_confirmation_gate"
    )
    exposures = exposure_diagnostics(
        scored, control_columns=KNOWN_FEATURES_V30
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "models": models,
                "model_columns": model_columns,
                "prediction_years": PREDICTION_YEARS_V43,
            },
            handle,
            protocol=5,
        )
    summary.to_csv(output / "construction_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig")
    fit_audit.to_csv(output / "walk_forward_audit.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "experiment_id": protocol["experiment_id"],
        "candidate": CANDIDATE_V43,
        "role": "adaptive_E_plus_F_construction",
        "construction_data_sha256": [sha256(Path(path)) for path in args.data],
        "industry_sha256": sha256(Path(args.industry)),
        "protocol_sha256": sha256(protocol_path),
        "model_sha256": sha256(model_path),
        "model_columns": model_columns,
        "walk_forward": fit_audit.to_dict(orient="records"),
        "return_manifests": return_manifests,
        "passes_construction_gate": bool(
            summary.iloc[0]["passes_construction_gate"]
        ),
        "bucket_g_opened": False,
        "out_of_sample_claim": False,
    }
    (output / "construction.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(yearly.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
