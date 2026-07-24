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
from ashare_alpha.model_v42 import (
    CANDIDATE_V42,
    FEATURES_V42,
    MODEL_PARAMETERS_V42,
    PREDICTION_YEARS_V42,
    TOP_TAIL_QUANTILE_V42,
    build_monthly_features_v42,
    finalize_features_v42,
    fit_walk_forward_models_v42,
    main_effect_controls_v42,
    rank_feature_matrix_v42,
)
from ashare_alpha.tdx_financial import FIELDS_V40
from run_confirmation_v41 import assert_hash, sha256
from run_discovery_v33 import load_validated_financial_history


def load_construction_data(
    data_paths: list[str],
    cache_paths: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(data_paths) != len(cache_paths):
        raise ValueError("Every construction dataset requires one qfq cache")
    daily_parts: list[pd.DataFrame] = []
    price_parts: list[pd.DataFrame] = []
    asset_sets: list[set[str]] = []
    for data_path, cache_path in zip(data_paths, cache_paths, strict=True):
        daily, prices = load_formation_daily(data_path, cache_path)
        daily_parts.append(daily)
        price_parts.append(prices)
        asset_sets.append(set(daily["asset"].astype(str).str.zfill(6)))
    for left in range(len(asset_sets)):
        for right in range(left + 1, len(asset_sets)):
            if asset_sets[left].intersection(asset_sets[right]):
                raise ValueError("V42 construction datasets have overlapping assets")
    daily = pd.concat(daily_parts, ignore_index=True)
    prices = pd.concat(price_parts, ignore_index=True)
    if daily.duplicated(["date", "asset"]).any():
        raise ValueError("V42 combined daily data has duplicate date/asset rows")
    if prices.duplicated(["date", "asset"]).any():
        raise ValueError("V42 combined price data has duplicate date/asset rows")
    return daily, prices


def neutralize_predictions(
    panel: pd.DataFrame,
    controls: list[str],
    minimum_rows: int,
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    for _, rows in panel.groupby("date", sort=True):
        scored = _neutralize_cross_section(
            rows, CANDIDATE_V42, controls, minimum_rows=minimum_rows
        )
        if not scored.empty:
            output.append(scored)
    if not output:
        raise ValueError("V42 produced no complete neutralized month")
    return pd.concat(output, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive E+F construction of the frozen walk-forward V42 model."
    )
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--hfq-cache", action="append", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--financial-directory", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["candidate"] != CANDIDATE_V42:
        raise ValueError("V42 candidate registry mismatch")
    if protocol["features"] != FEATURES_V42:
        raise ValueError("V42 feature registry mismatch")
    if protocol["training"]["fixed_hyperparameters"] != MODEL_PARAMETERS_V42:
        raise ValueError("V42 model parameters differ from protocol")
    if tuple(protocol["training"]["prediction_years"]) != PREDICTION_YEARS_V42:
        raise ValueError("V42 prediction years differ from protocol")
    if float(protocol["training"]["top_tail_quantile"]) != TOP_TAIL_QUANTILE_V42:
        raise ValueError("V42 target quantile differs from protocol")
    for path, expected in zip(
        args.data, protocol["freeze"]["construction_data_sha256"], strict=True
    ):
        assert_hash(Path(path), expected, "V42 construction data")
    assert_hash(
        Path(args.industry),
        protocol["freeze"]["industry_sha256"],
        "industry history",
    )

    financial, financial_audit, manifest_path = load_validated_financial_history(
        Path(args.financial_directory), field_map=FIELDS_V40
    )
    assert_hash(
        manifest_path,
        protocol["freeze"]["financial_manifest_sha256"],
        "financial manifest",
    )
    daily, prices = load_construction_data(args.data, args.hfq_cache)
    features = finalize_features_v42(
        align_industry_history(
            build_monthly_features_v42(daily, financial),
            pd.read_parquet(args.industry),
        )
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
    panel = features.merge(
        returns[outcomes], on=["date", "asset"], how="left", validate="one_to_one"
    )
    panel = panel[panel["date"].between("2020-01-01", "2025-12-31")].copy()
    matrix, model_columns = rank_feature_matrix_v42(panel)
    predictions, models, fit_audit = fit_walk_forward_models_v42(
        panel, matrix[model_columns]
    )
    panel[CANDIDATE_V42] = predictions
    main_effects, control_columns = main_effect_controls_v42(
        matrix[model_columns]
    )
    panel = pd.concat([panel, main_effects], axis=1)
    scored = neutralize_predictions(
        panel,
        control_columns,
        minimum_rows=int(protocol["neutralization"]["minimum_rows"]),
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
    exposures = exposure_diagnostics(scored, control_columns=control_columns)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "models": models,
                "model_columns": model_columns,
                "control_columns": control_columns,
                "prediction_years": PREDICTION_YEARS_V42,
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
        "candidate": CANDIDATE_V42,
        "role": "adaptive_E_plus_F_construction",
        "construction_data_sha256": [sha256(Path(path)) for path in args.data],
        "industry_sha256": sha256(Path(args.industry)),
        "financial_manifest_sha256": sha256(manifest_path),
        "protocol_sha256": sha256(protocol_path),
        "model_sha256": sha256(model_path),
        "financial_package_count": int(financial_audit["package_count"]),
        "model_columns": model_columns,
        "main_effect_control_count": len(control_columns),
        "walk_forward": fit_audit.to_dict(orient="records"),
        "return_manifest": return_manifest,
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
