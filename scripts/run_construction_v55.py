from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import yaml

from ashare_alpha.confirmation import evaluate_confirmation
from ashare_alpha.margin_data import sha256_file
from ashare_alpha.model_v55 import (
    CANDIDATE_V55,
    fit_final_model_v55,
    walk_forward_predictions_v55,
)
from ashare_alpha.v54_pipeline import (
    build_model_source_panel_v54,
    neutralize_model_predictions_v54,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen V55 A construction.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--dense-margin", required=True)
    parser.add_argument("--margin-manifest", required=True)
    parser.add_argument("--unseen-manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["candidate"] != CANDIDATE_V55:
        raise ValueError("V55 candidate differs from implementation")
    if protocol["status"] != "frozen_after_v54_A_failure_before_v55_outcomes":
        raise ValueError("V55 protocol status is invalid")
    frozen = protocol["data"]
    checks = {
        args.data: frozen["training_A_sha256"],
        args.industry: frozen["industry_sha256"],
        args.dense_margin: frozen["dense_margin_sha256"],
        args.margin_manifest: frozen["dense_margin_manifest_sha256"],
        args.unseen_manifest: frozen["unseen_manifest_sha256"],
    }
    for path, expected in checks.items():
        if sha256_file(path) != expected:
            raise ValueError(f"V55 frozen input changed: {path}")
    margin_manifest = json.loads(
        Path(args.margin_manifest).read_text(encoding="utf-8")
    )
    if margin_manifest["audit"]["status"] != "admitted":
        raise ValueError("V55 margin data is not admitted")

    portfolio = protocol["portfolio"]
    panel, matrix, coverage, return_manifest = build_model_source_panel_v54(
        structural_path=args.data,
        hfq_cache_directory=args.hfq_cache,
        industry_path=args.industry,
        dense_margin_path=args.dense_margin,
        start_date="2020-01-01",
        end_date="2023-12-31",
        entry_offset=int(portfolio["entry_offset_sessions"]),
        exit_offset=int(portfolio["exit_offset_sessions"]),
        maximum_delay=int(portfolio["maximum_execution_delay_sessions"]),
    )
    prediction = walk_forward_predictions_v55(
        panel,
        matrix,
        prediction_start="2021-01-01",
        prediction_end="2023-12-31",
        minimum_training_rows=int(protocol["minimum_training_rows"]),
    )
    evaluated, exposures = neutralize_model_predictions_v54(
        panel,
        prediction,
        candidate=CANDIDATE_V55,
        minimum_rows=int(protocol["minimum_rows_A"]),
    )
    gate = protocol["construction_gate"]
    summary, yearly, monthly = evaluate_confirmation(
        evaluated,
        minimum_months=int(gate["minimum_months"]),
        mean_ic_minimum=float(gate["mean_rank_ic_min"]),
        p_value_maximum=float(gate["hac_p_one_sided_max"]),
        positive_years_minimum=int(gate["positive_years_min"]),
        required_positive_year=int(gate["required_positive_year"]),
        mean_net_return_minimum=float(gate["mean_net_active_return_min"]),
        net_information_ratio_minimum=float(gate["net_information_ratio_min"]),
        monthly_return_coverage_minimum=float(gate["minimum_monthly_return_coverage"]),
        selected_return_coverage_minimum=float(gate["minimum_selected_return_coverage"]),
        portfolio_holdings=int(portfolio["holdings"]),
        retention_percentile=float(portfolio["retention_percentile"]),
        cost_bps_one_way=float(portfolio["cost_bps_one_way"]),
    )
    required = set(map(int, gate["required_core_years_positive"]))
    indexed = yearly.set_index("year")
    core = bool(
        required.issubset(set(indexed.index))
        and indexed.reindex(sorted(required))["mean_rank_ic"].gt(0).all()
    )
    summary["required_core_years_positive"] = core
    summary["passes_construction_gate"] = (
        summary.pop("passes_confirmation_gate") & core
    )
    selected = CANDIDATE_V55 if bool(summary.iloc[0]["passes_construction_gate"]) else None
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "construction_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output / "margin_coverage.csv", index=False, encoding="utf-8-sig")
    artifact_path = output / "selected_model.pkl"
    artifact_hash = None
    metadata = None
    if selected:
        bundle = fit_final_model_v55(
            panel,
            matrix,
            label_available_before="2024-01-01",
            minimum_training_rows=int(protocol["minimum_training_rows"]),
        )
        with artifact_path.open("wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        artifact_hash = sha256_file(artifact_path)
        metadata = {key: value for key, value in bundle.items() if key != "model"}
    result = {
        "selected_candidate": selected,
        "eligible_for_C_validation": selected is not None,
        "C_outcomes_read": False,
        "D_outcomes_read": False,
        "model_artifact_sha256": artifact_hash,
        "training_metadata": metadata,
        "return_manifest": return_manifest,
        "protocol_sha256": sha256_file(protocol_path),
        "summary_sha256": sha256_file(summary_path),
    }
    result_path = output / "construction.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    freeze = {
        "status": "C_authorized" if selected else "C_not_authorized",
        "candidate": selected,
        "model_artifact_sha256": artifact_hash,
        "protocol_sha256": sha256_file(protocol_path),
        "construction_json_sha256": sha256_file(result_path),
        "construction_summary_sha256": sha256_file(summary_path),
        "C_data_sha256": frozen["validation_C_sha256"],
        "D_data_sha256": frozen["confirmation_D_sha256"],
        "C_outcomes_read": False,
        "D_outcomes_read": False,
        "refitting_permitted": False,
    }
    (output / "C_freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(yearly.to_string(index=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
