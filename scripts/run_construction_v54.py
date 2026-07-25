from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml

from ashare_alpha.margin_data import sha256_file
from ashare_alpha.model_v54 import (
    CANDIDATES_V54,
    apply_construction_gate_v54,
    fit_final_model_v54,
    select_candidate_v54,
    walk_forward_predictions_v54,
)
from ashare_alpha.strategy_v54 import evaluate_long_short_v54
from ashare_alpha.v54_pipeline import (
    build_model_source_panel_v54,
    neutralize_model_predictions_v54,
)


def _evaluate(evaluated: pd.DataFrame, gate: dict, portfolio: dict):
    return evaluate_long_short_v54(
        evaluated,
        minimum_months=int(gate["minimum_months"]),
        mean_ic_minimum=float(gate["mean_rank_ic_min"]),
        p_value_maximum=float(gate["hac_p_one_sided_max"]),
        positive_years_minimum=int(gate["positive_years_min"]),
        required_positive_year=int(gate["required_positive_year"]),
        mean_net_return_minimum=float(gate["mean_net_spread_return_min"]),
        net_information_ratio_minimum=float(gate["net_information_ratio_min"]),
        monthly_return_coverage_minimum=float(gate["minimum_monthly_return_coverage"]),
        selected_return_coverage_minimum=float(gate["minimum_selected_return_coverage"]),
        holdings_each_side=int(portfolio["holdings_each_side"]),
        long_retention_percentile=float(portfolio["long_retention_percentile"]),
        short_retention_percentile=float(portfolio["short_retention_percentile"]),
        cost_bps_one_way=float(portfolio["cost_bps_one_way"]),
        annual_short_borrow_bps=float(portfolio["annual_short_borrow_bps"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen V54 A construction.")
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
    if tuple(protocol["candidate_registry"]) != CANDIDATES_V54:
        raise ValueError("V54 candidate registry differs")
    if protocol["status"] != "frozen_after_v53_B_economic_failure_before_v54_outcomes":
        raise ValueError("V54 protocol status is invalid")
    hashes = protocol["data"]
    checks = {
        "A data": (args.data, hashes["training_A_sha256"]),
        "industry": (args.industry, hashes["industry_sha256"]),
        "margin": (args.dense_margin, hashes["dense_margin_sha256"]),
        "margin manifest": (
            args.margin_manifest,
            hashes["dense_margin_manifest_sha256"],
        ),
        "unseen manifest": (
            args.unseen_manifest,
            hashes["unseen_manifest_sha256"],
        ),
    }
    for label, (path, expected) in checks.items():
        if sha256_file(path) != expected:
            raise ValueError(f"V54 frozen {label} hash differs")
    margin_manifest = json.loads(
        Path(args.margin_manifest).read_text(encoding="utf-8")
    )
    if margin_manifest["audit"]["status"] != "admitted":
        raise ValueError("V54 margin data is not admitted")

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
    gate = protocol["construction_gate"]
    summaries = []
    yearlies = []
    monthlies = []
    exposures = []
    evaluated_parts = []
    for candidate in CANDIDATES_V54:
        prediction = walk_forward_predictions_v54(
            panel,
            matrix,
            candidate=candidate,
            prediction_start="2021-01-01",
            prediction_end="2023-12-31",
            minimum_training_rows=int(
                protocol["point_in_time_training"]["minimum_training_rows"]
            ),
        )
        evaluated, exposure = neutralize_model_predictions_v54(
            panel,
            prediction,
            candidate=candidate,
            minimum_rows=int(protocol["neutralization"]["minimum_rows_A"]),
        )
        summary, yearly, monthly = _evaluate(evaluated, gate, portfolio)
        summaries.append(summary)
        yearlies.append(yearly.assign(candidate=candidate))
        monthlies.append(monthly)
        exposures.append(exposure)
        evaluated_parts.append(evaluated)
    yearly = pd.concat(yearlies, ignore_index=True)
    summary = apply_construction_gate_v54(
        pd.concat(summaries, ignore_index=True),
        yearly,
        required_core_years_positive=gate["required_core_years_positive"],
        family_bh_q_max=float(gate["family_bh_q_max"]),
    )
    evaluated_all = pd.concat(evaluated_parts, ignore_index=True)
    minimum_rows = evaluated_all.groupby(["candidate", "date"])["asset"].nunique()
    row_gate = minimum_rows.groupby("candidate").min().ge(
        int(gate["minimum_monthly_complete_rows"])
    )
    summary["passes_data_coverage_gate"] = summary["candidate"].map(row_gate).fillna(False)
    summary["passes_construction_gate"] &= summary["passes_data_coverage_gate"]
    selected = select_candidate_v54(summary)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "construction_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    pd.concat(monthlies, ignore_index=True).to_csv(
        output / "monthly_results.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(exposures, ignore_index=True).to_csv(
        output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    coverage.to_csv(output / "margin_coverage.csv", index=False, encoding="utf-8-sig")
    artifact_path = output / "selected_model.pkl"
    artifact_hash = None
    metadata = None
    if selected is not None:
        bundle = fit_final_model_v54(
            panel,
            matrix,
            candidate=selected,
            label_available_before="2024-01-01",
            minimum_training_rows=int(
                protocol["point_in_time_training"]["minimum_training_rows"]
            ),
        )
        with artifact_path.open("wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        artifact_hash = sha256_file(artifact_path)
        metadata = {key: value for key, value in bundle.items() if key != "model"}
    result = {
        "experiment_id": protocol["experiment_id"],
        "selected_candidate": selected,
        "eligible_for_C_validation": selected is not None,
        "C_outcomes_read": False,
        "D_outcomes_read": False,
        "model_artifact_sha256": artifact_hash,
        "training_metadata": metadata,
        "return_manifest": return_manifest,
        "protocol_sha256": sha256_file(protocol_path),
        "construction_summary_sha256": sha256_file(summary_path),
    }
    result_path = output / "construction.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    freeze = {
        "status": (
            "C_validation_authorized_and_frozen"
            if selected is not None
            else "C_validation_not_authorized"
        ),
        "candidate": selected,
        "model_artifact_sha256": artifact_hash,
        "protocol_sha256": sha256_file(protocol_path),
        "construction_json_sha256": sha256_file(result_path),
        "construction_summary_sha256": sha256_file(summary_path),
        "C_data_sha256": hashes["validation_C_sha256"],
        "D_data_sha256": hashes["confirmation_D_sha256"],
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
