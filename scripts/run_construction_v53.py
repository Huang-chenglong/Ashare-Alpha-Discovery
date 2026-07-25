from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml

from ashare_alpha.confirmation import evaluate_confirmation
from ashare_alpha.margin_data import sha256_file
from ashare_alpha.model_v53 import (
    CANDIDATES_V53,
    apply_construction_gate_v53,
    fit_final_model_v53,
    select_candidate_v53,
    walk_forward_predictions_v53,
)
from ashare_alpha.v53_pipeline import (
    build_model_source_panel_v53,
    neutralize_model_predictions_v53,
)


def _require_hash(path: str | Path, expected: str, label: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"V53 {label} differs from the frozen protocol")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen V53 direct-margin tail construction."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--dense-margin", required=True)
    parser.add_argument("--margin-manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if tuple(protocol["candidate_registry"]) != CANDIDATES_V53:
        raise ValueError("V53 registry differs from the frozen implementation")
    if protocol["status"] != "frozen_after_v52_A_failure_before_v53_A_outcomes":
        raise ValueError("V53 protocol status is invalid")
    frozen_hashes = protocol["data"]
    _require_hash(
        args.data,
        frozen_hashes["construction_structural_sha256"],
        "construction data",
    )
    _require_hash(args.industry, frozen_hashes["industry_sha256"], "industry data")
    _require_hash(
        args.dense_margin,
        frozen_hashes["dense_margin_data_sha256"],
        "dense margin data",
    )
    _require_hash(
        args.margin_manifest,
        frozen_hashes["dense_margin_manifest_sha256"],
        "dense margin manifest",
    )
    manifest_path = Path(args.margin_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["audit"]["status"] != "admitted"
        or manifest["output_sha256"] != sha256_file(args.dense_margin)
    ):
        raise ValueError("V53 dense margin data is not admitted")

    portfolio = protocol["portfolio"]
    panel, matrix, coverage, return_manifest = build_model_source_panel_v53(
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
    summaries: list[pd.DataFrame] = []
    yearlies: list[pd.DataFrame] = []
    monthlies: list[pd.DataFrame] = []
    exposure_parts: list[pd.DataFrame] = []
    evaluated_parts: list[pd.DataFrame] = []
    for candidate in CANDIDATES_V53:
        predictions = walk_forward_predictions_v53(
            panel,
            matrix,
            candidate=candidate,
            prediction_start="2021-01-01",
            prediction_end="2023-12-31",
            minimum_training_rows=int(
                protocol["point_in_time_training"]["minimum_training_rows"]
            ),
        )
        evaluated, exposures = neutralize_model_predictions_v53(
            panel,
            predictions,
            candidate=candidate,
            minimum_rows=int(
                protocol["final_score_neutralization"]["minimum_rows"]
            ),
        )
        summary, yearly, monthly = evaluate_confirmation(
            evaluated,
            minimum_months=int(gate["minimum_months"]),
            mean_ic_minimum=float(gate["mean_rank_ic_min"]),
            p_value_maximum=float(gate["hac_p_one_sided_max"]),
            positive_years_minimum=int(gate["positive_years_min"]),
            required_positive_year=int(gate["required_positive_year"]),
            mean_net_return_minimum=float(gate["mean_net_active_return_min"]),
            net_information_ratio_minimum=float(
                gate["net_information_ratio_min"]
            ),
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
        summaries.append(summary)
        yearlies.append(yearly.assign(candidate=candidate))
        monthlies.append(monthly)
        exposure_parts.append(exposures)
        evaluated_parts.append(evaluated)

    yearly = pd.concat(yearlies, ignore_index=True)
    summary = apply_construction_gate_v53(
        pd.concat(summaries, ignore_index=True),
        yearly,
        required_core_years_positive=gate["required_core_years_positive"],
        family_bh_q_max=float(gate["family_bh_q_max"]),
    )
    evaluated_all = pd.concat(evaluated_parts, ignore_index=True)
    complete_rows = evaluated_all.groupby(["candidate", "date"])["asset"].nunique()
    complete_months = complete_rows.groupby("candidate").size()
    row_gate = (
        complete_months.ge(int(gate["minimum_months"]))
        & complete_rows.groupby("candidate")
        .min()
        .ge(int(gate["minimum_monthly_complete_rows"]))
    )
    modality_coverage_gate = bool(
        coverage["new_modality_coverage"].median()
        >= float(gate["minimum_median_new_modality_coverage"])
    )
    summary["passes_data_coverage_gate"] = (
        summary["candidate"].map(row_gate).fillna(False)
        & modality_coverage_gate
    )
    summary["passes_construction_gate"] &= summary[
        "passes_data_coverage_gate"
    ]
    selected_candidate = select_candidate_v53(summary)
    monthly = pd.concat(monthlies, ignore_index=True)
    exposures = pd.concat(exposure_parts, ignore_index=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "construction_summary.csv"
    yearly_path = output / "yearly_results.csv"
    monthly_path = output / "monthly_results.csv"
    exposure_path = output / "exposure_diagnostics.csv"
    coverage_path = output / "margin_coverage.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    exposures.to_csv(exposure_path, index=False, encoding="utf-8-sig")
    coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")

    artifact_path = output / "selected_model.pkl"
    artifact_hash: str | None = None
    training_metadata: dict[str, object] | None = None
    if selected_candidate is not None:
        bundle = fit_final_model_v53(
            panel,
            matrix,
            candidate=selected_candidate,
            label_available_before="2024-01-01",
            minimum_training_rows=int(
                protocol["point_in_time_training"]["minimum_training_rows"]
            ),
        )
        with artifact_path.open("wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        artifact_hash = sha256_file(artifact_path)
        training_metadata = {
            key: value for key, value in bundle.items() if key != "model"
        }

    construction = {
        "experiment_id": protocol["experiment_id"],
        "role": "adaptive_A_walk_forward_tail_construction",
        "candidates": list(CANDIDATES_V53),
        "selected_candidate": selected_candidate,
        "eligible_to_freeze_confirmation": selected_candidate is not None,
        "confirmation_outcomes_read": False,
        "model_artifact_sha256": artifact_hash,
        "final_training_metadata": training_metadata,
        "median_new_modality_coverage": float(
            coverage["new_modality_coverage"].median()
        ),
        "maximum_absolute_neutralized_exposure": float(
            exposures.drop(
                columns=["candidate", "date", "assets"], errors="ignore"
            )
            .abs()
            .max()
            .max()
        ),
        "return_manifest": return_manifest,
        "hashes": {
            "structural_data_sha256": sha256_file(args.data),
            "industry_sha256": sha256_file(args.industry),
            "dense_margin_data_sha256": sha256_file(args.dense_margin),
            "margin_manifest_sha256": sha256_file(manifest_path),
            "protocol_sha256": sha256_file(protocol_path),
            "construction_summary_sha256": sha256_file(summary_path),
        },
        "margin_admission": manifest["audit"],
    }
    construction_path = output / "construction.json"
    construction_path.write_text(
        json.dumps(construction, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    freeze = {
        "experiment_id": "direct-margin-flow-tail-v53-confirmation",
        "status": (
            "confirmation_authorized_and_frozen"
            if selected_candidate is not None
            else "confirmation_not_authorized"
        ),
        "candidate": selected_candidate,
        "model_artifact_sha256": artifact_hash,
        "alternative_models_or_hyperparameters_permitted": False,
        "direction_flipping_permitted": False,
        "refitting_permitted": False,
        "construction_json_sha256": sha256_file(construction_path),
        "construction_summary_sha256": sha256_file(summary_path),
        "protocol_sha256": sha256_file(protocol_path),
        "dense_margin_data_sha256": sha256_file(args.dense_margin),
        "dense_margin_manifest_sha256": sha256_file(manifest_path),
        "confirmation_structural_sha256": frozen_hashes[
            "confirmation_structural_sha256"
        ],
        "confirmation_outcomes_read": False,
    }
    (output / "confirmation_freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(yearly.to_string(index=False))
    print(json.dumps(construction, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
