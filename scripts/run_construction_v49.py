from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from ashare_alpha.confirmation import evaluate_confirmation
from ashare_alpha.margin_data import sha256_file
from ashare_alpha.margin_factors_v49 import (
    CANDIDATES_V49,
    apply_construction_gate_v49,
    select_candidate_v49,
)
from ashare_alpha.v49_pipeline import build_evaluation_panel_v49


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen adaptive V49 bucket-A construction."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--margin", required=True)
    parser.add_argument("--margin-manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if tuple(protocol["candidate_registry"]) != CANDIDATES_V49:
        raise ValueError("V49 registry differs from the frozen implementation")
    if protocol["status"] != "frozen_after_v48_A_failure_before_v49_A_outcomes":
        raise ValueError("V49 protocol status is invalid")
    margin_path = Path(args.margin)
    manifest_path = Path(args.margin_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_data = protocol["data"]
    checks = {
        "margin data": (sha256_file(margin_path), frozen_data["margin_data_sha256"]),
        "margin manifest": (
            sha256_file(manifest_path),
            frozen_data["margin_manifest_sha256"],
        ),
        "construction structural": (
            sha256_file(args.data),
            frozen_data["construction_structural_sha256"],
        ),
        "industry": (sha256_file(args.industry), frozen_data["industry_sha256"]),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise ValueError(f"V49 {label} hash differs from the frozen protocol")
    if (
        manifest["audit"]["status"] != "admitted"
        or manifest["output_sha256"] != sha256_file(margin_path)
    ):
        raise ValueError("V49 margin data is not admitted")

    portfolio = protocol["portfolio"]
    evaluated, exposures, coverage, return_manifest = build_evaluation_panel_v49(
        structural_path=args.data,
        hfq_cache_directory=args.hfq_cache,
        industry_path=args.industry,
        margin_path=args.margin,
        start_date="2020-01-01",
        end_date="2023-12-31",
        candidates=CANDIDATES_V49,
        minimum_rows=int(protocol["neutralization"]["minimum_rows"]),
        entry_offset=int(portfolio["entry_offset_sessions"]),
        exit_offset=int(portfolio["exit_offset_sessions"]),
        maximum_delay=int(portfolio["maximum_execution_delay_sessions"]),
    )
    gate = protocol["construction_gate"]
    coverage_gate = bool(
        coverage["margin_coverage"].median()
        >= float(gate["minimum_median_margin_coverage"])
        and evaluated.groupby(["candidate", "date"])["asset"].nunique().min()
        >= int(gate["minimum_monthly_complete_rows"])
    )
    summaries: list[pd.DataFrame] = []
    yearlies: list[pd.DataFrame] = []
    monthlies: list[pd.DataFrame] = []
    for candidate in CANDIDATES_V49:
        summary, yearly, monthly = evaluate_confirmation(
            evaluated[evaluated["candidate"].eq(candidate)].copy(),
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
        summaries.append(summary)
        yearlies.append(yearly.assign(candidate=candidate))
        monthlies.append(monthly)
    yearly = pd.concat(yearlies, ignore_index=True)
    summary = apply_construction_gate_v49(
        pd.concat(summaries, ignore_index=True),
        yearly,
        required_core_years_positive=gate["required_core_years_positive"],
        family_bh_q_max=float(gate["family_bh_q_max"]),
    )
    summary["passes_data_coverage_gate"] = coverage_gate
    summary["passes_construction_gate"] &= coverage_gate
    selected_candidate = select_candidate_v49(summary)
    monthly = pd.concat(monthlies, ignore_index=True)

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

    construction = {
        "experiment_id": protocol["experiment_id"],
        "role": "adaptive_bucket_A_2020_2023_construction",
        "candidates": list(CANDIDATES_V49),
        "selected_candidate": selected_candidate,
        "eligible_to_freeze_confirmation": selected_candidate is not None,
        "confirmation_outcomes_read": False,
        "data_coverage_gate": coverage_gate,
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
            "margin_data_sha256": sha256_file(margin_path),
            "margin_manifest_sha256": sha256_file(manifest_path),
            "protocol_sha256": sha256_file(protocol_path),
            "construction_summary_sha256": sha256_file(summary_path),
        },
    }
    construction_path = output / "construction.json"
    construction_path.write_text(
        json.dumps(construction, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    freeze = {
        "experiment_id": "persistent-margin-contrarian-v49-confirmation",
        "status": (
            "confirmation_authorized_and_frozen"
            if selected_candidate is not None
            else "confirmation_not_authorized"
        ),
        "candidate": selected_candidate,
        "alternative_formulas_permitted": False,
        "direction_flipping_permitted": False,
        "refitting_permitted": False,
        "construction_json_sha256": sha256_file(construction_path),
        "construction_summary_sha256": sha256_file(summary_path),
        "protocol_sha256": sha256_file(protocol_path),
        "margin_data_sha256": sha256_file(margin_path),
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
