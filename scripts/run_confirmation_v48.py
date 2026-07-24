from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from ashare_alpha.confirmation import evaluate_confirmation
from ashare_alpha.margin_data import sha256_file
from ashare_alpha.margin_factors_v48 import CANDIDATES_V48
from ashare_alpha.v48_pipeline import build_evaluation_panel_v48


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one-shot bucket-B and future-time V48 confirmation."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--margin", required=True)
    parser.add_argument("--margin-manifest", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--construction-json", required=True)
    parser.add_argument("--construction-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    freeze_path = Path(args.freeze)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["status"] != "confirmation_authorized_and_frozen":
        raise ValueError("V48 confirmation is not authorized by construction")
    candidate = freeze["candidate"]
    if candidate not in CANDIDATES_V48:
        raise ValueError("V48 confirmation candidate is outside the frozen registry")
    if freeze["protocol_sha256"] != sha256_file(protocol_path):
        raise ValueError("V48 protocol changed after construction")
    if freeze["margin_data_sha256"] != sha256_file(args.margin):
        raise ValueError("V48 margin dataset changed after construction")
    if freeze["construction_json_sha256"] != sha256_file(args.construction_json):
        raise ValueError("V48 construction metadata changed after freeze")
    if freeze["construction_summary_sha256"] != sha256_file(
        args.construction_summary
    ):
        raise ValueError("V48 construction summary changed after freeze")
    manifest = json.loads(Path(args.margin_manifest).read_text(encoding="utf-8"))
    if (
        manifest["audit"]["status"] != "admitted"
        or manifest["output_sha256"] != sha256_file(args.margin)
    ):
        raise ValueError("V48 confirmation margin data is not admitted")

    portfolio = protocol["portfolio"]
    evaluated, exposures, coverage, return_manifest = build_evaluation_panel_v48(
        structural_path=args.data,
        hfq_cache_directory=args.hfq_cache,
        industry_path=args.industry,
        margin_path=args.margin,
        start_date="2024-01-01",
        end_date="2026-05-31",
        candidates=(candidate,),
        minimum_rows=int(protocol["neutralization"]["minimum_rows"]),
        entry_offset=int(portfolio["entry_offset_sessions"]),
        exit_offset=int(portfolio["exit_offset_sessions"]),
        maximum_delay=int(portfolio["maximum_execution_delay_sessions"]),
    )
    gate = protocol["confirmation_gate"]
    summary, yearly, monthly = evaluate_confirmation(
        evaluated.copy(),
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
    required_years = set(int(value) for value in gate["required_core_years_positive"])
    yearly_indexed = yearly.set_index("year")
    core_positive = bool(
        required_years.issubset(set(yearly_indexed.index))
        and yearly_indexed.reindex(sorted(required_years))["mean_rank_ic"].gt(0.0).all()
    )
    summary["required_core_years_positive"] = core_positive
    summary["passes_confirmation_gate"] &= core_positive

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        output / "confirmation_summary.csv", index=False, encoding="utf-8-sig"
    )
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(
        output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    coverage.to_csv(
        output / "margin_coverage.csv", index=False, encoding="utf-8-sig"
    )
    result = {
        "experiment_id": freeze["experiment_id"],
        "candidate": candidate,
        "role": "one_shot_asset_and_future_time_disjoint_confirmation",
        "bucket": "stable_hash_bucket_B",
        "period": "2024-01-01_to_2026-05-31",
        "asset_disjoint_from_construction": True,
        "future_time_relative_to_construction": True,
        "formula_unseen_before_run": True,
        "refitting": "none",
        "passes_confirmation_gate": bool(
            summary.iloc[0]["passes_confirmation_gate"]
        ),
        "effective_confirmed_factor": bool(
            summary.iloc[0]["passes_confirmation_gate"]
        ),
        "global_pristine_claim": False,
        "return_manifest": return_manifest,
        "hashes": {
            "structural_data_sha256": sha256_file(args.data),
            "industry_sha256": sha256_file(args.industry),
            "margin_data_sha256": sha256_file(args.margin),
            "protocol_sha256": sha256_file(protocol_path),
            "freeze_sha256": sha256_file(freeze_path),
        },
    }
    (output / "confirmation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(yearly.to_string(index=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
