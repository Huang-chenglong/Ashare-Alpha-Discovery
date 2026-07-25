from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml

from ashare_alpha.composite_v41 import CANDIDATE_V41, combine_component_scores_v41
from ashare_alpha.confirmation import evaluate_confirmation
from ashare_alpha.data import (
    align_industry_history,
    attach_execution_returns,
    load_formation_daily,
)
from ashare_alpha.evaluate import exposure_diagnostics
from ashare_alpha.factors_v33 import attach_point_in_time_financial_v33
from ashare_alpha.factors_v40 import (
    CONTROL_COLUMNS_BY_CANDIDATE_V40,
    FINANCIAL_MAIN_EFFECTS_V40,
    build_quarterly_financial_features_v40,
    finalize_candidates_v40,
)
from ashare_alpha.model_v30 import (
    KNOWN_FEATURES_V30,
    build_monthly_features_v30,
    rank_feature_matrix_v30,
)
from ashare_alpha.tdx_financial import FIELDS_V40
from run_confirmation_v41 import _neutralize_months, assert_hash, sha256
from run_discovery_v33 import load_validated_financial_history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive post-fit V41 diagnostic on construction bucket E."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--financial-directory", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--diagnostic-protocol", required=True)
    parser.add_argument("--confirmation-protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    diagnostic_path = Path(args.diagnostic_protocol)
    diagnostic = yaml.safe_load(diagnostic_path.read_text(encoding="utf-8"))
    confirmation_path = Path(args.confirmation_protocol)
    confirmation = yaml.safe_load(confirmation_path.read_text(encoding="utf-8"))
    freeze = confirmation["freeze"]

    assert_hash(Path(args.data), diagnostic["data_sha256"], "adaptive E data")
    assert_hash(Path(args.data), freeze["research_data_sha256"], "V41 research data")
    assert_hash(Path(args.industry), freeze["industry_sha256"], "industry history")
    assert_hash(Path(args.model), freeze["model_sha256"], "V41 structural model")
    assert_hash(
        Path("src/ashare_alpha/composite_v41.py"),
        freeze["composite_source_sha256"],
        "composite source",
    )
    assert_hash(
        Path("src/ashare_alpha/model_v30.py"),
        freeze["structural_source_sha256"],
        "structural source",
    )
    assert_hash(
        Path("src/ashare_alpha/factors_v40.py"),
        freeze["financial_source_sha256"],
        "financial source",
    )

    with Path(args.model).open("rb") as handle:
        model_bundle = pickle.load(handle)
    financial, financial_audit, manifest_path = load_validated_financial_history(
        Path(args.financial_directory), field_map=FIELDS_V40
    )
    assert_hash(
        manifest_path, freeze["financial_manifest_sha256"], "financial manifest"
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    daily, prices = load_formation_daily(args.data, args.hfq_cache)
    structural = align_industry_history(
        build_monthly_features_v30(daily), pd.read_parquet(args.industry)
    )
    matrix, model_columns = rank_feature_matrix_v30(structural)
    if model_columns != model_bundle["columns"]:
        raise ValueError("V41 model columns differ from frozen bundle")
    structural["nsim_v30_raw"] = model_bundle["model"].predict(
        matrix[model_columns]
    )
    nsim = _neutralize_months(
        structural, "nsim_v30_raw", KNOWN_FEATURES_V30, minimum_rows=100
    )[["date", "asset", "score"]].rename(columns={"score": "nsim_v30_score"})

    quarterly = build_quarterly_financial_features_v40(financial)
    financial_panel = finalize_candidates_v40(
        attach_point_in_time_financial_v33(structural, quarterly)
    )
    cfma_controls = CONTROL_COLUMNS_BY_CANDIDATE_V40["cfma_v40"]
    cfma = _neutralize_months(
        financial_panel, "cfma_v40", cfma_controls, minimum_rows=100
    )[["date", "asset", "score"]].rename(columns={"score": "cfma_v40_score"})
    components = nsim.merge(
        cfma, on=["date", "asset"], how="inner", validate="one_to_one"
    )
    components[CANDIDATE_V41] = combine_component_scores_v41(components)
    combined = financial_panel.merge(
        components[["date", "asset", CANDIDATE_V41]],
        on=["date", "asset"],
        how="inner",
        validate="one_to_one",
    )
    final_controls = list(
        dict.fromkeys([*KNOWN_FEATURES_V30, *FINANCIAL_MAIN_EFFECTS_V40])
    )
    panels = _neutralize_months(
        combined, CANDIDATE_V41, final_controls, minimum_rows=100
    )
    start, end = diagnostic["evaluation_period"]
    panels = panels[panels["date"].between(pd.Timestamp(start), pd.Timestamp(end))].copy()

    returns, return_manifest = attach_execution_returns(
        panels[["date", "asset"]], prices, daily["date"]
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
    evaluated = panels.merge(
        returns[outcomes], on=["date", "asset"], how="left", validate="one_to_one"
    )
    gate, portfolio = diagnostic["gate"], diagnostic["portfolio"]
    required_years = [int(year) for year in gate["required_positive_years"]]
    summary, yearly, monthly = evaluate_confirmation(
        evaluated,
        minimum_months=int(gate["minimum_months"]),
        mean_ic_minimum=float(gate["mean_rank_ic_min"]),
        p_value_maximum=float(gate["hac_p_one_sided_max"]),
        positive_years_minimum=int(gate["positive_years_min"]),
        required_positive_year=max(required_years),
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
    yearly_ic = yearly.set_index("year")["mean_rank_ic"].to_dict()
    required_years_positive = all(
        year in yearly_ic and yearly_ic[year] > 0.0 for year in required_years
    )
    summary["required_positive_years"] = ",".join(map(str, required_years))
    summary["required_years_positive"] = required_years_positive
    summary["passes_adaptive_gate"] = (
        summary.pop("passes_confirmation_gate") & required_years_positive
    )

    exposures = exposure_diagnostics(panels, control_columns=final_controls)
    summary.to_csv(
        output / "adaptive_summary.csv", index=False, encoding="utf-8-sig"
    )
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(
        output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "experiment_id": diagnostic["experiment_id"],
        "candidate": CANDIDATE_V41,
        "role": "adaptive_post_fit_diagnostic_not_independent_confirmation",
        "data_sha256": sha256(Path(args.data)),
        "industry_sha256": sha256(Path(args.industry)),
        "financial_manifest_sha256": sha256(manifest_path),
        "model_sha256": sha256(Path(args.model)),
        "diagnostic_protocol_sha256": sha256(diagnostic_path),
        "confirmation_protocol_sha256": sha256(confirmation_path),
        "financial_package_count": int(financial_audit["package_count"]),
        "quarterly_feature_rows": len(quarterly),
        "return_manifest": return_manifest,
        "passes_adaptive_gate": bool(summary.iloc[0]["passes_adaptive_gate"]),
        "out_of_sample_claim": False,
        "training_period_excluded_from_gate": True,
    }
    (output / "adaptive_diagnostic.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(yearly.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
