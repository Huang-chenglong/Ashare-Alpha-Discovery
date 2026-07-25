from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml

from ashare_alpha.data import (
    align_industry_history,
    attach_execution_returns,
    load_formation_daily,
)
from ashare_alpha.evaluate import (
    _neutralize_cross_section,
    evaluate_research,
    exposure_diagnostics,
)
from ashare_alpha.factors import Candidate
from ashare_alpha.model_v36 import (
    CANDIDATE_V36,
    FEATURES_V36,
    FINANCIAL_FEATURES_V36,
    FOLD_COUNT_V36,
    MODEL_PARAMETERS_V36,
    build_monthly_features_v36,
    build_quarterly_financial_features_v36,
    exclude_financial_industries_v36,
    fit_cross_fitted_model_v36,
    main_effect_controls_v36,
    rank_feature_matrix_v36,
)
from ashare_alpha.statistics import hac_mean_test
from ashare_alpha.tdx_financial import FIELDS_V36
from run_discovery_v33 import load_validated_financial_history, sha256


DEFINITION_V36 = {
    CANDIDATE_V36: Candidate(
        "fundamental_structural_interaction_model",
        (
            "A fixed shallow nonlinear map of point-in-time accounting, ownership, "
            "and trading-structure states predicts residual next-month returns "
            "after linear and quadratic ranked main effects, missingness, known "
            "factors, size, and industry are removed."
        ),
    )
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen V36 cross-fitted fundamental-structural model."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--financial-directory", required=True)
    parser.add_argument("--prior-summary", action="append", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["candidate_registry"] != [CANDIDATE_V36]:
        raise ValueError("V36 protocol candidate registry mismatch")
    if protocol["training"]["fixed_hyperparameters"] != MODEL_PARAMETERS_V36:
        raise ValueError("V36 model parameters differ from protocol")
    if int(protocol["training"]["asset_crossfit_folds"]) != FOLD_COUNT_V36:
        raise ValueError("V36 fold count differs from protocol")
    if protocol["features"]["all_model_features"] != FEATURES_V36:
        raise ValueError("V36 feature registry differs from protocol")
    if protocol["features"]["financial_features"] != FINANCIAL_FEATURES_V36:
        raise ValueError("V36 financial registry differs from protocol")
    expected_prior = int(protocol["multiplicity"]["prior_tests_included"])
    if int(protocol["multiplicity"]["total_tests_after_v36"]) != expected_prior + 1:
        raise ValueError("V36 multiplicity total is inconsistent")
    prior = pd.concat(
        [pd.read_csv(path) for path in args.prior_summary], ignore_index=True
    )
    if len(prior) != expected_prior or prior["candidate"].nunique() != expected_prior:
        raise ValueError(f"V36 requires exactly {expected_prior} unique prior tests")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    financial, financial_audit, manifest_path = load_validated_financial_history(
        Path(args.financial_directory), field_map=FIELDS_V36
    )
    quarterly = build_quarterly_financial_features_v36(financial)
    daily, prices = load_formation_daily(args.data, args.hfq_cache)
    features = build_monthly_features_v36(daily, financial)
    features = exclude_financial_industries_v36(
        align_industry_history(features, pd.read_parquet(args.industry))
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
    panel = panel[panel["date"].between("2020-01-01", "2024-12-31")].copy()

    matrix, model_columns = rank_feature_matrix_v36(panel)
    predictions, fold_models, final_model, fold_audit = fit_cross_fitted_model_v36(
        panel, matrix[model_columns]
    )
    panel[CANDIDATE_V36] = predictions
    main_effects, main_effect_columns = main_effect_controls_v36(
        matrix[model_columns]
    )
    panel = pd.concat([panel, main_effects], axis=1)

    scored_months: list[pd.DataFrame] = []
    minimum_rows = int(protocol["evaluation_neutralization"]["minimum_rows"])
    for _, rows in panel.groupby("date", sort=True):
        scored = _neutralize_cross_section(
            rows,
            CANDIDATE_V36,
            main_effect_columns,
            minimum_rows=minimum_rows,
        )
        if not scored.empty:
            scored_months.append(scored)
    if not scored_months:
        raise ValueError("V36 produced no neutralized month")
    scored_panel = pd.concat(scored_months, ignore_index=True)

    gate = protocol["research_gate"]
    portfolio = protocol["portfolio"]
    summary, yearly, monthly = evaluate_research(
        scored_panel,
        candidate_columns=[CANDIDATE_V36],
        candidate_definitions=DEFINITION_V36,
        prior_discovery_p_values=prior["discovery_p_one_sided"],
        portfolio_holdings=int(portfolio["holdings"]),
        retention_percentile=float(portfolio["retention_percentile"]),
        cost_bps_one_way=float(portfolio["cost_bps_one_way"]),
        discovery_minimum_months=int(gate["discovery_minimum_months"]),
        discovery_mean_ic_minimum=float(gate["discovery_mean_rank_ic_min"]),
        discovery_q_maximum=float(gate["discovery_bh_fdr_max"]),
        validation_minimum_months=int(gate["validation_minimum_months"]),
        validation_mean_ic_minimum=float(gate["validation_mean_rank_ic_min"]),
        validation_mean_net_return_minimum=float(
            gate["validation_mean_net_active_return_min"]
        ),
        validation_net_information_ratio_minimum=float(
            gate["validation_net_information_ratio_min"]
        ),
    )
    validation_ic = monthly.loc[
        monthly["date"].between("2023-01-01", "2024-12-31"), "rank_ic"
    ]
    validation_t, validation_p = hac_mean_test(validation_ic)
    summary["validation_hac_t"] = validation_t
    summary["validation_hac_p_one_sided"] = validation_p
    summary["passes_research_gate"] &= summary[
        "validation_hac_p_one_sided"
    ].le(float(gate["validation_hac_p_one_sided_max"]))
    exposures = exposure_diagnostics(
        scored_panel, control_columns=main_effect_columns
    )
    coverage = (
        panel.assign(
            complete_financial=panel[FINANCIAL_FEATURES_V36].notna().all(axis=1),
            scored=CANDIDATE_V36 in panel and panel[CANDIDATE_V36].notna(),
        )
        .groupby("date")
        .agg(
            universe=("asset", "size"),
            complete_financial=("complete_financial", "sum"),
            scored=("scored", "sum"),
        )
        .reset_index()
    )

    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "fold_models": fold_models,
                "final_model": final_model,
                "model_columns": model_columns,
                "main_effect_columns": main_effect_columns,
            },
            handle,
            protocol=5,
        )
    summary.to_csv(output / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output / "feature_coverage.csv", index=False, encoding="utf-8-sig")
    fold_audit.to_csv(output / "crossfit_audit.csv", index=False, encoding="utf-8-sig")
    passed = bool(summary.iloc[0]["passes_research_gate"])
    selection = {
        "experiment_id": protocol["experiment_id"],
        "prior_test_count": len(prior),
        "current_test_count": 1,
        "multiplicity_family_size": len(prior) + 1,
        "research_data_sha256": sha256(Path(args.data)),
        "industry_sha256": sha256(Path(args.industry)),
        "financial_manifest_sha256": sha256(manifest_path),
        "protocol_sha256": sha256(protocol_path),
        "model_sha256": sha256(model_path),
        "financial_package_count": int(financial_audit["package_count"]),
        "quarterly_joint_feature_rows": len(quarterly),
        "model_columns": model_columns,
        "main_effect_control_count": len(main_effect_columns),
        "crossfit": fold_audit.to_dict(orient="records"),
        "adaptive_reuse_disclosure": protocol["adaptive_reuse_disclosure"],
        "return_manifest": return_manifest,
        "passed_candidates": [CANDIDATE_V36] if passed else [],
        "selected_candidate": CANDIDATE_V36 if passed else None,
        "confirmation_bucket_opened": False,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
