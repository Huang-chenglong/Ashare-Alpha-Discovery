from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from ashare_alpha.data import (
    align_industry_history,
    attach_execution_returns,
    load_formation_daily,
)
from ashare_alpha.evaluate import (
    evaluate_research,
    exposure_diagnostics,
    neutralized_candidate_panels,
)
from ashare_alpha.factors_v35 import normalize_protocol_periods_v35
from ashare_alpha.factors_v37 import (
    ANNOUNCEMENT_WINDOW_SESSIONS_V37,
    CANDIDATES_V37,
    CANDIDATE_COLUMNS_V37,
    CONTROL_COLUMNS_BY_CANDIDATE_V37,
    FINANCIAL_MAIN_EFFECTS_V37,
    MAXIMUM_SIGNAL_AGE_DAYS_V37,
    build_monthly_financial_panel_v37,
    build_quarterly_financial_features_v37,
    finalize_candidates_v37,
)
from ashare_alpha.statistics import hac_mean_test
from ashare_alpha.tdx_financial import FIELDS_V37
from run_discovery_v33 import load_validated_financial_history, sha256


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen V37 cash-lead underattention factor."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--financial-directory", required=True)
    parser.add_argument("--prior-summary", action="append", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["candidate_registry"] != CANDIDATE_COLUMNS_V37:
        raise ValueError("V37 protocol candidate registry mismatch")
    expected_prior = int(protocol["multiplicity"]["prior_tests_included"])
    if int(protocol["multiplicity"]["total_tests_after_v37"]) != expected_prior + 1:
        raise ValueError("V37 multiplicity total is inconsistent")
    if (
        int(protocol["event_window"]["sessions"])
        != ANNOUNCEMENT_WINDOW_SESSIONS_V37
    ):
        raise ValueError("V37 event window differs from protocol")
    if (
        int(protocol["financial_data"]["maximum_signal_age_days"])
        != MAXIMUM_SIGNAL_AGE_DAYS_V37
    ):
        raise ValueError("V37 signal age differs from protocol")
    if (
        protocol["neutralization"]["financial_main_effects"]
        != FINANCIAL_MAIN_EFFECTS_V37
    ):
        raise ValueError("V37 financial controls differ from protocol")
    prior = pd.concat(
        [pd.read_csv(path) for path in args.prior_summary], ignore_index=True
    )
    if len(prior) != expected_prior or prior["candidate"].nunique() != expected_prior:
        raise ValueError(f"V37 requires exactly {expected_prior} unique prior tests")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    financial, financial_audit, manifest_path = load_validated_financial_history(
        Path(args.financial_directory), field_map=FIELDS_V37
    )
    daily, prices = load_formation_daily(args.data, args.hfq_cache)
    quarterly = build_quarterly_financial_features_v37(financial, daily)
    raw = build_monthly_financial_panel_v37(daily, financial)
    features = finalize_candidates_v37(
        align_industry_history(raw, pd.read_parquet(args.industry))
    )
    complete = features[
        ["clua_v37", *FINANCIAL_MAIN_EFFECTS_V37]
    ].notna().all(axis=1)
    coverage = (
        features.assign(
            complete_financial=complete,
            nonzero_interaction=features["clua_v37"].gt(0.0),
        )
        .groupby("date")
        .agg(
            universe=("asset", "size"),
            complete_financial=("complete_financial", "sum"),
            nonzero_interaction=("nonzero_interaction", "sum"),
        )
        .reset_index()
    )
    candidate = CANDIDATE_COLUMNS_V37[0]
    controls = CONTROL_COLUMNS_BY_CANDIDATE_V37[candidate]
    panels = neutralized_candidate_panels(
        features,
        candidate_columns=[candidate],
        control_columns=controls,
        minimum_rows=int(protocol["neutralization"]["minimum_rows"]),
    )
    periods = normalize_protocol_periods_v35(protocol["periods"])
    panels = panels[
        panels["date"].between(
            periods["discovery"][0], periods["internal_validation"][1]
        )
    ].copy()
    returns, return_manifest = attach_execution_returns(
        panels[["date", "asset"]].drop_duplicates(), prices, daily["date"]
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
        returns[outcomes], on=["date", "asset"], how="left", validate="many_to_one"
    )
    gate = protocol["research_gate"]
    portfolio = protocol["portfolio"]
    summary, yearly, monthly = evaluate_research(
        evaluated,
        candidate_columns=[candidate],
        candidate_definitions=CANDIDATES_V37,
        prior_discovery_p_values=prior["discovery_p_one_sided"],
        discovery_start=periods["discovery"][0],
        discovery_end=periods["discovery"][1],
        validation_start=periods["internal_validation"][0],
        validation_end=periods["internal_validation"][1],
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
        monthly["date"].between(
            periods["internal_validation"][0],
            periods["internal_validation"][1],
        ),
        "rank_ic",
    ]
    validation_t, validation_p = hac_mean_test(validation_ic)
    summary["validation_hac_t"] = validation_t
    summary["validation_hac_p_one_sided"] = validation_p
    summary["passes_research_gate"] &= summary[
        "validation_hac_p_one_sided"
    ].le(float(gate["validation_hac_p_one_sided_max"]))
    exposures = exposure_diagnostics(panels, control_columns=controls)

    summary.to_csv(output / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output / "financial_coverage.csv", index=False, encoding="utf-8-sig")
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
        "financial_package_count": int(financial_audit["package_count"]),
        "completed_announcement_event_rows": len(quarterly),
        "adaptive_reuse_disclosure": protocol["adaptive_reuse_disclosure"],
        "return_manifest": return_manifest,
        "passed_candidates": [candidate] if passed else [],
        "selected_candidate": candidate if passed else None,
        "confirmation_bucket_opened": False,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
