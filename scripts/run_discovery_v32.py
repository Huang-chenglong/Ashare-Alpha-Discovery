from __future__ import annotations

import argparse
import hashlib
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
from ashare_alpha.factors_v32 import (
    CANDIDATES_V32,
    CANDIDATE_COLUMNS_V32,
    CONTROL_COLUMNS_BY_CANDIDATE_V32,
    FINANCIAL_MAIN_EFFECTS_V32,
    MAXIMUM_ANNOUNCEMENT_LAG_DAYS_V32,
    MAXIMUM_SIGNAL_AGE_DAYS_V32,
    build_quarterly_financial_features_v32,
    compute_candidates_v32,
)
from ashare_alpha.statistics import hac_mean_test
from ashare_alpha.tdx_financial import (
    FinancialPackage,
    load_financial_history,
    validate_financial_package,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_validated_financial_history(
    directory: Path,
) -> tuple[pd.DataFrame, dict[str, object], Path]:
    manifest_path = directory / "manifest.json"
    audit = json.loads(manifest_path.read_text(encoding="utf-8"))
    package_rows = audit.get("packages", [])
    if audit.get("package_count") != len(package_rows) or not package_rows:
        raise ValueError("TongdaXin audit manifest has an inconsistent package count")
    expected_names = {str(row["filename"]) for row in package_rows}
    actual_names = {path.name for path in directory.glob("gpcw????????.zip")}
    if actual_names != expected_names:
        raise ValueError("Financial ZIP set does not exactly match its audit manifest")
    for row in package_rows:
        package = FinancialPackage(
            filename=str(row["filename"]),
            md5=str(row["md5"]),
            filesize=int(row["filesize"]),
            report_date=pd.Timestamp(row["report_date"]),
        )
        validate_financial_package(directory / package.filename, package)
    return load_financial_history(directory), audit, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the single frozen V32 quarterly ownership interaction."
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
    if protocol["candidate_registry"] != CANDIDATE_COLUMNS_V32:
        raise ValueError("V32 protocol candidate registry mismatch")
    expected_prior = int(protocol["multiplicity"]["prior_tests_included"])
    expected_total = int(protocol["multiplicity"]["total_tests_after_v32"])
    if expected_total != expected_prior + len(CANDIDATE_COLUMNS_V32):
        raise ValueError("V32 protocol multiplicity total is inconsistent")
    filters = protocol["financial_data"]["filters"]
    if int(filters["announcement_lag_days"][1]) != MAXIMUM_ANNOUNCEMENT_LAG_DAYS_V32:
        raise ValueError("V32 announcement-lag constant differs from the protocol")
    if int(filters["maximum_signal_age_days"]) != MAXIMUM_SIGNAL_AGE_DAYS_V32:
        raise ValueError("V32 staleness constant differs from the protocol")
    if (
        protocol["neutralization"]["financial_main_effects"]
        != FINANCIAL_MAIN_EFFECTS_V32
    ):
        raise ValueError("V32 financial controls differ from the protocol")

    prior = pd.concat(
        [pd.read_csv(path) for path in args.prior_summary], ignore_index=True
    )
    if len(prior) != expected_prior or prior["candidate"].nunique() != expected_prior:
        raise ValueError(f"V32 requires exactly {expected_prior} unique prior tests")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    financial, financial_audit, financial_manifest_path = (
        load_validated_financial_history(Path(args.financial_directory))
    )
    quarterly = build_quarterly_financial_features_v32(financial)
    daily, prices = load_formation_daily(args.data, args.hfq_cache)
    features = compute_candidates_v32(daily, financial)
    financial_complete = features[
        ["diba_v32", *FINANCIAL_MAIN_EFFECTS_V32]
    ].notna().all(axis=1)
    financial_coverage = (
        features.assign(
            complete_financial=financial_complete,
            nonzero_interaction=features["diba_v32"].gt(0.0),
        )
        .groupby("date")
        .agg(
            universe=("asset", "size"),
            complete_financial=("complete_financial", "sum"),
            nonzero_interaction=("nonzero_interaction", "sum"),
        )
        .reset_index()
    )
    aligned = align_industry_history(features, pd.read_parquet(args.industry))
    candidate = CANDIDATE_COLUMNS_V32[0]
    controls = CONTROL_COLUMNS_BY_CANDIDATE_V32[candidate]
    panels = neutralized_candidate_panels(
        aligned,
        candidate_columns=[candidate],
        control_columns=controls,
        minimum_rows=int(protocol["neutralization"]["minimum_rows"]),
    )
    panels = panels[panels["date"].between("2020-01-01", "2024-12-31")].copy()
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
        returns[outcomes],
        on=["date", "asset"],
        how="left",
        validate="many_to_one",
    )
    gate = protocol["research_gate"]
    portfolio = protocol["portfolio"]
    summary, yearly, monthly = evaluate_research(
        evaluated,
        candidate_columns=[candidate],
        candidate_definitions=CANDIDATES_V32,
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
    exposures = exposure_diagnostics(panels, control_columns=controls)

    summary.to_csv(output / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(
        output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    financial_coverage.to_csv(
        output / "financial_coverage.csv", index=False, encoding="utf-8-sig"
    )
    passed = bool(summary.iloc[0]["passes_research_gate"])
    selection = {
        "experiment_id": protocol["experiment_id"],
        "prior_test_count": len(prior),
        "current_test_count": 1,
        "multiplicity_family_size": len(prior) + 1,
        "research_data_sha256": sha256(Path(args.data)),
        "industry_sha256": sha256(Path(args.industry)),
        "financial_manifest_sha256": sha256(financial_manifest_path),
        "protocol_sha256": sha256(protocol_path),
        "financial_package_count": int(financial_audit["package_count"]),
        "financial_quarterly_feature_rows": len(quarterly),
        "financial_vintage_limitation": protocol["financial_data"][
            "vintage_limitation"
        ],
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
