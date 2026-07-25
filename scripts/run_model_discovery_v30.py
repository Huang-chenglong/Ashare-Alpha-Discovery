from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml

from ashare_alpha.data import align_industry_history, attach_execution_returns, load_formation_daily
from ashare_alpha.evaluate import (
    _neutralize_cross_section,
    evaluate_research,
    exposure_diagnostics,
)
from ashare_alpha.factors import Candidate
from ashare_alpha.model_v30 import (
    KNOWN_FEATURES_V30,
    MODEL_PARAMETERS_V30,
    build_monthly_features_v30,
    fit_model_v30,
    rank_feature_matrix_v30,
    residualize_training_target_v30,
)
from ashare_alpha.statistics import hac_mean_test


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and validate the frozen V30 model factor.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--prior-summary", action="append", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["candidate_registry"] != ["nsim_v30"]:
        raise ValueError("V30 protocol registry mismatch")
    if protocol["training"]["fixed_hyperparameters"] != MODEL_PARAMETERS_V30:
        raise ValueError("V30 model parameters do not match the frozen protocol")
    prior = pd.concat([pd.read_csv(path) for path in args.prior_summary], ignore_index=True)
    expected_prior = int(protocol["multiplicity"]["prior_tests_included"])
    if len(prior) != expected_prior or prior["candidate"].nunique() != expected_prior:
        raise ValueError(f"V30 requires exactly {expected_prior} unique prior tests")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    daily, prices = load_formation_daily(args.data, args.hfq_cache)
    features = build_monthly_features_v30(daily)
    features = align_industry_history(features, pd.read_parquet(args.industry))
    returns, return_manifest = attach_execution_returns(
        features[["date", "asset"]], prices, daily["date"]
    )
    outcomes = [
        "date", "asset", "entry_date", "exit_date", "entry_delay", "exit_delay",
        "return_status", "future_return_20", "label",
    ]
    panel = features.merge(returns[outcomes], on=["date", "asset"], how="left", validate="one_to_one")
    panel = panel[panel["date"].between("2020-01-01", "2024-12-31")].copy()
    matrix, model_columns = rank_feature_matrix_v30(panel)
    target = residualize_training_target_v30(panel)
    model = fit_model_v30(matrix[model_columns], target, panel["date"])
    panel["nsim_v30"] = model.predict(matrix[model_columns])

    neutralized = []
    for _, rows in panel.groupby("date", sort=True):
        result = _neutralize_cross_section(
            rows,
            "nsim_v30",
            KNOWN_FEATURES_V30,
            minimum_rows=int(protocol["evaluation_neutralization"]["minimum_rows"]),
        )
        if not result.empty:
            neutralized.append(result)
    if not neutralized:
        raise ValueError("V30 produced no neutralized month")
    scored = pd.concat(neutralized, ignore_index=True)
    definition = {
        "nsim_v30": Candidate(
            "nonlinear_structural_interaction_model",
            "Fixed shallow boosted interactions predict residual next-month returns beyond all registered known main effects.",
        )
    }
    gate = protocol["research_gate"]
    portfolio = protocol["portfolio"]
    summary, yearly, monthly = evaluate_research(
        scored,
        candidate_columns=["nsim_v30"],
        candidate_definitions=definition,
        prior_discovery_p_values=prior["discovery_p_one_sided"],
        portfolio_holdings=int(portfolio["holdings"]),
        retention_percentile=float(portfolio["retention_percentile"]),
        cost_bps_one_way=float(portfolio["cost_bps_one_way"]),
        discovery_minimum_months=int(gate["discovery_minimum_months"]),
        discovery_mean_ic_minimum=float(gate["discovery_mean_rank_ic_min"]),
        discovery_q_maximum=float(gate["discovery_bh_fdr_max"]),
        validation_minimum_months=int(gate["validation_minimum_months"]),
        validation_mean_ic_minimum=float(gate["validation_mean_rank_ic_min"]),
        validation_mean_net_return_minimum=float(gate["validation_mean_net_active_return_min"]),
        validation_net_information_ratio_minimum=float(gate["validation_net_information_ratio_min"]),
    )
    validation_ic = monthly.loc[
        monthly["date"].between("2023-01-01", "2024-12-31"), "rank_ic"
    ]
    validation_t, validation_p = hac_mean_test(validation_ic)
    summary["validation_hac_t"] = validation_t
    summary["validation_hac_p_one_sided"] = validation_p
    summary["passes_research_gate"] = summary["passes_research_gate"] & summary[
        "validation_hac_p_one_sided"
    ].le(float(gate["validation_hac_p_one_sided_max"]))
    exposures = exposure_diagnostics(scored, control_columns=KNOWN_FEATURES_V30)

    with model_path.open("wb") as handle:
        pickle.dump({"model": model, "columns": model_columns}, handle, protocol=5)
    summary.to_csv(output / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig")
    selection = {
        "experiment_id": protocol["experiment_id"],
        "prior_test_count": len(prior),
        "current_test_count": 1,
        "multiplicity_family_size": len(prior) + 1,
        "research_data_sha256": sha256(Path(args.data)),
        "industry_sha256": sha256(Path(args.industry)),
        "protocol_sha256": sha256(protocol_path),
        "model_sha256": sha256(model_path),
        "model_columns": model_columns,
        "return_manifest": return_manifest,
        "passed_candidates": summary.loc[summary["passes_research_gate"], "candidate"].tolist(),
        "selected_candidate": "nsim_v30" if bool(summary.iloc[0]["passes_research_gate"]) else None,
        "confirmation_bucket_opened": False,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

