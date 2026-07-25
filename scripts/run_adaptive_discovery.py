from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import pandas as pd
import yaml

from ashare_alpha.data import align_industry_history, attach_execution_returns, load_formation_daily
from ashare_alpha.evaluate import (
    evaluate_research,
    exposure_diagnostics,
    month_end_formation,
    neutralized_candidate_panels,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen adaptive discovery version on C.")
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--prior-summary", action="append", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vwap-manifest")
    args = parser.parse_args()

    suffix = f"V{args.version}"
    module = importlib.import_module(f"ashare_alpha.factors_v{args.version}")
    candidates = getattr(module, f"CANDIDATES_{suffix}")
    candidate_columns = getattr(module, f"CANDIDATE_COLUMNS_{suffix}")
    controls = getattr(module, f"CONTROL_COLUMNS_BY_CANDIDATE_{suffix}")
    compute = getattr(module, f"compute_candidates_v{args.version}")
    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    minimum_rows = int(protocol.get("neutralization", {}).get("minimum_rows", 60))
    portfolio = protocol.get("portfolio", {})
    gate = protocol.get("research_gate", {})
    holdings = int(portfolio.get("holdings", 20))
    retention_percentile = float(portfolio.get("retention_percentile", 0.60))
    cost_bps_one_way = float(portfolio.get("cost_bps_one_way", 20.0))
    if list(protocol["candidate_registry"]) != list(candidate_columns):
        raise ValueError("Protocol candidate registry does not match the factor module")
    expected_prior = int(protocol["multiplicity"]["prior_tests_included"])
    total_key = f"total_tests_after_v{args.version}"
    expected_total = int(protocol["multiplicity"].get(total_key, -1))
    if expected_total != expected_prior + len(candidate_columns):
        raise ValueError("Protocol multiplicity total is inconsistent")
    prior = pd.concat([pd.read_csv(path) for path in args.prior_summary], ignore_index=True)
    if len(prior) != expected_prior or prior["candidate"].nunique() != expected_prior:
        raise ValueError(f"V{args.version} requires exactly {expected_prior} unique prior tests")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    daily, prices = load_formation_daily(args.data, args.hfq_cache)
    features = compute(daily)
    aligned = align_industry_history(month_end_formation(features), pd.read_parquet(args.industry))
    panels = pd.concat(
        [
            neutralized_candidate_panels(
                aligned,
                candidate_columns=[candidate],
                control_columns=controls[candidate],
                minimum_rows=minimum_rows,
            )
            for candidate in candidate_columns
        ],
        ignore_index=True,
    )
    panels = panels[panels["date"].between("2020-01-01", "2024-12-31")].copy()
    returns, return_manifest = attach_execution_returns(
        panels[["date", "asset"]].drop_duplicates(), prices, daily["date"]
    )
    outcomes = [
        "date", "asset", "entry_date", "exit_date", "entry_delay", "exit_delay",
        "return_status", "future_return_20", "label",
    ]
    evaluated = panels.merge(
        returns[outcomes], on=["date", "asset"], how="left", validate="many_to_one"
    )
    summary, yearly, monthly = evaluate_research(
        evaluated,
        candidate_columns=candidate_columns,
        candidate_definitions=candidates,
        prior_discovery_p_values=prior["discovery_p_one_sided"],
        portfolio_holdings=holdings,
        retention_percentile=retention_percentile,
        cost_bps_one_way=cost_bps_one_way,
        discovery_minimum_months=int(gate.get("discovery_minimum_months", 34)),
        discovery_mean_ic_minimum=float(gate.get("discovery_mean_rank_ic_min", 0.015)),
        discovery_q_maximum=float(gate.get("discovery_bh_fdr_max", 0.10)),
        validation_minimum_months=int(gate.get("validation_minimum_months", 23)),
        validation_mean_ic_minimum=float(gate.get("validation_mean_rank_ic_min", 0.010)),
        validation_mean_net_return_minimum=float(
            gate.get("validation_mean_net_active_return_min", 0.0)
        ),
        validation_net_information_ratio_minimum=float(
            gate.get("validation_net_information_ratio_min", 0.30)
        ),
    )
    exposures = pd.concat(
        [
            exposure_diagnostics(
                panels[panels["candidate"].eq(candidate)],
                control_columns=controls[candidate],
            )
            for candidate in candidate_columns
        ],
        ignore_index=True,
    )
    passing = summary[summary["passes_research_gate"]].sort_values(
        ["validation_mean_ic", "candidate"], ascending=[False, True]
    )
    selected = str(passing.iloc[0]["candidate"]) if not passing.empty else None
    summary.to_csv(output / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "experiment_id": protocol["experiment_id"],
        "research_bucket": protocol.get("research_bucket", "C/hash-0"),
        "adaptive_reuse_disclosed": True,
        "prior_test_count": len(prior),
        "current_test_count": len(candidate_columns),
        "multiplicity_family_size": len(prior) + len(candidate_columns),
        "research_data_sha256": sha256(Path(args.data)),
        "industry_sha256": sha256(Path(args.industry)),
        "protocol_sha256": sha256(protocol_path),
        "return_manifest": return_manifest,
        "passed_candidates": passing["candidate"].tolist(),
        "selected_candidate": selected,
        "confirmation_bucket_opened": False,
    }
    if args.vwap_manifest:
        vwap_manifest = Path(args.vwap_manifest)
        audit = json.loads(vwap_manifest.read_text(encoding="utf-8"))
        if audit.get("confirmation_bucket_opened") is not False:
            raise ValueError("Research VWAP manifest must attest that confirmation remains unopened")
        metadata["vwap_manifest_sha256"] = sha256(vwap_manifest)
    (output / "selection.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
