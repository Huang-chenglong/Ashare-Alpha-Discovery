from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ashare_alpha.data import align_industry_history, attach_execution_returns, load_formation_daily
from ashare_alpha.evaluate import evaluate_research, exposure_diagnostics, month_end_formation, neutralized_candidate_panels
from ashare_alpha.factors_v5 import CANDIDATES_V5, CANDIDATE_COLUMNS_V5, CONTROL_COLUMNS_V5, compute_candidates_v5


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen V5 candidates on bucket A only.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--prior-summary", action="append", required=True)
    parser.add_argument("--output", default="reports/discovery_v5")
    args = parser.parse_args()
    prior_paths = [Path(value) for value in args.prior_summary]
    prior = pd.concat([pd.read_csv(path) for path in prior_paths], ignore_index=True)
    if len(prior) != 38 or prior["candidate"].nunique() != 38:
        raise ValueError("V5 requires all 38 unique prior research and diagnostic tests")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    daily, prices = load_formation_daily(args.data, args.hfq_cache)
    features = compute_candidates_v5(daily)
    aligned = align_industry_history(month_end_formation(features), pd.read_parquet(args.industry))
    panels = neutralized_candidate_panels(
        aligned, candidate_columns=CANDIDATE_COLUMNS_V5, control_columns=CONTROL_COLUMNS_V5
    )
    panels = panels[panels["date"].between("2020-01-01", "2024-12-31")].copy()
    returns, manifest = attach_execution_returns(
        panels[["date", "asset"]].drop_duplicates(), prices, daily["date"]
    )
    outcomes = ["date", "asset", "entry_date", "exit_date", "entry_delay", "exit_delay", "return_status", "future_return_20", "label"]
    evaluated = panels.merge(returns[outcomes], on=["date", "asset"], how="left", validate="many_to_one")
    summary, yearly, monthly = evaluate_research(
        evaluated,
        candidate_columns=CANDIDATE_COLUMNS_V5,
        candidate_definitions=CANDIDATES_V5,
        prior_discovery_p_values=prior["discovery_p_one_sided"],
    )
    exposures = exposure_diagnostics(panels, control_columns=CONTROL_COLUMNS_V5)
    selected_rows = summary[summary["passes_research_gate"]]
    selected = str(selected_rows.iloc[0]["candidate"]) if not selected_rows.empty else None

    summary.to_csv(output / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "experiment_id": "defensive-bottleneck-residual-discovery-v5",
        "prior_test_count": len(prior),
        "new_candidate_count": len(CANDIDATE_COLUMNS_V5),
        "multiplicity_family_size": len(prior) + len(CANDIDATE_COLUMNS_V5),
        "return_manifest": manifest,
        "passed_candidates": selected_rows["candidate"].tolist(),
        "selected_candidate": selected,
        "confirmation_opened": False,
    }
    (output / "selection.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
