from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ashare_alpha.baselines import BASELINES, BASELINE_COLUMNS, compute_baselines
from ashare_alpha.data import align_industry_history, attach_execution_returns, load_formation_daily
from ashare_alpha.evaluate import evaluate_research, exposure_diagnostics, month_end_formation, neutralized_candidate_panels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate known baselines; none is confirmable as novel.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--prior-summary", action="append", required=True)
    parser.add_argument("--output", default="reports/baseline_diagnostics")
    args = parser.parse_args()
    prior_paths = [Path(value) for value in args.prior_summary]
    prior = pd.concat([pd.read_csv(path) for path in prior_paths], ignore_index=True)
    if len(prior) != 24 or prior["candidate"].nunique() != 24:
        raise ValueError("Baseline diagnostics require all 24 unique V1-V4 candidates")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    daily, prices = load_formation_daily(args.data, args.hfq_cache)
    features = compute_baselines(daily)
    aligned = align_industry_history(month_end_formation(features), pd.read_parquet(args.industry))
    panels = neutralized_candidate_panels(
        aligned, candidate_columns=BASELINE_COLUMNS, control_columns=[]
    )
    panels = panels[panels["date"].between("2020-01-01", "2024-12-31")].copy()
    returns, manifest = attach_execution_returns(
        panels[["date", "asset"]].drop_duplicates(), prices, daily["date"]
    )
    outcomes = ["date", "asset", "entry_date", "exit_date", "entry_delay", "exit_delay", "return_status", "future_return_20", "label"]
    evaluated = panels.merge(returns[outcomes], on=["date", "asset"], how="left", validate="many_to_one")
    summary, yearly, monthly = evaluate_research(
        evaluated,
        candidate_columns=BASELINE_COLUMNS,
        candidate_definitions=BASELINES,
        prior_discovery_p_values=prior["discovery_p_one_sided"],
    )
    exposures = exposure_diagnostics(panels, control_columns=[])
    summary["eligible_for_confirmation"] = False
    summary.to_csv(output / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "experiment_id": "known-baseline-diagnostics",
        "purpose": "pipeline calibration; no baseline is eligible for novel-factor confirmation",
        "prior_candidate_count": len(prior),
        "baseline_count": len(BASELINE_COLUMNS),
        "multiplicity_family_size": len(prior) + len(BASELINE_COLUMNS),
        "return_manifest": manifest,
        "confirmation_opened": False,
    }
    (output / "selection.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
