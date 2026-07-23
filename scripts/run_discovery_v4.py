from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ashare_alpha.data import align_industry_history, attach_execution_returns, load_formation_daily
from ashare_alpha.evaluate import evaluate_research, exposure_diagnostics, month_end_formation, neutralized_candidate_panels
from ashare_alpha.factors_v4 import CANDIDATES_V4, CANDIDATE_COLUMNS_V4, CONTROL_COLUMNS_V4, compute_candidates_v4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen V4 candidates on bucket A only.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--v1-summary", default="reports/discovery_v1/candidate_summary.csv")
    parser.add_argument("--v2-summary", default="reports/discovery_v2/candidate_summary.csv")
    parser.add_argument("--v3-summary", default="reports/discovery_v3/candidate_summary.csv")
    parser.add_argument("--protocol", default="configs/discovery_v4.yaml")
    parser.add_argument("--output", default="reports/discovery_v4")
    args = parser.parse_args()

    data_path, industry_path, protocol_path = Path(args.data), Path(args.industry), Path(args.protocol)
    prior_paths = [Path(args.v1_summary), Path(args.v2_summary), Path(args.v3_summary)]
    prior = pd.concat([pd.read_csv(path) for path in prior_paths], ignore_index=True)
    if len(prior) != 18 or prior["candidate"].nunique() != 18:
        raise ValueError("V4 requires all eighteen unique V1-V3 candidates")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    daily, prices = load_formation_daily(data_path, args.hfq_cache)
    candidates = compute_candidates_v4(daily)
    aligned = align_industry_history(month_end_formation(candidates), pd.read_parquet(industry_path))
    panels = neutralized_candidate_panels(
        aligned, candidate_columns=CANDIDATE_COLUMNS_V4, control_columns=CONTROL_COLUMNS_V4
    )
    panels = panels[panels["date"].between("2020-01-01", "2024-12-31")].copy()
    formation = panels[["date", "asset"]].drop_duplicates()
    returns, return_manifest = attach_execution_returns(formation, prices, daily["date"])
    outcomes = ["date", "asset", "entry_date", "exit_date", "entry_delay", "exit_delay", "return_status", "future_return_20", "label"]
    evaluated = panels.merge(returns[outcomes], on=["date", "asset"], how="left", validate="many_to_one")
    summary, yearly, monthly = evaluate_research(
        evaluated,
        candidate_columns=CANDIDATE_COLUMNS_V4,
        candidate_definitions=CANDIDATES_V4,
        prior_discovery_p_values=prior["discovery_p_one_sided"],
    )
    exposures = exposure_diagnostics(panels, control_columns=CONTROL_COLUMNS_V4)
    selected_rows = summary[summary["passes_research_gate"]]
    selected = str(selected_rows.iloc[0]["candidate"]) if not selected_rows.empty else None

    summary.to_csv(output / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "experiment_id": "upside-tail-concentration-discovery-v4",
        "adaptive_after_v1_v2_v3": True,
        "protocol_sha256": sha256(protocol_path),
        "data_sha256": sha256(data_path),
        "industry_sha256": sha256(industry_path),
        "prior_summary_sha256": {path.stem + f"_{index+1}": sha256(path) for index, path in enumerate(prior_paths)},
        "prior_candidate_count": len(prior),
        "new_candidate_count": len(CANDIDATE_COLUMNS_V4),
        "multiplicity_family_size": len(prior) + len(CANDIDATE_COLUMNS_V4),
        "candidates": CANDIDATE_COLUMNS_V4,
        "return_manifest": return_manifest,
        "passed_candidates": selected_rows["candidate"].tolist(),
        "selected_candidate": selected,
        "confirmation_opened": False,
    }
    (output / "selection.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
