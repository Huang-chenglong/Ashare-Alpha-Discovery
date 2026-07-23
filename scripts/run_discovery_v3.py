from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ashare_alpha.data import align_industry_history, attach_execution_returns, load_formation_daily
from ashare_alpha.evaluate import (
    evaluate_research,
    exposure_diagnostics,
    month_end_formation,
    neutralized_candidate_panels,
)
from ashare_alpha.factors_v3 import (
    CANDIDATES_V3,
    CANDIDATE_COLUMNS_V3,
    CONTROL_COLUMNS_V3,
    compute_candidates_v3,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen V3 candidates on bucket A only.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--v1-summary", default="reports/discovery_v1/candidate_summary.csv")
    parser.add_argument("--v2-summary", default="reports/discovery_v2/candidate_summary.csv")
    parser.add_argument("--protocol", default="configs/discovery_v3.yaml")
    parser.add_argument("--output", default="reports/discovery_v3")
    arguments = parser.parse_args()

    data_path = Path(arguments.data)
    industry_path = Path(arguments.industry)
    protocol_path = Path(arguments.protocol)
    v1_path = Path(arguments.v1_summary)
    v2_path = Path(arguments.v2_summary)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    prior = pd.concat([pd.read_csv(v1_path), pd.read_csv(v2_path)], ignore_index=True)
    if len(prior) != 12 or prior["candidate"].nunique() != 12:
        raise ValueError("V3 requires all twelve unique V1/V2 candidates")

    daily, prices = load_formation_daily(data_path, arguments.hfq_cache)
    candidates = compute_candidates_v3(daily)
    month_end = month_end_formation(candidates)
    aligned = align_industry_history(month_end, pd.read_parquet(industry_path))
    panels = neutralized_candidate_panels(
        aligned,
        candidate_columns=CANDIDATE_COLUMNS_V3,
        control_columns=CONTROL_COLUMNS_V3,
    )
    panels = panels[panels["date"].between("2020-01-01", "2024-12-31")].copy()

    formation = panels[["date", "asset"]].drop_duplicates()
    returns, return_manifest = attach_execution_returns(formation, prices, daily["date"])
    outcomes = [
        "date", "asset", "entry_date", "exit_date", "entry_delay", "exit_delay",
        "return_status", "future_return_20", "label",
    ]
    evaluated = panels.merge(
        returns[outcomes], on=["date", "asset"], how="left", validate="many_to_one"
    )
    summary, yearly, monthly = evaluate_research(
        evaluated,
        candidate_columns=CANDIDATE_COLUMNS_V3,
        candidate_definitions=CANDIDATES_V3,
        prior_discovery_p_values=prior["discovery_p_one_sided"],
    )
    exposures = exposure_diagnostics(panels, control_columns=CONTROL_COLUMNS_V3)
    selected_rows = summary[summary["passes_research_gate"]]
    selected = str(selected_rows.iloc[0]["candidate"]) if not selected_rows.empty else None

    summary.to_csv(output / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "experiment_id": "dynamic-float-supply-discovery-v3",
        "adaptive_after_v1_v2": True,
        "protocol_sha256": sha256(protocol_path),
        "data_sha256": sha256(data_path),
        "industry_sha256": sha256(industry_path),
        "prior_summary_sha256": {"v1": sha256(v1_path), "v2": sha256(v2_path)},
        "prior_candidate_count": len(prior),
        "new_candidate_count": len(CANDIDATE_COLUMNS_V3),
        "multiplicity_family_size": len(prior) + len(CANDIDATE_COLUMNS_V3),
        "candidates": CANDIDATE_COLUMNS_V3,
        "return_manifest": return_manifest,
        "passed_candidates": selected_rows["candidate"].tolist(),
        "selected_candidate": selected,
        "confirmation_opened": False,
    }
    (output / "selection.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
