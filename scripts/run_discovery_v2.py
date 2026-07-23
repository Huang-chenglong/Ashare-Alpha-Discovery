from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ashare_alpha.data import (
    align_industry_history,
    attach_execution_returns,
    load_formation_daily,
)
from ashare_alpha.evaluate import (
    evaluate_research,
    exposure_diagnostics,
    month_end_formation,
    neutralized_candidate_panels,
)
from ashare_alpha.factors_v2 import (
    CANDIDATES_V2,
    CANDIDATE_COLUMNS_V2,
    CONTROL_COLUMNS_V2,
    compute_candidates_v2,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen V2 candidates on research bucket A only."
    )
    parser.add_argument("--data", required=True, help="Bucket-A structural parquet")
    parser.add_argument("--hfq-cache", required=True, help="Tencent hfq CSV directory")
    parser.add_argument("--industry", required=True, help="Shenwan history parquet")
    parser.add_argument(
        "--prior-summary", default="reports/discovery_v1/candidate_summary.csv"
    )
    parser.add_argument("--protocol", default="configs/discovery_v2.yaml")
    parser.add_argument("--output", default="reports/discovery_v2")
    arguments = parser.parse_args()

    data_path = Path(arguments.data)
    industry_path = Path(arguments.industry)
    prior_summary_path = Path(arguments.prior_summary)
    protocol_path = Path(arguments.protocol)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    prior = pd.read_csv(prior_summary_path)
    if len(prior) != 6 or prior["candidate"].nunique() != 6:
        raise ValueError("V2 requires the complete six-candidate V1 summary")

    daily, prices = load_formation_daily(data_path, arguments.hfq_cache)
    candidates = compute_candidates_v2(daily)
    month_end = month_end_formation(candidates)
    industry = pd.read_parquet(industry_path)
    aligned = align_industry_history(month_end, industry)
    panels = neutralized_candidate_panels(
        aligned,
        candidate_columns=CANDIDATE_COLUMNS_V2,
        control_columns=CONTROL_COLUMNS_V2,
    )
    panels = panels[panels["date"].between("2020-01-01", "2024-12-31")].copy()

    unique_formation = panels[["date", "asset"]].drop_duplicates()
    returns, return_manifest = attach_execution_returns(
        unique_formation,
        prices,
        daily["date"],
    )
    outcome_columns = [
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
        returns[outcome_columns],
        on=["date", "asset"],
        how="left",
        validate="many_to_one",
    )
    summary, yearly, monthly = evaluate_research(
        evaluated,
        candidate_columns=CANDIDATE_COLUMNS_V2,
        candidate_definitions=CANDIDATES_V2,
        prior_discovery_p_values=prior["discovery_p_one_sided"],
    )
    exposures = exposure_diagnostics(panels, control_columns=CONTROL_COLUMNS_V2)
    selected_rows = summary[summary["passes_research_gate"]]
    selected = str(selected_rows.iloc[0]["candidate"]) if not selected_rows.empty else None

    summary.to_csv(output / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "experiment_id": "active-gap-discovery-v2",
        "adaptive_after_v1": True,
        "protocol_sha256": sha256(protocol_path),
        "data_sha256": sha256(data_path),
        "industry_sha256": sha256(industry_path),
        "prior_summary_sha256": sha256(prior_summary_path),
        "prior_candidate_count": len(prior),
        "new_candidate_count": len(CANDIDATE_COLUMNS_V2),
        "multiplicity_family_size": len(prior) + len(CANDIDATE_COLUMNS_V2),
        "candidates": CANDIDATE_COLUMNS_V2,
        "research_period": ["2020-01-01", "2024-12-31"],
        "return_manifest": return_manifest,
        "passed_candidates": selected_rows["candidate"].tolist(),
        "selected_candidate": selected,
        "confirmation_opened": False,
        "selection_rule": (
            "highest validation net information ratio, then validation mean IC, then name"
        ),
    }
    (output / "selection.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
