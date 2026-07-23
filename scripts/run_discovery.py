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
from ashare_alpha.factors import CANDIDATE_COLUMNS, compute_candidates


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen factor discovery on bucket A.")
    parser.add_argument("--data", required=True, help="Bucket-A structural parquet")
    parser.add_argument("--hfq-cache", required=True, help="Tencent hfq CSV directory")
    parser.add_argument("--industry", required=True, help="Shenwan history parquet")
    parser.add_argument("--protocol", default="configs/discovery_v1.yaml")
    parser.add_argument("--output", default="reports/discovery_v1")
    arguments = parser.parse_args()

    data_path = Path(arguments.data)
    industry_path = Path(arguments.industry)
    protocol_path = Path(arguments.protocol)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)

    daily, prices = load_formation_daily(data_path, arguments.hfq_cache)
    candidates = compute_candidates(daily)
    month_end = month_end_formation(candidates)
    industry = pd.read_parquet(industry_path)
    aligned = align_industry_history(month_end, industry)
    panels = neutralized_candidate_panels(aligned)
    panels = panels[panels["date"].between("2020-01-01", "2024-12-31")].copy()

    unique_formation = panels[["date", "asset"]].drop_duplicates()
    returns, return_manifest = attach_execution_returns(
        unique_formation,
        prices,
        daily["date"],
    )
    outcome_columns = [
        "date", "asset", "entry_date", "exit_date", "entry_delay", "exit_delay",
        "return_status", "future_return_20", "label",
    ]
    evaluated = panels.merge(
        returns[outcome_columns],
        on=["date", "asset"],
        how="left",
        validate="many_to_one",
    )
    summary, yearly, monthly = evaluate_research(evaluated)
    exposures = exposure_diagnostics(panels)
    selected_rows = summary[summary["passes_research_gate"]]
    selected = str(selected_rows.iloc[0]["candidate"]) if not selected_rows.empty else None

    summary.to_csv(output / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "experiment_id": "temporal-participation-discovery-v1",
        "protocol_sha256": sha256(protocol_path),
        "data_sha256": sha256(data_path),
        "industry_sha256": sha256(industry_path),
        "candidate_count": len(CANDIDATE_COLUMNS),
        "candidates": CANDIDATE_COLUMNS,
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
