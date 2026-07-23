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
from ashare_alpha.factors_v7 import (
    CANDIDATES_V7,
    CANDIDATE_COLUMNS_V7,
    CONTROL_COLUMNS_BY_CANDIDATE_V7,
    compute_candidates_v7,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen DISA family on unseen research bucket C only."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--prior-summary", action="append", required=True)
    parser.add_argument("--protocol", default="configs/discovery_v7.yaml")
    parser.add_argument("--output", default="reports/discovery_v7")
    args = parser.parse_args()

    prior = pd.concat(
        [pd.read_csv(path) for path in args.prior_summary], ignore_index=True
    )
    if len(prior) != 45 or prior["candidate"].nunique() != 45:
        raise ValueError("V7 requires exactly all 45 unique prior candidate tests")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    daily, prices = load_formation_daily(args.data, args.hfq_cache)
    features = compute_candidates_v7(daily)
    formation = month_end_formation(features)
    aligned = align_industry_history(formation, pd.read_parquet(args.industry))
    panels = pd.concat(
        [
            neutralized_candidate_panels(
                aligned,
                candidate_columns=[candidate],
                control_columns=CONTROL_COLUMNS_BY_CANDIDATE_V7[candidate],
                minimum_rows=60,
            )
            for candidate in CANDIDATE_COLUMNS_V7
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
        candidate_columns=CANDIDATE_COLUMNS_V7,
        candidate_definitions=CANDIDATES_V7,
        prior_discovery_p_values=prior["discovery_p_one_sided"],
        portfolio_holdings=20,
        retention_percentile=0.60,
        cost_bps_one_way=20.0,
    )
    exposure_frames = [
        exposure_diagnostics(
            panels[panels["candidate"].eq(candidate)],
            control_columns=CONTROL_COLUMNS_BY_CANDIDATE_V7[candidate],
        )
        for candidate in CANDIDATE_COLUMNS_V7
    ]
    exposures = pd.concat(exposure_frames, ignore_index=True)
    coverage = (
        panels.groupby(["candidate", "date"])["asset"]
        .nunique()
        .rename("complete_assets")
        .reset_index()
    )

    passing = summary[summary["passes_research_gate"]].sort_values(
        ["validation_mean_ic", "candidate"], ascending=[False, True]
    )
    selected = str(passing.iloc[0]["candidate"]) if not passing.empty else None
    summary.to_csv(output / "candidate_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output / "cross_section_coverage.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "experiment_id": "directional-impact-survival-asymmetry-v7",
        "claim_type": "new primitive candidate; prior-art search cannot prove universal novelty",
        "research_bucket": "C/hash-0",
        "prior_test_count": len(prior),
        "current_test_count": len(CANDIDATE_COLUMNS_V7),
        "multiplicity_family_size": len(prior) + len(CANDIDATE_COLUMNS_V7),
        "research_data_sha256": sha256(Path(args.data)),
        "industry_sha256": sha256(Path(args.industry)),
        "protocol_sha256": sha256(Path(args.protocol)),
        "return_manifest": return_manifest,
        "passed_candidates": passing["candidate"].tolist(),
        "selected_candidate": selected,
        "selection_rule": "highest validation mean rank IC among candidates passing every gate",
        "confirmation_bucket_opened": False,
    }
    (output / "selection.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
