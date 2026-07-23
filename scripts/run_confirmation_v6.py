from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ashare_alpha.confirmation import evaluate_confirmation
from ashare_alpha.data import align_industry_history, attach_execution_returns, load_formation_daily
from ashare_alpha.evaluate import exposure_diagnostics, month_end_formation, neutralized_candidate_panels
from ashare_alpha.factors_v6 import CANDIDATE_COLUMNS_V6, compute_candidates_v6


SELECTED_CANDIDATE = "supply_aware_defensive_attention"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot V6 confirmation on asset-disjoint bucket B.")
    parser.add_argument("--research-data", required=True)
    parser.add_argument("--confirmation-data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--research-selection", default="reports/discovery_v6/selection.json")
    parser.add_argument("--protocol", default="configs/confirmation_v6.yaml")
    parser.add_argument("--output", default="reports/confirmation_v6")
    args = parser.parse_args()

    selection = json.loads(Path(args.research_selection).read_text(encoding="utf-8"))
    if selection.get("selected_candidate") != SELECTED_CANDIDATE:
        raise ValueError("Research selection does not authorize the frozen V6 candidate")
    research_assets = set(
        pd.read_parquet(args.research_data, columns=["asset"])["asset"].astype(str).str.zfill(6)
    )
    confirmation_assets = set(
        pd.read_parquet(args.confirmation_data, columns=["asset"])["asset"].astype(str).str.zfill(6)
    )
    intersection = sorted(research_assets.intersection(confirmation_assets))
    if intersection:
        raise ValueError(f"Research and confirmation assets overlap: {intersection[:10]}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    daily, prices = load_formation_daily(args.confirmation_data, args.hfq_cache)
    features = compute_candidates_v6(daily)
    aligned = align_industry_history(month_end_formation(features), pd.read_parquet(args.industry))
    panels = neutralized_candidate_panels(
        aligned, candidate_columns=CANDIDATE_COLUMNS_V6, control_columns=[]
    )
    panels = panels[panels["date"].between("2020-01-01", "2026-05-31")].copy()
    returns, return_manifest = attach_execution_returns(
        panels[["date", "asset"]].drop_duplicates(), prices, daily["date"]
    )
    outcomes = ["date", "asset", "entry_date", "exit_date", "entry_delay", "exit_delay", "return_status", "future_return_20", "label"]
    evaluated = panels.merge(returns[outcomes], on=["date", "asset"], how="left", validate="many_to_one")
    summary, yearly, monthly = evaluate_confirmation(evaluated)
    exposures = exposure_diagnostics(panels, control_columns=[])

    summary.to_csv(output / "confirmation_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "experiment_id": "supply-aware-defensive-attention-confirmation-v6",
        "candidate": SELECTED_CANDIDATE,
        "research_assets": len(research_assets),
        "confirmation_assets": len(confirmation_assets),
        "asset_intersection_count": len(intersection),
        "research_data_sha256": sha256(Path(args.research_data)),
        "confirmation_data_sha256": sha256(Path(args.confirmation_data)),
        "industry_sha256": sha256(Path(args.industry)),
        "protocol_sha256": sha256(Path(args.protocol)),
        "research_selection_sha256": sha256(Path(args.research_selection)),
        "return_manifest": return_manifest,
        "passes_confirmation_gate": bool(summary.iloc[0]["passes_confirmation_gate"]),
        "confirmation_opened": True,
        "formula_changed_after_open": False,
    }
    (output / "confirmation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(yearly.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
