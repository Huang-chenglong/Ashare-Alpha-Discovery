from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from ashare_alpha.confirmation import evaluate_confirmation
from ashare_alpha.evaluate import _neutralize_cross_section, exposure_diagnostics
from ashare_alpha.stability_v47 import (
    CANDIDATES_V47,
    apply_family_gate_v47,
    build_stability_candidates_v47,
    select_candidate_v47,
)
from run_confirmation_v41 import assert_hash, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen V47 E+F+G construction.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--cache-metadata", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if tuple(item["candidate"] for item in protocol["candidates"]) != CANDIDATES_V47:
        raise ValueError("V47 candidate registry mismatch")
    cache_path = Path(args.cache)
    assert_hash(cache_path, protocol["cache"]["sha256"], "V47 adaptive cache")
    metadata = json.loads(Path(args.cache_metadata).read_text(encoding="utf-8"))
    if metadata["contains_bucket_h"] or metadata["cache_sha256"] != protocol["cache"]["sha256"]:
        raise ValueError("V47 cache evidence roles are invalid")
    controls = metadata["final_main_effect_controls"]
    panel = build_stability_candidates_v47(pd.read_parquet(cache_path))
    scored_parts: list[pd.DataFrame] = []
    for candidate in CANDIDATES_V47:
        for _, rows in panel.groupby("date", sort=True):
            scored = _neutralize_cross_section(
                rows,
                candidate,
                controls,
                minimum_rows=int(protocol["neutralization"]["minimum_rows_construction"]),
            )
            if not scored.empty:
                scored_parts.append(scored)
    if not scored_parts:
        raise ValueError("V47 produced no neutralized month")
    scored = pd.concat(scored_parts, ignore_index=True)

    gate, portfolio = protocol["gate"], protocol["portfolio"]
    summaries, yearlies, monthlies = [], [], []
    for candidate in CANDIDATES_V47:
        summary, yearly, monthly = evaluate_confirmation(
            scored[scored["candidate"].eq(candidate)].copy(),
            minimum_months=int(gate["minimum_months"]),
            mean_ic_minimum=float(gate["mean_rank_ic_min"]),
            p_value_maximum=float(gate["hac_p_one_sided_max"]),
            positive_years_minimum=int(gate["positive_years_min"]),
            required_positive_year=int(gate["required_positive_year"]),
            mean_net_return_minimum=float(gate["mean_net_active_return_min"]),
            net_information_ratio_minimum=float(gate["net_information_ratio_min"]),
            monthly_return_coverage_minimum=float(gate["minimum_monthly_return_coverage"]),
            selected_return_coverage_minimum=float(gate["minimum_selected_return_coverage"]),
            portfolio_holdings=int(portfolio["holdings"]),
            retention_percentile=float(portfolio["retention_percentile"]),
            cost_bps_one_way=float(portfolio["cost_bps_one_way"]),
        )
        summaries.append(summary)
        yearlies.append(yearly.assign(candidate=candidate))
        monthlies.append(monthly)
    summary = apply_family_gate_v47(
        pd.concat(summaries, ignore_index=True),
        q_maximum=float(gate["family_bh_q_max"]),
    )
    selected_candidate = select_candidate_v47(summary)
    yearly = pd.concat(yearlies, ignore_index=True)
    monthly = pd.concat(monthlies, ignore_index=True)
    exposures = exposure_diagnostics(scored, control_columns=controls)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "construction_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output / "yearly_results.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_results.csv", index=False, encoding="utf-8-sig")
    exposures.to_csv(output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig")
    result = {
        "experiment_id": protocol["experiment_id"],
        "candidates": list(CANDIDATES_V47),
        "role": "adaptive_E_plus_F_plus_G_construction",
        "cache_sha256": sha256(cache_path),
        "cache_metadata_sha256": sha256(Path(args.cache_metadata)),
        "protocol_sha256": sha256(protocol_path),
        "selected_candidate": selected_candidate,
        "eligible_to_freeze_confirmation": selected_candidate is not None,
        "bucket_h_opened": False,
        "out_of_sample_claim": False,
    }
    (output / "construction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(yearly.to_string(index=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
