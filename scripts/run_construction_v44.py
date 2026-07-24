from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml

from ashare_alpha.composite_v44 import (
    CANDIDATES_V44,
    apply_family_gate_v44,
    combine_component_ranks_v44,
    rank_components_v44,
    select_candidate_v44,
)
from ashare_alpha.confirmation import evaluate_confirmation
from ashare_alpha.data import (
    align_industry_history,
    attach_execution_returns,
    load_formation_daily,
)
from ashare_alpha.evaluate import _neutralize_cross_section, exposure_diagnostics
from ashare_alpha.factors_v33 import attach_point_in_time_financial_v33
from ashare_alpha.factors_v40 import (
    build_quarterly_financial_features_v40,
    finalize_candidates_v40,
)
from ashare_alpha.model_v42 import (
    CANDIDATE_V42,
    PREDICTION_YEARS_V42,
    main_effect_controls_v42,
    predict_walk_forward_v42,
    rank_feature_matrix_v42,
)
from ashare_alpha.model_v43 import (
    CANDIDATE_V43,
    KNOWN_FEATURES_V30,
    PREDICTION_YEARS_V43,
    build_monthly_features_v30,
    predict_walk_forward_v43,
    rank_feature_matrix_v30,
)
from ashare_alpha.tdx_financial import FIELDS_V40
from run_confirmation_v41 import assert_hash, sha256
from run_discovery_v33 import load_validated_financial_history


def build_construction_panel_v44(
    data_paths: list[str],
    cache_paths: list[str],
    industry: pd.DataFrame,
    quarterly_financial: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if len(data_paths) != len(cache_paths):
        raise ValueError("Every construction dataset requires one qfq cache")
    panels: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []
    asset_sets: list[set[str]] = []
    for data_path, cache_path in zip(data_paths, cache_paths, strict=True):
        daily, prices = load_formation_daily(data_path, cache_path)
        asset_sets.append(set(daily["asset"].astype(str).str.zfill(6)))
        structural = align_industry_history(
            build_monthly_features_v30(daily), industry
        )
        features = finalize_candidates_v40(
            attach_point_in_time_financial_v33(
                structural, quarterly_financial
            )
        )
        returns, return_manifest = attach_execution_returns(
            features[["date", "asset"]], prices, daily["date"]
        )
        outcomes = [
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
        panels.append(
            features.merge(
                returns[outcomes],
                on=["date", "asset"],
                how="left",
                validate="one_to_one",
            )
        )
        manifests.append(return_manifest)
    for left in range(len(asset_sets)):
        for right in range(left + 1, len(asset_sets)):
            if asset_sets[left].intersection(asset_sets[right]):
                raise ValueError("V44 construction datasets have overlapping assets")
    panel = pd.concat(panels, ignore_index=True)
    if panel.duplicated(["date", "asset"]).any():
        raise ValueError("V44 combined panel has duplicate date/asset rows")
    return panel, manifests


def neutralized_component_v44(
    panel: pd.DataFrame,
    candidate: str,
    controls: list[str],
    *,
    minimum_rows: int,
    score_name: str,
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    for _, rows in panel.groupby("date", sort=True):
        scored = _neutralize_cross_section(
            rows,
            candidate,
            controls,
            minimum_rows=minimum_rows,
        )
        if not scored.empty:
            output.append(
                scored[["date", "asset", "score"]].rename(
                    columns={"score": score_name}
                )
            )
    if not output:
        raise ValueError(f"V44 component {candidate} produced no scored month")
    return pd.concat(output, ignore_index=True)


def final_neutralized_family_v44(
    panel: pd.DataFrame,
    controls: list[str],
    *,
    minimum_rows: int,
) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    for candidate in CANDIDATES_V44:
        for _, rows in panel.groupby("date", sort=True):
            scored = _neutralize_cross_section(
                rows,
                candidate,
                controls,
                minimum_rows=minimum_rows,
            )
            if not scored.empty:
                output.append(scored)
    if not output:
        raise ValueError("V44 produced no final neutralized month")
    return pd.concat(output, ignore_index=True)


def _load_model_bundle(path: Path, expected_sha256: str) -> dict:
    assert_hash(path, expected_sha256, "frozen component model")
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict):
        raise ValueError("Frozen model artifact is not a mapping")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive E+F construction of the frozen V44 family."
    )
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--hfq-cache", action="append", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--financial-directory", required=True)
    parser.add_argument("--v42-model", required=True)
    parser.add_argument("--v43-model", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    registered = tuple(item["candidate"] for item in protocol["candidates"])
    if registered != CANDIDATES_V44:
        raise ValueError("V44 candidate registry mismatch")
    if tuple(protocol["components"]["prediction_years"]) != PREDICTION_YEARS_V42:
        raise ValueError("V44 prediction years differ from V42")
    if PREDICTION_YEARS_V42 != PREDICTION_YEARS_V43:
        raise ValueError("V42 and V43 prediction years differ")
    for path, expected in zip(
        args.data, protocol["freeze"]["construction_data_sha256"], strict=True
    ):
        assert_hash(Path(path), expected, "V44 construction data")
    assert_hash(
        Path(args.industry),
        protocol["freeze"]["industry_sha256"],
        "industry history",
    )

    financial, financial_audit, manifest_path = load_validated_financial_history(
        Path(args.financial_directory), field_map=FIELDS_V40
    )
    assert_hash(
        manifest_path,
        protocol["freeze"]["financial_manifest_sha256"],
        "financial manifest",
    )
    quarterly = build_quarterly_financial_features_v40(financial)
    panel, return_manifests = build_construction_panel_v44(
        args.data,
        args.hfq_cache,
        pd.read_parquet(args.industry),
        quarterly,
    )
    panel = panel[panel["date"].between("2020-01-01", "2025-12-31")].copy()

    v42_path, v43_path = Path(args.v42_model), Path(args.v43_model)
    v42_bundle = _load_model_bundle(
        v42_path, protocol["components"]["tafs_v42"]["model_sha256"]
    )
    v43_bundle = _load_model_bundle(
        v43_path, protocol["components"]["rrsm_v43"]["model_sha256"]
    )
    matrix_v42, model_columns_v42 = rank_feature_matrix_v42(panel)
    matrix_v43, model_columns_v43 = rank_feature_matrix_v30(panel)
    if model_columns_v42 != v42_bundle["model_columns"]:
        raise ValueError("V42 model columns differ from the frozen artifact")
    if model_columns_v43 != v43_bundle["model_columns"]:
        raise ValueError("V43 model columns differ from the frozen artifact")
    panel[CANDIDATE_V42] = predict_walk_forward_v42(
        panel,
        matrix_v42[model_columns_v42],
        v42_bundle["models"],
    )
    panel[CANDIDATE_V43] = predict_walk_forward_v43(
        panel,
        matrix_v43[model_columns_v43],
        v43_bundle["models"],
    )
    main_effects, final_controls = main_effect_controls_v42(
        matrix_v42[model_columns_v42]
    )
    panel = pd.concat([panel, main_effects], axis=1)

    minimum_rows = int(protocol["neutralization"]["minimum_rows_construction"])
    rrsm = neutralized_component_v44(
        panel,
        CANDIDATE_V43,
        KNOWN_FEATURES_V30,
        minimum_rows=minimum_rows,
        score_name="rrsm_component_score_v44",
    )
    tafs = neutralized_component_v44(
        panel,
        CANDIDATE_V42,
        final_controls,
        minimum_rows=minimum_rows,
        score_name="tafs_component_score_v44",
    )
    components = rrsm.merge(
        tafs, on=["date", "asset"], how="inner", validate="one_to_one"
    )
    components = rank_components_v44(
        components,
        rrsm_score_column="rrsm_component_score_v44",
        tafs_score_column="tafs_component_score_v44",
    )
    components = combine_component_ranks_v44(components)
    composite = panel.merge(
        components[
            [
                "date",
                "asset",
                "rrsm_rank_v44",
                "tafs_rank_v44",
                *CANDIDATES_V44,
            ]
        ],
        on=["date", "asset"],
        how="inner",
        validate="one_to_one",
    )
    scored = final_neutralized_family_v44(
        composite,
        final_controls,
        minimum_rows=minimum_rows,
    )
    scored = scored[
        scored["date"].between("2023-01-01", "2025-12-31")
    ].copy()

    gate, portfolio = protocol["gate"], protocol["portfolio"]
    summaries: list[pd.DataFrame] = []
    yearlies: list[pd.DataFrame] = []
    monthlies: list[pd.DataFrame] = []
    for candidate in CANDIDATES_V44:
        candidate_panel = scored[scored["candidate"].eq(candidate)].copy()
        summary, yearly, monthly = evaluate_confirmation(
            candidate_panel,
            minimum_months=int(gate["minimum_months"]),
            mean_ic_minimum=float(gate["mean_rank_ic_min"]),
            p_value_maximum=float(gate["hac_p_one_sided_max"]),
            positive_years_minimum=int(gate["positive_years_min"]),
            required_positive_year=int(gate["required_positive_year"]),
            mean_net_return_minimum=float(gate["mean_net_active_return_min"]),
            net_information_ratio_minimum=float(
                gate["net_information_ratio_min"]
            ),
            monthly_return_coverage_minimum=float(
                gate["minimum_monthly_return_coverage"]
            ),
            selected_return_coverage_minimum=float(
                gate["minimum_selected_return_coverage"]
            ),
            portfolio_holdings=int(portfolio["holdings"]),
            retention_percentile=float(portfolio["retention_percentile"]),
            cost_bps_one_way=float(portfolio["cost_bps_one_way"]),
        )
        summaries.append(summary)
        yearlies.append(yearly.assign(candidate=candidate))
        monthlies.append(monthly)
    summary = apply_family_gate_v44(
        pd.concat(summaries, ignore_index=True),
        q_maximum=float(gate["family_bh_q_max"]),
    )
    selected_candidate = select_candidate_v44(summary)
    yearly = pd.concat(yearlies, ignore_index=True)
    monthly = pd.concat(monthlies, ignore_index=True)
    exposures = exposure_diagnostics(
        scored, control_columns=final_controls
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        output / "construction_summary.csv", index=False, encoding="utf-8-sig"
    )
    yearly.to_csv(
        output / "yearly_results.csv", index=False, encoding="utf-8-sig"
    )
    monthly.to_csv(
        output / "monthly_results.csv", index=False, encoding="utf-8-sig"
    )
    exposures.to_csv(
        output / "exposure_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "experiment_id": protocol["experiment_id"],
        "candidates": list(CANDIDATES_V44),
        "role": "adaptive_E_plus_F_construction",
        "construction_data_sha256": [sha256(Path(path)) for path in args.data],
        "industry_sha256": sha256(Path(args.industry)),
        "financial_manifest_sha256": sha256(manifest_path),
        "financial_package_count": int(financial_audit["package_count"]),
        "protocol_sha256": sha256(protocol_path),
        "component_model_sha256": {
            "tafs_v42": sha256(v42_path),
            "rrsm_v43": sha256(v43_path),
        },
        "component_model_columns": {
            "tafs_v42": model_columns_v42,
            "rrsm_v43": model_columns_v43,
        },
        "final_main_effect_control_count": len(final_controls),
        "return_manifests": return_manifests,
        "selected_candidate": selected_candidate,
        "eligible_to_freeze_confirmation": selected_candidate is not None,
        "bucket_g_opened": False,
        "out_of_sample_claim": False,
    }
    (output / "construction.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(yearly.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
