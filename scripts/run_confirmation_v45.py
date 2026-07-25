from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
import yaml

from ashare_alpha.composite_v44 import (
    combine_component_ranks_v44,
    rank_components_v44,
)
from ashare_alpha.confirmation import evaluate_confirmation
from ashare_alpha.evaluate import _neutralize_cross_section, exposure_diagnostics
from ashare_alpha.factors_v40 import build_quarterly_financial_features_v40
from ashare_alpha.model_v42 import (
    CANDIDATE_V42,
    main_effect_controls_v42,
    predict_walk_forward_v42,
    rank_feature_matrix_v42,
)
from ashare_alpha.model_v43 import (
    CANDIDATE_V43,
    KNOWN_FEATURES_V30,
    predict_walk_forward_v43,
    rank_feature_matrix_v30,
)
from ashare_alpha.persistence_v45 import build_persistence_candidates_v45
from ashare_alpha.tdx_financial import FIELDS_V40
from run_confirmation_v41 import assert_hash, sha256
from run_construction_v44 import (
    build_construction_panel_v44,
    neutralized_component_v44,
)
from run_discovery_v33 import load_validated_financial_history


def neutralize_selected_v45(
    panel: pd.DataFrame,
    candidate: str,
    controls: list[str],
    *,
    minimum_rows: int,
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
            output.append(scored)
    if not output:
        raise ValueError("V45 confirmation produced no neutralized month")
    return pd.concat(output, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot asset-disjoint bucket G confirmation for V45."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--financial-directory", required=True)
    parser.add_argument("--v42-model", required=True)
    parser.add_argument("--v43-model", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    candidate = protocol["candidate"]
    if candidate != "pcs60_25_15_v45":
        raise ValueError("V45 confirmation candidate differs from frozen selection")
    assert_hash(
        Path(args.data),
        protocol["bucket_g"]["structural_data_sha256"],
        "bucket G structural data",
    )
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
    v42_path, v43_path = Path(args.v42_model), Path(args.v43_model)
    assert_hash(v42_path, protocol["models"]["v42_sha256"], "V42 model")
    assert_hash(v43_path, protocol["models"]["v43_sha256"], "V43 model")
    with v42_path.open("rb") as handle:
        v42_bundle = pickle.load(handle)
    with v43_path.open("rb") as handle:
        v43_bundle = pickle.load(handle)

    quarterly = build_quarterly_financial_features_v40(financial)
    panel, return_manifests = build_construction_panel_v44(
        [args.data],
        [args.hfq_cache],
        pd.read_parquet(args.industry),
        quarterly,
    )
    panel = panel[panel["date"].between("2020-01-01", "2025-12-31")].copy()
    matrix_v42, model_columns_v42 = rank_feature_matrix_v42(panel)
    matrix_v43, model_columns_v43 = rank_feature_matrix_v30(panel)
    if model_columns_v42 != v42_bundle["model_columns"]:
        raise ValueError("V42 model columns differ from frozen artifact")
    if model_columns_v43 != v43_bundle["model_columns"]:
        raise ValueError("V43 model columns differ from frozen artifact")
    panel[CANDIDATE_V42] = predict_walk_forward_v42(
        panel, matrix_v42[model_columns_v42], v42_bundle["models"]
    )
    panel[CANDIDATE_V43] = predict_walk_forward_v43(
        panel, matrix_v43[model_columns_v43], v43_bundle["models"]
    )
    main_effects, final_controls = main_effect_controls_v42(
        matrix_v42[model_columns_v42]
    )
    panel = pd.concat([panel, main_effects], axis=1)
    minimum_rows = int(protocol["neutralization"]["minimum_rows"])
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
    components = rank_components_v44(
        rrsm.merge(
            tafs, on=["date", "asset"], how="inner", validate="one_to_one"
        ),
        rrsm_score_column="rrsm_component_score_v44",
        tafs_score_column="tafs_component_score_v44",
    )
    components = combine_component_ranks_v44(components)
    composite = panel.merge(
        components,
        on=["date", "asset"],
        how="inner",
        validate="one_to_one",
    )
    persistent = build_persistence_candidates_v45(
        composite[
            [
                "date",
                "asset",
                "rrsm_rank_v44",
                "tafs_rank_v44",
            ]
        ]
    )
    selected = composite.merge(
        persistent[["date", "asset", candidate]],
        on=["date", "asset"],
        how="inner",
        validate="one_to_one",
    )
    scored = neutralize_selected_v45(
        selected,
        candidate,
        final_controls,
        minimum_rows=minimum_rows,
    )

    gate, portfolio = protocol["gate"], protocol["portfolio"]
    summary, yearly, monthly = evaluate_confirmation(
        scored,
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
    exposures = exposure_diagnostics(scored, control_columns=final_controls)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        output / "confirmation_summary.csv", index=False, encoding="utf-8-sig"
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
        "candidate": candidate,
        "role": "one_shot_asset_disjoint_bucket_G_confirmation",
        "bucket_g_sha256": sha256(Path(args.data)),
        "industry_sha256": sha256(Path(args.industry)),
        "financial_manifest_sha256": sha256(manifest_path),
        "financial_package_count": int(financial_audit["package_count"]),
        "v42_model_sha256": sha256(v42_path),
        "v43_model_sha256": sha256(v43_path),
        "protocol_sha256": sha256(protocol_path),
        "return_manifests": return_manifests,
        "passes_confirmation_gate": bool(
            summary.iloc[0]["passes_confirmation_gate"]
        ),
        "bucket_g_opened": True,
        "formula_unseen_before_run": True,
        "asset_overlap_with_E_F": 0,
        "future_time_holdout": False,
        "global_pristine_claim": False,
    }
    (output / "confirmation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(yearly.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
