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
from ashare_alpha.tdx_financial import FIELDS_V40
from run_confirmation_v41 import assert_hash, sha256
from run_construction_v44 import (
    build_construction_panel_v44,
    neutralized_component_v44,
)
from run_discovery_v33 import load_validated_financial_history


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the consumed E+F component cache for adaptive V45+ work."
    )
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--hfq-cache", action="append", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--financial-directory", required=True)
    parser.add_argument("--v42-model", required=True)
    parser.add_argument("--v43-model", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    for path, expected in zip(
        args.data, protocol["freeze"]["construction_data_sha256"], strict=True
    ):
        assert_hash(Path(path), expected, "adaptive component source data")
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
    assert_hash(
        v42_path,
        protocol["components"]["tafs_v42"]["model_sha256"],
        "V42 model",
    )
    assert_hash(
        v43_path,
        protocol["components"]["rrsm_v43"]["model_sha256"],
        "V43 model",
    )
    with v42_path.open("rb") as handle:
        v42_bundle = pickle.load(handle)
    with v43_path.open("rb") as handle:
        v43_bundle = pickle.load(handle)
    matrix_v42, model_columns_v42 = rank_feature_matrix_v42(panel)
    matrix_v43, model_columns_v43 = rank_feature_matrix_v30(panel)
    if model_columns_v42 != v42_bundle["model_columns"]:
        raise ValueError("V42 model columns differ from the frozen artifact")
    if model_columns_v43 != v43_bundle["model_columns"]:
        raise ValueError("V43 model columns differ from the frozen artifact")
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
    components = rank_components_v44(
        rrsm.merge(
            tafs, on=["date", "asset"], how="inner", validate="one_to_one"
        ),
        rrsm_score_column="rrsm_component_score_v44",
        tafs_score_column="tafs_component_score_v44",
    )
    components = combine_component_ranks_v44(components)
    output_columns = [
        "date",
        "asset",
        "float_market_cap",
        "industry_l1",
        "future_return_20",
        "label",
        *final_controls,
    ]
    cache = panel[output_columns].merge(
        components,
        on=["date", "asset"],
        how="inner",
        validate="one_to_one",
    )
    cache = cache[
        cache["date"].between("2023-01-01", "2025-12-31")
    ].sort_values(["date", "asset"]).reset_index(drop=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(output_path, index=False)
    metadata = {
        "role": "consumed_E_plus_F_adaptive_component_cache",
        "contains_bucket_g": False,
        "rows": len(cache),
        "assets": int(cache["asset"].nunique()),
        "months": int(cache["date"].nunique()),
        "minimum_date": str(cache["date"].min().date()),
        "maximum_date": str(cache["date"].max().date()),
        "columns": list(cache.columns),
        "final_main_effect_controls": final_controls,
        "source_data_sha256": [sha256(Path(path)) for path in args.data],
        "industry_sha256": sha256(Path(args.industry)),
        "financial_manifest_sha256": sha256(manifest_path),
        "financial_package_count": int(financial_audit["package_count"]),
        "v42_model_sha256": sha256(v42_path),
        "v43_model_sha256": sha256(v43_path),
        "source_protocol_sha256": sha256(protocol_path),
        "cache_sha256": sha256(output_path),
        "return_manifests": return_manifests,
    }
    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
