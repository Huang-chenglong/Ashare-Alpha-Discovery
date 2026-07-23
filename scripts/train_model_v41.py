from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from ashare_alpha.data import align_industry_history, attach_execution_returns, load_formation_daily
from ashare_alpha.model_v30 import (
    MODEL_PARAMETERS_V30,
    build_monthly_features_v30,
    fit_model_v30,
    rank_feature_matrix_v30,
    residualize_training_target_v30,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the frozen E-construction structural component for V41."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--hfq-cache", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--metadata-output", required=True)
    args = parser.parse_args()

    daily, prices = load_formation_daily(args.data, args.hfq_cache)
    features = align_industry_history(
        build_monthly_features_v30(daily), pd.read_parquet(args.industry)
    )
    returns, return_manifest = attach_execution_returns(
        features[["date", "asset"]], prices, daily["date"]
    )
    panel = features.merge(
        returns[["date", "asset", "label"]],
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    panel = panel[panel["date"].between("2020-01-01", "2022-12-31")].copy()
    matrix, model_columns = rank_feature_matrix_v30(panel)
    target = residualize_training_target_v30(panel)
    model = fit_model_v30(matrix[model_columns], target, panel["date"])
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(
            {
                "model": model,
                "columns": model_columns,
                "parameters": MODEL_PARAMETERS_V30,
            },
            handle,
            protocol=5,
        )
    metadata = {
        "experiment_id": "v41-e-construction-model",
        "construction_period": ["2020-01-01", "2022-12-31"],
        "construction_data_sha256": sha256(Path(args.data)),
        "industry_sha256": sha256(Path(args.industry)),
        "model_sha256": sha256(model_path),
        "model_columns": model_columns,
        "training_target_rows": int(target.notna().sum()),
        "model_parameters": MODEL_PARAMETERS_V30,
        "return_manifest": return_manifest,
        "outcome_metrics_inspected": False,
        "confirmation_bucket_opened": False,
    }
    metadata_path = Path(args.metadata_output)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
