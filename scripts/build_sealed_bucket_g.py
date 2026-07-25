from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ashare_alpha.sealed_bucket import make_asset_disjoint_bucket, normalize_assets


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_asset_set(path: Path) -> set[str]:
    assets = pd.read_parquet(path, columns=["asset"])["asset"]
    return set(normalize_assets(assets))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sealed asset-disjoint bucket G without computing outcomes."
    )
    parser.add_argument("--candidate-data", required=True)
    parser.add_argument("--exclude-data", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-assets", type=int, default=300)
    parser.add_argument("--minimum-monthly-members", type=int, default=190)
    args = parser.parse_args()

    candidate_path = Path(args.candidate_data)
    exclusion_paths = [Path(path) for path in args.exclude_data]
    output_path = Path(args.output)
    if output_path.exists() or output_path.with_suffix(".manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite sealed output: {output_path}")

    excluded_sets = [read_asset_set(path) for path in exclusion_paths]
    candidate = pd.read_parquet(candidate_path)
    output, manifest = make_asset_disjoint_bucket(
        candidate,
        excluded_sets,
        minimum_assets=args.minimum_assets,
        minimum_monthly_members=args.minimum_monthly_members,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    manifest.update(
        {
            "role": "sealed formula-unseen bucket G for V42 only",
            "selection": (
                "All CSI500 structural-panel assets whose code never appears "
                "in construction buckets E or F"
            ),
            "global_pristine_claim": False,
            "factor_or_return_metrics_inspected": False,
            "candidate_data": str(candidate_path),
            "candidate_data_sha256": sha256(candidate_path),
            "excluded_data": [str(path) for path in exclusion_paths],
            "excluded_data_sha256": [sha256(path) for path in exclusion_paths],
            "output_sha256": sha256(output_path),
        }
    )
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
