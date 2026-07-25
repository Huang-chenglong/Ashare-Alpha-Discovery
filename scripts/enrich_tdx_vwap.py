from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from ashare_alpha.unseen_universe import attach_adjusted_tdx_vwap


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach adjusted actual-TDX VWAP to one sealed panel.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--tdx-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    source = Path(args.data)
    output = Path(args.output)
    manifest_path = Path(args.manifest)
    panel, audit = attach_adjusted_tdx_vwap(pd.read_parquet(source), args.tdx_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output, index=False)
    audit.update(
        {
            "source_data_sha256": sha256(source),
            "output_data_sha256": sha256(output),
            "confirmation_bucket_opened": False,
        }
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
