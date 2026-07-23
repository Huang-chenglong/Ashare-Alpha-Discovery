from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ashare_alpha.unseen_universe import (
    build_bucket_panel,
    build_unseen_memberships,
    collect_seen_assets,
    download_tencent_hfq,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build asset-disjoint C/D panels after excluding every prior experiment asset."
    )
    parser.add_argument("--tdx-root", required=True)
    parser.add_argument("--gbbq", required=True)
    parser.add_argument("--seen", action="append", required=True)
    parser.add_argument("--output", default="data/unseen")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-05-31")
    parser.add_argument("--price-start", default="2018-07-01")
    parser.add_argument("--price-end", default="2026-07-20")
    parser.add_argument("--minimum-members", type=int, default=80)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    output = Path(args.output)
    cache = output / "tencent_hfq"
    output.mkdir(parents=True, exist_ok=True)
    seen_assets, seen_sources = collect_seen_assets(args.seen)
    memberships, selection_manifest = build_unseen_memberships(
        args.tdx_root,
        seen_assets,
        start=args.start,
        end=args.end,
        minimum_members=args.minimum_members,
    )
    membership_paths: dict[int, Path] = {}
    for bucket, membership in memberships.items():
        path = output / f"bucket_{'c' if bucket == 0 else 'd'}_membership.csv"
        membership.to_csv(path, index=False, encoding="utf-8-sig")
        membership_paths[bucket] = path

    all_assets = sorted(set().union(*(set(frame["asset"]) for frame in memberships.values())))
    completed = 0

    def progress(message: str) -> None:
        nonlocal completed
        completed += 1
        if completed % 50 == 0 or "FAILED" in message or completed == len(all_assets):
            print(message)

    status = download_tencent_hfq(
        all_assets,
        cache,
        args.price_start,
        args.price_end,
        workers=args.workers,
        progress=progress,
    )
    panel_manifests: dict[str, object] = {}
    panel_paths: dict[int, Path] = {}
    for bucket, membership in memberships.items():
        label = "c" if bucket == 0 else "d"
        panel, panel_manifest = build_bucket_panel(
            membership, cache, args.tdx_root, args.gbbq
        )
        path = output / f"bucket_{label}_structural.parquet"
        panel.to_parquet(path, index=False)
        panel_paths[bucket] = path
        panel_manifests[label] = {
            **panel_manifest,
            "membership_sha256": sha256(membership_paths[bucket]),
            "panel_sha256": sha256(path),
        }

    manifest = {
        "experiment": "unseen-asset-buckets-c-d",
        "seen_sources": seen_sources,
        "seen_source_count": len(seen_sources),
        "downloaded_assets": sum(value == "downloaded" for value in status.values()),
        "cached_assets": sum(value == "cached" for value in status.values()),
        "gbbq_sha256": sha256(Path(args.gbbq)),
        "selection": selection_manifest,
        "panels": panel_manifests,
        "research_bucket": "C/hash-0",
        "confirmation_bucket": "D/hash-1",
        "confirmation_opened": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
