from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from ashare_alpha.tdx_financial import (
    md5_file,
    parse_financial_manifest,
    select_financial_packages,
    validate_financial_package,
)

OFFICIAL_CDN = "http://data.tdx.com.cn/tdxfin/"
USER_AGENT = "Ashare-Alpha-Discovery/1.0"


def read_url(url: str, *, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def download_url(url: str, destination: Path, *, timeout: float) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MD5-verified TongdaXin quarterly GPCW packages."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", default="2018-03-31")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--base-url", default=OFFICIAL_CDN)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/") + "/"
    manifest_url = urljoin(base_url, "gpcw.txt")
    packages = select_financial_packages(
        parse_financial_manifest(read_url(manifest_url, timeout=args.timeout)),
        start=args.start,
        end=args.end,
    )
    if not packages:
        raise ValueError("No TongdaXin packages fall in the requested date range")

    records: list[dict[str, object]] = []
    for index, package in enumerate(packages, start=1):
        destination = output / package.filename
        status = "cached"
        if not destination.exists() or md5_file(destination) != package.md5:
            temporary = destination.with_suffix(destination.suffix + ".part")
            download_url(
                urljoin(base_url, package.filename),
                temporary,
                timeout=args.timeout,
            )
            validate_financial_package(temporary, package)
            temporary.replace(destination)
            status = "downloaded"
        validate_financial_package(destination, package)
        print(
            f"[{index:02d}/{len(packages):02d}] {status}: "
            f"{package.filename} ({package.filesize} bytes)",
            flush=True,
        )
        records.append(
            {
                "filename": package.filename,
                "report_date": package.report_date.date().isoformat(),
                "filesize": package.filesize,
                "md5": package.md5,
                "status": status,
            }
        )

    audit = {
        "source": "TongdaXin official financial-data CDN",
        "remote_manifest": manifest_url,
        "base_url": base_url,
        "requested_period": [args.start, args.end],
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_count": len(records),
        "packages": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
