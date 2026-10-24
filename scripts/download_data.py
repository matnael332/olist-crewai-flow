"""Download and verify the Olist raw tables, then build the zip-prefix centroid lookup.

Usage:
    uv run python scripts/download_data.py                 # GitHub (pinned commit), verify manifest
    uv run python scripts/download_data.py --source kaggle # fallback source
    uv run python scripts/download_data.py --verify-only   # no network; check hashes only

Exit codes: 0 ok, 3 hash mismatch or missing files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pulse_core.cleaning.geo import build_centroid_lookup  # noqa: E402
from pulse_core.download import ManifestMismatch, download_raw  # noqa: E402
from pulse_core.raw import check_raw_schema  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["github", "hf", "kaggle"], default="github")
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--lookups-dir", default=str(ROOT / "data" / "lookups"))
    parser.add_argument("--update-manifest", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    try:
        download_raw(
            args.raw_dir,
            source=args.source,
            update_manifest=args.update_manifest,
            verify_only=args.verify_only,
        )
    except (ManifestMismatch, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 3

    report = check_raw_schema(args.raw_dir)
    if not report.passed:
        print("ERROR: raw schema check failed:")
        for error in report.errors:
            print(f"  - {error}")
        return 3
    print("raw schema OK")

    lookup = build_centroid_lookup(args.raw_dir, args.lookups_dir)
    print(f"centroid lookup: {lookup['rows']} zip prefixes -> {lookup['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
