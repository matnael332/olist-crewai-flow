"""Download the Olist raw tables with byte-level provenance.

Primary source: Olist's own GitHub repository at a pinned commit (MIT license). Fallbacks: a
HuggingFace mirror of the same files and the Kaggle anonymous zip (CC BY-NC-SA 4.0 listing).
Every file is hashed; data/raw/manifest.json pins the hashes on first download and later runs verify
them, so a silently changed source cannot alter the pipeline's inputs.
"""

from __future__ import annotations

import io as _io
import zipfile
from pathlib import Path

import requests

from pulse_core.io import atomic_replace, read_json, sha256_file, write_json_lf

GITHUB_COMMIT = "d9e49802f3e92d09ee94ab9ccc5e457f207a8959"
SOURCES = {
    "github": (
        f"https://raw.githubusercontent.com/olist/work-at-olist-data/{GITHUB_COMMIT}/datasets/"
    ),
    "hf": "https://huggingface.co/datasets/bulutttt/olist-raw-data/resolve/main/",
}
KAGGLE_ZIP_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce"
    "?datasetVersionNumber=2"
)
KAGGLE_ZIP_SHA256 = "967e41e04fc306fe604e2a693f488995a8b41e5047418f8a5c8e4abd6deca784"

RAW_FILES = (
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
)
MANIFEST_NAME = "manifest.json"
TIMEOUT_S = 120


class ManifestMismatch(Exception):
    """A raw file's sha256 differs from the committed manifest."""


def _stream(url: str, dest: Path) -> None:
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=TIMEOUT_S) as response:
        response.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    atomic_replace(tmp, dest)


def _download_kaggle_zip(raw_dir: Path) -> None:
    response = requests.get(KAGGLE_ZIP_URL, timeout=TIMEOUT_S)
    response.raise_for_status()
    import hashlib

    digest = hashlib.sha256(response.content).hexdigest()
    if digest != KAGGLE_ZIP_SHA256:
        raise ManifestMismatch(f"Kaggle zip sha256 {digest} != pinned {KAGGLE_ZIP_SHA256}")
    with zipfile.ZipFile(_io.BytesIO(response.content)) as archive:
        for name in RAW_FILES:
            target = raw_dir / name
            with archive.open(name) as src, open(target.with_suffix(".part"), "wb") as dst:
                dst.write(src.read())
            atomic_replace(target.with_suffix(".part"), target)


def download_raw(
    raw_dir: str | Path,
    source: str = "github",
    update_manifest: bool = False,
    verify_only: bool = False,
    log=print,
) -> dict:
    """Ensure all raw files exist and match the manifest. Returns the manifest dict.

    Raises ManifestMismatch when a file's hash differs from a committed manifest (unless
    update_manifest is set, which accepts and records the new hashes).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / MANIFEST_NAME
    manifest = read_json(manifest_path) if manifest_path.exists() else None

    missing = [name for name in RAW_FILES if not (raw_dir / name).exists()]
    if missing and verify_only:
        raise FileNotFoundError(f"missing raw files: {missing}")
    if missing:
        log(f"downloading {len(missing)} file(s) from {source}")
        if source == "kaggle":
            _download_kaggle_zip(raw_dir)
        else:
            base = SOURCES[source]
            for name in missing:
                log(f"  {name}")
                _stream(base + name, raw_dir / name)

    files = {}
    for name in RAW_FILES:
        path = raw_dir / name
        files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}

    if manifest is not None and not update_manifest:
        mismatched = [
            name
            for name in RAW_FILES
            if manifest.get("files", {}).get(name, {}).get("sha256") != files[name]["sha256"]
        ]
        if mismatched:
            raise ManifestMismatch(
                f"sha256 mismatch for {mismatched}; rerun with --update-manifest to accept"
            )
        log("all raw files match data/raw/manifest.json")
        return manifest

    new_manifest = {
        "dataset": "Olist Brazilian E-Commerce Public Dataset",
        "source": source,
        "github_commit": GITHUB_COMMIT if source == "github" else None,
        "base_url": SOURCES.get(source, KAGGLE_ZIP_URL),
        "files": files,
    }
    write_json_lf(manifest_path, new_manifest)
    log(f"wrote {manifest_path.as_posix()}")
    return new_manifest
