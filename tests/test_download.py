import shutil

import pytest

from olist_core.download import ManifestMismatch, download_raw
from olist_core.raw import check_raw_schema


def _copy(src, dst):
    shutil.copytree(src, dst)
    return dst


def test_verify_only_accepts_matching_files(fixture_raw_dir):
    manifest = download_raw(fixture_raw_dir, verify_only=True, log=lambda *_: None)
    assert len(manifest["files"]) == 9


def test_tampered_file_fails_manifest(tmp_path, fixture_raw_dir):
    raw = _copy(fixture_raw_dir, tmp_path / "raw")
    with open(raw / "olist_orders_dataset.csv", "ab") as fh:
        fh.write(b"\n")
    with pytest.raises(ManifestMismatch):
        download_raw(raw, verify_only=True, log=lambda *_: None)


def test_raw_schema_detects_missing_column(tmp_path, fixture_raw_dir):
    raw = _copy(fixture_raw_dir, tmp_path / "raw")
    assert check_raw_schema(raw).passed
    path = raw / "olist_sellers_dataset.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace(",seller_state", "")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = check_raw_schema(raw)
    assert not report.passed
    assert any("seller_state" in e for e in report.errors)
