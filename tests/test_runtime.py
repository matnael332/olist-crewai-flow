"""Promotion, status file and artifact comparison."""

import shutil

import pytest

from olist_core import promotion
from olist_core.compare import compare_dirs
from olist_core.promotion import PromotionError, promote
from olist_core.status import RunStatus, read_status


def _make(folder, text):
    folder.mkdir(parents=True)
    (folder / "file.txt").write_text(text, encoding="utf-8")
    return folder


def test_promote_swaps_and_keeps_extras(tmp_path):
    target = _make(tmp_path / "artifacts", "old")
    run = _make(tmp_path / "run1", "new")
    log = tmp_path / "pipeline.log"
    log.write_text("log", encoding="utf-8")
    promote(run, target, "r1", extra={"logs/pipeline.log": log})
    assert (target / "file.txt").read_text(encoding="utf-8") == "new"
    assert (target / "logs" / "pipeline.log").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["artifacts", "pipeline.log", "run1"]


def test_promote_rolls_back_when_the_swap_fails(tmp_path, monkeypatch):
    target = _make(tmp_path / "artifacts", "old")
    run = _make(tmp_path / "run1", "new")
    real = promotion.os.replace

    def locked(src, dst):
        if str(src).endswith("staging-r1"):
            raise PermissionError("file in use")
        return real(src, dst)

    monkeypatch.setattr(promotion.os, "replace", locked)
    monkeypatch.setattr(promotion, "DELAY_S", 0)
    with pytest.raises(PromotionError):
        promote(run, target, "r1")
    assert (target / "file.txt").read_text(encoding="utf-8") == "old"
    assert not (tmp_path / "artifacts.staging-r1").exists()
    assert not (tmp_path / "artifacts.prev-r1").exists()


def test_status_file_and_tolerant_reader(tmp_path):
    status = RunStatus(tmp_path / "run_status.json", run_id="r1")
    status.stage_start("check_inputs")
    status.stage_done("check_inputs")
    data = read_status(tmp_path / "run_status.json")
    assert data["stage"] == "check_inputs" and data["stages_done"] == ["check_inputs"]
    (tmp_path / "broken.json").write_text('{"half": ', encoding="utf-8")
    assert read_status(tmp_path / "broken.json") is None
    assert read_status(tmp_path / "missing.json") is None


def test_compare_dirs(tmp_path, pipeline_out):
    a, b = tmp_path / "a", tmp_path / "b"
    shutil.copytree(pipeline_out, a)
    shutil.copytree(pipeline_out, b)
    assert compare_dirs(a, b)["deterministic_identical"]
    with open(b / "features.csv", "ab") as fh:
        fh.write(b"x")
    result = compare_dirs(a, b)
    assert not result["deterministic_identical"]
    assert {r["artifact"]: r["result"] for r in result["rows"]}["features.csv"] == "DIFFERS"
