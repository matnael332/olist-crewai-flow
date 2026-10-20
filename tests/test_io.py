import json

import numpy as np
import pandas as pd
import pytest

from olist_core import io
from olist_core.sorting import counts_sorted, stable_sort


def test_write_text_lf_uses_utf8_and_lf(tmp_path):
    path = io.write_text_lf(tmp_path / "a" / "note.md", "line 1\nSão Paulo → ok\n")
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.decode("utf-8") == "line 1\nSão Paulo → ok\n"
    assert not (tmp_path / "a" / "note.md.tmp").exists()


def test_write_json_lf_is_sorted_rounded_and_builtin(tmp_path):
    data = {
        "b": np.float64(1.23456789),
        "a": np.int64(3),
        "c": [np.nan, pd.Timestamp("2018-01-02")],
    }
    path = io.write_json_lf(tmp_path / "x.json", data)
    text = path.read_text(encoding="utf-8")
    assert "np.float64" not in text and "\r" not in text
    assert list(json.loads(text)) == ["a", "b", "c"]
    assert json.loads(text) == {"a": 3, "b": 1.234568, "c": [None, "2018-01-02T00:00:00"]}


def test_write_csv_stable_is_byte_identical(tmp_path):
    df = pd.DataFrame(
        {"x": [1.123456, 2.0], "t": pd.to_datetime(["2018-01-01 10:00:00", "2018-01-02 00:00:00"])}
    )
    first = io.write_csv_stable(df, tmp_path / "a.csv").read_bytes()
    second = io.write_csv_stable(df, tmp_path / "b.csv").read_bytes()
    assert first == second
    assert b"\r" not in first
    assert first.decode().splitlines()[1] == "1.1235,2018-01-01 10:00:00"


def test_atomic_replace_retries_on_permission_error(tmp_path, monkeypatch):
    src, dst = tmp_path / "src.txt", tmp_path / "dst.txt"
    src.write_text("new", encoding="utf-8")
    real_replace = io.os.replace
    calls = {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("locked")
        return real_replace(a, b)

    monkeypatch.setattr(io.os, "replace", flaky)
    monkeypatch.setattr(io, "REPLACE_DELAY_S", 0)
    io.atomic_replace(src, dst)
    assert dst.read_text(encoding="utf-8") == "new"
    assert calls["n"] == 3


def test_atomic_replace_gives_up(tmp_path, monkeypatch):
    def locked(a, b):
        raise PermissionError("locked")

    monkeypatch.setattr(io.os, "replace", locked)
    monkeypatch.setattr(io, "REPLACE_DELAY_S", 0)
    with pytest.raises(PermissionError):
        io.atomic_replace(tmp_path / "a", tmp_path / "b")


def test_stable_sort_and_counts_sorted():
    df = pd.DataFrame({"k": [2, 1, 2, 1], "v": ["d", "c", "b", "a"]})
    assert stable_sort(df, "k")["v"].tolist() == ["c", "a", "d", "b"]
    counts = counts_sorted(pd.Series(["x", "y", "y", "z", "x"]))
    assert counts["value"].tolist() == ["x", "y", "z"]
    assert counts["count"].tolist() == [2, 2, 1]
