"""A tiny synthetic DatasetSpec that proves the generic pipeline flows end to end.

Not the product's synthetic data: just enough rows, columns, rules and windows for every gate,
fault and renderer to have something real to work on. Everything is seeded and stable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from pulse_core.columns import DATETIME_FORMAT, Column  # noqa: E402
from pulse_core.spec import DatasetSpec  # noqa: E402

ROWS = 300
DECISION = "t0"
RAW_SCHEMA = {
    "schema_version": 1,
    "datetime_format": DATETIME_FORMAT,
    "renames": {},
    "tables": {
        "events": {
            "file": "events.csv",
            "primary_key": ["row_id"],
            "columns": {
                "row_id": "string",
                "ts": "datetime",
                "x_num": "float",
                "x_cat": "string",
                "x_flag": "bool",
                "post_val": "float",
                "y": "bool",
            },
        }
    },
}


def make_events(rows: int = ROWS) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    ts = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.sort(rng.randint(0, 366, rows)), "D")
    ts = ts + pd.to_timedelta(rng.randint(0, 24, rows), "h")
    x_num = np.round(rng.normal(0, 1.5, rows), 4)
    x_cat = rng.choice(["a", "b", "c"], rows)
    x_flag = rng.rand(rows) < 0.4
    logit = 0.9 * x_num + 1.2 * (x_cat == "a") - 0.6 * x_flag + rng.normal(0, 0.8, rows) - 0.3
    post_val = np.round(1 / (1 + np.exp(-logit)), 4)
    return pd.DataFrame(
        {
            "row_id": [f"r{i:04d}" for i in range(rows)],
            "ts": ts.strftime(DATETIME_FORMAT),
            "x_num": x_num,
            "x_cat": x_cat,
            "x_flag": x_flag,
            "post_val": post_val,
            "y": post_val > 0.5,
        }
    )


def write_raw(raw_dir: Path) -> Path:
    """Write events.csv, raw_schema.json and manifest.json into raw_dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    events = raw_dir / "events.csv"
    make_events().to_csv(events, index=False, lineterminator="\n", encoding="utf-8")
    (raw_dir / "raw_schema.json").write_text(json.dumps(RAW_SCHEMA, indent=2) + "\n", "utf-8")
    digest = hashlib.sha256(events.read_bytes()).hexdigest()
    manifest = {
        "source": "toy events (synthetic, seeded)",
        "files": {"events.csv": {"bytes": events.stat().st_size, "sha256": digest}},
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", "utf-8")
    return raw_dir


def clean(tables: dict[str, pd.DataFrame], lookups_dir: Path) -> tuple[pd.DataFrame, dict]:
    df = tables["events"].copy()
    rows_in = len(df)
    kept = df.dropna(subset=["x_num"])
    rules = [
        {
            "rule": "R1_drop_missing_x_num",
            "description": "rows without x_num cannot be scored",
            "rows_before": int(rows_in),
            "rows_after": int(len(kept)),
            "rows_removed": int(rows_in - len(kept)),
            "detail": "x_num is a required feature",
        }
    ]
    kept = kept.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    kept["ts"] = kept["ts"].dt.strftime(DATETIME_FORMAT)
    report = {
        "rows_in": int(rows_in),
        "rows_out": int(len(kept)),
        "rules": rules,
        "definitions": {"y": "post_val > 0.5 (observed after t0)"},
    }
    return kept[[c.name for c in COLUMNS]], report


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x_num": df["x_num"].astype("float64"),
            "x_cat": df["x_cat"].astype(object),
            "x_flag": df["x_flag"].astype(bool).astype("int64"),
            "x_num_sq": (df["x_num"].astype("float64") ** 2).round(6),
        },
        index=df.index,
    )


def eda_summary(tables, clean_df: pd.DataFrame, report: dict) -> dict:
    rate = float(clean_df["y"].astype(bool).mean())
    by_cat = clean_df.groupby("x_cat")["y"].mean().round(4)
    return {
        "positive_rate": rate,
        "key_numbers": [
            {"label": "rows", "value": int(len(clean_df))},
            {"label": "positive rate", "value": rate},
            {"label": "mean x_num", "value": float(clean_df["x_num"].mean())},
        ],
        "positive_rate_by_x_cat": {str(k): float(v) for k, v in sorted(by_cat.items())},
    }


def eda_figures(summary: dict, clean_df: pd.DataFrame, out: Path) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(4, 3))
    ax = fig.subplots()
    counts = clean_df["x_cat"].value_counts().sort_index()
    ax.bar(counts.index, counts.to_numpy(), color="#2563eb")
    ax.set_title("rows per x_cat")
    fig.tight_layout()
    fig.savefig(out / "x_cat_counts.png", dpi=80, metadata={"Software": None})
    return ["x_cat_counts.png"]


COLUMNS = (
    Column("row_id", "string", "identifier", False, "row identifier"),
    Column("ts", "datetime", DECISION, False, "event time (decision time)"),
    Column("x_num", "float", DECISION, False, "a numeric signal"),
    Column("x_cat", "string", DECISION, False, "a category"),
    Column("x_flag", "bool", DECISION, False, "a flag"),
    Column("post_val", "float", "post_decision", False, "observed after t0"),
    Column("y", "bool", "label", False, "post_val > 0.5"),
)

SPEC = DatasetSpec(
    name="toy_events",
    source="synthetic events generated by tests/toy_spec.py",
    license="none (synthetic)",
    grain="one row per event",
    snapshot_date="2021-01-01",
    id_column="row_id",
    time_column="ts",
    decision_stage=DECISION,
    entity="event",
    decision_time="t0 (the event time)",
    target="y",
    target_definition="post_val > 0.5, observed after t0",
    target_label="positive",
    splits={
        "train": ("2020-01-01", "2020-08-31"),
        "test": ("2020-09-01", "2020-11-30"),
        "late_buffer": ("2020-12-01", "2020-12-31"),
    },
    columns=COLUMNS,
    clean=clean,
    patterns={"row_id": r"^r[0-9]{4}$"},
    bounds={"x_num": {"min": -10, "max": 10}},
    constraints=(
        {"id": "C01", "type": "unique", "columns": ["row_id"], "description": "one row per event"},
        {"id": "C02", "type": "query", "query": "y == (post_val > 0.5)", "description": "target"},
        {
            "id": "C03",
            "type": "date_range",
            "column": "ts",
            "min": "2020-01-01 00:00:00",
            "max_exclusive": "2021-01-01 00:00:00",
            "description": "event window",
        },
    ),
    allowed_values={"x_cat": ["a", "b", "c"]},
    population={"events": "every synthetic event", "excluded": "events without x_num"},
    privacy={"note": "synthetic data; no personal information"},
    feature_definitions=(
        {
            "name": "x_num",
            "kind": "numeric",
            "derived_from": ["x_num"],
            "nullable": False,
            "description": "x_num as is",
        },
        {
            "name": "x_cat",
            "kind": "categorical",
            "derived_from": ["x_cat"],
            "nullable": False,
            "description": "x_cat as is",
        },
        {
            "name": "x_flag",
            "kind": "numeric",
            "derived_from": ["x_flag"],
            "nullable": False,
            "description": "flag as 0/1",
        },
        {
            "name": "x_num_sq",
            "kind": "numeric",
            "derived_from": ["x_num"],
            "nullable": False,
            "description": "x_num squared",
        },
    ),
    required_features=(
        {"name": "x_num", "derived_from": ["x_num"]},
        {"name": "x_cat", "derived_from": ["x_cat"]},
    ),
    forbidden_features=("post_val", "y"),
    derive_features=derive_features,
    eda_summary=eda_summary,
    eda_figures=eda_figures,
    model_card_items={"ethical_considerations": ["Toy data carries no personal information."]},
)
