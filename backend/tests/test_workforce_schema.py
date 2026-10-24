"""The workforce_saas schema is internally consistent and fits the DatasetSpec interface."""

import json
from pathlib import Path

from pulse_core.columns import Column
from pulse_core.spec import DatasetSpec
from pulse_specs.workforce_saas import columns as wf

SCHEMA = Path(__file__).resolve().parents[1] / "pulse_specs" / "workforce_saas" / "raw_schema.json"
KINDS = {"string", "datetime", "int", "float", "bool"}
GROUPS = {"1 ", "2 ", "3 ", "4 ", "5 ", "6 ", "7 ", "8 ", "9 ", "10 "}


def test_raw_schema_covers_the_ten_groups_with_valid_tables():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert schema["datetime_format"] == "%Y-%m-%d %H:%M:%S" and schema["renames"] == {}
    tables = schema["tables"]
    assert len(tables) == 13
    files = [t["file"] for t in tables.values()]
    assert len(set(files)) == len(files)
    groups = {t["group"].split(" ")[0] + " " for t in tables.values()}
    assert groups == GROUPS
    for name, table in tables.items():
        assert table["file"] == f"{name}.csv"
        assert set(table["columns"].values()) <= KINDS, name
        assert set(table["primary_key"]) <= set(table["columns"]), name
        assert "account_id" in table["columns"], name


def test_clean_catalogue_is_consistent():
    names = [c.name for c in wf.COLUMNS]
    assert len(set(names)) == len(names)
    by_name = {c.name: c for c in wf.COLUMNS}
    assert by_name[wf.ID_COLUMN].dtype == "string"
    assert by_name[wf.ID_COLUMN].available_at == "identifier"
    assert by_name[wf.TIME_COLUMN].dtype == "datetime"
    assert by_name[wf.TIME_COLUMN].available_at == wf.DECISION_STAGE
    assert by_name[wf.TARGET].dtype == "bool" and by_name[wf.TARGET].available_at == "label"
    stages = {c.available_at for c in wf.COLUMNS}
    assert stages == {"identifier", wf.DECISION_STAGE, "post_decision", "label"}
    later = {c.name for c in wf.COLUMNS if c.available_at in ("post_decision", "label")}
    assert later == set(wf.FORBIDDEN)
    for declared in (wf.BOUNDS, wf.PATTERNS, wf.ALLOWED_VALUES):
        assert set(declared) <= set(names), set(declared) - set(names)
    for constraint in wf.CONSTRAINTS:
        for column in constraint.get("columns", []):
            assert column in names
    for name, allowed in wf.ALLOWED_VALUES.items():
        assert by_name[name].dtype == "string" and len(allowed) == len(set(allowed))
    for name, bounds in wf.BOUNDS.items():
        assert by_name[name].dtype in ("int", "float"), name
        assert set(bounds) <= {"min", "max"}
    assert all(isinstance(c, Column) for c in wf.COLUMNS)


def test_catalogue_fits_the_dataset_spec_interface():
    def not_yet(*args, **kwargs):
        raise NotImplementedError("step 4")

    spec = DatasetSpec(
        name="workforce_saas",
        source="synthetic",
        license="none",
        grain=wf.GRAIN,
        snapshot_date="2025-09-01",
        id_column=wf.ID_COLUMN,
        time_column=wf.TIME_COLUMN,
        decision_stage=wf.DECISION_STAGE,
        entity=wf.ENTITY,
        decision_time="the last second of the month",
        target=wf.TARGET,
        target_definition=f"cancelled within {wf.HORIZON_DAYS} days of the month end",
        target_label="churned",
        splits={"train": ("2023-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-06-30")},
        columns=wf.COLUMNS,
        clean=not_yet,
        patterns=wf.PATTERNS,
        bounds=wf.BOUNDS,
        constraints=wf.CONSTRAINTS,
        allowed_values=wf.ALLOWED_VALUES,
        feature_definitions=(),
        required_features=(),
        forbidden_features=wf.FORBIDDEN,
        derive_features=not_yet,
        eda_summary=not_yet,
    )
    assert spec.dtypes[wf.TARGET] == "bool" and spec.model_features == []
