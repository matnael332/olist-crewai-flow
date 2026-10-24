"""The workforce_saas schema is internally consistent and fits the DatasetSpec interface."""

import json
from pathlib import Path

from pulse_core.columns import Column
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


def test_real_spec_fits_the_catalogue():
    from pulse_specs.workforce_saas.spec import SPEC

    assert SPEC.dtypes[wf.TARGET] == "bool"
    assert SPEC.id_column == wf.ID_COLUMN and SPEC.time_column == wf.TIME_COLUMN
    ids = [c["id"] for c in wf.CONSTRAINTS]
    assert len(set(ids)) == len(ids)
    by_id = {c["id"]: c for c in wf.CONSTRAINTS}
    assert by_id["C02"]["where_not_null"] == ["days_to_cancel"]
    assert "C05" in by_id
    decision = {c.name for c in wf.COLUMNS if c.available_at == wf.DECISION_STAGE}
    names = [f["name"] for f in SPEC.feature_definitions]
    assert len(set(names)) == len(names) == len(SPEC.model_features)
    for feature in SPEC.feature_definitions:
        assert set(feature["derived_from"]) <= decision, feature["name"]
        assert feature["kind"] in ("numeric", "categorical")
        assert feature["name"] not in wf.FORBIDDEN
    assert {f["name"] for f in SPEC.required_features} <= set(names)
    assert set(SPEC.splits) == {"train", "horizon_gap", "test"}
