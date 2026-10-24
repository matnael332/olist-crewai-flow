"""One ingestion, from an uploaded zip to a data folder the pipeline can run on.

Customer folder layout (``customer_dir``):

    uploads/<ingestion>/         the uploaded zip              (deleted by apply)
    staging/<ingestion>/         extracted files, tables of strings (parquet), profile.json,
                                 mapping.proposed.json         (deleted by apply)
    mappings/v<N>.json           every approved mapping, by version
    data/<ingestion>/raw/        the normalised raw layer (CSV + raw_schema.json + manifest.json)
    data/<ingestion>/lookups/    empty (the spec needs none)
    data/<ingestion>/spec_params.json, preflight.json
    entity_map.csv               pseudonym <-> the customer's account id and name
    secret.key                   the key of the employee, site and device hashes

After ``apply`` nothing of the original export is left but the mapped columns, pseudonymised.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from pulse_core.io import read_json, write_json_lf
from pulse_core.raw import write_raw_dir
from pulse_ingest import heuristics, llm
from pulse_ingest.contract import IngestContract
from pulse_ingest.mapping import Mapping
from pulse_ingest.normalize import normalize
from pulse_ingest.profile import fingerprint, profile_tables, to_json, union_groups
from pulse_ingest.pseudonymize import EntityMap, load_secret
from pulse_ingest.readers import read_files
from pulse_ingest.sources import ZipSource

RAW_FLOAT_FORMAT = "%.4f"


def new_ingestion_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def _staging(customer_dir: Path, ingestion: str) -> Path:
    return customer_dir / "staging" / ingestion


def latest_approved(customer_dir: Path) -> Mapping | None:
    versions = sorted((customer_dir / "mappings").glob("v*.json"), key=lambda p: int(p.stem[1:]))
    return Mapping.model_validate(read_json(versions[-1])) if versions else None


def propose(
    zip_path: Path,
    customer_dir: Path,
    contract: IngestContract,
    *,
    use_llm: bool = True,
    ingestion: str | None = None,
    llm_client=None,
) -> tuple[str, Mapping]:
    """Stage the export and write ``mapping.proposed.json``; returns the ingestion id."""
    ingestion = ingestion or new_ingestion_id()
    upload = customer_dir / "uploads" / ingestion
    staging = _staging(customer_dir, ingestion)
    upload.mkdir(parents=True)
    staging.mkdir(parents=True)
    shutil.copyfile(zip_path, upload / zip_path.name)
    files = ZipSource(upload / zip_path.name).fetch(staging)
    tables, read_info = read_files(files, staging / "files")
    groups = union_groups(tables)
    grouped = {
        name: pd.concat([tables[m] for m in members], ignore_index=True)
        for name, members in groups.items()
    }
    (staging / "tables").mkdir()
    index = {}
    for i, (name, df) in enumerate(grouped.items()):
        df.to_parquet(staging / "tables" / f"{i:03d}.parquet", index=False)
        index[name] = f"{i:03d}.parquet"
    shutil.rmtree(staging / "files")  # the extracted originals: the parquet copies suffice
    profile = profile_tables(grouped)
    structure = fingerprint(profile)
    write_json_lf(staging / "tables.json", index)
    write_json_lf(staging / "profile.json", {"read": read_info, "tables": to_json(profile)})

    previous = latest_approved(customer_dir)
    if previous and previous.fingerprint == structure:
        mapping = previous.model_copy(deep=True)
        mapping.review = [f"same structure as the approved mapping v{previous.version}"]
    else:
        mapping = heuristics.propose(profile, contract, structure)
        if use_llm:
            try:
                answer, tokens = llm.ask(profile, contract, client=llm_client)
                llm.merge(mapping, answer, profile, contract)
                mapping.review.append(f"LLM mapping used ({tokens})")
            except Exception as exc:  # noqa: BLE001 - the heuristic proposal stands alone
                mapping.review.append(f"LLM mapping unavailable: {type(exc).__name__}: {exc}")
        heuristics.finalize(mapping, profile, contract)
        heuristics.propose_values(mapping, contract, grouped)
        mapping.version = (previous.version + 1) if previous else 1
        mapping.status = "proposed"
    write_json_lf(staging / "mapping.proposed.json", mapping.model_dump(), ndigits=None)
    return ingestion, mapping


def approve(customer_dir: Path, ingestion: str, mapping_path: Path | None = None) -> Mapping:
    """Mark the proposed (or an edited) mapping approved, in the staging folder."""
    path = mapping_path or _staging(customer_dir, ingestion) / "mapping.proposed.json"
    mapping = Mapping.model_validate(read_json(path))
    mapping.status = "approved"
    write_json_lf(
        _staging(customer_dir, ingestion) / "mapping.proposed.json",
        mapping.model_dump(),
        ndigits=None,
    )
    return mapping


def apply(customer_dir: Path, ingestion: str, contract: IngestContract) -> dict:
    """Normalise, preflight and write data/<ingestion>/; then delete the export's originals."""
    staging = _staging(customer_dir, ingestion)
    mapping = Mapping.model_validate(read_json(staging / "mapping.proposed.json"))
    index = read_json(staging / "tables.json")
    tables = {name: pd.read_parquet(staging / "tables" / file) for name, file in index.items()}
    present = contract.groups_present(list(mapping.tables))
    kept = [t.name for t in contract.tables if t.group in present and t.name in mapping.tables]
    mapping.tables = {name: mapping.tables[name] for name in kept}
    entity_map = EntityMap(customer_dir / "entity_map.csv")
    built = normalize(
        mapping, contract, tables, entity_map, load_secret(customer_dir / "secret.key")
    )
    result = contract.preflight(built, present) if contract.preflight else {"spec_params": {}}
    data = customer_dir / "data" / ingestion
    write_raw_dir(
        built,
        contract.raw_schema(kept),
        data / "raw",
        {"source": f"customer export, ingestion {ingestion}", "mapping_version": mapping.version},
        float_format=RAW_FLOAT_FORMAT,
    )
    (data / "lookups").mkdir(parents=True, exist_ok=True)
    write_json_lf(data / "spec_params.json", result["spec_params"], ndigits=None)
    write_json_lf(data / "preflight.json", result.get("report", {}), ndigits=None)
    entity_map.save()
    previous = latest_approved(customer_dir)
    if not previous or previous.model_dump() != mapping.model_dump():
        (customer_dir / "mappings").mkdir(parents=True, exist_ok=True)
        write_json_lf(
            customer_dir / "mappings" / f"v{mapping.version}.json",
            mapping.model_dump(),
            ndigits=None,
        )
    shutil.rmtree(customer_dir / "uploads" / ingestion, ignore_errors=True)
    shutil.rmtree(staging, ignore_errors=True)
    return {"data_dir": data, **result}


def pipeline_command(data_dir: Path, contract: IngestContract) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "-m",
        "pulse_flow.run",
        "--spec",
        contract.spec_ref,
        "--spec-params",
        str(data_dir / "spec_params.json"),
        "--data-dir",
        str(data_dir),
    ]


def describe(result: dict) -> str:
    return json.dumps(result.get("report", {}), indent=2, ensure_ascii=False)
