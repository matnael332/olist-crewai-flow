"""The only place that builds what the LLM mapper sees: names and kinds, never values.

Built from the profile (which holds no cell values) and the contract. A test scans every payload
against the export's own values, so a change here that leaks data fails the build.
"""

from __future__ import annotations

from pulse_ingest.contract import IngestContract
from pulse_ingest.profile import SourceTable


def mapping_payload(profile: list[SourceTable], contract: IngestContract) -> dict:
    return {
        "targets": [
            {
                "table": t.name,
                "columns": [
                    {"name": c.name, "kind": c.kind, "required": c.required}
                    for c in t.columns
                    if not c.generate
                ],
            }
            for t in contract.tables
        ],
        "sources": [
            {
                "table": s.name,
                "columns": [{"name": c.name, "kind": c.kind} for c in s.columns],
            }
            for s in profile
        ],
    }
