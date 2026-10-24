"""pulse-ingest: a customer's zip -> a data folder the pipeline runs on.

    uv run pulse-ingest propose customer.zip --customer-dir C [--no-llm]
    uv run pulse-ingest approve --customer-dir C --ingestion ID [--mapping edited.json]
    uv run pulse-ingest apply   --customer-dir C --ingestion ID

``propose`` prints the mapping summary and what needs a look; nothing is applied before a person
approves it. ``apply`` prints the pipeline command for the new data folder.

Exit codes: 0 done, 1 the export or the mapping was refused (the message says why).
"""

from __future__ import annotations

import argparse
import importlib
import os
import shlex
import sys
from pathlib import Path

from pulse_ingest import session
from pulse_ingest.contract import IngestContract, PreflightError
from pulse_ingest.normalize import MappingError
from pulse_ingest.sources import UnsafeArchive

DEFAULT_CONTRACT = "pulse_specs.workforce_saas.ingest_contract:CONTRACT"


def load_contract(ref: str) -> IngestContract:
    module, _, attr = ref.partition(":")
    return getattr(importlib.import_module(module), attr)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(prog="pulse-ingest", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["propose", "approve", "apply"])
    parser.add_argument("zip", nargs="?", help="the customer's export (propose)")
    parser.add_argument("--customer-dir", required=True)
    parser.add_argument("--ingestion", help="the ingestion id printed by propose")
    parser.add_argument("--mapping", help="an edited mapping to approve")
    parser.add_argument("--no-llm", action="store_true", help="heuristic mapping only")
    parser.add_argument("--contract", default=DEFAULT_CONTRACT)
    args = parser.parse_args(argv)
    contract = load_contract(args.contract)
    customer = Path(args.customer_dir)
    try:
        if args.command == "propose":
            if not args.zip:
                parser.error("propose needs the zip file")
            use_llm = not args.no_llm and bool(os.getenv("GEMINI_API_KEY"))
            ingestion, mapping = session.propose(
                Path(args.zip), customer, contract, use_llm=use_llm
            )
            print("\n".join(mapping.summary()))
            print(f"\ningestion {ingestion}: review {customer / 'staging' / ingestion}")
            print(f"then: pulse-ingest approve --customer-dir {customer} --ingestion {ingestion}")
        elif args.command == "approve":
            mapping = session.approve(
                customer, args.ingestion, Path(args.mapping) if args.mapping else None
            )
            print("\n".join(mapping.summary()))
        else:
            result = session.apply(customer, args.ingestion, contract)
            print(session.describe(result))
            command = session.pipeline_command(result["data_dir"], contract)
            print("\nrun the pipeline:\n  " + shlex.join(command))
    except (UnsafeArchive, MappingError, PreflightError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
