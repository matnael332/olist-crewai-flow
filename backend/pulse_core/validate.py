"""Run the three gates against an artifacts folder (for example the promoted artifacts/).

Usage: uv run python -m pulse_core.validate artifacts
"""

from __future__ import annotations

import sys
from pathlib import Path

from pulse_core.validators.artifacts_gate import validate_artifacts
from pulse_core.validators.contract_gate import validate_clean_data_against_contract
from pulse_core.validators.features_gate import validate_features


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    art = Path(args[0] if args else "artifacts")
    reports = [
        validate_clean_data_against_contract(art / "clean_data.csv", art / "dataset_contract.json"),
        validate_features(
            art / "features.csv", art / "feature_spec.json", art / "dataset_contract.json"
        ),
        validate_artifacts(art),
    ]
    for report in reports:
        print(f"{report.gate}: {report.summary}")
        for error in report.errors:
            print(f"  - {error}")
    return 0 if all(r.passed for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
