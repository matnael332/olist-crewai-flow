"""What each optional data group is worth: retrain without it and measure the drop.

    uv run synth --data-dir data
    uv run python -m pulse_specs.workforce_saas.ablation --data-dir data

Runs the deterministic pipeline (analyst, features, modeling) once with every group and once
without each optional group, on the same raw data, and writes ``group_value.json`` next to this
file. ``make_spec`` reads it to tell a customer what a missing group costs. The numbers come from
our synthetic reference data, so the contract quotes them as such: there is no honest way to
measure, on a customer's own data, the value of data the customer does not have.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from pulse_core.io import read_json, write_json_lf
from pulse_core.stages import run_analyst_stage, run_feature_stage, run_modeling_stage
from pulse_specs.workforce_saas import columns as wf

GROUP_VALUE = Path(__file__).with_name("group_value.json")
METRICS = ("pr_auc", "roc_auc", "precision_at_top10pct")


def _test_metrics(raw: Path, lookups: Path, out: Path, groups: tuple[str, ...]) -> dict:
    from pulse_specs.workforce_saas.spec import make_spec

    manifest = read_json(raw / "manifest.json")
    spec = make_spec(manifest["generator"]["snapshot_date"], groups=groups)
    run_analyst_stage(spec, raw, lookups, out)
    run_feature_stage(spec, out)
    summary = run_modeling_stage(spec, out)
    return {"model": summary["selected_model"], **{k: summary["test"][k] for k in METRICS}}


def measure(data_dir: Path) -> dict:
    raw, lookups = data_dir / "raw", data_dir / "lookups"
    optional = [g for g in wf.ALL_GROUPS if g not in wf.CORE_GROUPS]
    with tempfile.TemporaryDirectory() as tmp:
        full = _test_metrics(raw, lookups, Path(tmp) / "all", wf.ALL_GROUPS)
        print(f"all groups: {full}")
        without = {}
        for group in optional:
            groups = tuple(g for g in wf.ALL_GROUPS if g != group)
            metrics = _test_metrics(raw, lookups, Path(tmp) / group, groups)
            without[group] = {
                **metrics,
                **{f"{k}_drop": round(full[k] - metrics[k], 4) for k in METRICS},
            }
            print(f"without {group}: {without[group]}")
    return {
        "reference_data": read_json(raw / "manifest.json")["source"],
        "metric_note": (
            "test-window metrics of the selected model; *_drop = all groups minus without the "
            "group (positive: the group helps)"
        ),
        "all_groups": full,
        "without": without,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ablation", description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data", help="synthetic data from `uv run synth`")
    parser.add_argument("--out", default=str(GROUP_VALUE))
    args = parser.parse_args(argv)
    write_json_lf(Path(args.out), measure(Path(args.data_dir)))
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
