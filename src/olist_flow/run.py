"""Command line entry point for the Olist Flow.

    uv run kickoff                                   # LLM mode when GEMINI_API_KEY is set
    uv run python -m olist_flow.run --no-llm         # deterministic mode
    uv run python -m olist_flow.run --no-llm --data-dir tests/fixtures/olist_mini --no-promote \
        --inject-fault missing_feature               # graceful-failure demo
    uv run python -m olist_flow.run --from-run <id> --start-at modeling --no-promote
    uv run python -m olist_flow.run --no-llm --no-promote --compare-to artifacts
    uv run python -m olist_flow.run verify-repro --data-dir tests/fixtures/olist_mini

Exit codes: 0 success, 1 graceful failure (see runs/<id>/run_status.json), 2 unexpected error.
"""

import argparse
import os
import shutil
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import olist_flow  # noqa: F401  (sets runtime env defaults before crewai is imported)
from olist_core.compare import compare_dirs, format_table
from olist_core.faults import FAULTS
from olist_core.io import write_text_lf
from olist_core.logging_setup import Redactor, RunLog, get_logger
from olist_core.manifest import environment_check
from olist_flow.llm import has_gemini_key, resolve_model

REPO_ROOT = Path(__file__).resolve().parents[2]
EXIT_CODES = {"success": 0, "failed": 1, "promotion_failed": 1}
log = get_logger("run")


def new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="olist_flow.run", description="Run the Olist CrewAI Flow")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "verify-repro"])
    parser.add_argument(
        "--no-llm", action="store_true", help="deterministic mode (no Gemini calls)"
    )
    parser.add_argument("--mode", choices=["llm", "deterministic"])
    parser.add_argument("--data-dir", default="data", help="folder with raw/ and lookups/")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inject-fault", choices=sorted(FAULTS), default="")
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--model", help="Gemini model id, e.g. gemini-3.8-flash")
    parser.add_argument("--from-run", help="reuse the artifacts of a previous run")
    parser.add_argument("--start-at", choices=["analyst", "features", "modeling"], default="")
    parser.add_argument("--compare-to", help="compare this run's artifacts with a folder")
    return parser


def resolve_mode(args: argparse.Namespace) -> str:
    if args.no_llm:
        return "deterministic"
    if args.mode:
        return args.mode
    env_mode = os.getenv("PIPELINE_MODE", "").strip().lower()
    if env_mode in ("llm", "deterministic"):
        return env_mode
    return "llm" if has_gemini_key() else "deterministic"


def _configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def run(args: argparse.Namespace) -> int:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    _configure_console()
    data_dir = Path(args.data_dir).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    run_id = args.run_id or new_run_id()
    run_dir = runs_dir / run_id
    if run_dir.exists():
        print(f"run directory already exists: {run_dir}", file=sys.stderr)
        return 2
    (run_dir / "artifacts").mkdir(parents=True)
    if args.from_run:
        source = runs_dir / args.from_run / "artifacts"
        if not source.exists():
            print(f"--from-run {args.from_run}: {source} not found", file=sys.stderr)
            return 2
        shutil.copytree(source, run_dir / "artifacts", dirs_exist_ok=True)
    mode = resolve_mode(args)
    inputs = {
        "run_id": run_id,
        "run_dir": run_dir.as_posix(),
        "mode": mode,
        "inject_fault": args.inject_fault,
        "seed": args.seed,
        "raw_dir": (data_dir / "raw").as_posix(),
        "lookups_dir": (data_dir / "lookups").as_posix(),
        "artifacts_dir": Path(args.artifacts_dir).resolve().as_posix(),
        "promote": not args.no_promote,
        "llm_model": resolve_model(args.model) if mode == "llm" else "",
        "start_at": args.start_at if args.from_run else "",
        "from_run": args.from_run or "",
    }
    status = "error"
    with RunLog(run_dir / "pipeline.log"):
        env = environment_check(REPO_ROOT)
        if env["environment_mismatch"]:
            log.warning(
                "ENV running outside the project venv (%s); use `uv run` for reproducible results",
                env["python_prefix"],
            )
        try:
            from olist_flow.flow import OlistFlow

            flow = OlistFlow()
            flow.kickoff(inputs=inputs)
            status = flow.state.status
        except Exception as exc:  # noqa: BLE001 - last line of defence: exit 2, no console traceback
            import traceback

            write_text_lf(run_dir / "traceback.txt", Redactor()(traceback.format_exc()))
            log.error(
                "RUN %s END status=error %s: %s (see traceback.txt)",
                run_id,
                type(exc).__name__,
                exc,
            )
        finally:
            if "crewai" in sys.modules:
                from crewai.events import crewai_event_bus

                crewai_event_bus.flush(timeout=10)
    code = EXIT_CODES.get(status, 2)
    if args.compare_to and status == "success":
        print(format_table(compare_dirs(run_dir / "artifacts", Path(args.compare_to))))
    print(f"run {run_id}: {status} (exit {code}); details in {run_dir.as_posix()}")
    return code


def verify_repro(args: argparse.Namespace) -> int:
    """Run the deterministic pipeline twice in fresh interpreters with different hash seeds."""
    base = Path(args.runs_dir).resolve() / f"verify-repro-{new_run_id()}"
    for hash_seed, run_id in (("0", "a"), ("12345", "b")):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONUTF8": "1"}
        cmd = [
            sys.executable,
            "-m",
            "olist_flow.run",
            "--no-llm",
            "--no-promote",
            "--data-dir",
            args.data_dir,
            "--runs-dir",
            str(base),
            "--run-id",
            run_id,
        ]
        done = subprocess.run(
            cmd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if done.returncode != 0:
            print(f"run {run_id} (PYTHONHASHSEED={hash_seed}) failed with exit {done.returncode}")
            print(done.stdout[-2000:])
            return 1
    result = compare_dirs(base / "a" / "artifacts", base / "b" / "artifacts")
    print(format_table(result))
    return 0 if result["deterministic_identical"] else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify-repro":
        return verify_repro(args)
    return run(args)


def kickoff() -> None:
    sys.exit(main())


def plot() -> None:
    """Export the Flow diagram (HTML) and structure (JSON) into docs/."""
    import json

    from crewai.flow import flow_structure

    from olist_flow.flow import OlistFlow

    out = REPO_ROOT / "docs" / "flow_plot"
    out.mkdir(parents=True, exist_ok=True)
    html = Path(OlistFlow().plot("olist_flow.html", show=False))
    for file in sorted(html.parent.glob("olist_flow*")):
        shutil.copyfile(file, out / file.name)
    write_text_lf(
        REPO_ROOT / "docs" / "flow_structure.json",
        json.dumps(flow_structure(OlistFlow), indent=2, sort_keys=True, default=str) + "\n",
    )
    print(f"flow plot written to {out.as_posix()}")


if __name__ == "__main__":
    sys.exit(main())
