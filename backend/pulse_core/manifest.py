"""run_manifest.json: everything needed to reconstruct or audit a run."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pulse_core.io import read_json, sha256_file, write_json_lf

PACKAGES = (
    "crewai",
    "google-genai",
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "joblib",
    "threadpoolctl",
    "streamlit",
    "flask",
)


def _git(*args: str, cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def blas_name() -> str | None:
    try:
        import numpy as np

        return np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception:  # noqa: BLE001 - informational only
        return None


def environment_check(repo_root: Path) -> dict:
    """Warn when the interpreter is not a project venv (.venv or .venv-<platform>)."""
    expected = (repo_root / ".venv").resolve()
    actual = Path(sys.prefix).resolve()
    in_project = actual.parent == repo_root.resolve() and actual.name.startswith(".venv")
    return {
        "python_prefix": actual.as_posix(),
        "expected_prefix": expected.as_posix(),
        "environment_mismatch": not in_project,
    }


def hash_tree(folder: Path) -> dict[str, str]:
    if not folder.exists():
        return {}
    return {
        p.relative_to(folder).as_posix(): sha256_file(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def build_manifest(
    state: dict, run_dir: Path, artifacts_dir: Path, repo_root: Path, raw_dir: Path
) -> dict:
    raw_manifest = raw_dir / "manifest.json"
    inputs = read_json(raw_manifest)["files"] if raw_manifest.exists() else {}
    return {
        "run": {
            k: state.get(k)
            for k in (
                "run_id",
                "mode",
                "inject_fault",
                "seed",
                "llm_model",
                "status",
                "stage",
                "start_at",
                "from_run",
                "spec_ref",
                "spec_params",
            )
        },
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "errors": state.get("errors", []),
        "warnings": state.get("warnings", []),
        "narrative_sources": state.get("narrative_sources", {}),
        "token_usage": state.get("token_usage", {}),
        "environment": {
            **environment_check(repo_root),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "blas": blas_name(),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "packages": package_versions(),
        },
        "git": {
            "sha": _git("rev-parse", "HEAD", cwd=repo_root),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root),
            "dirty": bool(_git("status", "--porcelain", cwd=repo_root)),
        },
        "inputs": {name: entry.get("sha256") for name, entry in sorted(inputs.items())},
        "artifacts": hash_tree(artifacts_dir),
        "run_dir": run_dir.as_posix(),
    }


def write_manifest(path: Path, manifest: dict) -> None:
    write_json_lf(path, manifest, ndigits=None)
