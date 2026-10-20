"""Repository hygiene gate (CI and pre-push).

Fails if a tracked file looks like it contains an API key, or if tracked files exceed the size
budget (GitHub warns at 50 MB per file; this project keeps every file under 45 MB and the whole
tree under 120 MB).

Usage: uv run python scripts/check_repo_hygiene.py [--max-file-mb 45] [--max-total-mb 120]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

KEY_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"GEMINI_API_KEY\s*=\s*['\"]?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{32,}"),
]
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".html",
    ".log",
    ".cfg",
    ".ini",
    ".env",
    ".example",
    ".lock",
}


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
    ).stdout.decode("utf-8")
    return [root / name for name in out.split("\0") if name]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-file-mb", type=float, default=45.0)
    parser.add_argument("--max-total-mb", type=float, default=120.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    problems: list[str] = []
    total = 0
    for path in tracked_files(root):
        if not path.exists():
            continue
        size = path.stat().st_size
        total += size
        rel = path.relative_to(root).as_posix()
        if size > args.max_file_mb * 1024 * 1024:
            problems.append(f"{rel}: {size / 1048576:.1f} MB exceeds {args.max_file_mb} MB")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name.startswith(".env"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in KEY_PATTERNS:
                if pattern.search(text):
                    problems.append(f"{rel}: matches secret pattern {pattern.pattern[:24]}...")
    if total > args.max_total_mb * 1024 * 1024:
        problems.append(f"tracked files total {total / 1048576:.1f} MB > {args.max_total_mb} MB")

    if problems:
        print("Repository hygiene check FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"Repository hygiene OK ({total / 1048576:.1f} MB tracked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
