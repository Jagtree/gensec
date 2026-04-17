#!/usr/bin/env python3
"""
CLI USAGE GUIDE — setup_licensing.py
====================================

This file documents practical command-line usage examples for the
setup_licensing.py tool.

------------------------------------------------------------------
🚀 BASIC USAGE
------------------------------------------------------------------

Standard setup:
    uv run setup_licensing.py GenSec --headers

What it does:
- Downloads LICENSE
- Applies headers to src/ and test/
- Verifies required files

------------------------------------------------------------------
🔍 SAFE MODE (HIGHLY RECOMMENDED)
------------------------------------------------------------------

Dry-run (no file modification):
    uv run setup_licensing.py GenSec --headers --dry-run

------------------------------------------------------------------
🧪 DEBUG MODES
------------------------------------------------------------------

Verbose mode:
    uv run setup_licensing.py GenSec --headers --verbose

Dry-run + verbose (best debugging combo):
    uv run setup_licensing.py GenSec --headers --dry-run --verbose

------------------------------------------------------------------
📁 DIRECTORY CONTROL
------------------------------------------------------------------

Default (src + test):
    uv run setup_licensing.py --headers

Only src:
    uv run setup_licensing.py --headers --dirs src

src + examples:
    uv run setup_licensing.py --headers --dirs src examples

Entire repository:
    uv run setup_licensing.py --headers --dirs .

------------------------------------------------------------------
🧱 REAL WORKFLOWS
------------------------------------------------------------------

1. First-time integration:
    uv run setup_licensing.py GenSec --headers --dry-run --verbose
    uv run setup_licensing.py GenSec --headers

2. After adding new files:
    uv run setup_licensing.py --headers

3. Rename project:
    uv run setup_licensing.py NewName --headers

4. CI check (no modification):
    uv run setup_licensing.py --dry-run

------------------------------------------------------------------
⚠️ EDGE CASES
------------------------------------------------------------------

Only download LICENSE:
    uv run setup_licensing.py

If --headers is missing:
    ⚠️ Headers not applied (use --headers)

------------------------------------------------------------------
🧠 BEST PRACTICE WORKFLOW
------------------------------------------------------------------

    uv run setup_licensing.py --headers --dry-run --verbose
    uv run setup_licensing.py --headers
    git add .
    git commit -m "chore(license): apply AGPL headers"

------------------------------------------------------------------
🔥 OPTIONAL SHELL ALIAS
------------------------------------------------------------------

    alias lic="uv run setup_licensing.py"

Then:
    lic --headers
    lic --headers --dry-run

------------------------------------------------------------------
🏁 TL;DR
------------------------------------------------------------------

Most used commands:

    uv run setup_licensing.py --headers
    uv run setup_licensing.py --headers --dry-run
    uv run setup_licensing.py --headers --verbose

------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import os
import textwrap
import urllib.request
from pathlib import Path

AGPL_URL = "https://www.gnu.org/licenses/agpl-3.0.txt"

REQUIRED_FILES = [
    "LICENSE",
    "NOTICE",
    "CLA.md",
    "CONTRIBUTING.md",
    "COMMERCIAL_LICENSE.md",
    os.path.join(".github", "pull_request_template.md"),
    os.path.join(".github", "workflows", "cla.yml"),
]

# Header markers (robusti)
HEADER_START = "# ---------------------------------------------------------------------------"
HEADER_KEY = "GNU Affero General Public License"

# Directory da escludere SEMPRE
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
}

LICENSE_HEADER = textwrap.dedent("""\
# ---------------------------------------------------------------------------
# {project} — Copyright (c) 2026 Andrea Albero
#
# This file is part of {project}.
#
# {project} is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# {project} is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public
# License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with {project}. If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------

""")


# ---------------------------------------------------------------------------
# CORE UTILS
# ---------------------------------------------------------------------------

def find_repo_root(start: Path) -> Path:
    """Find repository root by looking for .git or pyproject.toml."""
    current = start.resolve()

    while current != current.parent:
        if (current / ".git").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent

    return start.resolve()


def is_excluded(path: Path) -> bool:
    """Check if path is inside excluded directories."""
    return any(part in EXCLUDED_DIRS for part in path.parts)


def collect_python_files(repo_root: Path, dirs: list[str]) -> list[Path]:
    """Collect .py files from given directories."""
    search_dirs = [repo_root / d for d in dirs if (repo_root / d).exists()]

    if not search_dirs:
        search_dirs = [repo_root]

    print("==> Scanning directories:")
    for d in search_dirs:
        print(f"   - {d}")

    files = {
        p for d in search_dirs
        for p in d.rglob("*.py")
        if not is_excluded(p)
    }

    return sorted(files)


def download_license(repo_root: Path, dry: bool) -> None:
    dest = repo_root / "LICENSE"
    print("==> Downloading AGPL-3.0")

    if dry:
        print(f"[DRY] Would download to {dest}")
        return

    try:
        urllib.request.urlretrieve(AGPL_URL, dest)
        print("    LICENSE downloaded")
    except Exception as exc:
        print(f"    [ERROR] {exc}")


def process_file(py: Path, header: str, repo_root: Path, dry: bool, verbose: bool) -> str:
    """Process a single file and return status."""
    try:
        content = py.read_text(encoding="utf-8")
    except Exception:
        return "error"

    if verbose:
        print(f"CHECK: {py}")

    # Caso 1: header già corretto
    if content.startswith(HEADER_START) and HEADER_KEY in content[:500]:
        return "skipped"

    # Caso 2: header parziale → sostituisci
    if HEADER_KEY in content[:1000]:
        if not dry:
            parts = content.split(HEADER_START)
            if len(parts) > 1:
                content = HEADER_START + parts[-1]
            py.write_text(header + content, encoding="utf-8")

        print(f"    ~ updated {py.relative_to(repo_root)}")
        return "updated"

    # Caso 3: aggiungi header
    if not dry:
        py.write_text(header + content, encoding="utf-8")

    print(f"    + added {py.relative_to(repo_root)}")
    return "added"


def prepend_headers(
    repo_root: Path,
    project: str,
    dirs: list[str],
    dry: bool,
    verbose: bool,
) -> None:
    header = LICENSE_HEADER.format(project=project)

    py_files = collect_python_files(repo_root, dirs)

    added = updated = skipped = errors = 0

    for py in py_files:
        result = process_file(py, header, repo_root, dry, verbose)

        if result == "added":
            added += 1
        elif result == "updated":
            updated += 1
        elif result == "skipped":
            skipped += 1
        else:
            errors += 1

    print("\nSummary:")
    print(f"    Added:   {added}")
    print(f"    Updated: {updated}")
    print(f"    Skipped: {skipped}")
    if errors:
        print(f"    Errors:  {errors}")


def verify(repo_root: Path) -> bool:
    print("==> Verifying licensing files")
    ok = True

    for rel in REQUIRED_FILES:
        path = repo_root / rel
        if path.exists():
            print(f"    [OK] {rel}")
        else:
            print(f"    [MISSING] {rel}")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AGPL licensing setup tool")

    parser.add_argument("project", nargs="?", default="GenSec")

    parser.add_argument(
        "--headers",
        action="store_true",
        help="Apply license headers",
    )

    parser.add_argument(
        "--dirs",
        nargs="*",
        default=["src", "test"],
        help="Directories to scan (default: src test)",
    )

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).parent)

    print(f"==> Repo root detected: {repo_root}")

    download_license(repo_root, args.dry_run)

    if args.headers:
        prepend_headers(
            repo_root,
            args.project,
            args.dirs,
            args.dry_run,
            args.verbose,
        )
    else:
        print("⚠️  Headers not applied (use --headers)")

    ok = verify(repo_root)

    print("\nDone.")
    if not ok:
        print("⚠️  Some files are missing.")


if __name__ == "__main__":
    main()



#uv run setup_licensing.py GenSec --headers
#uv run setup_licensing.py GenSec --headers --dry-run
#uv run setup_licensing.py GenSec --headers --verbose
#uv run setup_licensing.py GenSec --headers --dry-run --verbose
#uv run setup_licensing.py --headers --dirs src
#uv run setup_licensing.py --headers --dirs src examples
#uv run setup_licensing.py --headers --dirs .

