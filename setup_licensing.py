#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
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

HEADER_START = "# ---------------------------------------------------------------------------"
HEADER_KEY = "GNU Affero General Public License"

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
    """Find repo root by looking for .git or pyproject.toml."""
    current = start.resolve()

    while current != current.parent:
        if (current / ".git").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent

    return start.resolve()


def download_license(repo_root: Path, dry: bool) -> None:
    dest = repo_root / "LICENSE"
    print(f"==> Downloading AGPL-3.0")

    if dry:
        print(f"[DRY] Would download to {dest}")
        return

    try:
        urllib.request.urlretrieve(AGPL_URL, dest)
        print("    LICENSE downloaded")
    except Exception as exc:
        print(f"    [ERROR] {exc}")


def prepend_headers(repo_root: Path, project: str, dry: bool, verbose: bool) -> None:
    header = LICENSE_HEADER.format(project=project)

    src_dir = repo_root / "src"
    search_dir = src_dir if src_dir.exists() else repo_root

    print(f"==> Scanning: {search_dir}")

    py_files = list(search_dir.rglob("*.py"))

    added, updated, skipped = 0, 0, 0

    for py in py_files:
        content = py.read_text(encoding="utf-8")

        # DEBUG
        if verbose:
            print(f"CHECK: {py}")

        # Already has full header
        if content.startswith(HEADER_START) and HEADER_KEY in content[:500]:
            skipped += 1
            continue

        # Has partial header → replace
        if HEADER_KEY in content[:1000]:
            if not dry:
                # Remove existing header block
                parts = content.split(HEADER_START)
                if len(parts) > 1:
                    content = HEADER_START + parts[-1]

                py.write_text(header + content, encoding="utf-8")

            updated += 1
            print(f"    ~ updated {py.relative_to(repo_root)}")
            continue

        # Add new header
        if not dry:
            py.write_text(header + content, encoding="utf-8")

        added += 1
        print(f"    + added {py.relative_to(repo_root)}")

    print(f"\nSummary:")
    print(f"    Added:   {added}")
    print(f"    Updated: {updated}")
    print(f"    Skipped: {skipped}")


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
    parser = argparse.ArgumentParser(description="Setup AGPL licensing")

    parser.add_argument("project", nargs="?", default="GenSec")
    parser.add_argument("--headers", action="store_true")
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
            args.dry_run,
            args.verbose,
        )

    ok = verify(repo_root)

    print("\nDone.")
    if not ok:
        print("⚠️  Some files are missing.")


if __name__ == "__main__":
    main()