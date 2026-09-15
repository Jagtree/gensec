#!/usr/bin/env python3
r"""
Validator: the archive-currency gate can fail, and fails by name.

Why this file exists
--------------------
:mod:`tools.archive_currency` answers the question none of the other
gates ask — *is the bundle inside the delivered archive at the commit
the working tree was on when the archive was rolled, and was that tree
clean?*  At the close of Phase 33 it was not: bundle and tarball were
written at 19:55, the last commit landed at 20:03, and the delivered
bundle was one commit short of the tree.  All four provenance checks
passed, because they compare the tarball with **its own** bundle and
the two were stale together (E33-4).

A gate that has never been shown to fail is not yet a gate.  This
validator drives ``tools/archive_currency.py --falsify``, which builds
a throwaway repository, rolls an archive from it, and then corrupts
that archive in each of the three ways an archive can stop describing
its tree — bundle behind, bundle ahead, tree dirty — requiring the gate
to reject each **by its own diagnosis**, plus a null control it must
accept.

The exit status is read from the process object, never from a pipe:
``cmd | grep ...; echo $?`` reports ``grep``'s status, which is the
project's own standing rule and the one E30-9 records as having been
broken in a command whose only purpose was to read an exit code.

Usage
-----
::

    python tests/run_phase34_archive_currency_falsification.py

Exits 0 when the falsification passes, 1 otherwise.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The tool under test, relative to the repository root.
TOOL = os.path.join("tools", "archive_currency.py")

#: The diagnoses the falsification must print, one per corruption.
#: Named here as well as inside the tool so that a corruption silently
#: renamed on one side is caught by the other: two homes would be a
#: defect, but this is not a second home for the *logic*, only an
#: independent statement of what the output must contain.
REQUIRED = ("null_control", "bundle_behind", "bundle_ahead",
            "tree_dirty", "FALSIFICATION PASS")


def main() -> int:
    r"""
    Run the falsification and check what it printed.

    Returns
    -------
    int
        Process exit status: ``0`` when the tool exits 0 **and** its
        output names every required case.  Both are required: a tool
        that exited 0 while silently skipping a case would pass on the
        status alone.
    """
    print("PHASE-34 ARCHIVE-CURRENCY FALSIFICATION")
    print("=" * 62)
    proc = subprocess.run([sys.executable, TOOL, "--falsify"],
                          cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    print(proc.stdout, end="")
    status = proc.returncode

    missing = [k for k in REQUIRED if k not in proc.stdout]
    print("-" * 62)
    print(f"  tool exit status (from the process): {status}")
    if missing:
        print(f"  FAIL  cases_absent: the falsification never reported "
              f"{missing}")
    if status != 0 or missing:
        print("VALIDATOR FAIL: the archive-currency gate has not been "
              "shown to reject what it must reject.")
        return 1
    print("VALIDATOR PASS: three distinct corruptions rejected, each "
          "by its own diagnosis, null control accepted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
# GENSEC-COMPLETE phase=34 utc=2026-09-06T10:33:16Z sha256=3309b4c3673a429795fe73ebb97ff2fa3d4d97ecde190e4eba984c769e89c430 body_lines=92
