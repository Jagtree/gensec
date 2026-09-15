#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# GenSec — Copyright (c) 2026 Andrea Albero
#
# This file is part of GenSec.
#
# GenSec is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# GenSec is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public
# License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GenSec.  If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------
r"""
Falsify the Phase-38 patcher before its green is believed.

Why this file exists, and why it REPLACES its predecessor
---------------------------------------------------------
``--selfcheck`` exiting 0 is the strongest statement this project makes
about an archive: the tree on disk is the tree the phase delivered, byte
for byte.  A gate whose green has never been seen to go red is not a
gate.

This file replaces ``run_phase37_patcher_falsification.py`` rather than
sitting beside it: a patcher's selfcheck is pinned to the tree that
generated it, so the Phase-37 harness reports a false positive on its own
null control the moment Phase 38 moves a file the Phase-37 payload
carries (F-B14, E37-4).  A permanently red validator annuls the gate it
belongs to, which is why the replacement is mandatory and not a tidy-up.

Phase 38 is the first phase in which that replacement **travels**.  The
predecessor is named in the patcher's ``REMOVE`` block, so a tree rebuilt
from the payload alone no longer keeps it (``37_0`` section 10.5).  The
last two cases below are what stops that block from being a claim.

The corruptions, and why each one
---------------------------------
Chosen so that no two are detected by the same mechanism.

``payload_byte_flipped``
    One identifier renamed inside a carried source, **preserving its
    length**.  A checker comparing sizes sees nothing.
``carried_file_truncated``
    A carried file cut in half.  The blunt case, kept because a checker
    that missed it would be broken in a way the subtle cases might not
    reveal.
``line_ending_flipped``
    LF rewritten as CRLF on a file the payload declares as LF.  The tree
    is not uniform, and a patcher that normalised endings would rewrite
    files it was never asked to change.
``bootstrap_count_edited``
    A single digit changed in a carried document, in a place where the
    replacement has **the same length** as the original.  This is the
    corruption Phase 30 met twice and the reason ``--selfcheck``
    compares bytes and not sizes.
``removed_file_restored``
    A path the patcher declares removed, put back with the content it
    used to have.  This is the corruption the ``REMOVE`` block exists
    for: before Phase 38 the payload had no opinion about it at all, and
    a tree carrying it silently gained a nineteenth validator that was
    red by construction.
``removed_file_restored_empty``
    The same path put back **empty**.  It separates two mechanisms that
    would otherwise look alike: a removal is verified by asking whether
    the path is there, not by comparing its bytes against anything, so
    an empty file must be caught exactly as the full one is.
``patcher_digest_corrupted``
    The patcher's own declared digest altered.  A different mechanism
    entirely: the payload is verified on import, so a corrupted patcher
    must refuse to act rather than write bad bytes.

What the null control proves here, and did not before
-----------------------------------------------------
The null control leaves the tree exactly as delivered -- which, since
Phase 38, is a tree where every declared removal is **already absent**.
Its green is therefore also the idempotency statement the ``REMOVE``
block needs: a declared removal whose path is already gone is not an
error, it is the ordinary state, and ``--selfcheck`` must accept it.

Reading the output
------------------
Each corrupted case must exit non-zero **and** print a diagnosis naming
the file or the failure.  ``detected`` with no name would mean the
patcher knows something is wrong without being able to say what.

The ``edited`` column guards the harness itself: a mutation that
silently changed no byte would produce a green that proves nothing, and
it would look exactly like a passing null control.
"""
import ast
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHER = os.path.join("patchers",
                       "run_phase38_remove_block_and_annex.py")

#: The path the Phase-38 patcher declares removed.  Named once here and
#: read from the patcher in :func:`_declared_removals`, so that a
#: harness and a patcher that disagree are a failure rather than a
#: coincidence nobody checks.
RETIRED = "tests/run_phase37_patcher_falsification.py"

#: ``(name, path, mutation, what it simulates)``.  ``mutation`` receives
#: the current bytes of *path*, or ``b""`` when the path does not exist,
#: and returns the bytes to write.
CASES = [
    ("payload_byte_flipped", "tools/make_phase38_patcher.py",
     lambda b: b.replace(b"def _removal_state(", b"def _removal_stata(", 1),
     "a carried source edited without changing its length"),
    ("carried_file_truncated", "tools/selfcheck_inventory.py",
     lambda b: b[:len(b) // 2],
     "a carried file cut in half"),
    ("line_ending_flipped", "tools/selfcheck_inventory.py",
     lambda b: b.replace(b"\n", b"\r\n"),
     "LF rewritten as CRLF on a file declared as LF"),
    ("bootstrap_count_edited", "handoff/00_BOOTSTRAP.md",
     lambda b: b.replace(b"voci di carico", b"voci di carica", 1),
     "one same-length edit inside a carried document, the corruption\n"
     "      a size comparison cannot see"),
    ("removed_file_restored", RETIRED,
     lambda b: b"# restored\n" * 40,
     "a path declared removed, put back with content"),
    ("removed_file_restored_empty", RETIRED,
     lambda b: b"",
     "the same path put back empty: presence is the fact checked,\n"
     "      not the bytes"),
    ("patcher_digest_corrupted", PATCHER,
     lambda b: _flip_first_digest(b),
     "the patcher's own declared digest altered"),
]


def _flip_first_digest(raw):
    """
    Alter one hex character of the first declared digest.

    Located rather than pattern-matched on a file name: the first key
    of ``DIGEST`` is whatever ``FILES`` happens to start with, and a
    harness anchored on that name would stop corrupting anything the
    day the list is reordered -- and would then pass while testing
    nothing.

    Parameters
    ----------
    raw : bytes

    Returns
    -------
    bytes
    """
    i = raw.index(b"DIGEST: Dict[str, str] = {")
    j = raw.index(b'":\n        "', i) + len(b'":\n        "')
    flipped = b"1" if raw[j:j + 1] != b"1" else b"2"
    return raw[:j] + flipped + raw[j + 1:]


#: The tree left exactly as it is.  Without this row the table cannot
#: discriminate: a checker that reddens on everything would score seven
#: out of seven.  Since Phase 38 it carries a second statement -- see
#: the module docstring -- because the delivered tree is one in which
#: every declared removal is already absent.
NULL_CONTROL = ("null_control", "tools/selfcheck_inventory.py",
                lambda b: b, "nothing changed")


def _copy_tree(dest):
    """
    Copy the repository into *dest*, without the parts that are not it.

    ``.git``, the virtualenv and the build metadata are excluded: none
    is carried by the payload, and copying a virtualenv would make each
    case cost hundreds of megabytes.
    """
    def ignore(_dirpath, names):
        return [n for n in names
                if n in (".git", ".venv", "__pycache__", ".pytest_cache",
                         "htmlcov", "_build")
                or n.endswith(".egg-info")]
    shutil.copytree(ROOT, dest, ignore=ignore)


def _declared_removals():
    """
    Read the patcher's ``REMOVE`` block without importing its payload.

    Returns
    -------
    list of str

    Notes
    -----
    Read, not assumed.  A harness that named the retired path only in
    its own source would keep passing after the patcher stopped
    declaring it, and the green would then prove nothing about the
    mechanism under test.
    """
    src = open(os.path.join(ROOT, PATCHER), "rb").read().decode("utf-8")
    lines = src.splitlines()
    i = next(k for k, ln in enumerate(lines)
             if ln.startswith("REMOVE: List[str] = ["))
    body = []
    for ln in lines[i + 1:]:
        if ln.startswith("]"):
            break
        body.append(ln)
    return list(ast.literal_eval("[" + "\n".join(body) + "]"))


def _run_selfcheck(tree):
    """
    Run the patcher's ``--selfcheck`` in *tree*.

    Returns
    -------
    tuple
        ``(exit code, the diagnosis line)``.  The diagnosis preferred is
        the one naming a file; the summary line is the fallback, and a
        case that can only produce the summary is reported as such so
        that the reader sees the difference.
    """
    proc = subprocess.run(
        [sys.executable, PATCHER, "--selfcheck"], cwd=tree,
        capture_output=True, text=True, timeout=900,
    )
    out = proc.stdout + proc.stderr
    # Match the FAILURE tokens the patcher prints, not the words that
    # happen to appear in them.  A filter on "absent" also matched
    # "ok  <path>: absent as declared" -- the line that says a removal
    # is FINE -- so the two removal cases scored as detected while
    # quoting a green line as their diagnosis.  Caught by reading the
    # table rather than its verdict, which is the only way that class
    # of error is ever caught.
    named = [ln.strip() for ln in out.splitlines()
             if ln.strip().startswith(("BYTES", "REMOVE"))
             or "Error" in ln or "ERROR" in ln]
    if named:
        return proc.returncode, named[0][:86]
    tail = out.strip().splitlines()
    return proc.returncode, (tail[-1][:86] if tail else "")


def main():
    """
    Run every case in its own copy of the tree and report.

    Returns
    -------
    int
        0 if every corruption was detected **and** the null control was
        accepted; 1 otherwise.
    """
    declared = _declared_removals()
    if RETIRED not in declared:
        print(f"HARNESS FAIL: {RETIRED} is not in the patcher's REMOVE "
              f"block ({declared}); the two removal cases below would "
              f"test nothing.")
        return 1

    rows = []
    for name, rel, mutate, what in CASES + [NULL_CONTROL]:
        tmp = tempfile.mkdtemp(prefix="gensec_falsify_")
        tree = os.path.join(tmp, "t")
        try:
            _copy_tree(tree)
            target = os.path.join(tree, rel)
            existed = os.path.exists(target)
            raw = open(target, "rb").read() if existed else b""
            new = mutate(raw)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "wb") as fh:
                fh.write(new)
            # "edited" means the TREE changed, not that the bytes did.
            # Creating a path that was absent is a change even when the
            # content written is empty, and the empty-restore case is
            # exactly that: without this the harness would score its own
            # mutation as a no-op and read like a passing null control.
            edited = (new != raw) or (not existed)
            code, diag = _run_selfcheck(tree)
            rows.append((name, what, edited, code, diag))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 96)
    print("PHASE-38 PATCHER FALSIFICATION")
    print("=" * 96)
    print(f"{'case':30s} {'edited':7s} {'exit':5s} diagnosis")
    print("-" * 96)
    ok = True
    for name, _what, edited, code, diag in rows:
        if name == "null_control":
            good = (code == 0 and not edited)
            verdict = "accepted" if good else "FALSE POSITIVE"
        else:
            good = (code != 0 and edited and bool(diag))
            verdict = ("detected: " + diag) if good else "NOT DETECTED"
        ok = ok and good
        print(f"{name:30s} {str(edited):7s} {code:<5d} {verdict}")
    print()
    if ok:
        print(f"FALSIFICATION PASS: {len(CASES)} distinct corruptions "
              f"detected, each with a named diagnosis; null control "
              f"accepted, which is also the statement that a declared "
              f"removal already absent is not an error.")
        return 0
    print("FALSIFICATION FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
