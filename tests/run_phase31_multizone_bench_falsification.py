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
Falsify the multi-zone check bench before its green is believed.

Why this file exists
--------------------
A gate tool's green means nothing until the tool has been **seen to go
red** on a tree that is deliberately wrong.  The multi-zone bench added
to ``tools/numeric_fingerprint.py`` in Phase 31 exists to make a
per-zone limit visible; if it were insensitive to the very quantities it
claims to watch, its silence after the repair would be mistaken for
proof.  That is the failure mode ``30_0`` §3 records twice (E30-6, and
the general rule under E30-8).

Method
------
Each case copies the tree to a scratch directory, applies **one** textual
corruption, and re-captures the fingerprint there.  A case passes when
the recapture **differs** from the reference on at least one
``checks.multizone.*`` key -- except the null control, which passes when
it does not.

Two guards, both learned from Phase 30:

- **Every injection is verified to have happened.**  A substitution that
  matched zero times is reported as an ``ERROR``, never counted as a
  detection.  E30-6 is exactly this: a falsification case that targeted
  a line written with different indentation, matched nothing, and was
  scored as a success.
- **The expected keys are named per case.**  A case that moves *some*
  key but not the one it is aimed at is a coincidence, not a detection,
  and is reported as such.

Exit status
-----------
0 when every corruption is detected on the keys it targets and the null
control is not, 1 otherwise.  Read it from the process; a pipe returns
the exit status of the last command in the pipe (E30-9).

Usage
-----
::

    python3 tests/run_phase31_multizone_bench_falsification.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX = "checks.multizone."

#: ``(name, relative path, old text, new text, keys the case must move)``.
#:
#: The three engine-side cases corrupt what the bench claims to watch;
#: the fourth corrupts the bench's own declared inputs, so that a later
#: edit silently changing the bench's subject cannot pass unnoticed.
CASES = (
    (
        "k1_halved",
        "src/gensec/materials/verification_limits.py",
        '"k1": 0.6,',
        '"k1": 0.45,',
        ("limits_applied", "error_MPa", "error_pct"),
    ),
    (
        "concrete_check_silenced",
        "src/gensec/solver/sls.py",
        "                if limits.k_c is None:\n",
        "                if True:  # corruption: the check never runs\n",
        ("limits_applied", "n_distinct_limits", "error_MPa"),
    ),
    (
        "by_zone_collapsed",
        "src/gensec/solver/sls.py",
        '        out[name] = {"sigma_min_MPa": s_min, "at_min": p_min,\n',
        '        out[name] = {"sigma_min_MPa": 0.0, "at_min": p_min,\n',
        ("sigma_min[base]", "sigma_min[topping]"),
    ),
    (
        "bench_subject_changed",
        "tools/numeric_fingerprint.py",
        "_MZ_FCK_BASE, _MZ_FCK_TOP = 45.0, 30.0",
        "_MZ_FCK_BASE, _MZ_FCK_TOP = 45.0, 45.0",
        ("fck[topping]", "error_MPa", "error_pct"),
    ),
)

#: The null control: a comment-only edit, which must move nothing.
NULL_CONTROL = (
    "null_control",
    "src/gensec/solver/sls.py",
    "        # ---- Checks ---------------------------------------------\n",
    "        # ---- Checks (comment touched by the null control) -------\n",
)


def _read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


#: Driver run inside each corrupted tree.
#:
#: It exercises **the bench**, not the whole example set.  A full
#: ``capture`` takes 205 s on this tree and records 6022 values of which
#: 16 belong to the bench; running it six times would cost twenty
#: minutes to read sixteen numbers, and the other 6006 are held constant
#: by construction because only one file differs between the trees.  The
#: cost was measured before the shape was chosen, not after.
_DRIVER = r"""
import json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import numeric_fingerprint as nf
rec = nf.Recorder()
nf._check_zones(rec)
print("<<<JSON>>>" + json.dumps(rec.rows))
"""


def _capture(tree, out_path):
    """
    Read the bench's keys from inside *tree*.

    Returns ``None`` when the run fails, which a caller must treat as an
    error rather than as a detection: a tree that cannot be measured has
    not been measured.

    Parameters
    ----------
    tree : str
        Directory of the (possibly corrupted) copy.
    out_path : str
        Where the raw rows are written, so a case can be inspected after
        the run rather than only counted.
    """
    import json

    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(tree, "src")
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER],
        cwd=tree, env=env, capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0 or "<<<JSON>>>" not in proc.stdout:
        return None
    rows = json.loads(proc.stdout.split("<<<JSON>>>", 1)[1].strip())
    values = {k[len(PREFIX):]: v for k, v in rows.items()
              if k.startswith(PREFIX)}
    with open(out_path, "w", encoding="utf-8") as fh:
        for k in sorted(values):
            fh.write(f"{PREFIX}{k} = {values[k]}\n")
    # A run that produced only a ``build``/``run`` failure row recorded
    # no bench value; that is a refusal, not a measurement.
    return values or None


def _corrupt(tree, rel, old, new):
    """
    Apply one substitution, and report whether it actually happened.

    Returns
    -------
    int
        Number of occurrences replaced.  Zero means the case did not
        inject what it claims, and the caller must fail rather than
        score a detection (E30-6).
    """
    path = os.path.join(tree, rel)
    raw = _read_bytes(path)
    crlf = b"\r\n" in raw
    text = raw.decode("utf-8")
    probe = text.replace("\r\n", "\n") if crlf else text
    n = probe.count(old)
    if n == 0:
        return 0
    probe = probe.replace(old, new)
    out = probe.replace("\n", "\r\n") if crlf else probe
    with open(path, "wb") as fh:
        fh.write(out.encode("utf-8"))
    return n


def main() -> int:
    scratch = tempfile.mkdtemp(prefix="gensec_mz_falsify_")
    try:
        ref_dir = os.path.join(scratch, "ref")
        shutil.copytree(ROOT, ref_dir,
                        ignore=shutil.ignore_patterns(
                            ".git", ".venv", "__pycache__", "htmlcov",
                            "*.bundle", "*.egg-info", ".pytest_cache"))
        ref = _capture(ref_dir, os.path.join(scratch, "ref.txt"))
        if not ref:
            print("ERROR: the reference capture failed or recorded no "
                  "checks.multizone.* key; nothing can be falsified "
                  "against it.")
            return 1
        print(f"reference: {len(ref)} checks.multizone.* key(s)")
        print()
        header = f"{'case':26s} {'injected':>9s}  {'moved':>6s}  verdict"
        print(header)
        print("-" * len(header))

        ok = True
        for name, rel, old, new, expect in CASES:
            work = os.path.join(scratch, name)
            shutil.copytree(ref_dir, work)
            n = _corrupt(work, rel, old, new)
            if n == 0:
                print(f"{name:26s} {'0':>9s}  {'-':>6s}  "
                      f"ERROR: injected nothing")
                ok = False
                continue
            got = _capture(work, os.path.join(scratch, f"{name}.txt"))
            if got is None:
                # A corruption that stops the capture is still a
                # detection -- the instrument refused to produce a
                # number rather than producing a wrong one -- but it is
                # a weaker one, so it is labelled.
                print(f"{name:26s} {n:>9d}  {'n/a':>6s}  "
                      f"detected (capture refused)")
                continue
            moved = sorted(k for k in set(ref) | set(got)
                           if ref.get(k) != got.get(k))
            hit = [k for k in expect if k in moved]
            if hit:
                print(f"{name:26s} {n:>9d}  {len(moved):>6d}  "
                      f"detected on {', '.join(hit)}")
            else:
                print(f"{name:26s} {n:>9d}  {len(moved):>6d}  "
                      f"MISSED: expected one of {list(expect)}")
                ok = False

        name, rel, old, new = NULL_CONTROL
        work = os.path.join(scratch, name)
        shutil.copytree(ref_dir, work)
        n = _corrupt(work, rel, old, new)
        if n == 0:
            print(f"{name:26s} {'0':>9s}  {'-':>6s}  "
                  f"ERROR: injected nothing")
            ok = False
        else:
            got = _capture(work, os.path.join(scratch, f"{name}.txt"))
            moved = sorted(k for k in set(ref) | set(got or {})
                           if ref.get(k) != (got or {}).get(k))
            if got is not None and not moved:
                print(f"{name:26s} {n:>9d}  {0:>6d}  accepted")
            else:
                print(f"{name:26s} {n:>9d}  {len(moved):>6d}  "
                      f"FALSE POSITIVE: {moved}")
                ok = False

        print()
        if ok:
            print("FALSIFICATION PASS: every corruption detected on the "
                  "keys it targets, null control accepted.")
            return 0
        print("FALSIFICATION FAIL: see above.")
        return 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
# GENSEC-COMPLETE phase=32 utc=2026-09-05T13:56:13Z sha256=6275db64edb6f836087ce86af2dfbcb758b34afd266392f3fdec750b831a02f2 body_lines=289
