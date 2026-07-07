#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# GenSec — Phase-8 Task-2 (construction timeline) integration patcher.
# Copyright (C) 2026  Andrea (GenSec project).
r"""
Idempotent, CRLF-preserving patcher for the Phase-8 Task-2 wiring.

The Task-2 **new files** are dropped in directly (no surgery needed)::

    src/gensec/solver/timeline.py         # Layer-2 compiler
    src/gensec/solver/timeline_run.py     # orchestration driver
    tests/test_phase8_task2.py            # test suite
    run_phase8_task2_validation.py        # independent validator
    example_composite_timeline.yaml       # flagship example

This script performs only the **in-place** edits on two existing files:

  io_yaml.py
    (1) preserve the ``construction_history`` block in ``load_yaml``'s
        return (absent -> ``None`` -> the whole machinery is inert);
    (2) preserve the ``at`` / ``history_factors`` / ``gamma_P`` anchor
        keys on components-based combinations.

  cli.py
    (3) after the combinations are loaded, when a timeline is present,
        run :func:`gensec.solver.timeline_run.run_timeline` and print the
        per-anchor governing results, and drop the anchored combinations
        from the legacy per-combination loop (the driver owns them).

Each edit is guarded by a sentinel: re-running the patcher is a no-op
(triple-run idempotency is asserted in ``--selfcheck``).  Line endings
are detected per file and preserved at the byte level.

Usage::

    python apply_phase8_task2.py [--root .] [--selfcheck] [--dry-run]
"""

from __future__ import annotations

import argparse
import io
import os
import sys

# --- Edit units: (relative path, sentinel, anchor, inserted text) ----------
# ``anchor`` is matched on the LF-normalized text; the inserted text is
# spliced immediately BEFORE it.  ``sentinel`` presence => already applied.

IO_YAML = os.path.join("src", "gensec", "io_yaml.py")
CLI = os.path.join("src", "gensec", "cli.py")

EDIT_IO_CH_SENTINEL = "# Phase-8 Task-2: the single construction timeline"
EDIT_IO_CH_ANCHOR = (
    "    return {\n"
    "        \"materials\": materials,\n"
    "        \"section\": section,\n"
    "        \"demands\": demands,\n"
    "        \"combinations\": combinations,\n"
    "        \"envelopes\": envelopes,\n"
    "        \"output_options\": output_opts,\n"
    "    }\n"
)
EDIT_IO_CH_NEW = (
    "    # Phase-8 Task-2: the single construction timeline (G-D1).\n"
    "    # Carried raw (a list of single-key event mappings) for\n"
    "    # gensec.solver.timeline.ConstructionTimeline.from_block. Absent\n"
    "    # -> None and the whole timeline machinery is inert (the run is\n"
    "    # byte-identical to the pre-Task-2 behaviour).\n"
    "    construction_history = data.get(\"construction_history\")\n"
    "\n"
    "    return {\n"
    "        \"materials\": materials,\n"
    "        \"section\": section,\n"
    "        \"demands\": demands,\n"
    "        \"combinations\": combinations,\n"
    "        \"envelopes\": envelopes,\n"
    "        \"output_options\": output_opts,\n"
    "        \"construction_history\": construction_history,\n"
    "    }\n"
)

EDIT_IO_COMBO_SENTINEL = "# Phase-8 Task-2: a components-based combination"
EDIT_IO_COMBO_ANCHOR = (
    "        return {\n"
    "            \"name\": name,\n"
    "            \"components\": _parse_component_list(c_spec[\"components\"]),\n"
    "        }\n"
)
EDIT_IO_COMBO_NEW = (
    "        combo = {\n"
    "            \"name\": name,\n"
    "            \"components\": _parse_component_list(c_spec[\"components\"]),\n"
    "        }\n"
    "        # Phase-8 Task-2: a components-based combination may anchor at\n"
    "        # a construction-timeline point. The anchor metadata is passed\n"
    "        # verbatim to the timeline compiler; its presence does not\n"
    "        # conflict with the components/stages exclusivity (the compiler\n"
    "        # emits the stages). Combinations without 'at' are unaffected.\n"
    "        for _tl_key in (\"at\", \"history_factors\", \"gamma_P\"):\n"
    "            if _tl_key in c_spec:\n"
    "                combo[_tl_key] = c_spec[_tl_key]\n"
    "        return combo\n"
)

EDIT_CLI_SENTINEL = "# Phase-8 Task-2: construction-timeline driver"
EDIT_CLI_ANCHOR = "    staged_mgr = None\n"
EDIT_CLI_NEW = (
    "    # Phase-8 Task-2: construction-timeline driver (single opt-in\n"
    "    # gate). When a construction_history is present, one timeline is\n"
    "    # built and resolved once; every combination carrying an 'at'\n"
    "    # anchor is compiled against it and verified per anchor point\n"
    "    # (governing = transparent max, decision C2). Anchored\n"
    "    # combinations are then removed from the legacy loop below.\n"
    "    from .solver.timeline_run import run_timeline, timeline_active\n"
    "    if timeline_active(data):\n"
    "        _tl_out = run_timeline(data, n_points=args.n_points)\n"
    "        print(\"\\n  Construction timeline active: verifying anchored \"\n"
    "              \"combinations per point.\")\n"
    "        for _cname, _gov in _tl_out[\"anchored\"].items():\n"
    "            print(f\"    {_cname}: governing point \"\n"
    "                  f\"'{_gov['governing_point']}' \"\n"
    "                  f\"eta={_gov['eta_governing']} \"\n"
    "                  f\"({'OK' if _gov['verified'] else 'NOT VERIFIED'})\")\n"
    "            for _pt, _r in _gov[\"per_point\"].items():\n"
    "                print(f\"        {_pt}: eta={_r.get('eta_governing')}\")\n"
    "        _anchored_names = set(_tl_out[\"anchored\"])\n"
    "        combinations = [c for c in combinations\n"
    "                        if c.get(\"name\") not in _anchored_names]\n"
    "\n"
    "    staged_mgr = None\n"
)

EDITS = [
    (IO_YAML, EDIT_IO_CH_SENTINEL, EDIT_IO_CH_ANCHOR, EDIT_IO_CH_NEW),
    (IO_YAML, EDIT_IO_COMBO_SENTINEL, EDIT_IO_COMBO_ANCHOR, EDIT_IO_COMBO_NEW),
]

# ---------------------------------------------------------------------------
#  cli.py / api.py call-site (documented, applied by hand — one call).
#  cli.main's local layout differs enough between refactors that a
#  string-surgery anchor here is fragile (the ``staged_mgr = None`` anchor
#  is non-unique in the current tree, and this patcher refuses to guess).
#  The driver is a single call; place it right after ``combinations`` is
#  loaded and before the per-combination loop:
#
#      from .solver.timeline_run import run_timeline, timeline_active
#      if timeline_active(data):
#          tl_out = run_timeline(data, n_points=args.n_points)
#          for cname, gov in tl_out["anchored"].items():
#              print(f"  {cname}: governing '{gov['governing_point']}' "
#                    f"eta={gov['eta_governing']} "
#                    f"({'OK' if gov['verified'] else 'NOT VERIFIED'})")
#              for pt, r in gov["per_point"].items():
#                  print(f"      {pt}: eta={r.get('eta_governing')}")
#          # anchored combinations are owned by the driver; drop them from
#          # the legacy per-combination loop:
#          _anchored = set(tl_out["anchored"])
#          combinations = [c for c in combinations
#                          if c.get("name") not in _anchored]
#
#  ``EDIT_CLI_*`` below are retained for reference / a future tightened
#  anchor, but are NOT in EDITS.
# ---------------------------------------------------------------------------


def _read(path):
    r"""Read *path* as bytes; return (text_lf, newline) where newline is
    the detected line ending ('\r\n' or '\n')."""
    with open(path, "rb") as fh:
        raw = fh.read()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    return raw.decode("utf-8").replace("\r\n", "\n"), newline


def _write(path, text_lf, newline):
    r"""Write *text_lf* to *path* using *newline*, at the byte level."""
    data = text_lf.replace("\n", newline).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(data)


def apply_edit(root, rel, sentinel, anchor, new, *, dry_run=False):
    r"""
    Apply a single guarded edit; return a status string.

    Raises
    ------
    RuntimeError
        If the anchor is absent or non-unique (a refusal to guess).
    """
    path = os.path.join(root, rel)
    text, newline = _read(path)
    if sentinel in text:
        return f"  = already applied: {rel} ({sentinel[:40]}...)"
    n = text.count(anchor)
    if n == 0:
        raise RuntimeError(
            f"anchor not found in {rel}; refusing to guess. Anchor:\n"
            f"{anchor}")
    if n > 1:
        raise RuntimeError(
            f"anchor is non-unique ({n} matches) in {rel}; refusing to "
            f"guess. Tighten the anchor.")
    patched = text.replace(anchor, new, 1)
    if not dry_run:
        _write(path, patched, newline)
    return (f"  + applied ({'dry-run' if dry_run else newline!r} EOL): "
            f"{rel}")


def run(root, *, dry_run=False):
    r"""Apply every edit once; print a report."""
    print(f"GenSec Phase-8 Task-2 patcher  (root={root!r}, "
          f"dry_run={dry_run})")
    for rel, sentinel, anchor, new in EDITS:
        print(apply_edit(root, rel, sentinel, anchor, new, dry_run=dry_run))
    print("done.")


def selfcheck(root):
    r"""
    Triple-run idempotency + CRLF-preservation self-check.

    Applies the edits three times to scratch copies and asserts the
    result is stable after the first pass and the byte-level line ending
    is unchanged.
    """
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="gensec_t2_")
    try:
        for rel, *_ in EDITS:
            dst = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(os.path.join(root, rel), dst)
            with open(dst, "rb") as fh:
                eol0 = b"\r\n" if b"\r\n" in fh.read() else b"\n"
            # three passes
            for _ in range(3):
                for r, s, a, n in EDITS:
                    if r == rel:
                        apply_edit(tmp, r, s, a, n)
            with open(dst, "rb") as fh:
                raw = fh.read()
            assert raw.count(b"\r\n") if eol0 == b"\r\n" else True, \
                f"CRLF lost in {rel}"
            for _r, s, _a, _n in EDITS:
                if _r == rel:
                    assert raw.decode("utf-8").count(s) == 1, \
                        f"edit not idempotent in {rel} ({s[:30]})"
        print("selfcheck: PASS (triple-run idempotent, EOL preserved)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".",
                    help="repository root (default: cwd)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selfcheck", action="store_true",
                    help="run triple-run idempotency check and exit")
    args = ap.parse_args(argv)
    if args.selfcheck:
        return selfcheck(args.root)
    run(args.root, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
