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
Phase 31, task zero: three declarations measured false, pinned.

Why these are tests and not a note in a handoff
-----------------------------------------------
Each of the three is a *statement about this tree* that the tree
contradicts.  A note in a handoff records that someone once checked; a
test refuses to let the statement drift back.  All three cost almost
nothing to repair and were measured before being repaired, which is the
order the project requires -- a repair applied to an unmeasured claim
proves only that someone edited a file.

The three
---------
**Z-1** -- :meth:`gensec.solver.timeline.ConstructionTimeline.from_block`
promised, in its ``Raises`` section, a ``NotImplementedError`` for an
``interval`` carrying a ``losses`` key, "timeline losses land in Task 3".
Task 3 landed.  The live code validates the block and integrates it.  A
docstring that promises a refusal which does not happen is the exact
dual of a silent fallback: the reader is told a path is closed and it is
open.

**Z-2** -- ``handoff/10_6-GENSEC_PRESTRESS_ARC_CLOSURE.md`` §7 declared
"runtime run pending on your environment" and "Gate not yet green
end-to-end" for a gate measured green here and in every phase since.
The promotion of that section never happened.

**Z-6** -- ``handoff/00_BOOTSTRAP.md`` carried two counts that the tree
contradicts: the Phase-30 patcher's ``--selfcheck`` transports 31 files,
not 29, and the completion marker finds 19 stamped files, not 17.  This
one is the worst placed of the three, because ``00_BOOTSTRAP.md`` is the
file a session reads *first* and checks its own environment against: a
session opener comparing a real 31 against a declared 29 would conclude
the archive was corrupt.

What each test does and does not assert
---------------------------------------
The Z-1 test asserts **both halves**: that the behaviour is acceptance,
measured by running it, and that the docstring no longer promises a
refusal.  Asserting only the second would pass on a tree where the
docstring was corrected and the behaviour then changed back.

The Z-2 and Z-6 tests read the handoff files as text.  This is
deliberate: the defect *is* in the text, and a test that checked
anything else would be testing something the defect does not live in.
The Z-6 test does not hard-code 31 and 19 -- it reads them from the
tools, so that the file and the tree are compared with each other
rather than both against a constant that would itself go stale.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

HANDOFF = os.path.join(ROOT, "handoff")


def _read(path):
    """Read a file as text, whatever its line endings."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


# ==================================================================
#  Z-1 — a refusal declared that no longer happens
# ==================================================================


def _losses_timeline_block():
    """
    The smallest ``construction_history`` carrying ``interval.losses``.

    The shape is copied from ``examples/example_composite_losses.yaml``
    rather than invented: a losses spec is a mapping keyed by zone name,
    and an invented shape would be refused by
    :func:`gensec.solver.timeline._validate_losses_spec` for a reason
    that has nothing to do with what this test measures.
    """
    return [
        {"load": {"demand": "G1"}},
        {"interval": {"days": 62,
                      "losses": {"base": {"model": "rheo_precast",
                                          "age": 28, "curing": 3}}}},
    ]


def test_interval_with_losses_is_accepted_not_refused():
    """
    ``from_block`` accepts an ``interval`` carrying ``losses``.

    Measured behaviour, run rather than read.  This half of Z-1 is what
    makes the docstring half meaningful.
    """
    from gensec.io_yaml import load_yaml
    from gensec.solver.timeline import ConstructionTimeline

    doc = load_yaml(os.path.join(ROOT, "examples",
                                 "example_composite_losses.yaml"))
    tl = ConstructionTimeline.from_block(
        _losses_timeline_block(),
        losses_models=doc["losses_models"],
        zone_rheology=doc["zone_rheology"],
    )
    assert [e.kind for e in tl.events] == ["load", "interval"]
    assert "losses" in tl.events[1].payload


def test_from_block_docstring_promises_no_refusal_it_does_not_make():
    """
    The ``Raises`` section of ``from_block`` names no refusal for
    ``interval.losses``.

    A docstring is a declaration, and a false declaration is the defect
    this phase's task zero exists to remove.
    """
    from gensec.solver.timeline import ConstructionTimeline

    doc = ConstructionTimeline.from_block.__doc__ or ""
    assert "NotImplementedError" not in doc, (
        "from_block still declares a NotImplementedError; the live path "
        "validates and integrates interval.losses instead."
    )
    # The ``Raises`` section is the declaration; prose elsewhere in the
    # docstring may legitimately *mention* the old refusal in order to
    # say it is gone, and an earlier draft of this test forbade the
    # mention along with the declaration -- constraining the form of the
    # answer instead of its content (E30-3/4/5, committed again here and
    # recorded in the ledger).  Only the ``Raises`` section is read.
    raises = doc.split("Raises")[-1]
    assert "losses" not in raises or "malformed" in raises, (
        "the Raises section still refuses interval.losses outright; the "
        "only refusal on that path is a malformed or dangling spec."
    )
    assert "Task 3" not in raises and "task 3" not in raises, (
        "from_block still defers timeline losses to Task 3, which landed."
    )


# ==================================================================
#  Z-2 — a gate declared red that is green
# ==================================================================


@pytest.mark.parametrize("bullet", [
    "**runtime run pending on your\n  environment.**",
    "- **Gate not yet green end-to-end**:",
])
def test_prestress_arc_closure_no_longer_declares_a_red_gate(bullet):
    """
    ``10_6`` §7 does not *declare* pending what has been measured green.

    The predicate is the **bullet**, not the phrase.  The repaired
    section quotes both phrases in the paragraph that explains why they
    were removed, and a test forbidding the phrase anywhere in the file
    would make the explanation unwritable -- which is the same defect
    read backwards, since a repair that cannot say what it repaired
    leaves the next reader to rediscover it.

    The validator that section calls red is run by
    :func:`test_the_gate_10_6_called_red_is_green` below, so this file
    does not assert the gate's colour on trust.
    """
    text = _read(os.path.join(HANDOFF,
                              "10_6-GENSEC_PRESTRESS_ARC_CLOSURE.md"))
    assert bullet not in text, (
        f"10_6 still declares, as a status bullet, {bullet!r} for a gate "
        f"measured green."
    )


def test_prestress_arc_closure_declares_the_gate_green():
    """
    The positive half: ``10_6`` §7 states the measured status.

    Without this, the test above would pass on a file from which the
    section had simply been deleted.
    """
    text = _read(os.path.join(HANDOFF,
                              "10_6-GENSEC_PRESTRESS_ARC_CLOSURE.md"))
    assert "**Gate green end-to-end.**" in text


def test_the_gate_10_6_called_red_is_green():
    """
    The Task-3 validator exits 0, read from the process.

    Never through a pipe: ``cmd | grep; echo $?`` reads ``grep``'s exit
    status, which is the project's own standing rule and the subject of
    E30-9.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(
        [sys.executable,
         os.path.join(ROOT, "tests", "run_phase8_task3_validation.py")],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        f"run_phase8_task3_validation exited {proc.returncode}\n"
        f"{proc.stdout[-2000:]}"
    )


# ==================================================================
#  Z-6 — the opening file's own counts, wrong
# ==================================================================


def _bootstrap_text():
    return _read(os.path.join(HANDOFF, "00_BOOTSTRAP.md"))


def test_bootstrap_selfcheck_file_count_matches_the_patcher():
    """
    ``00_BOOTSTRAP.md`` §4 states the patcher's true payload size.

    The expected count is read from the patcher, not written here as a
    constant: a constant beside the instrument is the error pattern the
    ledger records under "writing a constant from a probe run beside the
    instrument rather than reading it from the instrument".
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    # Read from HEALTH_LIST, never named here (Phase 32).  This fixture
    # was written to catch a document naming a number by hand, and it
    # named a PATCHER by hand: when Phase 32 built its own, the fixture
    # kept measuring the Phase-30 payload and demanded the bootstrap
    # state a size belonging to a patcher section 4 no longer
    # prescribes.  HEALTH_LIST is the single home for "which patcher
    # section 4 says is green", so it is the only place to ask.
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import selfcheck_inventory
    assert len(selfcheck_inventory.HEALTH_LIST) == 1, (
        "F-B14: exactly one patcher is on the health list; this fixture "
        "reads it and has no rule for choosing among several")
    patcher = os.path.join(
        ROOT, "patchers", selfcheck_inventory.HEALTH_LIST[0])
    proc = subprocess.run(
        [sys.executable, patcher, "--selfcheck"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    # The exit status is deliberately **not** asserted.  What
    # 00_BOOTSTRAP.md §4 states is the size of the patcher's payload,
    # which is a property of the patcher; whether the comparison passes
    # is a property of the tree, and it is *supposed* to go red as soon
    # as a later phase edits a file the patcher transports (F-B14).
    # Phase 31 edits handoff/00_BOOTSTRAP.md itself, so requiring exit 0
    # here would make the statement untestable in the phase that repairs
    # it -- and, worse, would tie a stable number to a moving verdict.
    # Phase 38: the load is no longer contents alone.  The patcher now
    # carries a declared REMOVE block, and a removal is part of what it
    # transports -- a tree rebuilt from a payload that ignored removals
    # kept one validator too many, red by construction (37_0 s10.5).
    # The number this fixture pins is therefore PAYLOAD + REMOVE, and
    # the wording moved with it: "voci di carico", not "file
    # byte-identici", because calling a removed path byte-identical
    # would be a false sentence written to keep a regex happy.  The two
    # sub-counts stay visible in the patcher's own output; what
    # 00_BOOTSTRAP.md states, and what this reads, is the total.
    m = re.search(r"(\d+)\s+load entr\(ies\):", proc.stdout)
    assert m, f"unparsable selfcheck tail:\n{proc.stdout[-500:]}"
    n_true = int(m.group(1))
    text = _bootstrap_text()
    assert f"{n_true} voci di carico" in text, (
        f"00_BOOTSTRAP.md does not state the true load size "
        f"({n_true} entries: transported contents plus declared "
        f"removals)."
    )


def _worktree_is_clean():
    """
    Whether the tree has no modified or untracked file.

    ``completion_marker`` walks the **working tree**, so its counts move
    the moment a phase adds a file and stop moving when the phase
    commits.  The statement in ``00_BOOTSTRAP.md`` describes the
    *archive*, i.e. the tree as delivered, so the comparison is
    meaningful exactly when the tree is the delivered one.
    """
    try:
        proc = subprocess.run(["git", "status", "--porcelain"],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=120)
    except (OSError, subprocess.SubprocessError):
        return False
    if proc.returncode != 0:
        return False
    # An installed editable package regenerates src/gensec/_version.py
    # and drops src/gensec.egg-info/; neither is the phase's work and
    # neither changes a marker count.
    noise = ("src/gensec/_version.py", "src/gensec.egg-info")
    for line in proc.stdout.splitlines():
        path = line[3:].strip().strip('"')
        if any(path.startswith(n) for n in noise):
            continue
        return False
    return True


def test_bootstrap_completion_counts_match_the_tool():
    """
    ``00_BOOTSTRAP.md`` §4.2 states the true number of stamped files.

    Same rule as above: the number is read from
    ``tools/completion_marker.py``, never written here as a constant.

    Skipped on a dirty tree, with the reason stated rather than passing
    vacuously.  A phase in progress has files the archive does not, so
    on a dirty tree the two numbers are *supposed* to disagree and a
    green here would mean nothing.  The check bites where it matters: on
    the closed archive a session opener actually reads.
    """
    if not _worktree_is_clean():
        pytest.skip(
            "worktree is dirty: completion_marker counts the working "
            "tree, 00_BOOTSTRAP.md describes the delivered archive, and "
            "the two are the same tree only at closure."
        )
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools",
                                      "completion_marker.py"), "--gate"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-2000:]
    m = re.search(r"(\d+)\s+complete,\s+(\d+)\s+exempt", proc.stdout)
    assert m, f"unparsable gate tail:\n{proc.stdout[-500:]}"
    n_complete, n_exempt = int(m.group(1)), int(m.group(2))
    text = _bootstrap_text()
    assert f"{n_complete} timbrati, {n_exempt} esenti" in text, (
        f"00_BOOTSTRAP.md does not state the true marker counts "
        f"({n_complete} stamped, {n_exempt} exempt)."
    )
# GENSEC-COMPLETE phase=38 utc=2026-09-11T16:04:37Z sha256=047b47cb8de126c16b063ae92dcd96b9a664bec63c8dba0f91d0e834e52f3140 body_lines=365
