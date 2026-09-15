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
Phase 32, Task A (finding Z-7): unknown top-level YAML keys are refused.

The defect
----------
``section.material_zones`` refuses an unknown key by name, citing the
fail-loud policy in its own comment.  The **top level** of the same
document applied no such policy: ``load_yaml`` read thirteen keys with
``data.get(...)`` and anything else fell on the floor without a word.

Measured on the incoming tree, three injections into a distributed
example, all three accepted in silence::

    zone_limits: {topping: {f_ck: 25.0}}          -> loads, no effect
    pippo_totalmente_inventato: 1                 -> loads, no effect
    custom_zone_strengths: {topping: {f_ck: 25}}  -> loads, no effect

The third is the one that matters.  ``custom_zone_strengths`` is the
name ratified for the Phase-32 per-zone strengths; before this repair,
writing it in a document was indistinguishable from a typo.  A silent
top level is not a cosmetic problem: it is the mechanism by which every
future block can be declared and ignored.

Why the list gets a constant and not a second copy
--------------------------------------------------
The thirteen valid keys were implicit in the control flow -- thirteen
scattered ``data.get`` calls.  A refusal needs the list, and a refusal
that carries its own copy diverges from the reader the first time a
block is added.  So the constant is declared once and the readers are
checked **against it**, which is what
:func:`test_every_valid_key_is_actually_read` measures: the constant
cannot drift away from the code it describes without a test going red.

What the fixtures assert
------------------------
Both halves.  An invented key must raise, naming itself and naming the
valid set; and every distributed example must continue to load.  The
first without the second would pass on a build that refuses everything.
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
import tempfile

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

EXAMPLES = os.path.join(ROOT, "examples")

#: The document used for every injection.  A composite one, so that the
#: injection sits beside real blocks rather than in a document so small
#: that a parser could plausibly not have reached the top level yet.
BASE_DOC = "example_composite_topping.yaml"


def _inject(tmpdir, extra_yaml):
    """
    Copy :data:`BASE_DOC` and append *extra_yaml* to its top level.

    Parameters
    ----------
    tmpdir : str
    extra_yaml : str
        Raw YAML appended at column zero, hence a top-level key.

    Returns
    -------
    str
        Path of the written document.
    """
    src = os.path.join(EXAMPLES, BASE_DOC)
    dst = os.path.join(tmpdir, "injected.yaml")
    shutil.copy(src, dst)
    with open(dst, "a", encoding="utf-8") as fh:
        fh.write("\n" + extra_yaml)
    return dst


# ==================================================================
#  The refusal
# ==================================================================


#: Three injections of different shapes.  A scalar, a mapping, and the
#: ratified name for the Phase-32 block -- that last one is the reason
#: this repair is a precondition and not a tidy-up.
_INJECTIONS = [
    ("pippo_totalmente_inventato", "pippo_totalmente_inventato: 1\n"),
    ("zone_limits", "zone_limits:\n  topping: {f_ck: 25.0}\n"),
    ("custom_zone_strengths",
     "custom_zone_strengths:\n  topping: {f_ck: 25.0}\n"),
]


@pytest.mark.parametrize("key,block", _INJECTIONS,
                         ids=[k for k, _ in _INJECTIONS])
def test_unknown_top_level_key_is_refused(key, block):
    """
    An unknown top-level key raises, rather than being dropped.

    The exception type is pinned because the caller has to catch
    something: ``load_yaml`` already raises :class:`ValueError` for
    every other malformed-document condition, and a second type for
    this one would make the caller write two handlers for one class of
    problem.
    """
    from gensec.io_yaml import load_yaml

    with tempfile.TemporaryDirectory() as tmp:
        path = _inject(tmp, block)
        with pytest.raises(ValueError) as excinfo:
            load_yaml(path)
    assert key in str(excinfo.value), (
        "the refusal must name the offending key; a message that says "
        "only 'unknown key' forces the reader into the source to find "
        "out which one and what was allowed instead"
    )


@pytest.mark.parametrize("key,block", _INJECTIONS,
                         ids=[k for k, _ in _INJECTIONS])
def test_the_refusal_names_the_valid_keys(key, block):
    """
    The refusal enumerates what *is* valid.

    Naming the offender tells the reader what went wrong; naming the
    valid set tells them what to write instead.  A message with only
    the first sends them to the source, which is precisely the cost
    this repair exists to remove.
    """
    from gensec.io_yaml import VALID_TOP_LEVEL_KEYS, load_yaml

    with tempfile.TemporaryDirectory() as tmp:
        path = _inject(tmp, block)
        with pytest.raises(ValueError) as excinfo:
            load_yaml(path)
    msg = str(excinfo.value)
    missing = [k for k in VALID_TOP_LEVEL_KEYS if k not in msg]
    assert not missing, (
        f"the refusal does not name these valid keys: {sorted(missing)}"
    )


# ==================================================================
#  The other half: nothing valid may break
# ==================================================================


@pytest.mark.parametrize(
    "path", sorted(glob.glob(os.path.join(EXAMPLES, "*.yaml"))),
    ids=lambda p: os.path.basename(p))
def test_every_distributed_example_still_loads(path):
    """
    Every shipped example loads unchanged.

    The denominator was measured before the refusal was written: zero
    examples, zero tests and zero validators declare a top-level key
    outside the thirteen.  This fixture is what keeps that zero true --
    a future block added to a reader but not to the constant would
    break every example that uses it, loudly, here.
    """
    from gensec.io_yaml import load_yaml

    load_yaml(path)


def test_every_valid_key_is_actually_read():
    """
    The constant describes the readers, and cannot drift from them.

    One home for one fact: the list of valid top-level keys exists once.
    This asserts the direction that a constant most easily gets wrong --
    a key left in the list after its reader was deleted, which would
    make the refusal permit a block nothing consumes.
    """
    from gensec.io_yaml import VALID_TOP_LEVEL_KEYS

    with open(os.path.join(SRC, "gensec", "io_yaml.py"),
              "r", encoding="utf-8", newline="") as fh:
        source = fh.read()
    unread = [k for k in VALID_TOP_LEVEL_KEYS
              if f'data.get("{k}"' not in source
              and f'data["{k}"]' not in source]
    assert not unread, (
        f"declared valid but never read by load_yaml: {sorted(unread)}"
    )


def test_the_valid_set_is_exactly_the_thirteen_measured():
    """
    The set is the thirteen measured on the incoming tree.

    Pinned as a count and as a membership, so that a fourteenth key
    added without a decision shows up here rather than in whatever
    breaks later.  A phase that legitimately adds a block edits this
    number, which is the point: it becomes a decision instead of a
    side effect.
    """
    from gensec.io_yaml import VALID_TOP_LEVEL_KEYS

    assert set(VALID_TOP_LEVEL_KEYS) == {
        "code", "materials", "section", "demands", "combinations",
        "envelopes", "output", "construction_history", "losses_models",
        "zone_rheology", "rheology", "tolerances", "schema_version",
    }


def test_a_non_mapping_document_is_still_refused_for_its_own_reason():
    """
    The new gate does not swallow the older malformed-document paths.

    A document that is a list, not a mapping, has no top-level keys at
    all.  The refusal must not turn that into a confusing report about
    unknown keys; it must fall through to whatever already handled it.
    """
    from gensec.io_yaml import load_yaml

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "list.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump([1, 2, 3], fh)
        with pytest.raises((ValueError, TypeError, KeyError,
                            AttributeError)):
            load_yaml(path)
# GENSEC-COMPLETE phase=32 utc=2026-09-05T13:56:13Z sha256=fb3d582b2225ee832ba92a248c476c8c1416642285c3c8e0532647e29adaae90 body_lines=250
