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
Phase 17 regressions — the verification tools, and the closed vocabulary.

Two defect families, each with fixtures that were seen to fail on the
Phase 16 tree before the correction:

**N-8** — ``domain(name)`` accepted any string and answered ``None`` for
names it did not know.  ``None`` on that surface means *stated at every
age*, so a typo silently widened the declared domain to everything.  The
repair is a closed vocabulary, which needs no normative text: it is a
statement about which names exist.

**N-6 / N-7** — ``_patchkit.Patcher._selfcheck`` verified ``require`` and
``retire`` sentinels only, and ``create`` compared newline-normalised
*text*.  A patcher that transports whole files therefore reported PASS
after comparing nothing, and a tree whose line endings had been flipped
took the ``ALREADY`` branch: exit 0, neither repaired nor detected.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile

import pytest

from gensec.materials.base import (RheologicalModel, UnknownQuantity)
from gensec.materials.rheology import (EC2RheologicalModel,
                                       TabulatedRheologicalModel)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


class TestClosedVocabulary:
    r"""N-8: a name outside the vocabulary must raise, not answer."""

    def test_a_misspelt_quantity_raises_instead_of_answering(self):
        r"""
        The fixture that names the defect.

        On the Phase 16 tree this returned ``None`` and the assertion on
        the exception never ran.  ``None`` is the answer *stated at
        every age*, which is the most permissive declaration available,
        so the misspelling did not merely fail to help -- it removed the
        guard.
        """
        m = EC2RheologicalModel(fck=30.0)
        with pytest.raises(UnknownQuantity):
            m.domain("f_ctm_typo")

    def test_the_unsuffixed_spelling_is_rejected_too(self):
        r"""``fctm`` is the *property* name; ``f_ctm`` is the quantity
        name.  The two vocabularies are separate and neither may leak
        into the other by accident."""
        m = EC2RheologicalModel(fck=30.0)
        with pytest.raises(UnknownQuantity):
            m.domain("fctm")

    def test_states_and_domain_source_share_the_gate(self):
        r"""A guard on one entry point only is a guard with a door next
        to it."""
        m = EC2RheologicalModel(fck=30.0)
        with pytest.raises(UnknownQuantity):
            m.states("f_ctm_typo", 5.0)
        with pytest.raises(UnknownQuantity):
            m.domain_source("f_ctm_typo")

    def test_a_provider_speaks_the_names_it_exposes(self):
        r"""A quantity the provider *implements* is in its vocabulary
        whether or not it declared an interval for it.  Otherwise every
        third-party provider would have to restate its own method names,
        and a vocabulary that is expensive to extend gets routed
        around."""
        m = EC2RheologicalModel(fck=30.0)
        assert m.domain("E_c") is None
        assert m.domain("J") is None

    def test_the_container_invariants_are_in_the_base_vocabulary(self):
        r"""The four material functions plus the linearity limit are
        what the age-adjusted container calls on *any* provider, so they
        belong to the abstract base and not to a normative subclass."""
        assert {"J", "eps_imposed", "relaxation", "E_c",
                "linearity_limit"} <= RheologicalModel.QUANTITIES


class TestDeclaredDomainMatchesTheImplementation:
    r"""N-8, second half: what the provider declares must be what the
    code does."""

    def test_f_ctm_is_stated_at_every_age(self):
        r"""
        Expression (3.4) is written on :math:`\beta_{cc}(t)` and the
        28-day :math:`f_{ctm}`; ``fben2`` computes it from ``fck_28``
        and ``beta_cc`` at every age, below three days included.  The
        floor the provider used to declare described a route neither the
        code nor the clause takes -- the last fossil of the belief F-M5
        corrected when it restored :math:`\alpha = 1` there.

        On the Phase 16 tree this returned ``(3.0, inf)``.
        """
        m = EC2RheologicalModel(fck=30.0)
        assert m.domain("f_ctm") is None
        assert m.states("f_ctm", 0.5)

    def test_f_ck_and_f_cm_keep_their_floor(self):
        r"""The floor is not wrong everywhere: ``fben2`` zeroes both
        below three days as a deliberate marker, because
        :math:`f_{ck}(t) = f_{cm}(t) - 8` of §3.1.2(6) goes negative
        there.  Removing this one would be the opposite defect."""
        m = EC2RheologicalModel(fck=30.0)
        assert m.domain("f_ck") == (3.0, math.inf)
        assert m.domain("f_cm") == (3.0, math.inf)
        assert not m.states("f_ck", 0.5)

    def test_the_citation_that_named_the_wrong_route_is_gone(self):
        r"""``DOMAIN_SOURCE`` used to motivate the ``f_ctm`` floor
        "through f_cm(t) of §3.1.2(5)".  A citation that survives its
        interval cites nothing."""
        assert "f_ctm" not in EC2RheologicalModel.DOMAIN_SOURCE

    @pytest.mark.parametrize("cls", [EC2RheologicalModel,
                                     TabulatedRheologicalModel])
    def test_every_provider_is_internally_consistent(self, cls):
        r"""Cheap internal-consistency assertions run always: every
        declared interval is in the vocabulary, every citation has an
        interval."""
        report = cls.check_vocabulary()
        assert report["orphan_domain"] == []
        assert report["orphan_source"] == []

    def test_the_declared_but_unexposed_names_are_reported(self):
        r"""``f_ck`` and ``f_cm`` are declared and have no method: the
        provider computes them inside ``fben2`` and does not surface
        them.  That is legitimate, and it must be *visible* -- it was
        invisible until Phase 16 measured it."""
        assert EC2RheologicalModel.check_vocabulary(
            )["declared_no_surface"] == ["f_ck", "f_cm"]


def _run(args, cwd):
    """Run a patcher and return ``(exit_code, output)``, status read
    from the process and never through a pipe."""
    p = subprocess.run([sys.executable] + args, cwd=cwd,
                       capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(_ROOT, "patchers")),
    reason="patchers/ is not present in this checkout")
class TestPatchkitVerifiesBytes:
    r"""N-6 / N-7: the toolkit's own verification."""

    def _tree(self, tmp_path):
        r"""
        A copy of this tree, **restored to the state the Phase 16
        patcher authored**.

        The restore is the point, and it was added in Phase 18 after
        this fixture went red for a reason that was not a defect.  A
        patcher can only verify the files it transports; a later phase
        that legitimately edits one of them makes the predecessor's
        ``--selfcheck`` report FAIL, and that FAIL is a property of the
        method (``00_BOOTSTRAP.md`` section 4, rule F-B14), not a
        regression.  Asserting exit 0 on the *live* tree therefore
        pinned something false: "no future phase may touch these ten
        files".  Phase 18 touched three of them (``analysis.py`` for
        N-13, ``capacity.py`` for N-14, ``element_class_inventory.py``
        for the reasoned baseline) and the fixture went red without
        anything being wrong.

        ``--repair`` puts the transported bytes back, so what the
        selfcheck is asked about is the tree the patcher owns.  The
        claim under test is unchanged and is the one N-6 is about: the
        selfcheck **compares**, and says how many files it compared,
        instead of passing vacuously with zero checks.
        """
        import shutil
        dst = str(tmp_path / "gensec")
        shutil.copytree(_ROOT, dst, symlinks=True,
                        ignore=shutil.ignore_patterns(
                            ".git", "__pycache__", "*.bundle",
                            ".pytest_cache"))
        rc, out = _run(["patchers/run_phase16_tendon.py", "--repair"], dst)
        assert rc == 0, (
            "could not restore the temporary tree to the bytes the "
            "Phase 16 patcher transports; every assertion below would "
            "then be about a tree nobody owns.\n" + out)
        return dst

    def test_selfcheck_reports_what_it_compared(self, tmp_path):
        r"""A patcher that transports ten files and compares zero of
        them printed the same PASS as one that compared all ten."""
        tree = self._tree(tmp_path)
        rc, out = _run(["patchers/run_phase16_tendon.py", "--selfcheck"],
                       tree)
        assert rc == 0, out
        assert "file(s) byte-identical" in out

    def test_flipped_line_endings_are_detected(self, tmp_path):
        r"""
        The branch that used to be the worst combination available:
        exit 0, not repaired, not detected.  ``create`` compared the
        newline-normalised text, so a flipped tree was indistinguishable
        from a correct one.
        """
        tree = self._tree(tmp_path)
        target = os.path.join(tree, "src", "gensec", "materials",
                              "ec2_properties.py")
        raw = open(target, "rb").read()
        flipped = (raw.replace(b"\r\n", b"\n") if b"\r\n" in raw
                   else raw.replace(b"\n", b"\r\n"))
        open(target, "wb").write(flipped)

        rc, out = _run(["patchers/run_phase16_tendon.py", "--selfcheck"],
                       tree)
        assert rc == 1, out
        assert "line endings" in out

    def test_a_damaged_file_is_refused_by_default_and_named(self,
                                                            tmp_path):
        r"""Detect and refuse is the default; the diagnosis names the
        kind of damage, because "differs" sends the reader to a diff
        while "line endings differ" sends them to the transport."""
        tree = self._tree(tmp_path)
        target = os.path.join(tree, "src", "gensec", "materials",
                              "ec2_properties.py")
        raw = open(target, "rb").read()
        open(target, "wb").write(raw[: int(len(raw) * 0.6)])

        rc, out = _run(["patchers/run_phase16_tendon.py"], tree)
        assert rc == 1
        assert "ABORTED" in out
        assert "truncated" in out

    def test_repair_is_opt_in_and_restores_the_bytes(self, tmp_path):
        r"""Silent repair of a file the patcher did not author is the
        silent-restore failure this project rejects everywhere else, so
        it is behind an explicit flag."""
        tree = self._tree(tmp_path)
        target = os.path.join(tree, "src", "gensec", "materials",
                              "ec2_properties.py")
        want = open(target, "rb").read()
        open(target, "wb").write(want[: int(len(want) * 0.6)])

        rc, _ = _run(["patchers/run_phase16_tendon.py", "--repair"], tree)
        assert rc == 0
        assert open(target, "rb").read() == want
