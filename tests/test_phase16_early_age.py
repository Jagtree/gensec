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
Phase 16 — the tensile strength of concrete younger than three days.

The defect
----------
``ec2_properties.fben2`` carried, from Phase 12 to Phase 15, a branch

.. code-block:: python

    elif time < 3:
        self.fcm = 0
        self.fck = 0
        alpha = 0

in which ``alpha`` is the **exponent** of EN 1992-1-1 expression (3.4),

.. math::

    f_{ctm}(t) = \bigl[\beta_{cc}(t)\bigr]^{\alpha}\, f_{ctm},
    \qquad \alpha = 1 \;\; (t < 28\,\mathrm{d}),
    \qquad \alpha = 2/3 \;\; (t \ge 28\,\mathrm{d}).

With :math:`\alpha = 0` the factor collapses to one and a concrete of
half a day is handed the **28-day** tensile strength — a
**non-conservative** substitution, and a discontinuity at :math:`t = 3`
in the wrong direction: the tensile strength *fell* as the concrete
matured past three days.

Why the tests are written this way
----------------------------------
They name the closed form of expression (3.4) and the continuity of the
function, not the numbers the implementation happens to produce.  A
fixture that pins the output cannot tell a repair from a regression;
one that states the clause can.

The three-day floor of :attr:`EC2RheologicalModel.DOMAIN` is **not**
challenged here.  It belongs to :math:`f_{cm}(t)` and :math:`f_{ck}(t)`,
whose mean-to-characteristic offset of 8 MPa is what expression (3.1)
stops supporting below three days; expression (3.4) is written on
:math:`\beta_{cc}(t)` and the 28-day :math:`f_{ctm}` and never routes
through :math:`f_{cm}(t)`.  The zero those two properties return is a
marker guarded by the provider's declared domain, and it is left alone.
"""

import math
import unittest

import pytest

from gensec.materials.ec2_properties import fben2


def _beta_cc(t, s):
    r"""EN 1992-1-1 expression (3.2), :math:`\beta_{cc}(t)`."""
    return math.exp(s * (1.0 - math.sqrt(28.0 / t)))


#: ``s`` of EN 1992-1-1 §3.1.2(6) per cement class, as ``fben2`` uses it.
_S_CEM = {"R": 0.20, "N": 0.25, "S": 0.38}


def _props(fck, cem, t):
    r"""``fben2`` at age *t*, with the service (undesigned) factors."""
    return fben2(fck=fck, ls="S", loadtype="slow", TypeConc=cem, time=t)


class TestFctmBelowThreeDays(unittest.TestCase):
    r"""Expression (3.4) is stated for every age below 28 days."""

    CASES = ((30.0, "N"), (30.0, "R"), (60.0, "N"))
    YOUNG = (0.5, 1.0, 2.0, 2.9)

    def test_fctm_follows_expression_3_4_below_three_days(self):
        r"""
        :math:`f_{ctm}(t) = \beta_{cc}(t)\, f_{ctm,28}` for
        :math:`t < 3` d.

        The reference is built from the clause — :math:`\beta_{cc}` from
        (3.2), :math:`f_{ctm}` from Table 3.1 as the object itself
        reports it at 28 days — so the assertion is the Eurocode's, not
        a transcription of the implementation.
        """
        for fck, cem in self.CASES:
            fctm28 = _props(fck, cem, 28.0).fctm_28
            for t in self.YOUNG:
                p = _props(fck, cem, t)
                expected = _beta_cc(t, _S_CEM[cem]) ** 1.0 * fctm28
                assert p.fctm == pytest.approx(expected, rel=1e-12), (
                    f"C{fck:.0f}/{cem} at t = {t} d: f_ctm reads "
                    f"{p.fctm:.6f} MPa where EN 1992-1-1 (3.4) with "
                    f"alpha = 1 gives {expected:.6f} MPa "
                    f"(ratio {p.fctm / expected:.3f})"
                )

    def test_fctm_is_continuous_across_the_three_day_boundary(self):
        r"""
        No step at :math:`t = 3` d.

        A discontinuity in a material property across an age boundary is
        a modelling artefact whichever way it points; this one pointed
        the wrong way, the strength *dropping* by a factor 1.672 as the
        concrete matured.
        """
        for fck, cem in self.CASES:
            below = _props(fck, cem, 2.999).fctm
            above = _props(fck, cem, 3.001).fctm
            assert below == pytest.approx(above, rel=1e-3), (
                f"C{fck:.0f}/{cem}: f_ctm jumps from {below:.6f} MPa at "
                f"t = 2.999 d to {above:.6f} MPa at t = 3.001 d "
                f"(step {above - below:+.6f} MPa)"
            )

    def test_fctm_is_monotone_non_decreasing_in_age(self):
        r"""
        Tensile strength does not fall as concrete cures, below 28 days.

        Stated separately from continuity because a *smooth* inversion
        would satisfy the previous test and still be wrong.
        """
        ages = (0.5, 1.0, 2.0, 2.9, 3.0, 7.0, 14.0, 28.0)
        for fck, cem in self.CASES:
            values = [_props(fck, cem, t).fctm for t in ages]
            for (t0, v0), (t1, v1) in zip(zip(ages, values),
                                          zip(ages[1:], values[1:])):
                assert v1 >= v0 - 1e-12, (
                    f"C{fck:.0f}/{cem}: f_ctm falls from {v0:.6f} MPa at "
                    f"t = {t0} d to {v1:.6f} MPa at t = {t1} d"
                )

    def test_cracking_strain_inherits_the_correction(self):
        r"""
        :math:`\varepsilon_{ct} = f_{ctm}/E_{cm}` is the quantity through
        which the tensile strength actually reaches a criterion, so it
        is asserted in its own right rather than assumed to follow.
        """
        p = _props(30.0, "N", 1.0)
        expected = (_beta_cc(1.0, 0.25) * _props(30.0, "N", 28.0).fctm_28
                    / p.ecm)
        assert p.eps_ct == pytest.approx(expected, rel=1e-12)


class TestCompressiveMarkerIsUntouched(unittest.TestCase):
    r"""
    The zero on :math:`f_{cm}` / :math:`f_{ck}` below three days is a
    declared-domain marker and stays.

    Guards the repair against over-reach: expression (3.1) genuinely
    stops below three days, where :math:`f_{cm}(t) - 8` goes negative,
    and turning that marker into a number would trade a loud absence for
    a quiet nonsense.
    """

    def test_fcm_and_fck_remain_zero_below_three_days(self):
        for t in (0.5, 1.0, 2.0, 2.9):
            p = _props(30.0, "N", t)
            assert p.fcm == 0.0
            assert p.fck == 0.0

    def test_the_provider_still_refuses_the_linearity_limit(self):
        from gensec.materials.base import OutOfDeclaredDomain
        from gensec.materials.rheology import EC2RheologicalModel
        m = EC2RheologicalModel(fck=30, cement_class="N")
        with pytest.raises(OutOfDeclaredDomain):
            m.linearity_limit(1.0)


if __name__ == "__main__":                               # pragma: no cover
    unittest.main()
