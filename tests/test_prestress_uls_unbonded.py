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
Unit tests for prestress Phase 6 (unbonded / external tendons).

Covers the ULS action factory
:meth:`~gensec.solver.section_state.PrestressAction.from_force_uls`, the
EN 1992-1-1 §5.10.8(3) default helper
:func:`~gensec.materials.prestress_properties.delta_sigma_p_uls`, and the
strain-compatibility boundary guards (non-finite rejection, the
``eps_pe`` / ``sectional_strain`` guards, and the primary
``Tendon(bonded=False)`` construction guard).

There is deliberately no FiberSolver-level unbonded-tendon check: the
section exposes tendons to the solver as arrays with no per-tendon bond
flag (``materialize_view`` excludes the unbonded ones upstream), so the
boundary is enforced at ``Tendon.__post_init__``, not at the solver.

The reference values are closed-form; ``pytest.approx`` guards the
floating-point comparisons and ``pytest.raises`` / ``pytest.warns`` guard
the fail-loud paths.
"""

import unittest

import numpy as np
import pytest

from gensec.solver.section_state import PrestressAction
from gensec.materials.prestress_properties import delta_sigma_p_uls
from gensec.geometry.fiber import Tendon


class TestFromForceUls(unittest.TestCase):
    r"""ULS force composition and couple of an unbonded/external tendon."""

    P_EFF = 1.5e6
    AP = 1500.0
    DSP = 100.0
    X, Y = 200.0, 80.0
    XR, YR = 150.0, 300.0

    @staticmethod
    def _uls_force(P_eff, ap, dsp):
        r"""Closed-form ULS force :math:`P_{ULS}=P_{eff}+\Delta\sigma_p A_p`."""
        return P_eff + dsp * ap

    def test_force_composition_and_couple(self):
        a = PrestressAction.from_force_uls(
            self.P_EFF, self.X, self.Y, Ap=self.AP,
            delta_sigma_p=self.DSP, x_ref=self.XR, y_ref=self.YR)
        p_uls = self._uls_force(self.P_EFF, self.AP, self.DSP)
        self.assertEqual(a.N, pytest.approx(-p_uls))
        self.assertEqual(a.Mx, pytest.approx(-p_uls * (self.Y - self.YR)))
        self.assertEqual(a.My, pytest.approx(+p_uls * (self.X - self.XR)))

    def test_equivalent_to_from_force_at_uls_force(self):
        p_uls = self._uls_force(self.P_EFF, self.AP, self.DSP)
        a = PrestressAction.from_force_uls(
            self.P_EFF, self.X, self.Y, Ap=self.AP,
            delta_sigma_p=self.DSP, x_ref=self.XR, y_ref=self.YR)
        b = PrestressAction.from_force(
            p_uls, self.X, self.Y, x_ref=self.XR, y_ref=self.YR)
        self.assertEqual(a.triple(), pytest.approx(b.triple()))

    def test_zero_increment_collapses_to_effective_prestress(self):
        a = PrestressAction.from_force_uls(
            self.P_EFF, self.X, self.Y, Ap=self.AP,
            delta_sigma_p=0.0, x_ref=self.XR, y_ref=self.YR)
        b = PrestressAction.from_force(
            self.P_EFF, self.X, self.Y, x_ref=self.XR, y_ref=self.YR)
        self.assertEqual(a.triple(), pytest.approx(b.triple()))

    def test_origin_tag_default(self):
        a = PrestressAction.from_force_uls(
            self.P_EFF, self.X, self.Y, Ap=self.AP, delta_sigma_p=self.DSP)
        self.assertEqual(a.origin, "prestress_uls_unbonded")


class TestDesignStrengthCap(unittest.TestCase):
    r"""Optional f_pd cap: verbatim below, warns and caps above, never silent."""

    AP = 1500.0
    P_EFF, DSP = 1.5e6, 100.0     # composed sigma_uls = 1100 MPa
    X, Y = 100.0, 540.0

    def test_no_cap_is_verbatim(self):
        a = PrestressAction.from_force_uls(
            self.P_EFF, self.X, self.Y, Ap=self.AP, delta_sigma_p=self.DSP)
        self.assertEqual(a.N, pytest.approx(-(self.P_EFF + self.DSP * self.AP)))

    def test_cap_above_composed_stress_no_warning(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            a = PrestressAction.from_force_uls(
                self.P_EFF, self.X, self.Y, Ap=self.AP,
                delta_sigma_p=self.DSP, sigma_pd_cap=1500.0)
        self.assertEqual(a.N, pytest.approx(-(self.P_EFF + self.DSP * self.AP)))

    def test_cap_below_composed_stress_warns_and_caps(self):
        with pytest.warns(UserWarning):
            a = PrestressAction.from_force_uls(
                self.P_EFF, self.X, self.Y, Ap=self.AP,
                delta_sigma_p=self.DSP, sigma_pd_cap=1050.0)
        self.assertEqual(a.N, pytest.approx(-(1050.0 * self.AP)))


class TestInputGuards(unittest.TestCase):
    r"""``from_force_uls`` rejects a non-positive area / negative increment."""

    def test_non_positive_area_raises(self):
        for ap in (0.0, -5.0):
            with pytest.raises(ValueError):
                PrestressAction.from_force_uls(
                    1.5e6, 100.0, 540.0, Ap=ap, delta_sigma_p=100.0)

    def test_negative_increment_raises(self):
        with pytest.raises(ValueError):
            PrestressAction.from_force_uls(
                1.5e6, 100.0, 540.0, Ap=1500.0, delta_sigma_p=-1.0)


class TestDeltaSigmaPUls(unittest.TestCase):
    r"""EN 1992-1-1 §5.10.8(3) recommended increment and its NA machinery."""

    def test_ec2_recommended_value(self):
        self.assertEqual(delta_sigma_p_uls(), pytest.approx(100.0))
        self.assertEqual(delta_sigma_p_uls(NA="EC2"), pytest.approx(100.0))

    def test_override_bypasses_table(self):
        self.assertEqual(
            delta_sigma_p_uls(delta_override=137.0), pytest.approx(137.0))

    def test_override_wins_over_unknown_annex(self):
        self.assertEqual(
            delta_sigma_p_uls(NA="NTC18", delta_override=90.0),
            pytest.approx(90.0))

    def test_unimplemented_annex_raises(self):
        with pytest.raises(ValueError):
            delta_sigma_p_uls(NA="NTC18")


class TestBoundaryGuards(unittest.TestCase):
    r"""The strain-compatibility boundary must fail loudly, never silently."""

    def test_non_finite_action_rejected(self):
        for bad in (np.inf, -np.inf, np.nan):
            with pytest.raises(ValueError):
                PrestressAction(bad, 0.0, 0.0)
        with pytest.raises(ValueError):
            PrestressAction(0.0, np.nan, 0.0)

    def test_eps_pe_property_raises(self):
        act = PrestressAction.from_force(1.4e6, 200.0, 80.0)
        with pytest.raises(TypeError):
            _ = act.eps_pe

    def test_sectional_strain_method_raises(self):
        act = PrestressAction.from_force(1.4e6, 200.0, 80.0)
        with pytest.raises(TypeError):
            act.sectional_strain(0.0, 0.0, 0.0)

    def test_unbonded_tendon_not_constructible(self):
        # Primary enforcement upstream of the solver (Ap>0 clears the
        # area check first, then bonded=False raises).
        with pytest.raises(ValueError):
            Tendon(y=60.0, Ap=150.0, material=object(), bonded=False)


class TestDemandWalkIntegration(unittest.TestCase):
    r"""An unbonded ULS action sums into the cumulative demand as the
    staged engines do (``cum += sum(action.triple())``)."""

    def test_action_accumulates_into_demand(self):
        xr, yr = 150.0, 300.0
        p_eff, ap, dsp = 1.5e6, 1500.0, 100.0
        x, y = 200.0, 80.0
        act = PrestressAction.from_force_uls(
            p_eff, x, y, Ap=ap, delta_sigma_p=dsp, x_ref=xr, y_ref=yr)

        g = dict(N=-300e3, Mx=50e6, My=0.0)
        cum = np.array([g["N"], g["Mx"], g["My"]], dtype=float)
        cum += np.array(act.triple(), dtype=float)

        p_uls = p_eff + dsp * ap
        self.assertEqual(cum[0], pytest.approx(-300e3 - p_uls))
        self.assertEqual(cum[1], pytest.approx(50e6 - p_uls * (y - yr)))
        self.assertEqual(cum[2], pytest.approx(0.0 + p_uls * (x - xr)))


if __name__ == "__main__":
    unittest.main()
