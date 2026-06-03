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
Phase-2 prestress: pre-tensioning transfer and elastic-shortening loss.

The elastic-shortening loss is validated against the exact
transformed-section closed form for both the **centroidal** and the
**eccentric** tendon.  A linear-elastic bulk is used deliberately: the
fiber model is then algebraically identical to transformed-section
linear elasticity, so the agreement is at machine precision once the
reference uses the section's *meshed* area and inertia (isolating the
solver from the O(1/n_y^2) strip-inertia discretization).
"""

import numpy as np
import pytest

from gensec.geometry.section import RectSection
from gensec.geometry.fiber import Tendon
from gensec.solver.prestress_transfer import (
    solve_pretension_transfer,
    elastic_shortening_loss,
)


class LinearElastic:
    r"""Minimal linear-elastic material: :math:`\sigma = E\varepsilon`.

    Used only by these tests so the elastic-shortening check is exact
    and independent of the concrete parabola or the strand bilinear
    diagram (the strand also stays on its linear branch here).
    """

    def __init__(self, E, eps_lim=1.0):
        self.E = E
        self._lim = eps_lim

    @property
    def eps_min(self):
        return -self._lim

    @property
    def eps_max(self):
        return self._lim

    def stress(self, eps):
        return self.E * eps

    def stress_array(self, eps):
        return self.E * np.asarray(eps, dtype=float)

    def tangent(self, eps):
        return self.E

    def tangent_array(self, eps):
        return np.full_like(np.asarray(eps, dtype=float), self.E)


# Common transfer-state data.
B, H, N_Y = 300.0, 600.0, 300
EC, EP = 31000.0, 195000.0
AP = 1000.0
SIGMA_P0 = 1170.0           # jacking stress [MPa]; eps_p0 = 0.006 < eps_el


def _build(e):
    """Concrete + single tendon at eccentricity ``e`` from centroid."""
    conc = LinearElastic(EC)
    strand = LinearElastic(EP)
    eps_p0 = SIGMA_P0 / EP
    tendon = Tendon(y=H / 2.0 + e, Ap=AP, material=strand,
                    eps_pe=eps_p0, embedded=True)
    return RectSection(B, H, conc, [], n_fibers_y=N_Y, tendons=[tendon])


def _meshed_props(sec):
    """Gross area/inertia of the *meshed* section about its centroid."""
    Ac = float(np.sum(sec.A_fibers))
    yc = sec.y_centroid
    Ic = float(np.sum(sec.A_fibers * (sec.y_fibers - yc) ** 2))
    return Ac, Ic


@pytest.mark.parametrize("e", [0.0, -200.0, 150.0])
def test_elastic_shortening_loss_matches_closed_form(e):
    sec = _build(e)
    Ac, Ic = _meshed_props(sec)

    res = solve_pretension_transfer(sec, tol=1e-6, max_iter=300)
    cf = elastic_shortening_loss(EC, EP, Ac, Ic, AP, e, SIGMA_P0)

    assert res.converged
    # Released state is genuinely self-equilibrated: zero residual.
    # (A pure-axial chi=0 "solution" would leave a large moment here.)
    # Loss and post-transfer stress match the exact closed form.
    assert res.loss[0] == pytest.approx(cf["loss"], rel=1e-6)
    assert res.sigma_p[0] == pytest.approx(cf["sigma_p_after"], rel=1e-6)
    assert res.eps_sec[0] == pytest.approx(cf["eps_sec"], rel=1e-6)
    assert res.chi_x == pytest.approx(cf["chi"], abs=1e-12)


def test_abutment_force_equals_prestress_resultant():
    """Pre-transfer abutment reaction is the bare prestress resultant."""
    e = -200.0
    sec = _build(e)
    res = solve_pretension_transfer(sec)
    assert res.N_abutment == pytest.approx(SIGMA_P0 * AP, rel=1e-9)
    assert res.Mx_abutment == pytest.approx(SIGMA_P0 * AP * e, rel=1e-9)


def test_loss_increases_with_eccentricity():
    """Eccentric transfer loses more prestress than centroidal."""
    loss_c = solve_pretension_transfer(_build(0.0)).loss[0]
    loss_e = solve_pretension_transfer(_build(-200.0)).loss[0]
    assert loss_e > loss_c


def test_no_tendons_raises():
    conc = LinearElastic(EC)
    sec = RectSection(B, H, conc, [], n_fibers_y=50)
    with pytest.raises(ValueError):
        solve_pretension_transfer(sec)
