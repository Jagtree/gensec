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
"""
Validation of the prestress Phase 1 bonded-tendon initial-strain core.

These tests pin the hard correctness invariant of the per-element
initial-strain mechanism: the tendon evaluates its own constitutive law
at the offset total strain ``eps_section + eps_init`` while the displaced
bulk is evaluated at ``eps_section`` alone.

Checks
------
- Centroidal tendon, linear-elastic bulk: exact axial equilibrium.
- Eccentric tendon: ``Mx`` accumulation vs the closed form.
- Analytical 3x3 tangent vs central finite differences (hardening branch).
- Realistic section: ``solve_equilibrium`` + SLS fiber stresses, with the
  decoupled-argument invariant verified numerically.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from gensec.materials.base import Material
from gensec.materials.concrete import Concrete
from gensec.materials.steel import PrestressingSteel
from gensec.geometry.section import RectSection
from gensec.geometry.fiber import Tendon
from gensec.solver.integrator import FiberSolver


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

@dataclass
class LinElastic(Material):
    r"""
    Linear-elastic bulk material for closed-form comparison.

    A constant-modulus law :math:`\sigma = E\,\varepsilon` with no
    branch points, so the section response is exactly integrable by
    hand and any discrepancy with the solver is either a bug or a
    quantifiable mesh-discretization term.

    Parameters
    ----------
    E : float
        Elastic modulus [MPa].
    eps_min, eps_max : float
        Admissible strain range (wide, so limits never bind in these
        tests).
    """

    E: float = 30000.0
    eps_min: float = -1.0
    eps_max: float = 1.0

    def stress(self, e):
        return self.E * e

    def tangent(self, e):
        return self.E

    def stress_array(self, e):
        return self.E * np.asarray(e, float)

    def tangent_array(self, e):
        return np.full_like(np.asarray(e, float), self.E)


@pytest.fixture
def Ec():
    """Bulk modulus used across the linear-elastic checks [MPa]."""
    return 30000.0


@pytest.fixture
def Ep():
    """Prestressing-steel modulus [MPa]."""
    return 195000.0


# ---------------------------------------------------------------------------
#  1. Centroidal tendon: pure axial equilibrium, exact
# ---------------------------------------------------------------------------

def test_centroidal_axial_equilibrium(Ec, Ep):
    r"""
    A centroidal bonded tendon under a pure axial section strain.

    With a linear-elastic bulk the axial force is

    .. math::

        N = E_c\,\varepsilon_0\,A_c
            + \bigl[E_p(\varepsilon_0 + \varepsilon_{pe})
                    - E_c\,\varepsilon_0\bigr] A_p

    where the bulk integrates over the gross area and the bracketed
    term is the net embedded-tendon contribution (tendon law at
    :math:`\varepsilon_0 + \varepsilon_{pe}`, displaced bulk at
    :math:`\varepsilon_0`).  A centroidal tendon must produce no
    moment.
    """
    B, H, Ap, eps_pe = 300.0, 600.0, 1400.0, 0.0065
    bulk = LinElastic(E=Ec)
    ps = PrestressingSteel(f_p01d=1391.3, eps_ud=0.02, Ep=Ep)
    sec = RectSection(B=B, H=H, bulk_material=bulk, rebars=[],
                      n_fibers_y=200,
                      tendons=[Tendon(y=H / 2, x=B / 2, material=ps,
                                      Ap=Ap, eps_pe=eps_pe)])
    solver = FiberSolver(sec)

    eps0 = -3e-4
    N, Mx, My = solver.integrate(eps0, 0.0, 0.0)

    N_hand = (Ec * eps0 * (B * H)
              + (Ep * (eps0 + eps_pe) - Ec * eps0) * Ap)

    assert N == pytest.approx(N_hand, rel=1e-9)
    assert Mx == pytest.approx(0.0, abs=1e-3)
    assert My == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
#  2. Eccentric tendon: moment accumulation vs closed form
# ---------------------------------------------------------------------------

def test_eccentric_moment(Ec, Ep):
    r"""
    An eccentric tendon under combined axial strain and curvature.

    The moment about x is the bulk term plus the eccentric tendon term

    .. math::

        M_x = E_c\,\chi_x\,I_x
            + f_{\text{net}}\,(y_t - y_{\text{ref}})

    The tolerance on :math:`M_x` is looser than machine precision
    because the closed form uses the analytic :math:`I_x = B H^3/12`
    while the solver integrates a finite fiber mesh; the residual is
    the discretization of :math:`I_x`, not a formula error.
    """
    B, H, Ap, eps_pe = 300.0, 600.0, 1400.0, 0.0065
    bulk = LinElastic(E=Ec)
    ps = PrestressingSteel(f_p01d=1391.3, eps_ud=0.02, Ep=Ep)
    yt, xt = 100.0, B / 2
    sec = RectSection(B=B, H=H, bulk_material=bulk, rebars=[],
                      n_fibers_y=400,
                      tendons=[Tendon(y=yt, x=xt, material=ps,
                                      Ap=Ap, eps_pe=eps_pe)])
    solver = FiberSolver(sec)

    eps0, chi_x = -3e-4, 5e-7
    N, Mx, My = solver.integrate(eps0, chi_x, 0.0)

    ly = yt - solver.y_ref
    e_sec = eps0 + chi_x * ly
    f_net = (Ep * (e_sec + eps_pe) - Ec * e_sec) * Ap
    N_hand = Ec * eps0 * (B * H) + f_net
    Mx_hand = Ec * chi_x * (B * H ** 3 / 12) + f_net * ly

    assert N == pytest.approx(N_hand, rel=1e-9)
    assert Mx == pytest.approx(Mx_hand, rel=1e-4)
    assert My == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
#  3. Analytical tangent vs finite differences
# ---------------------------------------------------------------------------

def test_analytical_tangent_matches_fd(Ec, Ep):
    r"""
    The analytical 3x3 tangent stiffness vs central finite differences.

    The state is chosen so the tendon's total strain exceeds the proof
    strain, exercising the second-branch slope :math:`E_{\text{hard}}`
    in both the force and the tangent.  Agreement is measured by the
    Frobenius-norm relative error, which avoids the false alarms that a
    per-entry ratio produces on the structurally-zero off-diagonal
    couplings.
    """
    B, H = 300.0, 600.0
    bulk = LinElastic(E=Ec)
    ps = PrestressingSteel(f_p01d=1391.3, sigma_ud=1860.0,
                           eps_ud=0.0315, Ep=Ep)
    sec = RectSection(B=B, H=H, bulk_material=bulk, rebars=[],
                      n_fibers_y=120,
                      tendons=[Tendon(y=120.0, x=B / 2, material=ps,
                                      Ap=1400.0, eps_pe=0.0065)])
    solver = FiberSolver(sec)
    x0 = np.array([0.001, 1e-6, 2e-7])

    def F(x):
        return np.array(solver.integrate(*x))

    _, _, _, K = solver.integrate_with_tangent(*x0)

    h = [1e-7, 1e-10, 1e-10]
    Kfd = np.zeros((3, 3))
    for j in range(3):
        xp = x0.copy(); xp[j] += h[j]
        xm = x0.copy(); xm[j] -= h[j]
        Kfd[:, j] = (F(xp) - F(xm)) / (2 * h[j])

    rel = np.linalg.norm(K - Kfd) / np.linalg.norm(K)
    assert rel < 1e-8


# ---------------------------------------------------------------------------
#  4. End-to-end: equilibrium + SLS stresses + the decoupled invariant
# ---------------------------------------------------------------------------

def test_equilibrium_and_sls_invariant():
    r"""
    Solve a realistic prestressed section and check the SLS stresses.

    Verifies that:

    - ``solve_equilibrium`` converges with the tendon-aware tangent and
      reproduces the applied :math:`(N, M_x)`;
    - the reported tendon total strain equals
      :math:`\varepsilon_{\text{sec}} + \varepsilon_{\text{init}}`;
    - the net tendon stress equals
      :math:`\sigma_p(\varepsilon_{\text{tot}})
      - \sigma_c(\varepsilon_{\text{sec}})`, i.e. the displaced
      concrete is subtracted at the **section** strain — the hard
      correctness invariant.
    """
    c = Concrete(fck=40)
    ps = PrestressingSteel(f_p01d=1600 / 1.15, sigma_ud=1860.0,
                           eps_ud=0.0315, Ep=195000.0)
    B, H = 400.0, 800.0
    sec = RectSection(B=B, H=H, bulk_material=c, rebars=[],
                      n_fibers_y=160,
                      tendons=[Tendon(y=120.0, x=B / 2, material=ps,
                                      Ap=1400.0, eps_pe=0.0065)])
    solver = FiberSolver(sec)

    res = solver.solve_equilibrium(N_target=-1.5e6, Mx_target=300e6,
                                   My_target=0.0)
    assert res["converged"]

    eps0, chi_x, chi_y = res["eps0"], res["chi_x"], res["chi_y"]
    N, Mx, My = solver.integrate(eps0, chi_x, chi_y)
    assert N == pytest.approx(-1.5e6, abs=1.0)
    assert Mx == pytest.approx(300e6, abs=1.0)

    fr = solver.get_fiber_results(eps0, chi_x, chi_y)
    assert "tendons" in fr
    t = fr["tendons"]

    # eps_tot == eps_sec + eps_init
    assert t["eps_tot"][0] == pytest.approx(
        t["eps_sec"][0] + t["eps_init"][0], abs=1e-15)

    # sigma_net == sigma_p(eps_tot) - sigma_c(eps_sec)
    sig_c = c.stress(np.array([t["eps_sec"][0]]))[0]
    assert (t["sigma"][0] - sig_c) == pytest.approx(
        t["sigma_net"][0], abs=1e-6)


# ---------------------------------------------------------------------------
#  5. Tendon construction guards
# ---------------------------------------------------------------------------

def test_tendon_guards():
    """Phase 1 construction-time guards on the ``Tendon`` dataclass."""
    ps = PrestressingSteel(f_p01d=1391.3)

    # bonded=False is not a section element: an unbonded/external
    # prestressing force on hardened concrete is a load and must be
    # modelled as a PrestressAction in the demand layer, never as a
    # Tendon.  (A post-tension tendon that will be grouted is declared
    # bonded=True; its not-yet-grouted state is a per-stage mask in
    # SectionState, not a Tendon(bonded=False).)
    with pytest.raises(ValueError):
        Tendon(y=100.0, material=ps, Ap=1400.0, bonded=False)

    # Zero/absent area is an error.
    with pytest.raises(ValueError):
        Tendon(y=100.0, material=ps, Ap=0.0)

    # Area from strand count.
    t = Tendon(y=100.0, material=ps, n_strands=10, area_strand=140.0)
    assert t.Ap == pytest.approx(1400.0)
    # The element carries the effective prestrain as ``eps_pe`` (here the
    # default, since none was passed).  ``eps_init`` is the engine/bulk
    # generic-offset name (the section array ``eps_init_tendons``), not an
    # attribute of the element — the asymmetric naming is deliberate.
    assert t.eps_pe == pytest.approx(0.0)