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
Validation of the Phase-6 (unbonded / external) ULS action and the
strain-compatibility boundary guards.

Run from the project root (with the package installed / importable):

.. code-block:: console

    python run_prestress_uls_unbonded_validation.py

Every check is closed-form / hand-calc so it needs no reference tool:

1. **ULS force composition** of
   :meth:`PrestressAction.from_force_uls`:
   :math:`P_{\text{ULS}} = P_{\text{eff}} + \Delta\sigma_p\,A_p`, and its
   couple equals :meth:`PrestressAction.from_force` at that force.
2. **Design-strength cap** behaviour: verbatim below the cap, capped with
   a warning above it, never silently.
3. **EN 1992-1-1 §5.10.8(3) default** via :func:`delta_sigma_p_uls`
   (100 MPa recommended; override; unimplemented annex raises).
4. **Boundary guards**: non-finite action rejected; ``eps_pe`` /
   ``sectional_strain`` on a ``PrestressAction`` raise; and -- the
   *primary* enforcement -- ``Tendon(bonded=False)`` raises at
   construction (an unbonded/external tendon is not a section element).
5. **Demand-walk integration**: an unbonded ULS action sums into the
   staged cumulative demand exactly as the engines do.

Note: there is deliberately **no** FiberSolver-level "reject unbonded
tendon" check.  Sections expose tendons to the solver as arrays with no
per-tendon bond flag (``materialize_view`` has already excluded the
unbonded ones), so such a check would have no real hook -- the boundary is
enforced upstream at ``Tendon.__post_init__`` and by the view.
"""

import warnings

import numpy as np

from gensec.solver.section_state import PrestressAction
from gensec.materials.prestress_properties import delta_sigma_p_uls
from gensec.geometry.fiber import Tendon


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_uls_force_composition():
    print("[1] from_force_uls: P_ULS = P_eff + dsp*Ap, couple vs from_force")
    P_eff, Ap, dsp = 1.5e6, 1500.0, 100.0
    x, y, xr, yr = 200.0, 80.0, 150.0, 300.0

    a = PrestressAction.from_force_uls(
        P_eff, x, y, Ap=Ap, delta_sigma_p=dsp, x_ref=xr, y_ref=yr)

    P_uls = P_eff + dsp * Ap                    # 1.65e6 N
    assert approx(a.N, -P_uls), (a.N, -P_uls)
    assert approx(a.Mx, -P_uls * (y - yr))
    assert approx(a.My, +P_uls * (x - xr))

    b = PrestressAction.from_force(P_uls, x, y, x_ref=xr, y_ref=yr)
    assert approx(a.N, b.N) and approx(a.Mx, b.Mx) and approx(a.My, b.My)

    c = PrestressAction.from_force_uls(
        P_eff, x, y, Ap=Ap, delta_sigma_p=0.0, x_ref=xr, y_ref=yr)
    d = PrestressAction.from_force(P_eff, x, y, x_ref=xr, y_ref=yr)
    assert approx(c.N, d.N) and approx(c.Mx, d.Mx) and approx(c.My, d.My)
    print("    OK")


def test_design_cap():
    print("[2] f_pd cap: verbatim below, warns+caps above, never silent")
    Ap = 1500.0
    x, y = 100.0, 540.0
    P_eff, dsp = 1.5e6, 100.0                    # sigma_uls = 1100 MPa

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = PrestressAction.from_force_uls(
            P_eff, x, y, Ap=Ap, delta_sigma_p=dsp, sigma_pd_cap=1500.0)
    assert approx(a.N, -(P_eff + dsp * Ap))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b = PrestressAction.from_force_uls(
            P_eff, x, y, Ap=Ap, delta_sigma_p=dsp, sigma_pd_cap=1050.0)
    assert len(w) == 1 and issubclass(w[0].category, UserWarning)
    assert approx(b.N, -(1050.0 * Ap)), b.N
    print("    OK")


def test_delta_sigma_p_uls_default():
    print("[3] delta_sigma_p_uls: EC2=100 MPa, override, unknown NA raises")
    assert approx(delta_sigma_p_uls(), 100.0)
    assert approx(delta_sigma_p_uls(NA="EC2"), 100.0)
    assert approx(delta_sigma_p_uls(delta_override=137.0), 137.0)
    assert approx(delta_sigma_p_uls(NA="NTC18", delta_override=90.0), 90.0)
    raised = False
    try:
        delta_sigma_p_uls(NA="NTC18")
    except ValueError:
        raised = True
    assert raised, "unimplemented annex did not raise"
    print("    OK")


def test_input_guards():
    print("[4] from_force_uls input guards (Ap>0, dsp>=0)")
    for kw in (dict(Ap=0.0, delta_sigma_p=100.0),
               dict(Ap=-5.0, delta_sigma_p=100.0),
               dict(Ap=1500.0, delta_sigma_p=-1.0)):
        raised = False
        try:
            PrestressAction.from_force_uls(1.5e6, 100.0, 540.0, **kw)
        except ValueError:
            raised = True
        assert raised, f"guard did not fire for {kw}"
    print("    OK")


def test_boundary_guards():
    print("[5] boundary guards: non-finite / eps_pe / sectional_strain / Tendon")
    for bad in (np.inf, np.nan):
        raised = False
        try:
            PrestressAction(bad, 0.0, 0.0)
        except ValueError:
            raised = True
        assert raised, f"non-finite N={bad} not rejected"

    act = PrestressAction.from_force(1.4e6, 200.0, 80.0)
    for probe in (lambda: act.eps_pe, lambda: act.sectional_strain(0.0)):
        raised = False
        try:
            probe()
        except TypeError:
            raised = True
        assert raised, "strain-compatibility probe did not raise"

    # Primary enforcement: an unbonded Tendon is not constructible
    # (Ap>0 clears the area check first, then bonded=False raises).
    raised = False
    try:
        Tendon(y=60.0, Ap=150.0, material=object(), bonded=False)
    except ValueError:
        raised = True
    assert raised, "Tendon(bonded=False) did not raise"
    print("    OK")


def test_demand_walk_integration():
    r"""An unbonded ULS action sums into the cumulative demand exactly as
    ``check._check_staged`` / ``analysis._analyze_staged`` do."""
    print("[6] demand-walk integration of an unbonded ULS action")
    xr, yr = 150.0, 300.0
    P_eff, Ap, dsp = 1.5e6, 1500.0, 100.0
    x, y = 200.0, 80.0
    act = PrestressAction.from_force_uls(
        P_eff, x, y, Ap=Ap, delta_sigma_p=dsp, x_ref=xr, y_ref=yr)

    g = dict(N=-300e3, Mx=50e6, My=0.0)
    cum = [0.0, 0.0, 0.0]
    cum[0] += g["N"]; cum[1] += g["Mx"]; cum[2] += g["My"]     # stage 0
    aN, aMx, aMy = act.triple()
    cum[0] += aN; cum[1] += aMx; cum[2] += aMy                 # stage 1

    P_uls = P_eff + dsp * Ap
    assert approx(cum[0], -300e3 - P_uls)
    assert approx(cum[1], 50e6 - P_uls * (y - yr))
    assert approx(cum[2], 0.0 + P_uls * (x - xr))
    print("    OK")


if __name__ == "__main__":
    test_uls_force_composition()
    test_design_cap()
    test_delta_sigma_p_uls_default()
    test_input_guards()
    test_boundary_guards()
    test_demand_walk_integration()
    print("\nALL PHASE-6 UNBONDED/EXTERNAL VALIDATIONS PASSED")
