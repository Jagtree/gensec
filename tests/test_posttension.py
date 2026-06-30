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
Phase-4 prestress: sequential bonded post-tensioning driver
(:mod:`gensec.solver.posttension`).

The fiber driver is regression-checked against the closed-form
transformed-section reference
:func:`gensec.solver.prestress_transfer.sequential_posttension_loss`.  A
**linear-elastic** bulk is used deliberately: the fiber model is then
algebraically identical to the planar transformed-section solve, so the
per-tendon debit profiles must coincide to machine precision once the
reference uses the section's **meshed** ``Ac, Sc, Ic`` (isolating the
solver from the :math:`\mathcal{O}(1/n_y^2)` strip-inertia
discretization).

Grouting tests exercise the action→element transition: the reconciled
``eps_init`` baked into the resistance must reproduce the effective
post-loss strand stress at the grouting strain datum, and the
single-side invariant must hold by construction.
"""

import numpy as np
import pytest

from gensec.geometry.section import RectSection
from gensec.geometry.fiber import Tendon
from gensec.solver.integrator import FiberSolver
from gensec.solver.prestress_transfer import sequential_posttension_loss
from gensec.solver.posttension import (
    solve_posttension_sequence,
    grout,
)


# ---------------------------------------------------------------------------
#  Minimal linear-elastic material (sigma = E * eps), used so the
#  fiber/closed-form comparison is exact and the strand stays linear.
# ---------------------------------------------------------------------------

class LinearElastic:
    r"""Constant-modulus law :math:`\sigma = E\,\varepsilon`, no branches."""

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


# ---------------------------------------------------------------------------
#  Section fixtures
# ---------------------------------------------------------------------------

B, H, N_Y = 300.0, 700.0, 350
EC, EP = 31000.0, 195000.0
AP = 150.0
SP0 = 1395.0


def _section(tendon_ys, x=None):
    r"""RC-free rectangular section with tendons at the given y's."""
    x = B / 2 if x is None else x
    bulk = LinearElastic(EC)
    ps = LinearElastic(EP)
    tendons = [Tendon(y=float(yt), x=float(x), material=ps, Ap=AP,
                      eps_pe=0.0)
               for yt in tendon_ys]
    return RectSection(B=B, H=H, bulk_material=bulk, rebars=[],
                       n_fibers_y=N_Y, tendons=tendons)


def _meshed_props(section, y_ref):
    r"""Meshed ``Ac, Sc, Ic`` about *y_ref* from the bulk fiber arrays."""
    A = np.asarray(section.A_fibers, dtype=float)
    y = np.asarray(section.y_fibers, dtype=float)
    d = y - y_ref
    Ac = float(np.sum(A))
    Sc = float(np.sum(A * d))
    Ic = float(np.sum(A * d * d))
    return Ac, Sc, Ic


def _closed_form(section, sigma_p0, order, scheme, base_N=0.0, base_M=0.0):
    r"""Run the closed-form reference about the section centroid."""
    y_ref = section.y_centroid
    Ac, Sc, Ic = _meshed_props(section, y_ref)
    tendons = [{"y": float(yt), "Ap": AP, "sigma_p0": float(sp)}
               for yt, sp in zip(section.y_tendons, sigma_p0)]
    return sequential_posttension_loss(
        EC, 0.0, EP, Ac=Ac, Sc=Sc, Ic=Ic,
        tendons=tendons, order=order,
        base_N=base_N, base_M=base_M,
        y_ref=y_ref, scheme=scheme)


# ---------------------------------------------------------------------------
#  1. Regression vs closed form — one_pass, centroidal
# ---------------------------------------------------------------------------

def test_onepass_centroidal_matches_closed_form():
    sec = _section([H / 2] * 4)
    sp0 = [SP0] * 4
    res = solve_posttension_sequence(sec, sigma_p0=sp0, Ep=EP,
                                     scheme="one_pass")
    ref = _closed_form(sec, sp0, res.order, "one_pass")
    assert res.converged
    assert np.allclose(res.loss, ref["loss"], atol=1e-6)
    assert np.allclose(res.eps_ref_grout, ref["eps_ref_grout"], atol=1e-12)
    # textbook profile: last stressed loses nothing
    assert res.loss[res.order[-1]] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
#  2. Order dependence and eccentric (negative debit)
# ---------------------------------------------------------------------------

def test_order_dependence_matches_closed_form():
    sec = _section([H / 2] * 4)
    sp0 = [SP0] * 4
    fwd = solve_posttension_sequence(sec, sigma_p0=sp0, Ep=EP,
                                     order=[0, 1, 2, 3])
    rev = solve_posttension_sequence(sec, sigma_p0=sp0, Ep=EP,
                                     order=[3, 2, 1, 0])
    assert np.allclose(fwd.loss, rev.loss[::-1], atol=1e-6)


def test_eccentric_negative_debit_matches_closed_form():
    e = 250.0
    sec = _section([H / 2 - e, H / 2 + e])
    sp0 = [SP0, SP0]
    res = solve_posttension_sequence(sec, sigma_p0=sp0, Ep=EP,
                                     order=[1, 0], scheme="one_pass")
    ref = _closed_form(sec, sp0, [1, 0], "one_pass")
    assert np.allclose(res.loss, ref["loss"], atol=1e-5)
    # stressing the second-stressed tendon raises the first's stress:
    # exactly one debit is negative.
    assert np.min(res.loss) < 0.0


# ---------------------------------------------------------------------------
#  3. Coupled scheme: matches closed form and is <= one_pass
# ---------------------------------------------------------------------------

def test_coupled_matches_closed_form_and_le_onepass():
    sec = _section([H / 2] * 4)
    sp0 = [SP0] * 4
    op = solve_posttension_sequence(sec, sigma_p0=sp0, Ep=EP,
                                    scheme="one_pass")
    cp = solve_posttension_sequence(sec, sigma_p0=sp0, Ep=EP,
                                    scheme="coupled")
    ref = _closed_form(sec, sp0, cp.order, "coupled")
    assert np.allclose(cp.loss, ref["loss"], atol=1e-5)
    assert cp.loss.mean() <= op.loss.mean() + 1e-9
    assert cp.coupled_iterations >= 2


# ---------------------------------------------------------------------------
#  4. Base sollecitazione is debit-free
# ---------------------------------------------------------------------------

def test_base_load_is_debit_free():
    e = 250.0
    sec = _section([H / 2 - e, H / 2 + e])
    sp0 = [SP0, SP0]
    no_base = solve_posttension_sequence(sec, sigma_p0=sp0, Ep=EP,
                                         order=[0, 1])
    with_base = solve_posttension_sequence(
        sec, sigma_p0=sp0, Ep=EP, order=[0, 1],
        base_N=-2.0e5, base_Mx=5.0e7)
    # The loss must be identical; only eps_ref_grout shifts by the base
    # strain (which the grouting reconciliation absorbs).
    assert np.allclose(no_base.loss, with_base.loss, atol=1e-6)
    assert not np.allclose(no_base.eps_ref_grout, with_base.eps_ref_grout)


# ---------------------------------------------------------------------------
#  5. Grouting: reconciliation reproduces the effective stress
# ---------------------------------------------------------------------------

def test_grout_reconciliation_reproduces_effective_stress():
    sec = _section([H / 2])           # single centroidal tendon
    sp0 = [SP0]
    res = solve_posttension_sequence(sec, sigma_p0=sp0, Ep=EP)
    g = grout(sec, res)

    # Algebraic identity: eps_init + eps_ref_grout == sigma_after / Ep
    rec = g.report["reconciliation"][0]
    assert (rec["eps_init"] + rec["eps_ref_grout"]) == pytest.approx(
        res.sigma_p_after[0] / EP, abs=1e-15)

    # End-to-end through the materialized view: at the grouting plane
    # (centroidal tendon -> eps0 = eps_ref_grout, chi = 0) the strand
    # stress reproduces sigma_p_after.
    from gensec.solver.section_state import materialize_view
    view = materialize_view(sec, g.state)
    solver = FiberSolver(view)
    eps0 = float(res.eps_ref_grout[0])
    fr = solver.get_fiber_results(eps0, 0.0, 0.0)
    assert "tendons" in fr
    assert fr["tendons"]["sigma"][0] == pytest.approx(
        res.sigma_p_after[0], rel=1e-9)


# ---------------------------------------------------------------------------
#  6. Single-side invariant and capacity-hash change on grouting
# ---------------------------------------------------------------------------

def test_grout_single_side_invariant_and_hash_change():
    sec = _section([H / 2 - 200.0, H / 2 + 200.0])
    sp0 = [SP0, SP0]
    res = solve_posttension_sequence(sec, sigma_p0=sp0, Ep=EP)

    # Grout only tendon 0; tendon 1 stays a load.
    g = grout(sec, res, indices=[0])
    assert g.grouted == [0]
    assert len(g.dropped_actions) == 1
    assert len(g.residual_actions) == 1
    # bonded mask: rebar-free section, so union index 0 is tendon 0.
    assert bool(g.state.bonded[0]) is True
    assert bool(g.state.bonded[1]) is False

    # Hash differs from the all-loads stressing state -> domain rebuild.
    from gensec.solver.section_state import (
        StagedDomainManager, geometry_signature)
    mgr = StagedDomainManager(sec, biaxial=False,
                              gen_kwargs={"n_angles": 4, "n_points": 8})
    h_grout = mgr.hash_of(g.state)
    h_init = mgr.hash_of(mgr.initial_state())
    assert h_grout != h_init


# ---------------------------------------------------------------------------
#  7. Reconciliation linear-branch guard raises
# ---------------------------------------------------------------------------

def test_grout_linear_branch_guard_raises():
    sec = _section([H / 2])
    sp0 = [SP0]
    res = solve_posttension_sequence(sec, sigma_p0=sp0, Ep=EP)
    big = float(abs(res.eps_ref_grout[0]))
    # A limit below the actual grouting strain must raise.
    with pytest.raises(ValueError):
        grout(sec, res, eps_c_linear_limit=big / 2.0)
    # A generous limit must pass.
    g = grout(sec, res, eps_c_linear_limit=big * 2.0)
    assert g.report["reconciliation"][0]["beyond_linear"] is False


# ---------------------------------------------------------------------------
#  8. Input guards
# ---------------------------------------------------------------------------

def test_input_guards():
    sec = _section([H / 2, H / 2])
    with pytest.raises(ValueError):
        solve_posttension_sequence(sec, sigma_p0=[SP0], Ep=EP)        # wrong length
    with pytest.raises(ValueError):
        solve_posttension_sequence(sec, sigma_p0=[SP0, SP0], Ep=EP,
                                   order=[0, 0])                       # not a permutation
    with pytest.raises(ValueError):
        solve_posttension_sequence(sec, sigma_p0=[SP0, SP0], Ep=EP,
                                   scheme="bogus")
    # No tendons.
    bare = RectSection(B=B, H=H, bulk_material=LinearElastic(EC),
                       rebars=[], n_fibers_y=50)
    with pytest.raises(ValueError):
        solve_posttension_sequence(bare, sigma_p0=[], Ep=EP)

def test_intra_sequence_guards_raise():
    sec = _section([H / 2, H / 2 + 150.0, H / 2 - 150.0])   # 3 tendons
    sp0 = [SP0, SP0, SP0]
    # subset order -> NotImplementedError (not a generic ValueError)
    import pytest
    with pytest.raises(NotImplementedError):
        solve_posttension_sequence(sec, sigma_p0=sp0, Ep=EP, order=[0, 1])
    # declared pre-bonded tendon -> NotImplementedError
    with pytest.raises(NotImplementedError):
        solve_posttension_sequence(sec, sigma_p0=sp0, Ep=EP,
                                   already_bonded=[0])
    # out-of-range already_bonded -> ValueError
    with pytest.raises(ValueError):
        solve_posttension_sequence(sec, sigma_p0=sp0, Ep=EP,
                                    already_bonded=[9])


if __name__ == "__main__":          # pragma: no cover
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
