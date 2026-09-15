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
Phase 16 — the tendon, downstream of the solver.

Two consumers enumerate the section's element classes and stop at two of
the three.  They are not two defects of the same site: they are one
class of element invisible to two different consumers.

N-1 — :meth:`AnalysisEngine._decompose`
    Walks ``_bulk_groups`` and ``_rebar_groups``.  There is no branch
    for ``_tendon_groups``.  On a prestressed view the components
    therefore do not sum back to what :meth:`FiberSolver.integrate`
    returns, and a decomposition that does not reproduce what it
    decomposes is not one.

N-2 — the ultimate criterion of the curvature scans
    Pools the bulk and rebar strains into one array and compares its
    minimum against the **concrete** crushing limit and its maximum
    against the **pool's** maximum limit, which on a reinforced section
    is the steel's.  The tendon strain never enters.  A tendon carries
    :math:`\varepsilon_{pe}` on top of the section strain — of the order
    of :math:`7\times10^{-3}` — so it is the element **closest to its
    own limit** at every curvature, and it is the one nobody asks.

Method note
-----------
Both fixtures were run on the Phase-15 tree rebuilt from
``gensec_phase15.bundle`` and **seen to fail** before the repair was
written.  The probe planes of the decomposition test are the ones
``15_1`` §3 measured as discriminating: a point on the tension branch or
on the plateau of the concrete law gives the right stress from the wrong
argument and would pass with the defect present.

The ultimate test does not re-solve the scan's planes cold — ``15_1`` §3
measured that a cold re-solve at high curvature lands on a different
plane and would measure the solver rather than the criterion.  It reads
the tendon's own limit and the curvature the scan itself reports.

Two candidate fixtures for N-2 were written and **discarded for not
discriminating**, and the reasons are recorded because they generalise:

* *"the reported ultimate is not past the tendon's limit"* passed on the
  Phase-15 tree.  The tendon sits at :math:`l_y = -550` mm, so under
  sagging its total strain **falls** below :math:`\varepsilon_{pe}`
  (5.63e-3 at the sagging ultimate against a datum of 6.70e-3): on that
  branch no tendon limit can bind, with or without the defect.
* *"tightening the strand moves the reported ultimate"* also passed on
  the Phase-15 tree — for the wrong reason.  ``PrestressingSteel``
  derives its hardening modulus from ``eps_ud``, so tightening the limit
  also stiffens the **constitutive law**, which moves the equilibrium
  and hence the scan, whether or not the criterion ever reads the
  tendon.  A differential is only evidence when the thing it moves is
  the only thing that moves.
"""

import copy
import unittest

import numpy as np
import pytest

from gensec.io_yaml import load_yaml
from gensec.solver.analysis import AnalysisEngine
from gensec.solver.capacity import NMDiagram
from gensec.solver.integrator import FiberSolver
from gensec.solver.section_state import (StagedDomainManager,
                                         materialize_view)
from gensec.solver.timeline import ConstructionTimeline

_FLAGSHIP = "examples/example_composite_losses.yaml"


def _eps0_at(solver, mc, chi):
    r"""
    The axial strain of the scan's own plane at curvature *chi*.

    Recovered from the extreme strains the scan returns rather than by
    re-solving: ``15_1`` §3 measured that a cold re-solve at high
    curvature lands on a different plane than the vectorised scan's warm
    start, so a fixture built on one would be measuring the solver
    instead of the criterion.

    With ``eps_min`` and ``eps_max`` recorded at the extreme bulk
    ordinates the plane is determined, and the inversion is exact for a
    linear strain field.

    Parameters
    ----------
    solver : FiberSolver
    mc : dict
        The dictionary ``generate_moment_curvature`` returns.
    chi : float
        The curvature to read the plane at [1/mm].

    Returns
    -------
    float
        :math:`\varepsilon_0` of the scan's plane at *chi*.
    """
    chis = np.asarray(mc["chi"], dtype=float)
    k = int(np.nanargmin(np.abs(chis - chi)))
    e_max = float(np.asarray(mc["eps_max"], dtype=float)[k])
    ly = np.asarray(solver._ly_bulk, dtype=float)
    off = np.asarray(solver.bulk_strain_offset(), dtype=float)
    # eps(y) = eps0 + chi*ly + offset ; the recorded maximum is its max.
    return e_max - float(np.max(chis[k] * ly + off))


def _staged_view(path=_FLAGSHIP, point=None):
    r"""
    Materialize the construction view of an anchored combination point.

    Parameters
    ----------
    path : str
        Example YAML carrying a ``construction_history``.
    point : str, optional
        Which anchored point.  The last compiled one when ``None``.

    Returns
    -------
    GenericSection
        The materialized view, carrying the bonded tendon at its
        reconciled locked-in strain.
    """
    doc = load_yaml(path)
    section = doc["section"]
    ddb = {d["name"]: {"N": d.get("N", 0.0), "Mx": d.get("Mx", 0.0),
                       "My": d.get("My", 0.0)}
           for d in (doc.get("demands") or [])}
    tl = ConstructionTimeline.from_block(
        doc["construction_history"],
        losses_models=doc.get("losses_models"),
        zone_rheology=doc.get("zone_rheology"))
    resolution = tl.resolve(section, ddb)
    combo = [c for c in doc["combinations"] if c.get("at") is not None][0]
    compiled = tl.compile_combination(combo, resolution, section, ddb)
    chosen = compiled[-1]
    if point is not None:
        chosen = next(c for c in compiled if c[0] == point)
    _pt, stages, inactive = chosen
    mgr = StagedDomainManager(section, biaxial=False,
                              gen_kwargs={"n_points": 12})
    states, _h, _b, _d = mgr.resolve_stages(
        stages, initially_inactive=inactive)
    return materialize_view(section, states[-1])


class TestDecompositionRecomposes(unittest.TestCase):
    r"""N-1: the components sum back to the resultant they decompose."""

    #: Off the tension branch and off the plateau of the concrete law
    #: (``15_1`` §3): on either, a wrong argument gives the right
    #: stress by accident and the probe stops discriminating.
    PLANES = ((-1.0e-4, -2.0e-6, 0.0), (-6.0e-4, -8.0e-6, 0.0))

    def _closure(self, view):
        solver = FiberSolver(view)
        engine = AnalysisEngine(solver)
        out = []
        for eps0, chi_x, chi_y in self.PLANES:
            comps = engine._decompose(eps0, chi_x, chi_y)
            total = solver.integrate(eps0, chi_x, chi_y)
            summed = tuple(
                float(sum(float(c.get(k, 0.0)) for c in comps))
                for k in ("N", "Mx", "My"))
            out.append((total, summed, comps))
        return out

    def test_prestressed_view_recomposes(self):
        r"""
        On a view carrying a bonded tendon the decomposition closes.

        The tolerance is relative to the *resultant's own scale*, not
        absolute: the residual measured before the repair was
        :math:`+2.79\times10^{6}` N and :math:`-1.54\times10^{9}`
        N·mm, four orders of magnitude above any tolerance a converged
        integration needs.
        """
        view = _staged_view()
        for (total, summed, comps) in self._closure(view):
            scale = max(1.0, max(abs(v) for v in total))
            for name, t, s in zip(("N", "Mx", "My"), total, summed):
                assert abs(t - s) <= 1e-6 * scale, (
                    f"{name}: integrate gives {t:.6e} but the components "
                    f"sum to {s:.6e} (residual {t - s:+.6e}); component "
                    f"types present: "
                    f"{sorted({str(c.get('type')) for c in comps})}"
                )

    def test_a_tendon_component_is_emitted(self):
        r"""
        The tendon appears as a component in its own right.

        Stated separately from the closure: an implementation could
        close the sum by folding the tendon force into the bulk or the
        rebar entry, which would balance the arithmetic and destroy the
        thing a decomposition exists to report.
        """
        view = _staged_view()
        engine = AnalysisEngine(FiberSolver(view))
        comps = engine._decompose(*self.PLANES[0])
        types = sorted({str(c.get("type")) for c in comps})
        assert "tendon" in types, (
            f"no tendon component on a view with "
            f"{len(view.tendons)} tendon(s); types emitted: {types}"
        )

    def test_tendon_free_section_is_unchanged(self):
        r"""
        A section without tendons closes exactly, before and after.

        The control: it is what makes the previous two tests evidence of
        an omitted element class rather than of a broken integration.
        """
        doc = load_yaml("examples/example_input.yaml")
        solver = FiberSolver(doc["section"])
        engine = AnalysisEngine(solver)
        for eps0, chi_x, chi_y in self.PLANES:
            comps = engine._decompose(eps0, chi_x, chi_y)
            total = solver.integrate(eps0, chi_x, chi_y)
            for name, t, k in zip(("N", "Mx", "My"), total,
                                  ("N", "Mx", "My")):
                s = float(sum(float(c.get(k, 0.0)) for c in comps))
                assert t == pytest.approx(s, abs=1e-6 * max(1.0, abs(t))), (
                    f"{name} residual {t - s:+.6e} on a tendon-free section"
                )


class TestUltimateCriterionSeesTheTendon(unittest.TestCase):
    r"""N-2: no element is past its own limit at the reported ultimate."""

    #: A strand limit tight enough that the tendon reaches it before the
    #: concrete crushes and before the passive steel reaches its own
    #: limit.  The distributed strand (``eps_max`` = 3.15e-2) is never
    #: the governing element on this section, so with it the fixture
    #: would pass with the defect present -- and a fixture that passes
    #: before and after is not a fixture (``15_1`` §3).
    EPS_UD = 1.0e-2

    def _view_with_tight_strand(self):
        r"""
        The flagship view with the strand's ``eps_ud`` tightened.

        Rebuilt with :func:`dataclasses.replace` rather than mutated:
        ``PrestressingSteel`` derives its hardening modulus from
        ``eps_ud`` in ``__post_init__``, so assigning the field in place
        would leave a stale ``E_hard`` and the fixture would be
        measuring an inconsistent material instead of a tighter limit.
        """
        import dataclasses
        view = copy.deepcopy(_staged_view())
        tendons = []
        for t in view.tendons:
            mat = dataclasses.replace(t.material, eps_ud=self.EPS_UD)
            tendons.append(dataclasses.replace(t, material=mat))
        view.tendons = tendons
        solver = FiberSolver(view)
        return view, solver

    def test_the_bound_ultimate_is_where_the_tendon_crosses(self):
        r"""
        When the strand governs, the reported ultimate is the first grid
        step at which the tendon's total strain crosses its own limit.

        Names the event instead of pinning the number: the assertion is
        recomputed from the scan's own planes, so it follows the grid
        and the section rather than a value copied out of a run.
        """
        _view, solver = self._view_with_tight_strand()
        mc = NMDiagram(solver).generate_moment_curvature(
            N_fixed=0.0, n_chi=200, direction="x")
        chi_u = mc.get("ultimate_chi_neg")
        assert chi_u is not None, (
            f"no hogging ultimate with a strand limit of "
            f"{self.EPS_UD:.3e}; diagnostics: {mc.get('diagnostics_neg')}"
        )
        chis = np.asarray(mc["chi"], dtype=float)
        band = self.EPS_UD * 0.99
        crossing = None
        for k in np.argsort(np.abs(chis)):
            if chis[k] >= 0 or not np.isfinite(chis[k]):
                continue
            eps0 = _eps0_at(solver, mc, float(chis[k]))
            et = (solver._tendon_section_strain(eps0, float(chis[k]), 0.0)
                  + solver._eps_init_tendon)
            if float(np.max(et)) >= band:
                crossing = float(chis[k])
                break
        assert crossing is not None, "the tendon never crosses its limit"
        assert chi_u == pytest.approx(crossing, rel=1e-9), (
            f"the scan reports the hogging ultimate at {chi_u:.6e} but "
            f"the tendon first crosses {band:.6e} at {crossing:.6e}"
        )


if __name__ == "__main__":                               # pragma: no cover
    unittest.main()
