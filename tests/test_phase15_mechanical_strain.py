# SPDX-License-Identifier: AGPL-3.0-or-later
# GenSec — reinforced/prestressed concrete sectional analysis.
# Copyright (C) 2026  Andrea (GenSec project).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.  See <https://www.gnu.org/licenses/>.
r"""
Phase 15 C (F-M3) — a strain criterion must read the strain its own
constitutive law reads.

The integrator evaluates every bulk law at

.. math::

    \varepsilon^{\mathrm{mech}}_i = \varepsilon_{\mathrm{sec},i}
        + \varepsilon_{b,0}
        + \varepsilon_{0,z(i)}
        + \chi_{x,z(i)}\,(y_i - y_{\mathrm{ref}})
        - \chi_{y,z(i)}\,(x_i - x_{\mathrm{ref}}) ,

the section strain plus the locked-in imposed-strain field of the fibre's
own zone — ``bulk_eps_init`` plus the zone plane of ``bulk_planes``,
which on a staged section carries the casting datum and the viscous
eigenstrain of every interval.  A rebar's own law, by contrast, reads the
section strain alone: the imposed field of the concrete is a property of
the concrete, and the steel does not share it.

:meth:`~gensec.solver.integrator.FiberSolver.strains_within_limits`
already makes exactly that distinction, per group.  The curvature scans
of :mod:`gensec.solver.capacity` did not: they compared the bare
sectional strain against thresholds — cracking, crushing, ultimate —
that are properties of the material and are therefore stated against the
strain the material sees.  Two definitions of "the strain in this fibre"
inside one solver are a violation of *one fact, one home*, and the one
that is right is already written.

Order of magnitude, measured on the flagship: the topping datum reads
:math:`\varepsilon_0 = 1.81\cdot 10^{-4}`, about **twice**
:math:`\varepsilon_{cr}`.  Not rounding.

Every test below fails on the pre-Phase-15 tree.
"""
import os

import numpy as np
import pytest

from gensec.io_yaml import load_yaml
from gensec.solver.capacity import NMDiagram
from gensec.solver.integrator import FiberSolver
from gensec.solver.section_state import (StagedDomainManager,
                                         materialize_view)
from gensec.solver.timeline import ConstructionTimeline

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.join(_HERE, os.pardir, "examples")
_FLAGSHIP = os.path.join(_EXAMPLES, "example_composite_losses.yaml")


_TIMELINE = os.path.join(_EXAMPLES, "example_composite_timeline.yaml")


def _with_ec2_concrete(path):
    r"""
    Write a copy of *path* whose concretes are declared as
    ``concrete_ec2_gen1``, and return the new path.

    No shipped example puts a cracking-capable material on a staged
    section: the three files carrying a ``construction_history`` all
    declare ``type: concrete`` with an ``fck``, which attaches no EC2
    property object, so
    :meth:`~gensec.solver.capacity.NMDiagram._zone_cracking_strain`
    returns ``None`` for every zone and the marker never appears.  That
    gap is a fact about the example set, recorded in ``handoff/15_0``;
    it is not a reason to leave the cracking consumer of F-M3
    untested, so this fixture supplies the missing case itself.

    Parameters
    ----------
    path : str

    Returns
    -------
    str
        Path of the rewritten model, under the pytest temporary area.
    """
    import re
    import tempfile

    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    for name, cls in (("c_precast", "C45/55"), ("c_topping", "C30/37")):
        text = re.sub(
            # Every indented line of the block, not a fixed four: the
            # material declares its long-term reduction factor since
            # Phase 23, and a pattern that counted lines would leave
            # the tail of the old block orphaned under the new
            # inline mapping.
            rf"^  {name}:\n(?:    [^\n]*\n)+",
            f"  {name}: {{type: concrete_ec2_gen1, class: '{cls}', "
            f"ls: F, loadtype: slow, TypeConc: R, NA: French, "
            f"time: 28}}\n",
            text, flags=re.MULTILINE)
    fd, out = tempfile.mkstemp(suffix=".yaml", prefix="gensec_ec2_")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out


def _staged_view(path, ec2_materials=False):
    r"""
    Materialize the view of the last anchored stage of *path*.

    This is the object :func:`~gensec.solver.section_state.
    materialize_view` documents as *suitable for* ``NMDiagram(view)``:
    the public surface on which the defect is reachable.

    Returns
    -------
    GenericSection
        A view whose ``bulk_planes_active`` is non-zero.
    """
    if ec2_materials:
        path = _with_ec2_concrete(path)
    data = load_yaml(path)
    section = data["section"]
    ddb = {d["name"]: {"N": d.get("N", 0.0), "Mx": d.get("Mx", 0.0),
                       "My": d.get("My", 0.0)} for d in data["demands"]}
    tl = ConstructionTimeline.from_block(
        data["construction_history"],
        losses_models=data.get("losses_models"),
        zone_rheology=data.get("zone_rheology"))
    resolution = tl.resolve(section, ddb)
    combo = next(c for c in data["combinations"] if c.get("at") is not None)
    compiled = tl.compile_combination(combo, resolution, section, ddb)
    _point, stages, inactive = compiled[-1]
    mgr = StagedDomainManager(section, biaxial=False,
                              gen_kwargs={"n_points": 12})
    states, _h, _b, _d = mgr.resolve_stages(
        stages, initially_inactive=inactive)
    view = materialize_view(section, states[-1])
    planes = np.asarray(view.bulk_planes_active, dtype=float)
    assert np.any(planes), (
        "this fixture is worthless unless the view actually carries "
        "non-zero bulk planes")
    return view


@pytest.fixture(scope="module")
def flagship_view():
    r"""Materialized view of the flagship at its last anchor point."""
    return _staged_view(_FLAGSHIP)


# ======================================================================
#  1.  eps_min / eps_max must be the extremes the laws see
# ======================================================================

def test_reported_extreme_strains_are_the_ones_the_laws_read(
        flagship_view):
    r"""
    The ``eps_min`` / ``eps_max`` a scan reports must be computed from
    the mechanical strain of the concrete and the section strain of the
    steel — each fibre measured at the argument of its own law.

    The reference is rebuilt here from the solver's own offset accessor
    and the fibre coordinates, not read back from the scan.
    """
    solver = FiberSolver(flagship_view)
    mc = NMDiagram(solver).generate_moment_curvature(
        N_fixed=0.0, n_chi=60, direction="x")

    chis = np.asarray(mc["chi"], dtype=float)
    eps_min = np.asarray(mc["eps_min"], dtype=float)
    eps_max = np.asarray(mc["eps_max"], dtype=float)

    offset = solver._bulk_offset()
    assert np.any(np.asarray(offset)), "the view carries no bulk offset"

    # The scan solves eps0 per curvature step and does not report it, so
    # rebuild it from the reported extremes of a single reference step:
    # instead, recompute the whole field from the solver's own solve.
    checked = 0
    for k, chi in enumerate(chis):
        if not np.isfinite(eps_max[k]):
            continue
        eps0 = NMDiagram(solver)._solve_eps0_for_N(
            solver, 0.0, float(chi), 0.0, 0.0,
            solver.sec.bulk_material.eps_min)
        eb, er = solver.strain_field(eps0, float(chi), 0.0)
        pooled = np.concatenate([eb + offset, er]) if er.size else \
            (eb + offset)
        assert eps_max[k] == pytest.approx(float(pooled.max()), rel=1e-6,
                                           abs=1e-12), (
            f"step {k}: the scan reports eps_max={eps_max[k]:.6e} but the "
            f"laws see a maximum of {float(pooled.max()):.6e}")
        assert eps_min[k] == pytest.approx(float(pooled.min()), rel=1e-6,
                                           abs=1e-12), (
            f"step {k}: the scan reports eps_min={eps_min[k]:.6e} but the "
            f"laws see a minimum of {float(pooled.min()):.6e}")
        checked += 1
        if checked >= 5:
            break
    assert checked >= 3, "not enough converged steps to check"


# ======================================================================
#  2.  the ultimate must agree with strains_within_limits
# ======================================================================

def test_ultimate_and_extremes_move_when_the_offset_is_neutralised(
        flagship_view, monkeypatch):
    r"""
    Neutralising the imposed-strain offset must change the answer.

    A differential test, and the sharpest statement of F-M3 available:
    :meth:`~gensec.solver.integrator.FiberSolver.bulk_strain_offset` is
    used **only** by the criteria — the integration and the tangent read
    the private accessor — so forcing it to zero holds the equilibrium,
    the solve order and the warm starts exactly fixed and changes
    nothing but the strain the criteria interrogate.

    Before Phase 15 the criteria never consulted it, so the two runs
    were identical and this test failed.  That they now differ is the
    defect, measured: on a staged section the offset is of the order of
    :math:`5\cdot 10^{-4}`, and the reported extreme strains, the
    ultimate curvature and the ultimate moment all move.

    Re-solving the plane independently is deliberately avoided: the
    scan warm-starts its vectorized Newton, a cold solve lands on a
    different plane at high curvature, and the comparison would then be
    measuring the solver rather than the criterion.

    Re-anchored in Phase 16 (declared deviation, ``16_1`` §5).  The
    probe used to run at ``N_fixed = 0``, where -- before N-2 -- the
    ultimate was declared because the **pooled** maximum reached the
    limit of the passive steel, and the pool contained bulk fibres,
    which carry the offset.  With each class now tested against its own
    limits, that comparison is gone: at zero axial force the governing
    element is the rebar, whose law reads the bare section strain, so
    the offset legitimately cannot move the event and the assertion
    stopped discriminating.  The probe moves to a compression level at
    which **crushing** governs -- the bulk-side limit, the one the
    offset does reach.  F-M3 has not regressed; which element governs
    has changed, and a differential must be taken on a branch where the
    quantity under test can bind (the lesson of ``15_1`` §3, applied to
    a fixture that already existed).
    """
    solver = FiberSolver(flagship_view)
    offset = np.asarray(solver.bulk_strain_offset(), dtype=float)
    assert np.abs(offset).max() > 1e-5, "the view carries no offset"

    #: Compression high enough that the concrete crushing strain is the
    #: first limit reached.  Measured: at this level the reported
    #: ultimate moves from 8.305085e-06 to 6.949153e-06 when the offset
    #: is neutralised -- a 16 % shift, not a rounding difference.
    N_CRUSH = -4.0e6

    with_offset = NMDiagram(solver).generate_moment_curvature(
        N_fixed=N_CRUSH, n_chi=60, direction="x")

    monkeypatch.setattr(FiberSolver, "bulk_strain_offset",
                        lambda self, idx=None: 0.0)
    without = NMDiagram(FiberSolver(flagship_view)
                        ).generate_moment_curvature(
        N_fixed=N_CRUSH, n_chi=60, direction="x")

    # the strain planes must be the same: only the criteria changed
    assert np.allclose(np.asarray(with_offset["chi"], dtype=float),
                       np.asarray(without["chi"], dtype=float),
                       equal_nan=True)

    a = np.asarray(with_offset["eps_max"], dtype=float)
    b = np.asarray(without["eps_max"], dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    assert finite.any()
    assert np.abs(a[finite] - b[finite]).max() > 1e-6, (
        "the reported extreme strains do not depend on the imposed-strain "
        "field at all: the criteria are reading the sectional strain")

    assert with_offset.get("ultimate_chi_pos") is not None
    assert (with_offset["ultimate_chi_pos"]
            != without.get("ultimate_chi_pos")), (
        "the ultimate curvature is unchanged when the imposed-strain "
        "field is removed: the ultimate criterion is not reading the "
        "strain the constitutive laws read")


# ======================================================================
#  3.  cracking is detected against the mechanical strain
# ======================================================================

@pytest.fixture(scope="module")
def cracking_view():
    r"""Staged view of a composite whose concretes declare EC2 data."""
    return _staged_view(_TIMELINE, ec2_materials=True)


def test_cracking_marker_moves_when_the_offset_is_neutralised(
        cracking_view, monkeypatch):
    r"""
    The first-cracking marker must depend on the imposed-strain field.

    Same isolation as the ultimate test: only
    :meth:`~gensec.solver.integrator.FiberSolver.bulk_strain_offset` is
    neutralised, so the equilibrium and the solve order are untouched
    and the sole difference is the strain the cracking criterion reads.

    Cracking is the criterion with the least room for the error to hide:
    :math:`\varepsilon_{cr}` is of the order of :math:`10^{-4}` and so
    is the casting datum, so a criterion that ignores the datum can be
    wrong by a full threshold.  On the flagship the marker never appears
    at all — no shipped example puts a cracking-capable material on a
    staged section, a gap recorded in ``handoff/15_0`` — which is why
    this fixture builds its own model.
    """
    solver = FiberSolver(cracking_view)
    gen = NMDiagram(solver)
    offset = np.asarray(solver.bulk_strain_offset(), dtype=float)
    assert np.abs(offset).max() > 1e-6, "the view carries no offset"

    eps_cr = gen._cracking_strain_field(solver.sec)
    assert eps_cr is not None, (
        "this fixture is worthless unless the materials declare a "
        "cracking threshold")

    with_offset = gen.generate_moment_curvature(N_fixed=-2000.0e3,
                                                n_chi=200, direction="x")
    monkeypatch.setattr(FiberSolver, "bulk_strain_offset",
                        lambda self, idx=None: 0.0)
    without = NMDiagram(FiberSolver(cracking_view)
                        ).generate_moment_curvature(N_fixed=-2000.0e3,
                                                    n_chi=200,
                                                    direction="x")

    a = with_offset.get("cracking_chi_pos")
    b = without.get("cracking_chi_pos")
    assert a is not None, (
        "no cracking marker even with the offset: "
        f"{(with_offset.get('diagnostics_pos') or {}).get('cracking_reason')}")
    assert a != b, (
        f"first cracking is reported at chi={a:.6e} whether or not the "
        f"locked-in imposed strain is counted: the cracking criterion "
        f"is reading the sectional strain")


# ======================================================================
#  4.  the force decomposition must reproduce the integrator
# ======================================================================

@pytest.fixture(scope="module")
def timeline_view():
    r"""Staged view of the tendon-free composite timeline example."""
    return _staged_view(_TIMELINE)


def test_component_decomposition_reproduces_the_integrator(timeline_view):
    r"""
    ``AnalysisEngine._decompose`` splits :math:`N, M_x, M_y` by material
    component.  A split that does not sum back to the total is not a
    split: it evaluated the constitutive laws at a different argument
    from the one the integrator used.

    Measured on the **tendon-free** composite, deliberately: the
    decomposition enumerates bulk zones and rebar groups and has no
    tendon branch at all, so on a prestressed view it cannot close for a
    reason that has nothing to do with F-M3.  That omission is a defect
    in its own right, registered in ``handoff/15_0`` with its
    measurement rather than repaired inside a strain-argument fix.
    """
    from gensec.solver.analysis import AnalysisEngine

    solver = FiberSolver(timeline_view)
    # The plane lives on the topping, so the probe must land the
    # topping on the RISING branch of its law.  A concrete law returns
    # zero throughout the tension side and a constant on the plateau,
    # and at either of those a wrong argument gives the right stress by
    # accident: measured on this view, (-1e-4, +2e-6) and (-8e-4,
    # -6e-6) both close to the last digit even with the defect present.
    # This pair does not.
    eps0, chi_x, chi_y = -1.0e-4, -2.0e-6, 0.0
    N, Mx, My = solver.integrate(eps0, chi_x, chi_y)

    engine = AnalysisEngine(solver)
    parts = engine._decompose(eps0, chi_x, chi_y)
    N_sum = sum(float(p["N"]) for p in parts)
    Mx_sum = sum(float(p["Mx"]) for p in parts)
    My_sum = sum(float(p["My"]) for p in parts)

    scale = max(abs(N), 1.0)
    assert N_sum == pytest.approx(N, rel=1e-9, abs=1e-6 * scale), (
        f"components sum to N={N_sum:.6e} but the integrator reads "
        f"N={N:.6e}")
    scale_m = max(abs(Mx), 1.0)
    assert Mx_sum == pytest.approx(Mx, rel=1e-9, abs=1e-6 * scale_m)
    assert My_sum == pytest.approx(My, rel=1e-9,
                                   abs=1e-6 * max(abs(My), 1.0))


if __name__ == "__main__":         # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
