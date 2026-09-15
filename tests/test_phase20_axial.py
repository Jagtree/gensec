# SPDX-License-Identifier: AGPL-3.0-or-later
# GenSec — reinforced/prestressed concrete sectional analysis.
# Copyright (C) 2026  Andrea (GenSec project).
r"""
Phase 20 — N-17: the pure-axial corridor that had no reason.

What Phase 19 left open, and what Phase 20 measured
---------------------------------------------------
Phase 19 recorded that ``solve_equilibrium`` returns
``converged = False`` when asked for a pure-tension state close to a
strand-only section's axial capacity, and left the item open because it
had **not** measured whether the state exists.  Phase 20 measured that,
and the answer decided the shape of the closure:

* the largest axial force the Phase 19 slab carries with
  :math:`|M_x| \le 10^3` N.mm is **317 016 N**, measured over 121
  curvatures with 427 verified :math:`M_x = 0` roots;
* the interrogated target is **1 056 673.5 N**, which misses that
  reach by 70.00 %;
* on the locus :math:`N = N_{target}` the smallest attainable
  :math:`|M_x|` is :math:`6.340041 \times 10^7` N.mm — a *plateau*, not
  a minimum, because at that load the concrete is wholly cracked, the
  strand carries everything and :math:`M_x = T e` exactly.

So the state does not exist.  N-17 is **not** a search defect, and the
closure is a refusal that names its reason rather than a repair.

The three kinds of fixture, kept apart
--------------------------------------
Following ``19_1 §4`` and ``18_1 §4``:

``TestTheStateDoesNotExist``
    **Guard.**  Green on both trees.  It is the *measurement* — an
    independent scan built from the section through ``integrate`` and
    ``integrate_batch``, both preexisting — and it is what makes the
    refusal legitimate rather than convenient.  A refusal justified by
    the routine that refuses would be circular.

``TestTheRefusalNamesItsReason``
    **Discriminating.**  Red on the pristine Phase 19 tree, on the
    behaviour and not on a name: every key added in this phase is read
    through ``dict.get``, so an absent key yields ``None`` and an
    :class:`AssertionError`, never an :class:`AttributeError`.  No
    symbol added in this phase is imported at module scope.

``TestTheStrictFormIsOptIn``
    **Contract, not evidence.**  Names symbols introduced in this
    phase, so on the pristine tree it would fail with
    :class:`AttributeError` — which proves a name is missing, not that
    a defect is present.  Declared as such so the census of reds in
    ``handoff/20_1`` stays honest.

Where the geometry comes from
-----------------------------
From ``tests/test_phase19_hull.py``, imported.  The slab has **one**
home and it is the one Phase 19 measured on; the distributed example
``examples/example_strand_only_slab.yaml`` carries the same geometry
and says so in its header.  Importing rather than restating is what
keeps a number recorded against a model from drifting away from the
model (``19_0 §8``).  It also keeps this module loadable on the
pristine tree, where the distributed example does not yet exist: a red
raised at import time proves an absent file, not an absent behaviour.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from gensec.io_yaml import load_yaml
from gensec.solver.integrator import FiberSolver

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:                    # same idiom as test_properties
    sys.path.insert(0, _HERE)

from test_phase19_hull import _SLAB_YAML     # noqa: E402  (path set above)

#: Moment tolerance the reach is measured under [N.mm].  Equal to
#: ``solve_equilibrium``'s own ``tol * 1000`` at ``tol = 1.0``, so this
#: file and the solver agree on what "zero moment" means.
_M_TOL = 1.0e3

#: The distributed example Phase 20 adds.  Named, not loaded at import:
#: its absence must read as a failed assertion in one declared test,
#: not as a collection error in all of them.
_EXAMPLE = os.path.join(_ROOT, "examples", "example_strand_only_slab.yaml")


def _slab(tmp_path):
    r"""
    The strand-only slab, from its single home in Phase 19.

    Parameters
    ----------
    tmp_path : pathlib.Path

    Returns
    -------
    GenericSection
    """
    path = tmp_path / "slab.yaml"
    path.write_text(_SLAB_YAML, encoding="utf-8")
    return load_yaml(str(path))["section"]


def _true_axial_reach(solver, n=4000):
    r"""
    Peak of :math:`N(\varepsilon_0)` at zero curvature.

    Assembled from the section's own element census — a tendon's limit
    enters the section strain shifted by its locked-in strain,

    .. math::

        \varepsilon_{\mathrm{sec}} = \varepsilon_p - \varepsilon_{init}

    — so this number does not depend on the bracket the solver builds,
    which is the thing under observation.

    Parameters
    ----------
    solver : FiberSolver
    n : int, optional

    Returns
    -------
    float
        Largest axial force at zero curvature [N], tension positive.
    """
    reach = 0.0
    for mat, _bulk, _idx, eps_init in solver._tendon_groups:
        reach = max(reach, float(mat.eps_max) - float(np.min(eps_init)))
    grid = np.linspace(float(solver.sec.bulk_material.eps_min) * 1.01,
                       reach * 1.01, n)
    zeros = np.zeros_like(grid)
    N, _, _ = solver.integrate_batch(grid, zeros, zeros)
    return float(np.max(N))


def _max_axial_at_zero_moment(solver, n_chi=61, n_eps=900):
    r"""
    Largest :math:`N` over the locus :math:`|M_x| \le` ``_M_TOL``.

    Walks a logarithmic grid of curvatures, finds every sign change of
    :math:`M_x(\varepsilon_0)` along each, refines it by bisection and
    **verifies** the root before counting it.  Verification is not
    ceremony: :math:`M_x` jumps where a fibre leaves the active range,
    and a sign-change bracket around a jump is not a root.  An earlier
    version of this scan without the check reported a maximum of
    1 112 295 N at a point whose moment was :math:`-6.7\times10^7`
    N.mm, i.e. it would have called N-17 a search defect.

    Parameters
    ----------
    solver : FiberSolver
    n_chi, n_eps : int, optional
        Scan resolution; reported in the failure messages.

    Returns
    -------
    N_max : float
        Largest attainable axial force [N]; ``-inf`` if nothing
        verified.
    roots : int
        Number of verified roots.
    state : tuple of float
        ``(eps0, chi_x)`` achieving ``N_max``.
    """
    eps_hi = 0.0
    for mat, _bulk, _idx, eps_init in solver._tendon_groups:
        eps_hi = max(eps_hi, float(mat.eps_max) - float(np.min(eps_init)))
    grid = np.linspace(float(solver.sec.bulk_material.eps_min) * 1.01,
                       eps_hi * 1.01, n_eps)
    zeros = np.zeros_like(grid)
    mags = np.logspace(-8.0, -2.0, max(1, (n_chi - 1) // 2))
    curvatures = np.concatenate([-mags[::-1], [0.0], mags])

    N_max, roots, state = float("-inf"), 0, (0.0, 0.0)
    for chi in curvatures:
        _, Mx, _ = solver.integrate_batch(
            grid, np.full_like(grid, chi), zeros)
        for i in np.where(np.diff(np.sign(Mx)) != 0)[0]:
            a, b, fa = grid[i], grid[i + 1], Mx[i]
            for _ in range(60):
                mid = 0.5 * (a + b)
                _, fm, _ = solver.integrate(mid, chi, 0.0)
                if fa * fm <= 0.0:
                    b = mid
                else:
                    a, fa = mid, fm
            root = 0.5 * (a + b)
            N_r, Mx_r, _ = solver.integrate(root, chi, 0.0)
            if abs(Mx_r) > _M_TOL:
                continue                      # a jump, not a root
            roots += 1
            if N_r > N_max:
                N_max, state = float(N_r), (float(root), float(chi))
    return N_max, roots, state


def _outcome(solver, N_target):
    r"""
    The outcome of a pure-axial request, whatever its form.

    Both a returned mapping and a raised exception are captured, so the
    caller asserts on **behaviour** and stays valid on a tree where the
    shape of the outcome differs.  This is the ``_compile_outcome`` /
    ``_refusal`` pattern of ``18_1 §4``.

    Parameters
    ----------
    solver : FiberSolver
    N_target : float

    Returns
    -------
    sol : dict or None
        The returned mapping, or ``None`` when the call raised.
    text : str
        ``repr`` of the mapping, or ``"<Type>: <message>"``.
    """
    try:
        sol = solver.solve_equilibrium(N_target, 0.0, 0.0, tol=1.0)
    except Exception as exc:                # noqa: BLE001 — outcome is data
        return None, f"{type(exc).__name__}: {exc}"
    return sol, repr(sol)


class TestTheStateDoesNotExist:
    r"""
    The measurement that decided the form of the closure.

    **Guard, not evidence.**  Green on both trees by construction: it
    asserts a property of the *section*, which no repair to the solver
    changes.  It is here because a refusal is only legitimate if the
    state really is absent, and that has to be established by something
    other than the routine doing the refusing.
    """

    def test_the_target_is_outside_the_zero_moment_reach(self, tmp_path):
        r"""
        No strain plane carries the target with :math:`M_x \approx 0`.

        The target is 95 % of the zero-curvature axial tensile
        capacity; the largest force compatible with zero moment is
        about 30 % of it.
        """
        solver = FiberSolver(_slab(tmp_path))
        target = 0.95 * _true_axial_reach(solver)
        N_max, roots, state = _max_axial_at_zero_moment(solver)

        assert roots > 0, (
            "the Mx = 0 locus scan verified no root at all, so this "
            "measurement is not evidence either way: the instrument is "
            "blind, which is silence and not proof.")
        assert N_max < target, (
            f"this test's premise failed: the scan found a state at "
            f"N = {N_max:.1f} N with |Mx| <= {_M_TOL:g} N.mm, which "
            f"reaches the target of {target:.1f} N. If that is real, "
            f"N-17 is a search defect after all and closing it as a "
            f"refusal is wrong.")
        assert N_max == pytest.approx(3.17e5, rel=0.05), (
            f"the measured zero-moment reach moved to {N_max:.1f} N "
            f"from the 317 016 N recorded in handoff/20_1 for this "
            f"geometry ({roots} verified roots, extreme state "
            f"eps0 = {state[0]:.6g}, chi_x = {state[1]:.6g}). A number "
            f"that moves without a declared cause is what this suite "
            f"exists to catch.")

    def test_the_moment_on_the_target_locus_is_a_plateau(self, tmp_path):
        r"""
        On :math:`N = N_{target}`, :math:`M_x` equals :math:`T\,e`.

        The absence is not a near miss.  At that load the concrete is
        wholly cracked, the strand carries everything, and the moment
        is the product of the force by the lever arm.  Asserting the
        plateau rather than a minimum is what makes the absence
        structural instead of numerical.
        """
        section = _slab(tmp_path)
        solver = FiberSolver(section)
        target = 0.95 * _true_axial_reach(solver)
        lever = float(section.y_tendons[0]) - solver.y_ref

        found = []
        for chi in np.linspace(-3.0e-4, 3.0e-4, 121):
            grid = np.linspace(-0.012, 0.032, 400)
            N, _, _ = solver.integrate_batch(
                grid, np.full_like(grid, chi), np.zeros_like(grid))
            f = N - target
            for i in np.where(np.diff(np.sign(f)) != 0)[0]:
                a, b, fa = grid[i], grid[i + 1], f[i]
                for _ in range(60):
                    mid = 0.5 * (a + b)
                    Nm, _, _ = solver.integrate(mid, chi, 0.0)
                    if fa * (Nm - target) <= 0.0:
                        b = mid
                    else:
                        a, fa = mid, Nm - target
                root = 0.5 * (a + b)
                N_r, Mx_r, _ = solver.integrate(root, chi, 0.0)
                if abs(N_r - target) <= 1.0:
                    found.append(abs(Mx_r))

        assert found, "no state at N = target was found at all"
        assert min(found) == pytest.approx(abs(target * lever), rel=1e-6), (
            f"the smallest |Mx| on the N = target locus is "
            f"{min(found):.6e} N.mm over {len(found)} verified roots, "
            f"which is not the cracked-section product |T e| = "
            f"{abs(target * lever):.6e}. The plateau explanation in "
            f"handoff/20_1 does not hold on this tree.")


class TestTheRefusalNamesItsReason:
    r"""
    N-17 closed: a pure-axial request that cannot be met says so.

    **Discriminating.**  Red on the pristine Phase 19 tree, where
    ``solve_equilibrium`` returns a mapping with no reason in it.
    """

    def test_an_unattainable_pure_axial_request_carries_a_reason(
            self, tmp_path):
        r"""The outcome tells "no such state" from "search failed"."""
        solver = FiberSolver(_slab(tmp_path))
        target = 0.95 * _true_axial_reach(solver)
        sol, text = _outcome(solver, target)

        assert sol is None or not sol.get("converged"), (
            "the target is outside the section's zero-moment reach, so "
            "a converged result would mean the solver reports a state "
            "the section does not have.")
        reason = None if sol is None else sol.get("reason")
        assert reason == "axial_state_not_attainable", (
            f"a pure-axial request for N = {target:.1f} N, which this "
            f"section cannot carry at zero moment, came back with "
            f"reason={reason!r}. A bare non-convergence cannot tell "
            f"the caller whether the state is missing or the search "
            f"is: that is item N-17. Outcome was {text}")

    def test_the_reason_carries_the_measured_reach_and_the_residual(
            self, tmp_path):
        r"""
        The verdict travels with numbers, never on its own.

        ``converged`` is meaningless above certain residual
        magnitudes, so the refusal carries the relative residual and
        the tolerance it was taken under, exactly as the timeline dump
        does with ``solver_flag``.
        """
        solver = FiberSolver(_slab(tmp_path))
        target = 0.95 * _true_axial_reach(solver)
        sol, text = _outcome(solver, target)
        assert sol is not None, (
            f"the default solve path raised instead of returning; the "
            f"strict form is meant to be opt-in. Outcome was {text}")

        reach = sol.get("attainable_N")
        assert reach is not None and len(reach) == 2, (
            f"the outcome does not carry the measured zero-moment "
            f"reach, so its claim that the state is missing cannot be "
            f"checked against anything. Outcome was {text}")
        assert target > reach[1], (
            f"the outcome claims the target is unattainable while "
            f"reporting a reach of {reach} that contains it.")
        assert sol.get("residual_rel") is not None, (
            f"the verdict is published without its relative residual. "
            f"Outcome was {text}")
        assert sol.get("M_tol") is not None, (
            f"the verdict is published without the moment tolerance it "
            f"was taken under. Outcome was {text}")
        assert sol.get("reach_resolution") is not None, (
            f"the certificate does not declare the coarseness of the "
            f"scan behind it. Outcome was {text}")

    def test_a_section_the_plane_cannot_certify_is_not_certified(self):
        r"""
        Off the scanned plane, the refusal declines to be a certificate.

        The reach is measured over :math:`(\varepsilon_0, \chi_x)` with
        :math:`\chi_y = 0`.  On a section whose elements are not
        symmetric about the reference y-axis that plane is a *slice* of
        the search space, so a target outside the measured interval is
        not proven absent — the bound is a lower bound.

        ``examples/Pannello_187x30.yaml`` is that case, and it is not
        hypothetical: with only :math:`M_x` verified at each root the
        scan called its extremum attainable at 2 042 789 N while that
        state carried :math:`M_y = 4.73 \times 10^8` N.mm, and the
        outcome came back ``search_failed_inside_reach`` — a claim that
        a defect exists, issued by a half-blind instrument.  The claim
        was false and the instrument was the thing at fault.
        """
        path = os.path.join(_ROOT, "examples", "Pannello_187x30.yaml")
        assert os.path.exists(path)
        solver = FiberSolver(load_yaml(path)["section"])
        sol, text = _outcome(solver, 1_940_667.8)
        assert sol is not None and not sol.get("converged"), (
            f"the premise of this test moved: the target now solves. "
            f"Outcome was {text}")
        assert sol.get("reason") == "axial_reach_not_certified", (
            f"a section with an asymmetric element layout got "
            f"reason={sol.get('reason')!r}. Anything stronger is a "
            f"certificate of absence taken on a plane the scan did "
            f"not visit. Outcome was {text}")

    def test_an_attainable_pure_axial_request_is_not_refused(
            self, tmp_path):
        r"""
        Positive control: inside the reach nothing changes.

        250 kN is inside the measured reach of about 317 kN.  If this
        went red the refusal would be firing on states that exist,
        which is worse than the defect it closes.
        """
        solver = FiberSolver(_slab(tmp_path))
        sol, text = _outcome(solver, 250.0e3)
        assert sol is not None and sol.get("converged"), (
            f"250 kN is inside this section's zero-moment reach and "
            f"must still solve. Outcome was {text}")
        assert sol.get("reason") is None, (
            f"a converged solve must not carry a refusal reason. "
            f"Outcome was {text}")


class TestTheExampleIsDistributed:
    r"""
    The corridor is reachable from the battery, not only from here.

    **Declared discriminator against the tree without the example**,
    the same standing the N-3 fixtures of Phase 19 have: it is a claim
    about the shipped set of models, not about the code.  The check is
    an ``assert`` on ``os.path.exists`` rather than a load, so a tree
    without the file fails one test with a message instead of failing
    collection for the whole module.
    """

    def test_the_strand_only_example_ships_and_matches_the_fixture(
            self, tmp_path):
        r"""
        The example carries the geometry the measurements were taken on.
        """
        assert os.path.exists(_EXAMPLE), (
            f"examples/example_strand_only_slab.yaml is absent at "
            f"{_EXAMPLE}. Until Phase 20 no distributed model had "
            f"strands and no ordinary reinforcement, so the axial "
            f"corridor N-15 and N-17 live in was invisible to every "
            f"instrument of the battery and visible only to a fixture: "
            f"an INERT on that path was silence, not evidence "
            f"(19_0 §2).")
        shipped = FiberSolver(load_yaml(_EXAMPLE)["section"])
        inline = FiberSolver(_slab(tmp_path))
        assert _true_axial_reach(shipped) == pytest.approx(
            _true_axial_reach(inline), rel=1e-12), (
            "the distributed example and the Phase 19 fixture no "
            "longer describe the same section, so every number this "
            "phase recorded against one of them has stopped applying "
            "to the other.")

    def test_the_example_has_no_ordinary_reinforcement(self):
        r"""
        The property that makes the example worth distributing.

        A single rebar would move the axial peak to the rebar limit and
        the whole corridor would close, silently.
        """
        assert os.path.exists(_EXAMPLE), (
            f"examples/example_strand_only_slab.yaml is absent at "
            f"{_EXAMPLE}; see the sibling test for why it ships. The "
            f"existence check is repeated here rather than left to the "
            f"loader so the red reads as a missing file and not as a "
            f"FileNotFoundError, which would prove a path wrong rather "
            f"than a claim false.")
        section = load_yaml(_EXAMPLE)["section"]
        assert len(section.rebars) == 0, (
            f"the strand-only example has grown {len(section.rebars)} "
            f"rebar(s). On a section with ordinary reinforcement the "
            f"axial peak falls at the rebar limit and the band N-15 "
            f"measured drops from 26.37 % to ~0 % (19_0 §3): the file "
            f"would still load and would no longer exhibit anything.")
        assert len(section.tendons) >= 1


#: Whether the tree under test carries the Phase 20 refusal machinery.
#: Read once, at import, through :func:`hasattr` — the only place in
#: this module where a Phase 20 name is touched outside a function.
_HAS_REACH = hasattr(FiberSolver, "axial_reach_at_zero_moment")


@pytest.mark.skipif(
    not _HAS_REACH,
    reason="the strict pure-axial form is a Phase 20 contract; on a "
           "tree that predates it these tests would fail with "
           "ImportError and AttributeError, which prove a name is "
           "missing and not that a defect is present (19_1 §4)")
class TestTheStrictFormIsOptIn:
    r"""
    The raising form exists and is **not** the default.

    **Contract, not evidence.**  Names symbols introduced in Phase 20.
    On a tree that predates them these tests are *skipped*, not failed:
    a red raised by an absent name is not an observation of behaviour,
    and letting one into the census would pollute the count that
    ``handoff/20_1`` reports.  Skipping costs nothing on the delivered
    tree, where the names exist and the class runs in full.

    This is not a defect pinned by a green test.  The claim is about a
    contract added in this phase — that the raising form exists *and*
    that the lenient one still returns — and a later phase that removes
    either half must see this go red.
    """

    def test_require_equilibrium_raises_on_an_unattainable_request(
            self, tmp_path):
        r"""The opt-in form turns the label into an exception."""
        from gensec.solver.integrator import AxialStateNotAttainable

        solver = FiberSolver(_slab(tmp_path))
        target = 0.95 * _true_axial_reach(solver)
        with pytest.raises(AxialStateNotAttainable) as caught:
            solver.require_equilibrium(target, 0.0, 0.0, tol=1.0)
        assert caught.value.N_target == pytest.approx(target)
        assert caught.value.attainable[1] < target
        assert "does not exist" in str(caught.value)

    def test_the_lenient_form_keeps_returning_a_point(self, tmp_path):
        r"""
        The default path must not raise.

        A capability removed is a cost regardless of how it reads:
        diagram generators skip non-converged points, and one
        unattainable point must not abort a whole curve.
        """
        solver = FiberSolver(_slab(tmp_path))
        target = 0.95 * _true_axial_reach(solver)
        sol = solver.solve_equilibrium(target, 0.0, 0.0, tol=1.0)
        assert isinstance(sol, dict)
        assert {"eps0", "chi_x", "chi_y", "N", "Mx", "My",
                "converged"}.issubset(sol)

    def test_a_blind_reach_measurement_refuses_to_certify(
            self, tmp_path, monkeypatch):
        r"""
        Zero verified roots is labelled, never used as a certificate.

        The failure mode Phase 19 spent itself removing was an
        instrument reporting silence as proof, so the refusal must not
        be issued on the strength of a scan that found nothing.

        The blind scan is **substituted**, not provoked.  Trying to
        provoke it by squeezing the tolerance does not work and the
        failed attempt is worth recording: with
        ``M_tol = 1e-30, resolution = (3, 4)`` the scan still verifies
        8 roots, because bisection lands on strains where the moment is
        exactly ``0.0`` and ``0.0 <= 1e-30`` holds.  A test that drove
        the branch that way would have been asserting a coincidence.
        """
        solver = FiberSolver(_slab(tmp_path))

        def _blind(M_tol=1.0e3, resolution=None):
            return {"N_min": 0.0, "N_max": 0.0,
                    "state_min": (0.0, 0.0), "state_max": (0.0, 0.0),
                    "roots": 0, "resolution": (0, 0),
                    "M_tol": float(M_tol)}

        monkeypatch.setattr(solver, "axial_reach_at_zero_moment", _blind)
        out = solver._refuse_unattainable_axial({"N": 0.0}, 1.0e9, 1.0e3)
        assert out["reason"] == "reach_not_measurable"
        assert "nothing here is evidence either way" in out["refusal"]
        assert out["residual_rel"] == pytest.approx(1.0)
