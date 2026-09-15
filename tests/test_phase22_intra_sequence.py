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
Phase 22 — staged injection inside a post-tensioning sequence (D1).

What is under test
------------------
A tendon injected **between two jacks** is a section element from that
moment on: it transforms the section the later jacks are stressed
against by :math:`(E_p - E_c)A_p` at its own level.  Until Phase 22 the
driver's resisting view was tendon-free and *constant* across the whole
sequence, and it refused any history that implied otherwise.

How these fixtures are written
------------------------------
Every one of them observes **behaviour**, not the presence of a name.
The outcome of a call is captured as a ``(kind, payload)`` pair by
:func:`_outcome`, so a fixture stays meaningful on a tree where the
outcome has a different shape — a refusal instead of a number, a
different exception class, a different message.  A red produced by an
``ImportError`` or an ``AttributeError`` proves that a name is missing,
not that a defect exists, and is not evidence of anything
(``21_0 §5``).

The reference is **not the driver compared with itself**.  It is
:func:`~gensec.solver.prestress_transfer.sequential_posttension_loss`,
the closed transformed-section form, which has carried the
``bonded_before`` hook since Phase 4 and which the driver has never
been able to reach on this case.  A refusal justified by a different
instrument than the one refusing.
"""

import numpy as np
import pytest

from gensec.geometry.fiber import Tendon
from gensec.geometry.section import RectSection
from gensec.io_yaml import load_yaml
from gensec.solver.posttension import solve_posttension_sequence
from gensec.solver.prestress_transfer import sequential_posttension_loss
from gensec.solver.timeline import ConstructionTimeline


# ---------------------------------------------------------------------------
#  Outcome capture — the ``_outcome`` pattern of test_phase20_axial.py
# ---------------------------------------------------------------------------

def _outcome(fn, *args, **kwargs):
    r"""
    Run *fn* and return ``(kind, payload)``.

    ``kind`` is ``'ok'`` or the exception's class name; ``payload`` is
    the return value or the message.  Nothing is asserted here: the
    point is that a fixture can compare *what happened* across two
    trees, including a tree on which the call does not exist in the
    form the other tree offers.

    Returns
    -------
    tuple
    """
    try:
        return ("ok", fn(*args, **kwargs))
    except BaseException as exc:            # noqa: BLE001 — outcome is data
        return (type(exc).__name__, str(exc))


def _reason(fn, *args, **kwargs):
    r"""
    Run *fn* and return the **named** reason of its refusal.

    ``'none'`` when the call succeeds; the exception's declared
    ``reason`` when it carries one; ``'unnamed:<ExcType>'`` when it does
    not.  Nothing is imported to obtain this — the classes themselves
    are deliberately NOT imported at module scope anywhere in this file,
    so that a tree on which they do not exist produces a red on
    *behaviour* rather than an ``ImportError``, which would only prove
    that a name is missing (``21_0 §5``).

    Returns
    -------
    str
    """
    try:
        fn(*args, **kwargs)
        return "none"
    except BaseException as exc:            # noqa: BLE001 — outcome is data
        return str(getattr(exc, "reason", None)
                   or f"unnamed:{type(exc).__name__}")


# ---------------------------------------------------------------------------
#  A linear-elastic section, so the fiber solve and the closed form are
#  the same linear algebra and any disagreement is a defect, not a model
#  difference.
# ---------------------------------------------------------------------------

class _LinearElastic:
    r"""Constant-modulus law :math:`\sigma = E\varepsilon`."""

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

    def content_token(self):
        r"""Declared content: the modulus and the strain limit."""
        from gensec.materials.base import digest_bytes, encode_value
        return digest_bytes(
            encode_value("tests.test_phase22_intra_sequence._LinearElastic"),
            encode_value(float(self.E)),
            encode_value(float(self._lim)),
        )


# The declared geometry.  A number without its geometry is not a measure.
B, H, N_Y = 400.0, 1200.0, 240
EC, EP = 35220.462, 195000.0
AP, SP0 = 1500.0, 1300.0
Y_TENDONS = (150.0, 200.0, 250.0)


def _section():
    r"""400x1200 rectangle, three tendons at y = 150 / 200 / 250."""
    bulk = _LinearElastic(EC)
    strand = _LinearElastic(EP)
    tendons = [Tendon(y=float(yt), x=B / 2, material=strand, Ap=AP,
                      eps_pe=0.0) for yt in Y_TENDONS]
    return RectSection(B=B, H=H, bulk_material=bulk, rebars=[],
                       n_fibers_y=N_Y, tendons=tendons)


def _meshed_props(section, y_ref):
    r"""Meshed ``Ac, Sc, Ic`` about *y_ref*, from the bulk fiber arrays."""
    A = np.asarray(section.A_fibers, dtype=float)
    y = np.asarray(section.y_fibers, dtype=float)
    d = y - y_ref
    return (float(np.sum(A)), float(np.sum(A * d)), float(np.sum(A * d * d)))


def _closed_form(section, schedule, order=(0, 1, 2), scheme="one_pass"):
    r"""
    The independent reference for the same case.

    *schedule* is ``{tendon: step}``; it is translated into the closed
    form's per-tendon ``bonded_before`` entry, which carries exactly the
    same meaning and is where the name comes from.
    """
    y_ref = section.y_centroid
    Ac, Sc, Ic = _meshed_props(section, y_ref)
    tendons = [{"y": float(yt), "Ap": AP, "sigma_p0": SP0,
                "bonded_before": schedule.get(k)}
               for k, yt in enumerate(section.y_tendons)]
    return sequential_posttension_loss(
        EC, 0.0, EP, Ac=Ac, Sc=Sc, Ic=Ic, tendons=tendons,
        order=list(order), y_ref=y_ref, scheme=scheme)


# The staged-injection schedule: tendon k is grouted right after its own
# jack, so it is bonded from step k+1 onwards.  The last one is never
# bonded during the sequence -- there is no later jack to stiffen for.
STAGED = {0: 1, 1: 2}


# ---------------------------------------------------------------------------
#  1. THE DISCRIMINATING FIXTURE
# ---------------------------------------------------------------------------

def test_staged_injection_matches_the_closed_form():
    r"""
    The driver, given a grouting schedule, reproduces the closed form.

    This is the fixture that carries the repair.  On a pre-Phase-22 tree
    the driver has no way to be told about the schedule at all, so the
    outcome is a refusal and this is red on a *value*, not on a name.
    """
    sec = _section()
    kind, res = _outcome(
        solve_posttension_sequence, sec,
        sigma_p0=[SP0] * 3, Ep=EP, order=[0, 1, 2],
        bonded_before=STAGED, scheme="one_pass")
    assert kind == "ok", f"driver refused the staged case: {res}"

    ref = _closed_form(sec, STAGED)
    # 1e-6 MPa on a 1300 MPa stress: the two are the same linear system
    # solved twice, so anything above round-off is a defect.
    assert np.allclose(res.sigma_p_after, ref["sigma_p_after"], atol=1e-6)
    assert np.allclose(res.loss, ref["loss"], atol=1e-6)
    assert np.allclose(res.eps_ref_grout, ref["eps_ref_grout"], atol=1e-12)
    assert res.converged


def test_the_measured_gap_is_the_one_the_ledger_records():
    r"""
    The size of the defect, asserted rather than described.

    Tendon-free constant view against the staged-injection view, on the
    geometry declared at the top of this file.  The recorded numbers
    (``22_0 §1``) are +0.464 % on P1's final stress and -5.11 % on its
    loss; the brackets below are deliberately wide enough to survive a
    mesh change and narrow enough that a regression to the constant view
    cannot pass.
    """
    sec = _section()
    free = solve_posttension_sequence(
        sec, sigma_p0=[SP0] * 3, Ep=EP, order=[0, 1, 2],
        bonded_before=None, scheme="one_pass")
    staged = solve_posttension_sequence(
        sec, sigma_p0=[SP0] * 3, Ep=EP, order=[0, 1, 2],
        bonded_before=STAGED, scheme="one_pass")

    # The bonded cables stiffen the section, so the section shortens
    # less and the debit is smaller: sigma_p ends HIGHER, every time.
    assert np.all(staged.sigma_p_after >= free.sigma_p_after - 1e-12)
    pct = 100.0 * (staged.sigma_p_after[0] / free.sigma_p_after[0] - 1.0)
    assert 0.35 < pct < 0.60, f"P1 sigma_p gap {pct:.4f} % off the record"
    loss_pct = 100.0 * (staged.loss[0] / free.loss[0] - 1.0)
    assert -6.0 < loss_pct < -4.2, f"P1 loss gap {loss_pct:.4f} % off"
    # The last-jacked tendon takes no debit under either view: it is
    # never strain-compatible with anything while being stressed.
    assert free.loss[2] == pytest.approx(0.0, abs=1e-9)
    assert staged.loss[2] == pytest.approx(0.0, abs=1e-9)


def test_the_resisting_section_is_photographed_per_step():
    r"""
    Each stage records how many tendons resisted when it was jacked.

    An integer, and that is the point: the repair moves stresses in
    their third significant figure, so a photograph made only of
    stresses cannot tell a per-step view from a constant one
    (``20_1 §3``).
    """
    sec = _section()
    staged = solve_posttension_sequence(
        sec, sigma_p0=[SP0] * 3, Ep=EP, order=[0, 1, 2],
        bonded_before=STAGED)
    assert [st.report["n_bonded_at_jack"] for st in staged.stages] == [0, 1, 2]
    free = solve_posttension_sequence(
        sec, sigma_p0=[SP0] * 3, Ep=EP, order=[0, 1, 2])
    assert [st.report["n_bonded_at_jack"] for st in free.stages] == [0, 0, 0]


# ---------------------------------------------------------------------------
#  2. A capacity removed is a cost: the old path must still be the old path
# ---------------------------------------------------------------------------

def test_deferred_single_grouting_is_untouched():
    r"""
    With no schedule, the sequence is what it always was.

    ``None`` and ``{}`` must not be two different models, and neither
    may differ from the closed form on the tendon-free view.  The global
    version of this claim is the numeric fingerprint (``22_1 §3``); this
    is its local, falsifiable form.
    """
    sec = _section()
    a = solve_posttension_sequence(sec, sigma_p0=[SP0] * 3, Ep=EP)
    b = solve_posttension_sequence(sec, sigma_p0=[SP0] * 3, Ep=EP,
                                   bonded_before={}, pending=[])
    assert np.array_equal(a.sigma_p_after, b.sigma_p_after)
    assert np.array_equal(a.loss, b.loss)
    ref = _closed_form(sec, {})
    assert np.allclose(a.loss, ref["loss"], atol=1e-6)


# ---------------------------------------------------------------------------
#  3. The refusals that survive, and the one that was wrong
# ---------------------------------------------------------------------------

def test_a_pending_tendon_is_not_an_already_bonded_one():
    r"""
    The narrowing: an undeclared subset is a missing declaration.

    A strict subset of ``order`` used to be read as "the rest are
    already grouted" and refused with a message about bonded stiffness.
    In the timeline that reading was wrong at every staged-injection
    history: at the first ``grout`` the remaining tendons have simply
    not been jacked yet.  The refusal is now about the **absence of a
    declaration**, and declaring ``pending`` resolves it.
    """
    sec = _section()
    kind, msg = _outcome(solve_posttension_sequence, sec,
                         sigma_p0=[SP0] * 3, Ep=EP, order=[0, 1])
    assert kind == "IntraSequenceUndeclared", msg
    assert _reason(solve_posttension_sequence, sec, sigma_p0=[SP0] * 3,
                   Ep=EP, order=[0, 1]) == "intra_sequence_undeclared"
    assert "pending" in msg and "already_bonded" in msg

    kind, res = _outcome(solve_posttension_sequence, sec,
                         sigma_p0=[SP0, SP0, 0.0], Ep=EP, order=[0, 1],
                         pending=[2])
    assert kind == "ok", res
    # A pending tendon is neither a load nor an element: it takes no
    # debit and it contributes no stiffness.
    assert res.loss[2] == pytest.approx(0.0, abs=1e-12)


def test_the_retrofit_case_refuses_an_undeclared_state_and_says_why():
    r"""
    ``already_bonded`` as a **bare index list** still refuses — and
    Phase 24 changed why.

    What was missing was never the stiffness: that is the same algebra
    the staged case runs.  It was the effective stress of a cable GenSec
    did not jack and the strain state of concrete already in service.
    Phase 24 does not derive either; it makes both **declarable**
    (:class:`~gensec.solver.posttension.PreexistingState`).  So the
    refusal moved from "GenSec does not model this" to "you have not
    declared this", which is a different fact with a different remedy —
    the same distinction the normative-code registry draws between
    ``custom`` and ``undeclared``.
    """
    sec = _section()
    kind, msg = _outcome(solve_posttension_sequence, sec,
                         sigma_p0=[SP0] * 3, Ep=EP, already_bonded=[0])
    assert kind == "PreexistingStateUndeclared", msg
    assert _reason(solve_posttension_sequence, sec, sigma_p0=[SP0] * 3,
                   Ep=EP, already_bonded=[0]) == "preexisting_undeclared"
    # The refusal points at what IS modelled instead of stopping dead:
    # the declaration that lifts it, and the neighbouring case.
    assert "PreexistingState" in msg
    assert "bonded_before" in msg
    # An out-of-range index stays a plain input error, not a scope one.
    kind, msg = _outcome(solve_posttension_sequence, sec,
                         sigma_p0=[SP0] * 3, Ep=EP, already_bonded=[9])
    assert kind == "ValueError", msg


def test_a_tendon_cannot_be_grouted_before_it_is_stressed():
    r"""The schedule is validated, not trusted."""
    sec = _section()
    for bad in ({1: 1}, {0: 0}, {0: 99}, {7: 1}):
        kind, msg = _outcome(solve_posttension_sequence, sec,
                             sigma_p0=[SP0] * 3, Ep=EP, order=[0, 1, 2],
                             bonded_before=bad)
        assert kind == "ValueError", f"{bad} was accepted: {msg}"


# ---------------------------------------------------------------------------
#  4. The same case, from YAML, through the timeline
# ---------------------------------------------------------------------------

def _demand_db(doc):
    dem = doc.get("demands") or []
    if isinstance(dem, dict):
        return dem
    return {d["name"]: d for d in dem}


def test_the_distributed_example_resolves_and_is_staged():
    r"""
    ``examples/example_staged_injection.yaml`` walks end to end.

    Before Phase 22 this history died at its **first** ``grout`` event,
    because only one of the three tendons had been stressed by then and
    the driver read the missing two as already bonded.  The fixture
    asserts the walk completes and that the reconciled datum of each
    tendon exists — the observable end of the repair.
    """
    doc = load_yaml("examples/example_staged_injection.yaml")
    tl = ConstructionTimeline.from_document(doc)
    kind, res = _outcome(tl.resolve, doc["section"], _demand_db(doc))
    assert kind == "ok", f"the staged-injection history was refused: {res}"
    assert sorted(res.grout_eps_init) == ["P1", "P2", "P3"]
    assert sorted(res.stress_sigma) == ["P1", "P2", "P3"]
    # Reconciled datums grow along the sequence: each cable is injected
    # into a section that is already more compressed than the last one
    # found, so it locks in more strain.
    e = [res.grout_eps_init[t] for t in ("P1", "P2", "P3")]
    assert e[0] < e[1] < e[2]


if __name__ == "__main__":              # pragma: no cover
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
