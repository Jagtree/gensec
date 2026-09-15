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
Phase 24 — the declared retrofit (D1b-1).

What is under test
------------------
A new tendon jacked into a section **already in service**, which carries
grouted tendons at some effective stress.  Until Phase 24 the driver
refused the case outright, because three facts it needs are facts it
cannot derive: the effective stress of each existing tendon, the strain
state of the existing concrete, and the datum both are referred to.

Phase 24 does not derive them.  It makes them **declarable**, and
answers when — and only when — they are declared.  What is refused is
no longer the physics; it is an undeclared input, and the two carry
different remedies.  The distinction is the same one the normative-code
registry draws between ``custom`` and ``undeclared``.

How these fixtures are written
------------------------------
Nothing that Phase 24 introduces is imported at module scope.  The
declaration class is fetched with :func:`getattr` from the module
object, so a tree that does not have it produces a red **assertion**
rather than an :class:`ImportError`, and an
:class:`AttributeError` never stands in for evidence: a red that says
"this name does not exist" demonstrates that a name is missing, not
that a defect is present (``21_0 §5``, ``23_0 §5``).

The reference is not the driver compared with itself.  It is
:func:`_closed_form_retrofit`, transformed-section algebra written out
here: assemble :math:`EA, ES, EI` with the pre-existing tendon
transformed by :math:`(E_p - E_c)A_p`, invert the 2x2 for each jack,
and debit every anchored tendon by :math:`-E_p\,\Delta\varepsilon_c`.
The driver reaches the same numbers by meshing 240 fibres and running
Newton on each increment.  Two different computations; a disagreement
is a defect, not a model difference.
"""

import numpy as np
import pytest

from gensec.geometry.fiber import Tendon
from gensec.geometry.section import RectSection
import gensec.solver.posttension as pt


# ---------------------------------------------------------------------------
#  Outcome capture — the ``_outcome`` pattern of test_phase20_axial.py
# ---------------------------------------------------------------------------

def _outcome(fn, *args, **kwargs):
    r"""
    Run *fn* and return ``(kind, payload)``.

    ``kind`` is ``'ok'`` or the exception's class name; ``payload`` is
    the return value or the message.  Nothing is asserted here: a
    fixture can then compare *what happened* across two trees,
    including a tree on which the call does not exist in the form the
    other tree offers.

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

    ``'none'`` when the call succeeds, the exception's declared
    ``reason`` when it carries one, ``'unnamed:<ExcType>'`` when it
    does not.  No exception class is imported to obtain this.

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


def _declaration_cls():
    r"""
    The declaration object, fetched by name from the module.

    Returns ``None`` on a tree that does not carry it, so that every
    fixture below can turn its absence into an assertion instead of an
    :class:`ImportError`.

    Returns
    -------
    type or None
    """
    return getattr(pt, "PreexistingState", None)


def _declare(sigma_pe, eps0=0.0, chi_x=0.0, chi_y=0.0):
    r"""
    Build a declaration, or fail the calling test with a message.

    Parameters
    ----------
    sigma_pe : dict
        ``{tendon index: effective stress [MPa]}``.
    eps0, chi_x, chi_y : float, optional
        The declared strain plane of the existing concrete, about the
        driver's reference point.

    Returns
    -------
    object
    """
    cls = _declaration_cls()
    assert cls is not None, (
        "gensec.solver.posttension carries no PreexistingState: a "
        "retrofit cannot be declared, so it can only be refused.")
    return cls(sigma_pe=dict(sigma_pe), eps0=float(eps0),
               chi_x=float(chi_x), chi_y=float(chi_y),
               x_ref=X_REF_DECL, y_ref=Y_REF_DECL,
               concrete_age_days=AGE_C,
               tendon_age_days={int(k): AGE_P for k in sigma_pe})


# ---------------------------------------------------------------------------
#  A linear-elastic section: the fiber solve and the closed form are then
#  the same linear algebra, and any disagreement is a defect.
# ---------------------------------------------------------------------------

class _LinearElastic:
    r"""Constant-modulus law :math:`\sigma = E\varepsilon`."""

    def __init__(self, E, eps_lim=1.0):
        self.E = float(E)
        self._lim = float(eps_lim)

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
            encode_value("tests.test_phase24_retrofit._LinearElastic"),
            encode_value(self.E), encode_value(self._lim))


# The declared geometry, the one every published measurement of this
# defect family uses (``22_0 §1.1``).  A number without its geometry is
# not a measure.
B, H, N_Y = 400.0, 1200.0, 240
EC, EP = 35220.462, 195000.0
AP, SP0 = 1500.0, 1300.0
Y_TENDONS = (150.0, 200.0, 250.0)

# Tendon 0 is the existing cable: grouted long ago, now at 1100 MPa,
# in concrete carrying a compression plane of its own.
SIGMA_PE = 1100.0
EPS0_EX, CHIX_EX = -3.0e-4, 1.5e-7
# The plane is declared about the SOFFIT, where an engineer reading a
# strain gauge would declare it -- deliberately not about the driver's
# pinned reference, which sits at the tendon-free centroid (y = 600 for
# this rectangle) and which the caller has no way to know. The 600 mm
# lever makes a mis-anchored plane impossible to miss; declaring about
# mid-height would have put the two points on top of each other and the
# fixture would have passed on a coincidence.
X_REF_DECL, Y_REF_DECL = 0.0, 0.0
AGE_C, AGE_P = 14600.0, 14200.0        # forty years, and a year less

NAN = float("nan")


def _section():
    r"""400x1200 rectangle, three tendons at y = 150 / 200 / 250."""
    bulk = _LinearElastic(EC)
    strand = _LinearElastic(EP)
    tendons = [Tendon(y=float(yt), x=B / 2.0, material=strand, Ap=AP,
                      eps_pe=0.0) for yt in Y_TENDONS]
    return RectSection(B=B, H=H, bulk_material=bulk, rebars=[],
                       n_fibers_y=N_Y, tendons=tendons)


def _meshed_props(section, y_ref):
    r"""Meshed ``Ac, Sc, Ic`` about *y_ref*, from the bulk fiber arrays."""
    A = np.asarray(section.A_fibers, dtype=float)
    y = np.asarray(section.y_fibers, dtype=float)
    d = y - y_ref
    return (float(np.sum(A)), float(np.sum(A * d)), float(np.sum(A * d * d)))


def _closed_form_retrofit(section, preexisting, order, sigma_p0,
                          sigma_pe, eps_plane=(0.0, 0.0)):
    r"""
    Transformed-section reference for the declared retrofit.

    Independent of the driver: the section properties are assembled
    once with the pre-existing tendons transformed by
    :math:`(E_p - E_c)A_p`, and each jack is a 2x2 solve

    .. math::

        \begin{bmatrix} EA & ES \\ ES & EI \end{bmatrix}
        \begin{bmatrix} \Delta\varepsilon_0 \\ \Delta\chi \end{bmatrix}
        = \begin{bmatrix} N_j \\ M_j \end{bmatrix},
        \qquad
        \Delta\sigma_{p,i} = -E_p\,\Delta\varepsilon_{c,i}.

    A pre-existing tendon is **anchored from the start**: it is
    strain-compatible with the concrete before the first new jack, so
    it is debited by every one of them.  A tendon jacked in this
    sequence becomes anchored only after its own jack.

    Parameters
    ----------
    section : GenericSection
    preexisting : sequence of int
        Tendon indices already grouted when the sequence begins.
    order : sequence of int
        Stressing order of the new tendons.
    sigma_p0 : sequence of float
        Jacking stress per tendon [MPa]; entries for pre-existing
        tendons are not read.
    sigma_pe : dict
        ``{tendon index: effective stress [MPa]}`` of the existing ones.
    eps_plane : tuple of float, optional
        ``(eps0, chi_x)`` of the declared existing concrete, about the
        same reference point.

    Returns
    -------
    dict
        ``{'sigma_p_after': ndarray, 'loss': ndarray,
        'eps_ref_grout': ndarray}``.
    """
    y_ref = section.y_centroid
    Ac, Sc, Ic = _meshed_props(section, y_ref)
    yv = np.asarray(section.y_tendons, dtype=float)
    d = yv - y_ref
    nt = yv.size
    pre = set(int(p) for p in preexisting)

    EA, ES, EI = EC * Ac, EC * Sc, EC * Ic
    for k in sorted(pre):
        dE = EP - EC
        EA += dE * AP
        ES += dE * AP * d[k]
        EI += dE * AP * d[k] * d[k]

    sig = np.array(sigma_p0, dtype=float)
    for k, v in sigma_pe.items():
        sig[int(k)] = float(v)

    debit = np.zeros(nt)
    anchored = np.zeros(nt, dtype=bool)
    for k in sorted(pre):
        anchored[k] = True                  # bonded before the first jack
    eps0_ex, chix_ex = eps_plane
    # About the DECLARED reference point, then read at the tendons --
    # not about `y_ref`, which is the driver's and which the declaration
    # never saw.
    eps_c = eps0_ex + chix_ex * (yv - Y_REF_DECL)

    det = EA * EI - ES * ES
    for j in order:
        P = sig[j] * AP
        Nj, Mj = -P, -P * d[j]
        de0 = (EI * Nj - ES * Mj) / det
        dchi = (-ES * Nj + EA * Mj) / det
        d_eps = de0 + dchi * d
        eps_c = eps_c + d_eps
        mask = anchored.copy()
        mask[j] = False
        debit[mask] += -EP * d_eps[mask]
        anchored[j] = True

    return {"sigma_p_after": sig - debit, "loss": debit,
            "eps_ref_grout": eps_c}


def _run(**kwargs):
    r"""Call the driver on a fresh section with the given keywords."""
    return pt.solve_posttension_sequence(_section(), **kwargs)


# ===========================================================================
#  1. The refusal changes cause: unmodelled physics -> undeclared input
# ===========================================================================

def test_a_bare_index_list_is_refused_because_it_declares_no_stress():
    r"""
    ``already_bonded=[0]`` says *which* tendon is already grouted and
    not *at what stress*.  It is refused on every tree; what Phase 24
    changes is the cause, from "GenSec does not model this" to "you
    have not declared this", which carries a different remedy.
    """
    got = _reason(_run, sigma_p0=[NAN, SP0, SP0], Ep=EP,
                  order=[1, 2], pending=[], already_bonded=[0])
    assert got == "preexisting_undeclared", got


def test_the_bare_refusal_names_the_two_missing_facts_and_the_remedy():
    r"""
    A refusal names its reason **and** its remedy.  Both missing facts
    have to appear: the effective stress of the existing tendons and
    the strain state of the existing concrete.
    """
    kind, msg = _outcome(_run, sigma_p0=[NAN, SP0, SP0], Ep=EP,
                         order=[1, 2], pending=[], already_bonded=[0])
    assert kind != "ok"
    low = msg.lower()
    assert "sigma_pe" in low or "effective stress" in low, msg
    assert "strain" in low and "concrete" in low, msg
    assert "preexistingstate" in low.replace(" ", ""), msg


def test_deriving_the_effective_stress_is_named_as_out_of_scope():
    r"""
    The refusal must say GenSec will not *derive* the effective stress
    from the deferred history.  Naming the boundary is what stops a
    reader from expecting D1b-2 to be here.
    """
    kind, msg = _outcome(_run, sigma_p0=[NAN, SP0, SP0], Ep=EP,
                         order=[1, 2], pending=[], already_bonded=[0])
    assert kind != "ok"
    low = msg.lower()
    assert "derive" in low or "does not compute" in low, msg


# ===========================================================================
#  2. A complete declaration is answered
# ===========================================================================

def test_a_declared_retrofit_is_answered():
    r"""The whole point: declared, therefore answerable."""
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    kind, res = _outcome(_run, sigma_p0=[NAN, SP0, SP0], Ep=EP,
                         order=[1, 2], pending=[], already_bonded=state)
    assert kind == "ok", res
    assert len(res.stages) == 2


def test_the_existing_tendon_stiffens_the_section_the_new_jacks_pull_against():
    r"""
    The stiffness is the part that was never missing, and it must now
    be exact against the transformed-section reference.
    """
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    res = _run(sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2],
               pending=[], already_bonded=state)
    ref = _closed_form_retrofit(_section(), [0], [1, 2],
                                [NAN, SP0, SP0], {0: SIGMA_PE},
                                (EPS0_EX, CHIX_EX))
    for k in (1, 2):
        assert res.sigma_p_after[k] == pytest.approx(
            ref["sigma_p_after"][k], rel=1e-9)


def test_the_existing_tendon_is_debited_by_every_new_jack():
    r"""
    A grouted cable is strain-compatible with the concrete around it,
    so shortening the section shortens it too.  Its stress must fall
    below the declared value, by the amount the closed form gives.
    """
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    res = _run(sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2],
               pending=[], already_bonded=state)
    ref = _closed_form_retrofit(_section(), [0], [1, 2],
                                [NAN, SP0, SP0], {0: SIGMA_PE},
                                (EPS0_EX, CHIX_EX))
    assert res.sigma_p_after[0] < SIGMA_PE
    assert res.loss[0] > 0.0
    assert res.sigma_p_after[0] == pytest.approx(
        ref["sigma_p_after"][0], rel=1e-9)


def test_the_declared_stress_is_the_baseline_the_loss_is_taken_from():
    r"""
    :math:`\sigma_{p,\text{after}} = \sigma_{pe} - \Delta\sigma`.  The
    declared number is not decoration: it is the only baseline there
    is, which is exactly why the case could not be answered without it.
    """
    for spe in (900.0, 1100.0, 1250.0):
        state = _declare({0: spe}, EPS0_EX, CHIX_EX)
        res = _run(sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2],
                   pending=[], already_bonded=state)
        assert res.sigma_p_after[0] + res.loss[0] == pytest.approx(
            spe, rel=1e-12)


def test_the_loss_of_the_existing_tendon_does_not_depend_on_its_stress():
    r"""
    Under ``one_pass`` the debit is :math:`-E_p\Delta\varepsilon_c`,
    and :math:`\Delta\varepsilon_c` comes from the new jacks alone.
    Three declared stresses, one debit — the declaration moves the
    baseline and nothing else, which is what makes the previous
    fixture a test of bookkeeping rather than of algebra.
    """
    losses = []
    for spe in (900.0, 1100.0, 1250.0):
        state = _declare({0: spe}, EPS0_EX, CHIX_EX)
        res = _run(sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2],
                   pending=[], already_bonded=state)
        losses.append(float(res.loss[0]))
    assert losses[0] == pytest.approx(losses[1], rel=1e-12)
    assert losses[1] == pytest.approx(losses[2], rel=1e-12)


# ===========================================================================
#  3. The declared concrete datum is consumed, and where
# ===========================================================================

def test_the_declared_concrete_plane_lands_on_the_grouting_datum():
    r"""
    The existing concrete is not at zero strain when the new jack is
    pulled, and ``eps_ref_grout`` is the **absolute** strain the
    grouting reconciliation
    :math:`\varepsilon_{\text{init}} = \varepsilon_{pe,\text{eff}}
    - \varepsilon_{\text{sec,grout}}` is formed against.  So the
    declared plane has to be the starting value of that accumulation,
    not zero.
    """
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    res = _run(sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2],
               pending=[], already_bonded=state)
    ref = _closed_form_retrofit(_section(), [0], [1, 2],
                                [NAN, SP0, SP0], {0: SIGMA_PE},
                                (EPS0_EX, CHIX_EX))
    for k in range(3):
        assert res.eps_ref_grout[k] == pytest.approx(
            ref["eps_ref_grout"][k], rel=1e-9)


def test_a_different_declared_plane_moves_the_datum_and_not_the_loss():
    r"""
    The declared plane is a **state**, not an increment.  It must move
    ``eps_ref_grout`` by exactly itself and leave the elastic-shortening
    debit untouched — a plane that leaked into the per-jack solves
    would add its own force to every increment and pollute the loss.
    """
    a = _run(sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2], pending=[],
             already_bonded=_declare({0: SIGMA_PE}, 0.0, 0.0))
    b = _run(sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2], pending=[],
             already_bonded=_declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX))
    d = np.asarray(_section().y_tendons, dtype=float) - Y_REF_DECL
    shift = EPS0_EX + CHIX_EX * d
    assert b.loss == pytest.approx(a.loss, rel=1e-12)
    assert b.eps_ref_grout == pytest.approx(
        a.eps_ref_grout + shift, rel=1e-9, abs=1e-15)


# ===========================================================================
#  4. What a declaration may not contradict
# ===========================================================================

def test_a_jacking_stress_for_an_existing_tendon_is_refused():
    r"""
    A pre-existing tendon was jacked in a history GenSec did not see.
    A finite ``sigma_p0`` entry for it is either a duplicate of the
    declared effective stress or a contradiction of it, and GenSec
    cannot tell which — so it refuses and names ``nan`` as the remedy.
    """
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    kind, msg = _outcome(_run, sigma_p0=[SP0, SP0, SP0], Ep=EP,
                         order=[1, 2], pending=[], already_bonded=state)
    assert kind == "ValueError", (kind, msg)
    assert "nan" in msg.lower(), msg


def test_a_tendon_cannot_be_both_stressed_here_and_pre_existing():
    r"""One tendon, one role, and the fourth role does not exist."""
    state = _declare({1: SIGMA_PE}, EPS0_EX, CHIX_EX)
    kind, msg = _outcome(_run, sigma_p0=[SP0, NAN, SP0], Ep=EP,
                         order=[1, 2], pending=[], already_bonded=state)
    assert kind == "ValueError", (kind, msg)


def test_a_base_sollecitazione_alongside_a_retrofit_is_refused():
    r"""
    The existing state is declared **once**, as the concrete's strain
    plane.  A base sollecitazione would declare part of it a second
    time, and GenSec cannot tell which part — nor whether the existing
    tendon should be debited for it.  Refused, named, and queued.
    """
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    kind, msg = _outcome(_run, sigma_p0=[NAN, SP0, SP0], Ep=EP,
                         order=[1, 2], pending=[], already_bonded=state,
                         base_N=-1.0e6)
    assert kind == "ValueError", (kind, msg)
    assert "eps0" in msg.lower() or "plane" in msg.lower(), msg


def test_an_empty_declaration_is_not_a_declaration():
    r"""
    An empty ``sigma_pe`` declares nothing.  Refused at construction:
    the remedy is to omit ``already_bonded`` entirely, which is a
    different statement about the section.
    """
    cls = _declaration_cls()
    assert cls is not None
    kind, _ = _outcome(cls, sigma_pe={}, eps0=0.0, chi_x=0.0, chi_y=0.0,
                       x_ref=X_REF_DECL, y_ref=Y_REF_DECL,
                       concrete_age_days=AGE_C, tendon_age_days={})
    assert kind == "ValueError", kind


@pytest.mark.parametrize("bad", [0.0, -50.0, float("nan"),
                                 float("inf")])
def test_a_non_physical_effective_stress_is_refused(bad):
    r"""
    A tendon at zero or negative effective stress is not a prestressed
    tendon, and a non-finite one is not a number.  Either would run to
    completion and publish a wrong capacity in silence.
    """
    cls = _declaration_cls()
    assert cls is not None
    kind, _ = _outcome(cls, sigma_pe={0: bad}, eps0=0.0, chi_x=0.0,
                       chi_y=0.0, x_ref=X_REF_DECL, y_ref=Y_REF_DECL,
                       concrete_age_days=AGE_C,
                       tendon_age_days={0: AGE_P})
    assert kind == "ValueError", (bad, kind)


def test_an_out_of_range_tendon_index_is_a_use_error():
    r"""Distinct from the modelling boundary, as it was before."""
    state = _declare({9: SIGMA_PE}, EPS0_EX, CHIX_EX)
    kind, _ = _outcome(_run, sigma_p0=[SP0] * 3, Ep=EP, order=[0, 1, 2],
                       already_bonded=state)
    assert kind == "ValueError", kind


# ===========================================================================
#  5. Grouting the new tendon leaves the existing one in the domain
# ===========================================================================

def test_grouting_the_new_tendon_keeps_the_existing_one_bonded():
    r"""
    The existing tendon was bonded before the sequence began and is
    bonded after it ends.  A grouting transition that rebuilt the state
    from "every tendon an inactive load" would silently drop it out of
    the resistance domain — the section would lose a cable that is
    physically there.
    """
    sec = _section()
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    res = pt.solve_posttension_sequence(
        sec, sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2],
        pending=[], already_bonded=state)
    kind, gres = _outcome(pt.grout, sec, res, indices=[1])
    assert kind == "ok", gres
    n_reb = int(sec.x_rebars.size)
    assert bool(gres.state.bonded[n_reb + 0]), "existing tendon dropped"
    assert bool(gres.state.active[n_reb + 0]), "existing tendon dropped"
    assert bool(gres.state.bonded[n_reb + 1])
    assert not bool(gres.state.bonded[n_reb + 2])


def test_the_existing_tendon_reads_back_its_own_effective_stress():
    r"""
    Its reconciled datum must make it evaluate at
    :math:`\sigma_{pe} - \Delta\sigma`, the post-loss stress the
    sequence computed — the same contract a tendon GenSec jacked
    itself gets from :func:`~gensec.solver.posttension.grout`.
    """
    sec = _section()
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    res = pt.solve_posttension_sequence(
        sec, sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2],
        pending=[], already_bonded=state)
    gres = pt.grout(sec, res, indices=[1])
    rows = {int(r["tendon"]): r for r in gres.report["reconciliation"]}
    assert 0 in rows, "the existing tendon carries no reconciliation row"
    row = rows[0]
    read_back = EP * (row["eps_init"] + row["eps_ref_grout"])
    assert read_back == pytest.approx(float(res.sigma_p_after[0]),
                                      rel=1e-9)


def test_the_existing_tendon_has_no_prestress_action_to_drop():
    r"""
    Its force is not a demand of *this* sequence: it was applied in a
    history GenSec did not see, and the declared concrete plane is
    where its effect on the section is stated.  So grouting has
    nothing to drop for it, and the single-side invariant must hold
    all the same.
    """
    sec = _section()
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    res = pt.solve_posttension_sequence(
        sec, sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2],
        pending=[], already_bonded=state)
    gres = pt.grout(sec, res, indices=[1])
    dropped = [a.label for a in gres.dropped_actions]
    assert not any("tendon0" in lbl for lbl in dropped), dropped
    residual = [a.label for a in gres.residual_actions]
    assert not any("tendon0" in lbl for lbl in residual), residual


# ===========================================================================
#  6. Behavioural control — green on BOTH trees
# ===========================================================================

def test_the_deferred_grouting_model_is_untouched():
    r"""
    Green before the repair and after it.  The overwhelming majority
    of post-tensioned decks are stressed in sequence and grouted once
    at the end, and this repair must not have moved a single one of
    their numbers.  Measured against the closed form, which is
    tree-independent.
    """
    res = _run(sigma_p0=[SP0] * 3, Ep=EP, order=[0, 1, 2])
    ref = _closed_form_retrofit(_section(), [], [0, 1, 2],
                                [SP0] * 3, {})
    assert res.sigma_p_after == pytest.approx(ref["sigma_p_after"],
                                              rel=1e-9)
    assert res.eps_ref_grout == pytest.approx(ref["eps_ref_grout"],
                                              rel=1e-9)


def test_the_phase22_staged_injection_is_untouched():
    r"""
    Green before the repair and after it.  Phase 22's staged schedule
    shares the resisting-view machinery this phase extends, so it is
    the case most exposed to an accidental change.

    The reference is
    :func:`~gensec.solver.prestress_transfer.sequential_posttension_loss`,
    which has carried the ``bonded_before`` hook in closed form since
    Phase 4 — **not** a number copied out of this driver.  A fixture
    that asserted the driver's own output would go green on any tree
    where the driver was self-consistent, including a tree where it
    was consistently wrong.
    """
    from gensec.solver.prestress_transfer import sequential_posttension_loss

    sec = _section()
    y_ref = sec.y_centroid
    Ac, Sc, Ic = _meshed_props(sec, y_ref)
    schedule = {0: 1, 1: 2}
    ref = sequential_posttension_loss(
        EC, 0.0, EP, Ac=Ac, Sc=Sc, Ic=Ic,
        tendons=[{"y": float(yt), "Ap": AP, "sigma_p0": SP0,
                  "bonded_before": schedule.get(k)}
                 for k, yt in enumerate(sec.y_tendons)],
        order=[0, 1, 2], y_ref=y_ref, scheme="one_pass")
    res = _run(sigma_p0=[SP0] * 3, Ep=EP, order=[0, 1, 2],
               bonded_before=schedule)
    assert res.sigma_p_after == pytest.approx(ref["sigma_p_after"],
                                              rel=1e-9)


def test_the_undeclared_tendon_refusal_still_fires_and_keeps_its_name():
    r"""
    Green on both trees.  A strict subset of ``order`` with nothing
    declared is still :class:`IntraSequenceUndeclared`; Phase 24 adds a
    role to the declaration, it does not remove the requirement that
    every tendon have one.
    """
    got = _reason(_run, sigma_p0=[SP0] * 3, Ep=EP, order=[0, 1])
    assert got == "intra_sequence_undeclared", got


# ===========================================================================
#  7. applied_at — a sollecitazione interposed in the sequence
# ===========================================================================

def test_a_sollecitazione_before_the_first_jack_debits_the_existing_tendon():
    r"""
    The fact that made ``applied_at`` a separate parameter rather than a
    reinterpretation of ``base_N``: a load arriving on a section that
    already carries a grouted cable costs that cable stress, because it
    is strain-compatible with the concrete around it.  ``base_N`` is
    debit-free by contract and would have swallowed that silently.
    """
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    free = _run(sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2],
                pending=[], already_bonded=state)
    loaded = _run(sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2],
                  pending=[], already_bonded=state,
                  applied_at={0: (-2.0e6, 0.0, 0.0)})
    assert loaded.sigma_p_after[0] != pytest.approx(
        float(free.sigma_p_after[0]))
    # Compression shortens the section, so a grouted cable loses stress.
    assert loaded.sigma_p_after[0] < free.sigma_p_after[0]


def test_the_interposed_sollecitazione_matches_the_closed_form():
    r"""
    Same 2x2 transformed-section algebra, one extra right-hand side.
    The debit lands on every anchored tendon and on no other.
    """
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    N_a = -2.0e6
    res = _run(sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2],
               pending=[], already_bonded=state,
               applied_at={0: (N_a, 0.0, 0.0)})
    sec = _section()
    y_ref = sec.y_centroid
    Ac, Sc, Ic = _meshed_props(sec, y_ref)
    d = np.asarray(sec.y_tendons, dtype=float) - y_ref
    EA = EC * Ac + (EP - EC) * AP
    ES = EC * Sc + (EP - EC) * AP * d[0]
    EI = EC * Ic + (EP - EC) * AP * d[0] * d[0]
    det = EA * EI - ES * ES
    de0 = (EI * N_a) / det
    dchi = (-ES * N_a) / det
    extra = -EP * (de0 + dchi * d[0])
    ref = _closed_form_retrofit(_section(), [0], [1, 2],
                                [NAN, SP0, SP0], {0: SIGMA_PE},
                                (EPS0_EX, CHIX_EX))
    assert float(res.loss[0]) == pytest.approx(
        float(ref["loss"][0]) + extra, rel=1e-9)


def test_a_sollecitazione_while_a_cable_is_open_is_refused_by_name():
    r"""
    Tendon 1 is jacked and not grouted when the load arrives at step 2.
    It slides in its duct, so its force change follows the mean
    elongation over the free length — a member quantity.  Refused, and
    the remedy is grouting, not abandoning the model.
    """
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    kind, msg = _outcome(_run, sigma_p0=[NAN, SP0, SP0], Ep=EP,
                         order=[1, 2], pending=[], already_bonded=state,
                         applied_at={1: (-2.0e6, 0.0, 0.0)})
    assert kind == "LoadBetweenJacksNotModelled", (kind, msg)
    assert "1" in msg


def test_grouting_the_open_cable_first_makes_the_same_load_modelled():
    r"""
    The remedy has to work, or naming it is theatre.  Same load, same
    step; the only change is that tendon 1 is grouted before it arrives.
    """
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    kind, res = _outcome(_run, sigma_p0=[NAN, SP0, SP0], Ep=EP,
                         order=[1, 2], pending=[], already_bonded=state,
                         bonded_before={1: 1},
                         applied_at={1: (-2.0e6, 0.0, 0.0)})
    assert kind == "ok", res
    # Both grouted cables paid; the one still to be jacked did not.
    assert res.loss[0] > 0.0
    assert res.loss[1] > 0.0
    assert float(res.loss[2]) == pytest.approx(0.0, abs=1e-12)


def test_applied_at_step_zero_without_a_retrofit_is_base_under_another_name():
    r"""
    Nothing anchored means nothing debited means it *is* ``base_N``.
    Two names for one fact is how the wrong one survives.
    """
    kind, msg = _outcome(_run, sigma_p0=[SP0] * 3, Ep=EP, order=[0, 1, 2],
                         applied_at={0: (-2.0e6, 0.0, 0.0)})
    assert kind == "ValueError", (kind, msg)
    assert "base_n" in msg.lower(), msg


def test_a_step_outside_the_sequence_is_refused():
    r"""After the last jack is not *interposed*, and has no debit to
    contaminate."""
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    kind, _ = _outcome(_run, sigma_p0=[NAN, SP0, SP0], Ep=EP,
                       order=[1, 2], pending=[], already_bonded=state,
                       applied_at={2: (-1.0e6, 0.0, 0.0)})
    assert kind == "ValueError", kind


def test_the_coupled_scheme_does_not_scale_the_applied_sollecitazione():
    r"""
    The fixed point iterates on the tendon stresses.  An external
    sollecitazione does not depend on them, so it must arrive at full
    value on every sweep — a load reduced by the cables' own losses
    would be a quiet mismodel with no symptom.
    """
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    kw = dict(sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2], pending=[],
              already_bonded=state)
    one = _run(scheme="one_pass", applied_at={0: (-2.0e6, 0.0, 0.0)}, **kw)
    cpl = _run(scheme="coupled", applied_at={0: (-2.0e6, 0.0, 0.0)}, **kw)
    base_one = _run(scheme="one_pass", **kw)
    base_cpl = _run(scheme="coupled", **kw)
    # The load's own contribution is the same under both schemes; only
    # the jack-driven part differs between them.
    assert (float(one.loss[0]) - float(base_one.loss[0])) == pytest.approx(
        float(cpl.loss[0]) - float(base_cpl.loss[0]), rel=1e-6)


# ===========================================================================
#  8. The two clocks — declared, carried, not consumed here
# ===========================================================================

def test_the_two_clocks_are_separate_facts():
    r"""
    Relaxation runs from the day a cable was tensioned; creep and
    shrinkage run from the day its concrete was cast.  On a member being
    retrofitted the two are decades apart, and one number could not
    stand for both.
    """
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    assert float(state.concrete_age_days) == AGE_C
    assert float(state.tendon_age_days[0]) == AGE_P
    assert state.concrete_age_days != state.tendon_age_days[0]


def test_a_clock_without_its_stress_is_refused():
    r"""Every declared tendon needs both, or the pair describes nothing."""
    cls = _declaration_cls()
    assert cls is not None
    kind, _ = _outcome(cls, sigma_pe={0: SIGMA_PE}, eps0=0.0, chi_x=0.0,
                       chi_y=0.0, x_ref=X_REF_DECL, y_ref=Y_REF_DECL,
                       concrete_age_days=AGE_C,
                       tendon_age_days={0: AGE_P, 1: AGE_P})
    assert kind == "ValueError", kind


def test_the_declaration_reaches_the_result():
    r"""
    Carried on the result because the clocks have no consumer *here* and
    every consumer they will have needs them: a slot that arrived empty
    would be filled with a zero by the first caller who forgot.
    """
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    res = _run(sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2],
               pending=[], already_bonded=state)
    assert res.preexisting is not None
    assert float(res.preexisting.concrete_age_days) == AGE_C
    assert float(res.preexisting.tendon_age_days[0]) == AGE_P


# ===========================================================================
#  9. The exception taxonomy says what kind of refusal it is
# ===========================================================================

def test_an_undeclared_input_is_a_value_error_not_an_unmodelled_physics():
    r"""
    The CLI reads the difference: a :class:`NotImplementedError` prints
    "outside GenSec's modelled scope", which is a false statement about
    a case that is modelled the moment the numbers arrive.
    """
    assert issubclass(pt.PreexistingStateUndeclared, ValueError)
    assert issubclass(pt.IntraSequenceUndeclared, ValueError)
    assert not issubclass(pt.PreexistingStateUndeclared,
                          NotImplementedError)


def test_unmodelled_physics_keeps_its_own_type():
    r"""
    An ungrouted tendon really is out of a sectional model's reach, and
    that refusal must not be flattened into "you forgot to declare
    something".
    """
    assert issubclass(pt.LoadBetweenJacksNotModelled, NotImplementedError)
    assert not issubclass(pt.LoadBetweenJacksNotModelled, ValueError)


# ===========================================================================
# 10. The declared plane is read about the point it was declared about
# ===========================================================================

def test_the_plane_is_read_about_its_own_declared_reference_point():
    r"""
    The driver pins its reference at the centroid of the tendon-free
    view, which the caller has no way to know.  A declaration read about
    the driver's point instead of its own would be wrong by
    :math:`\chi\,\Delta y` — here about 5 % of the declared strain,
    with nothing to show for it.
    """
    state = _declare({0: SIGMA_PE}, EPS0_EX, CHIX_EX)
    res = _run(sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2],
               pending=[], already_bonded=state)
    free = _run(sigma_p0=[NAN, SP0, SP0], Ep=EP, order=[1, 2],
                pending=[], already_bonded=_declare({0: SIGMA_PE},
                                                    0.0, 0.0))
    yv = np.asarray(_section().y_tendons, dtype=float)
    expected = EPS0_EX + CHIX_EX * (yv - Y_REF_DECL)
    assert (res.eps_ref_grout - free.eps_ref_grout) == pytest.approx(
        expected, rel=1e-9, abs=1e-15)
    assert float(res.y_ref) != Y_REF_DECL, (
        "the driver's reference and the declared one coincide here, so "
        "this fixture cannot tell them apart")
