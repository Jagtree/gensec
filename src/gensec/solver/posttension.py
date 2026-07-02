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
Sequential **bonded post-tensioning** driver (prestress Phase 4 / Step C).

This module is the fiber-method peer of
:mod:`gensec.solver.prestress_transfer` for *post-tensioning*.  Where
pre-tensioning releases all strands **simultaneously** (the loss
*emerges* from a single equilibrium solve), post-tensioning stresses
the tendons **sequentially**, each against the already-hardened
concrete; the elastic-shortening loss therefore **cannot emerge** from
one solve — it must be read and debited event by event.

The pre/post asymmetry, in one paragraph
----------------------------------------
A post-tensioned tendon being stressed at step *k* is **not** a section
element: it is a free cable, jacked to its target stress exactly, the
jack reacting on the section as the external action

.. math::

    (N_k,\,M_{x,k},\,M_{y,k})
        = \bigl(-P_k,\; -P_k\,(y_k - y_\text{ref}),\;
                +P_k\,(x_k - x_\text{ref})\bigr),
    \qquad P_k = \sigma_{p0,k}\,A_{p,k},

i.e. a :class:`~gensec.solver.section_state.PrestressAction`
(:meth:`PrestressAction.from_force`).  Consequently:

* the tendon stressed at step *k* suffers **no** loss from its own
  stressing (it is not strain-compatible with the concrete at that
  instant);
* every tendon *j* stressed **before** *k* and already anchored takes
  the concrete strain increment at its level and is debited

  .. math::

      \Delta\sigma_{p,j} = -E_p\,\Delta\varepsilon_{c,j},

  where :math:`\Delta\varepsilon_{c,j}` is read from the **solved
  strain plane** of the step-*k* event at the coordinate of tendon *j*.

Because the action is a *fixed* triple, the loss is **not emergent**: it
is computed explicitly from the plane.  This is the central distinction
from :func:`gensec.solver.prestress_transfer.solve_pretension_transfer`.

Relationship to the closed-form validator
------------------------------------------
For a **linear-elastic** transfer section the fiber solve reproduces, to
machine precision, the planar transformed-section algebra of
:func:`gensec.solver.prestress_transfer.sequential_posttension_loss`
(same linear system, two code paths).  That closed form is this
driver's **acceptance test** — see ``tests/test_posttension.py`` and
``run_posttension_sequence_validation.py``.  In production the bulk may
be nonlinear; the driver is material-agnostic (it only reads strain
planes), but only the linear case is checkable against the closed form.

Decision locked at the start of this driver (per the Step-C spec)
-----------------------------------------------------------------
**Elastic-shortening sequence fidelity** — the default is the *explicit
one-pass* debit (EC2 §5.10.5.1 / NTC 2018 §4.1.8.1.4 level, the
:math:`(n-1)/2n` mean-debit rule for *n* identical centroidal tendons).
The *self-consistent / coupled* scheme (``scheme="coupled"``) is
**opt-in**: it iterates because debiting a tendon reduces its action,
which reduces the shortening — a second-order feedback worth ~0.3 %
(quantified by the validator), the very reason EC2 stops at the explicit
form.  Both fidelity levels live behind one signature; Phase 5
(time-dependent losses) will later realise the *same* code-level
interface (state → effective prestrain) while staying a conceptually
separate, interval-triggered operator.

The single-side prestress invariant (grouting)
----------------------------------------------
A tendon is a load **xor** a resistance element, never both, never
neither.  At the grouting stage the rigorous transition is **atomic**
(:func:`grout`): the tendon's :class:`PrestressAction` is dropped from
the demand **and** the tendon enters the resistance domain
(:meth:`SectionState.with_grouted`: ``bonded`` flipped ``True`` and a
*reconciled* ``eps_init`` baked in) in one step.  The conservative
treatment simply never calls :func:`grout`: the
:class:`PrestressAction` stays a load for life.

References
----------
- EN 1992-1-1:2004 §5.10.5.1 (immediate losses — elastic deformation of
  concrete; sequential post-tensioning).
- NTC 2018 §4.1.8.1.4 (perdite istantanee per deformazione elastica).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .integrator import FiberSolver
from .section_state import (
    PrestressAction,
    SectionState,
    materialize_view,
)


# ===========================================================================
#  Result containers
# ===========================================================================

@dataclass
class PosttensionStageResult:
    r"""
    Outcome of one stressing **stage** (one tendon jacked).

    Attributes
    ----------
    stage_index : int
        0-based position in the stressing sequence.
    stressed_tendon : int
        Index (into ``section.tendons``) of the tendon jacked at this
        stage.
    action : PrestressAction
        The **incremental** action applied at this stage (the jacked
        tendon's couple, at the effective jacking stress used by the
        active scheme).
    cumulative_demand : tuple of float
        Running prestress demand ``(N, Mx, My)`` after this stage
        [N, N·mm, N·mm] — the sum of the base sollecitazione and every
        action applied up to and including this stage.
    eps_c : numpy.ndarray
        Concrete section strain at **every** tendon location,
        accumulated through this stage [-] (shape ``(n_tendons,)``).
    sigma_p : numpy.ndarray
        Current effective strand stress at every tendon after this
        stage's debits [MPa] (shape ``(n_tendons,)``).
    converged : bool
        Whether the stage's equilibrium solve converged.
    report : dict
        Opaque, human-auditable payload for the stage ``report`` field
        (the values the engine deliberately surfaces rather than burying
        in the solver).
    """

    stage_index: int
    stressed_tendon: int
    action: PrestressAction
    cumulative_demand: Tuple[float, float, float]
    eps_c: np.ndarray
    sigma_p: np.ndarray
    converged: bool
    report: dict = field(default_factory=dict)


@dataclass
class PosttensionSequenceResult:
    r"""
    Aggregate outcome of a full sequential post-tensioning analysis.

    All per-tendon arrays are index-aligned with ``section.tendons``
    (declaration order), **not** with the stressing ``order``.

    Attributes
    ----------
    order : list of int
        Stressing order actually applied (tendon indices).
    scheme : {"one_pass", "coupled"}
        Elastic-shortening fidelity scheme used.
    sigma_p0 : numpy.ndarray
        Jacking stress per tendon [MPa] (the input, net of friction and
        anchorage slip).
    sigma_p_after : numpy.ndarray
        Effective strand stress after the whole sequence [MPa].
    loss : numpy.ndarray
        Cumulative immediate elastic-shortening loss per tendon
        :math:`\Delta\sigma_p = \sigma_{p0}-\sigma_{p,\text{after}}`
        [MPa] (positive = loss of tension; may be **negative** for a
        tendon whose stress is *raised* by an opposite-side eccentric
        event).
    eps_pe_eff : numpy.ndarray
        Effective jacking strain per tendon after losses,
        :math:`\sigma_{p,\text{after}}/E_p` [-] — the prestrain the
        grouting stage bakes into the resistance.
    eps_ref_grout : numpy.ndarray
        Concrete section strain at each tendon at the **end** of the
        sequence [-] — the datum the grouting reconciliation consumes
        so that :math:`\sigma_p(\varepsilon_{\text{sec,grout}}
        + \varepsilon_{\text{init}})` reproduces
        ``sigma_p_after``.
    Ep : numpy.ndarray
        Tendon modulus per tendon [MPa].
    Ap : numpy.ndarray
        Tendon area per tendon [mm²].
    x_ref, y_ref : float
        Reference point the actions and strains are taken about [mm].
    stages : list of PosttensionStageResult
        Per-stage history (length = number of stressed tendons).
    converged : bool
        Whether every stage's equilibrium solve converged.
    coupled_iterations : int
        Fixed-point sweeps used by ``scheme="coupled"`` (1 for
        ``one_pass``).
    """

    order: List[int]
    scheme: str
    sigma_p0: np.ndarray
    sigma_p_after: np.ndarray
    loss: np.ndarray
    eps_pe_eff: np.ndarray
    eps_ref_grout: np.ndarray
    Ep: np.ndarray
    Ap: np.ndarray
    x_ref: float
    y_ref: float
    stages: List[PosttensionStageResult]
    converged: bool
    coupled_iterations: int

    # -- derived, demand-side artefacts ------------------------------------

    def effective_actions(self) -> List[PrestressAction]:
        r"""
        Per-tendon life-long load actions at the **effective**
        (post-loss) strand stress.

        These are the demand-side loads of the **conservative**
        treatment (tendon never grouted): the section carries each
        tendon's couple at :math:`\sigma_{p,\text{after}}` for life.

        Returns
        -------
        list of PrestressAction
            One action per tendon, aligned with ``section.tendons``.
        """
        acts = []
        for j in range(self.sigma_p_after.size):
            P = float(self.sigma_p_after[j] * self.Ap[j])
            acts.append(PrestressAction.from_force(
                P, self._x[j], self._y[j],
                x_ref=self.x_ref, y_ref=self.y_ref,
                label=f"tendon{j}", origin="posttension_effective"))
        return acts

    def total_effective_action(self) -> PrestressAction:
        r"""Resultant of all :meth:`effective_actions` as one triple."""
        N = Mx = My = 0.0
        for a in self.effective_actions():
            N += a.N
            Mx += a.Mx
            My += a.My
        return PrestressAction(N, Mx, My, label="prestress_total",
                               origin="posttension_effective")

    # ``_x`` / ``_y`` are stamped by the driver (tendon coordinates).
    _x: np.ndarray = field(default=None, repr=False)
    _y: np.ndarray = field(default=None, repr=False)


@dataclass
class GroutResult:
    r"""
    Outcome of the rigorous (bonded) grouting transition.

    Attributes
    ----------
    state : SectionState
        The post-grouting section state: grouted tendons are
        ``active & bonded`` with their reconciled ``eps_init``; tendons
        left ungrouted stay inactive (their force remains a demand-side
        load).  Its capacity hash differs from the pre-grout state, so a
        :class:`~gensec.solver.section_state.StagedDomainManager` rebuilds
        the resistance domain automatically.
    grouted : list of int
        Tendon indices brought into the domain.
    dropped_actions : list of PrestressAction
        The loads removed from the demand at grouting (one per grouted
        tendon).  **These must be subtracted from the cumulative demand**
        in the same stage they enter the domain — that is the
        single-side invariant made explicit.
    residual_actions : list of PrestressAction
        The loads that **remain** in the demand (ungrouted tendons), at
        their effective post-loss stress.
    report : dict
        Per-grouted-tendon reconciliation record (visible correction):
        ``eps_ref_grout``, ``eps_pe_eff``, ``eps_init``,
        ``sigma_p_target`` and ``beyond_linear`` flag.
    """

    state: SectionState
    grouted: List[int]
    dropped_actions: List[PrestressAction]
    residual_actions: List[PrestressAction]
    report: dict


# ===========================================================================
#  Internal helpers
# ===========================================================================

def _tendon_geometry(section) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return ``(x, y, Ap)`` arrays for the section's tendons [mm, mm, mm²]."""
    tendons = getattr(section, "tendons", [])
    if not tendons:
        raise ValueError(
            "solve_posttension_sequence: section carries no tendons. "
            "Build the section with a non-empty `tendons` list "
            "(the post-tension cables, declared bonded=True).")
    x = np.asarray(section.x_tendons, dtype=float)
    y = np.asarray(section.y_tendons, dtype=float)
    Ap = np.asarray(section.A_tendons, dtype=float)
    return x, y, Ap


def _resolve_Ep(section, Ep, nt) -> np.ndarray:
    r"""
    Resolve the per-tendon modulus array.

    Order of precedence: explicit *Ep* argument (scalar broadcast or
    array), else each tendon material's ``Ep`` / ``E`` attribute.  Raises
    if a modulus cannot be resolved (no silent default).
    """
    if Ep is not None:
        arr = np.broadcast_to(np.asarray(Ep, dtype=float), (nt,)).astype(float)
        return arr.copy()
    out = np.empty(nt, dtype=float)
    for j, t in enumerate(section.tendons):
        mat = t.material
        E = getattr(mat, "Ep", None)
        if E is None:
            E = getattr(mat, "E", None)
        if E is None:
            raise ValueError(
                f"solve_posttension_sequence: tendon {j} material exposes "
                f"neither `Ep` nor `E`; pass an explicit `Ep=` argument.")
        out[j] = float(E)
    return out


def _stressing_view(section):
    r"""
    Build the section **view** that resists during stressing: bulk +
    passive rebars, with **all tendons excluded**.

    During the stressing sequence no tendon is bonded (the default
    deferred-grouting model), so the resisting section is exactly the
    hardened concrete plus passive reinforcement.  The tendons are loads,
    not elements: they must not contribute stiffness to the solve.  This
    is enforced by materializing a state in which every tendon is
    inactive (hence omitted from the view by
    :func:`~gensec.solver.section_state.materialize_view`).

    Returns
    -------
    GenericSection
        The materialized resisting view (no tendons).
    """
    n_reb = int(section.x_rebars.size)
    n_ten = int(getattr(section, "x_tendons", np.empty(0)).size)
    n = n_reb + n_ten
    active = np.ones(n, dtype=bool)
    bonded = np.ones(n, dtype=bool)
    if n_ten:
        active[n_reb:] = False          # tendons are loads while stressing
        bonded[n_reb:] = False
    state = SectionState(
        stage_index=0,
        active=active,
        bonded=bonded,
        eps_init=np.zeros(n, dtype=float),
        bulk_eps_init=float(getattr(section, "bulk_eps_init", 0.0)),
        label="stressing",
    )
    return materialize_view(section, state)


def _grouted_state(section, grouted_local, eps_init_local):
    r"""
    Build the post-grouting :class:`SectionState`.

    Constructs the base state (bulk + rebars active/bonded, every
    tendon an inactive load) and hands the grouted tendons to the
    atomic primitive :meth:`SectionState.with_grouted`, which flips
    ``active``/``bonded`` and sets the reconciled ``eps_init``
    together and enforces the single-side / no-re-grouting invariant.

    Parameters
    ----------
    section : GenericSection
    grouted_local : sequence of int
        Tendon-local indices to grout.
    eps_init_local : sequence of float
        Reconciled ``eps_init`` per grouted tendon (aligned with
        *grouted_local*); see :func:`reconcile_grouting`.

    Returns
    -------
    SectionState
    """
    n_reb = int(section.x_rebars.size)
    n_ten = int(section.x_tendons.size)
    n = n_reb + n_ten
    base = SectionState(
        stage_index=0,
        active=np.concatenate([np.ones(n_reb, bool), np.zeros(n_ten, bool)]),
        bonded=np.concatenate([np.ones(n_reb, bool), np.zeros(n_ten, bool)]),
        eps_init=np.zeros(n, dtype=float),
        bulk_eps_init=float(getattr(section, "bulk_eps_init", 0.0)),
        label="grout",
    )
    union_idx = [n_reb + int(j) for j in grouted_local]
    eps_map = {ui: float(e) for ui, e in zip(union_idx, eps_init_local)}
    return base.with_grouted(union_idx, eps_map)


# ===========================================================================
#  Driver
# ===========================================================================

def solve_posttension_sequence(
    section,
    *,
    sigma_p0: Sequence[float],
    Ep=None,
    order: Optional[Sequence[int]] = None,
    already_bonded: Optional[Sequence[int]] = None,
    base_N: float = 0.0,
    base_Mx: float = 0.0,
    base_My: float = 0.0,
    x_ref: Optional[float] = None,
    y_ref: Optional[float] = None,
    scheme: str = "one_pass",
    coupled_tol: float = 1e-12,
    coupled_max_iter: int = 50,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> PosttensionSequenceResult:
    r"""
    Sequential bonded post-tensioning on a single existing bulk.

    Orchestrates the stressing sequence in the fiber model: for each
    event it applies the jacked tendon's couple as a
    :class:`~gensec.solver.section_state.PrestressAction`, solves the
    section, reads the concrete strain increment at every
    already-anchored tendon, and debits

    .. math::

        \Delta\sigma_{p,j} = -E_{p,j}\,\Delta\varepsilon_{c,j}.

    The resisting section while stressing is the hardened bulk plus
    passive reinforcement; the tendons are loads, **not** elements, until
    grouting (see :func:`grout`).

    Parameters
    ----------
    section : GenericSection
        A meshed section whose ``tendons`` carry the post-tension cables
        (declared ``bonded=True``; their not-yet-grouted state is the
        per-stage mask handled here).
    sigma_p0 : sequence of float
        Jacking stress per tendon [MPa], **already net of friction and
        anchorage slip** (member-level effects, injected as input;
        out of scope for this sectional driver).  Aligned with
        ``section.tendons``.
    Ep : float or sequence of float, optional
        Tendon modulus [MPa].  Default: each tendon material's ``Ep``
        (or ``E``) attribute.
    order : sequence of int, optional
        Stressing order, as tendon indices.  Default: declaration order
        ``range(n_tendons)``.
    already_bonded : sequence of int, optional
        Tendon indices already grouted (bonded, in the resistance
        domain) **before** this stressing sequence begins.  Reserved
        for the intra-sequence bonded-stiffness extension and currently
        **not implemented**: a non-empty value raises
        :class:`NotImplementedError` rather than silently ignoring the
        added stiffness.  See ``WARNING_intra_sequence_bonded.md``.
    base_N, base_Mx, base_My : float, optional
        Base sollecitazione present at transfer (e.g. self-weight)
        [N, N·mm, N·mm].  Applied once, **debit-free**: it contributes
        to ``eps_ref_grout`` but never to the loss (only the *shortening*
        from a stressing event is a loss; load-induced strain is not).
        GenSec takes sollecitazioni, never loads — pass the resultant.
    x_ref, y_ref : float, optional
        Reference point for actions and strains [mm].  Default: section
        centroid.
    scheme : {"one_pass", "coupled"}, optional
        Elastic-shortening fidelity.  ``"one_pass"`` (default): each
        event debits once (EC2 §5.10.5.1 level).  ``"coupled"``:
        fixed-point iteration over ``sigma_eff = sigma_p0 - loss``,
        capturing the second-order feedback.
    coupled_tol, coupled_max_iter : float, int, optional
        Convergence control for ``scheme="coupled"`` (max change in
        per-tendon stress between sweeps [MPa]).
    tol, max_iter : float, int, optional
        Force tolerance [N] and iteration cap for each equilibrium solve.

    Returns
    -------
    PosttensionSequenceResult

    Raises
    ------
    ValueError
        If the section carries no tendons, if a tendon modulus cannot be
        resolved, or if ``scheme`` is unknown.

    Notes
    -----
    **Precondition on a stressing stage (the check that stops).** A
    stressing event varies the prestress; the base sollecitazione is
    applied separately and once.  This driver therefore never mixes a
    prestress increment with an external-load increment within one
    event, so the read :math:`\Delta\varepsilon_{c,j}` is pure
    shortening.  The YAML stage layer (deferred — see the Step-C status
    note) must enforce the same rule: a stage that varies **both**
    prestress and external sollecitazione must raise rather than report a
    polluted loss.

    **Why incremental solves reproduce the closed form.** For a
    linear-elastic transfer section, applying tendon *k*'s action alone
    and reading its plane increment is, by superposition, identical to
    applying the cumulative action set and differencing against the
    previous step.  The incremental form is implemented because the loss
    is defined *per event* and it is the exact algebra of
    :func:`~gensec.solver.prestress_transfer.sequential_posttension_loss`.
    """
    x, y, Ap = _tendon_geometry(section)
    nt = x.size
    sp0 = np.asarray(sigma_p0, dtype=float)
    if sp0.shape != (nt,):
        raise ValueError(
            f"solve_posttension_sequence: sigma_p0 has shape {sp0.shape}, "
            f"expected ({nt},) — one jacking stress per tendon.")
    Epv = _resolve_Ep(section, Ep, nt)
    if order is None:
        order = list(range(nt))
    order = [int(o) for o in order]
    # --- ordering guards -------------------------------------------------
    order_set = set(order)
    if len(order) != len(order_set) or not order_set.issubset(range(nt)):
        # Genuine input error: duplicates or out-of-range indices.
        raise ValueError(
            "solve_posttension_sequence: `order` must list each tendon index "
            f"in range({nt}) exactly once; got {order}.")
    if order_set != set(range(nt)):
        # A strict subset: some section tendons are not stressed here, which
        # only makes sense if they are already bonded and stiffen the section
        # the rest are stressed against — the intra-sequence case this driver
        # does not model.  Refuse rather than compute a wrong stiffness.
        missing = sorted(set(range(nt)) - order_set)
        raise NotImplementedError(
            f"solve_posttension_sequence: tendons {missing} are absent from "
            "`order`. Stressing a subset implies the remainder are already "
            "grouted (in the domain) and stiffen the section the subset is "
            "stressed against — intra-sequence bonded stiffness is not yet "
            "supported (the resisting view is tendon-free and constant across "
            "the sequence). Stress all tendons here (deferred single "
            "grouting), or use the closed form "
            "`sequential_posttension_loss(..., tendons=[{'bonded_before': k}, "
            "...])` for the staged-injection case.")

    # --- intra-sequence bonded-stiffness refusal -------------------------
    # The section carries no signal distinguishing an already-grouted tendon
    # from a to-be-stressed one (Tendon.bonded is the final property, True for
    # both). `already_bonded` is the explicit input by which a caller would
    # declare the existing-bonded set; until the extension lands, a non-empty
    # value MUST raise — silently dropping the added stiffness would mis-model
    # the very stiffness it represents.
    if already_bonded:
        bad = sorted(set(int(i) for i in already_bonded))
        if not set(bad).issubset(range(nt)):
            raise ValueError(
                f"solve_posttension_sequence: `already_bonded` indices {bad} "
                f"are not a subset of range({nt}).")
        raise NotImplementedError(
            f"solve_posttension_sequence: `already_bonded`={bad} requests "
            "tendons already grouted before this sequence, whose bonded "
            "stiffness would change the section the new tendons are stressed "
            "against. Intra-sequence / pre-existing bonded stiffness is not yet "
            "implemented; passing a non-empty set must not silently ignore it. "
            "Model the existing-prestress resistance separately, and run this "
            "driver only for sequences on the tendon-free (unbonded) view. "
            "See WARNING_intra_sequence_bonded.md for the implementation plan.")

    view = _stressing_view(section)
    solver = FiberSolver(view, x_ref=x_ref, y_ref=y_ref)
    xr, yr = solver.x_ref, solver.y_ref
    ly = y - yr
    lx = x - xr

    def _plane_at_tendons(eps0, chi_x, chi_y):
        r"""Section strain at every tendon for a solved plane [-].

        Off-view evaluation of the same formula as
        :meth:`FiberSolver._tendon_section_strain`:

        .. math::

            \varepsilon_{\text{sec},i} = \varepsilon_0
                + \chi_x\,(y_i - y_\text{ref})
                - \chi_y\,(x_i - x_\text{ref}).
        """
        return eps0 + chi_x * ly - chi_y * lx

    def _solve(N, Mx, My):
        r"""Solve the stressing view for a demand triple; return plane."""
        if abs(N) < 1e-30 and abs(Mx) < 1e-30 and abs(My) < 1e-30:
            return 0.0, 0.0, 0.0, True
        s = solver.solve_equilibrium(N, Mx, My, tol=tol, max_iter=max_iter)
        return s["eps0"], s["chi_x"], s["chi_y"], bool(s["converged"])

    def _run_sequence(sigma_eff):
        r"""One forward sweep; returns (debit, eps_c, stages, ok)."""
        debit = np.zeros(nt)
        anchored = np.zeros(nt, dtype=bool)
        eps_c = np.zeros(nt)
        stages: List[PosttensionStageResult] = []
        ok = True

        # Base sollecitazione: applied once, debit-free.
        e0, cx, cy, c_ok = _solve(base_N, base_Mx, base_My)
        ok &= c_ok
        eps_c += _plane_at_tendons(e0, cx, cy)
        cum = np.array([base_N, base_Mx, base_My], dtype=float)

        for k, j in enumerate(order):
            P = float(sigma_eff[j] * Ap[j])
            act = PrestressAction.from_force(
                P, float(x[j]), float(y[j]), x_ref=xr, y_ref=yr,
                label=f"jack_tendon{j}", origin="posttension")
            e0, cx, cy, c_ok = _solve(act.N, act.Mx, act.My)
            ok &= c_ok
            d_eps = _plane_at_tendons(e0, cx, cy)
            eps_c += d_eps

            # Debit every already-anchored tendon (never the one being
            # stressed: it is not strain-compatible at this instant).
            mask = anchored.copy()
            mask[j] = False
            debit[mask] += -Epv[mask] * d_eps[mask]

            anchored[j] = True
            cum += np.array(act.triple(), dtype=float)

            sigma_now = sigma_eff - debit
            stages.append(PosttensionStageResult(
                stage_index=k,
                stressed_tendon=j,
                action=act,
                cumulative_demand=(float(cum[0]), float(cum[1]),
                                   float(cum[2])),
                eps_c=eps_c.copy(),
                sigma_p=sigma_now.copy(),
                converged=c_ok,
                report={
                    "stressed_tendon": j,
                    "sigma_eff_applied": float(sigma_eff[j]),
                    "d_eps_c_at_tendons": d_eps.copy(),
                },
            ))
        return debit, eps_c, stages, ok

    if scheme == "one_pass":
        debit, eps_c, stages, ok = _run_sequence(sp0.copy())
        coupled_iters = 1
    elif scheme == "coupled":
        sigma_eff = sp0.copy()
        prev = None
        coupled_iters = 0
        for _ in range(coupled_max_iter):
            coupled_iters += 1
            debit, eps_c, stages, ok = _run_sequence(sigma_eff)
            sigma_eff = sp0 - debit
            if prev is not None and np.max(np.abs(sigma_eff - prev)) < coupled_tol:
                break
            prev = sigma_eff.copy()
    else:
        raise ValueError(
            f"solve_posttension_sequence: unknown scheme {scheme!r}; "
            f"use 'one_pass' or 'coupled'.")

    sigma_after = sp0 - debit
    eps_pe_eff = sigma_after / Epv

    res = PosttensionSequenceResult(
        order=order,
        scheme=scheme,
        sigma_p0=sp0.copy(),
        sigma_p_after=sigma_after,
        loss=debit,
        eps_pe_eff=eps_pe_eff,
        eps_ref_grout=eps_c,
        Ep=Epv,
        Ap=Ap.copy(),
        x_ref=float(xr),
        y_ref=float(yr),
        stages=stages,
        converged=bool(ok),
        coupled_iterations=int(coupled_iters),
    )
    res._x = x.copy()
    res._y = y.copy()
    return res


# ===========================================================================
#  Grouting (rigorous transition) — the single-side invariant, atomic
# ===========================================================================

def grout(
    section,
    result: PosttensionSequenceResult,
    *,
    indices: Optional[Sequence[int]] = None,
    eps_c_linear_limit: Optional[float] = None,
) -> GroutResult:
    r"""
    Rigorous (bonded) grouting transition for selected tendons.

    Atomically (i) drops each grouted tendon's
    :class:`~gensec.solver.section_state.PrestressAction` from the demand
    and (ii) brings the tendon into the resistance domain via
    :meth:`SectionState.with_grouted` — ``bonded`` flipped ``True`` with
    a **reconciled** locked-in strain

    .. math::

        \varepsilon_{\text{init},j}
            = \varepsilon_{pe,\text{eff},j}
              - \varepsilon_{\text{sec,grout},j},

    so that the resistance evaluates
    :math:`\sigma_p(\varepsilon_{\text{sec,grout},j}
    + \varepsilon_{\text{init},j})
    = \sigma_p(\varepsilon_{pe,\text{eff},j})`, i.e. reproduces the
    effective post-loss stress at the grouting strain datum.  Here
    :math:`\varepsilon_{\text{sec,grout},j}` is ``eps_ref_grout[j]`` and
    :math:`\varepsilon_{pe,\text{eff},j}` is ``eps_pe_eff[j]``.

    The post-grout state's capacity hash differs from the pre-grout
    state, so a
    :class:`~gensec.solver.section_state.StagedDomainManager` recomputes
    the resistance domain automatically — there is no user flag.

    Parameters
    ----------
    section : GenericSection
        The same meshed section passed to
        :func:`solve_posttension_sequence`.
    result : PosttensionSequenceResult
        The sequence result whose effective prestrain is baked in.
    indices : sequence of int, optional
        Tendon indices to grout.  Default: **all** tendons (the usual
        "stress all, then grout all" lifecycle).
    eps_c_linear_limit : float, optional
        If given, the absolute concrete strain at a grouted tendon
        (:math:`|\varepsilon_{\text{sec,grout},j}|`) exceeding this value
        **raises**: the reconciliation premise (concrete on its linear
        branch at transfer) is violated and a silent reconciliation would
        be wrong.  If ``None``, the reconciliation proceeds and the
        maximum :math:`|\varepsilon_{\text{sec,grout}}|` is surfaced in
        the report (visible, not buried) for the engineer to judge.

    Returns
    -------
    GroutResult

    Raises
    ------
    ValueError
        If *indices* is not a subset of the tendon range, or if a grouted
        tendon's concrete strain exceeds *eps_c_linear_limit*.
    RuntimeError
        If the single-side invariant is violated (a defensive
        construction-time check; should be unreachable).

    Notes
    -----
    The **conservative** treatment is simply *not calling this function*:
    ``result.effective_actions()`` then remain demand-side loads for
    life.  Mixing the two — grouting a tendon *and* keeping its load — is
    the §F double-count this routine exists to prevent.
    """
    nt = result.sigma_p_after.size
    if indices is None:
        indices = list(range(nt))
    indices = [int(i) for i in indices]
    if not set(indices).issubset(range(nt)):
        raise ValueError(
            f"grout: indices {indices} not a subset of range({nt}).")
    grouted = sorted(set(indices))
    ungrouted = [j for j in range(nt) if j not in grouted]

    eps_ref = result.eps_ref_grout
    eps_pe = result.eps_pe_eff

    # Reconciliation + linear-branch guard.
    eps_init = []
    max_abs_eps = 0.0
    for j in grouted:
        sj = float(eps_ref[j])
        max_abs_eps = max(max_abs_eps, abs(sj))
        if eps_c_linear_limit is not None and abs(sj) > eps_c_linear_limit:
            raise ValueError(
                f"grout: tendon {j} concrete strain at grouting "
                f"{sj:.3e} exceeds the linear-branch limit "
                f"{eps_c_linear_limit:.3e}; reconciliation premise "
                f"(linear concrete at transfer) is violated.")
        eps_init.append(float(eps_pe[j]) - sj)

    state = _grouted_state(section, grouted, eps_init)

    # Demand-side bookkeeping: drop grouted loads, keep ungrouted loads.
    eff = result.effective_actions()
    dropped = [eff[j] for j in grouted]
    residual = [eff[j] for j in ungrouted]

    # --- single-side invariant, made explicit ---------------------------
    # Every tendon is in EXACTLY one camp.
    n_reb = int(section.x_rebars.size)
    bonded_local = {int(ui) - n_reb for ui in np.nonzero(state.bonded)[0]
                    if int(ui) >= n_reb}
    load_local = set(ungrouted)
    if not bonded_local.isdisjoint(load_local):
        raise RuntimeError(
            "grout: single-side invariant violated — a tendon is both in "
            "the resistance domain and a demand load (the §F double-count).")
    if bonded_local | load_local != set(range(nt)):
        raise RuntimeError(
            "grout: single-side invariant violated — a tendon is neither "
            "grouted nor a load (prestress lost).")

    report = {
        "grouted_tendons": grouted,
        "ungrouted_tendons": ungrouted,
        "reconciliation": [
            {
                "tendon": j,
                "eps_ref_grout": float(eps_ref[j]),
                "eps_pe_eff": float(eps_pe[j]),
                "eps_init": float(eps_pe[j]) - float(eps_ref[j]),
                "sigma_p_target": float(result.sigma_p_after[j]),
                "beyond_linear": (None if eps_c_linear_limit is None
                                  else abs(float(eps_ref[j]))
                                  > eps_c_linear_limit),
            }
            for j in grouted
        ],
        "max_abs_eps_sec_grout": max_abs_eps,
        "linear_limit": eps_c_linear_limit,
    }

    return GroutResult(
        state=state,
        grouted=grouted,
        dropped_actions=dropped,
        residual_actions=residual,
        report=report,
    )
