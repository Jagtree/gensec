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
Section-state evolution and capacity-domain caching (prestress Phase 3).

This module is the infrastructure layer that lets a *single* meshed
section carry a **history** of resistance states across construction
and loading stages, without re-meshing and without mutating the
immutable :class:`~gensec.geometry.section.GenericSection`.

Motivation
----------
The pre-Phase-3 pipeline assumes one immutable section maps to one
resistance domain for the lifetime of a verification run
(:class:`~gensec.solver.check.VerificationEngine` builds a single
:class:`~gensec.solver.check.DomainChecker` at construction).  Staged
loading therefore only ever **accumulated demand** against a fixed
domain.  This is correct for seismic-on-the-in-service-section, but
cannot represent staged construction (an element becoming active at a
later stage), prestress losses (the locked-in strain evolving in
time), or de-stressing (a bonded tendon being cut in a retrofit).

The model
---------
Three ideas, kept strictly separate:

1. **Immutable base section, mutable *view*.**  The bulk mesh is built
   once.  A :class:`SectionState` is a cheap descriptor — an
   *active/bonded* mask over the point elements plus a per-element
   ``eps_init`` override and a bulk ``eps_init`` — that
   :func:`materialize_view` turns into a shallow copy of the base
   section with the point-element arrays re-sliced and the prestrain
   arrays overwritten.  The bulk arrays are shared by reference; they
   never change in Phase 3.  This resolves the
   ``GenericSection.ideal_gross_properties`` immutable-section TODO:
   the base section *is* immutable by contract, and every materialized
   view owns a fresh (invalidated) property cache.

2. **The capacity state hash is a domain identity.**  Two states with
   equal hash have, by construction, identical resistance domains, so
   the domain and every output derived from it are reused.  The hash
   covers everything the domain depends on and **nothing else** — in
   particular ``PrestressAction`` loads do *not* enter it, because a
   prestress force is an external **action** (it moves the demand
   point), not a change to the section's **resistance** (only the
   locked-in ``eps_init`` baked into the fibers does that).  Mixing
   the two is the same double-counting trap flagged in
   ``compute_ideal_properties``.

3. **Capacity recompute is automatic from the hash, never a flag.**
   :class:`StagedDomainManager` owns a ``{hash -> DomainBundle}`` cache.
   ``QUANT_EPS`` is the single speed knob: a coarser quantum collapses
   nearby ``eps_init`` states onto one hash and one domain build.

Bonded vs unbonded (resistance vs load)
---------------------------------------
A **bonded** active element is strain-compatible with the section: its
``eps_init`` shifts its constitutive evaluation and it *enters the
resistance domain*.  An **unbonded** active element is not
strain-compatible; its force is a pure external action and it is
**excluded from the materialized view**, contributing instead a
``PrestressAction`` to the demand path.  This is why ``bonded`` is a
hash component: flipping it changes which elements are in the domain.

Deactivation semantics (force release)
--------------------------------------
Removing a bonded element from the active set is the reverse of
grouting activation and is **not** a free modelling toggle:

- *Clean removal* (unstressed / hypothetical element): just drop it.
- *Force release* (a stressed bonded element, e.g. a cut tendon):
  its locked-in net force, evaluated at the current converged strain
  state of the pre-removal section, discharges.  To preserve
  equilibrium on the **reduced** section the released force is
  re-applied as a :class:`PrestressAction` of opposite sign, which
  enters the deactivation stage's demand increment.  The element does
  not *become* a load — its removal *generates* one.  The hash changes
  (the active set changed), the domain recomputes, and the path
  resets.  See :func:`released_force_action`.

Path metrics and the hash
--------------------------
``eta_path_*`` are normalised *by the domain*.  A ray cast in the
normalised space of domain A is meaningless against domain B.
Therefore **a hash change resets the demand path**: at a stage where
the hash differs from the previous stage, the path base is reset to
the carried-over cumulative demand and measured on the new domain.  A
run of consecutive same-hash stages keeps the path rays valid and
reproduces the pre-Phase-3 ``_check_staged`` behaviour exactly as a
special case (see :mod:`gensec.solver.check`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import numpy as np


# ==================================================================
#  Speed knob
# ==================================================================

#: Quantum [-] used to bucket ``eps_init`` values when forming the
#: capacity state hash.  States whose prestrains agree to within
#: ``QUANT_EPS`` collapse onto a single domain build.  This is the
#: sole speed/accuracy trade-off of the cache: smaller is more
#: accurate (more rebuilds), larger is faster (more reuse).  At
#: ``1e-5`` the bucketing error on a steel/tendon stress is of order
#: ``E_p * QUANT_EPS ~ 195000 * 1e-5 ~ 2 MPa``, i.e. negligible against
#: ULS design strengths.
QUANT_EPS: float = 1e-5


def _quantize(eps: float, quantum: float = QUANT_EPS) -> int:
    r"""
    Bucket a strain onto an integer grid of step *quantum*.

    .. math::

        q(\varepsilon) = \operatorname{round}(\varepsilon / \Delta)

    Returning an :class:`int` (not a rounded float) guarantees that
    two strains in the same bucket hash identically regardless of
    floating-point representation.

    Parameters
    ----------
    eps : float
        Strain to quantize [-].
    quantum : float, optional
        Bucket width [-].  Default :data:`QUANT_EPS`.

    Returns
    -------
    int
    """
    return int(round(float(eps) / quantum))


# ==================================================================
#  Path-reset scheduler
# ==================================================================

def path_schedule(stage_hashes):
    r"""
    Build the per-stage path schedule from a sequence of capacity
    hashes.

    ``eta_path_*`` are normalised *by the domain*, so a ray cast in the
    normalised space of one domain is meaningless against another.  A
    **hash change therefore resets the demand path**: the schedule
    flags, per stage, whether the path base is the carried-over
    cumulative demand on a (new) domain (``reset=True``) or a
    continuation of the previous same-hash stage (``reset=False``).

    Parameters
    ----------
    stage_hashes : list
        ``stage_hashes[k]`` is the capacity hash of the section state
        stage ``k``'s demand is verified against.

    Returns
    -------
    list of dict
        One entry per stage: ``stage`` (int), ``domain_hash``,
        ``reset`` (bool), ``path_base`` (``"origin"`` at stage 0,
        ``"carry"`` at a hash boundary ``k>0``, ``"prev_cum"`` for a
        continuing same-hash stage).

    Notes
    -----
    When all hashes are equal the boundary branch never fires and the
    schedule is bit-identical to the pre-Phase-3 ``_check_staged``
    schedule (origin base at stage 0, previous cumulative thereafter).
    This is the regression anchor for the seismic-on-fixed-section
    validation case.
    """
    sched = []
    prev_hash = None
    for k, h in enumerate(stage_hashes):
        if k == 0:
            reset, base = True, "origin"
        elif h != prev_hash:
            reset, base = True, "carry"
        else:
            reset, base = False, "prev_cum"
        sched.append({
            "stage": k,
            "domain_hash": h,
            "reset": reset,
            "path_base": base,
        })
        prev_hash = h
    return sched


# ==================================================================
#  Geometry / mesh signature
# ==================================================================

def geometry_signature(section) -> Tuple[Any, ...]:
    r"""
    Hashable signature of the fixed (bulk) part of a section.

    Captures everything the resistance domain depends on that does
    **not** vary across Phase-3 stages: the bulk mesh (fiber
    coordinates, areas, material zoning) and the bulk material
    identity.  In Phase 3 the bulk is immutable, so this is computed
    once and reused.

    The byte hashes of the coordinate/area arrays make the signature a
    genuine *content* identity: two sections meshed identically share
    a signature even if they are distinct objects.  Material identity
    uses :func:`id`, which is process-local — adequate because the
    domain cache lives only within a single run.  Two materials with
    identical parameters but different objects get different
    signatures and therefore do not share a cached domain; this is the
    *safe* direction (a missed reuse, never a wrong reuse).

    Parameters
    ----------
    section : GenericSection
        Base (un-materialized) section.

    Returns
    -------
    tuple
        Canonical, hashable signature.

    Notes
    -----
    A bulk-fiber active mask is intentionally **not** part of this
    signature in Phase 3 (all bulk fibers are always active).  When
    staged-construction of bulk material (composite toppings) is added
    later, fold the active-fiber mask hash in here; the per-element
    hash machinery below is unaffected.
    """
    x = np.ascontiguousarray(section.x_fibers, dtype=float)
    y = np.ascontiguousarray(section.y_fibers, dtype=float)
    a = np.ascontiguousarray(section.A_fibers, dtype=float)
    mi = np.ascontiguousarray(section.mat_indices, dtype=np.int64)
    return (
        id(section.bulk_material),
        int(x.size),
        hash(x.tobytes()),
        hash(y.tobytes()),
        hash(a.tobytes()),
        hash(mi.tobytes()),
        # Reference point: in Phase 3 it is the polygon centroid and
        # constant, but include it so that a future bulk-staging change
        # which moves the centroid invalidates the cache.
        _quantize(float(section.x_centroid), 1e-3),
        _quantize(float(section.y_centroid), 1e-3),
    )


# ==================================================================
#  SectionState
# ==================================================================

@dataclass
class SectionState:
    r"""
    Descriptor of the section at one construction/loading stage.

    A :class:`SectionState` does not hold geometry; it holds the
    *deltas* against the immutable base section that define a distinct
    resistance state.  All point-element arrays (``active``,
    ``bonded``, ``eps_init``) are indexed over the **union** element
    set in the canonical order ``rebars + tendons`` of the base
    section, so the same index always refers to the same physical
    element across stages.

    Parameters
    ----------
    stage_index : int
        Position in the stage list (0-based).
    active : numpy.ndarray of bool
        Element-present mask over the union element set.
    bonded : numpy.ndarray of bool
        Strain-compatibility mask.  An active element that is *not*
        bonded is excluded from the resistance domain (its force is a
        demand-side :class:`PrestressAction`).
    eps_init : numpy.ndarray of float
        Per-element locked-in initial strain [-] (jacking strain net
        of losses, for tendons; usually 0 for passive rebars).
    bulk_eps_init : float, optional
        Uniform bulk pre-strain [-] (e.g. shrinkage), 0 by default.
    time_days : float, optional
        Cumulative time since stage 0 [days].  Informational for the
        losses model; does **not** enter the hash directly — its
        effect enters only through the ``eps_init`` values it produces.
    label : str, optional
        Human-readable stage name.

    Notes
    -----
    The capacity state hash (:meth:`capacity_hash`) deliberately
    excludes ``time_days`` and any applied loads.  Time and losses act
    *through* ``eps_init``; loads act on the demand path.
    """

    stage_index: int
    active: np.ndarray
    bonded: np.ndarray
    eps_init: np.ndarray
    bulk_eps_init: float = 0.0
    time_days: float = 0.0
    label: str = ""

    def capacity_hash(self, geom_sig: Tuple[Any, ...],
                      union_materials: List[int]) -> int:
        r"""
        Capacity state hash: identity of the resistance domain.

        Composed of

        1. the fixed geometry/mesh signature *geom_sig*;
        2. for every **active and bonded** element, in ascending
           index order, the triple
           ``(material_id, quantize(eps_init), bonded)``;
        3. the quantized bulk pre-strain.

        Active-but-unbonded elements are excluded (they are not in the
        domain).  Applied loads / ``PrestressAction`` are excluded by
        construction — they never reach this method.

        Parameters
        ----------
        geom_sig : tuple
            Output of :func:`geometry_signature` for the base section.
        union_materials : list of int
            ``id(material)`` for each element in the union set, in the
            canonical ``rebars + tendons`` order.

        Returns
        -------
        int
            Python hash of the canonical state tuple.
        """
        elem_terms = []
        idx = np.nonzero(self.active & self.bonded)[0]
        for i in idx:
            elem_terms.append(
                (union_materials[int(i)],
                 _quantize(float(self.eps_init[int(i)])),
                 True)
            )
        return hash((
            geom_sig,
            tuple(elem_terms),
            _quantize(float(self.bulk_eps_init)),
        ))

    def copy_advanced(self, stage_index: int,
                      label: str = "") -> "SectionState":
        r"""
        Return a deep-ish copy with masks/strains duplicated.

        Used by the stage driver to derive the next state from the
        current one before applying that stage's mutations, so the
        per-stage history is preserved (each stage owns its arrays).

        Parameters
        ----------
        stage_index : int
        label : str, optional

        Returns
        -------
        SectionState
        """
        return SectionState(
            stage_index=stage_index,
            active=self.active.copy(),
            bonded=self.bonded.copy(),
            eps_init=self.eps_init.copy(),
            bulk_eps_init=self.bulk_eps_init,
            time_days=self.time_days,
            label=label or self.label,
        )

    # -- stage operations (each returns a new state; arrays copied) --

    def with_activated(self, indices) -> "SectionState":
        r"""
        New state with *indices* (union element indices) set active.

        Activating an element is the staged-construction primitive
        (the element was declared inactive at stage 0 and "added"
        later).  It changes the hash and triggers a domain rebuild.
        """
        s = self.copy_advanced(self.stage_index, self.label)
        s.active[np.asarray(indices, dtype=int)] = True
        return s

    def with_deactivated(self, indices) -> "SectionState":
        r"""
        New state with *indices* set inactive (removed).

        Deactivation alone only removes the element from the
        resistance domain.  The *force-release* accounting that keeps
        equilibrium when a **stressed bonded** element is cut is
        produced separately by
        :meth:`StagedDomainManager.deactivation_actions` and applied to
        the demand path — the element does not become a load, its
        removal generates one.
        """
        s = self.copy_advanced(self.stage_index, self.label)
        s.active[np.asarray(indices, dtype=int)] = False
        return s

    def with_eps_override(self, mapping) -> "SectionState":
        r"""
        New state with per-element ``eps_init`` overrides applied.

        Parameters
        ----------
        mapping : dict
            ``{union_index: eps_init}``.  This is how a losses/creep
            step at a stage advances the locked-in strain; the new
            values re-quantize and (if they cross a ``QUANT_EPS``
            bucket) change the hash, triggering a domain rebuild.
        """
        s = self.copy_advanced(self.stage_index, self.label)
        for i, e in mapping.items():
            s.eps_init[int(i)] = float(e)
        return s

    def with_bulk_eps(self, value) -> "SectionState":
        """New state with bulk pre-strain set to *value* [-]."""
        s = self.copy_advanced(self.stage_index, self.label)
        s.bulk_eps_init = float(value)
        return s


# ==================================================================
#  PrestressAction (Phase-3 interface; fleshed out in prestress v1)
# ==================================================================

@dataclass(frozen=True)
class PrestressAction:
    r"""
    An external action applied at a stage, contributing to the demand
    path but **not** to the capacity hash.

    In prestress v1 this will be produced by the transfer/losses
    driver (equivalent prestress loads) and by element force release
    on deactivation (:func:`released_force_action`).  At the Phase-3
    infrastructure level it is the typed carrier that keeps the
    resistance/demand separation honest: anything that is a *load*
    flows through here, never through :class:`SectionState`.

    Parameters
    ----------
    N, Mx, My : float
        Action resultant about the section reference point
        [N, N·mm, N·mm].
    label : str, optional
    origin : str, optional
        Provenance tag, e.g. ``"transfer"``, ``"release"``, ``"creep"``.
    """

    N: float
    Mx: float
    My: float
    label: str = ""
    origin: str = ""

    def triple(self) -> Tuple[float, float, float]:
        """Return ``(N, Mx, My)`` in [N, N·mm, N·mm]."""
        return self.N, self.Mx, self.My

    @classmethod
    def from_force(cls, P, x, y, x_ref=0.0, y_ref=0.0,
                   label="", origin="prestress") -> "PrestressAction":
        r"""
        Build the section action of a prestressing force.

        Turns a tendon force :math:`P` (tension positive) applied at
        the point :math:`(x, y)` into the resultant **action on the
        section**, expressed about the reference point
        :math:`(x_{\text{ref}}, y_{\text{ref}})` and in the **exact
        sign convention of the fiber integrator**.

        Sign convention
        ---------------
        The integrator forms the resultant of an internal element force
        of value :math:`F = \sigma A` at a lever
        :math:`(\ell_x, \ell_y) = (x - x_{\text{ref}},\,
        y - y_{\text{ref}})` as

        .. math::

            N = F, \qquad
            M_x = F\,\ell_y, \qquad
            M_y = -F\,\ell_x

        (see :func:`_element_net_force`, whose tendon/rebar loop this
        mirrors).  A jacked tendon under tension :math:`P>0` pulls its
        anchorages inward, so the action it transmits to the section is
        a **compression** of magnitude :math:`P`; in the integrator's
        tension-positive convention this is the element force
        :math:`F = -P`.  Substituting :math:`F = -P`:

        .. math::

            N_p = -P, \qquad
            M_x = -P\,(y - y_{\text{ref}}), \qquad
            M_y = +P\,(x - x_{\text{ref}}).

        Routing this action onto the demand path therefore reproduces,
        for a tendon on hardened concrete (post-tension / external /
        jacking), exactly the equivalent :math:`(N, M)` couple a hand
        calculation about the same reference point would give.

        This is the **load** counterpart of a bonded tendon's
        ``eps_pe``: the bonded prestrain shifts the *resistance* (it is
        baked into the fibers and enters :meth:`capacity_hash`), whereas
        a :class:`PrestressAction` produced here is a pure *action* that
        moves the demand point and is excluded from the hash by
        construction.

        Parameters
        ----------
        P : float
            Tendon force [N], tension positive.  (Resolve a
            ``sigma_p0 * Ap`` product to this value upstream.)
        x, y : float
            Tendon position [mm], in the same frame as the section
            geometry.
        x_ref, y_ref : float, optional
            Reference point the action is taken about [mm].  Must be the
            **same point the demand path uses** — the section centroid
            (``solver.x_ref`` / ``solver.y_ref``), constant across
            materialized views in Phase 3.  Default ``0.0``.
        label : str, optional
            Human-readable tag.
        origin : str, optional
            Provenance tag.  Default ``"prestress"``.

        Returns
        -------
        PrestressAction

        Examples
        --------
        A strand jacked to :math:`P = 1.4\,\text{MN}` at
        :math:`(x, y) = (200, 80)` mm about a centroid at
        :math:`(150, 300)` mm:

        >>> a = PrestressAction.from_force(1.4e6, 200.0, 80.0,
        ...                                x_ref=150.0, y_ref=300.0)
        >>> a.N
        -1400000.0
        >>> a.Mx          # -P (y - y_ref) = -1.4e6 * (80 - 300)
        308000000.0
        >>> a.My          # +P (x - x_ref) = +1.4e6 * (200 - 150)
        70000000.0
        """
        F = -float(P)
        ly = float(y) - float(y_ref)
        lx = float(x) - float(x_ref)
        return cls(N=F, Mx=F * ly, My=-F * lx,
                   label=label, origin=origin)


def released_force_action(prev_bundle, element_index, x_ref, y_ref,
                          cum_N, cum_Mx, cum_My,
                          label="release") -> PrestressAction:
    r"""
    Compute the compensating action when a stressed bonded element is
    deactivated (cut / de-stressed).

    The pre-removal section is solved for equilibrium at the current
    cumulative demand to recover the strain plane; the element's net
    internal force at that strain plane is then evaluated and returned
    with **opposite sign** as a :class:`PrestressAction` on the
    reduced section, so that

    .. math::

        \mathbf{S}_{\text{reduced}}
          = \mathbf{S}_{\text{full}} + (-\mathbf{F}_{p,\text{elem}})

    keeps the demand consistent across the cut.

    Parameters
    ----------
    prev_bundle : DomainBundle
        Bundle of the *pre-removal* state (its solver is used to
        recover the strain plane).
    element_index : int
        Index of the element being removed, in the materialized view
        of the pre-removal section.
    x_ref, y_ref : float
        Section reference point [mm].
    cum_N, cum_Mx, cum_My : float
        Current cumulative demand [N, N·mm].
    label : str, optional

    Returns
    -------
    PrestressAction

    Warnings
    --------
    The per-element net-force computation is unit-verified against a
    known linear material.  The **end-to-end** de-stressing workflow
    (cut state solve, domain rebuild, path reset) is not yet exercised
    by a validation toy; treat a full-pipeline de-stressing run as
    unverified until that toy exists (existing-structure assessment /
    retrofit).

    Notes
    -----
    Only meaningful for a *bonded, stressed* element.  Clean removal of
    an unstressed/hypothetical element produces a zero action and the
    caller should simply drop the element.
    """
    sv = prev_bundle.solver
    sol = sv.solve_equilibrium(cum_N, cum_Mx, cum_My)
    if not sol["converged"]:
        # Cannot recover the strain state; surface a zero action and
        # let the caller decide.  Conservative: equilibrium of the
        # reduced section is then the caller's responsibility.
        return PrestressAction(0.0, 0.0, 0.0, label=label,
                               origin="release_unconverged")
    # Net element force at this strain plane, sign-flipped.
    F, Mx_e, My_e = _element_net_force(
        sv, element_index, sol["eps0"], sol["chi_x"], sol["chi_y"])
    return PrestressAction(-F, -Mx_e, -My_e, label=label,
                           origin="release")


def _element_net_force(solver, element_index, eps0, chi_x, chi_y):
    r"""
    Net (steel minus displaced bulk, for embedded) force and moments
    of a single tendon/rebar element at a given strain plane.

    Mirrors the per-element contribution the integrator forms in its
    tendon/rebar loop, restricted to one element.  Returns
    ``(N, Mx, My)`` about the solver reference point.
    """
    sec = solver.sec
    # Resolve which array the element lives in.  Convention: indices
    # [0, n_rebars) are rebars, [n_rebars, n_rebars+n_tendons) tendons.
    n_reb = int(sec.x_rebars.size)
    if element_index < n_reb:
        j = element_index
        x = float(sec.x_rebars[j]); y = float(sec.y_rebars[j])
        A = float(sec.A_rebars[j])
        mat = sec.rebars[j].material
        emb = bool(sec.embedded_rebars[j])
        eps_init = 0.0
    else:
        j = element_index - n_reb
        x = float(sec.x_tendons[j]); y = float(sec.y_tendons[j])
        A = float(sec.A_tendons[j])
        mat = sec.tendons[j].material
        emb = bool(sec.embedded_tendons[j])
        eps_init = float(sec.eps_init_tendons[j])

    ly = y - solver.y_ref
    lx = x - solver.x_ref
    eps_sec = eps0 + chi_x * ly - chi_y * lx
    sigma = float(mat.stress_array(np.array([eps_sec + eps_init]))[0])
    if emb:
        sigma -= float(
            sec.bulk_material.stress_array(np.array([eps_sec]))[0])
    F = sigma * A
    return F, F * ly, -F * lx


# ==================================================================
#  Materialized section view
# ==================================================================

def materialize_view(base_section, state: SectionState):
    r"""
    Build the section *view* for a state without re-meshing.

    Returns a shallow copy of *base_section* whose **point-element**
    arrays (rebars and tendons) are restricted to the elements that
    are both *active* and (for the resistance domain) *bonded*, and
    whose tendon ``eps_init`` is overwritten by the state's override.
    The bulk mesh arrays are shared by reference; they are immutable
    in Phase 3.

    The view's lazy property caches are invalidated so that
    ``ideal_gross_properties`` recomputes for this state (resolving the
    immutable-section TODO: the *base* section stays immutable, each
    *view* owns its own cache).

    Parameters
    ----------
    base_section : GenericSection
    state : SectionState

    Returns
    -------
    GenericSection
        A view suitable for ``FiberSolver(view)`` and
        ``NMDiagram(view)``.

    Notes
    -----
    Active-but-unbonded elements are **omitted** from the view: they
    do not participate in strain compatibility and their force is a
    demand-side :class:`PrestressAction`.  The mapping from view index
    back to union index (needed by :func:`released_force_action`) is
    attached as ``view._union_index``.
    """
    view = copy.copy(base_section)

    n_reb = int(base_section.x_rebars.size)
    n_ten = int(getattr(base_section, "x_tendons",
                        np.empty(0)).size)

    resist = state.active & state.bonded
    reb_keep = np.nonzero(resist[:n_reb])[0]
    ten_keep = np.nonzero(resist[n_reb:n_reb + n_ten])[0]

    # ---- Rebars ----
    view.rebars = [base_section.rebars[i] for i in reb_keep]
    view.x_rebars = base_section.x_rebars[reb_keep]
    view.y_rebars = base_section.y_rebars[reb_keep]
    view.A_rebars = base_section.A_rebars[reb_keep]
    view.embedded_rebars = base_section.embedded_rebars[reb_keep]
    if hasattr(base_section, "mat_indices_rebar"):
        view.mat_indices_rebar = \
            base_section.mat_indices_rebar[reb_keep]

    # ---- Tendons ----
    if n_ten:
        view.tendons = [base_section.tendons[i] for i in ten_keep]
        view.x_tendons = base_section.x_tendons[ten_keep]
        view.y_tendons = base_section.y_tendons[ten_keep]
        view.A_tendons = base_section.A_tendons[ten_keep]
        view.embedded_tendons = base_section.embedded_tendons[ten_keep]
        view.mat_indices_tendon = \
            base_section.mat_indices_tendon[ten_keep]
        # Override eps_init from the state (losses-evolved prestrain).
        view.eps_init_tendons = state.eps_init[n_reb:n_reb + n_ten][ten_keep]

    # ---- Provenance + cache invalidation ----
    view._union_index = (
        [int(i) for i in reb_keep]
        + [int(n_reb + i) for i in ten_keep]
    )
    # Carry the state's bulk pre-strain onto the view.  This is the
    # offset *carrier*: the value is now attached to the section the
    # solver is built on (and already participates in the capacity
    # hash via :meth:`SectionState.capacity_hash`).  Making it bite on
    # the resistance requires the integrator to evaluate the bulk law
    # at ``eps_section + bulk_eps_init`` — a separate, kernel-level
    # change (see the Phase-5 note in the deliverable).  Until then the
    # view faithfully advertises the offset without it being consumed.
    view.bulk_eps_init = float(state.bulk_eps_init)
    view._ideal_gross_props_cache = None
    return view


# ==================================================================
#  Domain bundle + manager
# ==================================================================

@dataclass
class DomainBundle:
    r"""
    Everything derived from one resistance state, cached together.

    Parameters
    ----------
    solver : FiberSolver
    nm_gen : NMDiagram
    domain : DomainChecker
        Wraps the 3-D (or uniaxial) point cloud.
    cloud : dict
        The raw point cloud the domain was built from (kept for
        diagnostics / extent checks without touching DomainChecker
        internals).
    contour_cache : dict
        ``{round(N) -> MxMyContour | None}`` for ``eta_2D`` / path-2D
        on this domain.  Separate per bundle because contours are
        domain-specific.
    """

    solver: Any
    nm_gen: Any
    domain: Any
    cloud: Any = None
    contour_cache: Dict[float, Any] = field(default_factory=dict)


class StagedDomainManager:
    r"""
    Owns the ``{capacity_hash -> DomainBundle}`` cache and rebuilds a
    resistance domain on a hash miss.

    The manager is the single place where the
    "equal hash → reuse domain and all derived outputs" rule is
    enforced.  Capacity recompute is **automatic**: callers ask for the
    bundle of a :class:`SectionState`; the manager hashes the state and
    either returns the cached bundle or builds a new one.

    Parameters
    ----------
    base_section : GenericSection
    biaxial : bool
        Whether to build the 3-D ``(N, Mx, My)`` domain
        (``generate_biaxial``) or the uniaxial N-Mx fallback
        (``generate``).
    gen_kwargs : dict, optional
        Keyword arguments forwarded to the domain generator
        (``n_angles``, ``n_points_per_angle`` / ``n_points``).
    include_pivot_a : bool, optional
        Forwarded to :class:`NMDiagram`.  Default ``True``.

    Notes
    -----
    The manager imports :class:`FiberSolver`, :class:`NMDiagram` and
    :class:`DomainChecker` lazily to avoid a circular import with
    :mod:`gensec.solver.check`.
    """

    def __init__(self, base_section, biaxial, gen_kwargs=None,
                 include_pivot_a=True):
        self.base_section = base_section
        self.biaxial = biaxial
        self.gen_kwargs = dict(gen_kwargs or {})
        self.include_pivot_a = include_pivot_a

        self._geom_sig = geometry_signature(base_section)
        self._union_materials = self._collect_union_materials(
            base_section)
        self._cache: Dict[int, DomainBundle] = {}

    @staticmethod
    def _collect_union_materials(section) -> List[int]:
        r"""``id(material)`` per union element, ``rebars + tendons``."""
        ids = [id(r.material) for r in section.rebars]
        for t in getattr(section, "tendons", []):
            ids.append(id(t.material))
        return ids

    def hash_of(self, state: SectionState) -> int:
        """Capacity state hash for *state* against the base geometry."""
        return state.capacity_hash(self._geom_sig,
                                   self._union_materials)

    def initial_state(self) -> SectionState:
        r"""
        The stage-0 state: every union element active and bonded, with
        ``eps_init`` taken from the base section (tendon ``eps_pe``,
        zero for passive rebars) and zero bulk pre-strain.

        Stages mutate a copy of this via their state operations.

        Returns
        -------
        SectionState
        """
        sec = self.base_section
        n_reb = int(sec.x_rebars.size)
        n_ten = int(getattr(sec, "x_tendons", np.empty(0)).size)
        n = n_reb + n_ten
        eps = np.zeros(n)
        if n_ten:
            eps[n_reb:] = np.asarray(sec.eps_init_tendons, dtype=float)
        return SectionState(
            stage_index=0,
            active=np.ones(n, bool),
            bonded=np.ones(n, bool),
            eps_init=eps,
            # Seed the bulk pre-strain from the section so a YAML
            # ``prestrain`` / ``eps_init`` field is the stage-0 baseline
            # and enters the capacity hash.  Defaults to 0.0 for every
            # section that does not declare one, so all pre-existing
            # (non-prestress) runs are byte-identical.
            bulk_eps_init=float(getattr(sec, "bulk_eps_init", 0.0)),
            label="stage0",
        )

    def get_bundle(self, state: SectionState
                   ) -> Tuple[int, DomainBundle, bool]:
        r"""
        Return ``(hash, bundle, was_built)`` for *state*.

        On a cache hit *bundle* is reused verbatim and *was_built* is
        ``False``.  On a miss the domain is built and cached.

        Parameters
        ----------
        state : SectionState

        Returns
        -------
        tuple
            ``(hash, DomainBundle, bool)``.
        """
        h = self.hash_of(state)
        if h in self._cache:
            return h, self._cache[h], False
        bundle = self._build_bundle(state)
        self._cache[h] = bundle
        return h, bundle, True

    def resolve_stages(self, stages):
        r"""
        Derive the per-stage section states, capacity hashes and
        resistance bundles for a staged combination.

        Single source of truth for stage-operation interpretation,
        shared by :class:`~gensec.solver.check.VerificationEngine` and
        :class:`~gensec.solver.analysis.AnalysisEngine`.  Pure with
        respect to demand — states, hashes and bundles depend only on
        the section, never on loads — so it can run before any demand
        walk and feed :func:`path_schedule`.

        Parameters
        ----------
        stages : list of dict
            Each stage may carry a ``section_ops`` dict with keys
            ``activate`` / ``deactivate`` (lists of union element
            indices), ``eps_override`` (``{idx: eps}``), ``bulk_eps``
            (float) and ``release`` (bool; whether deactivations are
            force-released vs cleanly removed).  A stage may also carry
            ``time`` (cumulative days since stage 0), copied onto
            :attr:`SectionState.time_days` — **carry-through only**: it
            never enters :meth:`SectionState.capacity_hash` (time acts
            on the capacity only through the ``eps_override`` values a
            losses model derives from it).  Omitted → the previous
            stage's value carries forward.

        Returns
        -------
        states : list of SectionState
        hashes : list
        bundles : list of DomainBundle
        deact : list of tuple
            ``(indices, release)`` of elements deactivated *at* each
            stage, for the force-release accounting in the demand walk.
        """
        states, hashes, bundles, deact = [], [], [], []
        cur = self.initial_state()
        for k, stage in enumerate(stages):
            ops = stage.get("section_ops", {}) or {}
            deact_idx = list(ops.get("deactivate", []) or [])
            release = bool(ops.get("release", True))

            if ops.get("activate"):
                cur = cur.with_activated(ops["activate"])
            if deact_idx:
                cur = cur.with_deactivated(deact_idx)
            if ops.get("eps_override"):
                cur = cur.with_eps_override(ops["eps_override"])
            if "bulk_eps" in ops:
                cur = cur.with_bulk_eps(ops["bulk_eps"])
            # Informational time stamp [days], cumulative since stage 0.
            # Deliberately outside the ops dict (it is not a capacity
            # operation) and deliberately not hashed.
            if "time" in stage:
                cur.time_days = float(stage["time"])
            cur.stage_index = k

            h, bundle, _built = self.get_bundle(cur)
            states.append(cur)
            hashes.append(h)
            bundles.append(bundle)
            deact.append((deact_idx, release))
        return states, hashes, bundles, deact

    def deactivation_actions(self, prev_bundle, union_indices, cum,
                             release=True) -> List[PrestressAction]:
        r"""
        Force-release actions for elements deactivated at a stage.

        Element-type agnostic (works for rebars and tendons).  The
        pre-removal section (``prev_bundle``) is solved **once** for
        equilibrium at the current cumulative demand *cum*; each
        released element's net internal force at that strain plane is
        then re-applied with opposite sign as a :class:`PrestressAction`
        on the reduced section, preserving equilibrium across the cut.

        Parameters
        ----------
        prev_bundle : DomainBundle
            Bundle of the state **before** deactivation (its solver and
            materialized view are used).
        union_indices : iterable of int
            Union element indices being deactivated at this stage.
        cum : tuple of float
            Current cumulative demand ``(N, Mx, My)`` [N, N·mm].
        release : bool, optional
            ``True`` (default): force release (stressed bonded element
            cut).  ``False``: clean removal (unstressed/hypothetical) —
            returns an empty list.

        Returns
        -------
        list of PrestressAction
            Opposite-sign element forces, about the section reference
            point (the same reference the demand path uses).

        Warnings
        --------
        The per-element force computation is unit-verified against a
        known linear material (sign, moments, embedded net-subtraction,
        union→view mapping).  What remains unvalidated at Phase-3 is the
        **end-to-end** de-stressing run (real solver convergence at the
        cut state, domain rebuild, path reset) — no de-stressing toy
        exists yet.  See :func:`released_force_action`.

        Notes
        -----
        Moments are about ``solver.x_ref/y_ref`` (the polygon centroid),
        which is constant across views, so the released action is
        directly additive to the cumulative demand.  For an *embedded*
        element the released force is the **net** contribution
        (steel minus displaced bulk) — exactly the contribution the
        element was making to the full section, so removing it and
        re-injecting its opposite keeps the demand consistent.
        """
        if not release:
            return []
        sv = prev_bundle.solver
        cN, cMx, cMy = cum
        sol = sv.solve_equilibrium(cN, cMx, cMy)
        if not sol["converged"]:
            return [PrestressAction(0.0, 0.0, 0.0,
                                    origin="release_unconverged")]
        view = sv.sec
        inv = {u: i for i, u in enumerate(getattr(view, "_union_index", []))}
        actions = []
        for u in union_indices:
            vi = inv.get(int(u))
            if vi is None:
                # Element not present in the pre-removal view (already
                # inactive / unbonded) — nothing to release.
                continue
            F, Mx_e, My_e = _element_net_force(
                sv, vi, sol["eps0"], sol["chi_x"], sol["chi_y"])
            actions.append(PrestressAction(
                -F, -Mx_e, -My_e, origin="release"))
        return actions

    def _build_bundle(self, state: SectionState) -> DomainBundle:
        r"""Materialize the view and generate its resistance domain."""
        from .integrator import FiberSolver
        from .capacity import NMDiagram
        from .check import DomainChecker

        view = materialize_view(self.base_section, state)
        solver = FiberSolver(view)
        nm_gen = NMDiagram(solver, include_pivot_a=self.include_pivot_a)

        if self.biaxial:
            cloud = nm_gen.generate_biaxial(
                n_angles=self.gen_kwargs.get("n_angles", 36),
                n_points_per_angle=self.gen_kwargs.get(
                    "n_points_per_angle", 80),
            )
        else:
            cloud = nm_gen.generate(
                n_points=self.gen_kwargs.get("n_points", 200))

        return DomainBundle(
            solver=solver,
            nm_gen=nm_gen,
            domain=DomainChecker(cloud),
            cloud=cloud,
        )
