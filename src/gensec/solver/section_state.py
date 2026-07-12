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
    bulk_active : numpy.ndarray of bool or None, optional
        Per-zone activity mask over ``n_zones = 1 +
        len(bulk_materials)`` (Phase 8 bulk staging).  Zone 0 (the
        base ``bulk_material``) is always active.  ``None`` (default,
        legacy direct construction) means "all zones active"; states
        built by :meth:`StagedDomainManager.initial_state` always
        carry the explicit array.  The per-fiber mask is always
        *derived*, never stored:
        ``fiber_active = bulk_active[mat_indices]``.
    bulk_planes : numpy.ndarray or None, optional
        Per-zone locked-in datum planes, shape ``(n_zones, 3)``:
        :math:`(\varepsilon_{0,z}, \chi_{x,z}, \chi_{y,z})` per
        zone, evaluated with the sign convention of
        :meth:`~gensec.solver.integrator.FiberSolver.strain_field`
        about the solver reference point (the full-polygon centroid,
        pinned across stages).  The casting datum of a staged zone:
        the zone is stress-free on the plane :math:`-\,
        \mathrm{plane}_z` (linear-equivalence identity, master plan
        §3).  The legacy scalar ``bulk_eps_init`` is *not* folded in
        here — it remains a separate uniform term added to every
        active zone by the integrator's offset field (one internal
        mechanism, two inputs; ``with_bulk_eps`` keeps working
        unchanged).
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
    bulk_active: Optional[np.ndarray] = None
    bulk_planes: Optional[np.ndarray] = None
    time_days: float = 0.0
    label: str = ""

    def capacity_hash(self, geom_sig: Tuple[Any, ...],
                      union_materials: List[int],
                      chi_quantum: float = QUANT_EPS) -> int:
        r"""
        Capacity state hash: identity of the resistance domain.

        Composed of

        1. the fixed geometry/mesh signature *geom_sig*;
        2. for every **active and bonded** element, in ascending
           index order, the triple
           ``(material_id, quantize(eps_init), bonded)``;
        3. the quantized bulk pre-strain;
        4. (Phase 8, when :attr:`bulk_active` is set) the byte hash of
           the zone activity mask, and — per **active** zone in index
           order — the quantized locked-in plane triple

           .. math::

               \bigl(\,q(\varepsilon_{0,z}),\;
               q_\chi(\chi_{x,z}),\; q_\chi(\chi_{y,z})\,\bigr)

           with the curvature quantum
           :math:`q_\chi = \texttt{QUANT\_EPS} / D`,
           :math:`D = \max(H, B)` of the base section, so the
           bucketing error on the extreme-fiber strain
           :math:`\chi \cdot D` stays :math:`\le` ``QUANT_EPS`` —
           coherent with the documented ``QUANT_EPS`` trap.

        Active-but-unbonded elements are excluded (they are not in the
        domain).  Applied loads / ``PrestressAction`` are excluded by
        construction — they never reach this method.  States without
        zone arrays (``bulk_active is None``, legacy direct
        construction) hash exactly as before Phase 8; a manager mixes
        the two forms only in the *safe* direction (a missed cache
        reuse, never a wrong one).

        Parameters
        ----------
        geom_sig : tuple
            Output of :func:`geometry_signature` for the base section.
        union_materials : list of int
            ``id(material)`` for each element in the union set, in the
            canonical ``rebars + tendons`` order.
        chi_quantum : float, optional
            Curvature bucket width [1/mm].  Deterministic per manager:
            :class:`StagedDomainManager` computes it once from the
            base section and passes it down.  Default
            :data:`QUANT_EPS` (dimensionally a fallback for direct
            callers only).

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
        terms = [
            geom_sig,
            tuple(elem_terms),
            _quantize(float(self.bulk_eps_init)),
        ]
        if self.bulk_active is not None:
            ba = np.ascontiguousarray(self.bulk_active, dtype=bool)
            terms.append(hash(ba.tobytes()))
            if self.bulk_planes is None:
                planes = np.zeros((ba.size, 3), dtype=float)
            else:
                planes = np.asarray(self.bulk_planes, dtype=float)
            zone_terms = []
            for z in np.nonzero(ba)[0]:
                e0, cx, cy = planes[int(z)]
                zone_terms.append((
                    _quantize(float(e0)),
                    _quantize(float(cx), chi_quantum),
                    _quantize(float(cy), chi_quantum),
                ))
            terms.append(tuple(zone_terms))
        return hash(tuple(terms))

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
            bulk_active=(None if self.bulk_active is None
                         else self.bulk_active.copy()),
            bulk_planes=(None if self.bulk_planes is None
                         else self.bulk_planes.copy()),
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

    def with_grouted(self, indices, eps_init_map) -> "SectionState":
        r"""
        New state with *indices* **grouted**: made active and bonded with
        their reconciled post-loss ``eps_init`` set, **atomically**.

        Grouting is the post-tensioning injection event.  Before it, a
        post-tension tendon is a demand-side :class:`PrestressAction` (a
        load on hardened concrete); after it, it is a bonded section
        element contributing ULS resistance.  This method performs that
        whole transition in a single state derivation: it flips ``active``
        and ``bonded`` to ``True`` **and** writes the locked-in strain in
        one copy, so the resulting state can never be "in the domain but
        also a load" or "neither".

        Composing :meth:`with_activated` + :meth:`with_eps_override` would
        produce the same arrays but leaves *atomicity* — and therefore the
        single-side prestress invariant — to the caller.  The dedicated
        method makes the invariant a property of construction, not of
        caller discipline.

        Every grouted index **must** carry a reconciled ``eps_init`` in
        *eps_init_map*.  Grouting reconciles the locked-in strain to the
        concrete strain datum at the injection stage, so that

        .. math::

            \sigma_p\!\left(\varepsilon_{\mathrm{sec}}^{\,\mathrm{grout}}
            + \varepsilon_{\mathrm{init}}\right)

        reproduces the effective post-loss tendon stress.  Activating a
        tendon while leaving whatever stale value the ``eps_init`` array
        happened to hold would be a *silent* reconciliation — precisely
        the failure mode the prestress driver forbids.  A missing entry
        therefore raises rather than defaulting.

        Parameters
        ----------
        indices : sequence of int
            Union element indices (canonical ``rebars + tendons`` order)
            to grout.
        eps_init_map : dict
            ``{union_index: eps_init}`` — the reconciled post-loss strain
            datum for **every** index in *indices*.

        Returns
        -------
        SectionState
            A new state; the hash changes (a bonded element enters the
            domain), triggering an automatic domain rebuild.

        Raises
        ------
        KeyError
            If any index in *indices* has no entry in *eps_init_map*
            (atomicity guard: a tendon never enters the domain without an
            explicit reconciled prestrain).

        See Also
        --------
        with_activated : activate an element without setting strain.
        with_eps_override : advance ``eps_init`` without changing bondedness.
        gensec.solver.posttension.grout : the driver-level grouting that
            produces the reconciled *eps_init_map* and drops the matching
            demand actions in the same step.
        """
        idx = np.asarray(indices, dtype=int)
        missing = [int(i) for i in idx if int(i) not in eps_init_map]
        if missing:
            raise KeyError(
                f"with_grouted: no reconciled eps_init for grouted "
                f"index/indices {missing}. Grouting requires an explicit "
                f"locked-in strain for every grouted element (single-side "
                f"invariant: a tendon enters the resistance domain with its "
                f"reconciled post-loss prestrain, never a stale array value)."
            )
        s = self.copy_advanced(self.stage_index, self.label)
        s.active[idx] = True
        s.bonded[idx] = True
        for i in idx:
            s.eps_init[int(i)] = float(eps_init_map[int(i)])
        return s

    def with_bulk_activated(self, zones, plane_map) -> "SectionState":
        r"""
        New state with the bulk *zones* **cast**: made active with
        their locked-in datum planes set, **atomically**.

        The bulk analog of :meth:`with_grouted` (same single-side
        invariant): a zone enters the resistance domain only together
        with an **explicit** casting datum plane.  Casting a zone
        while leaving whatever plane the array happened to hold would
        be a *silent* reconciliation — precisely the failure mode the
        prestress driver forbids for tendons.  A missing entry
        therefore raises rather than defaulting;
        :math:`(0, 0, 0)` is legal but must be written.

        The datum plane :math:`(\varepsilon_{0,z}, \chi_{x,z},
        \chi_{y,z})` is expressed about the solver reference point
        (full-polygon centroid) with the sign convention of
        :meth:`~gensec.solver.integrator.FiberSolver.strain_field`.
        Physically: the zone is **stress-free** on the section strain
        plane :math:`-\,\mathrm{plane}_z`, so the datum of a zone
        cast on a deformed substrate is the *negated* substrate plane
        at casting (linear incremental ≡ one-shot equivalence, master
        plan §3).  Producing that plane automatically (``auto``) is
        the Task-2 timeline resolution walk; at engine level the datum
        is an input (demand purity of ``resolve_stages``).

        Parameters
        ----------
        zones : sequence of int
            Zone indices to activate (1-based zone list positions;
            zone 0 = ``'base'`` is always active and not activatable).
        plane_map : dict
            ``{zone_index: (eps0, chi_x, chi_y)}`` — the locked-in
            datum plane for **every** index in *zones*.

        Returns
        -------
        SectionState
            A new state; the hash changes (mask flip + plane terms),
            triggering an automatic domain rebuild.

        Raises
        ------
        KeyError
            If any zone in *zones* has no entry in *plane_map*
            (atomicity guard, ``with_grouted``-style).
        ValueError
            Zone 0 targeted; zone index out of range; state built
            without zone arrays; malformed or non-finite plane.
        """
        if self.bulk_active is None:
            raise ValueError(
                "with_bulk_activated: this state carries no zone "
                "arrays (bulk_active is None — legacy direct "
                "construction). Derive states from "
                "StagedDomainManager.initial_state(), which sizes "
                "bulk_active/bulk_planes on the section's zones."
            )
        zs = [int(z) for z in zones]
        missing = [z for z in zs if z not in
                   {int(k) for k in plane_map}]
        if missing:
            raise KeyError(
                f"with_bulk_activated: no locked-in datum plane for "
                f"zone(s) {missing}. Casting requires an explicit "
                f"(eps0, chi_x, chi_y) for every activated zone "
                f"(single-side invariant: a zone enters the "
                f"resistance domain with its casting datum, never a "
                f"stale array value; (0, 0, 0) is legal but must be "
                f"written)."
            )
        n_zones = int(self.bulk_active.size)
        s = self.copy_advanced(self.stage_index, self.label)
        if s.bulk_planes is None:
            s.bulk_planes = np.zeros((n_zones, 3), dtype=float)
        plane_by_int = {int(k): v for k, v in plane_map.items()}
        for z in zs:
            if z == 0:
                raise ValueError(
                    "with_bulk_activated: zone 0 ('base') is always "
                    "active and not activatable."
                )
            if not (0 < z < n_zones):
                raise ValueError(
                    f"with_bulk_activated: zone index {z} out of "
                    f"range (state has {n_zones} zone(s))."
                )
            plane = np.asarray(plane_by_int[z], dtype=float).ravel()
            if plane.size != 3 or not np.all(np.isfinite(plane)):
                raise ValueError(
                    f"with_bulk_activated: datum plane for zone {z} "
                    f"must be three finite floats "
                    f"(eps0, chi_x, chi_y), got "
                    f"{plane_by_int[z]!r}."
                )
            s.bulk_active[z] = True
            s.bulk_planes[z, :] = plane
        return s

    def with_bulk_plane_delta(self, delta_map) -> "SectionState":
        r"""
        **Add** a locked-in datum plane increment to zones that are
        already active (Phase-5 / C5).

        The counterpart of :meth:`with_bulk_activated`, which *sets* a
        zone's datum plane **absolutely, once, at casting**.  Nothing in
        the pre-C5 vocabulary could *increment* the plane of a zone that
        is already in the domain — and that is exactly what a
        time-dependent interval does: over :math:`[t_0, t]` the concrete
        accumulates a stress-independent eigenstrain (creep under the
        frozen stress, plus shrinkage), and creep under a *linear* stress
        field is a *linear* imposed strain, so all three components of
        the plane move, not only :math:`\varepsilon_0`.

        .. math::

            \text{bulk\_planes}[z] \;\mathrel{+}=\;
            \boldsymbol{\beta}_z
            \;=\; -\,\varepsilon_{\mathrm{imp},z}

        The sign is the fiber kernel's: it *adds* the offset
        (:math:`\sigma = \mathrm{law}(\varepsilon_{\mathrm{sec}}
        + \varepsilon_{\mathrm{bulk}})`) while the physics *subtracts*
        the eigenstrain (:math:`\sigma = E(\varepsilon_{\mathrm{tot}}
        - \varepsilon_{\mathrm{imp}})`).  Its falsifiable consequence:
        a fully restrained shrinking member goes into **tension**.
        :mod:`gensec.solver.losses` owns the sign flip and states it once.

        Being **additive** is what makes a step-by-step integration
        composable: N sub-steps of an interval sum to the interval.  An
        absolute setter would make the last step overwrite the history.

        Parameters
        ----------
        delta_map : dict
            ``{zone_index: (d_eps0, d_chi_x, d_chi_y)}``.  A zone absent
            from the map is untouched.

        Returns
        -------
        SectionState
            A copy.

        Raises
        ------
        ValueError
            State built without zone arrays; zone index out of range;
            malformed or non-finite increment; a target zone that is
            **not active** (an eigenstrain cannot accrue in concrete that
            has not been cast — that is a modelling error, not a no-op).

        Notes
        -----
        The increment lands in ``bulk_planes``, which the capacity hash
        already covers, so two different loss states can never collapse
        onto one cached domain.  No hash change is needed.
        """
        if self.bulk_active is None or self.bulk_planes is None:
            raise ValueError(
                "with_bulk_plane_delta: this state carries no zone "
                "arrays (legacy direct construction). Derive states "
                "from StagedDomainManager.initial_state()."
            )
        n_zones = int(self.bulk_active.size)
        s = self.copy_advanced(self.stage_index, self.label)
        for z, d in delta_map.items():
            z = int(z)
            if not (0 <= z < n_zones):
                raise ValueError(
                    f"with_bulk_plane_delta: zone index {z} out of "
                    f"range (state has {n_zones} zone(s))."
                )
            if not bool(self.bulk_active[z]):
                raise ValueError(
                    f"with_bulk_plane_delta: zone {z} is not active, so "
                    f"it cannot accumulate a creep/shrinkage eigenstrain "
                    f"— concrete that has not been cast does not creep. "
                    f"Cast the zone (activate_bulk) before the interval "
                    f"that loads it."
                )
            delta = np.asarray(d, dtype=float).ravel()
            if delta.size != 3 or not np.all(np.isfinite(delta)):
                raise ValueError(
                    f"with_bulk_plane_delta: the increment for zone {z} "
                    f"must be three finite floats "
                    f"(d_eps0, d_chi_x, d_chi_y), got {d!r}."
                )
            s.bulk_planes[z, :] += delta
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

    def __post_init__(self):
        r"""Reject non-finite resultants (they poison the demand walk)."""
        for _nm, _v in (("N", self.N), ("Mx", self.Mx), ("My", self.My)):
            if not np.isfinite(_v):
                raise ValueError(
                    f"PrestressAction.{_nm} is not finite ({_v!r}). A "
                    "non-finite action would silently corrupt the "
                    "cumulative demand."
                )

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


    # -- Hard boundary: a PrestressAction is demand-only ----------------
    #    It carries a resultant (N, Mx, My); it is NOT a strain-compatible
    #    element and has no sectional strain.  Any code that duck-types
    #    "give me your prestrain / sectional strain" is mismodelling an
    #    unbonded action as a bonded element -> fail loud.
    @property
    def eps_pe(self):
        r"""
        Guard: a :class:`PrestressAction` has **no** effective prestrain.

        Raises
        ------
        TypeError
            Always.  ``eps_pe`` is the datum of a *bonded*
            :class:`~gensec.geometry.fiber.Tendon`, whose stress is read
            off the sectional strain plane.  A :class:`PrestressAction`
            is a demand-side load with a known resultant and no sectional
            strain; asking it for a prestrain means an unbonded/external
            action is being routed through the strain-compatibility path.
        """
        raise TypeError(
            "PrestressAction has no eps_pe: it is a demand-side load "
            "(unbonded/external prestress), not a strain-compatible "
            "element. Model a bonded tendon with geometry.fiber.Tendon; "
            "for the ULS force of an unbonded tendon use "
            "PrestressAction.from_force_uls."
        )

    def sectional_strain(self, *args, **kwargs):
        r"""
        Guard: a :class:`PrestressAction` has **no** sectional strain.

        Raises
        ------
        TypeError
            Always -- see :attr:`eps_pe`.  The force is known a priori
            (effective prestress + member-level
            :math:`\Delta\sigma_p`), not obtained from the section strain
            plane.
        """
        raise TypeError(
            "PrestressAction has no sectional strain: requesting "
            "sectional strain compatibility of a PrestressAction is a "
            "modelling error (unbonded != bonded)."
        )

    @classmethod
    def from_force_uls(cls, P_eff, x, y, *, Ap, delta_sigma_p,
                       x_ref=0.0, y_ref=0.0, sigma_pd_cap=None,
                       label="", origin="prestress_uls_unbonded"
                       ) -> "PrestressAction":
        r"""
        ULS section action of an **unbonded / external** tendon.

        For a bonded tendon the ULS stress comes out of *sectional*
        strain compatibility (the
        :class:`~gensec.geometry.fiber.Tendon` path inside the
        integrator).  An unbonded/external tendon has **no** sectional
        strain: between two contact points (anchorages / deviators) its
        strain is the member-average, governed by the deformation of the
        whole member (EN 1992-1-1 §5.10.8(2)).  The stress increment from
        the effective prestress to the ULS stress is therefore a
        **member quantity GenSec cannot derive sectionally** and is
        supplied as input:

        .. math::

            \sigma_{pm,t} = \frac{P_{\text{eff}}}{A_p}, \qquad
            \sigma_{p,\text{ULS}} = \sigma_{pm,t}
                                    + \Delta\sigma_{p,\text{ULS}},

        with :math:`\Delta\sigma_{p,\text{ULS}}` either the §5.10.8(3)
        simplified value (recommended :math:`100\ \text{MPa}`, see
        :func:`~gensec.materials.prestress_properties.delta_sigma_p_uls`)
        or a member-level result computed **outside** the sectional
        solver.  The ULS force

        .. math::

            P_{\text{ULS}} = \sigma_{p,\text{ULS}}\,A_p
                           = P_{\text{eff}}
                             + \Delta\sigma_{p,\text{ULS}}\,A_p

        is turned into the demand couple by :meth:`from_force`, in the
        integrator sign convention (a jacked tendon of force
        :math:`P>0` transmits a compression :math:`-P`):

        .. math::

            N_p = -P_{\text{ULS}}, \quad
            M_x = -P_{\text{ULS}}\,(y - y_{\text{ref}}), \quad
            M_y = +P_{\text{ULS}}\,(x - x_{\text{ref}}).

        A **pure action**: never in the capacity hash, never strain-
        compatible.  That is the physical content of "unbonded" -- force
        known a priori, applied to the section, not read off the strain
        plane.

        Parameters
        ----------
        P_eff : float
            Effective prestress force after all losses [N], tension
            positive: :math:`P_{\text{eff}} = \sigma_{pm,t}\,A_p`.
        x, y : float
            Tendon position at the verified section [mm] (for an external
            tendon, the eccentricity set by the deviator layout), in the
            section geometry frame.
        Ap : float
            Tendon area [mm²] (keyword-only): converts the increment to a
            force and lets the optional cap be applied in stress units.
        delta_sigma_p : float
            ULS stress increment :math:`\Delta\sigma_{p,\text{ULS}}`
            [MPa], tension positive, per EN 1992-1-1 §5.10.8.  **Input**
            -- no member-level computation is performed here (see Notes).
            Must be non-negative.
        x_ref, y_ref : float, optional
            Reference point [mm] -- the **same** the demand path uses
            (the section centroid).  Default ``0.0``.
        sigma_pd_cap : float or None, optional
            Optional design-strength cap :math:`f_{pd}` [MPa].  If given
            and :math:`\sigma_{pm,t}+\Delta\sigma_{p,\text{ULS}}` exceeds
            it, the ULS stress is capped and a :class:`UserWarning` is
            emitted -- **never** applied silently.  Default ``None`` (no
            cap; keeping :math:`\sigma_{pm,t}+\Delta\sigma_p\le f_{pd}` is
            the caller's responsibility, as §5.10.8 does not impose it).
        label : str, optional
        origin : str, optional
            Default ``"prestress_uls_unbonded"``.

        Returns
        -------
        PrestressAction

        Raises
        ------
        ValueError
            If ``Ap`` is not positive or ``delta_sigma_p`` is negative.

        Warns
        -----
        UserWarning
            If ``sigma_pd_cap`` is given and the composed ULS stress
            exceeds it.

        Notes
        -----
        **No member-level analysis here, by design.**
        :math:`\Delta\sigma_{p,\text{ULS}}` depends on span, tendon
        profile and deflected shape between contact points -- 1-D member
        kinematics with no representation in a sectional fiber model.
        Deriving it from a beam analysis is a member-level module, out of
        scope for this tool.  The action carries the **representative**
        ULS force; the partial factor :math:`\gamma_P` (§5.10.9,
        favourable/unfavourable) is a combination-layer choice and is
        **not** applied here.

        References
        ----------
        - EN 1992-1-1:2004 §5.10.8, §5.10.9.

        Examples
        --------
        External tendon, :math:`A_p = 1500\ \text{mm}^2`,
        :math:`\sigma_{pm,t}=1000\ \text{MPa}`
        (:math:`P_{\text{eff}}=1.5\ \text{MN}`),
        :math:`\Delta\sigma_p=100\ \text{MPa}`, eccentricity
        :math:`(200,80)` mm about a centroid at :math:`(150,300)` mm:

        >>> a = PrestressAction.from_force_uls(
        ...     1.5e6, 200.0, 80.0, Ap=1500.0, delta_sigma_p=100.0,
        ...     x_ref=150.0, y_ref=300.0)
        >>> a.N            # -(P_eff + dsp*Ap)
        -1650000.0
        >>> a.Mx           # -P_uls * (y - y_ref)
        363000000.0
        >>> a.My           # +P_uls * (x - x_ref)
        82500000.0
        """
        Ap = float(Ap)
        dsp = float(delta_sigma_p)
        if Ap <= 0.0:
            raise ValueError(
                f"PrestressAction.from_force_uls: Ap must be positive "
                f"(got {Ap})."
            )
        if dsp < 0.0:
            raise ValueError(
                "PrestressAction.from_force_uls: delta_sigma_p is the ULS "
                f"stress *increase* and must be >= 0 (got {dsp}). A "
                "relaxation/loss reduction belongs in P_eff, not here."
            )
        sigma_uls = float(P_eff) / Ap + dsp
        if sigma_pd_cap is not None:
            cap = float(sigma_pd_cap)
            if sigma_uls > cap:
                import warnings
                warnings.warn(
                    f"PrestressAction.from_force_uls: composed ULS stress "
                    f"{sigma_uls:.1f} MPa exceeds the supplied cap "
                    f"f_pd={cap:.1f} MPa; capping to f_pd. Check the "
                    "effective prestress and the ULS increment.",
                    UserWarning, stacklevel=2,
                )
                sigma_uls = cap
        return cls.from_force(
            sigma_uls * Ap, x, y, x_ref=x_ref, y_ref=y_ref,
            label=label, origin=origin,
        )

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

def _staging_parents(section) -> np.ndarray:
    r"""
    Per-union-element staging-parent zone indices.

    Concatenates, in the canonical ``rebars + tendons`` order, the
    geometric containing zones (``mat_indices_rebar`` /
    ``mat_indices_tendon``) with the tendon override already resolved
    by the section (:attr:`staging_parent_tendon` — legal only for
    non-embedded tendons, see
    :meth:`~gensec.geometry.geometry.GenericSection._resolve_tendon_parents`).
    Sections built before the zone machinery fall back to zone 0 for
    every element.

    Parameters
    ----------
    section : GenericSection

    Returns
    -------
    numpy.ndarray of int
        Shape ``(n_union,)``.
    """
    n_reb = int(getattr(section, "x_rebars", np.empty(0)).size)
    n_ten = int(getattr(section, "x_tendons", np.empty(0)).size)
    par_r = getattr(section, "mat_indices_rebar", None)
    if par_r is None:
        par_r = np.zeros(n_reb, dtype=int)
    par_t = getattr(section, "staging_parent_tendon", None)
    if par_t is None:
        par_t = getattr(section, "mat_indices_tendon", None)
        if par_t is None:
            par_t = np.zeros(n_ten, dtype=int)
    return np.concatenate([np.asarray(par_r, dtype=int),
                           np.asarray(par_t, dtype=int)])


def _apply_bulk_staging(base_section, state: SectionState, view):
    r"""
    Apply the Phase-8 bulk-staging state to a materialized *view*.

    Three regimes (master plan §1 / primer §2, with the fast-path
    condition corrected to *mask all-True **and** planes all-zero* —
    an all-active state may still carry non-zero locked-in planes at
    the final stage of a composite, and those must reach the solver):

    1. **Trivial** (``bulk_active`` is ``None``, or all-True with
       zero planes): return immediately — no attribute is set, the
       single-bulk pipeline is byte-identical by construction.
    2. **All-active, non-zero planes**: bulk arrays stay shared by
       reference; only ``view.bulk_planes_active`` is attached.
    3. **Masked**: re-slice the bulk fiber arrays to the active zones
       (mask-in-kernel was rejected as a correctness bug — masked
       fibers would still veto planes in ``strains_within_limits``),
       enforce the containment invariant on the kept point elements,
       override the stale geometry attributes from the **exact**
       active polygon (base polygon minus the union of inactive zone
       polygons), and attach the planes.

    The reference point is **pinned**: ``x_centroid`` / ``y_centroid``
    are properties of the *shared full polygon*, which this function
    never replaces — the demand path requires a constant moment
    reference across stages.  ``bbox`` reads the plain attribute
    ``_bounds``, so the per-instance override takes effect without
    touching the class.

    Parameters
    ----------
    base_section : GenericSection
    state : SectionState
    view : GenericSection
        The shallow copy being materialized (mutated in place).

    Raises
    ------
    ValueError
        Zone 0 inactive; a kept (active & bonded) point element whose
        staging parent zone is inactive; empty active bulk.
    """
    ba = state.bulk_active
    planes = state.bulk_planes
    has_planes = planes is not None and bool(
        np.any(np.asarray(planes, dtype=float)))
    masked = ba is not None and not bool(np.all(ba))
    if not masked and not has_planes:
        return                                    # regime 1: fast path

    if masked:                                    # regime 3
        if not bool(ba[0]):
            raise ValueError(
                "materialize_view: zone 0 ('base') is inactive. The "
                "base bulk zone is always active by contract; bulk "
                "deactivation (demolition) is not supported."
            )
        mi = getattr(base_section, "mat_indices", None)
        if mi is None:
            mi = np.zeros(int(base_section.n_fibers), dtype=int)
        mi = np.asarray(mi, dtype=int)
        keep = np.nonzero(ba[mi])[0]
        if keep.size == 0:
            raise ValueError(
                "materialize_view: the active bulk is empty (no fiber "
                "belongs to an active zone). A stage with no bulk "
                "material is meaningless."
            )
        view.x_fibers = base_section.x_fibers[keep]
        view.y_fibers = base_section.y_fibers[keep]
        view.A_fibers = base_section.A_fibers[keep]
        view.mat_indices = mi[keep]
        view.n_fibers = int(keep.size)
        # --- GENSEC T3 C4 composite-SLS (idempotency sentinel) ---
        # C4: view->base fiber map, consumed by verify_sls_staged to
        # scatter view-order bulk stresses back into the base-indexed
        # accumulator (the bulk analogue of view._union_index).
        view._bulk_keep = keep

        # Containment invariant on the elements kept in this view
        # (resolve_stages enforces it for manager-built walks; this
        # protects direct materialize_view callers too).
        parents = _staging_parents(base_section)
        resist = state.active & state.bonded
        bad = np.nonzero(resist & ~ba[parents])[0]
        if bad.size:
            names = getattr(base_section, "zone_names", None)
            i0 = int(bad[0])
            z0 = int(parents[i0])
            zlab = (names[z0] if names and z0 < len(names)
                    else str(z0))
            raise ValueError(
                f"materialize_view: union element {i0} is active but "
                f"its staging-parent bulk zone '{zlab}' (index {z0}) "
                f"is inactive — an element cannot exist before the "
                f"zone that carries it is cast."
            )

        # Exact active geometry: base polygon minus inactive zones.
        from shapely.ops import unary_union
        inactive_polys = [
            base_section.bulk_materials[z - 1][0]
            for z in np.nonzero(~ba)[0] if z > 0
        ]
        active_poly = base_section.polygon.difference(
            unary_union(inactive_polys))
        if active_poly.is_empty:
            raise ValueError(
                "materialize_view: the active geometry is empty "
                "(inactive zones cover the whole section)."
            )
        minx, miny, maxx, maxy = active_poly.bounds
        view._bounds = (float(minx), float(miny),
                        float(maxx), float(maxy))
        view.B = float(maxx - minx)
        view.H = float(maxy - miny)
        view.ideal_gross_area = float(active_poly.area)

    # Regimes 2 and 3: the solver consumes the per-zone planes
    # through this attribute (index-aligned with mat_indices values,
    # which the re-slice above preserves).
    view.bulk_planes_active = (
        np.zeros((int(ba.size), 3), dtype=float) if planes is None
        else np.array(planes, dtype=float, copy=True))


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
    # offset *carrier*: the value is attached to the section the
    # solver is built on and participates in the capacity hash via
    # :meth:`SectionState.capacity_hash`.  As of Phase 5 the fiber
    # integrator **consumes** it -- it evaluates the bulk law at
    # ``eps_section + bulk_eps_init`` at the batch, scalar and tangent
    # sites -- so the offset moves the resistance domain, not only the
    # cache identity (validated by run_bulk_prestrain_validation_new.py).
    view.bulk_eps_init = float(state.bulk_eps_init)

    # ---- Bulk staging (Phase 8, Task 1) ----
    # Re-slice / plane attachment / geometry overrides, or a strict
    # no-op on the trivial state (single-bulk byte-identity anchor).
    _apply_bulk_staging(base_section, state, view)

    # C4: regimes 1-2 keep the full base mesh, so the view->base fiber
    # map is the identity over the (unchanged) fiber count.  Regime 3
    # attached the real re-slice map above.
    if not hasattr(view, "_bulk_keep"):
        view._bulk_keep = np.arange(int(view.n_fibers))

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

        # ---- Phase 8: bulk staging bookkeeping ----
        # Curvature quantum for the plane terms of the capacity hash:
        # QUANT_EPS / max(H, B), so the bucketing error on the
        # extreme-fiber strain chi * D stays <= QUANT_EPS.  Computed
        # once per manager (deterministic cache identity).
        _D = max(float(getattr(base_section, "H", 0.0)),
                 float(getattr(base_section, "B", 0.0)))
        self._chi_quantum = (QUANT_EPS / _D) if _D > 0 else QUANT_EPS
        self._n_zones = 1 + len(getattr(base_section,
                                        "bulk_materials", []) or [])
        self._staging_parents = _staging_parents(base_section)
        self._mat_indices = getattr(base_section, "mat_indices", None)

    @staticmethod
    def _collect_union_materials(section) -> List[int]:
        r"""``id(material)`` per union element, ``rebars + tendons``."""
        ids = [id(r.material) for r in section.rebars]
        for t in getattr(section, "tendons", []):
            ids.append(id(t.material))
        return ids

    def hash_of(self, state: SectionState) -> int:
        """Capacity state hash for *state* against the base geometry."""
        # The curvature quantum is a deterministic property of the
        # manager (set in __init__).  ``getattr`` keeps hash_of usable
        # on partially-built managers (a documented test idiom builds
        # them via ``__new__`` to hash states without ever building a
        # domain); the fallback is capacity_hash's own documented
        # default, and states without zone arrays never consume it.
        return state.capacity_hash(
            self._geom_sig,
            self._union_materials,
            chi_quantum=getattr(self, "_chi_quantum", QUANT_EPS))

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
            # Phase 8: every zone active with zero locked-in planes —
            # the trivial state, byte-identical to the pre-staging
            # pipeline through the materialize_view fast path.
            bulk_active=np.ones(self._n_zones, dtype=bool),
            bulk_planes=np.zeros((self._n_zones, 3), dtype=float),
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

    def resolve_stages(self, stages, *, initially_inactive=None):
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
            (float), ``release`` (bool; whether deactivations are
            force-released vs cleanly removed) and — Phase 8 —
            ``activate_bulk``
            (``{zone_index: (eps0, chi_x, chi_y)}``: cast a bulk zone
            with its **mandatory** locked-in datum plane).  A zone
            targeted by any stage's ``activate_bulk`` starts
            **inactive** (activation-declarative pre-scan: casting a
            zone at stage *k* declares it not yet cast before *k*);
            re-activating an active zone raises.  The keyword-only
            ``initially_inactive`` (sequence of zone indices) marks
            zones as not-yet-cast **without** an ``activate_bulk``
            in this stage list — required by a timeline compiler
            emitting a prefix anchored *before* a zone's casting
            event (Task 2), where the pre-scan has nothing to see.
            ``deactivate_bulk`` (demolition) raises
            ``NotImplementedError`` — it needs the released-stress
            resultant of a bulk region, the bulk analog of
            :meth:`deactivation_actions`.

            Two invariants are enforced per stage (fail-loud):
            the **containment invariant**
            :math:`\mathrm{active}[i] \Rightarrow
            \mathrm{bulk\_active}[\mathrm{parent}(i)]` for every
            union element (which subsumes "reject
            stage(tendon) < stage(bulk)" and protects API-built
            stages, not only YAML), and a **non-empty active bulk**
            (a stage whose active bulk holds no fiber is
            meaningless).  A stage may also carry
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

        # ---- Bulk-staging pre-scan (Phase 8) ---------------------
        # A zone activated at some stage is, by declaration, not yet
        # cast before it: collect every activate_bulk target and
        # start those zones inactive.  Activation-declarative — no
        # separate "initially inactive" list to keep in sync — and a
        # double activation becomes a hard error below.
        planned = set()
        for z in (initially_inactive or ()):
            zi = int(z)
            if not (1 <= zi < self._n_zones):
                raise ValueError(
                    f"resolve_stages: initially_inactive zone index "
                    f"{zi} out of range (section has "
                    f"{self._n_zones} zone(s); zone 0 = 'base' is "
                    f"always active)."
                )
            planned.add(zi)
        for stage in stages:
            ops = stage.get("section_ops", {}) or {}
            if "deactivate_bulk" in ops:
                raise NotImplementedError(
                    "bulk deactivation not yet supported (demolition "
                    "needs the released-stress resultant of a bulk "
                    "region, the bulk analog of deactivation_actions)."
                )
            for z in (ops.get("activate_bulk") or {}):
                zi = int(z)
                if not (1 <= zi < self._n_zones):
                    raise ValueError(
                        f"resolve_stages: activate_bulk zone index "
                        f"{zi} out of range (section has "
                        f"{self._n_zones} zone(s); zone 0 = 'base' "
                        f"is always active and not activatable)."
                    )
                planned.add(zi)
        if planned:
            cur.bulk_active[sorted(planned)] = False

        for k, stage in enumerate(stages):
            ops = stage.get("section_ops", {}) or {}
            deact_idx = list(ops.get("deactivate", []) or [])
            release = bool(ops.get("release", True))

            if ops.get("activate_bulk"):
                ab = ops["activate_bulk"]
                already = [int(z) for z in ab
                           if cur.bulk_active[int(z)]]
                if already:
                    raise ValueError(
                        f"resolve_stages: stage {k} "
                        f"('{stage.get('name', '')}') re-activates "
                        f"already-active bulk zone(s) {already}. A "
                        f"zone is cast exactly once; a second "
                        f"activation would silently overwrite its "
                        f"locked-in datum plane."
                    )
                cur = cur.with_bulk_activated(
                    [int(z) for z in ab],
                    {int(z): tuple(p) for z, p in ab.items()})
            if ops.get("activate"):
                cur = cur.with_activated(ops["activate"])
            if deact_idx:
                cur = cur.with_deactivated(deact_idx)
            if ops.get("eps_override"):
                cur = cur.with_eps_override(ops["eps_override"])
            if ops.get("bulk_plane_delta"):
                # C5: the time-dependent eigenstrain of an interval.
                # Additive, so N sub-steps sum to the interval, and it
                # lands in bulk_planes -- already hashed, so two loss
                # states never collapse onto one cached domain.
                cur = cur.with_bulk_plane_delta(ops["bulk_plane_delta"])
            if "bulk_eps" in ops:
                cur = cur.with_bulk_eps(ops["bulk_eps"])
            # Informational time stamp [days], cumulative since stage 0.
            # Deliberately outside the ops dict (it is not a capacity
            # operation) and deliberately not hashed.
            if "time" in stage:
                cur.time_days = float(stage["time"])
            cur.stage_index = k

            # ---- Containment invariant (Phase 8) -----------------
            # active[i] => bulk_active[parent(i)] for every union
            # element: nothing can be anchored in a zone that is not
            # yet cast.  Checked after all of the stage's ops, so the
            # order of ops within a stage is immaterial.
            if cur.bulk_active is not None:
                bad = np.nonzero(
                    cur.active
                    & ~cur.bulk_active[self._staging_parents])[0]
                if bad.size:
                    names = getattr(self.base_section,
                                    "zone_names", None)
                    i0 = int(bad[0])
                    z0 = int(self._staging_parents[i0])
                    zlab = (names[z0] if names and z0 < len(names)
                            else str(z0))
                    raise ValueError(
                        f"resolve_stages: stage {k} "
                        f"('{stage.get('name', '')}'): union element "
                        f"{i0} is active but its staging-parent bulk "
                        f"zone '{zlab}' (index {z0}) is inactive. "
                        f"Deactivate the element until the zone is "
                        f"cast (activate_bulk), then activate/grout "
                        f"it."
                    )
                # ---- Non-empty active bulk -----------------------
                if (self._mat_indices is not None
                        and not bool(np.any(
                            cur.bulk_active[self._mat_indices]))):
                    raise ValueError(
                        f"resolve_stages: stage {k} "
                        f"('{stage.get('name', '')}'): the active "
                        f"bulk is empty (no fiber belongs to an "
                        f"active zone). A stage with no bulk "
                        f"material is meaningless."
                    )

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
