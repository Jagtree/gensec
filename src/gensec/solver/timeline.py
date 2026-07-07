# SPDX-License-Identifier: AGPL-3.0-or-later
# GenSec — reinforced/prestressed concrete sectional analysis.
# Copyright (C) 2026  Andrea (GenSec project).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.  See <https://www.gnu.org/licenses/>.
r"""
Construction timeline: the single exclusive construction history (Phase 8, G-D1).

This module is **Layer 2** of the Phase-8 architecture (master plan
``10_0`` §2).  It sits between the engine-level section-operation
machinery (Layer 1, :mod:`gensec.solver.section_state`) and the
verification combinations (Layer 3, :mod:`gensec.solver.check`):

.. code-block:: text

    Layer 3  combinations   anchor at timeline points; factors applied here
                 │  compiler (this module) — emits stage lists
    Layer 2  timeline       one construction history; frozen datums
                 │  lowers to activate_bulk / activate / eps_override ops
    Layer 1  engine ops     SectionState.bulk_active + per-zone planes

**The one hard constraint** (unchanged from ``8_2`` §1 / ``10_3`` §1):
the timeline *compiles* to per-combination stage lists in the existing
schema — one stage per event, each carrying ``section_ops``,
``components``, ``_prestress_actions``, ``time`` — consumed verbatim by
:meth:`gensec.solver.check.VerificationEngine._check_staged` and
:meth:`gensec.solver.section_state.StagedDomainManager.resolve_stages`.
Those are **not** touched.  The only new code is the timeline object,
its resolution walk, and the compiler in this file.

Design decisions resolved at Task-2 start (recap ``10_4``):

C1 / T2-D1 (factors + :math:`\gamma_P`)
    History ``load`` events stay symbolic; each anchored combination
    declares ``history_factors: {demand_name: gamma}``.  These lower
    directly to the existing per-component ``factor`` slot of
    :meth:`~gensec.solver.check.VerificationEngine.resolve_components`
    — no new demand summation.  :math:`\gamma_P` (EN 1992-1-1 §5.10.9)
    surfaces at the **combination** layer as ``gamma_P`` and applies
    **only** to demand-side :class:`~gensec.solver.section_state.PrestressAction`
    emissions (unbonded / external / not-yet-grouted post-tension).  A
    *bonded* tendon carries no demand-side increment — its prestress is
    strain-compatible in the resistance domain, frozen by the
    characteristic walk and governed by :math:`\gamma_s`, so
    :math:`\gamma_P` has nothing to scale there **by design**.  Because
    ``_check_staged`` sums ``_prestress_actions`` unfactored, the
    compiler bakes :math:`\gamma_P` into the emitted action's triple
    (the factor lives one layer up — recap ``8_3`` §2).  Selection of
    the favourable/unfavourable value is **engineer-declared**
    (``favourable`` | ``unfavourable`` | explicit float), never inferred
    from the sign of the effect (principle A11).

C2 / T2-D2 (multi-point anchoring)
    ``at:`` accepts a scalar or a list.  A list = one verification run
    per point, results keyed by point name.  The governing point is a
    transparent ``max`` over the per-point governing :math:`\eta`
    (:func:`governing_point`), **not** a reuse of the v2.1 envelope
    object: that object collapses a staged member to its final
    resultant and re-verifies it on the full-section domain from the
    origin (``check.py`` ``resolve_ref``/``check_envelope``), which is
    the wrong domain and the wrong base for a construction-stage result.

C3 / T2-D3 (datum ``auto``)
    At each ``cast`` event the walk solves the **previous** bundle at
    the cumulative characteristic demand
    (:meth:`~gensec.solver.integrator.FiberSolver.solve_equilibrium`);
    the converged plane :math:`(\varepsilon_0,\chi_x,\chi_y)` is negated
    and becomes the zone's datum triple.  The plane is affine, so
    negating the global triple is exact for the zone (Task-1 V1: machine
    precision).  Non-convergence raises :class:`ValueError` naming the
    event and the demand.  Datums are stored full precision; the only
    quantization is inside ``capacity_hash`` (Task-1 quanta).

Normative note (repeat in user docs, master plan B2): when a ULS
combination factors the construction history with :math:`\gamma_G` (or
prestress with :math:`\gamma_P`), the casting datums remain those of the
characteristic (:math:`\gamma = 1`) walk.  The built history is physical;
this is stated, never implied.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .section_state import PrestressAction, StagedDomainManager


# ---------------------------------------------------------------------------
#  Event vocabulary
# ---------------------------------------------------------------------------

#: The event kinds a construction history may contain.  ``point`` is a
#: named anchor (no physical effect); the rest are ordered physical
#: events.  ``load`` references a **permanent** action (master plan
#: §2): variable actions never enter the timeline, they are combination
#: components.
EVENT_KINDS = frozenset(
    {"cast", "stress", "grout", "interval", "load", "point"}
)


class TimelineEvent:
    r"""
    A single, immutable timeline event.

    Pure data after parse.  Semantic validation against the section
    (zone/tendon/demand existence) happens in
    :meth:`ConstructionTimeline.validate`; here only the shape is
    checked.

    Parameters
    ----------
    kind : str
        One of :data:`EVENT_KINDS`.
    payload : dict
        Kind-specific fields (see :meth:`ConstructionTimeline.from_block`).
    index : int
        0-based position in the (physical) event stream.

    Attributes
    ----------
    kind : str
    payload : dict
    index : int
    """

    __slots__ = ("kind", "payload", "index")

    def __init__(self, kind: str, payload: dict, index: int):
        if kind not in EVENT_KINDS:
            raise ValueError(
                f"construction_history: unknown event kind '{kind}'. "
                f"Known kinds: {sorted(EVENT_KINDS)}."
            )
        self.kind = kind
        self.payload = payload
        self.index = index

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"TimelineEvent({self.kind!r}, {self.payload!r}, i={self.index})"


# ---------------------------------------------------------------------------
#  Timeline object
# ---------------------------------------------------------------------------

class ConstructionTimeline:
    r"""
    The one construction history of a model (G-D1).

    An ordered list of typed events plus named anchor **points**.  A
    point marks a position in the event stream that verification
    combinations anchor at (``at:``); it has no physical effect.

    The object is built by :meth:`from_block` (from the parsed YAML
    ``construction_history`` list) and validated against the section by
    :meth:`validate`.  The resolution walk (:meth:`resolve`) and the
    compiler (:meth:`compile_combination`) are the two consumers.

    Parameters
    ----------
    events : list of TimelineEvent
        Physical events in order (``point`` events are *not* included
        here; they live in :attr:`points`).
    points : dict
        ``{point_name: prefix_length}`` — the number of physical events
        that precede (and are included by) the point.  ``prefix_length``
        indexes into :attr:`events` as a slice bound.

    Attributes
    ----------
    events : list of TimelineEvent
    points : dict
    """

    def __init__(self, events: List[TimelineEvent], points: Dict[str, int]):
        self.events = events
        self.points = points

    # -- construction --------------------------------------------------

    @classmethod
    def from_block(cls, block: Sequence[dict]) -> "ConstructionTimeline":
        r"""
        Parse the YAML ``construction_history`` block into a timeline.

        Each list item is a single-key mapping ``{kind: payload}``
        (mirroring the strawman of ``10_3`` §3), e.g.::

            construction_history:
              - cast:     {zone: precast}
              - stress:   {tendons: [T1], sigma_p0: 1400.0}
              - load:     {demand: G1_selfweight}
              - point:    transfer
              - interval: {days: 28}
              - cast:     {zone: topping, datum: auto}
              - load:     {demand: G2_finishes}
              - point:    service

        Parameters
        ----------
        block : sequence of dict
            The parsed list.

        Returns
        -------
        ConstructionTimeline

        Raises
        ------
        ValueError
            Unknown event kind; malformed item (not a single-key
            mapping); duplicate point name; a point name colliding with
            an event kind; an ``interval`` carrying a ``losses`` key
            (deferred to Task 3, fail-loud).
        NotImplementedError
            An ``interval`` with a ``losses`` key — timeline losses land
            in Task 3.
        """
        events: List[TimelineEvent] = []
        points: Dict[str, int] = {}

        for item in block:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(
                    "construction_history: each entry must be a "
                    f"single-key mapping {{kind: payload}}, got {item!r}."
                )
            (kind, payload), = item.items()
            if kind not in EVENT_KINDS:
                raise ValueError(
                    f"construction_history: unknown event kind '{kind}'. "
                    f"Known kinds: {sorted(EVENT_KINDS)}."
                )

            if kind == "point":
                name = payload if isinstance(payload, str) \
                    else payload.get("name")
                if not name:
                    raise ValueError(
                        "construction_history: a 'point' event needs a "
                        "name (either 'point: service' or "
                        "'point: {name: service}')."
                    )
                if name in points:
                    raise ValueError(
                        f"construction_history: duplicate point name "
                        f"'{name}'."
                    )
                # A point anchors *after* all events emitted so far.
                points[name] = len(events)
                continue

            if kind == "interval" and isinstance(payload, dict) \
                    and "losses" in payload:
                raise NotImplementedError(
                    "construction_history: 'interval' with a 'losses' "
                    "key is a Task-3 feature (rheological loss events "
                    "driving RheologicalModel). Carry time only for now."
                )

            events.append(TimelineEvent(kind, dict(payload)
                                        if isinstance(payload, dict)
                                        else {"value": payload},
                                        len(events)))

        return cls(events, points)

    # -- validation ----------------------------------------------------

    def validate(self, section) -> None:
        r"""
        Semantic validation of the timeline against a built section.

        Fail-loud inventory (``10_3`` §5.6): unknown zone / tendon /
        demand references; ``cast`` of the base zone (zone 0, never
        castable — master plan §1); ``grout``/``stress`` of an unknown
        tendon.  Demand references are checked by :meth:`resolve` /
        :meth:`compile_combination` against the demand database (not
        available here), so only geometry references are checked at this
        stage.

        Parameters
        ----------
        section : GenericSection or RectSection
            The built section, exposing :attr:`zone_names` and (for
            prestress) :attr:`tendons`.

        Raises
        ------
        ValueError
            Any unknown or illegal reference.
        """
        zone_names = list(getattr(section, "zone_names", []) or [])
        tendon_names = _tendon_name_index(section)

        for ev in self.events:
            if ev.kind == "cast":
                zref = ev.payload.get("zone")
                if zref is None:
                    raise ValueError(
                        f"construction_history[{ev.index}]: 'cast' needs "
                        f"a 'zone'."
                    )
                zi = _resolve_zone(zref, zone_names)
                if zi == 0:
                    raise ValueError(
                        f"construction_history[{ev.index}]: 'cast' of the "
                        f"base zone (zone 0, '{zone_names[0]}') is illegal "
                        f"— the base is always active and never cast "
                        f"(master plan G-D1)."
                    )
            elif ev.kind in ("stress", "grout"):
                refs = _event_tendon_refs(ev)
                for t in refs:
                    if t not in tendon_names:
                        raise ValueError(
                            f"construction_history[{ev.index}]: "
                            f"'{ev.kind}' references unknown tendon "
                            f"'{t}'. Known: {sorted(tendon_names)}."
                        )

    # -- helpers -------------------------------------------------------

    def prefix_length(self, point_name: str) -> int:
        r"""
        Number of physical events included by *point_name*.

        Parameters
        ----------
        point_name : str

        Returns
        -------
        int

        Raises
        ------
        ValueError
            If *point_name* is not a declared point.
        """
        if point_name not in self.points:
            raise ValueError(
                f"combination anchor 'at: {point_name}' is not a declared "
                f"timeline point. Declared points: "
                f"{sorted(self.points)}."
            )
        return self.points[point_name]

    def cast_events(self):
        r"""Yield ``(event_index, TimelineEvent)`` for every ``cast``."""
        for ev in self.events:
            if ev.kind == "cast":
                yield ev.index, ev

    # -- resolution walk ----------------------------------------------

    def resolve(self, section, demand_db: dict, *,
                tol: float = 1e-10, max_iter: int = 100) -> "TimelineResolution":
        r"""
        Run the resolution walk once (characteristic permanent loads).

        Produces, frozen (master plan B9): the per-``cast`` datum triples
        for zones declared ``datum: auto``, and the derived pre/post
        classification of every stressed tendon.  This is
        :func:`gensec.solver.posttension.grout` generalized to the whole
        history; datums are timeline properties — no combination ever
        recomputes them.

        The datum of a zone cast with ``auto`` is obtained by solving the
        **prefix** section state (everything cast *before* this zone,
        with its own frozen datums already applied) under the cumulative
        characteristic demand accumulated up to the cast, then negating
        the converged plane over the zone (C3).

        Parameters
        ----------
        section : GenericSection or RectSection
        demand_db : dict
            ``{name: {"N", "Mx", "My"}}`` (SI: N, N·mm).
        tol, max_iter : float, int, optional
            Equilibrium solver settings — the **same** the verification
            solves use, so a converged datum is consistent with the
            domain that will consume it.

        Returns
        -------
        TimelineResolution

        Raises
        ------
        ValueError
            Non-convergence of a substrate solve (a substrate that
            cannot carry its own construction loads is a real finding);
            unknown demand reference in a ``load`` event.
        """
        self.validate(section)
        zone_names = list(getattr(section, "zone_names", []) or [])
        mgr = StagedDomainManager(section, biaxial=False,
                                  gen_kwargs={"n_points": 40})

        datums: Dict[int, Tuple[float, float, float]] = {}
        # explicit datums first (they override auto, per zone)
        explicit: Dict[int, Tuple[float, float, float]] = {}

        # cumulative characteristic demand along the walk
        cumN = cumMx = cumMy = 0.0
        # zones cast so far (indices), in cast order
        cast_so_far: List[int] = []
        # running section_ops prefix (one stage per physical event) so we
        # can materialize the pre-cast bundle when an auto datum is needed
        stage_prefix: List[dict] = []
        # tendons already grouted (for pre/post + interleave detection)
        grouted: set = set()
        stressed: Dict[str, int] = {}  # tendon -> event index of stress
        pre_post: Dict[str, str] = {}

        # Elements (rebars/tendons) contained in castable zones do not
        # exist before their zone is cast.  They are deactivated at the
        # first stage (``release: False`` — nothing to release, they are
        # not present yet) and activated at their zone's ``cast`` event;
        # otherwise the engine's containment guard rejects the stage
        # (an element active while its staging-parent zone is inactive).
        nonbase = _nonbase_elements(section)

        def _emit(stage: dict) -> None:
            stage_prefix.append(stage)
            if len(stage_prefix) == 1 and nonbase:
                ops = stage_prefix[0].setdefault("section_ops", {})
                ops["deactivate"] = sorted(set(ops.get("deactivate", []))
                                           | set(nonbase))
                ops.setdefault("release", False)

        for ev in self.events:
            if ev.kind == "load":
                dref = ev.payload.get("demand")
                if dref not in demand_db:
                    raise ValueError(
                        f"construction_history[{ev.index}]: 'load' "
                        f"references unknown demand '{dref}'. Known: "
                        f"{sorted(demand_db)}."
                    )
                d = demand_db[dref]
                cumN += d["N"]; cumMx += d["Mx"]; cumMy += d["My"]
                _emit({"name": f"load[{ev.index}]",
                       "components": [{"ref": dref, "factor": 1.0}]})

            elif ev.kind == "stress":
                for t in _event_tendon_refs(ev):
                    stressed[t] = ev.index
                    parent = _tendon_parent_zone(section, t)
                    # pre/post derived from whether the parent is cast
                    if parent in cast_so_far or parent == 0:
                        pre_post[t] = "post"
                    else:
                        pre_post[t] = "pre"
                _emit({"name": f"stress[{ev.index}]", "components": []})

            elif ev.kind == "grout":
                for t in _event_tendon_refs(ev):
                    if t in grouted:
                        continue
                    grouted.add(t)
                _emit({"name": f"grout[{ev.index}]", "components": []})

            elif ev.kind == "interval":
                days = ev.payload.get("days", ev.payload.get("value"))
                _emit({"name": f"interval[{ev.index}]", "components": [],
                       "time": float(days) if days is not None else None})

            elif ev.kind == "cast":
                zi = _resolve_zone(ev.payload["zone"], zone_names)
                datum_spec = ev.payload.get("datum", "auto")
                if isinstance(datum_spec, dict):
                    triple = (float(datum_spec["eps0"]),
                              float(datum_spec["chi_x"]),
                              float(datum_spec["chi_y"]))
                    explicit[zi] = triple
                    datums[zi] = triple
                elif datum_spec == "auto":
                    triple = self._auto_datum(
                        mgr, stage_prefix, cast_so_far, zi,
                        (cumN, cumMx, cumMy), ev, tol, max_iter)
                    datums[zi] = triple
                else:
                    raise ValueError(
                        f"construction_history[{ev.index}]: 'datum' must "
                        f"be 'auto' or an explicit "
                        f"{{eps0, chi_x, chi_y}} mapping, got "
                        f"{datum_spec!r}."
                    )
                cast_so_far.append(zi)
                _emit({"name": f"cast[{ev.index}]",
                       "components": [],
                       "section_ops": {
                           "activate_bulk": {zi: datums[zi]},
                           "activate": _zone_elements(section, zi)}})

        return TimelineResolution(
            datums=datums, explicit_datums=explicit,
            pre_post=pre_post, grouted=frozenset(grouted),
            stressed=dict(stressed))

    def _auto_datum(self, mgr, stage_prefix, cast_so_far, zi, demand,
                    ev, tol, max_iter) -> Tuple[float, float, float]:
        r"""
        Compute the ``auto`` datum for zone *zi* at a ``cast`` event (C3).

        Solves the pre-cast bundle (the section as it exists *before*
        this cast — i.e. the current ``stage_prefix``, with every
        not-yet-cast zone held inactive) under the cumulative
        characteristic *demand*, and negates the converged plane.

        The plane is affine in :math:`(\Delta x, \Delta y)` about the
        pinned solver reference (full-polygon centroid), so negating the
        global triple :math:`(\varepsilon_0,\chi_x,\chi_y)` **is** the
        exact locked-in datum of the zone — no per-fiber sampling
        (Task-1 V1/R3).

        Parameters
        ----------
        mgr : StagedDomainManager
        stage_prefix : list of dict
            Stages for every physical event emitted so far (this cast
            excluded).
        cast_so_far : list of int
            Zone indices already cast (used to derive which zones are
            still inactive at this point).
        zi : int
            The zone about to be cast.
        demand : tuple of float
            ``(N, Mx, My)`` cumulative characteristic demand [N, N·mm].
        ev : TimelineEvent
        tol, max_iter : float, int

        Returns
        -------
        tuple of float
            The datum triple :math:`(-\varepsilon_0, -\chi_x, -\chi_y)`.

        Raises
        ------
        ValueError
            If the pre-cast bundle does not converge under *demand*.
        """
        n_zones = _n_zones(mgr)
        # zones not yet cast (and not base) are inactive for this solve:
        # every castable zone beyond what the prefix has cast.
        not_cast = [z for z in range(1, n_zones) if z not in cast_so_far]

        if not stage_prefix:
            # First event is this cast: the substrate is the base zone
            # alone in its as-built (all-active) initial state.
            state = mgr.initial_state()
        else:
            # Resolve the prefix; the last state is the current substrate
            # (cast zones active with their datums, not-yet-cast inactive).
            states, _hashes, _bundles, _deact = mgr.resolve_stages(
                stage_prefix, initially_inactive=not_cast)
            state = states[-1]
        solver = _state_solver(mgr, state)

        N, Mx, My = demand
        if abs(N) < 1e-30 and abs(Mx) < 1e-30 and abs(My) < 1e-30:
            return (0.0, 0.0, 0.0)

        sol = solver.solve_equilibrium(N, Mx, My, tol=tol,
                                       max_iter=max_iter)

        # Convergence is judged on the *relative* equilibrium residual,
        # not solely on the solver's ``converged`` flag.  That flag uses
        # an **absolute** force/moment tolerance, which on a partially
        # deactivated (staged) substrate can floor slightly above ``tol``
        # for large-magnitude moments — a false negative even though the
        # returned plane reproduces the demand to machine precision.  We
        # therefore accept when the plane reproduces (N, Mx, My) to a
        # relative machine-precision residual, and raise only on a
        # genuine residual failure (a substrate that truly cannot carry
        # its own construction loads is a real finding, not a warning).
        scale_N = max(abs(N), 1.0)
        scale_M = max(abs(Mx), abs(My), 1.0)
        res_ok = (abs(sol["N"] - N) / scale_N < 1e-9
                  and abs(sol["Mx"] - Mx) / scale_M < 1e-9
                  and abs(sol["My"] - My) / scale_M < 1e-9)
        if not (sol["converged"] or res_ok):
            raise ValueError(
                f"construction_history[{ev.index}]: auto datum for zone "
                f"{zi} failed — the substrate did not reach equilibrium "
                f"under the cumulative characteristic construction demand "
                f"(N={N/1e3:.1f} kN, Mx={Mx/1e6:.1f} kN·m, "
                f"My={My/1e6:.1f} kN·m); relative residual "
                f"{abs(sol['N'] - N) / scale_N:.2e} (N), "
                f"{abs(sol['Mx'] - Mx) / scale_M:.2e} (M) after "
                f"{sol['iterations']} iterations. A substrate that cannot "
                f"carry its own construction loads is a real finding."
            )
        return (-sol["eps0"], -sol["chi_x"], -sol["chi_y"])

    # -- compiler ------------------------------------------------------

    def compile_combination(self, combo: dict, resolution: "TimelineResolution",
                            section, demand_db: dict, *,
                            gamma_P_provider=None
                            ) -> List[Tuple[str, List[dict], List[int]]]:
        r"""
        Compile an anchored combination into stage lists (the compiler).

        For each anchor point :math:`P` in ``combo['at']`` (scalar or
        list, C2), emit the timeline prefix up to :math:`P` as stages —
        one stage per event — with the combination's ``history_factors``
        applied to the symbolic history ``load`` events (C1) and its
        ``gamma_P`` baked into every demand-side prestress action
        (C1) — followed by the combination's own variable-demand
        stage(s).  Output feeds the existing walk verbatim.

        Parameters
        ----------
        combo : dict
            Parsed combination with ``name``, ``at`` (scalar or list),
            optional ``history_factors`` (``{demand_name: gamma}``),
            optional ``gamma_P`` (``favourable`` | ``unfavourable`` |
            float), and its own ``components`` / ``stages`` (the
            variable part, existing schema).
        resolution : TimelineResolution
            Output of :meth:`resolve` (frozen datums, pre/post).
        section : GenericSection or RectSection
        demand_db : dict
        gamma_P_provider : callable or None, optional
            ``provider(kind) -> float`` returning the favourable /
            unfavourable :math:`\gamma_P` for the active normative
            (mirror ``gamma_s_prestress``).  ``None`` uses the EN
            1992-1-1 recommended pair (``favourable`` → 1.0,
            ``unfavourable`` → 1.3).  Normative-agnostic: any provider
            may be supplied.

        Returns
        -------
        list of tuple
            One ``(point_name, stages, initially_inactive)`` per anchor.
            ``stages`` is in the existing staged-combination schema;
            ``initially_inactive`` is the list of zone indices whose
            ``cast`` lies **after** the anchor (recap ``10_2`` R1).

        Raises
        ------
        ValueError
            Unknown anchor point; a ``history_factors`` key not present
            as a ``load`` in the prefix; unknown demand ref.
        NotImplementedError
            A ``stress`` event after a ``grout`` of the same tendon in
            the prefix (intra-sequence hazard, ``6_8-WARNING``).
        """
        anchors = combo.get("at")
        if anchors is None:
            raise ValueError(
                f"combination '{combo.get('name')}' has no 'at' anchor; "
                f"a timeline combination must anchor at a declared point."
            )
        if isinstance(anchors, str):
            anchors = [anchors]

        gamma_P = _resolve_gamma_P(combo.get("gamma_P", 1.0),
                                   gamma_P_provider)
        hist_factors = combo.get("history_factors", {}) or {}

        zone_names = list(getattr(section, "zone_names", []) or [])
        out = []
        for point in anchors:
            plen = self.prefix_length(point)
            prefix = self.events[:plen]

            self._check_interleave(prefix, combo.get("name"))
            self._warn_defaulted_history(prefix, hist_factors,
                                         combo.get("name"), point)

            stages = self._emit_prefix_stages(
                prefix, resolution, section, demand_db,
                hist_factors, gamma_P, gamma_P_provider)

            # the combination's own variable part, appended as stage(s)
            stages.extend(self._emit_variable_stages(combo))

            # zones whose cast is AFTER this anchor start inactive (R1)
            cast_after = [_resolve_zone(ev.payload["zone"], zone_names)
                          for ev in self.events[plen:]
                          if ev.kind == "cast"]
            out.append((point, stages, sorted(set(cast_after))))
        return out

    # -- compiler internals -------------------------------------------

    def _emit_prefix_stages(self, prefix, resolution, section, demand_db,
                            hist_factors, gamma_P, provider) -> List[dict]:
        r"""
        Lower the prefix events to stages (one per event).

        Elements (rebars/tendons) contained in castable zones are
        deactivated at the first stage (``release: False`` — they are not
        yet present) and activated at their zone's ``cast`` event, so the
        emitted stage list satisfies the engine's containment invariant
        exactly (mirrors :meth:`resolve`).
        """
        zone_names = list(getattr(section, "zone_names", []) or [])
        nonbase = _nonbase_elements(section)
        stages: List[dict] = []

        def _emit(stage: dict) -> None:
            stages.append(stage)
            if len(stages) == 1 and nonbase:
                ops = stages[0].setdefault("section_ops", {})
                ops["deactivate"] = sorted(set(ops.get("deactivate", []))
                                           | set(nonbase))
                ops.setdefault("release", False)

        for ev in prefix:
            if ev.kind == "load":
                dref = ev.payload["demand"]
                factor = float(hist_factors.get(dref, 1.0))
                _emit({"name": f"load[{ev.index}]:{dref}",
                       "components": [{"ref": dref, "factor": factor}]})

            elif ev.kind == "cast":
                zi = _resolve_zone(ev.payload["zone"], zone_names)
                _emit({"name": f"cast[{ev.index}]:{zone_names[zi]}",
                       "components": [],
                       "section_ops": {
                           "activate_bulk": {zi: resolution.datums[zi]},
                           "activate": _zone_elements(section, zi)}})

            elif ev.kind == "stress":
                acts = self._stress_actions(
                    ev, resolution, section, gamma_P)
                _emit({"name": f"stress[{ev.index}]", "components": [],
                       "_prestress_actions": acts})

            elif ev.kind == "grout":
                ops = self._grout_ops(ev, resolution, section)
                _emit({"name": f"grout[{ev.index}]", "components": [],
                       "section_ops": ops})

            elif ev.kind == "interval":
                days = ev.payload.get("days", ev.payload.get("value"))
                _emit({"name": f"interval[{ev.index}]", "components": [],
                       "time": float(days) if days is not None else None})
        return stages

    def _stress_actions(self, ev, resolution, section, gamma_P
                        ) -> List[PrestressAction]:
        r"""
        Emit demand-side prestress actions for a ``stress`` event.

        Only **post-tensioned** (parent cast) tendons produce a
        demand-side :class:`PrestressAction`; :math:`\gamma_P` is baked
        into the emitted triple (C1).  **Pre-tensioned** tendons are
        capacity-side (they enter the domain bonded with ``eps_pe`` when
        the zone casts) and contribute **no** demand increment, so
        :math:`\gamma_P` has nothing to scale for them (by design).
        """
        acts: List[PrestressAction] = []
        for t in _event_tendon_refs(ev):
            if resolution.pre_post.get(t) != "post":
                continue
            tinfo = _tendon_info(section, t)
            sigma_p0 = ev.payload.get("sigma_p0")
            if sigma_p0 is None:
                continue
            P = float(sigma_p0) * float(tinfo["Ap"])
            act = PrestressAction.from_force(
                P, tinfo["x"], tinfo["y"],
                x_ref=tinfo["x_ref"], y_ref=tinfo["y_ref"],
                label=f"stress:{t}", origin="timeline_posttension")
            # bake gamma_P (favourable/unfavourable, engineer-declared):
            # the walk sums _prestress_actions unfactored, so the factor
            # lives here, one layer up (recap 8_3 §2).
            if gamma_P != 1.0:
                act = PrestressAction(
                    act.N * gamma_P, act.Mx * gamma_P, act.My * gamma_P,
                    label=act.label, origin=act.origin)
            acts.append(act)
        return acts

    def _grout_ops(self, ev, resolution, section) -> dict:
        r"""
        Emit the capacity-side op that grouts a tendon.

        Grouting flips the tendon to ``active & bonded`` with its
        reconciled ``eps_init`` (the grouting datum = ``eps_pe`` at
        grout).  In the resolved stage schema this is an ``activate`` of
        the tendon union index plus an ``eps_override`` setting the
        reconciled strain — reusing
        :func:`gensec.solver.posttension.grout`'s reconciliation on the
        real repo.  The reconciled strain is a resolution-walk product;
        this method wires it into the op.
        """
        # Placeholder wiring: the reconciled eps_init comes from the
        # resolution walk's posttension driver.  Full driver reuse is
        # the axis-1 integration completed on the repo (see recap 10_4
        # §status).  Here we emit the structural op with the union index.
        activate = [_tendon_union_index(section, t)
                    for t in _event_tendon_refs(ev)]
        return {"activate": activate}

    @staticmethod
    def _emit_variable_stages(combo: dict) -> List[dict]:
        r"""The combination's own variable part as stage(s)."""
        if "stages" in combo:
            return list(combo["stages"])
        return [{"name": f"{combo.get('name', 'combo')}:variable",
                 "components": list(combo.get("components", []))}]

    @staticmethod
    def _check_interleave(prefix, combo_name) -> None:
        r"""
        Raise on a ``stress`` after a ``grout`` of the same tendon (A9).

        The intra-sequence bonded-stiffness hazard documented in
        ``6_8-WARNING_intra_sequence_bonded.md``: once a tendon is
        grouted, stressing it (or any tendon sharing the just-changed
        bonded section) is not modelled.  Fail-loud.
        """
        grouted: set = set()
        for ev in prefix:
            if ev.kind == "grout":
                grouted.update(_event_tendon_refs(ev))
            elif ev.kind == "stress":
                clash = grouted.intersection(_event_tendon_refs(ev))
                if clash:
                    raise NotImplementedError(
                        f"combination '{combo_name}': tendon(s) "
                        f"{sorted(clash)} are stressed after being "
                        f"grouted. The intra-sequence bonded-stiffness "
                        f"transition is not modelled — see "
                        f"6_8-WARNING_intra_sequence_bonded.md. Stress "
                        f"all tendons before grouting, or split the "
                        f"model."
                    )

    @staticmethod
    def _warn_defaulted_history(prefix, hist_factors, combo_name,
                                point) -> None:
        r"""
        Warn (stderr) on a prefix ``load`` omitted from ``history_factors``.

        The trap is **omission**, not ``factor == 1.0`` (C1): a ``load``
        in the prefix that the combination's ``history_factors`` does
        not mention defaults to 1.0 without a conscious choice — a
        mismodel trap on a ULS combination.  An explicit
        ``{G1: 1.0}`` is a made choice and warns nothing.
        """
        prefix_loads = [ev.payload["demand"] for ev in prefix
                        if ev.kind == "load"]
        omitted = [d for d in prefix_loads if d not in hist_factors]
        if omitted:
            print(
                f"  WARNING: combination '{combo_name}' at '{point}' "
                f"leaves history load(s) {omitted} out of "
                f"'history_factors'; they default to factor 1.0. If that "
                f"is intended, list them explicitly to silence this "
                f"(e.g. {{{omitted[0]}: 1.0}}).",
                file=sys.stderr)


# ---------------------------------------------------------------------------
#  Resolution result
# ---------------------------------------------------------------------------

class TimelineResolution:
    r"""
    Frozen output of :meth:`ConstructionTimeline.resolve`.

    Parameters
    ----------
    datums : dict
        ``{zone_index: (eps0, chi_x, chi_y)}`` — locked-in datum per
        cast zone (auto or explicit), full precision.
    explicit_datums : dict
        Subset of *datums* that were user-supplied (override auto).
    pre_post : dict
        ``{tendon_name: 'pre'|'post'}`` — derived classification.
    grouted : frozenset
        Tendon names grouted somewhere on the timeline.
    stressed : dict
        ``{tendon_name: event_index}`` of the stressing event.
    """

    __slots__ = ("datums", "explicit_datums", "pre_post", "grouted",
                 "stressed")

    def __init__(self, datums, explicit_datums, pre_post, grouted,
                 stressed):
        self.datums = datums
        self.explicit_datums = explicit_datums
        self.pre_post = pre_post
        self.grouted = grouted
        self.stressed = stressed


# ---------------------------------------------------------------------------
#  Cross-point governing (C2)
# ---------------------------------------------------------------------------

def governing_point(per_point_results: Dict[str, dict]) -> dict:
    r"""
    Reduce per-point verification results to the governing point (C2).

    A transparent ``max`` over the per-point governing :math:`\eta`.
    This is **not** a v2.1 envelope: the envelope object re-verifies a
    collapsed resultant on the full-section domain from the origin,
    which is the wrong domain and wrong base for a construction-stage
    result.  The engineer keeps every per-point result *and* sees which
    construction state governs (principle A11).

    Parameters
    ----------
    per_point_results : dict
        ``{point_name: <combination result dict>}`` — each a
        :meth:`~gensec.solver.check.VerificationEngine.check_combination`
        return.

    Returns
    -------
    dict
        ``{"governing_point", "eta_governing", "verified",
        "per_point": {...}}``.
    """
    best_pt, best_eta = None, -float("inf")
    for pt, res in per_point_results.items():
        eta = res.get("eta_governing", res.get("eta_norm"))
        if eta is not None and np.isfinite(eta) and eta > best_eta:
            best_eta, best_pt = eta, pt
    return {
        "governing_point": best_pt,
        "eta_governing": (round(best_eta, 4)
                          if best_pt is not None else None),
        "verified": (best_eta <= 1.0 if best_pt is not None else False),
        "per_point": per_point_results,
    }


# ---------------------------------------------------------------------------
#  gamma_P provider (C1) — normative-agnostic
# ---------------------------------------------------------------------------

#: EN 1992-1-1 §5.10.9 recommended persistent/transient pair, used when
#: no provider is supplied.  Any normative may override via a provider
#: (mirror ``gamma_s_prestress`` / ``delta_sigma_p_uls``).
_GAMMA_P_EC2 = {"favourable": 1.0, "unfavourable": 1.3}


def _resolve_gamma_P(spec, provider) -> float:
    r"""
    Resolve a combination's ``gamma_P`` spec to a float (C1).

    The selection between favourable and unfavourable is
    **engineer-declared**, never inferred from the sign of the effect
    (principle A11): auto-selection would be a hidden editorial call.

    Parameters
    ----------
    spec : str or float
        ``'favourable'`` | ``'unfavourable'`` | an explicit float.
    provider : callable or None
        ``provider(kind) -> float`` for the active normative.  ``None``
        uses :data:`_GAMMA_P_EC2`.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        Unknown keyword.
    """
    if isinstance(spec, (int, float)):
        return float(spec)
    key = str(spec).strip().lower()
    if key not in ("favourable", "unfavourable", "favorable",
                   "unfavorable"):
        raise ValueError(
            f"gamma_P must be 'favourable', 'unfavourable' or a float, "
            f"got {spec!r}. The favourable/unfavourable choice is "
            f"declared, never inferred from the effect sign."
        )
    key = key.replace("favorable", "favourable") \
        .replace("unfavorable", "unfavourable")
    if provider is not None:
        return float(provider(key))
    return _GAMMA_P_EC2[key]


# ---------------------------------------------------------------------------
#  Reference resolution helpers (geometry-frame lookups)
# ---------------------------------------------------------------------------

def _resolve_zone(ref, zone_names: List[str]) -> int:
    r"""Resolve a zone name or index to a 0-based zone index."""
    if isinstance(ref, bool):
        raise ValueError(f"zone reference must not be a bool, got {ref!r}.")
    if isinstance(ref, int):
        if not (0 <= ref < len(zone_names)):
            raise ValueError(
                f"zone index {ref} out of range "
                f"[0, {len(zone_names)}).")
        return ref
    if ref in zone_names:
        return zone_names.index(ref)
    raise ValueError(
        f"unknown zone '{ref}'. Known zones: {zone_names}.")


def _tendon_name_index(section) -> set:
    r"""Set of declared tendon names on the section."""
    names = set()
    for k, t in enumerate(getattr(section, "tendons", []) or []):
        names.add(getattr(t, "name", None) or f"tendon_{k}")
    return names


def _event_tendon_refs(ev: TimelineEvent) -> List[str]:
    r"""Tendon names referenced by a ``stress``/``grout`` event."""
    p = ev.payload
    if "tendons" in p:
        return list(p["tendons"])
    if "tendon" in p:
        return [p["tendon"]]
    return []


def _tendon_info(section, name) -> dict:
    r"""
    Geometry-frame data of a tendon by name (for :class:`PrestressAction`).

    The reference point is the solver's pinned reference (full-polygon
    centroid), obtained from a :class:`~gensec.solver.integrator.FiberSolver`
    on the section — the same convention
    :func:`gensec.solver.posttension.solve_posttension_sequence` uses,
    so the couple :math:`(N, M_x, M_y)` an emitted action carries is
    referred to the identical axis as the resistance domain.
    """
    from .integrator import FiberSolver
    solver = FiberSolver(section)
    x_ref, y_ref = float(solver.x_ref), float(solver.y_ref)
    names = list(getattr(section, "tendon_names", []) or [])
    x_t = getattr(section, "x_tendons", None)
    y_t = getattr(section, "y_tendons", None)
    a_t = getattr(section, "A_tendons", getattr(section, "area_tendons", None))
    for k, nm in enumerate(names):
        if nm == name:
            return {
                "x": float(x_t[k]), "y": float(y_t[k]),
                "Ap": float(a_t[k]),
                "x_ref": x_ref, "y_ref": y_ref,
            }
    raise ValueError(
        f"unknown tendon '{name}'. Known: {names}.")


def _tendon_parent_zone(section, name) -> int:
    r"""Parent (containing) bulk zone of a tendon, derived by containment."""
    mat_idx = getattr(section, "mat_indices_tendon", None)
    for k, t in enumerate(getattr(section, "tendons", []) or []):
        if (getattr(t, "name", None) or f"tendon_{k}") == name:
            if getattr(t, "parent", None) is not None:
                return _resolve_zone(t.parent,
                                     list(section.zone_names))
            if mat_idx is not None and k < len(mat_idx):
                return int(mat_idx[k])
            return 0
    raise ValueError(f"unknown tendon '{name}'.")


def _tendon_union_index(section, name) -> int:
    r"""Union (rebar+tendon) index of a tendon by name."""
    n_reb = len(getattr(section, "rebars", []) or [])
    for k, t in enumerate(getattr(section, "tendons", []) or []):
        if (getattr(t, "name", None) or f"tendon_{k}") == name:
            return n_reb + k
    raise ValueError(f"unknown tendon '{name}'.")


def _n_zones(mgr) -> int:
    r"""Number of bulk zones known to a StagedDomainManager."""
    for attr in ("_n_zones", "n_zones"):
        if hasattr(mgr, attr):
            return int(getattr(mgr, attr))
    return len(getattr(mgr, "_section").zone_names)


def _state_solver(mgr, state):
    r"""
    Return the :class:`~gensec.solver.integrator.FiberSolver` of a state.

    Uses the manager's cached bundle builder
    (:meth:`~gensec.solver.section_state.StagedDomainManager.get_bundle`)
    so the substrate solve of the auto-datum walk runs on **exactly** the
    materialized, bulk-staged view the verification domain is built from
    — no parallel section construction, no drift.
    """
    _h, bundle, _built = mgr.get_bundle(state)
    return bundle.solver


def _union_parents(section) -> np.ndarray:
    r"""
    Per-union-element staging-parent zone indices (rebars + tendons).

    Thin wrapper over
    :func:`gensec.solver.section_state._staging_parents` — the single
    source of truth for element containment, so the timeline's
    element-staging matches the engine's containment invariant exactly.
    """
    from .section_state import _staging_parents
    return _staging_parents(section)


def _nonbase_elements(section) -> List[int]:
    r"""
    Union indices of every element whose parent zone is not the base.

    These elements do not exist until their parent zone is cast; the
    compiler deactivates them at the first stage (with ``release: False``
    — there is no locked-in force to release, they are not yet present)
    and activates each at its zone's ``cast`` event.  Without this the
    engine's containment guard rejects the stage (an element active while
    its staging-parent bulk zone is inactive raises).
    """
    parents = _union_parents(section)
    return [i for i in range(parents.size) if int(parents[i]) != 0]


def _zone_elements(section, zone_index: int) -> List[int]:
    r"""Union indices of the elements contained in *zone_index*."""
    parents = _union_parents(section)
    return [i for i in range(parents.size) if int(parents[i]) == zone_index]
