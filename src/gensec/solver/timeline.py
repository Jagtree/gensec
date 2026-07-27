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

    def __init__(self, events: List[TimelineEvent], points: Dict[str, int],
                 losses_models=None):
        self.events = events
        self.points = points
        #: ``{name: LossModel}`` — the rheological models the
        #: ``interval`` events reference (C5).  Empty when the model
        #: declares no ``losses_models`` block, in which case the
        #: timeline is byte-identical to the pre-C5 behaviour.
        self.losses_models = dict(losses_models or {})

    # -- construction --------------------------------------------------

    @classmethod
    def from_block(cls, block: Sequence[dict], losses_models=None
                   ) -> "ConstructionTimeline":
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
                _validate_losses_spec(payload["losses"], losses_models,
                                      len(events))

            events.append(TimelineEvent(kind, dict(payload)
                                        if isinstance(payload, dict)
                                        else {"value": payload},
                                        len(events)))

        return cls(events, points, losses_models=losses_models)

    # -- validation ----------------------------------------------------

    def validate(self, section) -> None:
        r"""
        Semantic validation of the timeline against a built section.

        Fail-loud inventory (``10_3`` §5.6): unknown zone / tendon /
        demand references; ``cast`` of the base zone (zone 0, never
        castable — master plan §1); ``grout``/``stress`` of an unknown
        tendon; ``grout`` of a **pre-tensioned** tendon (F-C(1) — it has
        no duct, and it never emitted the demand-side action a grout
        stage exists to cancel, so grouting it would inject an orphan
        negative prestress demand); an ``interval`` crossed while a
        stressed tendon is not yet live in the resistance state
        (**F-B**, ``NotImplementedError`` — a scope boundary rather than
        a malformed input, so the type differs deliberately).  Demand
        references are checked by :meth:`resolve` /
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

        # -- F-C(1): grouting a PRE-tensioned tendon --------------------
        # Every tendon reference is known good by here, so the classifier
        # cannot raise on a bad name and mask this message.  The rule it
        # applies is the same one ``resolve`` uses -- one home (rule 5).
        pre_post = _classify_pre_post(self.events, section)

        # -- F-B: a stressed tendon that is not yet LIVE ----------------
        # The question is not "is it post-tensioned", it is "is it in the
        # resistance state yet".  Both answers are one line each, and
        # both windows are silent by the same mechanism: an inactive
        # tendon is skipped by ``_interval_losses``
        # (``if not state.active[...]: continue``), and the first
        # interval in which it *is* active initialises ``relax_from`` at
        # T = t_now - t_stress, so the window is dropped rather than
        # deferred.  Statically decidable -- hence checked here, before
        # any solver exists, rather than in ``resolve``.
        cast_so_far: set = set()
        stressed_so_far: set = set()
        grouted_so_far: set = set()
        for ev in self.events:
            if ev.kind == "cast":
                zref = ev.payload.get("zone")
                if zref is not None:
                    cast_so_far.add(_resolve_zone(zref, zone_names))
            elif ev.kind == "stress":
                stressed_so_far.update(_event_tendon_refs(ev))
            elif ev.kind == "grout":
                grouted_so_far.update(_event_tendon_refs(ev))
            elif ev.kind == "interval":
                pending = []
                for t in sorted(stressed_so_far):
                    if pre_post.get(t) == "post":
                        live = t in grouted_so_far
                        window, remedy = "grout", "grout"
                    else:
                        live = (_tendon_parent_zone(section, t)
                                in cast_so_far)
                        window, remedy = "cast", "cast of its parent zone"
                    if not live:
                        pending.append((t, window, remedy))
                if pending:
                    names = sorted(p[0] for p in pending)
                    window = pending[0][1]
                    remedy = pending[0][2]
                    raise NotImplementedError(
                        f"construction_history[{ev.index}]: tendon(s) "
                        f"{names} are stressed but not yet live across "
                        f"this 'interval' — their '{window}' has not "
                        f"happened.  A tendon is a demand-side load "
                        f"until it enters the resistance state (a "
                        f"post-tensioned one at grouting, a "
                        f"pre-tensioned one at the cast of the concrete "
                        f"around it), so it is inactive here and the "
                        f"loss walk cannot see it: its intrinsic "
                        f"relaxation over the window would be applied "
                        f"nowhere.  The loss is permanent, not "
                        f"deferred — the first interval in which the "
                        f"tendon IS live resumes the increment from "
                        f"rho(t_now - t_stress), so rho(t_live) - "
                        f"rho(0) is never charged; and for a "
                        f"post-tensioned tendon the grouting "
                        f"reconciliation is time-blind too, re-solving "
                        f"at the jacking stress with no elapsed time.  "
                        f"This is a real physical window that GenSec "
                        f"does not integrate — it is not a modelling "
                        f"error on your part, and it is refused rather "
                        f"than silently priced at zero.  Either move "
                        f"the '{remedy}' before the interval, or, if "
                        f"the window is real, compute its relaxation "
                        f"externally and declare the reduced stress at "
                        f"'stress'.  (Scope boundary F-B, sibling of "
                        f"D8.)"
                    )
        for ev in self.events:
            if ev.kind != "grout":
                continue
            pretensioned = sorted(t for t in _event_tendon_refs(ev)
                                  if pre_post.get(t) == "pre")
            if pretensioned:
                raise ValueError(
                    f"construction_history[{ev.index}]: 'grout' of "
                    f"tendon(s) {pretensioned}, which this history "
                    f"classifies as PRE-tensioned — they are stressed "
                    f"before their parent zone is cast.  A pre-tensioned "
                    f"strand is bonded by the concrete cast around it: "
                    f"there is no duct and nothing to grout, and it is a "
                    f"capacity-side element from its zone's cast, so it "
                    f"never emits the demand-side prestress action that "
                    f"a grout stage exists to cancel.  Grouting it would "
                    f"subtract an action that was never added — an "
                    f"orphan NEGATIVE prestress demand.  Either cast the "
                    f"parent zone before the 'stress' (which makes the "
                    f"tendon post-tensioned), or drop the 'grout'."
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
        NotImplementedError
            Raised through :meth:`validate`: an ``interval`` is crossed
            while a stressed tendon is not yet live in the resistance
            state — a post-tensioned one not yet grouted, or a
            pre-tensioned one whose parent zone is not yet cast (F-B).
            The scope boundary is D8's sibling: the tendon is inactive,
            hence invisible to both the unbonded guard and the
            relaxation walk, and the window's relaxation is lost rather
            than deferred.
        """
        self.validate(section)
        zone_names = list(getattr(section, "zone_names", []) or [])
        # The pre/post rule has ONE home (rule 5): the module-level
        # classifier, which ``validate`` has just used on the same
        # events.  The walk below still populates ``pre_post``
        # incrementally -- a tendon stressed *after* an interval must not
        # be pending *at* it -- but it no longer re-derives the rule.
        pre_post_all = _classify_pre_post(self.events, section)
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
        # reconciled grouting strain per tendon, and the jacking stress
        # declared at each ``stress`` event (Task-2 prestress closure).
        grout_eps_init: Dict[str, float] = {}
        stress_sigma: Dict[str, float] = {}

        # ---- C5: the timeline clock and the rheological walk state ----
        # ``t_now`` [days] advances only on ``interval`` events -- every
        # other event is instantaneous.  It is what lets the walk say how
        # old a zone is, and how long a tendon has been under load, at the
        # moment an interval starts.
        t_now = 0.0
        t_stress: Dict[str, float] = {}      # tendon -> clock at stressing
        zone_age0: Dict[int, float] = {}     # zone   -> age at the clock
        zone_curing: Dict[int, float] = {}   # zone   -> t_s
        # The concrete stress HISTORY, per zone: [(age_at_application,
        # stress_plane)].  Creep obeys Boltzmann superposition, so a
        # stress standing for years creeps far less over the next interval
        # than one just applied -- and the emitted state cannot say which
        # is which (reading a stress back with the instantaneous modulus
        # gives the wrong number).  The walk must carry it.
        creep_history: Dict[int, list] = {}
        relax_from: Dict[str, tuple] = {}
        # The SERVICE datum of every staged zone -- distinct from the
        # capacity datum in ``bulk_planes`` (F4).  Both say "this zone is
        # unstressed at its own casting"; they differ because one is drawn
        # by the ULS solver and the other by the linear service solve.
        service_datums: Dict[int, tuple] = {}
        # The frozen emission: {event_index: section_ops}.  Computed ONCE,
        # here; the compiler REPLAYS it and never re-derives.  Losses are
        # timeline physics, frozen like the casting datums.
        losses_ops: Dict[int, dict] = {}
        losses_trace: Dict[int, list] = {}

        # Elements (rebars/tendons) contained in castable zones do not
        # exist before their zone is cast; and a *post-tensioned* tendon
        # (one that is grouted somewhere on the timeline) does not exist
        # in the resistance domain before it is grouted — it is a
        # demand-side load until then.  Both are deactivated at the first
        # stage (``release: False`` — nothing to release, not present
        # yet) and (re)activated at their event (``cast`` for a zone
        # element, ``grout`` for a post-tensioned tendon); otherwise the
        # engine's containment / single-side guards reject the stage.
        nonbase = _nonbase_elements(section)
        pt_tendons = _posttension_union_indices(self, section)
        initial_off = sorted(set(nonbase) | set(pt_tendons))

        def _emit(stage: dict) -> None:
            stage_prefix.append(stage)
            if len(stage_prefix) == 1 and initial_off:
                ops = stage_prefix[0].setdefault("section_ops", {})
                ops["deactivate"] = sorted(set(ops.get("deactivate", []))
                                           | set(initial_off))
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
                sigma_p0 = ev.payload.get("sigma_p0")
                for t in _event_tendon_refs(ev):
                    stressed[t] = ev.index
                    t_stress.setdefault(t, t_now)
                    if sigma_p0 is not None:
                        stress_sigma[t] = float(sigma_p0)
                    # pre/post read from the single classifier; the
                    # assignment stays *here* so the map grows with the
                    # walk (F-B reads it mid-walk).
                    pre_post[t] = pre_post_all[t]
                _emit({"name": f"stress[{ev.index}]", "components": []})

            elif ev.kind == "grout":
                grout_now = [t for t in _event_tendon_refs(ev)
                             if t not in grouted]
                eps_map = self._reconcile_grout(
                    section, grout_now, stress_sigma,
                    (cumN, cumMx, cumMy), ev)
                grout_eps_init.update(eps_map)
                grouted.update(grout_now)
                # Emit the grouting into the walk's stage prefix too, so a
                # later ``cast`` computes its auto datum on a substrate
                # that already carries the bonded tendon at its reconciled
                # strain.  ``activate`` + ``eps_override`` reproduces
                # ``SectionState.with_grouted`` exactly for a tendon whose
                # intrinsic ``bonded`` flag is already set (the ungrouted
                # phase is ``active=False``); the only thing the dedicated
                # primitive adds is atomicity of the single-side invariant.
                gops = {"activate": [_tendon_union_index(section, t)
                                     for t in grout_now],
                        "eps_override": {
                            _tendon_union_index(section, t): eps_map[t]
                            for t in grout_now if t in eps_map}}
                _emit({"name": f"grout[{ev.index}]", "components": [],
                       "section_ops": gops})

            elif ev.kind == "interval":
                # F-B is decided statically, in ``validate``:
                # it needs no equilibrium, only the event
                # bookkeeping.  Nothing to check here.
                days = ev.payload.get("days", ev.payload.get("value"))
                spec = ev.payload.get("losses")
                stage = {"name": f"interval[{ev.index}]", "components": [],
                         "time": float(days) if days is not None else None}
                if spec:
                    if days is None:
                        raise ValueError(
                            f"construction_history[{ev.index}]: an "
                            f"'interval' carrying 'losses' must declare "
                            f"its length in 'days'.  Creep, shrinkage and "
                            f"relaxation are all functions of elapsed "
                            f"time; there is nothing to integrate over."
                        )
                    (ops, tr, creep_history,
                     relax_from) = self._interval_losses(
                        section, mgr, stage_prefix, cast_so_far,
                        (cumN, cumMx, cumMy), spec, float(days), zone_names,
                        t_now, t_stress, zone_age0, zone_curing,
                        creep_history, relax_from, service_datums, ev, tol,
                        max_iter)
                    if ops:
                        stage["section_ops"] = ops
                    losses_ops[ev.index] = ops
                    losses_trace[ev.index] = tr
                _emit(stage)
                if days is not None:
                    t_now += float(days)
                    for z in list(zone_age0):
                        zone_age0[z] += float(days)

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
                service_datums[zi] = self._service_datum(
                    section, mgr, stage_prefix, cast_so_far, zi,
                    (cumN, cumMx, cumMy), zone_names, zone_age0,
                    zone_curing, tol, max_iter)
                cast_so_far.append(zi)
                _emit({"name": f"cast[{ev.index}]",
                       "components": [],
                       "section_ops": {
                           "activate_bulk": {zi: datums[zi]},
                           "activate": [e for e in
                                        _zone_elements(section, zi)
                                        if e not in pt_tendons]}})

        return TimelineResolution(
            datums=datums, explicit_datums=explicit,
            pre_post=pre_post, grouted=frozenset(grouted),
            stressed=dict(stressed), grout_eps_init=grout_eps_init,
            stress_sigma=stress_sigma, losses_ops=losses_ops,
            losses_trace=losses_trace)

    def _service_datum(self, section, mgr, stage_prefix, cast_so_far, zi,
                       demand, zone_names, zone_age0, zone_curing, tol,
                       max_iter):
        r"""
        The **service** casting datum of zone *zi* — the twin of
        :meth:`_auto_datum`, drawn with the *linear* view (C5, finding
        F4).

        :meth:`_auto_datum` solves the substrate with the ULS view
        (design constitutive laws, non-linear) and writes
        :math:`(-\varepsilon_0, -\chi_x, -\chi_y)` into
        ``bulk_planes``.  That is the right datum for the **resistance
        domain**: the new zone enters it from its own zero.

        It is the *wrong* datum for the **losses**, which are a
        linear-viscoelastic, service-level calculation.  Negating a
        ULS-drawn plane against a service-drawn one leaves a residue of
        order :math:`10^{-4}` of strain, i.e. a few MPa of stress that a
        freshly-cast zone does not actually carry — enough, on a young
        topping, to trip the non-linear-creep limit and stop the run.
        "A zone is unstressed at its own casting" is physics; it must not
        depend on which solver drew the plane.

        So the walk draws **both**: the ULS datum into ``bulk_planes``
        (capacity), and this one into ``TimelineResolution`` (service).
        Neither is derived from the other.

        Returns
        -------
        tuple of float
            :math:`(-\varepsilon_0, -\chi_x, -\chi_y)` of the linear
            service solve, or ``(0, 0, 0)`` if no rheological model is in
            play (the field is then never read).

        Notes
        -----
        Uses each zone's own :math:`E_c(t)` from its loss model when one
        is declared for it; otherwise the material's own SLS modulus.  A
        model that declares no ``losses_models`` never reaches here.
        """
        from .losses import (_zone_material, _zone_drying_geometry
                             as _zone_drying, _UNCAST_PLACEHOLDER_E)
        from .sls import resolve_sls_moduli, sls_view
        from .section_state import materialize_view
        from .integrator import FiberSolver

        if not self.losses_models:
            return (0.0, 0.0, 0.0)
        N, Mx, My = demand
        if abs(N) < 1e-30 and abs(Mx) < 1e-30 and abs(My) < 1e-30:
            return (0.0, 0.0, 0.0)

        n_zones = len(zone_names) or 1
        not_cast = [z for z in range(1, n_zones)
                    if z not in cast_so_far and z != zi]
        if not stage_prefix:
            state = mgr.initial_state()
        else:
            states, _h, _b, _d = mgr.resolve_stages(
                stage_prefix, initially_inactive=not_cast + [zi])
            state = states[-1]

        # moduli of the substrate as it stands at the cast
        mod = {}
        for z in range(n_zones):
            mat = _zone_material(section, z)
            if z == zi or not bool(state.bulk_active[z]):
                mod[id(mat)] = _UNCAST_PLACEHOLDER_E
                continue
            lm = self._zone_loss_model(z, zone_names)
            if lm is None:
                mod[id(mat)] = _UNCAST_PLACEHOLDER_E
            else:
                A_c, u_d = _zone_drying(section, z)
                prov = (lm.provider if lm.provider.A_c is not None
                        else lm.provider.with_geometry(A_c, u_d))
                mod[id(mat)] = prov.E_c(zone_age0.get(z, 28.0))

        view = materialize_view(section, state)
        solver = FiberSolver(sls_view(view, resolve_sls_moduli(section, mod)))
        sol = solver.solve_equilibrium(N, Mx, My, tol=tol, max_iter=max_iter)
        if not sol["converged"]:
            raise ValueError(
                f"construction_history: the *service* casting datum of "
                f"zone '{zone_names[zi]}' did not converge under the "
                f"cumulative demand (N={N / 1e3:.1f} kN, "
                f"Mx={Mx / 1e6:.1f} kN·m).  A linear view should always "
                f"converge -- the substrate is likely degenerate."
            )
        return (-sol["eps0"], -sol["chi_x"], -sol["chi_y"])

    def _zone_loss_model(self, z, zone_names):
        r"""The loss model any ``interval`` declares for zone *z*, or
        ``None``.  Read from the events, so the datum drawn at a cast
        knows the rheology the zone will later be given."""
        for ev in self.events:
            if ev.kind != "interval":
                continue
            for ref, spec in (ev.payload.get("losses") or {}).items():
                if _resolve_zone(ref, zone_names) == z:
                    return self.losses_models.get(spec.get("model"))
        return None

    def _interval_losses(self, section, mgr, stage_prefix, cast_so_far,
                         demand, spec, days, zone_names, t_now, t_stress,
                         zone_age0, zone_curing, creep_history, relax_from,
                         service_datums, ev, tol, max_iter):
        r"""
        Compute the time-dependent losses of one ``interval`` (C5).

        Resolves the current stage prefix to a :class:`SectionState` —
        the same device :meth:`_auto_datum` and :meth:`_reconcile_grout`
        use — hands it to the AAEM container
        (:func:`gensec.solver.losses.expand_losses`) and returns the
        ``section_ops`` that encode the end-of-interval strain state.

        **The emission happens exactly once, here.**  The compiler
        replays :attr:`TimelineResolution.losses_ops` and never
        re-derives: losses are timeline physics, frozen like the casting
        datums.  Re-deriving them per combination would let a
        combination's *variable* actions leak into a *permanent*
        creep history — the quiet mismodel this architecture exists to
        prevent.

        Parameters
        ----------
        section : GenericSection
        mgr : StagedDomainManager
        stage_prefix : list of dict
            The stages emitted so far.
        cast_so_far : list of int
            Zones cast so far.
        demand : tuple of float
            The cumulative characteristic ``(N, Mx, My)`` standing at the
            interval — the sustained action the AAEM integrates under.
        spec : dict
            The event's ``losses`` block, ``{zone_ref: {model, age,
            curing}}``.
        days : float
            Length of the interval [days].
        zone_names : list of str
        t_now : float
            The timeline clock at the *start* of the interval [days].
        t_stress : dict
            ``{tendon: clock at stressing}``.
        zone_age0, zone_curing : dict
            Per-zone age at the clock, and curing age — carried across
            intervals by the walk.  Mutated in place.
        creep_history : dict
            ``{zone: [(tau, sigma_plane)]}``.  Carried across intervals.
        ev : TimelineEvent
        tol, max_iter :
            Forwarded to the service solve.

        Returns
        -------
        tuple
            ``(section_ops, trace, creep_history)``.

        Raises
        ------
        ValueError
            A zone with no declared age on its first interval; a declared
            age inconsistent with the clock; an unknown model reference.
        """
        from .losses import expand_losses

        n_zones = len(zone_names) or 1
        not_cast = [z for z in range(1, n_zones) if z not in cast_so_far]
        if not stage_prefix:
            state = mgr.initial_state()
        else:
            states, _h, _b, _d = mgr.resolve_stages(
                stage_prefix, initially_inactive=not_cast)
            state = states[-1]

        models, ages0, ages1, curing = {}, {}, {}, {}
        for zone_ref, zs in spec.items():
            z = _resolve_zone(zone_ref, zone_names)
            name = zs["model"]
            lm = self.losses_models.get(name)
            if lm is None:
                raise ValueError(
                    f"construction_history[{ev.index}]: 'interval.losses' "
                    f"references loss model '{name}', which the "
                    f"'losses_models' block does not define. Known: "
                    f"{sorted(self.losses_models)}."
                )
            declared = zs.get("age")
            if z in zone_age0:
                derived = zone_age0[z]
                if declared is not None \
                        and abs(float(declared) - derived) > 1e-9:
                    raise ValueError(
                        f"construction_history[{ev.index}]: zone "
                        f"'{zone_ref}' is declared to be "
                        f"{float(declared):g} d old at this interval, but "
                        f"the timeline clock makes it {derived:g} d "
                        f"(declared {zone_age0[z] - t_now:+g} d earlier "
                        f"and advanced by the intervening intervals).  An "
                        f"age is declared once, then derived; a mismatch "
                        f"is a modelling error, not a re-declaration."
                    )
                age0 = derived
            else:
                if declared is None:
                    raise ValueError(
                        f"construction_history[{ev.index}]: zone "
                        f"'{zone_ref}' appears in an 'interval.losses' "
                        f"for the first time and declares no 'age'.  "
                        f"Creep and shrinkage both depend on how old the "
                        f"concrete is; the timeline cannot know, and "
                        f"assuming would be a silent normative choice."
                    )
                age0 = float(declared)
                zone_age0[z] = age0
            if z not in zone_curing:
                if "curing" not in zs:
                    raise ValueError(
                        f"construction_history[{ev.index}]: zone "
                        f"'{zone_ref}' declares no 'curing' age.  Drying "
                        f"shrinkage is measured from the end of curing; "
                        f"an assumed curing age is an assumed shrinkage."
                    )
                zone_curing[z] = float(zs["curing"])
            if age0 <= 0.0:
                raise ValueError(
                    f"construction_history[{ev.index}]: zone "
                    f"'{zone_ref}' is {age0:g} d old at the start of the "
                    f"interval.  Concrete cannot creep before it exists; "
                    f"cast it, let it harden, then load it."
                )
            models[z] = lm
            ages0[z] = age0
            ages1[z] = age0 + days
            curing[z] = zone_curing[z]

        tendon_ages0 = {}
        for j, t in enumerate(getattr(section, "tendons", [])):
            nm = getattr(t, "name", None) or f"tendon[{j}]"
            if not state.active[int(section.x_rebars.size) + j]:
                continue
            # A pre-tensioned tendon carries no ``stress`` event: it was
            # stressed against the abutments before the timeline began, so
            # its relaxation clock starts at t = 0.
            tendon_ages0[nm] = max(0.0, t_now - t_stress.get(nm, 0.0))

        ops, trace, creep_history, relax_from = expand_losses(
            section, state, models=models, demand=demand,
            zone_ages_t0=ages0, zone_ages_t=ages1, zone_curing_ages=curing,
            tendon_ages_t0=tendon_ages0, interval_days=days,
            history=creep_history, relax_from=relax_from,
            service_datums=service_datums, label=f"interval[{ev.index}]")
        return ops, trace, creep_history, relax_from

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

    # -- prestress reconciliation (driver reuse) ----------------------

    def _reconcile_grout(self, section, grout_now, stress_sigma, base_demand,
                         ev) -> Dict[str, float]:
        r"""
        Reconciled locked-in strain for the tendons grouted at *ev*.

        **Reuses** the sectional post-tensioning driver rather than
        reimplementing the elastic-shortening / reconciliation algebra:

        1. :func:`gensec.solver.posttension.solve_posttension_sequence`
           re-runs the stressing sequence on the current hardened bulk
           under the cumulative characteristic *base_demand* (the
           sollecitazione present at grouting, applied debit-free), and
           returns per-tendon :math:`\varepsilon_{pe,\text{eff}}` (the
           post-loss jacking strain) and :math:`\varepsilon_{\text{sec,
           grout}}` (the concrete strain at each tendon at grout).
        2. :func:`gensec.solver.posttension.grout` forms the reconciled
           datum

           .. math::

               \varepsilon_{\text{init},j}
                   = \varepsilon_{pe,\text{eff},j}
                     - \varepsilon_{\text{sec,grout},j},

           so that the bonded tendon evaluates
           :math:`\sigma_p(\varepsilon_{\text{sec,grout},j}
           + \varepsilon_{\text{init},j})
           = \sigma_p(\varepsilon_{pe,\text{eff},j})` — the post-loss
           effective stress at the grouting strain datum.

        The compiler then lowers this to an ``eps_override`` on the grout
        stage, so a compiled ``stress`` → ``grout`` sequence is
        byte-equivalent to calling the driver directly (``10_3`` §5
        axis-1).

        Scope: a single-bulk post-tensioning where every grouted tendon
        was stressed **before** any grouting (the standard "stress all,
        then grout all").  Grouting in stages (a second ``grout`` after a
        prior one) is the intra-sequence bonded-stiffness case and is
        **not** modelled — it raises, pointing at
        ``6_8-WARNING_intra_sequence_bonded.md``.

        Parameters
        ----------
        section : GenericSection
        grout_now : list of str
            Tendon names grouted at this event (already de-duplicated
            against previously grouted tendons).
        stress_sigma : dict
            ``{tendon_name: sigma_p0}`` accumulated up to this event.
        base_demand : tuple of float
            Cumulative characteristic ``(N, Mx, My)`` at the grout event
            [N, N·mm].
        ev : TimelineEvent
            The grout event (for error messages).

        Returns
        -------
        dict
            ``{tendon_name: eps_init}`` for the tendons grouted here.

        Raises
        ------
        ValueError
            A tendon is grouted without a prior ``stress`` (no jacking
            stress to reconcile).
        NotImplementedError
            Staged grouting (a prior grout already bonded tendons) — the
            intra-sequence case.
        """
        if not grout_now:
            return {}

        missing = [t for t in grout_now if t not in stress_sigma]
        if missing:
            raise ValueError(
                f"construction_history[{ev.index}]: tendon(s) {missing} "
                f"are grouted but were never stressed (no 'sigma_p0' to "
                f"reconcile). A tendon must be stressed before it is "
                f"grouted."
            )

        from .posttension import solve_posttension_sequence, grout

        tendons = list(getattr(section, "tendons", []) or [])
        nt = len(tendons)
        # sigma_p0 aligned with section.tendons; stressing order = the
        # order the stress events appeared (dict preserves it).
        sigma_arr = [0.0] * nt
        order = []
        for t, s in stress_sigma.items():
            li = _tendon_local_index(section, t)
            sigma_arr[li] = float(s)
            order.append(li)

        bN, bMx, bMy = base_demand
        result = solve_posttension_sequence(
            section, sigma_p0=sigma_arr, order=order,
            base_N=bN, base_Mx=bMx, base_My=bMy)

        grout_local = [_tendon_local_index(section, t) for t in grout_now]
        gres = grout(section, result, indices=grout_local)

        by_local = {rec["tendon"]: rec["eps_init"]
                    for rec in gres.report["reconciliation"]}
        return {t: float(by_local[_tendon_local_index(section, t)])
                for t in grout_now}

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
        pt_tendons = _posttension_union_indices(self, section)
        initial_off = sorted(set(nonbase) | set(pt_tendons))
        stages: List[dict] = []

        def _emit(stage: dict) -> None:
            stages.append(stage)
            if len(stages) == 1 and initial_off:
                ops = stages[0].setdefault("section_ops", {})
                ops["deactivate"] = sorted(set(ops.get("deactivate", []))
                                           | set(initial_off))
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
                           "activate": [e for e in
                                        _zone_elements(section, zi)
                                        if e not in pt_tendons]}})

            elif ev.kind == "stress":
                acts = self._stress_actions(
                    ev, resolution, section, gamma_P)
                _emit({"name": f"stress[{ev.index}]", "components": [],
                       "_prestress_actions": acts})

            elif ev.kind == "grout":
                ops, drop = self._grout_stage(
                    ev, resolution, section, gamma_P)
                _emit({"name": f"grout[{ev.index}]", "components": [],
                       "section_ops": ops, "_prestress_actions": drop})

            elif ev.kind == "interval":
                days = ev.payload.get("days", ev.payload.get("value"))
                stage = {"name": f"interval[{ev.index}]", "components": [],
                         "time": float(days) if days is not None else None}
                # C5: REPLAY the frozen emission.  Never re-derive: the
                # AAEM integrates the *permanent* action standing on the
                # timeline, and a combination's variable part has no
                # business in a creep history.
                ops = resolution.losses_ops.get(ev.index)
                if ops:
                    stage["section_ops"] = ops
                _emit(stage)
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

    def _grout_stage(self, ev, resolution, section, gamma_P):
        r"""
        Emit the grouting stage: activate + reconciled ``eps_init``, and
        cancel the tendon's demand-side action (single-side invariant).

        Grouting flips each tendon to ``active & bonded`` at its
        reconciled locked-in strain (``resolution.grout_eps_init``,
        produced by :meth:`_reconcile_grout` via driver reuse).  In the
        resolved-stage schema this is an ``activate`` of the tendon union
        index plus an ``eps_override`` — arraywise identical to
        :meth:`SectionState.with_grouted` for a tendon whose intrinsic
        ``bonded`` flag is set (the dedicated primitive only adds
        atomicity).

        The same stage must **remove** the tendon's demand-side
        :class:`PrestressAction` — the one the ``stress`` stage added —
        because after grouting the tendon is a resistance element, not a
        load.  ``_check_staged`` accumulates ``_prestress_actions``
        cumulatively, so the negative action here nets the tendon's demand
        contribution to zero from this stage on.  That is the §F
        single-side invariant (``GroutResult.dropped_actions``) made
        explicit; :math:`\gamma_P` is applied to the drop exactly as it
        was to the stressing load, so the cancellation is exact.

        Returns
        -------
        tuple
            ``(section_ops, drop_actions)`` — ``section_ops`` is
            ``{"activate": [...], "eps_override": {...}}``;
            ``drop_actions`` is the list of cancelling
            :class:`PrestressAction` for the grouted tendons.
        """
        activate: List[int] = []
        eps_override: Dict[int, float] = {}
        drop: List[PrestressAction] = []
        for t in _event_tendon_refs(ev):
            ui = _tendon_union_index(section, t)
            activate.append(ui)
            if t in resolution.grout_eps_init:
                eps_override[ui] = resolution.grout_eps_init[t]
            sigma = resolution.stress_sigma.get(t)
            if sigma is not None:
                tinfo = _tendon_info(section, t)
                P = float(sigma) * float(tinfo["Ap"]) * gamma_P
                a = PrestressAction.from_force(
                    P, tinfo["x"], tinfo["y"],
                    x_ref=tinfo["x_ref"], y_ref=tinfo["y_ref"],
                    label=f"grout_drop:{t}", origin="timeline_grout_drop")
                drop.append(PrestressAction(
                    -a.N, -a.Mx, -a.My, label=a.label, origin=a.origin))
        return ({"activate": activate, "eps_override": eps_override}, drop)

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
    grout_eps_init : dict
        ``{tendon_name: eps_init}`` — the **reconciled** locked-in strain
        each grouted tendon carries once bonded,
        :math:`\varepsilon_{\text{init}}
        = \varepsilon_{pe,\text{eff}} - \varepsilon_{\text{sec,grout}}`,
        obtained by **reusing**
        :func:`gensec.solver.posttension.solve_posttension_sequence` and
        :func:`gensec.solver.posttension.grout` (no reimplementation) so
        that the resistance evaluates
        :math:`\sigma_p(\varepsilon_{\text{sec,grout}}
        + \varepsilon_{\text{init}}) = \sigma_p(\varepsilon_{pe,
        \text{eff}})`.  This is what makes a compiled ``stress`` →
        ``grout`` sequence byte-equivalent to the driver (``10_3`` §5
        axis-1).
    stress_sigma : dict
        ``{tendon_name: sigma_p0}`` [MPa] — the jacking stress declared
        at the tendon's ``stress`` event, net of friction/anchorage slip
        (member-level, supplied as input).  Used by the compiler to size
        the demand-side action while the tendon is unbonded and to cancel
        it at grout (the single-side invariant).
    """

    __slots__ = ("datums", "explicit_datums", "pre_post", "grouted",
                 "stressed", "grout_eps_init", "stress_sigma",
                 "losses_ops", "losses_trace")

    def __init__(self, datums, explicit_datums, pre_post, grouted,
                 stressed, grout_eps_init=None, stress_sigma=None,
                 losses_ops=None, losses_trace=None):
        self.datums = datums
        self.explicit_datums = explicit_datums
        self.pre_post = pre_post
        self.grouted = grouted
        self.stressed = stressed
        self.grout_eps_init = grout_eps_init or {}
        #: C5 -- the FROZEN loss emission: ``{event_index: section_ops}``.
        #: Computed exactly once, in :meth:`ConstructionTimeline.resolve`;
        #: :meth:`ConstructionTimeline.compile_combination` **replays** it
        #: and never re-derives.  Re-deriving per combination would let a
        #: combination's *variable* actions leak into a *permanent* creep
        #: history -- losses are timeline physics, frozen like the casting
        #: datums.
        self.losses_ops = losses_ops or {}
        #: ``{event_index: [IntervalLosses, ...]}`` -- the per-step trace,
        #: for reporting.  GenSec's purpose is to give the engineer every
        #: number it used, not only the one it acted on.
        self.losses_trace = losses_trace or {}
        self.stress_sigma = stress_sigma or {}


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


def _validate_losses_spec(spec, losses_models, ev_index):
    r"""
    Structural validation of an ``interval``'s ``losses`` block (C5).

    Runs at :meth:`ConstructionTimeline.from_block` time — before any
    section exists — so a typo in a model name or a missing age surfaces
    at parse, not three solves later.

    Expected shape::

        losses:
          base:    {model: rheo_c45, age: 90, curing: 3}
          topping: {model: rheo_c30, age: 2,  curing: 1}

    ``age`` is the age of the zone [days] **at the start of this
    interval** and ``curing`` the age at the end of curing (:math:`t_s`,
    the origin of drying shrinkage).  Both are **required the first time
    a zone appears** and are *derived* from the timeline clock on later
    intervals; when given again they are cross-checked.  There is no
    default: an assumed curing age is an assumed shrinkage, and this
    project does not make normative choices silently.

    Parameters
    ----------
    spec : dict
        The raw ``losses`` payload.
    losses_models : dict or None
        ``{name: LossModel}`` from the model's ``losses_models`` block.
    ev_index : int
        Event position, for the message.

    Raises
    ------
    ValueError
        Malformed block, unknown key, or a model reference that the
        ``losses_models`` block does not define.
    """
    where = f"construction_history[{ev_index}]: 'interval.losses'"
    if not isinstance(spec, dict) or not spec:
        raise ValueError(
            f"{where} must be a non-empty mapping "
            f"{{zone: {{model, age, curing}}}}, got {spec!r}.  A bare "
            f"model name is not accepted: the rheology of a zone is "
            f"meaningless without its age and its curing age, and "
            f"guessing them would be a silent normative choice."
        )
    known = set(losses_models or {})
    for zone_ref, z in spec.items():
        w = f"{where}, zone '{zone_ref}'"
        if not isinstance(z, dict):
            raise ValueError(
                f"{w}: must be a mapping {{model, age, curing}}, got "
                f"{z!r}."
            )
        unknown = sorted(set(z) - {"model", "age", "curing"})
        if unknown:
            raise ValueError(
                f"{w}: unknown key(s) {unknown}.  Valid: model, age, "
                f"curing."
            )
        if "model" not in z:
            raise ValueError(f"{w}: 'model' is required.")
        if known and z["model"] not in known:
            raise ValueError(
                f"{w}: loss model '{z['model']}' is not defined.  The "
                f"'losses_models' block declares {sorted(known)}."
            )
        for k in ("age", "curing"):
            if k in z and not (isinstance(z[k], (int, float))
                               and float(z[k]) >= 0.0):
                raise ValueError(
                    f"{w}: '{k}' must be a non-negative number of days, "
                    f"got {z[k]!r}."
                )


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
    # Enumerate the tendon objects directly (the ``tendon_names`` /
    # ``x_tendons`` parallel arrays are not populated on every section
    # builder); name resolution matches _tendon_local_index / _tendon_
    # union_index exactly so the three helpers never disagree.
    tendons = list(getattr(section, "tendons", []) or [])
    for k, t in enumerate(tendons):
        nm = getattr(t, "name", None) or f"tendon_{k}"
        if nm == name:
            return {
                "x": float(getattr(t, "x", 0.0)),
                "y": float(getattr(t, "y", 0.0)),
                "Ap": float(getattr(t, "area", getattr(t, "Ap", 0.0))),
                "x_ref": x_ref, "y_ref": y_ref,
            }
    known = [getattr(t, "name", None) or f"tendon_{k}"
             for k, t in enumerate(tendons)]
    raise ValueError(
        f"unknown tendon '{name}'. Known: {known}.")


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


def _tendon_local_index(section, name) -> int:
    r"""Local (tendon-array) index of a tendon by name."""
    for k, t in enumerate(getattr(section, "tendons", []) or []):
        if (getattr(t, "name", None) or f"tendon_{k}") == name:
            return k
    raise ValueError(f"unknown tendon '{name}'.")


def _posttension_union_indices(timeline, section) -> List[int]:
    r"""
    Union indices of tendons grouted somewhere on the timeline.

    A tendon that is grouted is **post-tensioned**: a demand-side load
    until its ``grout`` event, a bonded resistance element after.  It
    must therefore be inactive during the ungrouted phase (activated at
    grout, not at its zone's cast), unlike a pre-tensioned tendon which
    is bonded from casting.
    """
    names = set()
    for ev in timeline.events:
        if ev.kind == "grout":
            names.update(_event_tendon_refs(ev))
    out: List[int] = []
    for t in names:
        try:
            out.append(_tendon_union_index(section, t))
        except ValueError:
            pass
    return sorted(out)


def _classify_pre_post(events, section) -> Dict[str, str]:
    r"""
    Pre/post classification of every stressed tendon — **the one home**.

    A tendon is *post-tensioned* when its parent bulk zone already exists
    at the moment it is stressed (the base zone, index 0, always exists),
    and *pre-tensioned* when it does not: the strand is stressed against
    the abutments and the concrete is cast around it afterwards.  The
    distinction is **temporal** and derived from the history itself;
    :attr:`Tendon.system`, if present, is deliberately not consulted — a
    declared system contradicting the history would be a second,
    divergent home for one fact.

    The walk is purely syntactic: only ``cast`` and ``stress`` events are
    read and no equilibrium is solved, so
    :meth:`ConstructionTimeline.validate` — which runs before any solver
    exists — and :meth:`ConstructionTimeline.resolve` obtain the
    identical answer *by construction* rather than by agreement.

    Parameters
    ----------
    events : sequence of TimelineEvent
        The timeline's physical events, in order.
    section : GenericSection or RectSection
        Supplies :attr:`zone_names` and the tendon-to-parent-zone
        containment map.

    Returns
    -------
    dict
        ``{tendon_name: "pre" | "post"}``, one entry per tendon carrying
        at least one ``stress`` event.  A tendon that is never stressed
        is absent (grouting it is caught downstream, by
        :meth:`ConstructionTimeline._reconcile_grout`, with its own
        message).

    Raises
    ------
    ValueError
        Unknown tendon or zone reference.  Callers that need the
        reference errors reported in their own words must check them
        first.
    """
    zone_names = list(getattr(section, "zone_names", []) or [])
    cast_so_far: set = set()
    out: Dict[str, str] = {}
    for ev in events:
        if ev.kind == "cast":
            zref = ev.payload.get("zone")
            if zref is not None:
                cast_so_far.add(_resolve_zone(zref, zone_names))
        elif ev.kind == "stress":
            for t in _event_tendon_refs(ev):
                parent = _tendon_parent_zone(section, t)
                out[t] = ("post" if (parent in cast_so_far or parent == 0)
                          else "pre")
    return out


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
