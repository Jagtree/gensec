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
from .prestress_ledger import (PrestressLedger, STATE_STRESSED,
                               STATE_BONDED)


# ---------------------------------------------------------------------------
#  Event vocabulary
# ---------------------------------------------------------------------------

#: The event kinds a construction history may contain.  ``point`` is a
#: named anchor (no physical effect); the rest are ordered physical
#: events.  ``load`` references a **permanent** action (master plan
#: §2): variable actions never enter the timeline, they are combination
#: components.
EVENT_KINDS = frozenset(
    {"cast", "stress", "grout", "interval", "load", "point", "retrofit"}
)

#: The keys a ``retrofit`` event may carry, and the **only** home of
#: that set: :func:`_parse_retrofit_event` refuses anything outside it
#: by name and prints this tuple beside the refusal.  Every one of them
#: is required — there is no default effective stress, no default strain
#: plane and no default clock, for the reason there is no default
#: concrete class (D1b, ``PreexistingState``).
RETROFIT_KEYS = ("concrete", "tendons")

#: The keys of the ``concrete`` sub-block: the strain plane, the point
#: it is declared about, and the concrete's own clock.  A plane without
#: its reference point is not a plane: read about the wrong point it is
#: wrong by :math:`\chi\,\Delta y`, silently, so ``x_ref`` and
#: ``y_ref`` are as required as the three plane coefficients.
#:
#: The clock lives **here**, beside the plane it belongs to, and not at
#: the top of the block: the two sub-blocks are then symmetric — each
#: declared fact carries its own clock, the concrete's counted from
#: casting and each cable's from its own jacking — and the two clocks
#: cannot be confused for one another by position.
RETROFIT_CONCRETE_KEYS = ("eps0", "chi_x", "chi_y", "x_ref", "y_ref",
                          "age_days")

#: The keys of one entry of the ``tendons`` mapping.  ``age_days`` is a
#: **second, different clock** from ``concrete.age_days``: relaxation
#: runs from the day a cable was tensioned, not from the day its
#: concrete was cast, and on a member being retrofitted the two are
#: decades apart.
RETROFIT_TENDON_KEYS = ("sigma_pe", "age_days")


def _retrofit_number(value, where, key):
    r"""
    Read one declared number, or refuse naming it.

    Parameters
    ----------
    value : object
        The declared value.
    where : str
        Human-readable position, used in the refusal.
    key : str
        The field name, used in the refusal.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If the value is missing, not a number, or not finite.  A
        non-finite declared state would propagate into the strain plane
        of every fibre and the run would complete carrying it.
    """
    import math as _math

    if value is None:
        raise ValueError(
            f"{where}: '{key}' is required and was not declared. "
            f"GenSec derives no part of a pre-existing state: the whole "
            f"point of declaring a retrofit is that these numbers come "
            f"from a previous analysis, a site measurement or an "
            f"external calculation, and nothing in the section says "
            f"them.")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{where}: '{key}'={value!r} is not a number.") from None
    if not _math.isfinite(out):
        raise ValueError(
            f"{where}: '{key}'={value!r} is not finite.")
    return out


def _parse_retrofit_event(payload, index):
    r"""
    Parse and validate a ``retrofit`` event — the declared state of a
    member already in service, before the history begins.

    The physics of the retrofit is
    :class:`~gensec.solver.posttension.PreexistingState`, which has
    existed since Phase 24 with eight required fields and no defaults.
    What did not exist until Phase 34 is any way to *say* it in a
    document: ``io_yaml.py`` never named it, so a writer describing an
    existing member with new cables got the analysis of a virgin
    section and nothing told them (D1b-3).  This function is that
    surface, and it lives here rather than in
    :mod:`gensec.io_yaml` because ``construction_history`` is carried
    raw to :meth:`ConstructionTimeline.from_block`: one home for the
    parse of one block.

    The declared form::

        construction_history:
          - retrofit:
              concrete:
                eps0:     -3.0e-4
                chi_x:     1.5e-7
                chi_y:     0.0
                x_ref:   200.0
                y_ref:   600.0
                age_days: 14600.0
              tendons:
                P_old: {sigma_pe: 1100.0, age_days: 14200.0}

    What the plane means, and what it does not
    ------------------------------------------
    The triple :math:`(\varepsilon_0, \chi_x, \chi_y)` about
    :math:`(x_{\mathrm{ref}}, y_{\mathrm{ref}})` is the **total** strain
    state of the existing concrete:

    .. math::

        \varepsilon_c(x, y) = \varepsilon_0
            + \chi_x\,(y - y_{\mathrm{ref}})
            - \chi_y\,(x - x_{\mathrm{ref}}),

    existing prestress, permanent loads and the whole deferred history
    already combined.  GenSec does not decompose it and does not ask
    which part came from where: a decomposition it cannot verify would
    be a guess dressed as an input.

    Parameters
    ----------
    payload : dict
        The event payload as written.
    index : int
        Position in the event stream, named in every refusal.

    Returns
    -------
    dict
        ``{"concrete_age_days": float, "plane": {...},
        "tendons": {name: {"sigma_pe": float, "age_days": float}}}``,
        every value read and validated.

    Raises
    ------
    ValueError
        A key outside :data:`RETROFIT_KEYS`,
        :data:`RETROFIT_CONCRETE_KEYS` or :data:`RETROFIT_TENDON_KEYS`; a
        missing required field; an empty tendon mapping; a
        non-positive effective stress; a negative clock.  Every refusal
        names the offending key **and** the valid set: naming only the
        first says what went wrong, naming the second says what to
        write instead.
    """
    where = f"construction_history[{index}] (retrofit)"
    if not isinstance(payload, dict):
        raise ValueError(
            f"{where}: the payload must be a mapping carrying "
            f"{list(RETROFIT_KEYS)}, got {payload!r}.")

    unknown = sorted(set(payload) - set(RETROFIT_KEYS))
    if unknown:
        raise ValueError(
            f"{where}: unknown field(s) {unknown}. A 'retrofit' event "
            f"reads exactly these fields: {list(RETROFIT_KEYS)}. A "
            f"field nobody reads is a declaration the engine will not "
            f"act on, and the writer has no way to find that out.")

    conc_spec = payload.get("concrete")
    if not isinstance(conc_spec, dict):
        raise ValueError(
            f"{where}: 'concrete' is required and must be a mapping "
            f"carrying {list(RETROFIT_CONCRETE_KEYS)}, got "
            f"{conc_spec!r}. It is the strain state of the existing "
            f"concrete, and GenSec cannot derive it: the history that "
            f"produced it is not in this document.")
    unknown = sorted(set(conc_spec) - set(RETROFIT_CONCRETE_KEYS))
    if unknown:
        raise ValueError(
            f"{where}: unknown field(s) {unknown} in 'concrete'. The "
            f"block reads exactly {list(RETROFIT_CONCRETE_KEYS)}.")
    plane = {k: _retrofit_number(conc_spec.get(k), f"{where}.concrete", k)
             for k in RETROFIT_CONCRETE_KEYS if k != "age_days"}
    concrete_age = _retrofit_number(
        conc_spec.get("age_days"), f"{where}.concrete", "age_days")
    if concrete_age < 0.0:
        raise ValueError(
            f"{where}.concrete: 'age_days'={concrete_age!r} is "
            f"negative. It is the age of the existing concrete at the "
            f"instant this history begins, counted from casting.")

    tendon_spec = payload.get("tendons")
    if not isinstance(tendon_spec, dict) or not tendon_spec:
        raise ValueError(
            f"{where}: 'tendons' is required and must be a non-empty "
            f"mapping {{tendon name: {{sigma_pe: ..., age_days: "
            f"...}}}}. A retrofit with no pre-existing tendon is not a "
            f"retrofit; declare no 'retrofit' event instead.")
    tendons = {}
    for name, spec in tendon_spec.items():
        tw = f"{where}.tendons['{name}']"
        if not isinstance(spec, dict):
            raise ValueError(
                f"{tw}: must be a mapping carrying "
                f"{list(RETROFIT_TENDON_KEYS)}, got {spec!r}.")
        unknown = sorted(set(spec) - set(RETROFIT_TENDON_KEYS))
        if unknown:
            raise ValueError(
                f"{tw}: unknown field(s) {unknown}. A pre-existing "
                f"tendon reads exactly {list(RETROFIT_TENDON_KEYS)}.")
        sigma = _retrofit_number(spec.get("sigma_pe"), tw, "sigma_pe")
        if sigma <= 0.0:
            raise ValueError(
                f"{tw}: 'sigma_pe'={sigma!r} is not strictly positive. "
                f"It is the stress the cable is at NOW -- the jacking "
                f"stress minus the whole deferred history of creep, "
                f"shrinkage and relaxation -- and a cable at zero or "
                f"negative stress is not a pre-existing tendon.")
        age = _retrofit_number(spec.get("age_days"), tw, "age_days")
        if age < 0.0:
            raise ValueError(
                f"{tw}: 'age_days'={age!r} is negative. It is the time "
                f"since THIS cable was tensioned, a different clock "
                f"from 'concrete.age_days': relaxation runs from the "
                f"day a cable was jacked, not from the day its "
                f"concrete was cast.")
        tendons[str(name)] = {"sigma_pe": sigma, "age_days": age}

    return {"concrete_age_days": concrete_age, "plane": plane,
            "tendons": tendons}


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
                 losses_models=None, zone_rheology=None,
                 reference_modulus=None, rheology=None,
                 tolerances=None, code=None):
        self.events = events
        self.points = points
        #: ``{name: LossModel}`` — the rheological models the
        #: ``interval`` events reference (C5).  Empty when the model
        #: declares no ``losses_models`` block, in which case the
        #: timeline is byte-identical to the pre-C5 behaviour.
        self.losses_models = dict(losses_models or {})
        #: ``{zone name: {"model", "age", "curing"}}`` -- the rheological
        #: identity of a zone, declared once and outside any interval
        #: (P3).  It is what lets a ``cast`` know the substrate's
        #: :math:`E_c(t)` when no interval mentions the substrate.
        self.zone_rheology = dict(zone_rheology or {})
        #: The code's conventional service modulus rule
        #: (:class:`~gensec.materials.base.ReferenceModulusRule`), used
        #: for a datum **only** when the model declares no time axis at
        #: all.  ``None`` selects the shipped default of this build,
        #: which is the Eurocode's; a model under another code passes
        #: its own and no engine code changes.
        self.reference_modulus = reference_modulus
        #: The model's normative declaration
        #: (:class:`~gensec.materials.codes.NormativeCode`), read from
        #: the input's top-level ``code:``.  It is the one place a
        #: model says which code it is verified to, and everything code
        #: dependent is derived from it rather than re-decided per call
        #: site.  ``None`` means the input declared none, in which case
        #: the build's shipped rule applies and each material's own
        #: stamp still governs it.
        from ..materials.codes import resolve_code
        self.code = (None if code is None
                     else resolve_code(model_code=code))
        #: ``'declared'`` | ``'on'`` | ``'off'``; see :meth:`has_time_axis`.
        self.rheology = _normalise_rheology_switch(rheology)
        #: The resolved numerical tolerances
        #: (:class:`gensec.tolerances.Tolerances`).  Never ``None``:
        #: a site that reads a tolerance must find one, not decide.
        from ..tolerances import Tolerances
        self.tolerances = (tolerances if isinstance(tolerances, Tolerances)
                           else Tolerances.from_mapping(tolerances))

    @property
    def has_time_axis(self) -> bool:
        r"""
        Whether this analysis carries time-dependent behaviour.

        Three states, because two of them are answers and the third is
        an inference:

        ``'declared'`` (default)
            Inferred: true when *any* zone is given a rheological
            identity through ``losses_models`` or ``zone_rheology``.
        ``'on'``
            Forced true.  A zone with no declared model then refuses
            instead of falling back to a conventional modulus -- for a
            model that must not quietly degrade into an indicative one.
        ``'off'``
            Forced false.  Creep, shrinkage and relaxation are not
            integrated, and every datum is drawn at the code's
            conventional modulus (:math:`E_{cm}` at 28 d under EN
            1992-1-1).  This is a **declared approximation**, not a
            defect: an engineer checking a section at
            :math:`t = \infty` often wants the conventional answer,
            which is coarser but reproducible and far cheaper to audit
            than a full rheological walk.  What makes it legitimate is
            that it was *asked for* -- the result is wrong in a way the
            model states, which is the opposite of a silent default.
            Declared ``interval.losses`` are then not integrated, and
            the events they were declared on are listed in
            :attr:`TimelineResolution.skipped_losses` so that no report
            can present the result as a rheological one.

        It is a property of the **model**, not of a zone, and
        deliberately so: a model with a time axis in which one zone
        declares no part of it is an incomplete declaration -- that
        zone still restrains the ones that creep, at a stiffness nobody
        stated (finding F-A) -- and must be refused, not filled in from
        a conventional value.  Read per zone, this predicate would
        re-open exactly the branch Phase 12 closed.

        Returns
        -------
        bool
        """
        if self.rheology == "off":
            return False
        if self.rheology == "on":
            return True
        return bool(self.losses_models or self.zone_rheology)

    def _reference_rule(self):
        r"""
        The reference-modulus rule to hand a datum resolution, or
        ``None``.

        Returns
        -------
        ReferenceModulusRule or None
            ``None`` whenever the model has a time axis: there the age
            is a declared fact and a conventional substitute for it is
            the defect this phase removes.
        """
        if self.has_time_axis:
            return None
        if self.reference_modulus is not None:
            return self.reference_modulus
        # Resolve per material from its own stamp, so a model that
        # mixes codes -- a precast beam under EN 1992-1-1 with a
        # topping under ACI 318-19 -- has each element answered under
        # the code it was declared with, instead of one rule applied
        # to all of them.
        #
        # There is no fallback below this line.  Until Phase 21 an
        # undeclared model got the build's shipped rule, which was the
        # Eurocode's: the last surface on which a normative choice
        # happened without being declared.  A material that reaches
        # here with no declaration gets UNDECLARED, which refuses and
        # names the omission -- and the loader refuses earlier still,
        # so reaching here undeclared means the object was built in
        # Python rather than loaded.
        from ..materials.codes import CodeReferenceModulus, UNDECLARED
        return CodeReferenceModulus(
            fallback=self.code if self.code is not None else UNDECLARED)

    # -- construction --------------------------------------------------

    @classmethod
    def from_document(cls, data, **kwargs) -> "ConstructionTimeline":
        r"""
        Build a timeline from a **parsed model document**.

        The one home for *"which top-level keys a timeline is made
        of"*.  :meth:`from_block` takes them apart, which means every
        caller that builds a timeline by hand has to know the list --
        and a caller that does not know it drops the key **silently**.
        Measured: ``zone_rheology`` reached the timeline from exactly
        one site in the tree, and every other construction path lost
        it without a word.  A new timeline-level key added here is
        inherited by every caller instead of forgotten by all but one.

        Parameters
        ----------
        data : dict
            A document as returned by
            :func:`gensec.io_yaml.load_yaml`.
        **kwargs
            Forwarded to :meth:`from_block` -- e.g.
            ``reference_modulus`` for a non-Eurocode build.

        Returns
        -------
        ConstructionTimeline

        Raises
        ------
        KeyError
            If *data* carries no ``construction_history``.

        See Also
        --------
        from_block : the parser this delegates to.
        """
        kwargs.setdefault("rheology", data.get("rheology"))
        kwargs.setdefault("tolerances", data.get("tolerances"))
        kwargs.setdefault("code", data.get("code"))
        return cls.from_block(
            data["construction_history"],
            losses_models=data.get("losses_models"),
            zone_rheology=data.get("zone_rheology"),
            **kwargs)

    @classmethod
    def from_block(cls, block: Sequence[dict], losses_models=None,
                   zone_rheology=None, reference_modulus=None,
                   rheology=None,
                   tolerances=None, code=None) -> "ConstructionTimeline":
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
        losses_models : dict, optional
            ``{name: LossModel}`` an ``interval`` may reference.
        zone_rheology : dict, optional
            ``{zone name: {"model", "age", "curing"}}`` -- see
            :func:`gensec.io_yaml._parse_zone_rheology`.

        Returns
        -------
        ConstructionTimeline

        An ``interval`` carrying a ``losses`` key is **accepted**: the
        spec is validated here by
        :func:`_validate_losses_spec` and integrated by the walk.  Until
        Phase 8 it was refused, and this docstring went on declaring the
        refusal for several phases after the refusal was removed -- a
        declaration whose only effect was to tell a reader that an open
        path was closed.  The exact dual of a silent fallback, and
        removed for the same reason (Phase 31, task zero, Z-1).

        Raises
        ------
        ValueError
            Unknown event kind; malformed item (not a single-key
            mapping); duplicate point name; a point name colliding with
            an event kind; a malformed or dangling ``interval.losses``
            spec, refused by :func:`_validate_losses_spec`.
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

            if kind == "retrofit":
                # A retrofit is not an event in the ordinary sense: it
                # is the state the member is ALREADY in when the
                # history opens.  Anything before it would be an event
                # of a member whose state has not been said yet, so the
                # position is a constraint and not a convention.
                # Order matters between these two refusals.  A second
                # retrofit is ALSO not first, so the ordinal check
                # would answer it -- with the wrong diagnosis.  The
                # more specific reason is taken first.
                if any(e.kind == "retrofit" for e in events):
                    raise ValueError(
                        "construction_history: two 'retrofit' events. "
                        "The pre-existing state is declared once, in "
                        "full; two declarations are two homes for one "
                        "fact and nothing would say which to believe.")
                if events:
                    raise ValueError(
                        f"construction_history[{len(events)}]: a "
                        f"'retrofit' must be the first entry of the "
                        f"history. It declares the state the member is "
                        f"already in before anything happens; an event "
                        f"placed before it would act on a member whose "
                        f"state has not been declared yet.")
                events.append(TimelineEvent(
                    kind, _parse_retrofit_event(payload, len(events)),
                    len(events)))
                continue

            if kind == "interval" and isinstance(payload, dict) \
                    and "losses" in payload:
                _validate_losses_spec(payload["losses"], losses_models,
                                      len(events))

            events.append(TimelineEvent(kind, dict(payload)
                                        if isinstance(payload, dict)
                                        else {"value": payload},
                                        len(events)))

        return cls(events, points, losses_models=losses_models,
                   zone_rheology=zone_rheology,
                   reference_modulus=reference_modulus,
                   rheology=rheology, tolerances=tolerances,
                   code=code)

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

    # -- the declared pre-existing state (D1b-3) -----------------------

    @property
    def retrofit(self):
        r"""
        The parsed ``retrofit`` payload, or ``None``.

        One home for *"does this history begin on an existing
        member"*.  The event is constrained by :meth:`from_block` to be
        the first entry and to occur at most once, so this property
        reads position 0 and nothing else; a search over the whole
        stream would suggest the constraint is softer than it is.

        Returns
        -------
        dict or None
        """
        if self.events and self.events[0].kind == "retrofit":
            return self.events[0].payload
        return None

    def _build_preexisting(self, section):
        r"""
        Turn the declared ``retrofit`` event into the physics object.

        The declaration is written in **tendon names**, because that is
        what a document can say; the driver is written in **tendon
        indices**, because that is what a section array is.  This
        method is the single translation between the two, and it is the
        only place in the walk that knows both.

        Every refusal here names the tendon and the remedy.  In
        particular a cable declared pre-existing *and* named by a
        ``stress`` or ``grout`` event is refused rather than resolved:
        a cable was jacked before this history or during it, it cannot
        be both, and nothing in the document says which declaration to
        believe.  The driver raises the same refusal one level down
        (``already grouted and listed in `order```), but by then the
        message speaks of indices the writer never wrote.

        Parameters
        ----------
        section : GenericSection
            The built section; supplies the tendon names, coordinates
            and moduli.

        Returns
        -------
        gensec.solver.posttension.PreexistingState or None
            ``None`` when the history declares no ``retrofit``, in
            which case every caller below is byte-identical to the
            pre-Phase-34 behaviour.

        Raises
        ------
        ValueError
            An unknown tendon name; a declared cable that a ``stress``
            or ``grout`` event also names.
        """
        payload = self.retrofit
        if payload is None:
            return None
        from .posttension import PreexistingState

        where = "construction_history[0] (retrofit)"
        known = [(getattr(t, "name", None) or f"tendon_{k}")
                 for k, t in enumerate(
                     getattr(section, "tendons", []) or [])]

        # A cable cannot be jacked before this history AND during it.
        touched = set()
        for ev in self.events:
            if ev.kind in ("stress", "grout"):
                touched.update(_event_tendon_refs(ev))

        sigma_pe, ages = {}, {}
        for name, spec in payload["tendons"].items():
            if name not in known:
                raise ValueError(
                    f"{where}: declares tendon '{name}', which the "
                    f"section does not carry. Known tendons: "
                    f"{known}.")
            if name in touched:
                raise ValueError(
                    f"{where}: tendon '{name}' is declared "
                    f"pre-existing and is also named by a 'stress' or "
                    f"'grout' event of this same history. A cable was "
                    f"jacked before this history began or it is jacked "
                    f"during it; it cannot be both, and nothing here "
                    f"says which declaration to believe. Remove it "
                    f"from the 'retrofit' block, or remove the event.")
            j = _tendon_local_index(section, name)
            sigma_pe[j] = float(spec["sigma_pe"])
            ages[j] = float(spec["age_days"])

        plane = payload["plane"]
        return PreexistingState(
            sigma_pe=sigma_pe,
            eps0=float(plane["eps0"]),
            chi_x=float(plane["chi_x"]),
            chi_y=float(plane["chi_y"]),
            x_ref=float(plane["x_ref"]),
            y_ref=float(plane["y_ref"]),
            concrete_age_days=float(payload["concrete_age_days"]),
            tendon_age_days=ages,
        )

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
        # ``{tendon: step index at which it became bonded}``, the step
        # counted in JACKS pulled so far.  This is the timeline's half of
        # the Phase-22 staged-injection repair: it is what lets the driver
        # put an already-injected tendon into the section the NEXT jack is
        # pulled against.  Measured cost of not doing it, on a 400x1200
        # with three tendons: the elastic-shortening loss comes out 5.1 %
        # too large (``22_0 §1``).
        grout_step: Dict[str, int] = {}
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
        # A zone declared in ``zone_rheology`` and NOT cast by this
        # history is there from t = 0, so its declared age is its age at
        # the clock right now.  A zone that IS cast gets its age seeded
        # at its own ``cast`` event instead -- the declared age is "how
        # old when the timeline first sees it", and for a cast zone that
        # instant is the cast.  Seeding here rather than lazily is what
        # makes the age available to the very first event.
        cast_zones = {_resolve_zone(e.payload["zone"], zone_names)
                      for e in self.events if e.kind == "cast"}
        for ref, spec in self.zone_rheology.items():
            z = _resolve_zone(ref, zone_names)
            if z in cast_zones:
                continue
            zone_age0[z] = float(spec["age"])
            if spec.get("curing") is not None:
                zone_curing[z] = float(spec["curing"])
        # The concrete stress HISTORY, per zone: [(age_at_application,
        # stress_plane)].  Creep obeys Boltzmann superposition, so a
        # stress standing for years creeps far less over the next interval
        # than one just applied -- and the emitted state cannot say which
        # is which (reading a stress back with the instantaneous modulus
        # gives the wrong number).  The walk must carry it.
        creep_history: Dict[int, list] = {}
        # The demand the history above is synchronised at.  A ``load``
        # between two intervals moves the demand without touching the
        # history, and the resulting stress increment has an age of
        # application like any other -- carrying the pair (history, the
        # demand it stands at) is what lets the next interval date it.
        # ``None`` until the first interval seeds the history.
        creep_history_demand = None
        relax_from: Dict[str, tuple] = {}
        # The frozen emission: {event_index: section_ops}.  Computed ONCE,
        # here; the compiler REPLAYS it and never re-derives.  Losses are
        # timeline physics, frozen like the casting datums.
        losses_ops: Dict[int, dict] = {}
        losses_trace: Dict[int, list] = {}
        # The per-tendon, per-step record of the walk.  Built here, once,
        # and frozen onto the resolution: it is the only object that
        # knows which tendons were demand-side and which resistance-side
        # at each step -- the information ``_check_staged`` destroys when
        # it accumulates ``_prestress_actions`` into three scalars.
        ledger = PrestressLedger(tag="")
        # ``interval`` events whose declared losses were not integrated
        # because rheology is off (see ``has_time_axis``).
        skipped_losses: List[int] = []

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

        # ---- D1b-3: seed the declared pre-existing state -------------
        # The retrofit is not an event that happens; it is the state
        # the member is in before the first one does.  So it is applied
        # HERE, before the walk starts, and the walk then sees an
        # ordinary section that already carries bonded cables.
        #
        # Three books are seeded, and each one is read by a different
        # consumer downstream:
        #
        # * ``grouted`` / ``grout_step`` -- the driver's view of which
        #   cables stiffen the section a later jack is pulled against
        #   (Phase 22).  ``grout_step = 0`` says "bonded from before
        #   the first jack of this sequence", which is what
        #   ``bonded_before`` means at step 0;
        # * ``grout_eps_init`` -- the locked-in strain the cable
        #   enters the resisting section with, formed by the
        #   declaration object from the declared plane and effective
        #   stress, never re-derived here;
        # * the emitted stage -- ``activate`` + ``eps_override``, the
        #   same pair a ``grout`` event emits, so that every consumer
        #   of the stage prefix (auto datums, the SLS walk, the
        #   staged domain) sees one section and not two.
        preexisting = self._build_preexisting(section)
        pre_local: Dict[str, int] = {}
        if preexisting is not None:
            import numpy as _np
            from .posttension import _resolve_Ep, _tendon_geometry
            names = sorted(self.retrofit["tendons"],
                           key=lambda n: _tendon_local_index(section, n))
            pre_local = {n: _tendon_local_index(section, n) for n in names}
            order_idx = sorted(preexisting.tendons)
            # Coordinates and moduli read from the driver's own
            # accessors, not re-derived off the Tendon objects: a
            # tendon's ``x`` may be ``None`` (centred by default) and
            # its modulus may come from either ``Ep`` or ``E``.  Both
            # facts already have a home.
            _x, _y, _Ap = _tendon_geometry(section)
            _Ep = _resolve_Ep(section, None, len(_x))
            xs = _np.array([float(_x[j]) for j in order_idx])
            ys = _np.array([float(_y[j]) for j in order_idx])
            eps_arr = preexisting.locked_in_strain(
                xs, ys, _np.array([float(_Ep[j]) for j in order_idx]))
            by_local = {j: float(eps_arr[k])
                        for k, j in enumerate(order_idx)}
            ops = {"activate": [], "eps_override": {}}
            for name, j in pre_local.items():
                grouted.add(name)
                grout_step[name] = 0
                pre_post[name] = "pre-existing"
                grout_eps_init[name] = by_local[j]
                ui = _tendon_union_index(section, name)
                ops["activate"].append(ui)
                ops["eps_override"][ui] = by_local[j]
                if ledger is not None:
                    ledger.record_state(0, name, STATE_BONDED,
                                        sigma=float(
                                            preexisting.sigma_pe[j]))
            _emit({"name": "retrofit[0]", "components": [],
                   "section_ops": ops})

        for ev in self.events:
            if ev.kind == "retrofit":
                # Already applied above, before the walk began.  Not a
                # no-op reached by falling through: an explicit skip,
                # so that a reader of the loop does not have to prove
                # the absence of a branch.
                continue
            t_at_event = t_now
            loads_here: List[dict] = []
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
                loads_here.append({"demand": str(dref), "N": float(d["N"]),
                                   "Mx": float(d["Mx"]),
                                   "My": float(d["My"])})
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
                    ledger.record_state(
                        ev.index, t, STATE_STRESSED,
                        sigma=stress_sigma.get(t), pre_post=pre_post[t])
                _emit({"name": f"stress[{ev.index}]", "components": []})

            elif ev.kind == "grout":
                grout_now = [t for t in _event_tendon_refs(ev)
                             if t not in grouted]
                # The state at the grout: the same prefix resolution the
                # casting datum uses, so both datums see one section.
                n_zones_now = len(zone_names) or 1
                not_cast_now = [z for z in range(1, n_zones_now)
                                if z not in cast_so_far]
                if not stage_prefix:
                    grout_state = mgr.initial_state()
                else:
                    _st, _h, _b, _d = mgr.resolve_stages(
                        stage_prefix, initially_inactive=not_cast_now)
                    grout_state = _st[-1]
                eps_map = self._reconcile_grout(
                    section, grout_now, stress_sigma,
                    (cumN, cumMx, cumMy), ev, ledger=ledger,
                    state=grout_state, zone_names=zone_names,
                    zone_ages=zone_age0, grout_step=dict(grout_step),
                    preexisting=preexisting)
                grout_eps_init.update(eps_map)
                grouted.update(grout_now)
                # From now on these tendons stiffen the section that any
                # LATER jack is pulled against.  The step index is the
                # number of jacks pulled so far, i.e. len(stress_sigma):
                # the sequence the driver replays is exactly the tendons
                # stressed up to this point, in stress order.
                for _t in grout_now:
                    grout_step[_t] = len(stress_sigma)
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
                if spec and not self.has_time_axis:
                    # Declared, and deliberately not integrated: the
                    # model asked for the conventional answer.  Recorded
                    # rather than dropped -- a skipped creep model that
                    # leaves no trace cannot be told apart from one that
                    # was never read.
                    skipped_losses.append(ev.index)
                elif spec:
                    if days is None:
                        raise ValueError(
                            f"construction_history[{ev.index}]: an "
                            f"'interval' carrying 'losses' must declare "
                            f"its length in 'days'.  Creep, shrinkage and "
                            f"relaxation are all functions of elapsed "
                            f"time; there is nothing to integrate over."
                        )
                    (ops, tr, creep_history, creep_history_demand,
                     relax_from) = self._interval_losses(
                        section, mgr, stage_prefix, cast_so_far,
                        (cumN, cumMx, cumMy), spec, float(days), zone_names,
                        t_now, t_stress, zone_age0, zone_curing,
                        creep_history, creep_history_demand, relax_from,
                        ev, tol, max_iter)
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
                    # An explicit datum is not solved for, so there is
                    # no residual.  Said, not left blank.
                    ledger.record_zone(
                        ev.index, zi,
                        datum_convergence={
                            "solved": False,
                            "cause": "the datum is declared explicitly "
                                     "in the construction history; no "
                                     "equilibrium was solved for it.",
                        })
                elif datum_spec == "auto":
                    triple = self._auto_datum(
                        section, mgr, stage_prefix, cast_so_far, zi,
                        (cumN, cumMx, cumMy), zone_names, zone_age0,
                        ev, tol, max_iter, ledger=ledger)
                    datums[zi] = triple
                else:
                    raise ValueError(
                        f"construction_history[{ev.index}]: 'datum' must "
                        f"be 'auto' or an explicit "
                        f"{{eps0, chi_x, chi_y}} mapping, got "
                        f"{datum_spec!r}."
                    )
                spec_zr = self._zone_rheology_entry(zi, zone_names)
                if spec_zr is not None:
                    # "How old when the timeline first sees it" -- and
                    # for a cast zone that instant is this one.
                    zone_age0[zi] = float(spec_zr["age"])
                    if spec_zr.get("curing") is not None:
                        zone_curing[zi] = float(spec_zr["curing"])
                ledger.record_zone(ev.index, zi,
                                   auto_datum=datums[zi],
                                   origin=("explicit" if zi in explicit
                                           else "auto"))
                cast_so_far.append(zi)
                _emit({"name": f"cast[{ev.index}]",
                       "components": [],
                       "section_ops": {
                           "activate_bulk": {zi: datums[zi]},
                           "activate": [e for e in
                                        _zone_elements(section, zi)
                                        if e not in pt_tendons]}})

            # -- the per-step record (Phase 13) ------------------------
            # Written by the walk, at the instant it is true, so that a
            # machine-readable dump reads what happened instead of
            # re-deriving it from the event list.  ``t_days`` is the
            # clock at which the step *completes*: for every kind but
            # ``interval`` that equals ``t_days_start``, and for an
            # ``interval`` the pair plus ``duration_days`` states the
            # span the step covers.
            ledger.record_step(
                ev.index,
                kind=ev.kind,
                t_days_start=float(t_at_event),
                t_days=float(t_now),
                duration_days=float(t_now - t_at_event),
                zones_active=[0] + sorted(cast_so_far),
                zones_cast_here=([zi] if ev.kind == "cast" else []),
                loads_applied=loads_here,
                demand_cumulative={"N": float(cumN), "Mx": float(cumMx),
                                   "My": float(cumMy)},
                rheology=self._rheology_snapshot(
                    zone_names, zone_age0, zone_curing,
                    [0] + sorted(cast_so_far)),
            )

        return TimelineResolution(
            datums=datums, explicit_datums=explicit,
            pre_post=pre_post, grouted=frozenset(grouted),
            stressed=dict(stressed), grout_eps_init=grout_eps_init,
            stress_sigma=stress_sigma, losses_ops=losses_ops,
            losses_trace=losses_trace, ledger=ledger,
            skipped_losses=tuple(skipped_losses))

    def _rheology_snapshot(self, zone_names, zone_age0, zone_curing,
                           zones_active):
        r"""
        Per-zone rheological provenance at one instant of the walk.

        This is **time-dependent provenance**, the same species as a
        :class:`~gensec.materials.base.DeclaredRule`'s ``source`` one
        floor down: a run that does not say at what age it let the
        concrete flow is not reproducible, however exact its numbers
        are.

        Three quantities per zone, and the distinction between the
        first two is the whole point:

        ``age_declared``
            what the document states -- how old the zone is *when the
            timeline first sees it*.  An input.
        ``age_derived``
            how old it is **now**, the declared age plus every
            ``interval`` the walk has crossed since.  A consequence of
            the path, and it moves if an interval is inserted.
        ``declared_at``
            which of the two declaration sites said it,
            ``'zone_rheology'`` or ``'interval.losses'``.

        Parameters
        ----------
        zone_names : sequence of str
        zone_age0 : dict
            ``{zone index: derived age [days]}`` as the walk holds it.
        zone_curing : dict
            ``{zone index: curing age [days]}``.
        zones_active : sequence of int
            Zone indices that exist at this instant.

        Returns
        -------
        dict
            ``{zone index: {...}}``.  A zone with no declared model
            carries ``model = None`` and a ``cause`` saying so, which
            the dump turns into a recorded absence rather than a
            silent gap.
        """
        out = {}
        for z in sorted(set(int(v) for v in zones_active)):
            spec = self._zone_rheology_entry(z, zone_names)
            model = self._zone_loss_model(z, zone_names)
            name = zone_names[z] if z < len(zone_names) else f"zone_{z}"
            if spec is not None:
                site = "zone_rheology"
            elif model is not None:
                site = "interval.losses"
            else:
                site = None
            entry = {
                "zone_name": str(name),
                "model": (None if spec is None else str(spec["model"])),
                "age_declared": (None if spec is None
                                 else float(spec["age"])),
                "age_derived": (None if z not in zone_age0
                                else float(zone_age0[z])),
                "curing": (None if z not in zone_curing
                           else float(zone_curing[z])),
                "declared_at": site,
            }
            if site is None:
                entry["cause"] = (
                    f"zone '{name}' declares no rheological model: it "
                    f"appears neither in 'zone_rheology' nor in any "
                    f"'interval.losses'.  Its age is therefore not "
                    f"tracked and no E_c(t) is attributed to it.")
            out[z] = entry
        return out

    def _zone_rheology_entry(self, z, zone_names):
        r"""
        The :attr:`zone_rheology` entry of zone *z*, or ``None``.

        Parameters
        ----------
        z : int
        zone_names : sequence of str

        Returns
        -------
        dict or None
            ``{"model", "age", "curing"}``.
        """
        for ref, spec in self.zone_rheology.items():
            if _resolve_zone(ref, zone_names) == z:
                return spec
        return None

    def _zone_loss_model(self, z, zone_names):
        r"""
        The rheological model of zone *z*, or ``None``.

        Two declaration sites, one fact, and a guard between them:

        1. :attr:`zone_rheology` --- the zone's declared identity, its
           one home.  Reachable from **any** event, which is what lets a
           ``cast`` draw its datum on the substrate's :math:`E_c(t)`
           even when no ``interval`` mentions the substrate.
        2. the ``interval.losses`` entries --- the pre-P3 site, kept
           because it is what every shipped model uses.  Scanned over
           *all* events, future ones included, so the identity a zone
           will be given later is already known at its cast.

        A zone declared in both must name the **same** model.  Two homes
        for one fact are permitted only while they are enforced to
        agree; unenforced redundancy is two chances to be wrong and none
        to notice (F7's lesson, 12_0 §2).

        Parameters
        ----------
        z : int
        zone_names : sequence of str

        Returns
        -------
        LossModel or None

        Raises
        ------
        ValueError
            The two sites name different models for the same zone.
        """
        declared = self._zone_rheology_entry(z, zone_names)
        name_declared = declared["model"] if declared else None

        name_interval = None
        for ev in self.events:
            if ev.kind != "interval":
                continue
            for ref, spec in (ev.payload.get("losses") or {}).items():
                if _resolve_zone(ref, zone_names) != z:
                    continue
                if spec.get("model") is not None and name_interval is None:
                    name_interval = spec.get("model")

        if (name_declared is not None and name_interval is not None
                and name_declared != name_interval):
            zn = zone_names[z] if z < len(zone_names) else str(z)
            raise ValueError(
                f"zone '{zn}' is given two different rheological models: "
                f"'{name_declared}' in 'zone_rheology' and "
                f"'{name_interval}' in an 'interval.losses'.  A zone has "
                f"one concrete; declare it once, in 'zone_rheology', and "
                f"let the intervals reference the zone."
            )
        name = name_declared if name_declared is not None else name_interval
        return self.losses_models.get(name) if name is not None else None

    def _interval_losses(self, section, mgr, stage_prefix, cast_so_far,
                         demand, spec, days, zone_names, t_now, t_stress,
                         zone_age0, zone_curing, creep_history,
                         creep_history_demand, relax_from,
                         ev, tol, max_iter):
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
        creep_history_demand : tuple of float or None
            The ``(N, Mx, My)`` *creep_history* is synchronised at.  The
            difference against the cumulative demand standing now is
            everything the ``load`` events between the two intervals
            added, and it is dated at this interval's start rather than
            left to flow zero days.  ``None`` before the first interval.
        ev : TimelineEvent
        tol, max_iter :
            Forwarded to the service solve.

        Returns
        -------
        tuple
            ``(section_ops, trace, creep_history, creep_history_demand,
            relax_from)``.

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

        (ops, trace, creep_history, creep_history_demand,
         relax_from) = expand_losses(
            section, state, models=models, demand=demand,
            zone_ages_t0=ages0, zone_ages_t=ages1, zone_curing_ages=curing,
            tendon_ages_t0=tendon_ages0, interval_days=days,
            history=creep_history, history_demand=creep_history_demand,
            relax_from=relax_from, label=f"interval[{ev.index}]")
        return (ops, trace, creep_history, creep_history_demand,
                relax_from)

    def _zone_providers(self, section, zone_names, n_zones):
        r"""
        Geometry-bound rheological provider of every modelled zone.

        The provider is bound to the zone's own drying geometry
        (:math:`A_c`, :math:`u`) here because the caller is the only
        party that knows it: the timeline holds the section, the
        rheological model does not.

        Parameters
        ----------
        section : GenericSection
        zone_names : sequence of str
        n_zones : int

        Returns
        -------
        dict
            ``{zone index: RheologicalModel}``, only for zones an
            ``interval`` declares a loss model for.
        """
        from .losses import _zone_drying_geometry

        out = {}
        for z in range(n_zones):
            lm = self._zone_loss_model(z, zone_names)
            if lm is None:
                continue
            A_c, u_d = _zone_drying_geometry(section, z)
            p = lm.provider
            out[z] = p if p.A_c is not None else p.with_geometry(A_c, u_d)
        return out

    def _datum_view(self, section, state, zone_names, zone_ages, *,
                    casting_zone=None, moduli=None):
        r"""
        The **service** view a datum is drawn on, and its modulus map.

        The one home for this construction (rule 5).  Three sites need
        it and they must agree to the last digit, because a datum drawn
        on one modulus and read back on another leaves a residue
        :math:`E_p\,(\varepsilon_{\mathrm{sec,read}}
        - \varepsilon_{\mathrm{sec,draw}})` at every tendon -- which
        is the whole of the F4/F5 defect, measured at
        :math:`+7.08` MPa on the flagship (12_5 §7).

        The view is the materialized state of the section, linearised
        by :func:`~gensec.solver.sls.sls_view` at the moduli
        :class:`~gensec.solver.sls.DatumPolicy` resolves.  Same state,
        same geometry, same solver reference as the resistance
        bundle -- only the constitutive law is replaced, because a
        datum is a service fact and the design law describes a concrete
        that does not exist (12_0 §0).

        Parameters
        ----------
        section : GenericSection
        state : SectionState
            The state the datum is drawn at.
        zone_names : sequence of str
        zone_ages : dict
            ``{zone index: age [days]}`` at this instant.
        casting_zone : int or None, optional
            The zone being cast, held absent for its own solve.
        moduli : dict, optional
            Explicit per-slot overrides.  They win over the policy
            (12_7 §2): an override that a provider could silently
            outrank would be a dead override by design.

        Returns
        -------
        tuple
            ``(FiberSolver, MaterialMap)``.

        Raises
        ------
        ValueError
            A zone exists at this instant and neither declares a
            rheological model nor receives an explicit modulus.  The
            message carries :meth:`~gensec.solver.sls.DatumPolicy.explain`'s
            note, which says what to do about it.
        """
        from .sls import sls_view
        from .section_state import materialize_view
        from .integrator import FiberSolver

        modmap = self._datum_moduli(section, state, zone_names, zone_ages,
                                    casting_zone=casting_zone,
                                    moduli=moduli)
        view = materialize_view(section, state)
        return FiberSolver(sls_view(view, modmap)), modmap

    def _datum_moduli(self, section, state, zone_names, zone_ages, *,
                      casting_zone=None, moduli=None):
        r"""
        The service moduli a datum is drawn at -- **one** resolution.

        Both datum sites (the casting datum and the grout
        reconciliation) resolve here, through the single
        :func:`~gensec.solver.sls.resolve_moduli`, with the ratified
        precedence

        .. math::

            \text{explicit} \;\rightarrow\;
            \text{policy} \;\rightarrow\; \text{raise}

        (12_7 §2).  An explicit override that a provider could outrank
        would be a dead override by design, which is the same class of
        defect as a dead override by accident.

        Parameters
        ----------
        section : GenericSection
        state : SectionState
        zone_names : sequence of str
        zone_ages : dict
            ``{zone index: age [days]}`` at this instant.
        casting_zone : int or None, optional
        moduli : dict, optional
            Explicit per-slot overrides.

        Returns
        -------
        MaterialMap

        Raises
        ------
        ValueError
            A zone exists here and neither declares a rheological
            model, nor is covered by a reference rule, nor receives an
            explicit modulus.
        """
        from .sls import DatumPolicy, resolve_moduli

        n_zones = int(np.asarray(state.bulk_active).size)
        providers = self._zone_providers(section, zone_names or [], n_zones)
        return resolve_moduli(
            section,
            DatumPolicy(state=state, providers=providers,
                        ages=zone_ages or {}, casting_zone=casting_zone,
                        reference=self._reference_rule()),
            explicit=dict(moduli or {}),
        )

    def _auto_datum(self, section, mgr, stage_prefix, cast_so_far, zi,
                    demand, zone_names, zone_ages, ev, tol, max_iter,
                    moduli=None, ledger=None) -> Tuple[float, float, float]:
        r"""
        Compute the ``auto`` datum for zone *zi* at a ``cast`` event (C3).

        Solves the pre-cast substrate (the section as it exists
        *before* this cast — i.e. the current ``stage_prefix``, with
        every not-yet-cast zone held inactive) under the cumulative
        characteristic *demand*, and negates the converged plane.

        The solve runs on the **service** view (:meth:`_datum_view`),
        not on the resistance bundle.  A datum is the statement *"this
        zone is unstressed at this instant"*, and an instant of the
        real structure is elastic and service-level; the design law,
        with :math:`\gamma_c` and :math:`\alpha_{cc}` folded into its
        tangent, describes a concrete 30–41 % too soft and would lock
        in that error as the zone's own zero (12_0 §1).  What is kept
        from the earlier design is the *state*: the same materialized
        view the verification domain is built from, linearised — no
        parallel section construction, no drift.

        The plane is affine in :math:`(\Delta x, \Delta y)` about the
        pinned solver reference (full-polygon centroid), so negating the
        global triple :math:`(\varepsilon_0,\chi_x,\chi_y)` **is** the
        exact locked-in datum of the zone — no per-fiber sampling
        (Task-1 V1/R3).

        Parameters
        ----------
        section : GenericSection
            The base section; the service view is materialized from it.
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
        zone_names : sequence of str
        zone_ages : dict
            ``{zone index: age [days]}`` of every zone at this cast.
        ev : TimelineEvent
        tol, max_iter : float, int
        moduli : dict, optional
            Explicit per-slot moduli, forwarded to :meth:`_datum_view`.
        ledger : PrestressLedger, optional
            When given, the **convergence of this solve** is recorded
            against zone *zi*: relative residual, the tolerance it was
            judged against, the iteration count and the solver's own
            flag.  Recorded here, at the measurement, rather than
            reconstructed later from a second solve -- a reconstruction
            would be a second home for the number, and the number in
            question is the one Phase 12 found mis-stated in one of its
            two homes (``12_12`` §2).

        Returns
        -------
        tuple of float
            The datum triple :math:`(-\varepsilon_0, -\chi_x, -\chi_y)`.

        Raises
        ------
        ValueError
            If the pre-cast substrate does not converge under *demand*,
            or if a zone that exists at this instant has neither a
            rheological model nor an explicit modulus.

        See Also
        --------
        _datum_view : the one home of the service view.
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

        N, Mx, My = demand
        if abs(N) < 1e-30 and abs(Mx) < 1e-30 and abs(My) < 1e-30:
            # No demand, no deformation, no datum to lock in.  Returned
            # before the view is built so that a section with nothing to
            # resolve is never asked to resolve it.  The dump is told
            # *that this is why* there is no residual to report, rather
            # than being left to read a missing key.
            if ledger is not None:
                ledger.record_zone(
                    ev.index, zi,
                    datum_convergence={
                        "solved": False,
                        "cause": "no cumulative demand at this cast; "
                                 "the datum is identically zero and no "
                                 "equilibrium was solved.",
                    })
            return (0.0, 0.0, 0.0)

        solver, _modmap = self._datum_view(
            section, state, zone_names, zone_ages,
            casting_zone=zi, moduli=moduli)

        sol = solver.solve_equilibrium(N, Mx, My, tol=tol,
                                       max_iter=max_iter)

        conv = _datum_plane_residual(sol, demand,
                                     tolerances=self.tolerances)
        if ledger is not None:
            ledger.record_zone(ev.index, zi,
                               datum_convergence=dict(conv, solved=True))

        scale_N = max(abs(N), 1.0)
        scale_M = max(abs(Mx), abs(My), 1.0)
        if not conv["accepted"]:
            raise ValueError(
                f"construction_history[{ev.index}]: auto datum for zone "
                f"{zi} failed — the substrate did not reach equilibrium "
                f"on the service view "
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
                         ev, ledger=None, state=None, zone_names=None,
                         zone_ages=None, moduli=None,
                         grout_step=None,
                         preexisting=None) -> Dict[str, float]:
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

        Scope: a single-bulk post-tensioning.  Grouting **in stages** --
        stress, inject, stress the next against a section that now
        contains the first -- is modelled from Phase 22: the walk hands
        the driver a ``bonded_before`` schedule and a ``pending`` set, and
        the driver rebuilds its resisting view at each step.  What is
        still refused **here** is the **retrofit**: tendons already
        bonded before the timeline begins.  The driver models it from
        Phase 24 when the state that comes with it is declared
        (:class:`~gensec.solver.posttension.PreexistingState`); what
        this walk lacks is the declaration surface -- no timeline event
        and no YAML block states the effective stress, strain plane and
        two clocks of an existing member, so the case cannot be
        expressed here at all (D1b-3,
        ``6_8-WARNING_intra_sequence_bonded.md`` §2 case 2).

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
        state : SectionState, optional
            The construction state at the grout: which zones exist, with
            which datums.  Handed to the driver so the tendon is
            stressed against the section that exists, not against every
            zone the section will ever have.
        zone_names : sequence of str, optional
        zone_ages : dict, optional
            ``{zone index: age [days]}`` at the grout, for the service
            moduli the datum is drawn on.
        moduli : dict, optional
            Explicit per-slot overrides, forwarded to the resolution.
        ledger : PrestressLedger, optional
            When given, the **per-sub-step photograph** of the driver is
            recorded into it: for every jacking stage, the current
            stress and concrete strain at *every* tendon, plus the
            cumulative demand.  The driver already computes all of it
            (:class:`~gensec.solver.posttension.PosttensionStageResult`);
            before the ledger existed the whole result was discarded
            after a single number per tendon was read out of it, so the
            section's state at each stressing -- the thing an engineer
            actually wants to see -- was computed and thrown away.

        Returns
        -------
        dict
            ``{tendon_name: eps_init}`` for the tendons grouted here.

        Raises
        ------
        ValueError
            A tendon is grouted without a prior ``stress`` (no jacking
            stress to reconcile).
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

        from .posttension import (solve_posttension_sequence,
                                  grout_selfconsistent)

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

        # Every tendon of the section gets a declared role, because the
        # driver refuses to guess (``IntraSequenceUndeclared``).  A tendon
        # this walk has not stressed yet is PENDING: at this instant it is
        # neither a load nor a resistance element, and before Phase 22 the
        # driver read its absence from ``order`` as "already grouted" and
        # refused an ordinary staged-injection history at its FIRST grout.
        pending = sorted(set(range(nt)) - set(order))
        # ---- D1b-3: the roles a retrofit changes ---------------------
        # Three declared roles and no fourth (``order`` / ``pending`` /
        # ``already_bonded``).  A pre-existing cable belongs to the
        # third, and the driver refuses every way of saying it twice:
        #
        # * it must leave ``pending``, which means NOT YET stressed;
        # * its ``sigma_p0`` slot must be NaN, because that slot is the
        #   stress a jack of THIS sequence applies and an existing
        #   cable has none -- a finite number there is either a
        #   duplicate of the declared effective stress or a
        #   contradiction of it, and nothing would say which.
        #
        # None of this bypasses a refusal: it is the walk finally
        # saying which role each cable has, which is what the driver
        # has asked for since Phase 22.
        pre_set = set(preexisting.tendons) if preexisting is not None \
            else set()
        if pre_set:
            pending = [p for p in pending if p not in pre_set]
            for _j in pre_set:
                sigma_arr[_j] = float("nan")
        # ... and a tendon this walk has ALREADY grouted is bonded from the
        # step at which it was injected: the number of jacks that had been
        # pulled at that moment, which is exactly the ``bonded_before``
        # index of the closed form.  ``grout_step`` is carried by the walk
        # rather than re-derived here -- the walk is the only object that
        # knows the event order, and re-deriving it would be a second home
        # for one fact.
        # ``bonded_before`` schedules a grouting that happens INSIDE
        # this sequence, so it may only name cables the sequence
        # stresses.  A pre-existing cable is bonded before the
        # sequence starts and is declared through ``already_bonded``;
        # the walk keeps it in its own ``grout_step`` book because the
        # interleave guards read that book, but it does not travel to
        # the driver under a name that means something else.
        schedule = {li: int(step)
                    for li, step in (
                        (_tendon_local_index(section, t), st)
                        for t, st in (grout_step or {}).items())
                    if li not in pre_set}
        bN, bMx, bMy = base_demand
        modmap = None
        if state is not None:
            modmap = self._datum_moduli(section, state, zone_names,
                                        zone_ages, moduli=moduli)
        if preexisting is None:
            result = solve_posttension_sequence(
                section, sigma_p0=sigma_arr, order=order,
                pending=pending, bonded_before=schedule,
                base_N=bN, base_Mx=bMx, base_My=bMy,
                base_state=state, modmap=modmap)
        else:
            # On a retrofit the cumulative demand cannot be a
            # ``base_*``.  ``base_*`` is **debit-free by contract** --
            # it is the state before anything is anchored, so nothing
            # can be charged for it.  Here cables ARE anchored from
            # step 0, and a sollecitazione therefore costs them
            # stress.  ``applied_at={0: ...}`` is the parameter for
            # exactly that case, gated inside the driver on the bond
            # state; passing the same triple as ``base_*`` would
            # swallow the debit silently, which is the unconservative
            # direction for sigma_p at the critical section.
            #
            # The driver refuses the two together, and the refusal is
            # not worked around: ``base_*`` is left at zero and the
            # whole triple travels through the parameter that debits.
            applied = ({0: (bN, bMx, bMy)}
                       if (bN, bMx, bMy) != (0.0, 0.0, 0.0) else None)
            result = solve_posttension_sequence(
                section, sigma_p0=sigma_arr, order=order,
                pending=pending, bonded_before=schedule,
                already_bonded=preexisting, applied_at=applied,
                base_state=state, modmap=modmap)

        grout_local = [_tendon_local_index(section, t) for t in grout_now]

        # The demand the section carries *after* the grout: the base
        # demand plus whatever the still-ungrouted tendons keep applying
        # as loads.  The grouted ones are gone from the demand side --
        # they are internal now, and keeping them would be the §F
        # double-count.
        # The cables that are STILL LOADS: stressed here, not grouted
        # here, and not pre-existing.  A pre-existing cable is bonded
        # before the sequence begins and its force already lives
        # inside the declared concrete strain plane, so asking for its
        # demand-side action would count it twice -- which is what the
        # driver's refusal says, and why the set is named rather than
        # the refusal suppressed.
        eff_only = [j for j in range(nt)
                    if j not in grout_local and j not in pre_set]
        eff = {j: a for j, a in
               zip(eff_only, result.effective_actions(only=eff_only))}
        aN, aMx, aMy = bN, bMx, bMy
        for j, act in eff.items():
            aN += float(act.N)
            aMx += float(act.Mx)
            aMy += float(act.My)

        gres, by_local, n_it, res_rel = grout_selfconsistent(
            section, result, grout_local,
            demand_after=(aN, aMx, aMy),
            base_state=state, modmap=modmap,
            readback_rel=self.tolerances.grout_readback_rel)
        eps_map = {t: float(by_local[_tendon_local_index(section, t)])
                   for t in grout_now}

        if ledger is not None:
            self._record_driver(ledger, section, ev, result, eps_map,
                                stress_sigma)
            # The closure marker travels with the datum, attached to the
            # cell ``_record_driver`` has just opened -- after it, not
            # before, so that no state is written twice and the recorded
            # state stays exactly what it was.  Before Phase 13 the
            # iteration count and the residue were computed here and
            # dropped on the floor, so a dump had no way to tell a datum
            # that reached its fixed point from one left at the single
            # pass, which carries the O(n*rho) residue Phase 12 measured
            # at -7.08 MPa on the flagship.  Per tendon because the
            # datum is per tendon, even though the iteration is shared.
            closure = dict(gres.report["closure"])
            for t in grout_now:
                cell = ledger.at(ev.index, t)
                if cell is not None:
                    cell.closure = closure
        return eps_map

    @staticmethod
    def _record_driver(ledger, section, ev, result, eps_map,
                       stress_sigma) -> None:
        r"""
        Write the driver's per-stage photograph into *ledger*.

        One row per ``(stage, tendon)``: the section does not change
        during a stressing sequence -- the resistance view is tendon-free
        and constant (``6_8-WARNING_intra_sequence_bonded.md``) -- what
        changes is the strain plane, the per-tendon stress as later
        tendons take their elastic-shortening debit, and the accumulated
        demand.  Those three are the photograph.

        Parameters
        ----------
        ledger : PrestressLedger
        section : GenericSection
        ev : TimelineEvent
            The ``grout`` event whose reconciliation ran the driver.
        result : PosttensionSequenceResult
        eps_map : dict
            ``{tendon_name: eps_init}`` for the tendons grouted here.
        stress_sigma : dict
            ``{tendon_name: sigma_p0}`` as declared at jacking.
        """
        # Naming: the tree's ONE convention (``_tendon_local_index`` and
        # its four siblings), not a second one.  The previous form,
        # ``getattr(t, "name", f"T{i}")``, had two faults that the
        # ``getattr`` inventory of Phase 13 surfaced and a measurement
        # confirmed (finding F-N1): ``name`` is a dataclass field that
        # defaults to ``None``, so the fallback never fired and an
        # unnamed tendon was recorded under the literal key ``"None"`` --
        # two unnamed tendons collapsing into one cell -- and that key
        # matched nothing in ``_tendon_name_index``, so the substep rows
        # and the state rows of the same ledger were keyed differently
        # and a join across them dropped silently.
        names = [t.name or f"tendon_{k}"
                 for k, t in enumerate(section.tendons or [])]
        # Attribute access, not ``getattr`` with a default: every field
        # read below is mandatory on ``PosttensionStageResult``, so a
        # default could only ever mask the dataclass changing under us.
        # ``converged`` defaulting to ``True`` was the worst of them --
        # a silent default that *asserts* the thing it stands in for --
        # and this photograph is serialised into a versioned dump, where
        # a silent default is permanent (13_0 §11).
        for st in result.stages:
            si = int(st.stage_index)
            jacked = int(st.stressed_tendon)
            dem = tuple(st.cumulative_demand)
            for li, nm in enumerate(names):
                ledger.record_substep(
                    ev.index, si, nm,
                    sigma_p=float(st.sigma_p[li]),
                    eps_c=float(st.eps_c[li]),
                    jacked=bool(li == jacked),
                    demand_N=float(dem[0]),
                    demand_Mx=float(dem[1]),
                    demand_My=float(dem[2]),
                    converged=bool(st.converged))
        # The grouted tendons flip to the resistance side HERE, at their
        # post-loss effective stress and reconciled datum.
        for t, eps_init in eps_map.items():
            li = _tendon_local_index(section, t)
            ledger.record_state(
                ev.index, t, STATE_BONDED,
                sigma=float(result.sigma_p_after[li]),
                eps_init=float(eps_init))

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
            self._check_load_between_jacks(prefix, combo.get("name"))
            self._warn_defaulted_history(prefix, hist_factors,
                                         combo.get("name"), point)

            stages = self._emit_prefix_stages(
                prefix, resolution, section, demand_db,
                hist_factors, gamma_P, gamma_P_provider,
                combo_name=combo.get("name"), point=point)

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
                            hist_factors, gamma_P, provider,
                            combo_name=None, point=None) -> List[dict]:
        r"""
        Lower the prefix events to stages (one per event).

        Elements (rebars/tendons) contained in castable zones are
        deactivated at the first stage (``release: False`` — they are not
        yet present) and activated at their zone's ``cast`` event, so the
        emitted stage list satisfies the engine's containment invariant
        exactly (mirrors :meth:`resolve`).

        An **emission ledger** is built alongside the stages and left on
        ``resolution.emission_ledgers[(combo_name, point)]``.  It has two
        jobs.  First, the grouting stage cancels a tendon's demand-side
        action by **negating what was recorded** rather than rebuilding
        :math:`P = \sigma_{p0} A_p \gamma_P` a second time (F-C): two
        independent computations of one force agree only until somebody
        edits one of them.  Second, the single-side invariant

        .. math::

            A \oplus B = \text{true},
            \qquad
            A = \text{resistance-side},
            \quad
            B = \text{demand-side},

        is **asserted** over the finished ledger, unconditionally, at the
        end of the lowering -- not left as a comment and not hidden
        behind a debug switch.  The check is a walk over the ledger with
        no equilibrium in it, and what it catches (prestress counted
        twice, or lost) is not an error worth deferring.

        Parameters
        ----------
        prefix : sequence of TimelineEvent
        resolution : TimelineResolution
        section : GenericSection
        demand_db : dict
        hist_factors : dict
        gamma_P : float
        provider : callable or None
        combo_name : str, optional
            Combination name, for the ledger key and error messages.
        point : str, optional
            Anchor point name, likewise.

        Returns
        -------
        list of dict
            The lowered stages.

        Raises
        ------
        SingleSideViolation
            The XOR invariant failed at some ``(step, tendon)``.
        """
        zone_names = list(getattr(section, "zone_names", []) or [])
        led = PrestressLedger(tag=f"{combo_name or 'combo'}@{point or '-'}")
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
                    ev, resolution, section, gamma_P, ledger=led)
                _emit({"name": f"stress[{ev.index}]", "components": [],
                       "_prestress_actions": acts})

            elif ev.kind == "grout":
                ops, drop = self._grout_stage(
                    ev, resolution, section, gamma_P, ledger=led)
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

        # The invariant, asserted -- always, not optionally.
        led.assert_single_side(
            context=f"combination '{combo_name}' at '{point}'")
        resolution.emission_ledgers[(combo_name, point)] = led
        return stages

    def _stress_actions(self, ev, resolution, section, gamma_P,
                        ledger=None) -> List[PrestressAction]:
        r"""
        Emit demand-side prestress actions for a ``stress`` event.

        Only **post-tensioned** (parent cast) tendons produce a
        demand-side :class:`PrestressAction`; :math:`\gamma_P` is baked
        into the emitted triple (C1).  **Pre-tensioned** tendons are
        capacity-side (they enter the domain bonded with ``eps_pe`` when
        the zone casts) and contribute **no** demand increment, so
        :math:`\gamma_P` has nothing to scale for them (by design).

        Parameters
        ----------
        ev : TimelineEvent
        resolution : TimelineResolution
        section : GenericSection
        gamma_P : float
        ledger : PrestressLedger, optional
            Every emitted action is recorded into it **as emitted**, with
            :math:`\gamma_P` already baked in, so that the cancellation
            at grouting is an exact negation rather than a second
            reconstruction (F-C).

        Returns
        -------
        list of PrestressAction
        """
        acts: List[PrestressAction] = []
        for t in _event_tendon_refs(ev):
            if ledger is not None:
                ledger.record_state(
                    ev.index, t, STATE_STRESSED,
                    sigma=resolution.stress_sigma.get(t),
                    pre_post=resolution.pre_post.get(t))
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
            if ledger is not None:
                ledger.record_emission(ev.index, t, act)
        return acts

    def _grout_stage(self, ev, resolution, section, gamma_P, ledger=None):
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

        Parameters
        ----------
        ev : TimelineEvent
        resolution : TimelineResolution
        section : GenericSection
        gamma_P : float
        ledger : PrestressLedger, optional
            The emission ledger.  When present the drop is the
            **negation of what it recorded** (F-C).

        Returns
        -------
        tuple
            ``(section_ops, drop_actions)`` — ``section_ops`` is
            ``{"activate": [...], "eps_override": {...}}``;
            ``drop_actions`` is the list of cancelling
            :class:`PrestressAction` for the grouted tendons.

        Notes
        -----
        The pre-ledger implementation rebuilt
        :math:`P = \sigma_{p0} A_p \gamma_P` from
        ``resolution.stress_sigma`` and negated *that* -- a second,
        independent computation of a force the ``stress`` stage had
        already computed.  Two homes for one fact.  It also fired for
        **any** tendon carrying a jacking stress, while the emission
        fired only for post-tensioned ones, so a pre-tensioned tendon
        reaching a ``grout`` event received a negative action that never
        had a positive twin -- prestress conjured out of nothing, with
        nothing to catch it.  Negating the recorded emission closes both
        at once: nothing is cancelled that was not emitted, and the
        cancellation is exact by construction.
        """
        activate: List[int] = []
        eps_override: Dict[int, float] = {}
        drop: List[PrestressAction] = []
        for t in _event_tendon_refs(ev):
            ui = _tendon_union_index(section, t)
            activate.append(ui)
            if t in resolution.grout_eps_init:
                eps_override[ui] = resolution.grout_eps_init[t]
            if ledger is None:
                continue
            ledger.record_state(
                ev.index, t, STATE_BONDED,
                sigma=resolution.stress_sigma.get(t),
                pre_post=resolution.pre_post.get(t),
                eps_init=resolution.grout_eps_init.get(t))
            drop.extend(ledger.cancel(ev.index, t, _negate_action))
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
        Raise on a ``stress`` after a ``grout`` **of the same tendon**.

        Re-stressing a cable already bonded to the concrete around it is
        a different mechanical problem from jacking a free one: the jack
        no longer pulls against the anchorages alone, and the tendon is
        no longer a demand-side load.  GenSec does not model it.
        Fail-loud.

        The scope is exactly the same tendon, and since Phase 22 that is
        also all it should be.  This docstring used to claim the check
        covered "any tendon sharing the just-changed bonded section" --
        it never did (the code has always intersected the event's own
        tendon refs only), and it no longer should: stressing ANOTHER
        tendon against an already-injected one is the staged-injection
        case, which is now modelled through ``bonded_before``.
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
    def _check_load_between_jacks(prefix, combo_name) -> None:
        r"""
        Raise on a ``load`` interposed between two jacking events while a
        tendon is stressed but not yet grouted.

        The scope boundary of the sectional post-tensioning driver, made
        loud.  The reasoning belongs to the driver and is written once,
        in full, in
        :func:`~gensec.solver.posttension.solve_posttension_sequence`
        and in
        :class:`~gensec.solver.posttension.LoadBetweenJacksNotModelled`:
        an ungrouted tendon has no strain compatibility with the section,
        so its force change follows a member-level integral that a
        sectional model cannot evaluate, and the local value it *can*
        evaluate is unconservative for :math:`\sigma_p` at the critical
        section.

        The condition is deliberately narrow — three clauses, all
        required:

        1.  the event is a ``load``;
        2.  it falls **between** two ``stress`` events, i.e. at least one
            jacking event precedes it and at least one follows it;
        3.  at that instant at least one tendon is **stressed and not yet
            grouted**.

        Each exclusion is a case that *is* modelled, not an oversight:

        * a load **before the first jack** is the meaning of
          ``base_N``/``base_Mx``/``base_My`` — the state at the start of
          the sequence — **as long as nothing is anchored at that
          instant**, which is the case for every history this walk can
          compile.  It stops being true on a *retrofit*, where tendons
          grouted before the sequence began are anchored from the first
          step and a sollecitazione therefore costs them stress;
          ``base_*`` is debit-free by contract and would swallow that
          silently.  The driver's ``applied_at={step: (N, Mx, My)}``
          is the parameter for it, gated on the same bond-state
          condition as clause 3.  Since Phase 34 a ``retrofit`` event
          (D1b-3) does reach this walk, and the stressing compiler
          routes the cumulative pre-jack demand of a retrofitted member
          through ``applied_at={0: ...}`` with ``base_*`` left at zero,
          so the debit is charged and not swallowed;
        * a load **after the last jack** is not interposed in a sequence;
          there is no subsequent jacking event whose debit it would
          contaminate;
        * a load acting when every stressed tendon is **already
          grouted** is compatible in strain with the section and is
          resolved by the ordinary bonded path.

        Parameters
        ----------
        prefix : sequence of TimelineEvent
            The event prefix being compiled, in physical order.
        combo_name : str or None
            Combination name, for the message.

        Raises
        ------
        gensec.solver.posttension.LoadBetweenJacksNotModelled
            Naming the tendons, the load event and the two jacking
            events it falls between.

        See Also
        --------
        ConstructionTimeline._check_interleave : the sibling guard, for a
            ``stress`` after a ``grout`` of the same tendon.
        """
        stress_idx = [k for k, ev in enumerate(prefix)
                      if ev.kind == "stress"]
        if len(stress_idx) < 2:
            # Fewer than two jacks: nothing can fall *between* two.
            return

        first_stress, last_stress = stress_idx[0], stress_idx[-1]

        stressed: set = set()
        grouted: set = set()
        for k, ev in enumerate(prefix):
            if ev.kind == "stress":
                stressed.update(_event_tendon_refs(ev))
                continue
            if ev.kind == "grout":
                grouted.update(_event_tendon_refs(ev))
                continue
            if ev.kind != "load":
                continue
            if not (first_stress < k < last_stress):
                continue
            open_tendons = sorted(stressed - grouted)
            if not open_tendons:
                continue
            prev_j = max(j for j in stress_idx if j < k)
            next_j = min(j for j in stress_idx if j > k)
            from .posttension import LoadBetweenJacksNotModelled
            raise LoadBetweenJacksNotModelled(
                tendons=open_tendons,
                load_index=prefix[k].index,
                load_demand=prefix[k].payload.get("demand", "?"),
                prev_stress_index=prefix[prev_j].index,
                next_stress_index=prefix[next_j].index,
                combo_name=combo_name,
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
                 "losses_ops", "losses_trace", "ledger",
                 "emission_ledgers", "skipped_losses")

    def __init__(self, datums, explicit_datums, pre_post, grouted,
                 stressed, grout_eps_init=None, stress_sigma=None,
                 losses_ops=None, losses_trace=None, ledger=None,
                 skipped_losses=()):
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
        #: Indices of the ``interval`` events whose declared ``losses``
        #: were **not** integrated, because the model turned rheology
        #: off.  Empty in every other case.  A report that does not say
        #: this is a report that misdescribes its own result.
        self.skipped_losses = tuple(skipped_losses)
        self.stress_sigma = stress_sigma or {}
        #: :class:`~gensec.solver.prestress_ledger.PrestressLedger` -- the
        #: per-tendon, per-step record of the resolution walk.  Public
        #: and queryable: ``resolution.ledger.state_of("P1", upto=4)``,
        #: ``resolution.ledger.at(6, "P1").sigma``.  Timeline physics,
        #: computed once here and never re-derived.
        self.ledger = ledger if ledger is not None else PrestressLedger()
        #: ``{zone_index: (eps0, chi_x, chi_y)}`` -- the SERVICE casting
        #: datum, twin of ``datums`` (which is the ULS one).  Previously
        #: computed in the walk and dropped on the floor; the datum
        #: fingerprint needs to read it (F4/F5).
        #: ``{(combination, point): PrestressLedger}`` -- the compile-time
        #: emission ledgers.  Separate from :attr:`ledger` because
        #: :math:`\gamma_P` belongs to the combination, not the timeline,
        #: so the actions actually emitted differ per combination.
        self.emission_ledgers = {}


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


def _negate_action(act: PrestressAction) -> PrestressAction:
    r"""
    The exact opposite of *act*, label and origin retagged.

    The cancelling twin of a demand-side prestress action.  Built by
    negation of the emitted triple, never by recomputing the force, so
    :math:`\gamma_P` and every rounding decision baked into the emission
    are cancelled exactly rather than approximately.

    Parameters
    ----------
    act : PrestressAction

    Returns
    -------
    PrestressAction
        ``(-N, -Mx, -My)``.
    """
    label = getattr(act, "label", "") or ""
    return PrestressAction(
        -act.N, -act.Mx, -act.My,
        label=label.replace("stress:", "grout_drop:", 1)
        if label.startswith("stress:") else f"grout_drop:{label}",
        origin="timeline_grout_drop")


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


def _normalise_rheology_switch(value):
    r"""
    Normalise the ``rheology`` switch to ``'declared'``/``'on'``/``'off'``.

    Accepts ``None`` (declared), the booleans, and the strings
    ``'declared'``, ``'auto'``, ``'on'``, ``'off'``, ``'true'``,
    ``'false'``, ``'yes'``, ``'no'``, in any case.

    Parameters
    ----------
    value : None or bool or str

    Returns
    -------
    str

    Raises
    ------
    ValueError
        On anything else.  A misspelt switch that fell through to the
        default would silently re-enable a rheology the model asked to
        turn off -- the class of quiet mismodel this project refuses.
    """
    if value is None:
        return "declared"
    if isinstance(value, bool):
        return "on" if value else "off"
    text = str(value).strip().lower()
    if text in ("declared", "auto"):
        return "declared"
    if text in ("on", "true", "yes", "1"):
        return "on"
    if text in ("off", "false", "no", "0", "none"):
        return "off"
    raise ValueError(
        f"'rheology' must be one of 'declared' (the default: follow the "
        f"model's own declarations), 'on' (refuse any zone that declares "
        f"no rheological model) or 'off' (no creep, shrinkage or "
        f"relaxation; every datum at the code's conventional modulus), "
        f"got {value!r}."
    )


def _datum_plane_accepted(sol, demand, rel_tol=None,
                          tolerances=None) -> bool:
    r"""
    Whether a datum solve reproduced its demand.

    **One home for one acceptance rule.**  The solver's own
    ``converged`` flag is judged on an *absolute* force/moment
    residual, so on a section carrying moments of order
    :math:`10^{8}` N·mm the flag asks for a relative precision of
    :math:`10^{-14}`, which ``float64`` cannot deliver.  It therefore
    reports failure on planes that are exact.  Measured on the
    flagship's topping datum:

    .. math::

        N,\,M_x \;\text{exact};\qquad
        M_y = -2.79\times10^{-9}\ \mathrm{N\,mm}
        \;\text{against a target of }0,

    a relative residual of :math:`1.3\times10^{-17}` on a scale of
    :math:`2.2\times10^{8}` -- reported as *not converged* after 100
    iterations.

    The rule was already corrected once, inside
    :meth:`ConstructionTimeline._auto_datum`, and **not** in its twin
    ``_service_datum``; the twin then stopped the flagship the moment
    the datum retarget changed its stage prefix.  One criterion living
    in two places, only one of them repaired, is F7's lesson applied
    to convergence instead of to datums.  It lives here now, so a
    third datum site inherits it rather than re-deriving it.

    A genuine residual failure still raises at the call site: a
    substrate that truly cannot carry its own construction loads is a
    real finding, not a warning.

    Parameters
    ----------
    sol : dict
        A ``solve_equilibrium`` result: ``converged``, ``N``, ``Mx``,
        ``My``.
    demand : tuple of float
        The target ``(N, Mx, My)`` [N, N·mm].
    rel_tol : float, optional
        Overrides both tolerances with one number.  ``None`` (the
        default) reads them from *tolerances*.
    tolerances : Tolerances, optional
        The resolved tolerance set; the shipped defaults if absent.

    Returns
    -------
    bool

    Notes
    -----
    Scope, stated because it is narrower than the defect: 27 sites in
    ``src/`` read ``sol["converged"]`` and only the datum sites are
    covered here.  The others are registered, not fixed -- moving them
    would move numbers, and this phase moves numbers only on purpose.
    """
    return _datum_plane_residual(sol, demand, rel_tol=rel_tol,
                                 tolerances=tolerances)["accepted"]


def _datum_plane_residual(sol, demand, rel_tol=None,
                          tolerances=None) -> dict:
    r"""
    The residual of a datum solve, **and** what it was judged against.

    Companion of :func:`_datum_plane_accepted`, which derives its
    boolean from this dict rather than recomputing the arithmetic:
    two computations of one acceptance would be two homes for one
    fact, and the fact here is precisely the one Phase 12 found
    mis-stated in one of its two homes.

    It exists because the machine-readable dump must record the
    residual and the tolerance, never the solver's ``converged``
    flag.  That flag is judged on an *absolute* residual, so it is
    structurally meaningless above :math:`10^{8}` N·mm and has already
    caused one defect (``12_12`` §2).  Serialising it would freeze the
    trap into a versioned format, where it becomes permanent.

    .. math::

        r_N = \frac{|N_{\text{sol}} - N|}{\max(|N|,\,1)},
        \qquad
        r_M = \max_{c \in \{x, y\}}
              \frac{|M_{c,\text{sol}} - M_c|}
                   {\max(|M_x|,\,|M_y|,\,1)}

    Parameters
    ----------
    sol : dict
        A ``solve_equilibrium`` result.
    demand : tuple of float
        The target ``(N, Mx, My)`` [N, N·mm].
    rel_tol : float, optional
        Overrides both tolerances with one number.
    tolerances : Tolerances, optional
        The resolved tolerance set; the shipped defaults if absent.

    Returns
    -------
    dict
        ``residual_force_rel``, ``residual_moment_rel``,
        ``tolerance_force_rel``, ``tolerance_moment_rel``,
        ``iterations``, ``solver_flag`` and the derived ``accepted``.

    Notes
    -----
    ``accepted`` is redundant on purpose: it is

    .. math::

        \texttt{solver\_flag} \;\vee\;
        \bigl(r_N < \text{tol}_N \wedge r_M < \text{tol}_M\bigr),

    so a reader of the dump can re-derive it from the numbers beside
    it, **and** can see which of the two disjuncts carried it.  A
    boolean recomputable from its ingredients is a checkable
    assertion; an opaque one is a request for trust.

    ``solver_flag`` is carried rather than dropped precisely because
    it is the untrustworthy term: it is what the engine's rule reads
    first, so a dump that omitted it would describe an acceptance
    rule GenSec does not use.  Recording the flag as *one disjunct
    beside its residual* is the opposite of recording it alone.
    """
    if tolerances is None:
        from ..tolerances import Tolerances
        tolerances = Tolerances()
    tol_N = rel_tol if rel_tol is not None else tolerances.equilibrium_force_rel
    tol_M = rel_tol if rel_tol is not None else tolerances.equilibrium_moment_rel
    N, Mx, My = (float(v) for v in demand)
    scale_N = max(abs(N), 1.0)
    scale_M = max(abs(Mx), abs(My), 1.0)
    r_N = abs(float(sol["N"]) - N) / scale_N
    r_M = max(abs(float(sol["Mx"]) - Mx) / scale_M,
              abs(float(sol["My"]) - My) / scale_M)
    flag = bool(sol.get("converged"))
    return {
        "residual_force_rel": r_N,
        "residual_moment_rel": r_M,
        "tolerance_force_rel": float(tol_N),
        "tolerance_moment_rel": float(tol_M),
        "iterations": (None if sol.get("iterations") is None
                       else int(sol["iterations"])),
        "solver_flag": flag,
        "accepted": bool(flag or (r_N < tol_N and r_M < tol_M)),
    }


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
# GENSEC-COMPLETE phase=34 utc=2026-09-06T10:33:16Z sha256=c4dcaf6457de64aede6cbe9d737abb62b8f56218179182426c8ca73d60ad7e69 body_lines=3615
