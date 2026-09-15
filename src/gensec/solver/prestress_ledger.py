r"""
The per-tendon, per-step prestress ledger.

Purpose
-------
A construction timeline moves a tendon through three regimes, and the
sectional model treats each one differently:

============  ====================================================
regime        how the section sees the tendon
============  ====================================================
``stressed``  a **demand-side load**: the jack pushes on the
              concrete, the strand is not part of the resistance
``grouted``   the grout has been injected at this step — the
              transition itself
``bonded``    a **resistance element**: it strains with the
              section from its reconciled datum
              :math:`\varepsilon_{\text{init}}`
============  ====================================================

The physical statement that governs the transition is the *single-side
invariant*: at any step, a tendon whose force acts on the section is on
**exactly one** of the two sides — never both (the prestress would be
counted twice) and never neither (the prestress would vanish).  Written
as a predicate over the two propositions

.. math::

    A \;=\; \text{"the tendon is live in the resistance state"},
    \qquad
    B \;=\; \text{"the tendon has a live demand-side action"},

the invariant is the exclusive disjunction

.. math::

    A \oplus B \;=\; \text{true}
    \qquad
    \text{for every step at or after the tendon's stressing.}

This module records the walk in enough detail to *assert* that predicate
rather than comment it, and to answer the engineer's question — "what did
the section look like at each stressing?" — from stored data instead of
from a reconstruction.

Two ledgers, one schema
-----------------------
The same key schema carries two different bodies of data, produced at two
different times:

**Resolution ledger** (:meth:`ConstructionTimeline.resolve`).  Timeline
physics, computed once: per-tendon state and stress at every event, the
per-sub-step photograph of the post-tensioning driver, the casting
datums.  Frozen onto :class:`~gensec.solver.timeline.TimelineResolution`
and never recomputed.  This is what
:mod:`tools.datum_fingerprint` walks.

**Emission ledger** (:meth:`ConstructionTimeline._emit_prefix_stages`).
Compile-time, one per ``(combination, point)`` pair, because
:math:`\gamma_P` is a property of the combination and not of the
timeline.  It records the :class:`~gensec.solver.prestress.PrestressAction`
objects actually emitted, so that the cancellation at grouting is the
**negation of what was recorded** rather than a second, independent
reconstruction of :math:`P = \sigma_{p0} A_p \gamma_P` (finding F-C).

Why negation rather than reconstruction
---------------------------------------
The pre-ledger ``_grout_stage`` rebuilt the force from
``resolution.stress_sigma`` and negated *that*.  Two independent
computations of one quantity is two homes for one fact: they agree only
as long as nobody edits one of them, and they disagree silently.  Worse,
the reconstruction fired for **any** tendon carrying a jacking stress,
while the emission fired only for post-tensioned ones — so a
pre-tensioned tendon reaching a ``grout`` event received a negative
action that had never had a positive twin.  Cancelling by negation of the
recorded emission closes both: nothing is cancelled that was not emitted,
and the cancellation is exact by construction rather than by agreement.

Notes
-----
The ledger stores **physical values**, never hashes.  Its
:meth:`PrestressLedger.items` projection emits ``key = value`` pairs in
the format :mod:`tools.numeric_fingerprint` consumes, so the existing
``compare`` command works on it without a second implementation.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Tuple

__all__ = [
    "STATE_STRESSED",
    "STATE_GROUTED",
    "STATE_BONDED",
    "SingleSideViolation",
    "TendonRecord",
    "PrestressLedger",
]


#: The tendon has been jacked; it acts on the section as a demand-side
#: load and is absent from the resistance state.
STATE_STRESSED = "stressed"

#: The grout has been injected at this very step — the transition.
STATE_GROUTED = "grouted"

#: The tendon is live in the resistance state at its reconciled datum.
STATE_BONDED = "bonded"


class SingleSideViolation(AssertionError):
    r"""
    The single-side (XOR) invariant failed at some ``(step, tendon)``.

    Raised by :meth:`PrestressLedger.assert_single_side`.  This is an
    :class:`AssertionError` subclass on purpose: it reports a broken
    internal contract, not a rejected user input.  A user-facing scope
    limit raises :class:`NotImplementedError` and a bad model raises
    :class:`ValueError`; neither is what a double-counted prestress is.
    """


class TendonRecord:
    r"""
    One ``(step, tendon)`` cell of the ledger.

    Parameters
    ----------
    event_index : int
        Index of the timeline event this cell belongs to.
    tendon : str
        Tendon name.
    state : str
        One of :data:`STATE_STRESSED`, :data:`STATE_GROUTED`,
        :data:`STATE_BONDED`.
    sigma : float or None
        Current tendon stress :math:`\sigma_p` [MPa] as far as the walk
        knows it at this step.  ``None`` when the walk has no number
        (e.g. a ``stress`` event that declares no ``sigma_p0``).
    pre_post : str or None
        ``'pre'`` or ``'post'`` — the derived classification.
    emitted : list of PrestressAction
        Demand-side actions emitted *at this step* for this tendon.
    cancelled_at : list of int or None
        Parallel to *emitted*: the event index at which each emission
        was cancelled, or ``None`` while it is still live.  A *dated*
        cancellation, not a terminal flag: an action emitted at step
        :math:`i` and cancelled at step :math:`j` is live on
        :math:`[i, j)` and absent from :math:`j` onward.  A terminal
        flag would make the emission read as never-live once cancelled,
        and the single-side invariant would then report the tendon as
        having vanished at its own stressing step.
    eps_init : float or None
        The reconciled locked-in strain, present only on the grouting
        step.
    closure : dict or None
        How the locked-in strain was closed, present only on the
        grouting step: ``{"kind": "selfconsistent", "iterations": n,
        "residual_rel": r, "tolerance_rel": tol}`` or
        ``{"kind": "single_pass"}``.  It is what lets a consumer tell a
        datum that reached its fixed point from one left at
        :func:`~gensec.solver.posttension.grout`'s single pass, which
        carries the :math:`O(n\rho)` residue.  The machine-readable
        dump **refuses** the latter (``13_0`` §8.1).
    """

    __slots__ = ("event_index", "tendon", "state", "sigma", "pre_post",
                 "emitted", "cancelled_at", "eps_init", "closure")

    def __init__(self, event_index: int, tendon: str, state: str,
                 sigma: Optional[float] = None,
                 pre_post: Optional[str] = None,
                 eps_init: Optional[float] = None):
        self.event_index = int(event_index)
        self.tendon = str(tendon)
        self.state = str(state)
        self.sigma = None if sigma is None else float(sigma)
        self.pre_post = pre_post
        self.emitted: List[object] = []
        self.cancelled_at: List[Optional[int]] = []
        self.eps_init = None if eps_init is None else float(eps_init)
        self.closure: Optional[dict] = None

    def __repr__(self) -> str:                       # pragma: no cover
        return (f"<TendonRecord step={self.event_index} "
                f"tendon={self.tendon!r} state={self.state} "
                f"sigma={self.sigma} emitted={len(self.emitted)}>")

    def live_count(self, upto: Optional[int] = None) -> int:
        r"""
        Emissions of this cell live at *upto*.

        Parameters
        ----------
        upto : int, optional
            Event index.  ``None`` means "at the end of the walk".

        Returns
        -------
        int
        """
        n = 0
        for c in self.cancelled_at:
            if c is None:
                n += 1
            elif upto is not None and int(c) > int(upto):
                n += 1
        return n


class PrestressLedger:
    r"""
    Per-step, per-tendon record of the prestress bookkeeping.

    The ledger is append-only during a walk and read-only afterwards.
    It is the single place that knows, for every step, which tendons
    were on the demand side and which on the resistance side — the
    information ``_check_staged`` destroys when it accumulates
    ``_prestress_actions`` into three scalars.

    Parameters
    ----------
    tag : str, optional
        Short label prefixed to every key in :meth:`items` — typically
        the example stem, or ``combination@point`` for an emission
        ledger.

    Attributes
    ----------
    records : dict
        ``{(event_index, tendon): TendonRecord}``, insertion-ordered.
    substeps : dict
        ``{(event_index, sub_index, tendon): dict}`` — the photograph of
        the post-tensioning driver at each jacking sub-step.
    zones : dict
        ``{(event_index, zone_index): dict}`` — casting datums.

    Examples
    --------
    >>> led = PrestressLedger(tag="demo")
    >>> _ = led.record_state(1, "P1", STATE_STRESSED, sigma=1300.0,
    ...                      pre_post="post")
    >>> led.state_of("P1")
    'stressed'
    >>> _ = led.record_state(2, "P1", STATE_BONDED, sigma=1290.0,
    ...                      pre_post="post")
    >>> led.state_of("P1")
    'bonded'
    >>> led.state_of("P1", upto=1)
    'stressed'
    """

    __slots__ = ("tag", "records", "substeps", "zones", "step_facts",
                 "_order")

    def __init__(self, tag: str = ""):
        self.tag = str(tag)
        self.records: Dict[Tuple[int, str], TendonRecord] = {}
        self.substeps: Dict[Tuple[int, int, str], dict] = {}
        self.zones: Dict[Tuple[int, int], dict] = {}
        #: ``{event_index: dict}`` -- the per-step facts of the walk:
        #: the clock, the cumulative demand, which zones are active,
        #: and each zone's declared and derived age.  Recorded **by the
        #: walk**, at the instant they are true, so that a consumer
        #: reads them instead of re-deriving them from the event list.
        #: A second derivation would be a second home for a fact whose
        #: whole value is that it says what actually happened.
        self.step_facts: Dict[int, dict] = {}
        self._order: List[Tuple[int, str]] = []

    # -- recording -----------------------------------------------------

    def record_state(self, event_index: int, tendon: str, state: str, *,
                     sigma: Optional[float] = None,
                     pre_post: Optional[str] = None,
                     eps_init: Optional[float] = None) -> TendonRecord:
        r"""
        Open (or update) the cell for *tendon* at *event_index*.

        Parameters
        ----------
        event_index : int
        tendon : str
        state : str
            :data:`STATE_STRESSED`, :data:`STATE_GROUTED` or
            :data:`STATE_BONDED`.
        sigma : float, optional
            Current :math:`\sigma_p` [MPa].
        pre_post : str, optional
        eps_init : float, optional

        Returns
        -------
        TendonRecord
            The cell, so the caller can attach emissions to it.
        """
        key = (int(event_index), str(tendon))
        rec = self.records.get(key)
        if rec is None:
            rec = TendonRecord(event_index, tendon, state, sigma=sigma,
                               pre_post=pre_post, eps_init=eps_init)
            self.records[key] = rec
            self._order.append(key)
        else:
            rec.state = str(state)
            if sigma is not None:
                rec.sigma = float(sigma)
            if pre_post is not None:
                rec.pre_post = pre_post
            if eps_init is not None:
                rec.eps_init = float(eps_init)
        return rec

    def record_emission(self, event_index: int, tendon: str,
                        action) -> None:
        r"""
        Register a demand-side action emitted for *tendon*.

        The cell must already exist (:meth:`record_state`); an emission
        without a state is a bookkeeping error, not a tolerable gap.

        Parameters
        ----------
        event_index : int
        tendon : str
        action : PrestressAction
            Recorded **as emitted**, i.e. with :math:`\gamma_P` already
            baked in, so the later cancellation is exact.

        Raises
        ------
        KeyError
            No cell open for ``(event_index, tendon)``.
        """
        rec = self.records[(int(event_index), str(tendon))]
        rec.emitted.append(action)
        rec.cancelled_at.append(None)

    def record_substep(self, event_index: int, sub_index: int, tendon: str,
                       **values) -> None:
        r"""
        Record one jacking sub-step of the post-tensioning driver.

        This is the "photograph at each stressing": the driver already
        computes, for every stage, the stress and concrete strain at
        *every* tendon and the cumulative demand — and the pre-ledger
        ``_reconcile_grout`` discarded the whole result after extracting
        a single number per tendon.

        Parameters
        ----------
        event_index : int
            Index of the ``grout`` event whose reconciliation ran the
            driver.
        sub_index : int
            Driver stage index (one per jacked tendon, in stressing
            order).
        tendon : str
            The tendon this row describes — **not** necessarily the one
            being jacked at *sub_index*.
        **values
            Scalars to store, e.g. ``sigma_p``, ``eps_c``,
            ``stressed_tendon``, ``demand_N``.
        """
        self.substeps[(int(event_index), int(sub_index), str(tendon))] = \
            dict(values)

    def record_zone(self, event_index: int, zone_index: int,
                    **values) -> None:
        r"""
        Record facts about a zone at an event, **merging** into the cell.

        Merging and not replacing, because the cell has more than one
        writer: the datum solve records its convergence from inside
        :meth:`~gensec.solver.timeline.ConstructionTimeline._auto_datum`,
        and the walk records the resulting triple a few lines later.
        A replacing write would silently drop whichever of the two ran
        first -- and the one that runs first is the convergence, which
        is exactly the datum nobody would notice missing.

        Parameters
        ----------
        event_index : int
        zone_index : int
        **values
            Typically ``auto_datum`` (an ``(eps0, chi_x, chi_y)``
            triple), ``origin`` and ``datum_convergence``.

        Examples
        --------
        >>> led = PrestressLedger()
        >>> led.record_zone(1, 0, origin="auto")
        >>> led.record_zone(1, 0, auto_datum=(0.0, 0.0, 0.0))
        >>> sorted(led.zones[(1, 0)])
        ['auto_datum', 'origin']
        """
        cell = self.zones.setdefault((int(event_index), int(zone_index)), {})
        cell.update(values)

    def record_step(self, event_index: int, **values) -> None:
        r"""
        Record the walk's facts about one event, **merging** into the cell.

        These are the quantities that a machine-readable dump needs and
        that only the walk knows without re-deriving: the clock at the
        event, the cumulative characteristic demand, which zones are
        active, and each zone's declared and derived age.

        The derived age in particular must come from here.  It is the
        sum of the ``interval`` lengths the walk has crossed, and
        recomputing it downstream would mean re-implementing the walk's
        clock — two homes for a fact whose only content is *what the
        walk did*.

        Parameters
        ----------
        event_index : int
        **values
            Typically ``kind``, ``t_days_start``, ``t_days``,
            ``duration_days``, ``zones_active``, ``demand_cumulative``,
            ``loads_applied`` and ``rheology``.
        """
        cell = self.step_facts.setdefault(int(event_index), {})
        cell.update(values)

    # -- the F-C cancellation ------------------------------------------

    def cancel(self, event_index: int, tendon: str, factory) -> List[object]:
        r"""
        Cancel every live emission of *tendon* by **negating** it.

        This is finding F-C: the drop at grouting is the negation of what
        the ledger recorded, not a second reconstruction of the jacking
        force.  Nothing is cancelled that was not emitted — in
        particular a pre-tensioned tendon, which never emits a
        demand-side action, receives no spurious negative drop.

        Parameters
        ----------
        event_index : int
            The ``grout`` event at which the cancellation happens.
        tendon : str
        factory : callable
            ``factory(action) -> action`` returning the negated twin.
            Injected rather than imported so this module stays free of a
            dependency on :mod:`gensec.solver.prestress`.

        Returns
        -------
        list
            The negated actions, in emission order.  Empty when the
            tendon had nothing live.
        """
        out: List[object] = []
        for key in self._order:
            ei, tn = key
            if tn != str(tendon) or ei > int(event_index):
                continue
            rec = self.records[key]
            for i, act in enumerate(rec.emitted):
                if rec.cancelled_at[i] is None:
                    rec.cancelled_at[i] = int(event_index)
                    out.append(factory(act))
        return out

    # -- querying ------------------------------------------------------

    def tendons(self) -> List[str]:
        r"""Every tendon name the ledger has seen, sorted."""
        return sorted({t for _, t in self.records})

    def steps(self) -> List[int]:
        r"""Every event index the ledger has seen, sorted."""
        return sorted({i for i, _ in self.records})

    def at(self, event_index: int, tendon: str) -> Optional[TendonRecord]:
        r"""
        The cell for ``(event_index, tendon)``, or ``None``.

        Parameters
        ----------
        event_index : int
        tendon : str

        Returns
        -------
        TendonRecord or None
        """
        return self.records.get((int(event_index), str(tendon)))

    def state_of(self, tendon: str,
                 upto: Optional[int] = None) -> Optional[str]:
        r"""
        The tendon's state as of *upto* (default: the whole walk).

        Parameters
        ----------
        tendon : str
        upto : int, optional
            Event index; the latest cell at or before it wins.

        Returns
        -------
        str or None
            ``None`` when the tendon has no cell in range.
        """
        best = None
        for ei, tn in self._order:
            if tn != str(tendon):
                continue
            if upto is not None and ei > int(upto):
                continue
            best = self.records[(ei, tn)].state
        return best

    def live_actions(self, tendon: str,
                     upto: Optional[int] = None) -> int:
        r"""
        Count of uncancelled emissions for *tendon* up to *upto*.

        Parameters
        ----------
        tendon : str
        upto : int, optional

        Returns
        -------
        int
        """
        n = 0
        for ei, tn in self._order:
            if tn != str(tendon):
                continue
            if upto is not None and ei > int(upto):
                continue
            n += self.records[(ei, tn)].live_count(upto)
        return n

    # -- the invariant -------------------------------------------------

    def assert_single_side(self, *, bonded_at=None, context: str = "") -> None:
        r"""
        Assert :math:`A \oplus B` at every step, for every tendon.

        For a **post-tensioned** tendon, from its stressing onward
        exactly one of the two sides must hold: a live demand-side
        action, or presence in the resistance state.  For a
        **pre-tensioned** tendon the demand side is never used at all —
        it enters the domain bonded when its parent zone casts — so the
        assertion is that it emitted nothing, ever.

        This runs unconditionally, not behind a debug flag.  The
        invariant is cheap (a walk over the ledger, no equilibrium) and
        the failure it catches — prestress counted twice, or lost — is
        not the kind of error that should wait for someone to switch on
        a guard.

        Parameters
        ----------
        bonded_at : callable, optional
            ``bonded_at(event_index, tendon) -> bool``, the resistance
            side of the predicate.  When ``None`` the ledger's own
            recorded state is used
            (:data:`STATE_BONDED` or :data:`STATE_GROUTED`).
        context : str, optional
            Prefixed to the failure message.

        Raises
        ------
        SingleSideViolation
            Both sides true, or neither, at some ``(step, tendon)``.
        """
        pre = f"{context}: " if context else ""
        steps = self.steps()
        for tendon in self.tendons():
            first = min(ei for ei, tn in self.records if tn == tendon)
            kind = None
            for ei, tn in self._order:
                if tn == tendon and self.records[(ei, tn)].pre_post:
                    kind = self.records[(ei, tn)].pre_post
                    break

            if kind == "pre":
                total = sum(len(self.records[(ei, tn)].emitted)
                            for ei, tn in self.records if tn == tendon)
                if total:
                    raise SingleSideViolation(
                        f"{pre}pre-tensioned tendon '{tendon}' emitted "
                        f"{total} demand-side action(s).  A pre-tensioned "
                        f"tendon is capacity-side from the cast of its "
                        f"parent zone and never loads the section."
                    )
                continue

            for ei in steps:
                if ei < first:
                    continue
                demand_live = self.live_actions(tendon, upto=ei) > 0
                if bonded_at is not None:
                    resist_live = bool(bonded_at(ei, tendon))
                else:
                    st = self.state_of(tendon, upto=ei)
                    resist_live = st in (STATE_BONDED, STATE_GROUTED)
                if demand_live and resist_live:
                    raise SingleSideViolation(
                        f"{pre}tendon '{tendon}' is BOTH a live "
                        f"demand-side action and a resistance element at "
                        f"step {ei}.  The prestress is counted twice."
                    )
                if not demand_live and not resist_live:
                    raise SingleSideViolation(
                        f"{pre}tendon '{tendon}' is NEITHER a live "
                        f"demand-side action nor a resistance element at "
                        f"step {ei}, having been stressed at step "
                        f"{first}.  The prestress has vanished."
                    )

    # -- projection ----------------------------------------------------

    def items(self) -> Iterator[Tuple[str, object]]:
        r"""
        Yield ``(key, value)`` pairs in ``numeric_fingerprint`` format.

        Keys are hierarchical and deterministic, so two ledgers taken
        before and after a change can be diffed key by key by the
        existing ``compare`` command.

        Yields
        ------
        tuple
            ``(key, value)`` with *value* a float, int, str or bool.
        """
        t = f"{self.tag}." if self.tag else ""
        for ei, tn in self._order:
            rec = self.records[(ei, tn)]
            base = f"{t}step[{ei:03d}].tendon[{tn}]"
            yield (f"{base}.state", rec.state)
            if rec.pre_post is not None:
                yield (f"{base}.pre_post", rec.pre_post)
            if rec.sigma is not None:
                yield (f"{base}.sigma", rec.sigma)
            if rec.eps_init is not None:
                yield (f"{base}.grout_eps_init", rec.eps_init)
            if rec.closure is not None:
                for sub in sorted(rec.closure):
                    yield (f"{base}.closure.{sub}", rec.closure[sub])
            for i, act in enumerate(rec.emitted):
                for comp in ("N", "Mx", "My"):
                    yield (f"{base}.emitted[{i}].{comp}",
                           float(getattr(act, comp)))
                ca = rec.cancelled_at[i]
                yield (f"{base}.emitted[{i}].cancelled",
                       bool(ca is not None))
                if ca is not None:
                    yield (f"{base}.emitted[{i}].cancelled_at", int(ca))

        for (ei, si, tn) in sorted(self.substeps):
            base = f"{t}step[{ei:03d}].sub[{si:03d}].tendon[{tn}]"
            for k, v in sorted(self.substeps[(ei, si, tn)].items()):
                yield (f"{base}.{k}", v)

        for (ei, zi) in sorted(self.zones):
            base = f"{t}step[{ei:03d}].zone[{zi:02d}]"
            for k, v in sorted(self.zones[(ei, zi)].items()):
                if isinstance(v, (tuple, list)):
                    for name, comp in zip(("eps0", "chi_x", "chi_y"), v):
                        yield (f"{base}.{k}.{name}", float(comp))
                elif isinstance(v, dict):
                    # A nested fact -- the datum convergence record --
                    # flattened one level rather than yielded whole.
                    # ``numeric_fingerprint`` reads ``key = value``
                    # lines of scalars; handing it a dict would print a
                    # Python repr into a comparison file, where a
                    # reordering would read as a numeric movement.
                    for sub in sorted(v):
                        if v[sub] is None:
                            continue
                        yield (f"{base}.{k}.{sub}", v[sub])
                else:
                    yield (f"{base}.{k}", v)

    def as_dict(self) -> Dict[str, object]:
        r"""
        The whole ledger as a flat ``{key: value}`` mapping.

        Returns
        -------
        dict
        """
        return dict(self.items())

    def __repr__(self) -> str:                       # pragma: no cover
        return (f"<PrestressLedger tag={self.tag!r} "
                f"cells={len(self.records)} "
                f"substeps={len(self.substeps)} "
                f"zones={len(self.zones)}>")
