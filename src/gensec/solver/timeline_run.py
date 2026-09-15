# SPDX-License-Identifier: AGPL-3.0-or-later
# GenSec — reinforced/prestressed concrete sectional analysis.
# Copyright (C) 2026  Andrea (GenSec project).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.  See <https://www.gnu.org/licenses/>.
r"""
Construction-timeline run driver (Phase 8, Task 2 — orchestration layer).

This is the thin orchestration layer that binds the timeline compiler
(:mod:`gensec.solver.timeline`, Layer 2) to the verification engine
(:mod:`gensec.solver.check`, Layer 3).  It lives *above* both — it may
import :class:`~gensec.solver.check.VerificationEngine` and
:class:`~gensec.solver.integrator.NMDiagram`, which the timeline module
deliberately does not — so the layering of ``10_0`` §2 is preserved
(the compiler never sees a domain; the driver assembles both).

Contract (single opt-in gate, mirroring
:func:`gensec.solver.io_yaml.staged_ops_present`):

* ``construction_history`` **absent** (``None``) → :func:`run_timeline`
  returns ``None`` and the caller falls through to the legacy
  per-combination path unchanged (byte-identical to the pre-Task-2 run).
* ``construction_history`` **present** → a single
  :class:`~gensec.solver.timeline.ConstructionTimeline` is built and
  resolved **once**; every combination that carries an ``at`` anchor is
  compiled against it and verified per anchor point; the governing point
  is the transparent maximum (decision C2).  Combinations without ``at``
  are returned in ``legacy`` for the caller to run the ordinary way.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .check import VerificationEngine
from .capacity import NMDiagram
from .integrator import FiberSolver
from .section_state import StagedDomainManager
from .timeline import ConstructionTimeline, governing_point
from .sls import verify_sls_staged


def _demand_db(demands) -> Dict[str, dict]:
    r"""Build the SI demand database ``{name: {N, Mx, My}}``."""
    return {d["name"]: {"N": d.get("N", 0.0), "Mx": d.get("Mx", 0.0),
                        "My": d.get("My", 0.0)} for d in demands}


def timeline_active(data: dict) -> bool:
    r"""
    Whether a model carries a construction timeline (the opt-in gate).

    Parameters
    ----------
    data : dict
        The parsed model (:func:`gensec.solver.io_yaml.load_yaml` output).

    Returns
    -------
    bool
        ``True`` iff ``data['construction_history']`` is a non-empty list.
    """
    return bool(data.get("construction_history"))


def run_timeline(data: dict, *, n_points: int = 40,
                 biaxial: Optional[bool] = None,
                 gamma_P_provider=None) -> Optional[dict]:
    r"""
    Resolve the construction timeline and verify all anchored combinations.

    Parameters
    ----------
    data : dict
        Parsed model with ``section``, ``demands``, ``combinations`` and
        (optionally) ``construction_history``.
    n_points : int, optional
        Domain resolution (mirrors the CLI ``--n-points``); the staged
        manager and the main domain are built at this resolution so the
        per-stage metrics are commensurable.
    biaxial : bool or None, optional
        Force biaxial (3-D hull) domains.  ``None`` auto-detects from the
        section mesh, matching the CLI.
    gamma_P_provider : callable or None, optional
        ``provider(kind) -> float`` supplying the favourable /
        unfavourable :math:`\gamma_P` per the active normative
        (EN 1992-1-1, NTC 2018, ACI, …).  ``None`` uses the EN
        recommended pair.

    Returns
    -------
    dict or None
        ``None`` if no timeline is present (caller uses the legacy path).
        Otherwise a mapping with:

        ``resolution``
            the :class:`~gensec.solver.timeline.TimelineResolution`
            (frozen datums, pre/post classification);
        ``anchored``
            ``{combination_name: governing_point(...)}`` for every
            ``at``-anchored combination — each value carries
            ``governing_point``, ``eta_governing``, ``verified`` and the
            full ``per_point`` breakdown;
        ``legacy``
            the list of combinations **without** ``at`` (the caller runs
            these through the ordinary per-combination path).
    """
    if not timeline_active(data):
        # Fail loud: an ``at`` anchor with no timeline to anchor against is
        # a modelling error, not a silently-ignored key.  (Without this
        # guard the combination would fall through to the legacy path and
        # its ``at`` would be dropped — exactly the kind of quiet
        # mismodel the project forbids.)
        orphan = [c.get("name") for c in data.get("combinations", [])
                  if c.get("at") is not None]
        if orphan:
            raise ValueError(
                f"combination(s) {orphan} declare an 'at' anchor but the "
                f"model has no 'construction_history' to anchor against. "
                f"Add a construction_history, or remove the anchor."
            )
        return None

    section = data["section"]
    demands = data["demands"]
    combinations = data.get("combinations", [])
    ddb = _demand_db(demands)

    tl = ConstructionTimeline.from_block(
        data["construction_history"],
        losses_models=data.get("losses_models"),
        zone_rheology=data.get("zone_rheology"))
    resolution = tl.resolve(section, ddb)

    if biaxial is None:
        biaxial = bool(getattr(section, "n_fibers_x", 1) > 1
                       and getattr(section, "n_fibers_y", 1) > 1)

    solver_diagram = NMDiagram(FiberSolver(section))
    if biaxial:
        domain = solver_diagram.generate_biaxial(
            n_angles=36, n_points_per_angle=n_points)
        gen_kwargs = {"n_angles": 36, "n_points_per_angle": n_points}
    else:
        domain = solver_diagram.generate(n_points=n_points)
        gen_kwargs = {"n_points": n_points}

    mgr = StagedDomainManager(section, biaxial=biaxial, gen_kwargs=gen_kwargs)
    engine = VerificationEngine(domain, solver_diagram, {}, n_points=n_points,
                                staged_manager=mgr)

    anchored: Dict[str, dict] = {}
    legacy: List[dict] = []
    for combo in combinations:
        if combo.get("at") is None:
            legacy.append(combo)
            continue
        compiled = tl.compile_combination(
            combo, resolution, section, ddb,
            gamma_P_provider=gamma_P_provider)
        per_point = {}
        for point, stages, _inactive in compiled:
            synthetic = {"name": f"{combo['name']}@{point}",
                         "stages": stages}
            per_point[point] = engine.check_combination(synthetic, ddb)
        anchored[combo["name"]] = governing_point(per_point)

    return {"resolution": resolution, "anchored": anchored,
            "legacy": legacy}


def _sls_stage_items(combo, tl, resolution, section, ddb,
                     gamma_P_provider):
    r"""
    The ``(point name, stages)`` pairs one combination is verified over.

    Three shapes reach here, and they are the three a combination can
    have.  An anchored combination is compiled against the timeline and
    yields one entry per anchor point; a staged combination carries its
    stages already; a simple combination is one stage holding its own
    components.  The last case exists so that a document without a
    construction history can still ask for a stress verification --
    there is nothing about SLS that requires staged construction.

    Parameters
    ----------
    combo : dict
        A parsed combination carrying an ``sls`` block.
    tl : ConstructionTimeline or None
    resolution : TimelineResolution or None
    section : GenericSection
    ddb : dict
        ``{name: {"N", "Mx", "My"}}``.
    gamma_P_provider : callable or None

    Returns
    -------
    list of tuple
        ``[(point_name, stages), ...]``.  The point name is ``'-'``
        when the combination is not anchored, so a key built from it
        is still well formed.

    Raises
    ------
    ValueError
        If the combination anchors at a point but the model carries no
        construction history.  The same refusal
        :func:`run_timeline` makes, for the same reason: an ``at``
        with nothing to anchor against is a modelling error, and
        dropping it would verify a state the author did not describe.
    """
    if combo.get("at") is not None:
        if tl is None:
            raise ValueError(
                f"combination '{combo['name']}' declares an 'at' anchor "
                f"but the model has no 'construction_history' to anchor "
                f"against.  Add a construction_history, or remove the "
                f"anchor."
            )
        return [(point, stages) for point, stages, _inactive
                in tl.compile_combination(
                    combo, resolution, section, ddb,
                    gamma_P_provider=gamma_P_provider)]
    if "stages" in combo:
        return [("-", combo["stages"])]
    return [("-", [{"name": combo["name"],
                    "components": combo.get("components", [])}])]


def run_sls(data, *, n_points=40, biaxial=None, gamma_P_provider=None):
    r"""
    Verify the SLS stresses of every combination that declares them.

    The defect this closes
    ----------------------
    Until Phase 33 the staged SLS engine
    (:func:`gensec.solver.sls.verify_sls_staged` -- cracking, per-zone
    compression limits, decompression, tendon stresses) was reachable
    **only from the Python API**.  A user writing YAML received a ULS
    domain check and no stress check at all, and nothing said so.  That
    is not a wrong number: it is the answer to a different question,
    delivered in silence.

    What a combination must declare
    -------------------------------
    An ``sls`` block, parsed by
    :func:`gensec.io_yaml._parse_sls_block`:

    .. code-block:: yaml

        combinations:
          - name: SLS_char
            at: service
            components: [{ref: Q, factor: 1.0}]
            sls:
              moduli: {base: 36000, topping: 31000}
              limits: {kind: characteristic, fyk: 450, fpk: 1860}

    Both sub-blocks are mandatory and neither has a default: the
    modulus fixes the stiffness the section is analysed with
    (:math:`E_{c,\mathrm{eff}} = E_{cm}/(1+\varphi)` at quasi-permanent,
    :math:`E_{cm}` at characteristic -- two numbers for one zone,
    separated by the combination), and the limits fix what the
    resulting stresses are held to.

    Absence is a record, not a silence
    ----------------------------------
    A combination that declares no ``sls`` block produces an explicit
    ``skipped`` entry naming the lever, never an empty result.  The two
    statements *"no stress verification was requested here"* and *"no
    stress verification exists"* have to be distinguishable in the
    report, because the first is a choice and the second was, until
    this phase, the whole of GenSec's YAML behaviour.

    Where the demand increment comes from
    -------------------------------------
    :meth:`~gensec.solver.check.VerificationEngine.resolve_components`,
    the same static method the ULS staged walk calls -- one home for
    one fact.  The other two addends of that walk are not needed here
    and are not duplicated: ``verify_sls_staged`` sums a stage's
    ``prestress_actions`` itself, and it *refuses* a stage that removes
    elements from the resistance set, which is the only case that
    produces a deactivation-release term.

    Parameters
    ----------
    data : dict
        Parsed model (:func:`gensec.io_yaml.load_yaml` output).
    n_points : int, optional
        Domain resolution for the staged manager, so the section
        states are built the same way the ULS path builds them.
    biaxial : bool or None, optional
        Force biaxial domains; ``None`` auto-detects from the mesh,
        matching the CLI and :func:`run_timeline`.
    gamma_P_provider : callable or None, optional
        ``provider(kind) -> float`` for the favourable / unfavourable
        :math:`\gamma_P` of the governing document, forwarded to the
        timeline compiler.

    Returns
    -------
    dict
        ``{"combinations": [...], "requested": int, "verified": bool}``.
        Each entry is either

        ``{"name", "skipped": True, "reason": str}``
            the combination declared no ``sls`` block, or

        ``{"name", "points": {point: <verify_sls_staged result>},
        "verified": bool}``
            one result per anchor point (``'-'`` when unanchored).

        ``verified`` at the top level is the conjunction over the
        combinations that **were** requested; a document in which none
        was requested reports ``requested = 0`` rather than a vacuous
        ``True``.

    Raises
    ------
    ValueError
        A combination anchors at a point with no timeline to anchor
        against, or its moduli do not resolve every slot.
    NotImplementedError
        A stage removes elements from the resistance set (the SLS
        accumulation across deactivation is deferred).

    Examples
    --------
    >>> out = run_sls(load_yaml("model.yaml"))       # doctest: +SKIP
    >>> out["combinations"][0]["points"]["service"]["verified"]
    True
    """
    section = data["section"]
    ddb = _demand_db(data["demands"])
    combinations = data.get("combinations", [])

    tl = None
    resolution = None
    if timeline_active(data):
        tl = ConstructionTimeline.from_block(
            data["construction_history"],
            losses_models=data.get("losses_models"),
            zone_rheology=data.get("zone_rheology"))
        resolution = tl.resolve(section, ddb)

    if biaxial is None:
        biaxial = bool(getattr(section, "n_fibers_x", 1) > 1
                       and getattr(section, "n_fibers_y", 1) > 1)
    gen_kwargs = ({"n_angles": 36, "n_points_per_angle": n_points}
                  if biaxial else {"n_points": n_points})
    mgr = StagedDomainManager(section, biaxial=biaxial,
                              gen_kwargs=gen_kwargs)

    records: List[dict] = []
    requested = 0
    for combo in combinations:
        spec = combo.get("sls")
        if spec is None:
            records.append({
                "name": combo["name"],
                "skipped": True,
                "reason": (
                    "no 'sls' block declared on this combination, so no "
                    "stress verification was requested.  Add "
                    "sls: {moduli: {...}, limits: {kind: ...}} to have "
                    "the concrete, reinforcement and tendon stresses "
                    "checked at this combination."
                ),
            })
            continue
        requested += 1
        # ('zone', name) is the slot key resolve_moduli understands; the
        # YAML surface writes the zone name alone, because a tuple is
        # not expressible in YAML and a bare name is unambiguous once
        # the axis is fixed.
        moduli = {("zone", z): v for z, v in spec["moduli"].items()}
        limits = spec["limits"]

        points: Dict[str, dict] = {}
        for pname, stages in _sls_stage_items(
                combo, tl, resolution, section, ddb, gamma_P_provider):
            states, _hashes, _bundles, _deact = mgr.resolve_stages(stages)
            sls_stages = []
            for k, stage in enumerate(stages):
                inc = VerificationEngine.resolve_components(
                    stage.get("components", []), ddb)
                sls_stages.append({
                    "name": stage["name"],
                    "state": states[k],
                    "increment": (inc["N"], inc["Mx"], inc["My"]),
                    "prestress_actions": stage.get(
                        "_prestress_actions", []),
                    "limits": limits,
                })
            points[pname] = verify_sls_staged(
                section, sls_stages, moduli=moduli)
        records.append({
            "name": combo["name"],
            "points": points,
            "verified": all(r["verified"] for r in points.values()),
        })

    return {
        "combinations": records,
        "requested": requested,
        "verified": all(r["verified"] for r in records
                        if not r.get("skipped")),
    }
# GENSEC-COMPLETE phase=33 utc=2026-09-05T19:52:30Z sha256=8e115fb74382e998f6c33bad1d02fcfab6d785c376dbd46db251f6fc7a7b9908 body_lines=411
