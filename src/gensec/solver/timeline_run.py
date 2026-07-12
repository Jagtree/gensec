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
        losses_models=data.get("losses_models"))
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
