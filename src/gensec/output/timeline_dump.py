# SPDX-License-Identifier: AGPL-3.0-or-later
# GenSec — reinforced/prestressed concrete sectional analysis.
# Copyright (C) 2026  Andrea (GenSec project).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.  See <https://www.gnu.org/licenses/>.
r"""
The machine-readable construction-timeline dump — **the single source**.

Contract
--------
The schema is specified in ``handoff/13_0-GENSEC_DUMP_SCHEMA.md`` and
that document governs; this module implements it.  Three properties are
load-bearing, and each is measurable rather than asserted:

**One source.**  :func:`build_timeline_dump` produces a ``dict``.  Every
other artefact of a timeline run — the CSV tables, ``timeline.png`` — is
a pure function of that dict, computed in
:mod:`gensec.output.timeline_projections`, which imports nothing from
:mod:`gensec.solver` at all.  Building a figure and a table *in parallel*
from the resolution is the failure this arrangement removes: two
renderings of one run agree only until somebody edits one of them.

**Full provenance.**  Every time-dependent quantity carries where it
came from — the provider's declared domain with its citation, each
:class:`~gensec.materials.base.DeclaredRule` with its source, each zone's
declared *and* derived age per event — and every numerical choice carries
what it was judged against: the resolved tolerance set with the origin of
each entry, and convergence as a **relative residual beside its
tolerance**, never as the solver's ``converged`` flag.

**Recorded absence.**  A value that is not available is ``null`` *and* a
row in ``dump["absences"]`` naming the cause.  There is no
``getattr(obj, "k", default)`` on this path: inside a versioned contract
a silent default is permanent, because a later reader cannot tell it
from a measurement.  :mod:`tools.getattr_inventory` enforces this with
``--gate``.

Reproducibility
---------------
Every digest goes through
:func:`~gensec.materials.base.digest_bytes` (BLAKE2b) or
:meth:`~gensec.materials.base.Material.content_token`.  The interpreter's
built-in ``hash`` appears nowhere, directly or through an unordered
iteration: two dumps of one tree are byte-identical under any
``PYTHONHASHSEED``.  The ``A7'`` axis of
``tests/run_material_identity_falsification.py`` measures exactly this,
in two subprocesses with the seed forced apart.

Examples
--------
>>> dump = build_timeline_dump(tl, res, section, ddb)   # doctest: +SKIP
>>> dump["schema_version"]                              # doctest: +SKIP
'1.0'
"""

from __future__ import annotations

import json
import math
import platform
from typing import Any, Dict, List, Optional, Sequence

from ..materials.base import DIGEST_SIZE, digest_bytes, encode_value
from ..solver.section_state import geometry_digest

__all__ = [
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SCHEMA_MAJOR",
    "DumpContractViolation",
    "build_timeline_dump",
    "dump_json",
    "load_dump",
]

#: Identifier of this document family.  A consumer checks it before the
#: version: a JSON file that is not a timeline dump is not an *old*
#: timeline dump.
SCHEMA_NAME: str = "gensec.timeline.dump"

#: ``MAJOR.MINOR``.  MINOR rises when a field is **added** (an old
#: reader stays correct, merely incomplete); MAJOR rises when a field is
#: removed, renamed, or changes meaning (an old reader becomes *wrong*).
SCHEMA_VERSION: str = "1.0"

#: The MAJOR this build understands.
SCHEMA_MAJOR: int = 1


class DumpContractViolation(ValueError):
    r"""
    The dump contract was broken.

    Raised when a value offered to the serialiser cannot be recorded
    honestly — a locked-in strain closed by the single-pass
    :func:`~gensec.solver.posttension.grout` rather than by
    :func:`~gensec.solver.posttension.grout_selfconsistent`, for
    instance — or when a document being read declares no version, or a
    MAJOR this build does not know.

    A :class:`ValueError` and not an :class:`AssertionError`: the input
    is genuinely malformed for the purpose, which is what
    :class:`ValueError` means in this codebase.  A broken *internal*
    invariant raises
    :class:`~gensec.solver.prestress_ledger.SingleSideViolation`
    instead.
    """


# ======================================================================
#  Absence recording
# ======================================================================

class _Absences:
    r"""
    The register of things that are not there, with their causes.

    Every ``null`` the dump emits is paired with a row here.  A test
    checks the pairing in **both** directions — no ``null`` without a
    cause, no cause without a ``null`` — because a register that may
    contain stale rows is a register nobody trusts.

    Notes
    -----
    This is the structural replacement for the substituting
    ``getattr(obj, "k", default)``.  The default and the absence record
    cost the same to write; the difference is entirely in what the
    reader of the artefact can conclude two years later.
    """

    __slots__ = ("rows",)

    def __init__(self):
        self.rows: List[Dict[str, str]] = []

    def note(self, path: str, cause: str) -> None:
        r"""
        Record one absence.

        Parameters
        ----------
        path : str
            Dotted/bracketed path of the missing value, e.g.
            ``'zones[2].datum'``.
        cause : str
            Why it is missing, in a sentence an engineer can act on.
        """
        self.rows.append({"path": str(path), "cause": str(cause)})

    def sorted_rows(self) -> List[Dict[str, str]]:
        r"""
        The rows, ordered by path.

        Returns
        -------
        list of dict
        """
        return sorted(self.rows, key=lambda r: r["path"])


# ======================================================================
#  Scalar canonicalisation
# ======================================================================

def _num(value) -> Optional[float]:
    r"""
    A float JSON can carry, or ``None``.

    ``NaN`` and infinities are not JSON; Python's :mod:`json` writes
    them as bare ``NaN``/``Infinity`` tokens that a strict parser
    rejects.  Mapping them to ``null`` here means the absence register
    sees them, which is the point: an infinite domain bound is a fact
    to state, not a token to smuggle.

    Parameters
    ----------
    value : float or None

    Returns
    -------
    float or None
    """
    if value is None:
        return None
    out = float(value)
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _hex(digest: bytes) -> str:
    r"""
    Hexadecimal form of a digest.

    Parameters
    ----------
    digest : bytes

    Returns
    -------
    str
    """
    return bytes(digest).hex()


def _callable_name(value) -> Optional[str]:
    r"""
    Importable ``module.qualname`` of a callable, or ``None``.

    ``None`` rather than a ``'?'`` placeholder.  A placeholder would
    occupy the field a reader uses to *find* the rule, and finding
    nothing there is a different fact from finding a name that does not
    resolve — the second sends someone looking for a module that was
    never written.

    Parameters
    ----------
    value : callable

    Returns
    -------
    str or None
        ``None`` for a lambda, a closure, or any callable without both
        attributes: none of them can be identified across processes,
        which is the same bar
        :func:`~gensec.materials.base.encode_value` applies.
    """
    if not (hasattr(value, "__module__") and hasattr(value, "__qualname__")):
        return None
    name = f"{value.__module__}.{value.__qualname__}"
    if "<lambda>" in name or "<locals>" in name:
        return None
    return name


def _material_entry(material, path: str, absences: _Absences) -> Optional[dict]:
    r"""
    Identity of a material, or a recorded absence.

    Parameters
    ----------
    material : Material or None
    path : str
        Where this sits in the dump, for the absence register.
    absences : _Absences

    Returns
    -------
    dict or None
        ``{"class", "content_token"}``, plus whichever of ``fck`` /
        ``fyk`` / ``Ep`` the class actually declares.
    """
    if material is None:
        absences.note(path, "the element carries no material object; "
                            "its identity cannot be recorded.")
        return None
    entry = {
        "class": type(material).__name__,
        "content_token": _hex(material.content_token()),
    }
    for field in ("fck", "fyk", "f_p01d", "Ep"):
        if hasattr(material, field):
            entry[field] = _num(getattr(material, field))
    return entry


# ======================================================================
#  declared:  the provider domains and the engineer's rules
# ======================================================================

def _declared_block(timeline, absences: _Absences) -> dict:
    r"""
    Serialise every provider's declared domain and declared rules.

    Keyed by **loss-model name**, not by zone: a provider is shared by
    however many zones reference it, and writing it once per zone would
    be two homes for one fact.

    Parameters
    ----------
    timeline : ConstructionTimeline
    absences : _Absences

    Returns
    -------
    dict
        The ``declared`` object of the schema (``13_0`` §5).
    """
    providers: Dict[str, dict] = {}
    for name in sorted(timeline.losses_models):
        model = timeline.losses_models[name]
        provider = model.provider
        path = f"declared.providers[{name}]"

        domain: Dict[str, dict] = {}
        # The quantities a provider *declares an interval for*.  One
        # that it states at every age is deliberately ABSENT here, not
        # present with an infinite bound: that is what ``domain(q) is
        # None`` means on the ABC, and inventing a second convention
        # for it would put one fact in two shapes.
        names = sorted(set(type(provider).DOMAIN)
                       | set(getattr(provider, "_domain", None) or {}))
        for quantity in names:
            interval = provider.domain(quantity)
            if interval is None:
                continue
            lo, hi = interval
            domain[quantity] = {
                "interval": [_num(lo), _num(hi)],
                "source": provider.domain_source(quantity),
            }
            if _num(hi) is None:
                # ``null`` as the upper bound means unbounded above.
                # Said in the register rather than left to inference.
                absences.note(
                    f"{path}.domain[{quantity}].interval[1]",
                    f"the provider states {quantity!r} up to unbounded "
                    f"age; JSON has no infinity, so the bound is null.")

        rules: Dict[str, dict] = {}
        for quantity in sorted(provider.declared):
            rule = provider.declared[quantity]
            lo, hi = rule.domain
            entry = {
                "domain": [_num(lo), _num(hi)],
                "source": rule.source,
            }
            if callable(rule.value):
                entry["kind"] = "callable"
                entry["value"] = None
                entry["callable"] = _callable_name(rule.value)
                absences.note(
                    f"{path}.rules[{quantity}].value",
                    f"the declared rule for {quantity!r} is a callable "
                    f"of age, not a constant; its qualified name is "
                    f"recorded instead of a value.")
                if entry["callable"] is None:
                    absences.note(
                        f"{path}.rules[{quantity}].callable",
                        f"the declared rule for {quantity!r} is a "
                        f"callable that carries no importable name -- a "
                        f"lambda, a closure, or an instance with "
                        f"__call__ -- so it cannot be identified across "
                        f"processes and none is recorded.")
            else:
                entry["kind"] = "constant"
                entry["value"] = _num(rule.value)
            for bound, index in ((lo, 0), (hi, 1)):
                if _num(bound) is None:
                    absences.note(
                        f"{path}.rules[{quantity}].domain[{index}]",
                        f"the declared rule for {quantity!r} is "
                        f"unbounded on this side; JSON has no infinity, "
                        f"so the bound is null.")
            rules[quantity] = entry

        providers[name] = {
            "class": type(provider).__name__,
            "content_token": _hex(provider.content_token()),
            "domain": domain,
            "rules": rules,
        }
    return {"providers": providers}


# ======================================================================
#  Zones, tendons, rebars
# ======================================================================

def _zone_entries(timeline, resolution, section,
                  absences: _Absences) -> List[dict]:
    r"""
    One entry per bulk zone, with its datum and that datum's convergence.

    Parameters
    ----------
    timeline : ConstructionTimeline
    resolution : TimelineResolution
    section : GenericSection
    absences : _Absences

    Returns
    -------
    list of dict
    """
    zone_names = list(getattr(section, "zone_names", None) or ["base"])
    ledger = resolution.ledger
    cast_at = {}
    for (event_index, zone_index) in ledger.zones:
        cast_at.setdefault(zone_index, event_index)

    # ``bulk_material`` is mandatory on both section builders, so it is
    # read as an attribute: a default here could only mask the section
    # contract changing under us.
    materials = [section.bulk_material]
    materials += [mat for _poly, mat in
                  (getattr(section, "bulk_materials", None) or [])]

    out: List[dict] = []
    for index, name in enumerate(zone_names):
        path = f"zones[{index}]"
        material = materials[index] if index < len(materials) else None
        entry = {
            "index": index,
            "name": str(name),
            "material": _material_entry(material, f"{path}.material",
                                        absences),
            "cast_at_step": cast_at.get(index),
            "datum": None,
        }
        if index not in cast_at:
            absences.note(
                f"{path}.cast_at_step",
                f"zone '{name}' is declared on the section but is never "
                f"cast by this construction history"
                + (" (zone 0 is the base: it is there from the start "
                   "and is never castable)." if index == 0 else "."))
            absences.note(
                f"{path}.datum",
                f"zone '{name}' is never cast, so no datum is locked in "
                f"for it and no equilibrium was solved to find one.")
            out.append(entry)
            continue

        cell = ledger.zones[(cast_at[index], index)]
        triple = cell.get("auto_datum")
        convergence = cell.get("datum_convergence")
        entry["datum"] = {
            "eps0": _num(triple[0]),
            "chi_x": _num(triple[1]),
            "chi_y": _num(triple[2]),
            "origin": cell.get("origin"),
            "convergence": _convergence_entry(
                convergence, f"{path}.datum.convergence", absences),
        }
        out.append(entry)
    return out


def _convergence_entry(record, path: str,
                       absences: _Absences) -> Optional[dict]:
    r"""
    A convergence record, or a recorded absence.

    The solver's ``converged`` flag is carried as ``solver_flag``, one
    disjunct of ``accepted``, **never alone**.  On its own it is judged
    on an absolute residual and is therefore meaningless above
    :math:`10^{8}` N·mm — it has already caused one defect
    (``12_12`` §2), and freezing it into a versioned format alone would
    make that trap permanent.

    Parameters
    ----------
    record : dict or None
        As produced by
        :func:`gensec.solver.timeline._datum_plane_residual`, or the
        ``{"solved": False, "cause": ...}`` form.
    path : str
    absences : _Absences

    Returns
    -------
    dict or None
    """
    if record is None:
        absences.note(path, "no convergence record was written for this "
                            "quantity by the resolution walk.")
        return None
    if not record.get("solved", True):
        absences.note(path, record["cause"])
        return None
    return {
        "residual_force_rel": _num(record["residual_force_rel"]),
        "residual_moment_rel": _num(record["residual_moment_rel"]),
        "tolerance_force_rel": _num(record["tolerance_force_rel"]),
        "tolerance_moment_rel": _num(record["tolerance_moment_rel"]),
        "iterations": record["iterations"],
        "solver_flag": bool(record["solver_flag"]),
        "accepted": bool(record["accepted"]),
    }


def _tendon_entries(resolution, section, absences: _Absences) -> List[dict]:
    r"""
    One entry per tendon, read from the ledger.

    Nothing here is reconstructed.  The locked-in strain comes from
    ``resolution.grout_eps_init``, the closure from the ledger cell the
    grout wrote, the state and stress from the ledger walk.  Rebuilding
    :math:`P = \sigma_{p0} A_p \gamma_P` from the jacking stress was
    finding F-C: two independent computations of one quantity agree
    only until one of them is edited, and disagree in silence.

    Parameters
    ----------
    resolution : TimelineResolution
    section : GenericSection
    absences : _Absences

    Returns
    -------
    list of dict

    Raises
    ------
    DumpContractViolation
        A locked-in strain closed by the single-pass ``grout``.
    """
    ledger = resolution.ledger
    tendons = list(getattr(section, "tendons", None) or [])
    # The tree's one naming convention (``_tendon_local_index`` and its
    # siblings).  Not re-invented here: a dump that named tendons its
    # own way would not join to the ledger it reads.
    names = [t.name or f"tendon_{k}" for k, t in enumerate(tendons)]

    out: List[dict] = []
    for local, name in enumerate(names):
        path = f"tendons[{local}]"
        tendon = tendons[local]
        grouted_at = None
        closure = None
        for event_index in ledger.steps():
            cell = ledger.at(event_index, name)
            if cell is not None and cell.closure is not None:
                grouted_at = event_index
                closure = dict(cell.closure)

        if closure is not None and closure.get("kind") == "single_pass":
            raise DumpContractViolation(
                f"tendon '{name}': its locked-in strain was closed by "
                f"the single-pass grout(), which forms the datum against "
                f"the *unbonded* stiffness and leaves the O(n*rho) "
                f"residue Phase 12 measured at -7.08 MPa on the "
                f"flagship.  grout() is a legitimate primitive without "
                f"solver context and is not deprecated -- "
                f"grout_selfconsistent() calls it as its first iterate "
                f"-- but its residue must not enter a versioned dump "
                f"unannounced.  Close the datum with "
                f"grout_selfconsistent() (13_0 section 8.1).")

        entry = {
            "name": str(name),
            "pre_post": resolution.pre_post.get(name),
            "Ap": _num(tendon.Ap),
            "x": _num(tendon.x),
            "y": _num(tendon.y),
            "material": _material_entry(tendon.material,
                                        f"{path}.material", absences),
            "stressed_at_step": resolution.stressed.get(name),
            "grouted_at_step": grouted_at,
            "sigma_p0": _num(resolution.stress_sigma.get(name)),
            "eps_init": _num(resolution.grout_eps_init.get(name)),
            "closure": closure,
        }
        if entry["pre_post"] is None:
            absences.note(
                f"{path}.pre_post",
                f"tendon '{name}' is never referenced by a 'stress' "
                f"event, so the walk never classified it as pre- or "
                f"post-tensioned.")
        if entry["stressed_at_step"] is None:
            absences.note(
                f"{path}.stressed_at_step",
                f"tendon '{name}' is never stressed by this "
                f"construction history.")
        if entry["sigma_p0"] is None:
            absences.note(
                f"{path}.sigma_p0",
                f"no jacking stress is declared for tendon '{name}': "
                f"its 'stress' event carries no 'sigma_p0'.")
        if entry["grouted_at_step"] is None:
            absences.note(
                f"{path}.grouted_at_step",
                f"tendon '{name}' is never grouted; it is pre-tensioned, "
                f"or it stays a demand-side load for the whole history.")
            absences.note(
                f"{path}.closure",
                f"tendon '{name}' has no reconciled locked-in strain, "
                f"because it is never grouted; there is no closure to "
                f"report.")
        if entry["eps_init"] is None and grouted_at is not None:
            absences.note(
                f"{path}.eps_init",
                f"tendon '{name}' reached a grout event but the "
                f"resolution recorded no reconciled strain for it.")
        if tendon.x is None:
            absences.note(
                f"{path}.x",
                f"tendon '{name}' declares no x coordinate; the section "
                f"is treated as uniaxial in the y direction.")
        out.append(entry)
    return out


def _rebar_entries(section, zones_active,
                   absences: _Absences) -> List[dict]:
    r"""
    Rebar detail at one step, with each bar's activity.

    A bar exists once its parent zone is cast; before that it is not in
    the resistance state.  Which is a *geometric* consequence of the
    zone activity the walk recorded, not a second physics.

    Parameters
    ----------
    section : GenericSection
    zones_active : sequence of int
    absences : _Absences

    Returns
    -------
    list of dict
    """
    active = set(int(z) for z in zones_active)
    out = []
    for index, bar in enumerate(getattr(section, "rebars", None) or []):
        zone = _rebar_zone(section, index, bar)
        # ``name``, ``x``, ``y`` and ``As`` are all declared fields of
        # ``RebarLayer``; ``As`` is validated positive at construction.
        # Read as attributes, so a change to that contract raises here
        # instead of quietly writing a default into the dump.
        out.append({
            "index": index,
            "name": bar.name if bar.name else f"rebar_{index}",
            "x": _num(bar.x),
            "y": _num(bar.y),
            "As": _num(bar.As),
            "zone": zone,
            "active": bool(zone in active),
        })
    return out


def _rebar_zone(section, index: int, bar) -> int:
    r"""
    Parent bulk zone of a rebar.

    Parameters
    ----------
    section : GenericSection or RectSection
    index : int
    bar : RebarLayer

    Returns
    -------
    int

    Notes
    -----
    The three branches are exhaustive and each is a *statement*, not a
    fallback: an explicit ``parent`` wins where the element type has
    one at all (:class:`~gensec.geometry.fiber.Tendon` does,
    :class:`~gensec.geometry.fiber.RebarLayer` does not); else the
    mesher's containment map decides; else the section has no zones
    (``RectSection``), where zone 0 is not a default but the only zone
    there is.
    """
    if hasattr(bar, "parent") and bar.parent is not None:
        return int(section.zone_index(bar.parent))
    if hasattr(section, "mat_indices_rebar"):
        mat_idx = section.mat_indices_rebar
        if index < len(mat_idx):
            return int(mat_idx[index])
    return 0


# ======================================================================
#  Steps
# ======================================================================

def _latest_cell(ledger, name: str, upto: int):
    r"""
    The tendon's most recent cell at or before *upto*.

    Parameters
    ----------
    ledger : PrestressLedger
    name : str
    upto : int

    Returns
    -------
    TendonRecord or None
    """
    best = None
    for event_index in ledger.steps():
        if event_index > int(upto):
            break
        cell = ledger.at(event_index, name)
        if cell is not None:
            best = cell
    return best


def _step_entries(timeline, resolution, section,
                  absences: _Absences) -> List[dict]:
    r"""
    One entry per physical event, in history order.

    ``point`` events are not physical and get no entry of their own:
    they appear in ``anchors`` on the step they follow, which is
    exactly what :attr:`ConstructionTimeline.points` means (a prefix
    length).

    Parameters
    ----------
    timeline : ConstructionTimeline
    resolution : TimelineResolution
    section : GenericSection
    absences : _Absences

    Returns
    -------
    list of dict
    """
    ledger = resolution.ledger
    anchors: Dict[int, List[str]] = {}
    for point, prefix in timeline.points.items():
        anchors.setdefault(prefix - 1, []).append(str(point))

    # Whether the demand side is knowable at this level at all.  The
    # resolution ledger records *timeline physics*; the demand-side
    # emissions live in the per-(combination, point) emission ledgers,
    # because gamma_P belongs to the combination and not to the
    # timeline.  On a resolution-only dump term B of the single-side
    # invariant is therefore not zero -- it is **unknown**, and writing
    # 0 would be a silent default wearing a number's clothes.
    emissions_known = any(cell.emitted for cell in ledger.records.values())

    out: List[dict] = []
    for event in timeline.events:
        path = f"steps[{event.index}]"
        facts = ledger.step_facts.get(event.index)
        if facts is None:
            absences.note(
                path,
                f"the resolution walk recorded no facts for event "
                f"{event.index}; the walk did not reach it.")
            continue

        rheology = {}
        for zone_index in sorted(facts["rheology"]):
            entry = dict(facts["rheology"][zone_index])
            cause = entry.pop("cause", None)
            if cause is not None:
                absences.note(f"{path}.rheology[{zone_index}].model",
                              cause)
            for key in ("age_declared", "age_derived", "curing"):
                if entry[key] is None and cause is None:
                    absences.note(
                        f"{path}.rheology[{zone_index}].{key}",
                        f"zone {zone_index} declares a rheological "
                        f"model but no {key.replace('_', ' ')}.")
                entry[key] = _num(entry[key])
            rheology[str(zone_index)] = entry

        tendons = {}
        for name in ledger.tendons():
            first = min(ei for ei, tn in ledger.records if tn == name)
            if event.index < first:
                # The tendon does not exist in the bookkeeping yet.
                continue
            # State is carried forward, in the *dump*, not left to each
            # consumer to carry: a tendon grouted at step 2 is still
            # bonded at step 7, and a dump that only spoke at the steps
            # where the ledger happens to hold a cell would make every
            # reader re-derive the carry — differently, eventually.
            # ``state_of`` and ``live_actions`` are the ledger's own
            # queries, so the carry has one home and it is not here.
            cell = ledger.at(event.index, name)
            latest = _latest_cell(ledger, name, event.index)
            # ``pre_post`` is deliberately NOT repeated here.  A tendon
            # does not become pre-tensioned halfway through a history:
            # the classification is a property of the tendon, it has one
            # home in ``tendons[*].pre_post``, and copying it onto every
            # step would be two homes for one fact -- with the copy free
            # to fall out of step, which is precisely what it did (the
            # grout cell carries no classification, so the copy read
            # null from the grouting step onward while the fact itself
            # was unchanged).
            tendons[str(name)] = {
                "state": ledger.state_of(name, upto=event.index),
                "sigma": _num(None if latest is None else latest.sigma),
                "eps_init": _num(None if latest is None
                                 else latest.eps_init),
                "emitted": [
                    {"N": _num(action.N), "Mx": _num(action.Mx),
                     "My": _num(action.My),
                     "cancelled_at": cell.cancelled_at[i]}
                    for i, action in enumerate(
                        cell.emitted if cell is not None else [])
                ],
                "live_actions": (
                    ledger.live_actions(name, upto=event.index)
                    if emissions_known else None),
                "recorded_here": cell is not None,
            }
            if not emissions_known:
                absences.note(
                    f"{path}.tendons[{name}].live_actions",
                    f"the demand side of tendon '{name}' is not known "
                    f"at this level: gamma_P belongs to a combination, "
                    f"so the emitted actions live in the per-"
                    f"(combination, point) emission ledgers under "
                    f"combinations[*].emissions, and this dump records "
                    f"a resolution only.  Zero here would read as 'no "
                    f"demand-side action', which is a different claim.")
            if tendons[str(name)]["sigma"] is None:
                absences.note(
                    f"{path}.tendons[{name}].sigma",
                    f"the walk knows no stress for tendon '{name}' at "
                    f"this step: its 'stress' event declares no "
                    f"'sigma_p0'.")
            if tendons[str(name)]["eps_init"] is None:
                absences.note(
                    f"{path}.tendons[{name}].eps_init",
                    f"tendon '{name}' is not yet grouted at this step, "
                    f"so it carries no reconciled locked-in strain: it "
                    f"is a demand-side load, not a resistance element.")

        out.append({
            "index": event.index,
            "kind": facts["kind"],
            "t_days_start": _num(facts["t_days_start"]),
            "t_days": _num(facts["t_days"]),
            "duration_days": _num(facts["duration_days"]),
            "anchors": sorted(anchors.get(event.index, [])),
            "zones_active": list(facts["zones_active"]),
            "zones_cast_here": list(facts["zones_cast_here"]),
            "loads": {
                "applied_here": list(facts["loads_applied"]),
                "cumulative": dict(facts["demand_cumulative"]),
            },
            "rheology": rheology,
            "rebars": _rebar_entries(section, facts["zones_active"],
                                     absences),
            "tendons": tendons,
            "losses_integrated": bool(
                event.index in resolution.losses_ops),
            "losses_skipped": bool(
                event.index in resolution.skipped_losses),
        })
    return out


# ======================================================================
#  Combinations
# ======================================================================

def _emission_entries(resolution, combo_name, absences: _Absences,
                      path: str) -> Optional[dict]:
    r"""
    The per-``(point, step, tendon)`` emission record of a combination.

    This is where the **demand side** of the single-side invariant
    lives.  It is per combination and not per timeline because
    :math:`\gamma_P` is a property of the combination, so the actions
    actually emitted differ between two combinations resolved against
    one history.

    With this block a consumer can decide

    .. math::

        A \oplus B = \text{true}, \qquad
        A = \text{"in the resistance state"}, \quad
        B = \text{"a live demand-side action"}

    from the artefact, instead of trusting that the producer checked
    it.  Without it the invariant is not merely unverified — it is not
    *expressible*, which is why the resolution-only case records the
    demand side as unknown rather than as zero.

    Parameters
    ----------
    resolution : TimelineResolution
    combo_name : str
    absences : _Absences
    path : str

    Returns
    -------
    dict or None
        ``{point: {step: {tendon: {...}}}}``, or ``None`` when the
        combination was never compiled.
    """
    ledgers = {point: ledger
               for (name, point), ledger
               in (resolution.emission_ledgers or {}).items()
               if name == combo_name}
    if not ledgers:
        absences.note(
            f"{path}.emissions",
            f"combination '{combo_name}' was never compiled against "
            f"this timeline, so no demand-side actions were emitted "
            f"and the single-side invariant cannot be decided from "
            f"this dump.")
        return None

    out: Dict[str, dict] = {}
    for point in sorted(ledgers, key=str):
        ledger = ledgers[point]
        by_step: Dict[str, dict] = {}
        for event_index in ledger.steps():
            cells = {}
            for name in ledger.tendons():
                cell = ledger.at(event_index, name)
                if cell is None:
                    continue
                cells[str(name)] = {
                    "state": ledger.state_of(name, upto=event_index),
                    "live_actions": ledger.live_actions(
                        name, upto=event_index),
                    "emitted": [
                        {"N": _num(action.N), "Mx": _num(action.Mx),
                         "My": _num(action.My),
                         "cancelled_at": cell.cancelled_at[i]}
                        for i, action in enumerate(cell.emitted)
                    ],
                }
            if cells:
                by_step[str(event_index)] = cells
        out[str(point)] = by_step
    return out


def _combination_entries(combinations, resolution, verification,
                         absences: _Absences) -> List[dict]:
    r"""
    The declared combinations, with per-point metrics when they exist.

    Parameters
    ----------
    combinations : sequence of dict
    verification : dict or None
        ``{combination name: governing_point(...)}`` — the ``anchored``
        member of :func:`gensec.solver.timeline_run.run_timeline`.
    absences : _Absences

    Returns
    -------
    list of dict
    """
    out = []
    for position, combo in enumerate(combinations or []):
        name = combo.get("name")
        path = f"combinations[{position}]"
        at = combo.get("at")
        entry = {
            "name": (None if name is None else str(name)),
            "at": ([] if at is None else
                   ([str(at)] if isinstance(at, str)
                    else [str(a) for a in at])),
            "history_factors": {str(k): _num(v) for k, v in
                                (combo.get("history_factors") or {}).items()},
            "gamma_P": _gamma_p_entry(combo.get("gamma_P")),
            "components": [
                {"ref": str(c.get("ref")), "factor": _num(c.get("factor"))}
                for c in (combo.get("components") or [])
            ],
            "per_point": None,
            "emissions": _emission_entries(resolution, name, absences,
                                           path),
        }
        if at is None:
            absences.note(
                f"{path}.at",
                f"combination '{name}' declares no timeline anchor; it "
                f"is verified outside the timeline, on the legacy path.")
        if entry["gamma_P"] is None:
            absences.note(
                f"{path}.gamma_P",
                f"combination '{name}' declares no gamma_P; the "
                f"compiler applies the active normative's recommended "
                f"pair, and none is written down in the document.")
        result = (verification or {}).get(name)
        if result is None:
            absences.note(
                f"{path}.per_point",
                f"no verification result is available for combination "
                f"'{name}': this dump records a resolution only, or the "
                f"combination was not anchored.")
        else:
            entry["per_point"] = {
                str(point): {
                    "eta_governing": _num(payload.get("eta_governing")),
                    "verified": (None if payload.get("verified") is None
                                 else bool(payload["verified"])),
                }
                for point, payload in sorted(
                    (result.get("per_point") or {}).items())
            }
        out.append(entry)
    return out


def _gamma_p_entry(value):
    r"""
    ``gamma_P`` as a JSON scalar.

    Parameters
    ----------
    value : str, float or None

    Returns
    -------
    str, float or None
    """
    if value is None or isinstance(value, str):
        return value
    return _num(value)


# ======================================================================
#  The builder
# ======================================================================

def build_timeline_dump(timeline, resolution, section, demand_db, *,
                        stem: str = "",
                        combinations: Sequence[dict] = (),
                        verification: Optional[dict] = None) -> dict:
    r"""
    Build the machine-readable dump of a resolved construction timeline.

    This is **the** artefact.  Nothing else about a timeline run is
    rendered from the resolution: the CSV projections and
    ``timeline.png`` are pure functions of the dict returned here (see
    :mod:`gensec.output.timeline_projections`).

    Parameters
    ----------
    timeline : ConstructionTimeline
        The resolved timeline — its events, points, declared rheology
        and tolerance set are all serialised.
    resolution : TimelineResolution
        Output of :meth:`~gensec.solver.timeline.ConstructionTimeline.resolve`.
        Read, never re-run.
    section : GenericSection
        The base section, for geometry, materials and element detail.
    demand_db : dict
        ``{name: {"N", "Mx", "My"}}`` — the permanent demands the
        history references.
    stem : str, optional
        Short label for the model, typically the example file stem.
    combinations : sequence of dict, optional
        The declared verification combinations.
    verification : dict or None, optional
        ``{combination name: governing_point(...)}``.  Absent means
        this is a resolution-only dump, and every ``per_point`` is
        ``null`` with its cause recorded.

    Returns
    -------
    dict
        Conforming to ``handoff/13_0-GENSEC_DUMP_SCHEMA.md``.

    Raises
    ------
    DumpContractViolation
        A locked-in strain was closed by the single-pass
        :func:`~gensec.solver.posttension.grout` (``13_0`` §8.1).

    Examples
    --------
    >>> dump = build_timeline_dump(tl, res, sec, ddb)     # doctest: +SKIP
    >>> dump["schema"]                                    # doctest: +SKIP
    'gensec.timeline.dump'
    """
    from .._version import __version__ as _gensec_version

    absences = _Absences()

    dump: Dict[str, Any] = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "generator": {
            "tool": "gensec",
            "version": str(_gensec_version),
            "python": platform.python_version(),
        },
        "model": {
            "stem": str(stem),
            "geometry_digest": _hex(geometry_digest(section)),
            "zone_count": len(getattr(section, "zone_names", None)
                              or ["base"]),
            "tendon_count": len(getattr(section, "tendons", None) or []),
            "rebar_count": len(getattr(section, "rebars", None) or []),
            "has_time_axis": bool(timeline.has_time_axis),
            "rheology_switch": timeline.rheology,
        },
        "demands": {
            str(name): {"N": _num(d["N"]), "Mx": _num(d["Mx"]),
                        "My": _num(d["My"])}
            for name, d in sorted((demand_db or {}).items())
        },
        "declared": _declared_block(timeline, absences),
        "tolerances": {
            "resolved": timeline.tolerances.as_document(),
            "provenance": timeline.tolerances.provenance_document(),
        },
        "zones": _zone_entries(timeline, resolution, section, absences),
        "tendons": _tendon_entries(resolution, section, absences),
        "steps": _step_entries(timeline, resolution, section, absences),
        "combinations": _combination_entries(combinations, resolution,
                                             verification, absences),
    }
    dump["absences"] = absences.sorted_rows()
    dump["content_digest"] = _hex(_content_digest(dump))
    return dump


def _content_digest(dump: dict) -> bytes:
    r"""
    BLAKE2b over the whole document, minus the digest field itself.

    Through :func:`~gensec.materials.base.encode_value`, whose encoding
    is canonical and type-tagged: dict keys are sorted **as encoded
    bytes**, so no set iteration order and no ``hash()`` reaches the
    result.  Two dumps of one tree therefore agree under any
    ``PYTHONHASHSEED``, which the ``A7'`` axis measures with the seed
    forced apart.

    Parameters
    ----------
    dump : dict

    Returns
    -------
    bytes
        Digest of length :data:`~gensec.materials.base.DIGEST_SIZE`.
    """
    body = {k: v for k, v in dump.items() if k != "content_digest"}
    return digest_bytes(encode_value(body))


def dump_json(dump: dict, *, indent: Optional[int] = 2) -> str:
    r"""
    Serialise a dump to JSON, canonically.

    ``sort_keys=True`` and fixed separators, so two dumps of one tree
    are **byte-identical** and ``diff`` becomes an instrument of
    analysis rather than a source of noise.  ``allow_nan=False``:
    ``NaN`` and ``Infinity`` are not JSON, and a file a strict parser
    rejects is not a machine-readable artefact.  :func:`_num` has
    already mapped them to ``null`` with an absence row, so this only
    ever fires on a bug.

    Parameters
    ----------
    dump : dict
    indent : int or None, optional
        ``None`` gives the most compact form.

    Returns
    -------
    str
    """
    return json.dumps(dump, sort_keys=True, indent=indent,
                      separators=((",", ": ") if indent
                                  else (",", ":")),
                      allow_nan=False, ensure_ascii=False)


def load_dump(obj) -> dict:
    r"""
    Read a dump, refusing anything this build cannot read honestly.

    The **one home** of the version rule (``13_0`` §1).

    Parameters
    ----------
    obj : dict or str
        A parsed dump, or the JSON text of one.

    Returns
    -------
    dict

    Raises
    ------
    DumpContractViolation
        Not a dump; no ``schema_version``; a MAJOR this build does not
        know.

    Notes
    -----
    A missing version is refused rather than assumed to be the current
    one.  Assuming is how a format loses the ability to evolve: the
    first reader that guesses turns every future MAJOR bump into a
    silent misreading of old files.

    A **higher MINOR** is accepted, with unknown fields ignored — that
    is what MINOR means.  A higher MAJOR is refused.

    Examples
    --------
    >>> load_dump({"schema": SCHEMA_NAME, "schema_version": "1.0"}
    ...           )["schema_version"]
    '1.0'
    >>> load_dump({"schema": SCHEMA_NAME})    # doctest: +ELLIPSIS
    Traceback (most recent call last):
    gensec.output.timeline_dump.DumpContractViolation: ...
    """
    if isinstance(obj, (str, bytes, bytearray)):
        obj = json.loads(obj)
    if not isinstance(obj, dict):
        raise DumpContractViolation(
            f"a timeline dump is a JSON object, got "
            f"{type(obj).__name__}.")
    if obj.get("schema") != SCHEMA_NAME:
        raise DumpContractViolation(
            f"this is not a GenSec timeline dump: 'schema' is "
            f"{obj.get('schema')!r}, expected {SCHEMA_NAME!r}.  A JSON "
            f"file that is not a timeline dump is not an *old* timeline "
            f"dump.")
    version = obj.get("schema_version")
    if version is None:
        raise DumpContractViolation(
            f"the document declares no 'schema_version'.  It is refused "
            f"rather than read as {SCHEMA_VERSION!r}: a dump of unknown "
            f"provenance read as the current version is how a format "
            f"loses the ability to evolve -- every later MAJOR bump then "
            f"silently misreads every old file.  Add "
            f"'schema_version: \"{SCHEMA_VERSION}\"' if the document "
            f"really is one.")
    try:
        major = int(str(version).split(".")[0])
    except (TypeError, ValueError):
        raise DumpContractViolation(
            f"'schema_version' must be 'MAJOR.MINOR', got "
            f"{version!r}.") from None
    if major != SCHEMA_MAJOR:
        raise DumpContractViolation(
            f"this build reads schema MAJOR {SCHEMA_MAJOR}, the document "
            f"declares {major} (version {version!r}).  A MAJOR bump means "
            f"a field was removed, renamed or changed meaning, so an "
            f"older reader would be *wrong* rather than incomplete.")
    return obj
