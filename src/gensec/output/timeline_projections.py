# SPDX-License-Identifier: AGPL-3.0-or-later
# GenSec — reinforced/prestressed concrete sectional analysis.
# Copyright (C) 2026  Andrea (GenSec project).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.  See <https://www.gnu.org/licenses/>.
r"""
Projections of the timeline dump — CSV tables and ``timeline.png``.

**Every function here takes the dump dict and nothing else.**  That is
the whole design, and it is structural rather than conventional:

* each projection has exactly **one positional parameter**, the dict;
* this module imports nothing from :mod:`gensec.solver`, so it *cannot*
  reach the resolution even by accident;
* a test asserts both — the signature with :mod:`inspect`, and the
  absence of the import by walking this module's own AST.

The property being protected is that a figure and a table of one run
never disagree.  Built in parallel from a resolution they agree only
until somebody edits one of them, and the disagreement is silent: two
renderings, both plausible, no third thing to arbitrate.  Derived from
one dict they cannot disagree at all, because there is only one set of
numbers.

A clean signature with a hidden import would be parallelism with an
alibi, which is why the test checks the import too.

Notes
-----
The dump is the contract (``handoff/13_0-GENSEC_DUMP_SCHEMA.md``).
These functions read it the way any other consumer would, which is also
how they stay honest: a field they need and the schema does not promise
is a bug in one of the two, surfaced here rather than papered over.
"""

from __future__ import annotations

import csv
import io
from typing import Dict, List, Optional, Sequence

__all__ = [
    "CAST_PALETTE",
    "cast_colour",
    "cast_steps_from_dump",
    "project_zones_csv",
    "project_tendons_csv",
    "project_steps_csv",
    "project_timeline_figure",
]


# ======================================================================
#  The casting palette — colour encodes *when*, never *how much*
# ======================================================================

#: Qualitative colours, in casting order.  Deliberately **not** a
#: sequential colormap: a zone cast at :math:`T_0` and one cast at
#: :math:`T_1` are two different events, and nothing lies between them.
#: A continuous ramp would invite the reader to interpolate — to read
#: the midpoint as "half cast" — which is meaningless.  Chosen to stay
#: distinguishable in grey-scale print and under the common forms of
#: colour blindness (Okabe–Ito, reordered so the first two are the pair
#: an ordinary precast + topping model uses).
CAST_PALETTE = (
    "#0072B2",   # blue      — first cast
    "#E69F00",   # orange    — second
    "#009E73",   # green
    "#CC79A7",   # violet
    "#56B4E9",   # sky
    "#D55E00",   # vermillion
    "#F0E442",   # yellow
    "#000000",   # black
)

#: Colour for a zone this history never casts.  Grey, not a palette
#: entry: "never cast" is not a later cast, and giving it a position in
#: the sequence would say it was one.
UNCAST_COLOUR = "#BDBDBD"


def cast_colour(order: Optional[int]) -> str:
    r"""
    Colour for the *order*-th casting event.

    Parameters
    ----------
    order : int or None
        0-based rank of the cast in the history.  ``None`` means the
        zone is never cast.

    Returns
    -------
    str
        A hex colour.

    Examples
    --------
    >>> cast_colour(0) == CAST_PALETTE[0]
    True
    >>> cast_colour(None) == UNCAST_COLOUR
    True
    """
    if order is None:
        return UNCAST_COLOUR
    return CAST_PALETTE[int(order) % len(CAST_PALETTE)]


def cast_steps_from_dump(dump: dict) -> Dict[int, Optional[int]]:
    r"""
    ``{zone index: casting step}`` from a dump.

    The bridge between the dump and
    :func:`gensec.output.plots.plot_section`'s ``cast_steps``: the
    section drawing colours its bulk zones by *when they were cast*,
    and this is where that fact comes from.  The drawing never reads a
    resolution.

    Parameters
    ----------
    dump : dict

    Returns
    -------
    dict
        ``{zone index: step index or None}``.
    """
    return {int(z["index"]): z["cast_at_step"]
            for z in dump.get("zones", [])}


def _cast_order(dump: dict) -> Dict[int, Optional[int]]:
    r"""
    ``{zone index: 0-based casting rank}``.

    Zone 0 is the base and always takes rank 0.  It is never the target
    of a ``cast`` event — it is what the section *is* before the
    history starts — but "there from the beginning" is a position on
    the timeline, and the earliest one.  Colouring it as though it had
    never been cast would say the opposite of what is true, and the
    distinction the colour exists to carry is exactly
    :math:`T_0` against :math:`T_1`.

    A zone that is declared on the section and never cast by *this*
    history has no rank: it is not a later cast, it is absent, and
    giving it a position in the sequence would claim otherwise.

    Parameters
    ----------
    dump : dict

    Returns
    -------
    dict
    """
    steps = cast_steps_from_dump(dump)
    ordered = sorted((s, z) for z, s in steps.items()
                     if s is not None and z != 0)
    rank = {z: k + 1 for k, (_s, z) in enumerate(ordered)}
    if 0 in steps:
        rank[0] = 0
    return {z: rank.get(z) for z in steps}


# ======================================================================
#  CSV projections
# ======================================================================

def _write(rows: Sequence[Sequence], header: Sequence[str]) -> str:
    r"""
    Render rows as CSV text.

    Parameters
    ----------
    rows : sequence of sequence
    header : sequence of str

    Returns
    -------
    str
        CSV with ``\n`` line endings, so the text is identical on every
        platform.  A projection that differed by line ending between
        two machines would fail a byte comparison for a reason that has
        nothing to do with the analysis.
    """
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def project_zones_csv(dump: dict) -> str:
    r"""
    One row per bulk zone: datum, origin, and the datum's convergence.

    Parameters
    ----------
    dump : dict
        The timeline dump.  **The only input.**

    Returns
    -------
    str
        CSV text.
    """
    header = ("zone_index", "zone_name", "material_class",
              "material_token", "cast_at_step", "datum_origin",
              "eps0", "chi_x", "chi_y",
              "residual_force_rel", "residual_moment_rel",
              "tolerance_force_rel", "tolerance_moment_rel",
              "iterations", "solver_flag", "accepted")
    rows = []
    for zone in dump.get("zones", []):
        material = zone.get("material") or {}
        datum = zone.get("datum") or {}
        conv = datum.get("convergence") or {}
        rows.append((
            zone["index"], zone["name"],
            material.get("class"), material.get("content_token"),
            zone.get("cast_at_step"), datum.get("origin"),
            datum.get("eps0"), datum.get("chi_x"), datum.get("chi_y"),
            conv.get("residual_force_rel"),
            conv.get("residual_moment_rel"),
            conv.get("tolerance_force_rel"),
            conv.get("tolerance_moment_rel"),
            conv.get("iterations"), conv.get("solver_flag"),
            conv.get("accepted"),
        ))
    return _write(rows, header)


def project_tendons_csv(dump: dict) -> str:
    r"""
    One row per tendon: geometry, jacking stress, datum and closure.

    The ``closure_*`` columns are the point of the table.  A locked-in
    strain means nothing without knowing whether it reached its fixed
    point or stopped at the single pass, and against what tolerance it
    was judged (``13_0`` §8.1).

    Parameters
    ----------
    dump : dict
        The timeline dump.  **The only input.**

    Returns
    -------
    str
        CSV text.
    """
    header = ("tendon", "pre_post", "Ap", "x", "y",
              "stressed_at_step", "grouted_at_step", "sigma_p0",
              "eps_init", "closure_kind", "closure_iterations",
              "closure_residual_rel", "closure_tolerance_rel")
    rows = []
    for tendon in dump.get("tendons", []):
        closure = tendon.get("closure") or {}
        rows.append((
            tendon["name"], tendon.get("pre_post"), tendon.get("Ap"),
            tendon.get("x"), tendon.get("y"),
            tendon.get("stressed_at_step"),
            tendon.get("grouted_at_step"), tendon.get("sigma_p0"),
            tendon.get("eps_init"), closure.get("kind"),
            closure.get("iterations"), closure.get("residual_rel"),
            closure.get("tolerance_rel"),
        ))
    return _write(rows, header)


def project_steps_csv(dump: dict) -> str:
    r"""
    One row per ``(step, zone)``: the clock, the demand, the ages.

    Rows are per zone rather than per step because the time-dependent
    provenance — declared age against derived age — is a *per zone*
    fact, and a table with one row per step would have to fold it into
    a cell.

    Parameters
    ----------
    dump : dict
        The timeline dump.  **The only input.**

    Returns
    -------
    str
        CSV text.
    """
    header = ("step", "kind", "t_days_start", "t_days", "duration_days",
              "anchors", "zone_index", "zone_name", "rheology_model",
              "age_declared", "age_derived", "curing", "declared_at",
              "loads_applied", "cum_N", "cum_Mx", "cum_My",
              "losses_integrated", "losses_skipped")
    rows = []
    for step in dump.get("steps", []):
        loads = "|".join(str(load["demand"])
                         for load in step["loads"]["applied_here"])
        cumulative = step["loads"]["cumulative"]
        anchors = "|".join(step.get("anchors") or [])
        rheology = step.get("rheology") or {}
        if not rheology:
            rows.append((step["index"], step["kind"],
                         step["t_days_start"], step["t_days"],
                         step["duration_days"], anchors,
                         None, None, None, None, None, None, None,
                         loads, cumulative["N"], cumulative["Mx"],
                         cumulative["My"], step["losses_integrated"],
                         step["losses_skipped"]))
            continue
        for key in sorted(rheology, key=int):
            entry = rheology[key]
            rows.append((
                step["index"], step["kind"], step["t_days_start"],
                step["t_days"], step["duration_days"], anchors,
                int(key), entry.get("zone_name"), entry.get("model"),
                entry.get("age_declared"), entry.get("age_derived"),
                entry.get("curing"), entry.get("declared_at"),
                loads, cumulative["N"], cumulative["Mx"],
                cumulative["My"], step["losses_integrated"],
                step["losses_skipped"],
            ))
    return _write(rows, header)


# ======================================================================
#  The figure
# ======================================================================

#: Colours for the tendon states, one per regime.
STATE_COLOURS = {
    "stressed": "#D55E00",     # a demand-side load
    "grouted": "#F0E442",      # the transition itself
    "bonded": "#009E73",       # a resistance element
}


def project_timeline_figure(dump: dict):
    r"""
    Render ``timeline.png``: zone activity and tendon regime per step.

    Two stacked bands, sharing one step axis:

    * **zones** — a bar from the casting step to the end of the
      history, in the colour of its casting rank.  The same palette the
      section drawing uses (:func:`cast_colour`), so a reader moving
      between the two figures is reading one code.
    * **tendons** — a bar per step in the colour of the regime the
      tendon is in at that step: a demand-side load while stressed, a
      resistance element once bonded.

    The x axis is the **step index**, not the clock.  A real history
    spans a couple of months of construction and then a single
    ``interval`` to :math:`t_\infty` — the flagship runs to 25 550 days
    — so a time axis would compress everything that happens into the
    first two percent of the width.  The clock is printed on the tick
    labels instead, where it is readable, and the step spacing stays
    even.

    Parameters
    ----------
    dump : dict
        The timeline dump.  **The only input** — this function never
        sees a :class:`~gensec.solver.timeline.TimelineResolution`, and
        this module cannot import one.

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Matplotlib is imported inside the function, not at module import:
    the CSV projections are useful in a headless pipeline that has no
    display stack, and making them depend on a plotting library would
    be an unrelated cost.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    steps = dump.get("steps", [])
    zones = dump.get("zones", [])
    tendons = dump.get("tendons", [])
    n_steps = max(len(steps), 1)
    order = _cast_order(dump)

    n_rows = max(len(zones) + len(tendons), 1)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.15 * n_steps),
                                    max(3.0, 0.55 * n_rows + 2.0)))

    labels: List[str] = []
    row = 0

    # -- zones ---------------------------------------------------------
    for zone in zones:
        cast_at = zone.get("cast_at_step")
        colour = cast_colour(order.get(int(zone["index"])))
        if cast_at is None and int(zone["index"]) != 0:
            spans = []
        else:
            start = 0 if cast_at is None else int(cast_at)
            spans = [(start - 0.4, n_steps - start - 0.2)]
        ax.broken_barh(spans, (row - 0.35, 0.7),
                       facecolors=colour, edgecolor="black",
                       linewidth=0.6, zorder=3)
        marker = "" if cast_at is None else f"  (cast at {cast_at})"
        labels.append(f"zone {zone['index']} · {zone['name']}{marker}")
        row += 1

    # -- tendons -------------------------------------------------------
    for tendon in tendons:
        for step in steps:
            cell = (step.get("tendons") or {}).get(tendon["name"])
            if cell is None:
                continue
            colour = STATE_COLOURS.get(cell["state"], "#999999")
            ax.broken_barh([(step["index"] - 0.4, 0.8)],
                           (row - 0.3, 0.6),
                           facecolors=colour, edgecolor="black",
                           linewidth=0.6, zorder=3)
        labels.append(f"tendon · {tendon['name']}")
        row += 1

    # -- load markers --------------------------------------------------
    for step in steps:
        applied = step["loads"]["applied_here"]
        if not applied:
            continue
        ax.axvline(step["index"], color="#444444", linestyle=":",
                   linewidth=1.0, zorder=2)
        ax.annotate("+" + ", ".join(load["demand"] for load in applied),
                    (step["index"], -0.7),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=7, color="#444444")

    # -- axes ----------------------------------------------------------
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-0.8, max(len(labels) - 0.2, 0.5))
    ax.invert_yaxis()

    ticks, tick_labels = [], []
    for step in steps:
        ticks.append(step["index"])
        anchors = step.get("anchors") or []
        anchor = ("\n" + "/".join(anchors)) if anchors else ""
        tick_labels.append(
            f"{step['index']}\n{step['kind']}\n"
            f"t={_fmt_days(step['t_days'])}{anchor}")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_xlim(-0.7, n_steps - 0.3)
    ax.set_xlabel("construction step  (spacing is ordinal, not to scale "
                  "in time)")
    ax.grid(True, axis="x", alpha=0.2)

    handles = [Patch(facecolor=cast_colour(k), edgecolor="black",
                     label=("in place at T0" if k == 0
                            else f"cast #{k}"))
               for k in sorted({v for v in order.values()
                                if v is not None})]
    handles += [Patch(facecolor=colour, edgecolor="black", label=state)
                for state, colour in STATE_COLOURS.items()
                if any((s.get("tendons") or {}).get(t["name"], {})
                       .get("state") == state
                       for s in steps for t in tendons)]
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="upper right",
                  framealpha=0.9)

    stem = (dump.get("model") or {}).get("stem") or "construction timeline"
    ax.set_title(f"Construction timeline — {stem}", fontsize=11)
    fig.tight_layout()
    return fig


def _fmt_days(value) -> str:
    r"""
    Format a clock reading for a tick label.

    Parameters
    ----------
    value : float or None

    Returns
    -------
    str
    """
    if value is None:
        return "—"
    if float(value) >= 1000.0:
        return f"{float(value):.3g} d"
    return f"{float(value):g} d"
