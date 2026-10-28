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
Per-demand section-state plots, with an explicit outcome for every demand.

Why this module exists
----------------------
``gensec run`` writes, for each demand selected for fibre-level
post-processing, a fibre table (``fibers_<name>.csv``), a section map
(``section_<name>.png``) and a strain/stress map (``state_<name>.png``).
All three need a strain plane that satisfies equilibrium with the
demand.

Such a plane does **not exist** when the demand lies outside the
resistance domain: the constitutive laws are bounded by their ultimate
strains and no admissible strain plane can carry the action.  Before
this module the CLI printed one line for such a demand and wrote
nothing.  Selection is by descending governing :math:`\eta`, so the
demands that have no equilibrium state are exactly the ones ranked
first: with at least ``top_k`` of them, the selection contained no
demand that could be plotted and the run produced no ``state_*.png``
at all, while demands that were verified were never reached.

Contract
--------
For every selected demand exactly one of three outcomes is produced and
reported; none is silent.

``"demand"``
    Equilibrium found at the demand.  Files are named
    ``fibers_<name>.csv``, ``section_<name>.png``, ``state_<name>.png``
    -- unchanged from the previous behaviour.

``"boundary"``
    No equilibrium at the demand.  The state plotted is the one at the
    point :math:`\mathbf{R} = \mathbf{T}/\eta_{\mathrm{ray}}` where the
    ray from the origin through the demand :math:`\mathbf{T}` leaves
    the resistance hull (see
    :meth:`~gensec.solver.check.DomainChecker.boundary_point_along_ray`),
    i.e. the state at which *proportional* loading along the demand's
    own direction reaches the domain boundary.  Files carry the suffix
    ``_boundary`` and the titles say so, with the value of
    :math:`\eta_{\mathrm{ray}}`.  This is **not** the state under the
    demand and must not be read as such.

``"none"``
    No image could be produced: the boundary point is not defined, or
    the solver finds no equilibrium there either.  The record carries
    the reason; the solver's own ``reason`` / ``refusal`` are passed
    through when it supplies them.

Notes
-----
The boundary point belongs to the convex hull of the sampled domain,
which is an envelope.  Where the true domain is not convex the hull
boundary can lie outside it, and then no strain state exists at
:math:`\mathbf{R}`: the outcome is ``"none"`` with the solver's
message, never a substitute point.
"""

import os

import matplotlib.pyplot as plt

from .export import export_fiber_results_csv
from .plots import plot_section, plot_stress_profile
from .report import print_fiber_results


def _why_not(sol):
    r"""
    Describe, for a human, why a solve did not converge.

    Parameters
    ----------
    sol : dict
        Non-converged return of
        :meth:`~gensec.solver.integrator.FiberSolver.solve_equilibrium`.

    Returns
    -------
    str
        The solver's ``reason`` and ``refusal`` when it supplied them
        (pure-axial requests), otherwise the iteration count.
    """
    reason = sol.get("reason")
    refusal = sol.get("refusal")
    if reason is not None:
        return f"{reason}: {refusal}"
    return f"no convergence after {sol['iterations']} iterations"


def _write_state(section, solver, sol, label, suffix, heading,
                 N, Mx, My, outdir, show_rebar_labels=True):
    r"""
    Write the fibre table and the two figures for a converged state.

    Parameters
    ----------
    section : GenericSection
    solver : FiberSolver
    sol : dict
        Converged solve result (``eps0``, ``chi_x``, ``chi_y``).
    label : str
        Demand name; used in the file names.
    suffix : str
        ``""`` for the state at the demand, ``"_boundary"`` otherwise.
    heading : str
        Text placed before the force triple in the figure titles.
    N, Mx, My : float
        Actions the plotted state is in equilibrium with [N, N·mm].
    outdir : str
        Output directory.
    show_rebar_labels : bool, optional
        Annotate each rebar with its index and field value on
        ``section_*.png`` and ``state_*.png``. Rebar markers are
        always drawn; set to ``False`` to keep only the markers when
        a dense reinforcement layout makes the text illegible.
        Default ``True``.

    Returns
    -------
    list of str
        Paths written.
    """
    fr = solver.get_fiber_results(sol["eps0"], sol["chi_x"], sol["chi_y"])
    print(f"\n  --- {label}{suffix} ---")
    print_fiber_results(fr, section)

    written = []
    p = os.path.join(outdir, f"fibers_{label}{suffix}.csv")
    export_fiber_results_csv(fr, p)
    print(f"    Exported: {p}")
    written.append(p)

    fig = plot_section(
        section, fr,
        title=(f"{heading}: N={N/1e3:.0f},"
               f" Mx={Mx/1e6:.0f},"
               f" My={My/1e6:.0f} kNm"),
        show_rebar_labels=show_rebar_labels)
    p = os.path.join(outdir, f"section_{label}{suffix}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Exported: {p}")
    written.append(p)

    fig = plot_stress_profile(
        fr, section,
        title=(f"{heading}: N={N/1e3:.0f},"
               f" Mx={Mx/1e6:.0f},"
               f" My={My/1e6:.0f} kN*m"),
        show_rebar_labels=show_rebar_labels)
    p = os.path.join(outdir, f"state_{label}{suffix}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"    Exported: {p}")
    written.append(p)
    return written


def export_demand_state_plots(section, solver, domain, demands,
                              selected_names, outdir,
                              show_rebar_labels=True):
    r"""
    Write fibre tables and state plots for the selected demands.

    Every selected demand ends in exactly one of the three outcomes
    described in the module docstring, and every outcome is printed and
    returned.

    Parameters
    ----------
    section : GenericSection
        Section being analysed.
    solver : FiberSolver
        Solver for *section*.
    domain : DomainChecker
        Resistance hull used by the verification, needed for the
        boundary point of a demand that has no equilibrium state.
    demands : list of dict
        Parsed demands, ``{"name", "N", "Mx", "My"}`` in N and N·mm.
    selected_names : collection of str
        Names of the demands to post-process, typically the output of
        :func:`~gensec.output.summary.select_demands_for_fiber_details`.
    outdir : str
        Output directory; must exist.
    show_rebar_labels : bool, optional
        Annotate each rebar with its index and field value on the
        written ``section_*.png`` / ``state_*.png`` figures. Rebar
        markers are always drawn; set to ``False`` to keep only the
        markers when a dense reinforcement layout makes the text
        illegible. Default ``True``.

    Returns
    -------
    list of dict
        One record per processed demand, in the order of *demands*:

        ``name`` : str
            Demand name.
        ``kind`` : {``"demand"``, ``"boundary"``, ``"none"``}
            Outcome, as described in the module docstring.
        ``eta_ray`` : float or None
            :math:`\eta_{\mathrm{ray}}` of the demand, set only for
            ``"boundary"`` and for a ``"none"`` that got as far as
            computing it.
        ``files`` : list of str
            Paths written (empty for ``"none"``).
        ``message`` : str or None
            Why the state at the demand does not exist (``"boundary"``)
            or why nothing was produced (``"none"``).

    Notes
    -----
    Only the demand's own ``N``, ``Mx``, ``My`` enter the solve; no
    initial guess, tolerance or iteration limit is changed from the
    solver's own.  A demand that is *inside* the hull yet has no
    equilibrium is not treated differently from one outside: the
    outcome is decided by the solver, and the message states which
    case it was.
    """
    records = []
    for d in demands:
        label = d["name"]
        if label not in selected_names:
            continue
        N, Mx, My = d["N"], d["Mx"], d["My"]
        rec = {"name": label, "kind": "none", "eta_ray": None,
               "files": [], "message": None}

        sol = solver.solve_equilibrium(N, Mx, My)
        if sol["converged"]:
            rec["kind"] = "demand"
            rec["files"] = _write_state(
                section, solver, sol, label, "", label,
                N, Mx, My, outdir,
                show_rebar_labels=show_rebar_labels)
            records.append(rec)
            continue

        where = ("inside" if domain.is_inside(N, Mx, My) else "outside")
        no_state = (f"no equilibrium state at the demand "
                    f"({where} the hull): {_why_not(sol)}")
        try:
            bp = domain.boundary_point_along_ray(N, Mx, My)
        except ValueError as exc:
            rec["message"] = f"{no_state}; boundary point undefined: {exc}"
            records.append(rec)
            print(f"\n  --- {label} --- NO STATE PRODUCED: "
                  f"{rec['message']}")
            continue

        rec["eta_ray"] = bp["eta"]
        sol_b = solver.solve_equilibrium(bp["N"], bp["Mx"], bp["My"])
        if not sol_b["converged"]:
            rec["message"] = (
                f"{no_state}; and none at the hull boundary along the "
                f"demand ray (eta_ray={bp['eta']:.4f}) either: "
                f"{_why_not(sol_b)}")
            records.append(rec)
            print(f"\n  --- {label} --- NO STATE PRODUCED: "
                  f"{rec['message']}")
            continue

        rec["kind"] = "boundary"
        rec["message"] = no_state
        heading = (f"{label} @ HULL BOUNDARY along demand ray "
                   f"(eta_ray={bp['eta']:.3f}; demand has no "
                   f"equilibrium state)")
        print(f"\n  --- {label} --- no equilibrium state at the demand; "
              f"plotting the hull boundary along its ray "
              f"(eta_ray={bp['eta']:.4f})")
        rec["files"] = _write_state(
            section, solver, sol_b, label, "_boundary", heading,
            bp["N"], bp["Mx"], bp["My"], outdir,
            show_rebar_labels=show_rebar_labels)
        records.append(rec)

    _print_summary(records)
    return records


def _print_summary(records):
    r"""
    Print the outcome counts and every non-trivial outcome.

    Parameters
    ----------
    records : list of dict
        Output of :func:`export_demand_state_plots`.
    """
    if not records:
        return
    n_dem = sum(r["kind"] == "demand" for r in records)
    n_bnd = sum(r["kind"] == "boundary" for r in records)
    n_none = sum(r["kind"] == "none" for r in records)
    print(f"\n  State plots: {n_dem} at the demand, {n_bnd} at the hull "
          f"boundary along the demand ray (no state at the demand), "
          f"{n_none} not produced.")
    for r in records:
        if r["kind"] == "boundary":
            print(f"    BOUNDARY  {r['name']}: eta_ray="
                  f"{r['eta_ray']:.4f} -> *_boundary files")
        elif r["kind"] == "none":
            print(f"    NOT PRODUCED  {r['name']}: {r['message']}")
