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
# along with GenSec. If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------

"""
Command-line interface for GenSec.

Usage::

    uv run gensec run input.yaml [--n-points 400] [--output-dir ./results]
    uv run gensec plot data_file.json [--output plot.png] [--dpi 150]
"""

import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .io_yaml import load_yaml, staged_ops_present
from .solver import FiberSolver, NMDiagram
from .solver.check import VerificationEngine
from .output import (
    print_section_info, print_fiber_results,
    plot_nm_diagram, plot_stress_profile, plot_mx_my_diagram,
    plot_moment_curvature, plot_section, plot_section_state,
    plot_demand_heatmap, plot_3d_surface,
    plot_moment_curvature_bundle, plot_polar_ductility,
    plot_polar_ductility_refactored,
    plot_moment_curvature_surface,
    plot_from_json,
    export_nm_domain_csv, export_nm_domain_json,
    export_demand_results_csv, export_demand_results_json,
    export_fiber_results_csv,
    export_3d_surface_csv, export_3d_surface_json,
    export_verification_json,
    export_combination_results_json,
    export_envelope_results_json,
    export_moment_curvature_json, export_moment_curvature_csv,
    export_mx_my_json, export_mx_my_csv,
)
from .output.summary import (
    print_demand_summary,
    print_combination_summary,
    print_envelope_summary,
    build_verification_summary,
    select_demands_for_fiber_details,
    rank_results as _rank_results,
)


# Legacy table printers removed — now in output.summary module.


#: Resolution [points] of the uniaxial N-Mx domain generated as a
#: by-product when ``gensec analyze`` builds per-stage
#: :class:`~gensec.solver.section_state.DomainBundle` objects for a
#: ``section_ops``-carrying combination.  The analyze path consumes
#: only each bundle's *solver* (force decomposition, force-release
#: equilibrium); the domain is generated but never queried, so this is
#: a pure construction-cost knob with **zero effect on results** — kept
#: coarse and hardcoded (self-documenting constant over a YAML flag).
_ANALYZE_DOMAIN_N_POINTS = 60


# ==================================================================
#  Main entry point
# ==================================================================

def main(argv=None):
    """GenSec CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        prog="gensec",
        description="GenSec -- fiber-based cross-section analysis.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- ``gensec run`` ---
    p_run = sub.add_parser(
        "run", help="Run analysis from YAML input file.")
    p_run.add_argument("input_file", help="YAML input file.")
    p_run.add_argument("--n-points", type=int, default=50,
                       help="N-M diagram resolution (default: 50).")
    p_run.add_argument("--n-chi", type=int, default=100,
                       help="Curvature scan steps per direction for "
                            "moment-curvature and polar ductility "
                            "(default: 100).")
    p_run.add_argument("--output-dir", type=str, default=".",
                       help="Output directory (default: current dir).")

    # --- ``gensec plot`` ---
    p_plot = sub.add_parser(
        "plot",
        help="Regenerate plot from a previously exported JSON file.")
    p_plot.add_argument("data_file",
                        help="JSON data file (or CSV).")
    p_plot.add_argument("--output", "-o", type=str, default=None,
                        help="Output PNG path (default: same as input"
                             " with .png extension).")
    p_plot.add_argument("--dpi", type=int, default=150,
                        help="Plot resolution (default: 150).")

    # --- ``gensec analyze`` ---
    p_analyze = sub.add_parser(
        "analyze",
        help="Force decomposition without domain generation.")
    p_analyze.add_argument("input_file",
                           help="YAML input file.")
    p_analyze.add_argument("--output-dir", type=str, default=".",
                           help="Output directory (default: cwd).")
    p_analyze.add_argument("--eta", action="store_true",
                           help="Also compute on-demand eta for "
                                "each demand / combination.")

    args = parser.parse_args(argv)

    # A refusal is a result, not a crash.  ``NotImplementedError`` marks
    # a scope boundary the library states in words (D2, D8, F-B) and
    # ``ValueError`` a rejected input; neither deserves a stack trace
    # with ``<frozen runpy>`` frames in it.  Anything else still raises,
    # because an unexpected exception is a bug and hiding it would be
    # the opposite of fail-loud.
    try:
        return _dispatch(args, argv, parser)
    except NotImplementedError as exc:
        print(f"\nERROR (outside GenSec's modelled scope): {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)


def _dispatch(args, argv, parser):
    r"""
    Route a parsed command to its handler.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    argv : list of str or None
        Raw argument vector; the no-subcommand fallback re-reads it.
    parser : argparse.ArgumentParser
        Used by the same fallback to print help.
    """
    if args.command == "run":
        _run(args)
    elif args.command == "plot":
        _plot(args)
    elif args.command == "analyze":
        _analyze(args)
    else:
        # No subcommand: if a positional arg looks like a YAML,
        # treat as ``run`` for backward compat during development.
        # Otherwise show help.
        if argv is None:
            argv = sys.argv[1:]
        if argv and (argv[0].endswith('.yaml')
                     or argv[0].endswith('.yml')):
            run_argv = ["run"] + argv
            main(run_argv)
        else:
            parser.print_help()


def _plot(args):
    """
    ``gensec plot`` subcommand: regenerate a plot from JSON data.

    Parameters
    ----------
    args : argparse.Namespace
        Must have ``data_file``, ``output``, ``dpi``.
    """
    filepath = args.data_file
    if not os.path.isfile(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    try:
        out = plot_from_json(filepath,
                             output_path=args.output,
                             dpi=args.dpi)
        print(f"  Plot saved: {out}")
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: Failed to plot '{filepath}': {exc}")
        sys.exit(1)


def _run(args):
    """``gensec run`` subcommand: full analysis pipeline."""
    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    # --- Load ---
    print(f"\nLoading: {args.input_file}")
    data = load_yaml(args.input_file)
    section = data["section"]
    demands = data["demands"]
    combinations = data.get("combinations", [])
    envelopes = data.get("envelopes", [])

    # --- Construction-history pre-flight ------------------------------
    # ``run_timeline`` is called at the far end of this function, after
    # the N-M diagrams, the 3D resistance surface and a printed
    # verification verdict.  A history the library must refuse would
    # therefore be refused *after* the user has read a green banner —
    # one that reports on the raw demands, not on the anchored
    # combinations, which have not been evaluated at that point.
    # ``validate`` is a syntactic walk with no solver in it: run it now.
    if data.get("construction_history"):
        from .solver.timeline import ConstructionTimeline
        ConstructionTimeline.from_block(
            data["construction_history"],
            losses_models=data.get("losses_models")).validate(section)
        print(f"  Construction history: "
              f"{len(data['construction_history'])} events, pre-flight OK.")

    # Output flags (v2.1, with defaults applied by io_yaml).
    output_opts = data.get("output_options", {})
    do_mx_my = output_opts.get("generate_mx_my", False)
    do_3d_surface = output_opts.get("generate_3d_surface", False)
    do_moment_curvature = output_opts.get(
        "generate_moment_curvature", False)
    do_polar_ductility = output_opts.get(
        "generate_polar_ductility", False)
    do_3d_moment_curvature = output_opts.get(
        "generate_3d_moment_curvature", False)

    # N-level strategy.
    n_levels_mode = output_opts.get("n_levels_mode", "demands")
    n_levels_count = int(output_opts.get("n_levels_count", 10))
    n_levels_explicit = output_opts.get("n_levels_values", [])
    n_angles_mx_my = int(output_opts.get("n_angles_mx_my", 72))

    # Tiered reporting parameters.
    verification_top_k = int(output_opts.get("verification_top_k", 10))
    fiber_details_top_k = int(output_opts.get("fiber_details_top_k", 5))

    print_section_info(section)

    # --- Solver ---
    solver = FiberSolver(section)
    # A section is biaxial only when both in-plane extents have
    # actual mesh resolution.  Using ``n_fibers_x > 1`` alone
    # misclassified narrow tall walls (1 fiber column, many rows) as
    # biaxial — they have real extent in y but zero extent in x, and
    # the resulting Mx-My contour was degenerate.
    is_biaxial = section.n_fibers_x > 1 and section.n_fibers_y > 1

    # --- N-Mx diagram (always) ---
    print("\nGenerating N-Mx diagram...")
    nm_gen = NMDiagram(solver)
    nm_data = nm_gen.generate(n_points=args.n_points)
    print(f"  {len(nm_data['N'])} points")
    print(f"  N:  [{nm_data['N_kN'].min():.1f},"
          f" {nm_data['N_kN'].max():.1f}] kN")
    print(f"  Mx: [{nm_data['M_kNm'].min():.1f},"
          f" {nm_data['M_kNm'].max():.1f}] kN*m")

    export_nm_domain_csv(nm_data,
                         os.path.join(outdir, "n_mx_domain.csv"))
    export_nm_domain_json(nm_data,
                          os.path.join(outdir, "n_mx_domain.json"))

    # --- N-My diagram (biaxial only) ---
    nm_data_y = None
    if is_biaxial:
        print("\nGenerating N-My diagram...")
        nm_data_y = nm_gen.generate(n_points=args.n_points,
                                    direction='y')
        print(f"  {len(nm_data_y['N'])} points")
        print(f"  My: [{nm_data_y['M_kNm'].min():.1f},"
              f" {nm_data_y['M_kNm'].max():.1f}] kN*m")
        export_nm_domain_csv(nm_data_y,
                             os.path.join(outdir, "n_my_domain.csv"))

    # --- 3D resistance surface ---
    nm_3d = None
    if is_biaxial:
        print("\nGenerating 3D resistance surface (N-Mx-My)...")
        nm_3d = nm_gen.generate_biaxial(
            n_angles=36, n_points_per_angle=args.n_points)
        print(f"  {len(nm_3d['N'])} points")

        if do_3d_surface:
            export_3d_surface_csv(
                nm_3d, os.path.join(outdir, "surface_3d.csv"))
            export_3d_surface_json(
                nm_3d, os.path.join(outdir, "surface_3d.json"))
            print("  Exported: surface_3d.csv / .json")

    # ==============================================================
    #  VERIFICATION ENGINE (v2.1)
    # ==============================================================
    domain_data = nm_3d if nm_3d is not None else nm_data

    # Phase-3 staged section-state evolution: build a manager only when
    # at least one staged combination carries ``section_ops`` (the
    # single gate is :func:`gensec.io_yaml.staged_ops_present`).
    # Without ops the construction below is byte-identical to the
    # legacy capacity-frozen run (``staged_manager=None`` is the
    # constructor default).  The manager's domain generator settings
    # mirror the main-domain build above, so the bundle of the
    # all-active state reproduces ``domain_data`` at the same
    # resolution and the per-stage metrics are commensurable with the
    # single-domain ones.
    staged_mgr = None
    if staged_ops_present(combinations):
        from .solver.section_state import StagedDomainManager
        if nm_3d is not None:
            staged_gen_kwargs = {"n_angles": 36,
                                 "n_points_per_angle": args.n_points}
        else:
            staged_gen_kwargs = {"n_points": args.n_points}
        staged_mgr = StagedDomainManager(
            section, biaxial=(nm_3d is not None),
            gen_kwargs=staged_gen_kwargs)
        print("\n  Staged section-state evolution active "
              "(section_ops present): per-stage resistance domains "
              "will be built/reused by capacity hash.")

    engine = VerificationEngine(
        domain_data, nm_gen, output_opts,
        n_points=args.n_points,
        staged_manager=staged_mgr)

    # Build demand database for combination / envelope resolution.
    demand_db = {d["name"]: d for d in demands}

    # --- 1. Demand verification ---
    demand_results = []
    if demands:
        demand_results = engine.check_demands(demands)
        print_demand_summary(demand_results, top_k=verification_top_k)

        export_demand_results_csv(
            demand_results,
            os.path.join(outdir, "demand_summary.csv"))
        export_demand_results_json(
            demand_results,
            os.path.join(outdir, "demand_summary.json"))

    # ==============================================================
    #  Phase-8 Task-2: construction-timeline driver (single opt-in gate)
    # ==============================================================
    # When the model carries a construction_history, one timeline is
    # built and resolved once; every combination that declares an
    # 'at' anchor is compiled against it and verified per anchor
    # point, the governing point being the transparent maximum (C2).
    # Anchored combinations are removed from the legacy loop below.
    # Without a timeline the call returns None and this block is inert
    # (byte-identical to the pre-Task-2 run); an 'at' anchor with no
    # timeline raises inside the driver (fail-loud).
    from .solver.timeline_run import run_timeline
    timeline_out = run_timeline(data, n_points=args.n_points,
                                biaxial=(nm_3d is not None))
    if timeline_out is not None:
        print("\n  Construction timeline active: verifying anchored "
              "combinations per point.")
        for _cname, _gov in timeline_out["anchored"].items():
            _status = "OK" if _gov["verified"] else "NOT VERIFIED"
            print(f"    {_cname}: governing point "
                  f"'{_gov['governing_point']}' "
                  f"eta={_gov['eta_governing']} ({_status})")
            for _pt, _r in _gov["per_point"].items():
                print(f"        {_pt}: eta={_r.get('eta_governing')}")
        _anchored_names = set(timeline_out["anchored"])
        combinations = [c for c in combinations
                        if c.get("name") not in _anchored_names]

    # --- 2. Combination verification ---
    combination_results = []
    combination_results_db = {}
    if combinations:
        print("\n  Verifying combinations...")
        for combo in combinations:
            try:
                cr = engine.check_combination(combo, demand_db)
                combination_results.append(cr)
                combination_results_db[combo["name"]] = cr
            except KeyError as exc:
                print(f"  ERROR in combination "
                      f"'{combo['name']}': {exc}")

        print_combination_summary(combination_results,
                                  top_k=verification_top_k)

        export_combination_results_json(
            combination_results,
            os.path.join(outdir, "combination_summary.json"))

    # --- 3. Envelope verification ---
    envelope_results = []
    if envelopes:
        print("\n  Verifying envelopes...")
        for env in envelopes:
            try:
                er = engine.check_envelope(
                    env, demand_db, combination_results_db)
                envelope_results.append(er)
            except KeyError as exc:
                print(f"  ERROR in envelope "
                      f"'{env['name']}': {exc}")

        print_envelope_summary(envelope_results,
                               top_k=verification_top_k)

        export_envelope_results_json(
            envelope_results,
            os.path.join(outdir, "envelope_summary.json"))

    # --- Unified verification export (with statistics) ---
    if demand_results or combination_results or envelope_results:
        export_verification_json(
            demand_results, combination_results, envelope_results,
            os.path.join(outdir, "verification_summary.json"))

        import json
        summary = build_verification_summary(
            demand_results, combination_results, envelope_results)
        with open(os.path.join(outdir,
                               "verification_statistics.json"),
                  'w') as f:
            json.dump(summary["statistics"], f, indent=2)

    # ==============================================================
    #  Per-demand fiber details and state plots
    # ==============================================================
    demand_plot = []
    if demands:
        # Geometry-only section drawing.
        fig_geom = plot_section(section, title="Section geometry")
        p = os.path.join(outdir, "section_geometry.png")
        fig_geom.savefig(p, dpi=150)
        plt.close(fig_geom)
        print(f"\n  Exported: {p}")

        # Build demand_plot for all demands (N-M diagram annotation).
        for d in demands:
            demand_plot.append(
                (d["N"] / 1e3, d["Mx"] / 1e6, d["name"]))

        # Select which demands get full fiber details.
        fiber_names = set(select_demands_for_fiber_details(
            demand_results, top_k=fiber_details_top_k))
        n_skipped = len(demands) - len(fiber_names)

        if n_skipped > 0:
            print(f"\n  Fiber details for top {len(fiber_names)} "
                  f"demands (by eta); {n_skipped} skipped.")
        else:
            print("\n  Per-demand details:")

        for d in demands:
            label = d["name"]
            if label not in fiber_names:
                continue

            N_d, Mx_d, My_d = d["N"], d["Mx"], d["My"]
            sol = solver.solve_equilibrium(N_d, Mx_d, My_d)

            if sol["converged"]:
                fr = solver.get_fiber_results(
                    sol["eps0"], sol["chi_x"], sol["chi_y"])
                print(f"\n  --- {label} ---")
                print_fiber_results(fr, section)

                p = os.path.join(outdir, f"fibers_{label}.csv")
                export_fiber_results_csv(fr, p)
                print(f"    Exported: {p}")

                fig = plot_section(
                    section, fr,
                    title=(f"{label}: N={N_d/1e3:.0f},"
                           f" Mx={Mx_d/1e6:.0f},"
                           f" My={My_d/1e6:.0f} kNm"))
                p = os.path.join(outdir, f"section_{label}.png")
                fig.savefig(p, dpi=150)
                plt.close(fig)
                print(f"  Exported: {p}")

                fig = plot_stress_profile(
                    fr, section,
                    title=(f"{label}: N={N_d/1e3:.0f},"
                           f" Mx={Mx_d/1e6:.0f},"
                           f" My={My_d/1e6:.0f} kN*m"))
                p = os.path.join(outdir, f"state_{label}.png")
                fig.savefig(p, dpi=150)
                plt.close(fig)
                print(f"    Exported: {p}")
            else:
                print(f"\n  --- {label} --- SOLVER DID NOT CONVERGE")

    # --- Demand utilization heatmap ---
    if demand_results:
        # For large result sets, plot only the top-K in the heatmap
        # (the bar chart becomes unreadable beyond ~30 items).
        _HEATMAP_MAX = max(verification_top_k, 30)
        if len(demand_results) > _HEATMAP_MAX:
            hm_data = _rank_results(demand_results, "demand")[:_HEATMAP_MAX]
            hm_title = (f"Demand Utilization "
                        f"(top {_HEATMAP_MAX} of "
                        f"{len(demand_results)})")
        else:
            hm_data = demand_results
            hm_title = ""
        fig_hm = plot_demand_heatmap(hm_data, title=hm_title)
        p = os.path.join(outdir, "demand_utilization.png")
        fig_hm.savefig(p, dpi=150)
        plt.close(fig_hm)
        print(f"\n  Exported: {p}")

    # ==============================================================
    #  N-M diagram plots
    # ==============================================================
    fig_nm = plot_nm_diagram(nm_data, demand_plot or None)
    p = os.path.join(outdir, "n_mx_diagram.png")
    fig_nm.savefig(p, dpi=150)
    plt.close(fig_nm)
    print(f"\n  Exported: {p}")

    if nm_data_y is not None:
        demand_plot_y = [
            (d["N"] / 1e3, d["My"] / 1e6, d["name"])
            for d in demands if abs(d["My"]) > 0
        ]
        fig_nmy = plot_nm_diagram(
            nm_data_y, demand_plot_y or None,
            title="N-My Interaction Diagram")
        p = os.path.join(outdir, "n_my_diagram.png")
        fig_nmy.savefig(p, dpi=150)
        plt.close(fig_nmy)
        print(f"  Exported: {p}")

    if nm_3d is not None and do_3d_surface:
        print("\n  Generating 3D surface plot...")
        fig_3d = plot_3d_surface(nm_3d, demands=demands)
        p = os.path.join(outdir, "surface_3d.png")
        fig_3d.savefig(p, dpi=150)
        plt.close(fig_3d)
        print(f"  Exported: {p}")

    # ==============================================================
    #  N levels for M-chi and Mx-My diagrams
    # ==============================================================
    # Only compute N levels if at least one downstream generator
    # is enabled; otherwise skip the entire block.
    _need_n_levels = (
        do_mx_my or do_moment_curvature
        or do_polar_ductility or do_3d_moment_curvature
    )

    N_levels_analysis = []
    if _need_n_levels:
        if n_levels_mode == "explicit" and n_levels_explicit:
            N_levels_analysis = sorted(
                float(v) * 1e3 for v in n_levels_explicit)
            print(f"\n  N levels (explicit): "
                  f"{[v/1e3 for v in N_levels_analysis]} kN")
        elif n_levels_mode == "auto":
            N_min_d = float(nm_data["N"].min())
            N_max_d = float(nm_data["N"].max())
            # n_levels_count interior values, endpoints (N_Rd,min / N_Rd,max,
            # where M ≡ 0) deliberately excluded. To analyse the axial limits,
            # request them explicitly via n_levels_mode: explicit.
            N_levels_analysis = sorted(
                np.linspace(N_min_d, N_max_d, n_levels_count + 2)[1:-1].tolist())
            print(f"\n  N levels (auto, {n_levels_count} interior values): "
                  f"[{N_min_d/1e3:.0f}, {N_max_d/1e3:.0f}] kN")
        else:
            N_levels_analysis = sorted(
                set(d["N"] for d in demands)) if demands else [0.0]
            print(f"\n  N levels (from demands): "
                  f"{[v/1e3 for v in N_levels_analysis]} kN")

        if not any(abs(n) < 1.0 for n in N_levels_analysis):
            N_levels_analysis.append(0.0)
            N_levels_analysis.sort()
            print(f"  Added N=0 (pure bending)")

    # --- Mx-My diagrams ---
    if is_biaxial and do_mx_my:
        for N_fixed in N_levels_analysis:
            print(f"  Generating Mx-My at"
                  f" N={N_fixed/1e3:.0f} kN...")
            mx_my = nm_gen.generate_mx_my(
                N_fixed, n_angles=n_angles_mx_my)
            N_label = f"{N_fixed/1e3:.0f}".replace("-", "m")
            # Data export (always).
            p_json = os.path.join(outdir, f"mx_my_N{N_label}.json")
            export_mx_my_json(mx_my, p_json)
            export_mx_my_csv(
                mx_my, os.path.join(outdir, f"mx_my_N{N_label}.csv"))
            # Plot.
            mx_my_demands = [
                (d["Mx"] / 1e6, d["My"] / 1e6, d["name"])
                for d in demands
                if abs(d["N"] - N_fixed) < abs(N_fixed) * 0.05 + 1
            ]
            fig = plot_mx_my_diagram(mx_my,
                                     mx_my_demands or None)
            p = os.path.join(outdir, f"mx_my_N{N_label}.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"    Exported: {p_json}, {p}")

    # --- Moment-curvature diagrams ---
    mc_collection_x = []
    mc_collection_y = []
    if do_moment_curvature:
        for N_fixed in N_levels_analysis:
            N_label = f"{N_fixed/1e3:.0f}".replace("-", "m")

            print(f"  Generating Mx-chi at N={N_fixed/1e3:.0f} kN...")
            mc_x = nm_gen.generate_moment_curvature(
                N_fixed, n_chi=args.n_chi, direction='x')
            mc_collection_x.append(mc_x)
            # Data export (always).
            p_json = os.path.join(outdir, f"mx_chi_N{N_label}.json")
            export_moment_curvature_json(mc_x, p_json)
            export_moment_curvature_csv(
                mc_x, os.path.join(outdir, f"mx_chi_N{N_label}.csv"))
            # Plot.
            fig = plot_moment_curvature(mc_x)
            p = os.path.join(outdir, f"mx_chi_N{N_label}.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"    Exported: {p_json}, {p}")

            if is_biaxial:
                print(f"  Generating My-chi at"
                      f" N={N_fixed/1e3:.0f} kN...")
                mc_y = nm_gen.generate_moment_curvature(
                    N_fixed, n_chi=args.n_chi, direction='y')
                mc_collection_y.append(mc_y)
                p_json = os.path.join(outdir, f"my_chi_N{N_label}.json")
                export_moment_curvature_json(mc_y, p_json)
                export_moment_curvature_csv(
                    mc_y, os.path.join(outdir, f"my_chi_N{N_label}.csv"))
                fig = plot_moment_curvature(mc_y)
                p = os.path.join(outdir, f"my_chi_N{N_label}.png")
                fig.savefig(p, dpi=150)
                plt.close(fig)
                print(f"    Exported: {p_json}, {p}")

        if len(mc_collection_x) > 1:
            print("\n  Generating Mx-chi bundle...")
            fig = plot_moment_curvature_bundle(mc_collection_x,
                                               direction='x')
            p = os.path.join(outdir, "mx_chi_bundle.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"  Exported: {p}")

        if len(mc_collection_y) > 1:
            print("  Generating My-chi bundle...")
            fig = plot_moment_curvature_bundle(mc_collection_y,
                                               direction='y')
            p = os.path.join(outdir, "my_chi_bundle.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"  Exported: {p}")

    # --- Polar ductility diagrams ---
    if is_biaxial and do_polar_ductility:
        for N_fixed in N_levels_analysis:
            N_label = f"{N_fixed/1e3:.0f}".replace("-", "m")
            print(f"  Generating polar ductility at"
                  f" N={N_fixed/1e3:.0f} kN...")
            fig = plot_polar_ductility_refactored(
                nm_gen, N_fixed, n_angles=n_angles_mx_my,
                n_chi=args.n_chi)
            p = os.path.join(
                outdir, f"polar_ductility_N{N_label}.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"    Exported: {p}")

    # --- 3D moment-curvature surfaces ---
    if do_3d_moment_curvature:
        if len(mc_collection_x) > 1:
            print("\n  Generating 3D Mx-chi-N surface...")
            fig = plot_moment_curvature_surface(
                mc_collection_x, direction='x')
            p = os.path.join(outdir, "mx_chi_N_surface.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"  Exported: {p}")

        if len(mc_collection_y) > 1:
            print("  Generating 3D My-chi-N surface...")
            fig = plot_moment_curvature_surface(
                mc_collection_y, direction='y')
            p = os.path.join(outdir, "my_chi_N_surface.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"  Exported: {p}")

    # ==============================================================
    #  Final summary
    # ==============================================================
    print(f"\n{'=' * 80}")
    print(f"  All outputs in: {os.path.abspath(outdir)}/")
    files = sorted(os.listdir(outdir))
    for fn in files:
        size = os.path.getsize(os.path.join(outdir, fn))
        print(f"    {fn:45s}  {size:>8,} bytes")
    print("=" * 80)
    print("\nDone.")


def _analyze(args):
    r"""
    ``gensec analyze`` subcommand: lightweight force decomposition.

    Solves equilibrium for each demand and combination and reports
    per-material force contributions.  Does **not** generate the
    resistance domain (no NMDiagram, no ConvexHull).
    """
    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    print(f"\nLoading: {args.input_file}")
    data = load_yaml(args.input_file)
    section = data["section"]
    demands = data["demands"]
    combinations = data.get("combinations", [])

    print_section_info(section)

    solver = FiberSolver(section)
    from .solver.analysis import AnalysisEngine

    # Phase-3 staged section-state evolution (same gate as ``_run``).
    # ``analyze`` never consults the resistance domain — it only uses
    # each bundle's *solver* (force decomposition on the stage's
    # materialized view, force-release equilibrium on deactivation) —
    # but ``StagedDomainManager._build_bundle`` generates a domain per
    # state by design.  A coarse uniaxial generator keeps that
    # by-product cheap without touching the decomposition numbers,
    # which depend on the solver alone.
    staged_mgr = None
    if staged_ops_present(combinations):
        from .solver.section_state import StagedDomainManager
        staged_mgr = StagedDomainManager(
            section, biaxial=False,
            gen_kwargs={"n_points": _ANALYZE_DOMAIN_N_POINTS})
        print("\n  Staged section-state evolution active "
              "(section_ops present): per-stage decomposition on the "
              "materialized section views.")

    engine = AnalysisEngine(solver, staged_manager=staged_mgr)

    demand_db = {d["name"]: d for d in demands}

    # ==============================================================
    #  1. Demand analysis
    # ==============================================================
    demand_results = []
    if demands:
        print(f"\nAnalyzing {len(demands)} demand(s)...")
        demand_results = engine.analyze_demands(demands)
        _print_analysis_table("FORCE DECOMPOSITION — DEMANDS",
                              demand_results)

        if args.eta:
            print("\nComputing on-demand η...")
            for dr in demand_results:
                if dr["converged"]:
                    eta_r = engine.compute_eta(
                        dr["total"]["N"],
                        dr["total"]["Mx"],
                        dr["total"]["My"])
                    dr["eta"] = eta_r["eta"]
                    dr["demand_inside"] = eta_r["demand_inside"]
                    print(f"  {dr['name']:>20s}  η = {eta_r['eta']:.4f}"
                          f"  inside={eta_r['demand_inside']}")
                else:
                    dr["eta"] = None
                    print(f"  {dr['name']:>20s}  NOT CONVERGED")

    # ==============================================================
    #  2. Combination analysis
    # ==============================================================
    combination_results = []
    if combinations:
        print(f"\nAnalyzing {len(combinations)} combination(s)...")
        combination_results = engine.analyze_combinations(
            combinations, demand_db)
        for cr in combination_results:
            if cr.get("type") == "staged":
                _print_staged_analysis(cr)
            else:
                _print_analysis_table(
                    f"FORCE DECOMPOSITION — {cr['name']}",
                    [cr])

            if args.eta and cr.get("converged", False):
                total = cr.get("total", cr.get("resultant", {}))
                eta_r = engine.compute_eta(
                    total.get("N", 0),
                    total.get("Mx", 0),
                    total.get("My", 0))
                cr["eta"] = eta_r["eta"]
                print(f"  η = {eta_r['eta']:.4f}"
                      f"  inside={eta_r['demand_inside']}")

    # ==============================================================
    #  3. Export
    # ==============================================================
    from .output.export import export_analysis_json
    export_analysis_json(
        demand_results, combination_results,
        os.path.join(outdir, "analysis_results.json"))
    print(f"\n  Exported: {outdir}/analysis_results.json")

    print("\nDone.")


def _print_analysis_table(title, results):
    r"""Print force decomposition summary for analyzed demands."""
    if not results:
        return

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    for r in results:
        name = r.get("name", "unnamed")
        conv = r.get("converged", False)
        sok = r.get("strains_ok", False)
        if not conv:
            status = "NO CONV"
        elif not sok:
            status = "WARN"
        else:
            status = "OK"

        total = r.get("total", {})
        print(f"\n  {name}  (N={total.get('N',0)/1e3:.1f} kN,"
              f" Mx={total.get('Mx',0)/1e6:.2f} kNm,"
              f" My={total.get('My',0)/1e6:.2f} kNm)"
              f"  [{status}]")

        ss = r.get("strain_state", {})
        print(f"    ε₀ = {ss.get('eps0',0)*1e3:.4f} ‰"
              f"   χx = {ss.get('chi_x',0)*1e6:.2f} 1/km"
              f"   χy = {ss.get('chi_y',0)*1e6:.2f} 1/km")

        for comp in r.get("components", []):
            ctype = comp["type"]
            mname = comp["material_name"]
            if ctype == "bulk":
                zone = comp.get("zone", 0)
                print(f"    [{ctype}  zone={zone}] {mname:>16s}"
                      f"  N={comp['N_kN']:>10.2f} kN"
                      f"  Mx={comp['Mx_kNm']:>10.4f} kNm"
                      f"  My={comp['My_kNm']:>10.4f} kNm")
            else:
                print(f"    [{ctype}]          {mname:>16s}"
                      f"  N={comp['N_kN']:>10.2f} kN"
                      f"  Mx={comp['Mx_kNm']:>10.4f} kNm"
                      f"  My={comp['My_kNm']:>10.4f} kNm")
                for lay in comp.get("layers", []):
                    print(f"      bar {lay['index']+1:>2d}"
                          f"  y={lay['y']:>7.1f}"
                          f"  ε={lay['eps']*1e3:>8.3f}‰"
                          f"  σ_net={lay['sigma_net']:>8.2f} MPa"
                          f"  F={lay['F_net_kN']:>8.2f} kN")

    print(f"\n{'=' * 80}")


def _print_staged_analysis(cr):
    r"""Print staged combination analysis summary."""
    name = cr["name"]
    res = cr.get("resultant", {})
    print(f"\n{'=' * 80}")
    print(f"  FORCE DECOMPOSITION — {name} (staged)")
    print(f"  Resultant: N={res.get('N_kN',0):.1f} kN,"
          f" Mx={res.get('Mx_kNm',0):.2f} kNm,"
          f" My={res.get('My_kNm',0):.2f} kNm")
    print(f"{'=' * 80}")

    for stage in cr.get("stages", []):
        sname = stage.get("name", "?")
        cum = stage.get("cumulative", {})
        conv = stage.get("converged", False)
        print(f"\n  Stage: {sname}  (cum N={cum.get('N_kN',0):.1f},"
              f" Mx={cum.get('Mx_kNm',0):.2f},"
              f" My={cum.get('My_kNm',0):.2f})"
              f"  [{'OK' if conv else 'NO CONV'}]")

        for comp in stage.get("components", []):
            ctype = comp["type"]
            mname = comp["material_name"]
            print(f"    [{ctype:>5s}] {mname:>16s}"
                  f"  N={comp['N_kN']:>10.2f} kN"
                  f"  Mx={comp['Mx_kNm']:>10.4f} kNm")

    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()