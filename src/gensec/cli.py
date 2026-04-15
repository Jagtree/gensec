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

from .io_yaml import load_yaml
from .solver import FiberSolver, NMDiagram
from .solver.check import VerificationEngine
from .output import (
    print_section_info, print_fiber_results,
    plot_nm_diagram, plot_stress_profile, plot_mx_my_diagram,
    plot_moment_curvature, plot_section, plot_section_state,
    plot_demand_heatmap, plot_3d_surface,
    plot_moment_curvature_bundle, plot_polar_ductility,
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


# ==================================================================
#  Verification table printers
# ==================================================================

def _eta_columns(results):
    """Detect which eta columns are present in results."""
    cols = []
    for key in ("eta_3D", "eta_2D"):
        if any(key in r for r in results):
            cols.append(key)
    return cols


def _print_demand_table(title, results):
    """
    Print a formatted demand verification table.

    Adapts columns to whichever eta types are present.

    Parameters
    ----------
    title : str
    results : list of dict
    """
    if not results:
        return

    eta_cols = _eta_columns(results)
    has_my = any(r.get("My_kNm", 0) != 0 for r in results)

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    # Build header.
    hdr = f"  {'Name':>20} {'N[kN]':>8} {'Mx[kNm]':>9}"
    if has_my:
        hdr += f" {'My[kNm]':>9}"
    for ec in eta_cols:
        hdr += f" {ec:>8}"
    hdr += f" {'Status':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    all_ok = True
    for r in results:
        status = "OK" if r["verified"] else "FAIL"
        if not r["verified"]:
            all_ok = False
        line = (f"  {r['name']:>20} {r['N_kN']:>8.1f}"
                f" {r['Mx_kNm']:>9.2f}")
        if has_my:
            line += f" {r.get('My_kNm', 0):>9.2f}"
        for ec in eta_cols:
            val = r.get(ec)
            if val is not None:
                line += f" {val:>8.4f}"
            else:
                line += f" {'---':>8}"
        line += f" {status:>8}"
        print(line)

    print("  " + "-" * (len(hdr) - 2))
    if all_ok:
        print("  All demands VERIFIED.")
    else:
        n_fail = sum(1 for r in results if not r["verified"])
        print(f"  *** {n_fail} demand(s) NOT VERIFIED ***")
    print("=" * 80)


def _print_combination_table(title, combo_results):
    """
    Print combination verification summary.

    For staged combinations, prints a sub-table per stage.

    Parameters
    ----------
    title : str
    combo_results : list of dict
    """
    if not combo_results:
        return

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    for cr in combo_results:
        name = cr["name"]
        ctype = cr.get("type", "simple")
        res = cr.get("resultant", {})
        eta_gov = cr.get("eta_governing", cr.get("eta_3D", "---"))
        status = "OK" if cr.get("verified", False) else "FAIL"

        print(f"\n  -- {name} ({ctype}) --")
        print(f"     Resultant: N={res.get('N_kN',0):.1f} kN,"
              f" Mx={res.get('Mx_kNm',0):.2f} kNm,"
              f" My={res.get('My_kNm',0):.2f} kNm")

        if "stages" in cr:
            # Detect which etas appear in stages.
            stage_etas = []
            for k in ("eta_3D", "eta_2D", "eta_path", "eta_path_2D"):
                if any(k in s and s[k] is not None
                       for s in cr["stages"]):
                    stage_etas.append(k)

            hdr = (f"     {'Stage':>20} {'N[kN]':>8} {'Mx':>9}"
                   f" {'My':>9}")
            for se in stage_etas:
                hdr += f" {se:>12}"
            print(hdr)
            print("     " + "-" * (len(hdr) - 5))

            for s in cr["stages"]:
                cum = s.get("cumulative", {})
                line = (f"     {s['name']:>20}"
                        f" {cum.get('N_kN',0):>8.1f}"
                        f" {cum.get('Mx_kNm',0):>9.2f}"
                        f" {cum.get('My_kNm',0):>9.2f}")
                for se in stage_etas:
                    val = s.get(se)
                    if val is not None:
                        line += f" {val:>12.4f}"
                    else:
                        line += f" {'---':>12}"
                print(line)
                if "warning" in s:
                    print(f"       WARNING: {s['warning']}")

        print(f"     eta_governing = {eta_gov}  ->  {status}")

    print(f"\n{'=' * 80}")


def _print_envelope_table(title, envelope_results):
    """
    Print envelope verification summary.

    Parameters
    ----------
    title : str
    envelope_results : list of dict
    """
    if not envelope_results:
        return

    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"  {'Envelope':>25} {'eta_max':>8} {'Governing':>25}"
          f" {'Status':>8}")
    print("  " + "-" * 70)

    for er in envelope_results:
        status = "OK" if er["verified"] else "FAIL"
        print(f"  {er['name']:>25} {er['eta_max']:>8.4f}"
              f" {er['governing_member']:>25} {status:>8}")

    print("=" * 80)


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
    p_run.add_argument("--n-points", type=int, default=400,
                       help="N-M diagram resolution (default: 400).")
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

    args = parser.parse_args(argv)

    if args.command == "run":
        _run(args)
    elif args.command == "plot":
        _plot(args)
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

    # Output flags (v2.1, with defaults applied by io_yaml).
    output_opts = data.get("output_options", {})
    do_mx_my = output_opts.get("generate_mx_my", False)
    do_3d_surface = output_opts.get("generate_3d_surface", False)

    # N-level strategy.
    n_levels_mode = output_opts.get("n_levels_mode", "demands")
    n_levels_count = int(output_opts.get("n_levels_count", 10))
    n_levels_explicit = output_opts.get("n_levels_values", [])
    n_angles_mx_my = int(output_opts.get("n_angles_mx_my", 144))

    print_section_info(section)

    # --- Solver ---
    solver = FiberSolver(section)
    is_biaxial = section.n_fibers_x > 1

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
            n_angles=72, n_points_per_angle=args.n_points // 2)
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
    engine = VerificationEngine(
        domain_data, nm_gen, output_opts,
        n_points=args.n_points // 2)

    # Build demand database for combination / envelope resolution.
    demand_db = {d["name"]: d for d in demands}

    # --- 1. Demand verification ---
    demand_results = []
    if demands:
        demand_results = engine.check_demands(demands)
        _print_demand_table("DEMAND VERIFICATION", demand_results)

        export_demand_results_csv(
            demand_results,
            os.path.join(outdir, "demand_summary.csv"))
        export_demand_results_json(
            demand_results,
            os.path.join(outdir, "demand_summary.json"))

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

        _print_combination_table("COMBINATION VERIFICATION",
                                 combination_results)

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

        _print_envelope_table("ENVELOPE VERIFICATION",
                              envelope_results)

        export_envelope_results_json(
            envelope_results,
            os.path.join(outdir, "envelope_summary.json"))

    # --- Unified verification export ---
    if demand_results or combination_results or envelope_results:
        export_verification_json(
            demand_results, combination_results, envelope_results,
            os.path.join(outdir, "verification_summary.json"))

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

        print("\n  Per-demand details:")
        for d in demands:
            N_d, Mx_d, My_d = d["N"], d["Mx"], d["My"]
            sol = solver.solve_equilibrium(N_d, Mx_d, My_d)
            label = d["name"]
            demand_plot.append((N_d / 1e3, Mx_d / 1e6, label))

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
        fig_hm = plot_demand_heatmap(demand_results)
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
    if n_levels_mode == "explicit" and n_levels_explicit:
        N_levels_analysis = sorted(
            float(v) * 1e3 for v in n_levels_explicit)
        print(f"\n  N levels (explicit): "
              f"{[v/1e3 for v in N_levels_analysis]} kN")
    elif n_levels_mode == "auto":
        N_min_d = float(nm_data["N"].min())
        N_max_d = float(nm_data["N"].max())
        N_levels_analysis = sorted(
            np.linspace(N_min_d, N_max_d, n_levels_count).tolist())
        print(f"\n  N levels (auto, {n_levels_count} values): "
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
                N_fixed, n_angles=n_angles_mx_my,
                n_points_per_angle=args.n_points // 2)
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
    for N_fixed in N_levels_analysis:
        N_label = f"{N_fixed/1e3:.0f}".replace("-", "m")

        print(f"  Generating Mx-chi at N={N_fixed/1e3:.0f} kN...")
        mc_x = nm_gen.generate_moment_curvature(
            N_fixed, n_points=args.n_points, direction='x')
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
                N_fixed, n_points=args.n_points, direction='y')
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

    if is_biaxial:
        for N_fixed in N_levels_analysis:
            N_label = f"{N_fixed/1e3:.0f}".replace("-", "m")
            print(f"  Generating polar ductility at"
                  f" N={N_fixed/1e3:.0f} kN...")
            fig = plot_polar_ductility(
                nm_gen, N_fixed, n_angles=n_angles_mx_my,
                n_points=args.n_points)
            p = os.path.join(
                outdir, f"polar_ductility_N{N_label}.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"    Exported: {p}")

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


if __name__ == "__main__":
    main()
