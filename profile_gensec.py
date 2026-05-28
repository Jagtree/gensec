# profile_gensec.py
"""
Focused GenSec profiler. Times each solver stage in isolation,
then runs cProfile on whichever stage dominated.

Usage:
    python profile_gensec.py path/to/input.yaml
    python profile_gensec.py path/to/input.yaml --deep
    python profile_gensec.py path/to/input.yaml --stage biaxial
"""
import argparse
import cProfile
import pstats
import time
import os
import sys

# Matplotlib completely off for timing (no figure allocations).
os.environ["MPLBACKEND"] = "Agg"

from gensec.io_yaml import load_yaml
from gensec.solver import FiberSolver, NMDiagram
from gensec.solver.check import VerificationEngine


def setup(yaml_path, n_points):
    """Load YAML and build solver + NMDiagram. Returns a dict of
    reusable objects. Timing starts after this."""
    data = load_yaml(yaml_path)
    section = data["section"]
    solver = FiberSolver(section)
    nm_gen = NMDiagram(solver)
    return {
        "data": data,
        "section": section,
        "solver": solver,
        "nm_gen": nm_gen,
        "is_biaxial": section.n_fibers_x > 1,
        "n_points": n_points,
    }


def time_stage(label, fn):
    """Run fn(), print elapsed, return result."""
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    print(f"  [{dt:7.2f} s]  {label}")
    return result, dt


def run_all_stages(ctx):
    """Run every solver stage independently, accumulating timings."""
    nm_gen = ctx["nm_gen"]
    n_points = ctx["n_points"]
    is_biaxial = ctx["is_biaxial"]
    timings = {}

    print(f"\nSection: n_fibers = {ctx['section'].n_fibers}, "
          f"biaxial = {is_biaxial}")
    print(f"n_points = {n_points}\n")

    # 1. Uniaxial N-Mx
    nm_x, t = time_stage(
        "generate()              N-Mx diagram",
        lambda: nm_gen.generate(n_points=n_points, direction='x'))
    timings["nm_x"] = t

    # 2. Uniaxial N-My (biaxial only)
    if is_biaxial:
        _, t = time_stage(
            "generate(direction=y)  N-My diagram",
            lambda: nm_gen.generate(n_points=n_points, direction='y'))
        timings["nm_y"] = t

    # 3. 3D biaxial surface
    nm_3d = None
    if is_biaxial:
        nm_3d, t = time_stage(
            "generate_biaxial()     3D surface (72 angles)",
            lambda: nm_gen.generate_biaxial(
                n_angles=72, n_points_per_angle=n_points // 2))
        timings["biaxial"] = t

    # 4. Mx-My contours at 3 representative N levels
    N_levels = [0.0]
    if is_biaxial and nm_3d is not None:
        N_min, N_max = float(nm_3d["N"].min()), float(nm_3d["N"].max())
        N_levels = [0.3 * N_min, 0.0, 0.3 * N_max]

        def _all_contours():
            for N in N_levels:
                nm_gen.generate_mx_my(
                    N, n_angles=144, n_points_per_angle=n_points // 2)
        _, t = time_stage(
            f"generate_mx_my() x {len(N_levels)} N levels",
            _all_contours)
        timings["mx_my"] = t

    # 5. Moment-curvature at 3 N levels (vectorized via _scan_chi_vectorized)
    def _all_mc():
        for N in N_levels:
            nm_gen.generate_moment_curvature(
                N, n_chi=n_points, direction='x')
    _, t = time_stage(
        f"generate_moment_curvature() x {len(N_levels)} N levels",
        _all_mc)
    timings["mc"] = t

    # 6. Verification (generates Mx-My contour per demand; separate cache from stage 4)
    demands = ctx["data"].get("demands", [])
    if demands:
        engine = VerificationEngine(
            nm_3d if nm_3d is not None else nm_x,
            nm_gen, ctx["data"].get("output_options", {}),
            n_points=n_points // 2)
        _, t = time_stage(
            f"VerificationEngine.check_demands() ({len(demands)} demands)",
            lambda: engine.check_demands(demands))
        timings["verify"] = t

    print("\n" + "-" * 60)
    total = sum(timings.values())
    print(f"  TOTAL: {total:.2f} s")
    print("  Breakdown (% of total):")
    for k, v in sorted(timings.items(), key=lambda x: -x[1]):
        print(f"    {k:12s} {v:7.2f} s   {100*v/total:5.1f}%")
    return timings


def deep_profile_stage(ctx, stage):
    """cProfile on a single stage. Prints top 30 by cumulative time."""
    nm_gen = ctx["nm_gen"]
    n_points = ctx["n_points"]

    stages = {
        "biaxial": lambda: nm_gen.generate_biaxial(
            n_angles=72, n_points_per_angle=n_points // 2),
        "mx_my": lambda: nm_gen.generate_mx_my(
            0.0, n_angles=144, n_points_per_angle=n_points // 2),
        "mc": lambda: nm_gen.generate_moment_curvature(
            0.0, n_chi=n_points, direction='x'),
        "nm_x": lambda: nm_gen.generate(
            n_points=n_points, direction='x'),
    }
    if stage not in stages:
        print(f"Unknown stage '{stage}'. Choose: {list(stages)}")
        sys.exit(1)

    print(f"\ncProfile: {stage}")
    print("=" * 60)
    pr = cProfile.Profile()
    pr.enable()
    stages[stage]()
    pr.disable()

    st = pstats.Stats(pr).sort_stats("cumulative")
    st.print_stats(30)

    # Also save for snakeviz
    out = f"gensec_{stage}.prof"
    pr.dump_stats(out)
    print(f"\nSaved: {out}")
    print(f"Visualize with:  snakeviz {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_path")
    ap.add_argument("--n-points", type=int, default=50)
    ap.add_argument("--deep", action="store_true",
                    help="Run cProfile on the heaviest stage.")
    ap.add_argument("--stage", default=None,
                    help="Force cProfile on a specific stage "
                         "(biaxial|mx_my|mc|nm_x).")
    args = ap.parse_args()

    ctx = setup(args.yaml_path, args.n_points)

    # Warm-up: one small call to trigger Numba JIT before timing
    ctx["solver"].integrate(0.0, 1e-6, 0.0)
    ctx["solver"].integrate_batch(
        [0.0, -1e-3], [1e-6, 1e-6], [0.0, 0.0])

    timings = run_all_stages(ctx)

    if args.deep or args.stage:
        worst = args.stage or max(timings, key=timings.get)
        deep_profile_stage(ctx, worst)