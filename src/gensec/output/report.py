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
Text reporting utilities.

All rebar layers are numbered sequentially (1-based) and this
numbering is consistent across terminal output, plots, and CSV
exports.

Supports both :class:`GenericSection` and legacy :class:`RectSection`.
"""


def print_section_info(sec):
    r"""
    Print a formatted summary of section properties.

    Adapts to the section type: shows bounding box dimensions,
    ideal_gross area, mesh method and quality for generic sections;
    shows B x H and grid resolution for rectangular sections.

    Parameters
    ----------
    sec : GenericSection or RectSection
    """
    print("=" * 70)
    print("CROSS-SECTION SUMMARY")
    print("=" * 70)

    # Bounding box
    print(f"  Bounding box: {sec.B:.0f} x {sec.H:.0f} mm")

    # ideal_gross area
    ideal_gross = getattr(sec, 'ideal_gross_area', sec.B * sec.H)
    print(f"  ideal_gross area: {ideal_gross:.1f} mm²")

    # Mesh info
    mesh_summary = getattr(sec, 'mesh_summary', None)
    if mesh_summary is not None:
        ms = mesh_summary()
        print(f"  Mesh: {ms['mesh_method']} @ {ms['mesh_size']:.1f} mm"
              f"  ({ms['n_fibers']} fibers)")
        print(f"  Area coverage: {ms['total_area']:.1f} mm²"
              f"  (error: {ms['area_error_pct']:.2f}%)")
    else:
        nx = getattr(sec, 'n_fibers_x', 1)
        ny = getattr(sec, 'n_fibers_y', 1)
        print(f"  Bulk fibers: {sec.n_fibers}"
              f" ({nx}x{ny})")

    # Bulk material
    b = sec.bulk_material
    print(f"\n  Bulk: {type(b).__name__}"
          f"  eps=[{b.eps_min:.5f}, {b.eps_max:.5f}]")
    if hasattr(b, 'fcd'):
        print(f"    fcd = {b.fcd:.2f} MPa")

    # Multi-material zones
    bulk_mats = getattr(sec, 'bulk_materials', [])
    if bulk_mats:
        print(f"\n  Material zones: {len(bulk_mats)}")
        for i, (_, mat) in enumerate(bulk_mats):
            mn = getattr(mat, 'name', type(mat).__name__)
            print(f"    Zone {i+1}: {mn}"
                  f"  eps=[{mat.eps_min:.5f}, {mat.eps_max:.5f}]")

    # Rebars table
    print(f"\n  {'#':>3} {'x':>7} {'y':>7} {'dia':>5} {'As':>8}"
          f" {'mat':>10} {'emb':>5}")
    print("  " + "-" * 50)
    for i, r in enumerate(sec.rebars):
        mn = getattr(r.material, 'name', type(r.material).__name__)
        x_str = f"{r.x:.1f}" if r.x is not None else "-"
        d_str = f"{r.diameter:.0f}" if r.diameter > 0 else "-"
        emb_str = "yes" if r.embedded else "no"
        print(f"  {i+1:>3} {x_str:>7} {r.y:>7.1f} {d_str:>5}"
              f" {r.As:>8.1f} {mn:>10} {emb_str:>5}")

    As_tot = sum(r.As for r in sec.rebars)
    rho = As_tot / ideal_gross * 100 if ideal_gross > 0 else 0
    print(f"\n  As,tot = {As_tot:.1f} mm²  (rho = {rho:.2f}%)")
    print("=" * 70)


def print_fiber_results(results, sec):
    r"""
    Print strain and stress at rebars and sampled bulk fibers.

    Parameters
    ----------
    results : dict
        Output of :meth:`FiberSolver.get_fiber_results`.
    sec : GenericSection or RectSection
    """
    import numpy as np

    print("\n" + "=" * 70)
    print("STRAIN AND STRESS STATE")
    print("=" * 70)
    sr = results["rebars"]

    has_x = "x" in sr and len(sr["x"]) > 0
    if has_x:
        header = (f"  {'#':>3} {'x':>7} {'y':>7} {'eps[permil]':>11}"
                  f" {'sig[MPa]':>10} {'sig_net':>8} {'F_net[kN]':>10}")
    else:
        header = (f"  {'#':>3} {'y':>7} {'eps[permil]':>11}"
                  f" {'sig[MPa]':>10} {'sig_net':>8} {'F_net[kN]':>10}")
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    for i in range(len(sr["y"])):
        sig_net = sr["sigma_net"][i] if "sigma_net" in sr else sr["sigma"][i]
        F_net = sig_net * sr["A"][i] / 1e3
        if has_x:
            print(f"  {i+1:>3} {sr['x'][i]:>7.1f} {sr['y'][i]:>7.1f}"
                  f" {sr['eps'][i]*1e3:>11.3f}"
                  f" {sr['sigma'][i]:>10.2f}"
                  f" {sig_net:>8.2f}"
                  f" {F_net:>10.2f}")
        else:
            print(f"  {i+1:>3} {sr['y'][i]:>7.1f}"
                  f" {sr['eps'][i]*1e3:>11.3f}"
                  f" {sr['sigma'][i]:>10.2f}"
                  f" {sig_net:>8.2f}"
                  f" {F_net:>10.2f}")

    # Bulk summary: sample at 5 y-levels
    cr = results["bulk"]
    by = cr["y"]
    y_unique = np.unique(by)
    n_levels = len(y_unique)
    print(f"\n  Bulk (sampled at 5 levels):")
    for idx in [0, n_levels // 4, n_levels // 2,
                3 * n_levels // 4, n_levels - 1]:
        y_val = y_unique[idx]
        mask = by == y_val
        eps_mean = cr["eps"][mask].mean()
        sig_mean = cr["sigma"][mask].mean()
        print(f"    y={y_val:>7.1f}"
              f"  eps={eps_mean*1e3:>8.3f} permil"
              f"  sig={sig_mean:>8.2f} MPa")
