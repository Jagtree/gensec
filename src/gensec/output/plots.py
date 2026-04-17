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

r"""
Plotting utilities for N-M diagrams, section state maps, and demand
analysis.

All section state plots (strain, stress) are drawn as coloured maps
on the actual section geometry. Rebar values are annotated directly
on the section, replacing numbered labels.

Supports both rectangular and arbitrary polygon sections via the
``polygon`` attribute from :class:`GenericSection`.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Circle, Rectangle
from matplotlib.path import Path
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import ConvexHull


# ==================================================================
#  Section outline helper (used by all section-based plots)
# ==================================================================

def _draw_polygon_outline(ax, sec, **kwargs):
    r"""
    Draw the section outline as a matplotlib path patch.

    Correctly renders holes (interior rings) using compound paths.
    Falls back to a rectangle if no ``polygon`` attribute exists.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    sec : GenericSection or RectSection
    **kwargs
        Overrides for :class:`~matplotlib.patches.PathPatch`
        (``facecolor``, ``edgecolor``, ``linewidth``, …).
    """
    poly = getattr(sec, 'polygon', None)

    if poly is not None:
        verts = []
        codes = []

        # Exterior ring — must be CCW for matplotlib even-odd fill.
        # Shapely guarantees CCW for exteriors.
        ext = np.array(poly.exterior.coords)
        n_ext = len(ext)
        verts.extend(ext.tolist())
        codes.append(Path.MOVETO)
        codes.extend([Path.LINETO] * (n_ext - 2))
        codes.append(Path.CLOSEPOLY)

        # Interior rings (holes) — must be CW (opposite to exterior)
        # for the even-odd fill rule to carve them out.
        for interior in poly.interiors:
            ring = np.array(interior.coords)
            # Ensure CW winding: if signed area > 0, ring is CCW → reverse
            signed_area = np.sum(
                ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1]
            ) / 2
            if signed_area > 0:
                ring = ring[::-1]
            n_ring = len(ring)
            verts.extend(ring.tolist())
            codes.append(Path.MOVETO)
            codes.extend([Path.LINETO] * (n_ring - 2))
            codes.append(Path.CLOSEPOLY)

        path = Path(verts, codes)
        defaults = dict(linewidth=2, edgecolor='black',
                        facecolor='#E8E8E8')
        defaults.update(kwargs)
        patch = PathPatch(path, **defaults)
        ax.add_patch(patch)
    else:
        defaults = dict(linewidth=2, edgecolor='black',
                        facecolor='#E8E8E8')
        defaults.update(kwargs)
        rect = Rectangle((0, 0), sec.B, sec.H, **defaults)
        ax.add_patch(rect)


def _section_axis_limits(ax, sec):
    r"""
    Set axis limits from section bounding box with margins.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    sec : GenericSection or RectSection
    """
    bounds = getattr(sec, '_bounds', (0, 0, sec.B, sec.H))
    minx, miny, maxx, maxy = bounds
    mx = (maxx - minx) * 0.12
    my = (maxy - miny) * 0.12
    ax.set_xlim(minx - mx, maxx + mx)
    ax.set_ylim(miny - my, maxy + my)
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')


def _draw_neutral_axis(ax, sec, fiber_results):
    r"""
    Draw the neutral axis line, extending symmetrically beyond the
    section boundary on both sides.

    For uniaxial sections, draws a horizontal line. For biaxial,
    fits the strain plane and draws the :math:`\varepsilon = 0` line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    sec : GenericSection or RectSection
    fiber_results : dict
    """
    b = fiber_results["bulk"]
    bx, by, be = b["x"], b["y"], b["eps"]
    bounds = getattr(sec, '_bounds', (0, 0, sec.B, sec.H))
    minx, miny, maxx, maxy = bounds

    # Extension beyond bounding box for visual clarity
    ext = max(maxx - minx, maxy - miny) * 0.15

    is_2d = len(np.unique(bx)) > 1

    if is_2d:
        # Fit plane eps = a0 + a1*x + a2*y
        A = np.column_stack([np.ones(len(bx)), bx, by])
        coeffs, _, _, _ = np.linalg.lstsq(A, be, rcond=None)
        a0, a1, a2 = coeffs

        # Find intersections of eps=0 line with an enlarged bbox
        na_pts = []
        for x_test in [minx - ext, maxx + ext]:
            if abs(a2) > 1e-15:
                y_na = -(a0 + a1 * x_test) / a2
                na_pts.append((x_test, y_na))
        for y_test in [miny - ext, maxy + ext]:
            if abs(a1) > 1e-15:
                x_na = -(a0 + a2 * y_test) / a1
                na_pts.append((x_na, y_test))

        # Keep only points within the enlarged view
        view_minx, view_maxx = minx - ext, maxx + ext
        view_miny, view_maxy = miny - ext, maxy + ext
        na_pts = [(x, y) for x, y in na_pts
                  if view_minx <= x <= view_maxx
                  and view_miny <= y <= view_maxy]

        if len(na_pts) >= 2:
            na_pts.sort()
            ax.plot([na_pts[0][0], na_pts[-1][0]],
                    [na_pts[0][1], na_pts[-1][1]],
                    'r--', lw=2.5, label='Neutral axis', zorder=10)
    else:
        # Uniaxial: find y where eps crosses zero
        y_sorted = np.argsort(by)
        ys, es = by[y_sorted], be[y_sorted]
        crossings = np.where(np.diff(np.sign(es)))[0]
        if len(crossings) > 0:
            ic = crossings[0]
            y_na = ys[ic] - es[ic] * (ys[ic + 1] - ys[ic]) / (
                es[ic + 1] - es[ic])
            ax.plot([minx - ext, maxx + ext], [y_na, y_na],
                    'r--', lw=2.5, label=f'NA y={y_na:.1f} mm',
                    zorder=10)


def _make_color_norm_and_cmap(values):
    r"""
    Choose colour map and normalisation adapted to the data range.

    Strategy:

    - **Monopolar** (all values same sign, or the minor side is
      < 5% of the range): sequential colourmap. Blue for compression
      (negative), red/orange for tension (positive).
    - **Bipolar** (significant data on both sides of zero):
      diverging colourmap centred on zero, with asymmetric limits.
    - **Flat field**: neutral grey.

    Parameters
    ----------
    values : numpy.ndarray

    Returns
    -------
    norm : matplotlib.colors.Normalize or TwoSlopeNorm
    cmap : str
        Matplotlib colourmap name.
    """
    vmin, vmax = float(values.min()), float(values.max())
    span = vmax - vmin

    # Guard against flat fields
    if span < 1e-12:
        return plt.Normalize(vmin=vmin - 1, vmax=vmax + 1), 'Greys'

    # Determine polarity balance
    if vmin >= 0:
        # All positive (tension) → sequential warm
        return plt.Normalize(vmin=0, vmax=vmax), 'Reds'
    if vmax <= 0:
        # All negative (compression) → sequential cool, reversed
        # so that most compressed = darkest blue
        return plt.Normalize(vmin=vmin, vmax=0), 'Blues_r'

    # Mixed sign: check if one side is negligible (< 5% of range)
    pos_frac = vmax / span
    neg_frac = abs(vmin) / span

    if pos_frac < 0.05:
        # Almost all compression with a tiny positive tail
        return plt.Normalize(vmin=vmin, vmax=0), 'Blues_r'
    if neg_frac < 0.05:
        # Almost all tension with a tiny negative tail
        return plt.Normalize(vmin=0, vmax=vmax), 'Reds'

    # Genuinely bipolar → diverging, centred on zero
    return TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax), 'RdBu_r'


def _draw_reference_axes(ax, sec):
    r"""
    Draw X and Y reference axes at the section centroid.

    Short arrows with labels to clarify the sign convention for
    moments and curvatures.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    sec : GenericSection or RectSection
    """
    cx = sec.x_centroid
    cy = sec.y_centroid
    B, H = sec.B, sec.H
    arrow_len = min(B, H) * 0.15

    arrow_kw = dict(
        arrowstyle='->', color='black', lw=1.8,
        mutation_scale=15,
    )
    from matplotlib.patches import FancyArrowPatch

    # X-axis arrow
    ax.annotate('', xy=(cx + arrow_len, cy),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.8))
    ax.text(cx + arrow_len * 1.15, cy, 'x', fontsize=10,
            fontweight='bold', va='center')

    # Y-axis arrow
    ax.annotate('', xy=(cx, cy + arrow_len),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.8))
    ax.text(cx, cy + arrow_len * 1.15, 'y', fontsize=10,
            fontweight='bold', ha='center')

    # Small dot at centroid
    ax.plot(cx, cy, 'k+', ms=10, mew=1.5, zorder=10)


# ==================================================================
#  Section state map — strain or stress on the section
# ==================================================================

def plot_section_state(sec, fiber_results, field='eps', title=""):
    r"""
    Draw the section with a colour-mapped field (strain or stress).

    Each bulk fiber is drawn as a coloured scatter point. Rebar
    values are annotated directly on the section instead of
    sequential numbers. The neutral axis is drawn extending beyond
    the section on both sides.

    Parameters
    ----------
    sec : GenericSection or RectSection
    fiber_results : dict
        Output of :meth:`FiberSolver.get_fiber_results`.
    field : ``'eps'`` or ``'sigma'``
        Which field to plot. Default ``'eps'``.
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Section outline (with holes)
    _draw_polygon_outline(ax, sec, facecolor='#F5F5F5',
                          edgecolor='black', linewidth=2)

    # Bulk fibers
    bulk = fiber_results["bulk"]
    bx, by = bulk["x"], bulk["y"]
    if field == 'eps':
        bval = bulk["eps"] * 1000  # permil
        unit = "ε [‰]"
    else:
        bval = bulk["sigma"]
        unit = "σ [MPa]"

    norm, cmap = _make_color_norm_and_cmap(bval)
    sc = ax.scatter(bx, by, c=bval, cmap=cmap, norm=norm,
                    s=max(3, 10000 / max(len(bx), 1)),
                    marker='s', edgecolors='none', zorder=2)
    plt.colorbar(sc, ax=ax, label=unit, shrink=0.8)

    # Rebars: draw circle + annotate value (not number)
    rb = fiber_results["rebars"]
    if len(rb["y"]) > 0:
        if field == 'eps':
            rval = rb["eps"] * 1000
        else:
            rval = rb["sigma_net"] if "sigma_net" in rb else rb["sigma"]

        # Colour rebars with same norm as bulk
        ax.scatter(rb["x"], rb["y"], c=rval, cmap=cmap, norm=norm,
                   s=100, marker='o', edgecolors='black',
                   linewidths=1.5, zorder=5)

        for i in range(len(rb["y"])):
            txt = f"{i+1}: {rval[i]:.1f}"
            ax.annotate(txt, (rb["x"][i], rb["y"][i]),
                        xytext=(12, -4), textcoords="offset points",
                        fontsize=7, fontweight='bold',
                        color='black', zorder=6,
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.8,
                                  edgecolor='none'))

    # Neutral axis
    _draw_neutral_axis(ax, sec, fiber_results)

    # Reference axes at centroid
    _draw_reference_axes(ax, sec)

    _section_axis_limits(ax, sec)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.15)
    ax.set_title(title or f"Section — {unit}")
    fig.tight_layout()
    return fig


# ==================================================================
#  Section geometry (no results — just outline + numbered rebars)
# ==================================================================

def plot_section(sec, fiber_results=None, title=""):
    r"""
    Draw the cross-section geometry with rebars.

    If ``fiber_results`` are provided, draws a two-panel figure:
    left = geometry with numbered rebars + neutral axis,
    right = stress field.

    Parameters
    ----------
    sec : GenericSection or RectSection
    fiber_results : dict, optional
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if fiber_results is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        _draw_geometry_panel(ax1, sec, fiber_results)
        _draw_field_panel(ax2, sec, fiber_results, field='sigma')
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 8))
        _draw_geometry_panel(ax1, sec, fiber_results=None)

    fig.suptitle(title or "Cross-Section", fontsize=14)
    fig.tight_layout()
    return fig


def _draw_geometry_panel(ax, sec, fiber_results=None):
    """
    Draw section outline with numbered rebars and optional
    neutral axis.
    """
    _draw_polygon_outline(ax, sec)

    for i, rb in enumerate(sec.rebars):
        d = rb.diameter if rb.diameter > 0 else 16
        r_vis = d / 2
        color = '#444444' if rb.embedded else '#CC6600'
        c = Circle((rb.x, rb.y), r_vis,
                    facecolor=color, edgecolor='black', lw=0.8,
                    zorder=4)
        ax.add_patch(c)
        ax.annotate(f"{i+1}", (rb.x, rb.y),
                    xytext=(12, -4), textcoords="offset points",
                    fontsize=7, fontweight='bold',
                    color='black', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.8,
                              edgecolor='none'))

    if fiber_results is not None:
        _draw_neutral_axis(ax, sec, fiber_results)
        ax.legend(fontsize=9, loc='best')
        ax.set_title('Geometry + Neutral Axis')
    else:
        ax.set_title('Geometry')

    _draw_reference_axes(ax, sec)
    _section_axis_limits(ax, sec)
    ax.grid(True, alpha=0.2)


def _draw_field_panel(ax, sec, fiber_results, field='sigma'):
    """
    Draw a coloured field (stress or strain) on the section.
    """
    _draw_polygon_outline(ax, sec, facecolor='none',
                          edgecolor='black', linewidth=1.5)

    bulk = fiber_results["bulk"]
    bx, by = bulk["x"], bulk["y"]
    if field == 'eps':
        bval = bulk["eps"] * 1000
        unit = "ε [‰]"
    else:
        bval = bulk["sigma"]
        unit = "σ [MPa]"

    norm, cmap = _make_color_norm_and_cmap(bval)
    sc = ax.scatter(bx, by, c=bval, cmap=cmap, norm=norm,
                    s=max(2, 8000 / max(len(bx), 1)),
                    marker='s', edgecolors='none')
    plt.colorbar(sc, ax=ax, label=unit, shrink=0.8)

    # Rebar field values
    rb = fiber_results["rebars"]
    if len(rb["x"]) > 0:
        if field == 'eps':
            rval = rb["eps"] * 1000
        else:
            rval = rb["sigma_net"] if "sigma_net" in rb else rb["sigma"]

        ax.scatter(rb["x"], rb["y"], c=rval, cmap=cmap, norm=norm,
                   s=80, marker='o', edgecolors='black',
                   linewidths=1.2, zorder=5)
        for i in range(len(rb["x"])):
            ax.annotate(f"{i+1}", (rb["x"][i], rb["y"][i]),
                        xytext=(12, -4), textcoords="offset points",
                        fontsize=7, fontweight='bold',
                        color='black', zorder=6,
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.8,
                                  edgecolor='none'))

    _draw_neutral_axis(ax, sec, fiber_results)
    _draw_reference_axes(ax, sec)

    _section_axis_limits(ax, sec)
    ax.set_title(f'Stress Field [{unit}]')


# ==================================================================
#  N-M interaction diagram
# ==================================================================

def plot_nm_diagram(nm_data, demands=None, title=""):
    """
    Plot the N-M interaction diagram with convex hull boundary.

    Parameters
    ----------
    nm_data : dict
    demands : list of tuple, optional
        ``(N_kN, M_kNm, label)`` demand points.
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    pts = np.column_stack([nm_data["M_kNm"], nm_data["N_kN"]])
    try:
        h = ConvexHull(pts)
        hi = np.append(h.vertices, h.vertices[0])
        ax.plot(pts[hi, 0], pts[hi, 1], 'b-', lw=1.5,
                label="N-M domain")
        ax.fill(pts[hi, 0], pts[hi, 1], alpha=0.15, color='blue')
    except Exception:
        ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.3)
    if demands:
        for n_d, m_d, lb in demands:
            ax.plot(m_d, n_d, 'ro', ms=8)
            ax.annotate(lb, (m_d, n_d), xytext=(8, 5),
                        textcoords="offset points", fontsize=9)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    direction = nm_data.get("direction", "x")
    M_label = f"M{direction}" if direction else "M"
    ax.set_xlabel(f"{M_label} [kN·m]")
    ax.set_ylabel("N [kN]")
    ax.set_title(title or f"N-{M_label} Interaction Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ==================================================================
#  Mx-My interaction diagram
# ==================================================================

def plot_mx_my_diagram(mx_my_data, demands=None, title=""):
    """
    Plot the Mx-My interaction contour at fixed N.

    Parameters
    ----------
    mx_my_data : dict
    demands : list of tuple, optional
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    Mx = mx_my_data["Mx_kNm"]
    My = mx_my_data["My_kNm"]
    N_fixed = mx_my_data.get("N_fixed_kN", 0)

    pts = np.column_stack([Mx, My])
    try:
        hull = ConvexHull(pts)
        hv = np.append(hull.vertices, hull.vertices[0])
        ax.plot(pts[hv, 0], pts[hv, 1], 'b-', linewidth=1.5,
                label="Mx-My domain")
        ax.fill(pts[hv, 0], pts[hv, 1], alpha=0.15, color='blue')
    except Exception:
        ax.plot(Mx, My, 'b.', ms=3, label="Mx-My domain")

    if demands:
        for mx_d, my_d, lb in demands:
            ax.plot(mx_d, my_d, 'ro', ms=8)
            ax.annotate(lb, (mx_d, my_d), xytext=(8, 5),
                        textcoords="offset points", fontsize=9)

    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel("Mx [kN·m]")
    ax.set_ylabel("My [kN·m]")
    ax.set_title(title or f"Mx-My interaction at N = {N_fixed:.0f} kN")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
              fontsize=9, ncol=3, borderaxespad=0)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    return fig


# ==================================================================
#  Moment-curvature diagram
# ==================================================================

def plot_moment_curvature(mc_data, title=""):
    r"""
    Plot the moment-curvature diagram with cracking, yield and
    ultimate markers and a numerical legend.

    Parameters
    ----------
    mc_data : dict
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    chi = mc_data["chi_km"]
    M = mc_data["M_kNm"]
    N_kN = mc_data["N_fixed_kN"]
    direction = mc_data.get("direction", "x")
    M_label = f"M{direction}" if direction else "M"

    uc_pos = mc_data.get("ultimate_chi_pos")
    uc_neg = mc_data.get("ultimate_chi_neg")
    chi_min_plot = (uc_neg * 1e6 * 1.05) if uc_neg is not None else chi.min()
    chi_max_plot = (uc_pos * 1e6 * 1.05) if uc_pos is not None else chi.max()
    mask = (chi >= chi_min_plot) & (chi <= chi_max_plot)
    if np.sum(mask) < 10:
        mask = np.ones(len(chi), dtype=bool)

    ax.plot(chi[mask], M[mask], 'b-', lw=1.5, label=f"{M_label}-χ")

    # Collect key-point lines for the numerical legend.
    legend_lines = []

    # First cracking markers.
    for suffix, marker in [("_pos", "D"), ("_neg", "D")]:
        cc = mc_data.get(f"cracking_chi{suffix}")
        cm = mc_data.get(f"cracking_M{suffix}")
        if cc is not None and cm is not None:
            ax.plot(cc * 1e6, cm / 1e6, color='#FF8C00',
                    marker=marker, ms=9, zorder=5,
                    label="First cracking" if suffix == "_pos"
                    else None)
            sign = "+" if suffix == "_pos" else "−"
            legend_lines.append(
                f"Cracking ({sign}): {M_label}={cm/1e6:.1f} kNm,"
                f" χ={cc*1e6:.1f} 1/km")

    for suffix, color, marker in [("_pos", "green", "^"),
                                   ("_neg", "green", "v")]:
        yc = mc_data.get(f"yield_chi{suffix}")
        ym = mc_data.get(f"yield_M{suffix}")
        if yc is not None and ym is not None:
            ax.plot(yc * 1e6, ym / 1e6, color=color, marker=marker,
                    ms=12, zorder=5,
                    label="First yield" if suffix == "_pos" else None)
            sign = "+" if suffix == "_pos" else "−"
            legend_lines.append(
                f"Yield ({sign}): {M_label}={ym/1e6:.1f} kNm,"
                f" χ={yc*1e6:.1f} 1/km")

    for suffix, color, marker in [("_pos", "red", "x"),
                                   ("_neg", "red", "x")]:
        uc = mc_data.get(f"ultimate_chi{suffix}")
        um = mc_data.get(f"ultimate_M{suffix}")
        if uc is not None and um is not None:
            ax.plot(uc * 1e6, um / 1e6, color=color, marker=marker,
                    ms=12, mew=3, zorder=5,
                    label="Ultimate" if suffix == "_pos" else None)
            sign = "+" if suffix == "_pos" else "−"
            legend_lines.append(
                f"Ultimate ({sign}): {M_label}={um/1e6:.1f} kNm,"
                f" χ={uc*1e6:.1f} 1/km")

    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel("χ [1/km]")
    ax.set_ylabel(f"{M_label} [kN·m]")
    ax.set_title(title or
                 f"{M_label}-χ diagram at N = {N_kN:.0f} kN")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Numerical legend box (bottom-right).
    if legend_lines:
        txt = "\n".join(legend_lines)
        ax.text(0.98, 0.02, txt, transform=ax.transAxes,
                fontsize=7.5, family='monospace',
                va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='#FAFAFA', alpha=0.9,
                          edgecolor='#CCCCCC'))

    fig.tight_layout()
    return fig


# ==================================================================
#  Demand utilization heatmap
# ==================================================================

def plot_demand_heatmap(check_results, title=""):
    r"""
    Grouped horizontal bar chart of all enabled utilization ratios.

    Each demand gets one bar per :math:`\eta` type present in the
    results.  Bars exceeding 1.0 are coloured red; those within
    the limit are green.

    Parameters
    ----------
    check_results : list of dict
        Output of :meth:`VerificationEngine.check_demands`.
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not check_results:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, "No demands to display",
                ha='center', va='center', transform=ax.transAxes)
        return fig

    # Detect which eta columns are present.
    _all_keys = [("eta_3D", "η_3D", "#1f77b4"),
                 ("eta_2D", "η_2D", "#ff7f0e")]
    eta_cols = [(k, lab, col) for k, lab, col in _all_keys
                if any(k in r and r[k] is not None
                       for r in check_results)]
    if not eta_cols:
        eta_cols = [("eta_3D", "η_3D", "#1f77b4")]

    n_types = len(eta_cols)

    # Sort demands by governing eta (max across all types).
    def _gov_eta(r):
        vals = [r.get(k, 0) or 0 for k, _, _ in eta_cols]
        return max(vals) if vals else 0

    sorted_res = sorted(check_results,
                        key=_gov_eta, reverse=True)
    names = [r["name"] for r in sorted_res]
    n_demands = len(names)

    bar_height = 0.7 / n_types
    fig, ax = plt.subplots(
        1, 1, figsize=(max(8, n_demands * 0.5 + 2),
                       max(5, n_demands * 0.8 + 1)))
    y_pos = np.arange(n_demands)

    for t_idx, (key, label, base_color) in enumerate(eta_cols):
        offsets = y_pos + (t_idx - (n_types - 1) / 2) * bar_height
        vals = []
        colors = []
        for r in sorted_res:
            v = r.get(key)
            v = v if v is not None else 0.0
            vals.append(v)
            colors.append('#F44336' if v > 1.0 else '#4CAF50')

        bars = ax.barh(offsets, vals, height=bar_height * 0.9,
                       color=colors, edgecolor='black',
                       linewidth=0.4, label=label, alpha=0.85)

        for i, (bar, v) in enumerate(zip(bars, vals)):
            if v > 0:
                x_txt = v + 0.02
                ax.text(x_txt, offsets[i], f"{v:.3f}",
                        va='center', fontsize=7.5,
                        fontweight='bold', color='#333333')

    ax.axvline(1.0, color='red', ls='--', lw=2, label='η = 1.0')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Utilization ratio η")
    ax.set_title(title or "Demand Utilization")
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()

    # Type legend in corner.
    type_txt = ", ".join(lab for _, lab, _ in eta_cols)
    ax.text(0.98, 0.02, f"Showing: {type_txt}",
            transform=ax.transAxes, fontsize=7.5, ha='right',
            va='bottom', style='italic', color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.8, edgecolor='#CCCCCC'))

    fig.tight_layout()
    return fig


# ==================================================================
#  3D resistance surface (N, Mx, My)
# ==================================================================

def _hull_slice_at_N(hull_pts, simplices, N_level):
    r"""
    Intersect hull triangles at a fixed *N* level and return
    the outer boundary of the resulting 2-D cross-section.

    Uses a 2-D ConvexHull on the intersection points to guarantee
    a clean, non-self-intersecting contour even when the raw
    intersection set contains interior duplicates.

    Parameters
    ----------
    hull_pts : numpy.ndarray
        ``(n, 3)`` with columns ``[Mx, My, N]``.
    simplices : numpy.ndarray
    N_level : float

    Returns
    -------
    numpy.ndarray or None
        ``(m, 2)`` ordered boundary points ``[Mx, My]``, or ``None``
        if fewer than 3 intersection points are found.
    """
    pts_2d = []
    for tri in simplices:
        v = hull_pts[tri]
        n_vals = v[:, 2]
        for i in range(3):
            j = (i + 1) % 3
            if (n_vals[i] - N_level) * (n_vals[j] - N_level) < 0:
                t = (N_level - n_vals[i]) / (n_vals[j] - n_vals[i])
                p = v[i] + t * (v[j] - v[i])
                pts_2d.append(p[:2])
            elif abs(n_vals[i] - N_level) < 1e-6:
                pts_2d.append(v[i, :2].copy())
    if len(pts_2d) < 3:
        return None

    pts_2d = np.array(pts_2d)

    # Use 2-D ConvexHull to extract only the outer boundary.
    # This eliminates interior points and guarantees a clean,
    # ordered contour.
    try:
        h2 = ConvexHull(pts_2d)
        boundary = pts_2d[h2.vertices]
    except Exception:
        # Fallback: sort by angle.
        cx, cy = pts_2d[:, 0].mean(), pts_2d[:, 1].mean()
        angles = np.arctan2(pts_2d[:, 1] - cy,
                            pts_2d[:, 0] - cx)
        boundary = pts_2d[np.argsort(angles)]

    return boundary


def _resample_contour(contour, n_angles, cx=None, cy=None):
    r"""
    Resample a closed convex 2-D contour to *n_angles* evenly-spaced
    **angular** positions from a reference centroid.

    Angular parameterization guarantees that point *j* on every
    contour corresponds to the same direction in the Mx-My plane,
    which is essential for ``plot_surface`` to produce a well-
    structured mesh between adjacent N-level contours.

    Using a **common centroid** across all contours (passed via
    *cx*, *cy*) ensures angular consistency even when contour shapes
    change significantly with N.

    Parameters
    ----------
    contour : numpy.ndarray
        ``(m, 2)`` ordered convex boundary points.
    n_angles : int
    cx, cy : float or None
        Reference centroid for angular parameterization.  If ``None``,
        computed from the contour's own mean.

    Returns
    -------
    numpy.ndarray
        ``(n_angles + 1, 2)`` resampled contour.  The last point
        repeats the first to close the loop for ``plot_surface``.
    """
    if cx is None:
        cx = contour[:, 0].mean()
    if cy is None:
        cy = contour[:, 1].mean()

    # Compute angle from common centroid for each boundary point.
    dx = contour[:, 0] - cx
    dy = contour[:, 1] - cy

    angles = np.arctan2(dy, dx)

    # Sort by angle.
    order = np.argsort(angles)
    a_sorted = angles[order]
    x_sorted = contour[order, 0]
    y_sorted = contour[order, 1]

    # Wrap: extend by ±2π for clean interpolation across the seam.
    a_ext = np.concatenate([a_sorted - 2 * np.pi,
                            a_sorted,
                            a_sorted + 2 * np.pi])
    x_ext = np.concatenate([x_sorted, x_sorted, x_sorted])
    y_ext = np.concatenate([y_sorted, y_sorted, y_sorted])

    # Target: n_angles equally spaced in [-π, π).
    target_angles = np.linspace(-np.pi, np.pi, n_angles,
                                endpoint=False)

    # Interpolate x and y as functions of angle (Cartesian interp,
    # not polar radius — more stable on elongated sections).
    x_interp = np.interp(target_angles, a_ext, x_ext)
    y_interp = np.interp(target_angles, a_ext, y_ext)

    # Close the loop: append first point so plot_surface draws
    # a face across the seam instead of leaving a slit.
    x_closed = np.append(x_interp, x_interp[0])
    y_closed = np.append(y_interp, y_interp[0])

    return np.column_stack([x_closed, y_closed])


def plot_3d_surface(nm_3d, demands=None, title="",
                    n_levels=20, n_angles=72):
    r"""
    Plot the 3D resistance surface as a lofted surface built from
    Mx-My contour slices at regular N intervals.

    Two side-by-side perspective views are generated: one from above
    (compression-dominant) and one from below (tension-dominant, N
    axis visually inverted).

    Contour rings ("parallels") and meridian lines ("longitudes")
    are drawn on the surface to aid shape comprehension.

    Parameters
    ----------
    nm_3d : dict
        Output of :meth:`NMDiagram.generate_biaxial`. Must contain
        ``N_kN``, ``Mx_kNm``, ``My_kNm``.
    demands : list of dict, optional
    title : str, optional
    n_levels : int, optional
        Number of N-level slices. Default 20.
    n_angles : int, optional
        Angular resolution of each contour. Default 72 (every 5°).

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.colors import Normalize

    N_all = nm_3d["N_kN"]
    Mx_all = nm_3d["Mx_kNm"]
    My_all = nm_3d["My_kNm"]
    pts = np.column_stack([Mx_all, My_all, N_all])

    N_min, N_max = N_all.min(), N_all.max()
    cmap = plt.cm.RdYlBu_r
    norm_c = Normalize(vmin=N_min, vmax=N_max)

    # ---- Build structured surface from hull contour slices ----
    hull = ConvexHull(pts)
    N_levels = np.linspace(N_min * 0.98, N_max * 0.98, n_levels)

    # First pass: collect raw contours and find the largest one
    # to use its centroid as the common angular reference.
    raw_contours = []
    raw_N = []
    max_area = 0.0
    ref_cx, ref_cy = 0.0, 0.0

    for nl in N_levels:
        contour = _hull_slice_at_N(pts, hull.simplices, nl)
        if contour is None or len(contour) < 4:
            continue
        span_x = contour[:, 0].max() - contour[:, 0].min()
        span_y = contour[:, 1].max() - contour[:, 1].min()
        if span_x < 1e-3 and span_y < 1e-3:
            continue
        raw_contours.append(contour)
        raw_N.append(nl)
        area = span_x * span_y
        if area > max_area:
            max_area = area
            ref_cx = contour[:, 0].mean()
            ref_cy = contour[:, 1].mean()

    # Second pass: resample all contours with the common centroid.
    grid_Mx = []
    grid_My = []
    grid_N = []
    valid_levels = []

    for contour, nl in zip(raw_contours, raw_N):
        resampled = _resample_contour(contour, n_angles,
                                      cx=ref_cx, cy=ref_cy)
        n_cols = resampled.shape[0]  # n_angles + 1 (closed loop)
        grid_Mx.append(resampled[:, 0])
        grid_My.append(resampled[:, 1])
        grid_N.append(np.full(n_cols, nl))
        valid_levels.append(nl)

    if len(grid_Mx) < 3:
        # Fallback: scatter.
        fig, ax = plt.subplots(1, 1, figsize=(10, 8),
                               subplot_kw={'projection': '3d'})
        ax.scatter(Mx_all, My_all, N_all, s=1, alpha=0.3, c='blue')
        ax.set_xlabel("Mx [kN·m]")
        ax.set_ylabel("My [kN·m]")
        ax.set_zlabel("N [kN]")
        ax.set_title(title or "3D Resistance Surface (fallback)")
        fig.tight_layout()
        return fig

    GMx = np.array(grid_Mx)
    GMy = np.array(grid_My)
    GN = np.array(grid_N)

    # ---- Face colours from N ----
    face_colors = cmap(norm_c(GN))

    # ---- Two-panel figure: above + below ----
    fig = plt.figure(figsize=(18, 9))
    views = [
        (1, 30, -55, "Perspective — tension side"),
        (2, -30, -55, "Perspective — compression side"),
    ]

    n_meridians = min(12, n_angles)
    meridian_idx = np.linspace(0, n_angles - 1, n_meridians,
                               dtype=int)

    for idx, elev, azim, vtitle in views:
        ax = fig.add_subplot(1, 2, idx, projection='3d')

        # Surface.
        ax.plot_surface(GMx, GMy, GN,
                        facecolors=face_colors,
                        rstride=1, cstride=1,
                        shade=False, alpha=0.55,
                        antialiased=True)

        # Contour rings (parallels) — grid is already closed,
        # so plotting the full row draws a closed ring.
        for i in range(len(valid_levels)):
            ax.plot(GMx[i], GMy[i], GN[i],
                    color='#333333', lw=0.4, alpha=0.5)

        # Meridians (longitudes).
        for j in meridian_idx:
            ax.plot(GMx[:, j], GMy[:, j], GN[:, j],
                    color='#555555', lw=0.3, alpha=0.4)

        # Demand points.
        if demands:
            for d in demands:
                n_d = d["N"] / 1e3
                mx_d = d["Mx"] / 1e6
                my_d = d.get("My", 0) / 1e6
                ax.scatter([mx_d], [my_d], [n_d], c='red', s=80,
                           zorder=10, depthshade=False,
                           edgecolors='darkred', linewidths=0.8)
                ax.text(mx_d, my_d, n_d, f"  {d['name']}",
                        fontsize=9, color='darkred',
                        fontweight='bold')

        ax.set_xlabel("Mx [kN·m]", fontsize=9, labelpad=8)
        ax.set_ylabel("My [kN·m]", fontsize=9, labelpad=8)
        ax.set_zlabel("N [kN]", fontsize=9, labelpad=8)
        ax.set_title(vtitle, fontsize=11)
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(labelsize=8)

    # Dedicated colour-bar axes between the two plots.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])  # [left, bottom, width, height]
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label="N [kN]")

    fig.suptitle(
        title or "3D Resistance Surface (N, Mx, My)",
        fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.95, bottom=0.15,
                        top=0.92, wspace=0.15)
    return fig


# ==================================================================
#  A) Bundle of M-χ curves at multiple N levels
# ==================================================================

def plot_moment_curvature_bundle(mc_list, direction='x', title=""):
    r"""
    Plot a family of moment-curvature curves at different axial
    force levels on the same axes.

    Each curve is coloured by its N value using a sequential
    colourmap, providing an immediate view of how ductility and
    moment capacity change with axial force.

    Parameters
    ----------
    mc_list : list of dict
        Each dict is the output of
        :meth:`NMDiagram.generate_moment_curvature` at a different
        ``N_fixed``.
    direction : ``'x'`` or ``'y'``, optional
        Default ``'x'``.
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    M_label = f"M{direction}"

    N_values = [mc["N_fixed_kN"] for mc in mc_list]
    N_min, N_max = min(N_values), max(N_values)
    if abs(N_max - N_min) < 1e-6:
        N_max = N_min + 1
    # Use a diverging palette centred on N=0: blue for compression,
    # red for tension, clearly distinguishable.
    cmap_bundle = plt.cm.coolwarm
    norm_bundle = plt.Normalize(vmin=N_min, vmax=N_max)

    for mc in mc_list:
        chi = mc["chi_km"]
        M = mc["M_kNm"]
        N_kN = mc["N_fixed_kN"]

        # Truncate at ultimate
        uc_pos = mc.get("ultimate_chi_pos")
        uc_neg = mc.get("ultimate_chi_neg")
        chi_lo = (uc_neg * 1e6 * 1.02) if uc_neg is not None else chi.min()
        chi_hi = (uc_pos * 1e6 * 1.02) if uc_pos is not None else chi.max()
        mask = (chi >= chi_lo) & (chi <= chi_hi)
        if np.sum(mask) < 5:
            mask = np.ones(len(chi), dtype=bool)

        color = cmap_bundle(norm_bundle(N_kN))
        ax.plot(chi[mask], M[mask], '-', lw=1.5, color=color,
                label=f"N={N_kN:.0f} kN")

        # Mark ultimate points
        for suffix in ("_pos", "_neg"):
            uc = mc.get(f"ultimate_chi{suffix}")
            um = mc.get(f"ultimate_M{suffix}")
            if uc is not None and um is not None:
                ax.plot(uc * 1e6, um / 1e6, 'x', color=color,
                        ms=8, mew=2.5, zorder=5)

    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel("χ [1/km]")
    ax.set_ylabel(f"{M_label} [kN·m]")

    sm = plt.cm.ScalarMappable(cmap=cmap_bundle, norm=norm_bundle)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("N [kN]")

    ax.set_title(title or f"{M_label}-χ diagrams at varying N")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ==================================================================
#  B) Polar ductility diagram — ultimate curvature vs direction
# ==================================================================

def plot_polar_ductility(nm_gen, N_fixed, n_angles=72,
                         n_points=400, title=""):
    r"""
    Polar diagram of ultimate curvature as a function of bending
    direction.

    For each angle :math:`\theta` in the :math:`(\chi_x, \chi_y)`
    plane, the section is loaded to failure at the given axial force.
    The radial coordinate is the ultimate curvature magnitude
    :math:`\chi_u(\theta)`.

    A circular plot means isotropic ductility; elongation along one
    axis reveals higher ductility in that bending direction.

    The scan uses warm-starting: the equilibrium ``eps0`` from the
    previous angle is carried over as initial guess, which greatly
    improves convergence smoothness.

    Parameters
    ----------
    nm_gen : NMDiagram
        The diagram generator (wraps solver).
    N_fixed : float
        Axial force [N].
    n_angles : int, optional
        Angular resolution. Default 72.
    n_points : int, optional
        Curvature steps per direction. Default 400.
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    sec = nm_gen.solver.sec
    emg, exg, emb, _ = nm_gen._collect_strain_limits()

    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    chi_ultimates = np.zeros(n_angles)

    # Warm-start eps0: carry over from previous angle
    eps0_warm = 0.0

    for i, theta in enumerate(thetas):
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Estimate max curvature from geometry
        chi_max_candidates = []
        if abs(cos_t) > 0.01 and sec.H > 0:
            chi_max_candidates.append(abs(emb) / (sec.H * 0.3) * 1.5)
        if abs(sin_t) > 0.01 and sec.B > 0:
            chi_max_candidates.append(abs(emb) / (sec.B * 0.3) * 1.5)
        chi_max = max(chi_max_candidates) if chi_max_candidates else 1e-4

        # Scan from 0 to chi_max with warm-started eps0
        chi_u = 0.0
        eps0_prev = eps0_warm

        for chi_mag in np.linspace(0, chi_max, n_points):
            chi_x = chi_mag * cos_t
            chi_y = chi_mag * sin_t

            # Solve with warm start from previous step
            eps0 = nm_gen._solve_eps0_for_N(
                nm_gen.solver, N_fixed, chi_x, chi_y, eps0_prev, emb)
            eps0_prev = eps0  # warm-start within this angle

            eb, er = nm_gen.solver.strain_field(eps0, chi_x, chi_y)
            all_eps = np.concatenate([eb, er]) if len(er) > 0 else eb

            if (all_eps.min() <= emb * 0.99
                    or all_eps.max() >= exg * 0.99):
                chi_u = chi_mag
                break
            chi_u = chi_mag

        chi_ultimates[i] = chi_u
        eps0_warm = eps0_prev  # carry to next angle

    # Convert to 1/km for readability
    chi_km = chi_ultimates * 1e6

    # --- Outlier rejection and smoothing ---
    # Replace outliers (values deviating > 2x from local median)
    # with interpolated values. This handles solver non-convergence
    # at isolated angles.
    if n_angles >= 8:
        from scipy.ndimage import median_filter, uniform_filter1d
        # Circular median filter (window=5)
        chi_extended = np.concatenate([chi_km[-3:], chi_km, chi_km[:3]])
        med = median_filter(chi_extended, size=5)[3:-3]
        # Replace points where value deviates > 50% from local median
        for k in range(len(chi_km)):
            if med[k] > 0 and abs(chi_km[k] - med[k]) / med[k] > 0.5:
                chi_km[k] = med[k]
        # Light circular smoothing (moving average, window=3)
        chi_ext2 = np.concatenate([chi_km[-2:], chi_km, chi_km[:2]])
        chi_km = uniform_filter1d(chi_ext2, size=3)[2:-2]

    # Close the polar loop
    thetas_closed = np.append(thetas, thetas[0])
    chi_closed = np.append(chi_km, chi_km[0])

    fig, ax = plt.subplots(1, 1, figsize=(9, 9),
                           subplot_kw=dict(projection='polar'))

    ax.plot(thetas_closed, chi_closed, 'b-', lw=2)
    ax.fill(thetas_closed, chi_closed, alpha=0.15, color='blue')

    # Mark cardinal directions
    ax.set_thetagrids([0, 90, 180, 270],
                      labels=['χx+ (Mx+)', 'χy+ (My+)',
                              'χx− (Mx−)', 'χy− (My−)'])

    ax.set_title(
        title or
        f"Ultimate curvature χ_u [1/km] at N={N_fixed/1e3:.0f} kN",
        pad=20)
    ax.set_rlabel_position(45)

    fig.tight_layout()
    return fig


# ==================================================================
#  C) 3D surface M-χ-N
# ==================================================================

def plot_moment_curvature_surface(mc_list, direction='x', title=""):
    r"""
    3D surface of moment vs curvature vs axial force.

    The individual M-χ curves are interpolated onto a common χ grid
    and rendered as a continuous surface using
    :meth:`~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface`.

    Parameters
    ----------
    mc_list : list of dict
        Each dict is the output of
        :meth:`NMDiagram.generate_moment_curvature`.
    direction : ``'x'`` or ``'y'``, optional
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy.interpolate import interp1d

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    M_label = f"M{direction}"

    # Collect truncated curves
    curves = []
    N_values = []
    for mc in mc_list:
        chi = mc["chi_km"]
        M = mc["M_kNm"]
        N_kN = mc["N_fixed_kN"]

        uc_pos = mc.get("ultimate_chi_pos")
        uc_neg = mc.get("ultimate_chi_neg")
        chi_lo = (uc_neg * 1e6 * 1.02) if uc_neg is not None else chi.min()
        chi_hi = (uc_pos * 1e6 * 1.02) if uc_pos is not None else chi.max()
        mask = (chi >= chi_lo) & (chi <= chi_hi)
        if np.sum(mask) < 5:
            mask = np.ones(len(chi), dtype=bool)

        curves.append((chi[mask], M[mask]))
        N_values.append(N_kN)

    # Build common chi grid spanning the intersection of all ranges
    chi_min_all = max(c[0].min() for c in curves)
    chi_max_all = min(c[0].max() for c in curves)
    if chi_max_all <= chi_min_all:
        # Fallback: use union range
        chi_min_all = min(c[0].min() for c in curves)
        chi_max_all = max(c[0].max() for c in curves)

    n_chi = 150
    chi_common = np.linspace(chi_min_all, chi_max_all, n_chi)

    # Interpolate each curve onto common grid
    M_grid = np.zeros((len(curves), n_chi))
    for i, (chi_c, M_c) in enumerate(curves):
        # Sort by chi (should already be, but safety)
        order = np.argsort(chi_c)
        f_interp = interp1d(chi_c[order], M_c[order],
                            kind='linear', bounds_error=False,
                            fill_value=np.nan)
        M_grid[i, :] = f_interp(chi_common)

    N_arr = np.array(N_values)

    # Create meshgrid for surface
    CHI, N_MESH = np.meshgrid(chi_common, N_arr)

    # Plot surface
    cmap_surf = plt.cm.coolwarm #viridis
    ax.plot_surface(CHI, N_MESH, M_grid, cmap=cmap_surf,
                    alpha=0.7, edgecolor='none', antialiased=True)

    # Also draw the individual curves as wireframe for clarity
    norm_surf = plt.Normalize(vmin=N_arr.min(), vmax=N_arr.max())
    for i, N_kN in enumerate(N_values):
        valid = ~np.isnan(M_grid[i, :])
        color = cmap_surf(norm_surf(N_kN))
        ax.plot(chi_common[valid],
                np.full(np.sum(valid), N_kN),
                M_grid[i, valid],
                '-', lw=1.5, color=color, zorder=5)

    ax.set_xlabel("χ [1/km]")
    ax.set_ylabel("N [kN]")
    ax.set_zlabel(f"{M_label} [kN·m]")

    sm = plt.cm.ScalarMappable(cmap=cmap_surf, norm=norm_surf)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("N [kN]")

    ax.set_title(title or f"{M_label}-χ-N surface")
    fig.tight_layout()
    return fig


# ==================================================================
#  Legacy alias (backward compatibility with cli.py)
# ==================================================================

def plot_stress_profile(results, sec, title=""):
    r"""
    Legacy wrapper — now generates two section-state maps (strain
    and stress) side by side.

    Parameters
    ----------
    results : dict
    sec : GenericSection or RectSection
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Strain map
    _draw_polygon_outline(ax1, sec, facecolor='#F5F5F5',
                          edgecolor='black', linewidth=2)
    bulk = results["bulk"]
    bval_eps = bulk["eps"] * 1000
    norm_e, cmap_e = _make_color_norm_and_cmap(bval_eps)
    sc1 = ax1.scatter(bulk["x"], bulk["y"], c=bval_eps,
                      cmap=cmap_e, norm=norm_e,
                      s=max(3, 8000 / max(len(bulk["x"]), 1)),
                      marker='s', edgecolors='none', zorder=2)
    plt.colorbar(sc1, ax=ax1, label='ε [‰]', shrink=0.8)

    rb = results["rebars"]
    if len(rb["y"]) > 0:
        rval_eps = rb["eps"] * 1000
        ax1.scatter(rb["x"], rb["y"], c=rval_eps,
                    cmap=cmap_e, norm=norm_e,
                    s=100, marker='o', edgecolors='black',
                    linewidths=1.5, zorder=5)
        for i in range(len(rb["y"])):
            ax1.annotate(f"{i+1}: {rval_eps[i]:.2f}‰",
                         (rb["x"][i], rb["y"][i]),
                         xytext=(10, -4), textcoords="offset points",
                         fontsize=7, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2',
                                   facecolor='white', alpha=0.85,
                                   edgecolor='none'), zorder=6)

    _draw_neutral_axis(ax1, sec, results)
    _draw_reference_axes(ax1, sec)
    _section_axis_limits(ax1, sec)
    ax1.legend(fontsize=9, loc='best')
    ax1.set_title("Strain ε [‰]")
    ax1.grid(True, alpha=0.15)

    # Stress map
    _draw_polygon_outline(ax2, sec, facecolor='#F5F5F5',
                          edgecolor='black', linewidth=2)
    bval_sig = bulk["sigma"]
    norm_s, cmap_s = _make_color_norm_and_cmap(bval_sig)
    sc2 = ax2.scatter(bulk["x"], bulk["y"], c=bval_sig,
                      cmap=cmap_s, norm=norm_s,
                      s=max(3, 8000 / max(len(bulk["x"]), 1)),
                      marker='s', edgecolors='none', zorder=2)
    plt.colorbar(sc2, ax=ax2, label='σ [MPa]', shrink=0.8)

    if len(rb["y"]) > 0:
        rval_sig = rb["sigma_net"] if "sigma_net" in rb else rb["sigma"]
        ax2.scatter(rb["x"], rb["y"], c=rval_sig,
                    cmap=cmap_s, norm=norm_s,
                    s=100, marker='o', edgecolors='black',
                    linewidths=1.5, zorder=5)
        for i in range(len(rb["y"])):
            ax2.annotate(f"{i+1}: {rval_sig[i]:.1f}",
                         (rb["x"][i], rb["y"][i]),
                         xytext=(10, -4), textcoords="offset points",
                         fontsize=7, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2',
                                   facecolor='white', alpha=0.85,
                                   edgecolor='none'), zorder=6)

    _draw_neutral_axis(ax2, sec, results)
    _draw_reference_axes(ax2, sec)
    _section_axis_limits(ax2, sec)
    ax2.legend(fontsize=9, loc='best')
    ax2.set_title("Stress σ [MPa]")
    ax2.grid(True, alpha=0.15)

    fig.suptitle(title or "Section State", fontsize=14)
    fig.tight_layout()
    return fig


# ==================================================================
#  JSON → plot dispatcher (for ``gensec plot`` subcommand)
# ==================================================================

def plot_from_json(filepath, output_path=None, dpi=150):
    """
    Regenerate a plot from a previously exported JSON data file.

    Detects the data type from the ``"type"`` key in the JSON and
    dispatches to the appropriate plotting function.

    Supported types:

    - ``moment_curvature`` → :func:`plot_moment_curvature`
    - ``mx_my_contour`` → :func:`plot_mx_my_diagram`

    For files without a ``"type"`` key, the function inspects the
    available keys to infer the data type (N-M domain, 3-D surface,
    demand summary, etc.).

    Parameters
    ----------
    filepath : str
        Path to JSON data file.
    output_path : str or None
        Output PNG path. If ``None``, derived from the JSON filename.
    dpi : int, optional
        Resolution. Default 150.

    Returns
    -------
    str
        Path of the generated PNG file.

    Raises
    ------
    ValueError
        If the data type cannot be determined.
    """
    import json as _json

    with open(filepath, 'r') as f:
        data = _json.load(f)

    dtype = data.get("type")

    if output_path is None:
        base = filepath.rsplit('.', 1)[0]
        output_path = base + ".png"

    fig = None

    if dtype == "moment_curvature":
        # Reconstruct mc_data dict expected by plot_moment_curvature.
        mc = {
            "chi_km": np.array(data["chi_km"]),
            "M_kNm": np.array(data["M_kNm"]),
            "N_fixed_kN": data["N_fixed_kN"],
            "direction": data.get("direction", "x"),
        }
        # Restore key points.
        for prefix in ("cracking", "yield", "ultimate"):
            for suffix in ("_pos", "_neg"):
                chi_key = f"{prefix}_chi{suffix}"
                M_key = f"{prefix}_M{suffix}"
                chi_km_key = f"{chi_key}_km"
                M_kNm_key = f"{M_key}_kNm"
                if chi_km_key in data:
                    mc[chi_key] = data[chi_km_key] / 1e6  # back to 1/mm
                if M_kNm_key in data:
                    mc[M_key] = data[M_kNm_key] * 1e6  # back to N*mm
        fig = plot_moment_curvature(mc)

    elif dtype == "mx_my_contour":
        mx_my = {
            "Mx_kNm": np.array(data["Mx_kNm"]),
            "My_kNm": np.array(data["My_kNm"]),
            "N_fixed_kN": data.get("N_fixed_kN", 0),
        }
        fig = plot_mx_my_diagram(mx_my)

    elif "Mx_kNm" in data and "My_kNm" in data and "N_kN" in data:
        # 3D surface point cloud.
        nm_3d = {
            "N_kN": np.array(data["N_kN"]),
            "Mx_kNm": np.array(data["Mx_kNm"]),
            "My_kNm": np.array(data["My_kNm"]),
        }
        fig = plot_3d_surface(nm_3d)

    elif "N_kN" in data and "Mx_kNm" in data and "My_kNm" not in data:
        # N-M domain.
        nm = {
            "N_kN": np.array(data["N_kN"]),
            "M_kNm": np.array(data["Mx_kNm"]),
        }
        fig = plot_nm_diagram(nm)

    else:
        raise ValueError(
            f"Cannot determine plot type from '{filepath}'. "
            f"Keys: {list(data.keys())}"
        )

    if fig is not None:
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)

    return output_path