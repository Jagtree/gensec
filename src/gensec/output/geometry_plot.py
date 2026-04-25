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
# ---------------------------------------------------------------------------
r"""
Homogenized section geometry plot and textual report.

Provides three public entry points:

- :func:`plot_section_properties` — figure with centroid,
  principal axes :math:`(\xi, \eta)`, central inertia ellipse,
  and kern, of the homogenized section;
- :func:`print_section_properties` — formatted stdout report
  covering area, centroid, centroidal and principal
  second-moments, extreme-fiber distances, elastic and plastic
  section moduli;
- :func:`write_section_report` — the same content written to a
  UTF-8 text file.

All three functions can either accept a pre-computed
:class:`~gensec.geometry.properties.SectionProperties` instance
via ``props=...``, or compute it on the fly from
``sec.ideal_gross_properties`` (when :class:`~gensec.geometry.geometry.GenericSection`
exposes the attribute).
"""

import io
import contextlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Circle
from matplotlib.path import Path

from ..geometry.properties import (
    compute_inertia_ellipse,
    compute_kern_polygon,
)


# ==================================================================
#  Robust polygon outline with explicit hole carve-out
# ==================================================================


def _signed_area_np(ring):
    r"""Shoelace signed area; positive = CCW."""
    pts = ring
    if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _extend_path(verts, codes, ring):
    r"""Append a ring to (verts, codes) using CLOSEPOLY."""
    if len(ring) >= 2 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    n = len(ring)
    verts.extend(ring.tolist())
    verts.append(ring[0].tolist())
    codes.append(Path.MOVETO)
    codes.extend([Path.LINETO] * (n - 1))
    codes.append(Path.CLOSEPOLY)


def _draw_polygon_with_holes(ax, polygon, *,
                              facecolor='#EFEFEF',
                              edgecolor='black',
                              linewidth=1.8,
                              zorder=1):
    r"""
    Render a Shapely polygon (with holes) on a Matplotlib axes,
    using a compound :class:`~matplotlib.path.Path` with exterior
    in CCW and every hole in CW orientation, so that the
    non-zero winding rule carves holes reliably.
    """
    verts = []
    codes = []
    ext = np.asarray(polygon.exterior.coords)
    if _signed_area_np(ext) < 0:
        ext = ext[::-1]
    _extend_path(verts, codes, ext)
    for interior in polygon.interiors:
        ring = np.asarray(interior.coords)
        if _signed_area_np(ring) > 0:
            ring = ring[::-1]
        _extend_path(verts, codes, ring)
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                      linewidth=linewidth, zorder=zorder)
    ax.add_patch(patch)


# ==================================================================
#  Axis-through helpers (clipped to frame, labels pulled inside)
# ==================================================================


def _box_line_intersections(xg, yg, direction_rad, limits):
    r"""
    Parameterise a line through :math:`G` at angle
    ``direction_rad``, and return the parameter range
    :math:`[t_{\min}, t_{\max}]` for which the line lies inside
    the axis-limits rectangle.
    """
    xmin, xmax, ymin, ymax = limits
    dx = float(np.cos(direction_rad))
    dy = float(np.sin(direction_rad))
    ts = []
    eps = 1.0e-9 * max(xmax - xmin, ymax - ymin)
    if abs(dx) > 1.0e-12:
        for x_b in (xmin, xmax):
            t = (x_b - xg) / dx
            y_t = yg + t * dy
            if ymin - eps <= y_t <= ymax + eps:
                ts.append(t)
    if abs(dy) > 1.0e-12:
        for y_b in (ymin, ymax):
            t = (y_b - yg) / dy
            x_t = xg + t * dx
            if xmin - eps <= x_t <= xmax + eps:
                ts.append(t)
    if len(ts) < 2:
        return 0.0, 0.0
    return min(ts), max(ts)


def _draw_axis_through(ax, xg, yg, direction_rad, limits, *,
                        color, linewidth=1.5, zorder=5,
                        dashed=False):
    r"""Draw a line through :math:`G` clipped to the axis box."""
    t_min, t_max = _box_line_intersections(xg, yg, direction_rad,
                                             limits)
    if t_max <= t_min:
        return
    dx = float(np.cos(direction_rad))
    dy = float(np.sin(direction_rad))
    x1, y1 = xg + t_min * dx, yg + t_min * dy
    x2, y2 = xg + t_max * dx, yg + t_max * dy
    ls = (0, (4, 3)) if dashed else '-'
    ax.plot([x1, x2], [y1, y2], linestyle=ls, color=color,
            linewidth=linewidth, zorder=zorder)


def _place_edge_label(ax, xg, yg, direction_rad, limits, text, *,
                       color, fontsize=12, bold=False, box=False,
                       inset_frac=0.06):
    r"""
    Place a text label just inside the frame on the
    positive-direction end of a line through :math:`G`.  The text
    alignment is chosen to extend inward, guaranteeing the label
    stays inside the axes.
    """
    t_min, t_max = _box_line_intersections(xg, yg, direction_rad,
                                             limits)
    if t_max <= 0.0:
        return
    t_label = (1.0 - inset_frac) * t_max
    dx = float(np.cos(direction_rad))
    dy = float(np.sin(direction_rad))
    x_anchor = xg + t_label * dx
    y_anchor = yg + t_label * dy
    ha = 'right' if dx > 0.15 else ('left' if dx < -0.15 else 'center')
    va = 'top' if dy > 0.15 else ('bottom' if dy < -0.15 else 'center')
    weight = 'bold' if bold else 'normal'
    kwargs = dict(color=color, ha=ha, va=va,
                  fontsize=fontsize, fontweight=weight, zorder=11)
    if box:
        kwargs['bbox'] = dict(facecolor='white', edgecolor='none',
                              alpha=0.80, pad=1.4)
    ax.text(x_anchor, y_anchor, text, **kwargs)


def _axis_label(ax, x, y, text, *,
                color='black',
                ha='left', va='bottom',
                fontsize=11, bold=False,
                offset=(3, 3)):
    r"""Draw a small label offset from an anchor point."""
    weight = 'bold' if bold else 'normal'
    ax.annotate(text, xy=(x, y),
                xytext=offset, textcoords='offset points',
                fontsize=fontsize, fontweight=weight,
                color=color, ha=ha, va=va, zorder=11)


def _set_axis_limits(ax, bounds):
    r"""Set axis limits with a 20 % margin around a bounding box."""
    minx, miny, maxx, maxy = bounds
    mx = (maxx - minx) * 0.20
    my = (maxy - miny) * 0.20
    ax.set_xlim(minx - mx, maxx + mx)
    ax.set_ylim(miny - my, maxy + my)


# ==================================================================
#  Info box with key numeric properties
# ==================================================================


def _compact_sci(value, decimals=2):
    r"""
    Format a float in compact scientific notation for the info
    box, e.g. ``1.23e+06``.  Returns ``'--'`` for NaN or None.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '--'
    return f"{value:.{decimals}e}"


def _draw_properties_info_box(ax, props):
    r"""
    Draw an anchored text box in the upper-right corner of *ax*
    showing the most important homogenized geometric properties.

    The box contains:

    - homogenized area :math:`A_{\mathrm{id}}`;
    - centroid coordinates :math:`(x_G,\;y_G)`;
    - centroidal second-moments :math:`I_x,\;I_y`;
    - elastic section moduli :math:`W_x,\;W_y` (minimum of the
      two sides);
    - principal-axis angle :math:`\alpha` and principal moments
      :math:`I_\xi,\;I_\eta` (only when the user and principal
      frames differ by more than :math:`0.01^{\circ}`).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    props : SectionProperties
        Homogenized properties bundle.
    """
    deg = np.degrees(props.alpha)

    lines = [
        f"A   = {_compact_sci(props.area)} mm²",
        f"xG  = {props.xg:.1f} mm",
        f"yG  = {props.yg:.1f} mm",
        f"Ix  = {_compact_sci(props.Ix)} mm⁴",
        f"Iy  = {_compact_sci(props.Iy)} mm⁴",
    ]

    # Elastic section moduli — report the smaller side.
    W_x_min = min(getattr(props, 'W_x_top', float('inf')),
                  getattr(props, 'W_x_bot', float('inf')))
    W_y_min = min(getattr(props, 'W_y_left', float('inf')),
                  getattr(props, 'W_y_right', float('inf')))
    if W_x_min < float('inf'):
        lines.append(f"Wx  = {_compact_sci(W_x_min)} mm³")
    if W_y_min < float('inf'):
        lines.append(f"Wy  = {_compact_sci(W_y_min)} mm³")

    # Show principal frame only when it differs from the user
    # frame by more than a negligible amount.
    near_zero = abs(deg) < 0.01
    near_90 = abs(abs(deg) - 90.0) < 0.01
    if not (near_zero or near_90):
        lines.append(f"α   = {deg:.2f}°")
        lines.append(f"Iξ  = {_compact_sci(props.I_xi)} mm⁴")
        lines.append(f"Iη  = {_compact_sci(props.I_eta)} mm⁴")

    text = "\n".join(lines)
    ax.text(0.98, 0.98, text,
            transform=ax.transAxes,
            fontsize=8, fontfamily='monospace',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='white', edgecolor='#AAAAAA',
                      alpha=0.90),
            zorder=15)


# ==================================================================
#  Main plot
# ==================================================================


def plot_section_properties(sec, props=None, *,
                             show_ellipse=True,
                             show_kern=True,
                             show_rebars=True,
                             show_rebar_numbers=True,
                             show_principal=True,
                             show_reference=True,
                             show_info_box=True,
                             title=None):
    r"""
    Draw the homogenized cross-section with its inertial overlays.

    The figure contains:

    - the polygon outline (with hole carve-out);
    - rebar layers with optional numbering;
    - the centroid :math:`G` (of the *homogenized* section);
    - the user reference axes :math:`x, y` (dashed grey) through
      :math:`G`;
    - the principal axes :math:`\xi, \eta` (solid coloured)
      through :math:`G`;
    - the central inertia ellipse (Culmann, blue);
    - the kern (semi-transparent orange fill);
    - an info box with the key numerical properties (optional).

    Axes are clipped to the plot frame and their labels are
    pulled inside to avoid overflow.

    Parameters
    ----------
    sec : GenericSection or compatible
        Must expose ``polygon`` and ``rebars``; if ``props`` is
        not provided, must also expose ``ideal_gross_properties``.
    props : SectionProperties, optional
        Pre-computed homogenized properties.  If ``None``, uses
        ``sec.ideal_gross_properties``.
    show_ellipse : bool, optional
        Show the central inertia ellipse.  Default ``True``.
    show_kern : bool, optional
        Show the kern polygon.  Default ``True``.
    show_rebars : bool, optional
        Show rebar circles.  Default ``True``.
    show_rebar_numbers : bool, optional
        Number each rebar with a 1-based index label.
        Only effective when ``show_rebars`` is ``True``.
        Default ``True``.
    show_principal : bool, optional
        Show the principal axes :math:`\xi, \eta`.  Default ``True``.
    show_reference : bool, optional
        Show the user reference axes :math:`x, y`.
        Default ``True``.
    show_info_box : bool, optional
        Show a summary box with key properties
        (:math:`A`, centroid, :math:`I_x`, :math:`I_y`,
        :math:`\alpha`).  Default ``True``.
    title : str or None, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if props is None:
        props = sec.ideal_gross_properties
    poly = sec.polygon

    fig, ax = plt.subplots(1, 1, figsize=(12.0, 10.0))

    _draw_polygon_with_holes(ax, poly,
                              facecolor='#EFEFEF',
                              edgecolor='black',
                              linewidth=1.8,
                              zorder=1)

    if show_rebars:
        for i, rb in enumerate(sec.rebars):
            if rb.x is None:
                continue
            d = rb.diameter if getattr(rb, 'diameter', 0) > 0 \
                else 16.0
            color = '#333333' if getattr(rb, 'embedded', True) \
                else '#CC6600'
            circ = Circle((rb.x, rb.y), d / 2.0,
                          facecolor=color,
                          edgecolor='black',
                          linewidth=0.7, zorder=4)
            ax.add_patch(circ)

            if show_rebar_numbers:
                ax.annotate(
                    f"{i + 1}", (rb.x, rb.y),
                    xytext=(12, -4),
                    textcoords='offset points',
                    fontsize=7, fontweight='bold',
                    color='black', zorder=12,
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', alpha=0.8,
                              edgecolor='none'))

    if show_ellipse:
        ell = compute_inertia_ellipse(props, n_points=240)
        ax.plot(ell[:, 0], ell[:, 1], '-',
                color='#1B7FCC', linewidth=1.8,
                label='Central inertia ellipse',
                zorder=6)

    note = ""
    if show_kern:
        kern = compute_kern_polygon(poly, props)
        if len(kern) >= 4:
            if props.is_convex:
                lbl = 'Kern (core)'
                alpha_fill = 0.28
            else:
                lbl = 'Kern (from convex hull — approximate)'
                alpha_fill = 0.14
                note = "  [non-convex exterior: kern overestimated]"
            ax.fill(kern[:, 0], kern[:, 1],
                    facecolor='#E67F00', alpha=alpha_fill,
                    edgecolor='#B55500', linewidth=1.3,
                    label=lbl, zorder=5)

    bounds = getattr(sec, '_bounds', None)
    if bounds is None:
        bounds = poly.bounds
    _set_axis_limits(ax, bounds)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    limits = (1.2*xmin, 1.2*xmax, 1.2*ymin, 1.2*ymax) ### TODO: check the x1.2 factor
    ax.set_aspect('auto') ###TODO: Check this

    xg, yg = props.xg, props.yg
    alpha = props.alpha
    near_zero = abs(alpha) < 1.0e-6
    near_90 = abs(abs(alpha) - np.pi / 2) < 1.0e-6
    coincident = near_zero or near_90

    if show_reference and not coincident:
        _draw_axis_through(ax, xg, yg, 0.0, limits,
                            color='#888888', linewidth=1.0,
                            zorder=3, dashed=True)
        _draw_axis_through(ax, xg, yg, np.pi / 2, limits,
                            color='#888888', linewidth=1.0,
                            zorder=3, dashed=True)
        _place_edge_label(ax, xg, yg, 0.0, limits,
                           'x', color='#888888', fontsize=11)
        _place_edge_label(ax, xg, yg, np.pi / 2, limits,
                           'y', color='#888888', fontsize=11)

    if show_principal:
        _draw_axis_through(ax, xg, yg, alpha, limits,
                            color='#C0392B', linewidth=1.7,
                            zorder=7)
        _draw_axis_through(ax, xg, yg, alpha + np.pi / 2, limits,
                            color='#2E7D32', linewidth=1.7,
                            zorder=7)
        if coincident and near_zero:
            xi_lbl, eta_lbl = r'$\xi\equiv x$', r'$\eta\equiv y$'
        elif coincident and near_90:
            xi_lbl, eta_lbl = r'$\xi\equiv y$', r'$\eta\equiv x$'
        else:
            xi_lbl, eta_lbl = r'$\xi$', r'$\eta$'
        _place_edge_label(ax, xg, yg, alpha, limits,
                           xi_lbl, color='#C0392B', fontsize=13,
                           bold=True, box=True)
        _place_edge_label(ax, xg, yg, alpha + np.pi / 2, limits,
                           eta_lbl, color='#2E7D32', fontsize=13,
                           bold=True, box=True)

    ax.plot([xg], [yg], marker='o', color='black',
             markersize=6, zorder=10)
    _axis_label(ax, xg, yg, ' G', color='black',
                ha='left', va='bottom',
                fontsize=12, bold=True, offset=(4, 4))

    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.grid(True, alpha=0.2)

    if title is None:
        # Annotate homogenization only when rebars with n ≠ n_bulk
        # actually perturb the geometry.
        is_homogenized = any(
            abs(getattr(rb, 'E', props.E_bulk) - props.E_bulk)
            > 1.0e-6 * props.E_bulk
            for rb in getattr(sec, 'rebars', []) or []
            if getattr(rb, 'x', None) is not None
        )
        if is_homogenized:
            title = ('Homogenized section — geometric properties'
                     f'  (E_ref = {props.E_ref:.0f} MPa)')
        else:
            title = 'Section — geometric properties'
    ax.set_title(title + note)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='best',
                  fontsize=9, framealpha=0.92)

    # ---- Summary info box ----
    if show_info_box:
        _draw_properties_info_box(ax, props)

    fig.tight_layout()
    return fig


# Legacy alias kept for backward compatibility.
plot_ideal_gross_section = plot_section_properties


# ==================================================================
#  Textual report
# ==================================================================


def _fmt_sci(value, width=15, decimals=4):
    r"""Format a float in scientific notation, or ``--`` if NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"{'--':>{width}}"
    return f"{value:>{width}.{decimals}e}"


def _fmt_float(value, width=15, decimals=4):
    r"""Format a float, or ``--`` if NaN / None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"{'--':>{width}}"
    return f"{value:>{width}.{decimals}f}"


def print_section_properties(sec, props=None):
    r"""
    Print a formatted report of the homogenized geometric
    properties.

    Parameters
    ----------
    sec : GenericSection or compatible
    props : SectionProperties, optional
        If ``None``, uses ``sec.ideal_gross_properties``.
    """
    if props is None:
        props = sec.ideal_gross_properties

    deg = np.degrees(props.alpha)
    w = 70
    hdr = "=" * w

    print(hdr)
    print("HOMOGENIZED SECTION — GEOMETRIC PROPERTIES")
    print(hdr)

    # ---- Homogenization reference ----
    print(f"  E_ref  = {props.E_ref:>15.4e} MPa")
    print(f"  E_bulk = {props.E_bulk:>15.4e} MPa   "
          f"(n_bulk = {props.n_bulk:.4f})")

    # ---- Area and centroid ----
    print()
    print(f"  Area              A     = {_fmt_sci(props.area)} mm^2")
    print(f"  Static moments    Sx    = {_fmt_sci(props.Sx)} mm^3")
    print(f"                    Sy    = {_fmt_sci(props.Sy)} mm^3")
    print(f"  Centroid          xG    = {_fmt_float(props.xg)} mm")
    print(f"                    yG    = {_fmt_float(props.yg)} mm")

    # ---- Second moments (user frame) ----
    print()
    print("  Centroidal second-moments (user frame):")
    print(f"                    Ix    = {_fmt_sci(props.Ix)} mm^4")
    print(f"                    Iy    = {_fmt_sci(props.Iy)} mm^4")
    print(f"                    Ixy   = {_fmt_sci(props.Ixy)} mm^4")
    print(f"                    Ip    = {_fmt_sci(props.I_polar)} mm^4")

    # ---- Principal frame ----
    print()
    print("  Principal axes (xi, eta):")
    print(f"                    alpha = {deg:>15.4f} deg")
    print(f"                    I_xi  = {_fmt_sci(props.I_xi)} mm^4")
    print(f"                    I_eta = {_fmt_sci(props.I_eta)} mm^4")

    # ---- Radii of gyration ----
    print()
    print("  Radii of gyration:")
    print(f"                  rho_x   = {_fmt_float(props.rho_x)} mm")
    print(f"                  rho_y   = {_fmt_float(props.rho_y)} mm")
    print(f"                  rho_xi  = {_fmt_float(props.rho_xi)} mm")
    print(f"                  rho_eta = {_fmt_float(props.rho_eta)} mm")

    # ---- Extreme-fiber distances ----
    print()
    print("  Extreme-fiber distances (from centroid):")
    print(f"                  c_y_top = {_fmt_float(props.c_y_top)} mm")
    print(f"                  c_y_bot = {_fmt_float(props.c_y_bot)} mm")
    print(f"                  c_x_lft = {_fmt_float(props.c_x_left)} mm")
    print(f"                  c_x_rht = {_fmt_float(props.c_x_right)} mm")
    print(f"                  c_xi(+) = {_fmt_float(props.c_xi_pos)} mm")
    print(f"                  c_xi(-) = {_fmt_float(props.c_xi_neg)} mm")
    print(f"                  c_eta(+)= {_fmt_float(props.c_eta_pos)} mm")
    print(f"                  c_eta(-)= {_fmt_float(props.c_eta_neg)} mm")

    # ---- Elastic section moduli ----
    print()
    print("  Elastic section moduli (W = I / c):")
    print(f"                  W_x_top = {_fmt_sci(props.W_x_top)} mm^3")
    print(f"                  W_x_bot = {_fmt_sci(props.W_x_bot)} mm^3")
    print(f"                  W_y_lft = {_fmt_sci(props.W_y_left)} mm^3")
    print(f"                  W_y_rht = {_fmt_sci(props.W_y_right)} mm^3")
    print(f"                  W_xi(+) = {_fmt_sci(props.W_xi_pos)} mm^3")
    print(f"                  W_xi(-) = {_fmt_sci(props.W_xi_neg)} mm^3")
    print(f"                  W_eta(+)= {_fmt_sci(props.W_eta_pos)} mm^3")
    print(f"                  W_eta(-)= {_fmt_sci(props.W_eta_neg)} mm^3")

    # ---- Plastic section moduli ----
    print()
    print("  Plastic section moduli (homogenized, PNA from area split):")
    print(f"                  Z_x     = {_fmt_sci(props.Z_x)} mm^3")
    print(f"                  Z_y     = {_fmt_sci(props.Z_y)} mm^3")
    print(f"                  Z_xi    = {_fmt_sci(props.Z_xi)} mm^3")
    print(f"                  Z_eta   = {_fmt_sci(props.Z_eta)} mm^3")

    # ---- Shape factors (Z / W) for mono-material context ----
    # Only meaningful when W and Z are finite.
    if (props.W_x_top != float('inf')
        and not np.isnan(props.Z_x)
        and props.W_x_top > 0):
        # Shape factor is conventionally defined using the minimum
        # elastic modulus in each axis.
        W_x_min = min(props.W_x_top, props.W_x_bot)
        W_y_min = min(props.W_y_left, props.W_y_right)
        W_xi_min = min(props.W_xi_pos, props.W_xi_neg)
        W_eta_min = min(props.W_eta_pos, props.W_eta_neg)
        print()
        print("  Shape factors (Z / W_min, mono-material "
              "interpretation):")
        print(f"                  Z/W | x   = "
              f"{_fmt_float(props.Z_x / W_x_min, decimals=4)}")
        print(f"                  Z/W | y   = "
              f"{_fmt_float(props.Z_y / W_y_min, decimals=4)}")
        print(f"                  Z/W | xi  = "
              f"{_fmt_float(props.Z_xi / W_xi_min, decimals=4)}")
        print(f"                  Z/W | eta = "
              f"{_fmt_float(props.Z_eta / W_eta_min, decimals=4)}")

    # ---- Torsional constant (placeholder) ----
    print()
    if props.I_t is None:
        print("  Torsional constant I_t: not computed "
              "(St.-Venant solver not yet available)")
    else:
        print(f"  Torsional constant I_t = "
              f"{_fmt_sci(props.I_t)} mm^4")

    # ---- Convexity flag ----
    print()
    cvx = "yes" if props.is_convex else "no"
    print(f"  Exterior convex: {cvx}")
    if not props.is_convex:
        print("    (kern computed from the exterior convex hull;")
        print("     the true kern may be smaller than the one "
              "plotted)")
    print(hdr)


def write_section_report(sec, filepath, props=None):
    r"""
    Write the homogenized geometric properties to a UTF-8 text
    file.  Content identical to :func:`print_section_properties`.

    Parameters
    ----------
    sec : GenericSection or compatible
    filepath : str or os.PathLike
    props : SectionProperties, optional
    """
    if props is None:
        props = sec.ideal_gross_properties
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_section_properties(sec, props)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(buf.getvalue())


# Legacy aliases.
print_geometric_properties = print_section_properties
write_geometry_report = write_section_report