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
Homogenized geometric properties of an arbitrary cross-section.

All quantities in this module are computed on the **homogenized
section** (also called *ideal section* or *transformed section*):
the polygon area contributes with its own elastic modulus, and
every point fiber (rebar, tendon, FRP strip) contributes with the
*differential* area :math:`(n_s - n_{\mathrm{bulk}}) A_s`, where
:math:`n_i = E_i / E_{\mathrm{ref}}`.

This is the EC2 / NTC 2018 convention: the polygon is treated as
a continuous region of the bulk material, and each reinforcement
is added *in excess* to avoid double-counting the substrate
displaced by the bar.  Degeneracy is automatic:

- when no point fibers are present, the section coincides with
  the polygon alone (scaled by :math:`n_{\mathrm{bulk}}`);
- when :math:`E_s = E_{\mathrm{bulk}}` for every fiber, the
  homogenization factors vanish and the result is purely
  geometrical.

The central quantities are:

- homogenized area :math:`A`, static moments :math:`S_x, S_y`,
  centroid :math:`(x_G, y_G)`;
- centroidal second-moments :math:`I_x, I_y, I_{xy}` in the user
  frame, and :math:`I_\xi \ge I_\eta` in the principal frame,
  with rotation angle :math:`\alpha`;
- radii of gyration, central inertia ellipse, kern;
- distances to the extreme fibers (polygon vertices *and* point
  fibers are considered);
- elastic section moduli :math:`W_x^{\pm}, W_y^{\pm}, W_\xi^{\pm},
  W_\eta^{\pm}`;
- plastic section moduli :math:`Z_x, Z_y, Z_\xi, Z_\eta`
  obtained by locating the plastic neutral axis that splits the
  homogenized area in half;
- placeholder for the torsional constant :math:`I_t`
  (filled in by a future St.-Venant solver).
"""

from dataclasses import dataclass
from typing import Optional, Sequence, List
import numpy as np
from shapely.geometry import Polygon, box as sbox
from shapely.affinity import rotate as _rotate, translate as _translate


# ==================================================================
#  Low-level ring integrals (Green's theorem)
# ==================================================================


def _signed_area(coords):
    r"""
    Signed area of a (possibly open) ring via the shoelace formula.

    Parameters
    ----------
    coords : array_like of shape ``(n, 2)`` or ``(n+1, 2)``

    Returns
    -------
    float
        Positive for counter-clockwise rings, negative for
        clockwise.
    """
    pts = np.asarray(coords, dtype=float)
    if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ring_moments(coords):
    r"""
    Polygon moment integrals of a single ring via Green's theorem.

    .. math::

        A &= \tfrac{1}{2}\sum_i c_i,\\
        S_x &= \tfrac{1}{6}\sum_i (y_i + y_{i+1})\, c_i,\\
        S_y &= \tfrac{1}{6}\sum_i (x_i + x_{i+1})\, c_i,\\
        I_{xx,O} &= \tfrac{1}{12}\sum_i
                    (y_i^2 + y_i y_{i+1} + y_{i+1}^2)\, c_i,\\
        I_{yy,O} &= \tfrac{1}{12}\sum_i
                    (x_i^2 + x_i x_{i+1} + x_{i+1}^2)\, c_i,\\
        I_{xy,O} &= \tfrac{1}{24}\sum_i
                    \bigl(x_i y_{i+1} + 2 x_i y_i
                          + 2 x_{i+1} y_{i+1}
                          + x_{i+1} y_i\bigr)\, c_i,

    with :math:`c_i = x_i y_{i+1} - x_{i+1} y_i`.  Sign follows
    the orientation: CCW positive, CW negative.

    Parameters
    ----------
    coords : array_like of shape ``(n, 2)`` or ``(n+1, 2)``

    Returns
    -------
    tuple of float
        ``(A, Sx, Sy, Ixx_o, Iyy_o, Ixy_o)`` about the origin.
    """
    pts = np.asarray(coords, dtype=float)
    if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    x = pts[:, 0]
    y = pts[:, 1]
    x_n = np.roll(x, -1)
    y_n = np.roll(y, -1)
    cross = x * y_n - x_n * y
    A = 0.5 * float(np.sum(cross))
    Sx = (1.0 / 6.0) * float(np.sum((y + y_n) * cross))
    Sy = (1.0 / 6.0) * float(np.sum((x + x_n) * cross))
    Iyy = (1.0 / 12.0) * float(
        np.sum((x * x + x * x_n + x_n * x_n) * cross))
    Ixx = (1.0 / 12.0) * float(
        np.sum((y * y + y * y_n + y_n * y_n) * cross))
    Ixy = (1.0 / 24.0) * float(np.sum(
        (x * y_n + 2.0 * x * y + 2.0 * x_n * y_n + x_n * y) * cross))
    return A, Sx, Sy, Ixx, Iyy, Ixy


def _polygon_moments_about_origin(polygon):
    r"""
    Moments of a Shapely polygon about the origin, summed over
    exterior (CCW-forced) and holes (CW-forced).

    Parameters
    ----------
    polygon : shapely.geometry.Polygon

    Returns
    -------
    tuple of float
        ``(A, Sx, Sy, Ixx_o, Iyy_o, Ixy_o)``.
    """
    ext = np.asarray(polygon.exterior.coords, dtype=float)
    if _signed_area(ext) < 0:
        ext = ext[::-1]
    A, Sx, Sy, Ixx, Iyy, Ixy = _ring_moments(ext)
    for interior in polygon.interiors:
        ring = np.asarray(interior.coords, dtype=float)
        if _signed_area(ring) > 0:
            ring = ring[::-1]
        a_, sx_, sy_, ixx_, iyy_, ixy_ = _ring_moments(ring)
        A += a_
        Sx += sx_
        Sy += sy_
        Ixx += ixx_
        Iyy += iyy_
        Ixy += ixy_
    return A, Sx, Sy, Ixx, Iyy, Ixy


# ==================================================================
#  Input data class
# ==================================================================


@dataclass(frozen=True)
class HomogenizedRebar:
    r"""
    Minimal description of a point fiber for homogenization.

    Parameters
    ----------
    x, y : float
        Location in the user frame :math:`[\mathrm{mm}]`.
    area : float
        Cross-sectional area :math:`A_s\,[\mathrm{mm}^2]`.
    E : float
        Elastic modulus :math:`E_s\,[\mathrm{MPa}]`.
    """
    x: float
    y: float
    area: float
    E: float


# ==================================================================
#  Output data class
# ==================================================================


@dataclass(frozen=True)
class SectionProperties:
    r"""
    Bundle of homogenized geometric properties of a cross-section.

    All second-moment quantities are reported in
    :math:`\mathrm{mm}^4`, areas in :math:`\mathrm{mm}^2`, lengths
    in :math:`\mathrm{mm}`, and the principal-axis orientation
    angle in radians.  All integral quantities (A, S, I, Z) are
    computed on the **homogenized** section, using the convention

    .. math::

        \mathrm{d}A_{\mathrm{id}} = n_{\mathrm{bulk}}\,\mathrm{d}A
        \quad\text{(on the polygon)},\qquad
        \Delta A_{s,\mathrm{id}} = (n_s - n_{\mathrm{bulk}})\,A_s
        \quad\text{(per point fiber)},

    with :math:`n_i = E_i / E_{\mathrm{ref}}`.

    Attributes
    ----------
    E_ref : float
        Reference modulus used for homogenization.
    E_bulk : float
        Elastic modulus of the polygon material.
    n_bulk : float
        :math:`E_{\mathrm{bulk}} / E_{\mathrm{ref}}` (= 1 by
        default).
    area : float
        Homogenized area :math:`A_{\mathrm{id}}`.
    Sx, Sy : float
        Static moments about the user :math:`x, y` axes through
        the origin.
    xg, yg : float
        Centroid of the homogenized section.
    Ixx_o, Iyy_o, Ixy_o : float
        Homogenized second-moments about the user origin.
    Ix, Iy, Ixy : float
        Homogenized centroidal second-moments in the user frame.
    I_xi, I_eta : float
        Homogenized principal centroidal second-moments
        :math:`I_\xi \ge I_\eta`.
    alpha : float
        Rotation (CCW, radians) from the user :math:`x` axis to
        the principal :math:`\xi` axis, in
        :math:`(-\pi/2, +\pi/2]`.
    rho_x, rho_y, rho_xi, rho_eta : float
        Radii of gyration.
    I_polar : float
        :math:`I_p = I_x + I_y = I_\xi + I_\eta`.
    is_convex : bool
        Whether the exterior ring is convex.
    c_y_top, c_y_bot : float
        Distances from the centroid to the extreme :math:`y`
        fibers (always :math:`\ge 0`).  Polygon vertices *and*
        point-fiber positions are both considered.
    c_x_left, c_x_right : float
        Same for :math:`x`.
    c_xi_pos, c_xi_neg : float
        Distances to the extreme fibers along :math:`\xi`.
    c_eta_pos, c_eta_neg : float
        Distances to the extreme fibers along :math:`\eta`.
    W_x_top, W_x_bot : float
        Elastic section moduli for bending about :math:`x`:
        :math:`W_x^{\mathrm{top}} = I_x / c_y^{\mathrm{top}}`,
        :math:`W_x^{\mathrm{bot}} = I_x / c_y^{\mathrm{bot}}`.
    W_y_left, W_y_right : float
        Analogous for bending about :math:`y`.
    W_xi_pos, W_xi_neg : float
        Elastic moduli about the principal :math:`\xi` axis:
        :math:`W_\xi = I_\xi / c_\eta`.
    W_eta_pos, W_eta_neg : float
        Elastic moduli about the principal :math:`\eta` axis:
        :math:`W_\eta = I_\eta / c_\xi`.
    Z_x, Z_y, Z_xi, Z_eta : float
        Plastic section moduli of the homogenized section about
        the four reference axes through the centroid.  The
        plastic neutral axis is the line that splits
        :math:`A_{\mathrm{id}}` in two equal halves; :math:`Z`
        is the sum of the absolute first moments of the two
        halves about the PNA.
    I_t : float or None
        Torsional constant.  Not yet computed (deferred to a
        future St.-Venant solver); kept in the schema so that
        JSON export and the future GUI have a stable layout.
    """
    # ---- Homogenization ----
    E_ref: float
    E_bulk: float
    n_bulk: float
    # ---- Area and centroid ----
    area: float
    Sx: float
    Sy: float
    xg: float
    yg: float
    # ---- Second moments ----
    Ixx_o: float
    Iyy_o: float
    Ixy_o: float
    Ix: float
    Iy: float
    Ixy: float
    I_xi: float
    I_eta: float
    alpha: float
    # ---- Derived ----
    rho_x: float
    rho_y: float
    rho_xi: float
    rho_eta: float
    I_polar: float
    is_convex: bool
    # ---- Extreme-fiber distances ----
    c_y_top: float
    c_y_bot: float
    c_x_left: float
    c_x_right: float
    c_xi_pos: float
    c_xi_neg: float
    c_eta_pos: float
    c_eta_neg: float
    # ---- Elastic section moduli ----
    W_x_top: float
    W_x_bot: float
    W_y_left: float
    W_y_right: float
    W_xi_pos: float
    W_xi_neg: float
    W_eta_pos: float
    W_eta_neg: float
    # ---- Plastic section moduli ----
    Z_x: float
    Z_y: float
    Z_xi: float
    Z_eta: float
    # ---- Torsional constant (placeholder) ----
    I_t: Optional[float] = None


# ==================================================================
#  Helpers: extreme-fiber distances
# ==================================================================


def _extreme_distances_along_direction(polygon, rebars, xg, yg,
                                        direction_rad):
    r"""
    Signed extreme fiber distances along a unit direction through
    the centroid.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
    rebars : list of HomogenizedRebar
    xg, yg : float
    direction_rad : float

    Returns
    -------
    tuple of float
        ``(c_pos, c_neg)`` with :math:`c_{\pm} \ge 0`.
    """
    c = np.cos(direction_rad)
    s = np.sin(direction_rad)
    verts = [np.asarray(polygon.exterior.coords, dtype=float)]
    for interior in polygon.interiors:
        verts.append(np.asarray(interior.coords, dtype=float))
    all_pts = np.vstack(verts)
    if rebars:
        rb_pts = np.array([(r.x, r.y) for r in rebars], dtype=float)
        all_pts = np.vstack([all_pts, rb_pts])
    proj = (all_pts[:, 0] - xg) * c + (all_pts[:, 1] - yg) * s
    c_pos = float(max(proj.max(), 0.0))
    c_neg = float(max(-proj.min(), 0.0))
    return c_pos, c_neg


# ==================================================================
#  Helpers: plastic modulus via bisection
# ==================================================================


def _plastic_modulus_along_direction(polygon, rebars,
                                      xg, yg, direction_rad,
                                      n_bulk, E_ref,
                                      max_iter=80,
                                      tol_rel=1.0e-10):
    r"""
    Compute the plastic section modulus of the homogenized section
    for bending about an axis through :math:`G` oriented at
    ``direction_rad`` (angle measured CCW from user :math:`x`).

    The plastic neutral axis (PNA) is the line perpendicular to
    the stress-gradient direction that splits the homogenized area
    in two equal halves:

    .. math::

        A_{\mathrm{id}}^+(t_{\mathrm{pna}})
        = A_{\mathrm{id}}^-(t_{\mathrm{pna}})
        = A_{\mathrm{id}}/2.

    The plastic modulus is

    .. math::

        Z = \bigl| S_{\mathrm{id}}^+ \bigr|
          + \bigl| S_{\mathrm{id}}^- \bigr|,

    with :math:`S^{\pm}` the first moments of the two halves about
    the PNA.

    Implementation: rotate the polygon and the rebars about
    :math:`G` so that the bending axis aligns with local
    :math:`X` (PNA horizontal in the local frame), then bisect on
    :math:`Y_{\mathrm{pna}}` using Shapely half-plane intersection
    for the polygon and summing rebar contributions analytically.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
    rebars : list of HomogenizedRebar
    xg, yg : float
    direction_rad : float
        Angle from user-:math:`x` to the **bending axis**.
    n_bulk : float
    E_ref : float
    max_iter : int, optional
    tol_rel : float, optional

    Returns
    -------
    float
        Plastic section modulus :math:`Z` in :math:`\mathrm{mm}^3`.
    """
    # Rotate by -angle so bending axis aligns with local X.
    angle_deg = -float(np.degrees(direction_rad))
    poly_loc = _rotate(polygon, angle_deg, origin=(xg, yg))
    poly_loc = _translate(poly_loc, xoff=-xg, yoff=-yg)
    reb_loc = []
    c = np.cos(-direction_rad)
    s = np.sin(-direction_rad)
    for r in rebars or []:
        dx = r.x - xg
        dy = r.y - yg
        x_loc = c * dx - s * dy
        y_loc = s * dx + c * dy
        reb_loc.append((x_loc, y_loc, r.area, r.E))

    # Total homogenized area.
    A_poly, *_ = _polygon_moments_about_origin(poly_loc)
    A_poly_id = A_poly * n_bulk
    A_rebars_id = 0.0
    for _, _, As, Es in reb_loc:
        n_s = Es / E_ref
        A_rebars_id += As * (n_s - n_bulk)
    A_total_id = A_poly_id + A_rebars_id
    if A_total_id <= 0.0:
        return 0.0
    A_target = 0.5 * A_total_id

    minx, miny, maxx, maxy = poly_loc.bounds
    if reb_loc:
        ry = [r[1] for r in reb_loc]
        miny = min(miny, min(ry))
        maxy = max(maxy, max(ry))
    extent = max(abs(maxx - minx), abs(maxy - miny)) + 1.0
    slab_xmin = minx - 10.0 * extent
    slab_xmax = maxx + 10.0 * extent

    def area_below(y_cut):
        """Homogenized area strictly below the line y=y_cut."""
        below_poly = poly_loc.intersection(
            sbox(slab_xmin, miny - 10.0 * extent,
                 slab_xmax, y_cut))
        A_b_poly = below_poly.area * n_bulk
        A_b_reb = 0.0
        for _, yr, As, Es in reb_loc:
            if yr < y_cut:
                n_s = Es / E_ref
                A_b_reb += As * (n_s - n_bulk)
        return A_b_poly + A_b_reb

    # Bisection on y_pna.
    lo, hi = miny, maxy
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        A_b = area_below(mid)
        if abs(A_b - A_target) < tol_rel * A_total_id:
            break
        if A_b < A_target:
            lo = mid
        else:
            hi = mid
    y_pna = 0.5 * (lo + hi)

    # First moments of the two halves about the PNA.
    below = poly_loc.intersection(
        sbox(slab_xmin, miny - 10.0 * extent, slab_xmax, y_pna))
    above = poly_loc.intersection(
        sbox(slab_xmin, y_pna, slab_xmax, maxy + 10.0 * extent))

    def _first_moment_y(geom, y_ref):
        """Integral of (y - y_ref) dA over a polygonal geom."""
        if geom.is_empty:
            return 0.0
        total = 0.0
        geoms = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
        for g in geoms:
            if g.is_empty or g.area == 0.0:
                continue
            A_g, Sx_g, _, _, _, _ = _polygon_moments_about_origin(g)
            total += Sx_g - y_ref * A_g
        return total

    S_above_poly = _first_moment_y(above, y_pna) * n_bulk
    S_below_poly = _first_moment_y(below, y_pna) * n_bulk
    S_above_reb = 0.0
    S_below_reb = 0.0
    for _, yr, As, Es in reb_loc:
        n_s = Es / E_ref
        dA_id = As * (n_s - n_bulk)
        dy = yr - y_pna
        if dy > 0.0:
            S_above_reb += dA_id * dy
        elif dy < 0.0:
            S_below_reb += dA_id * dy

    S_above = S_above_poly + S_above_reb
    S_below = S_below_poly + S_below_reb
    return float(abs(S_above) + abs(S_below))


# ==================================================================
#  Public API
# ==================================================================


def compute_section_properties(
    polygon: Polygon,
    rebars: Optional[Sequence[HomogenizedRebar]] = None,
    E_bulk: float = 1.0,
    E_ref: Optional[float] = None,
    compute_plastic: bool = True,
) -> SectionProperties:
    r"""
    Compute the homogenized geometric properties of a section.

    Every polygon area element contributes as
    :math:`n_{\mathrm{bulk}}\,\mathrm{d}A`, and every point fiber
    contributes as :math:`(n_s - n_{\mathrm{bulk}}) A_s`.  In the
    typical RC case one takes :math:`E_{\mathrm{ref}} =
    E_{\mathrm{bulk}} = E_{\mathrm{cm}}`, giving
    :math:`n_{\mathrm{bulk}} = 1` and :math:`n_s = E_s /
    E_{\mathrm{cm}}`.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Section outline with any number of holes.
    rebars : sequence of HomogenizedRebar, optional
        Point fibers.  If ``None`` or empty, the section is
        treated as pure bulk material.
    E_bulk : float, optional
        Elastic modulus of the polygon material.  Default ``1.0``.
    E_ref : float or None, optional
        Reference modulus.  If ``None``, defaults to ``E_bulk``.
    compute_plastic : bool, optional
        Whether to compute the plastic moduli.  Default ``True``.

    Returns
    -------
    SectionProperties

    Raises
    ------
    ValueError
        If the polygon is empty / invalid / has non-positive area,
        or if any modulus is non-positive.
    """
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError(
            "Polygon is empty or invalid; cannot compute properties.")
    if E_bulk <= 0.0:
        raise ValueError(f"E_bulk must be positive, got {E_bulk}.")
    if E_ref is None:
        E_ref = E_bulk
    if E_ref <= 0.0:
        raise ValueError(f"E_ref must be positive, got {E_ref}.")

    n_bulk = E_bulk / E_ref
    rebars = list(rebars) if rebars else []

    # ---- Polygon integrals. -----------------------------------
    A_p, Sx_p, Sy_p, Ixx_p, Iyy_p, Ixy_p = \
        _polygon_moments_about_origin(polygon)
    if A_p <= 0.0:
        raise ValueError(
            f"Non-positive polygon area ({A_p:.3e}).")

    A = A_p * n_bulk
    Sx = Sx_p * n_bulk
    Sy = Sy_p * n_bulk
    Ixx = Ixx_p * n_bulk
    Iyy = Iyy_p * n_bulk
    Ixy = Ixy_p * n_bulk

    # ---- Differential contributions of point fibers. ----------
    for r in rebars:
        n_s = r.E / E_ref
        dA = r.area * (n_s - n_bulk)
        A += dA
        Sx += dA * r.y
        Sy += dA * r.x
        Ixx += dA * r.y * r.y
        Iyy += dA * r.x * r.x
        Ixy += dA * r.x * r.y

    if A <= 0.0:
        raise ValueError(
            f"Non-positive homogenized area ({A:.3e}).")

    xg = Sy / A
    yg = Sx / A

    Ix_c = Ixx - A * yg * yg
    Iy_c = Iyy - A * xg * xg
    Ixy_c = Ixy - A * xg * yg

    # ---- Principal moments and orientation. -------------------
    mean = 0.5 * (Ix_c + Iy_c)
    diff = 0.5 * (Ix_c - Iy_c)
    radius = float(np.hypot(diff, Ixy_c))
    I_major = mean + radius
    I_minor = mean - radius

    if radius <= 1.0e-10 * abs(mean):
        alpha = 0.0
    else:
        alpha = 0.5 * float(np.arctan2(-2.0 * Ixy_c, Ix_c - Iy_c))
    I_minor = max(I_minor, 0.0)
    I_major = max(I_major, I_minor)

    rho_x = float(np.sqrt(max(Ix_c, 0.0) / A))
    rho_y = float(np.sqrt(max(Iy_c, 0.0) / A))
    rho_xi = float(np.sqrt(I_major / A))
    rho_eta = float(np.sqrt(I_minor / A))

    # ---- Convexity. -------------------------------------------
    ext_poly = Polygon(polygon.exterior)
    hull = ext_poly.convex_hull
    is_convex = (ext_poly.symmetric_difference(hull).area
                 < 1.0e-6 * ext_poly.area)

    # ---- Extreme-fiber distances. -----------------------------
    c_x_right, c_x_left = _extreme_distances_along_direction(
        polygon, rebars, xg, yg, 0.0)
    c_y_top, c_y_bot = _extreme_distances_along_direction(
        polygon, rebars, xg, yg, np.pi / 2)
    c_xi_pos, c_xi_neg = _extreme_distances_along_direction(
        polygon, rebars, xg, yg, alpha)
    c_eta_pos, c_eta_neg = _extreme_distances_along_direction(
        polygon, rebars, xg, yg, alpha + np.pi / 2)

    # ---- Elastic section moduli. ------------------------------
    def _safe_div(I_val, c):
        return float(I_val / c) if c > 0.0 else float('inf')

    W_x_top = _safe_div(Ix_c, c_y_top)
    W_x_bot = _safe_div(Ix_c, c_y_bot)
    W_y_left = _safe_div(Iy_c, c_x_left)
    W_y_right = _safe_div(Iy_c, c_x_right)
    W_xi_pos = _safe_div(I_major, c_eta_pos)
    W_xi_neg = _safe_div(I_major, c_eta_neg)
    W_eta_pos = _safe_div(I_minor, c_xi_pos)
    W_eta_neg = _safe_div(I_minor, c_xi_neg)

    # ---- Plastic section moduli. ------------------------------
    if compute_plastic:
        Z_x = _plastic_modulus_along_direction(
            polygon, rebars, xg, yg, 0.0, n_bulk, E_ref)
        Z_y = _plastic_modulus_along_direction(
            polygon, rebars, xg, yg, np.pi / 2, n_bulk, E_ref)
        Z_xi = _plastic_modulus_along_direction(
            polygon, rebars, xg, yg, alpha, n_bulk, E_ref)
        Z_eta = _plastic_modulus_along_direction(
            polygon, rebars, xg, yg, alpha + np.pi / 2,
            n_bulk, E_ref)
    else:
        Z_x = Z_y = Z_xi = Z_eta = float('nan')

    return SectionProperties(
        E_ref=E_ref, E_bulk=E_bulk, n_bulk=n_bulk,
        area=A, Sx=Sx, Sy=Sy, xg=xg, yg=yg,
        Ixx_o=Ixx, Iyy_o=Iyy, Ixy_o=Ixy,
        Ix=Ix_c, Iy=Iy_c, Ixy=Ixy_c,
        I_xi=I_major, I_eta=I_minor, alpha=alpha,
        rho_x=rho_x, rho_y=rho_y, rho_xi=rho_xi, rho_eta=rho_eta,
        I_polar=Ix_c + Iy_c, is_convex=bool(is_convex),
        c_y_top=c_y_top, c_y_bot=c_y_bot,
        c_x_left=c_x_left, c_x_right=c_x_right,
        c_xi_pos=c_xi_pos, c_xi_neg=c_xi_neg,
        c_eta_pos=c_eta_pos, c_eta_neg=c_eta_neg,
        W_x_top=W_x_top, W_x_bot=W_x_bot,
        W_y_left=W_y_left, W_y_right=W_y_right,
        W_xi_pos=W_xi_pos, W_xi_neg=W_xi_neg,
        W_eta_pos=W_eta_pos, W_eta_neg=W_eta_neg,
        Z_x=Z_x, Z_y=Z_y, Z_xi=Z_xi, Z_eta=Z_eta,
        I_t=None,
    )


# ==================================================================
#  Ellipse and kern (unchanged in signature and math)
# ==================================================================


def compute_inertia_ellipse(props, n_points=240):
    r"""
    Sample points on the central inertia ellipse (Culmann).

    In the centroidal principal frame :math:`(\xi, \eta)`,

    .. math::

        \frac{\xi^2}{\rho_\eta^2} + \frac{\eta^2}{\rho_\xi^2} = 1.

    Parameters
    ----------
    props : SectionProperties
    n_points : int, optional

    Returns
    -------
    numpy.ndarray of shape ``(n_points, 2)``
    """
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=True)
    xi_p = props.rho_eta * np.cos(t)
    eta_p = props.rho_xi * np.sin(t)
    c = np.cos(props.alpha)
    s = np.sin(props.alpha)
    x = props.xg + c * xi_p - s * eta_p
    y = props.yg + s * xi_p + c * eta_p
    return np.column_stack([x, y])


def compute_kern_polygon(polygon, props):
    r"""
    Compute the central kern (core) of the cross-section.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
    props : SectionProperties

    Returns
    -------
    numpy.ndarray of shape ``(N+1, 2)`` or ``(0, 2)``
    """
    ext_poly = Polygon(polygon.exterior)
    hull = ext_poly.convex_hull
    hull_coords = np.asarray(hull.exterior.coords, dtype=float)
    if _signed_area(hull_coords) < 0:
        hull_coords = hull_coords[::-1]
    c = np.cos(props.alpha)
    s = np.sin(props.alpha)
    dx = hull_coords[:, 0] - props.xg
    dy = hull_coords[:, 1] - props.yg
    xi = c * dx + s * dy
    eta = -s * dx + c * dy
    rho_xi2 = props.rho_xi ** 2
    rho_eta2 = props.rho_eta ** 2
    kern_xi: List[float] = []
    kern_eta: List[float] = []
    for i in range(len(xi) - 1):
        xi1, eta1 = xi[i], eta[i]
        xi2, eta2 = xi[i + 1], eta[i + 1]
        ex = xi2 - xi1
        ey = eta2 - eta1
        nx = ey
        ny = -ex
        L = float(np.hypot(nx, ny))
        if L < 1.0e-12:
            continue
        nx /= L
        ny /= L
        d = nx * xi1 + ny * eta1
        if d <= 1.0e-12:
            continue
        kern_xi.append(-rho_eta2 * nx / d)
        kern_eta.append(-rho_xi2 * ny / d)
    if len(kern_xi) < 3:
        return np.empty((0, 2))
    kern_xi.append(kern_xi[0])
    kern_eta.append(kern_eta[0])
    xi_arr = np.asarray(kern_xi)
    eta_arr = np.asarray(kern_eta)
    x = props.xg + c * xi_arr - s * eta_arr
    y = props.yg + s * xi_arr + c * eta_arr
    return np.column_stack([x, y])
