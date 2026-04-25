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
Generic cross-section geometry with automatic fiber meshing.

Replaces the former ``RectSection`` with a fully general approach:
any cross-section defined as a Shapely polygon (with holes) is
automatically discretized into a fiber mesh suitable for
:class:`~gensec.solver.FiberSolver`.

Two meshing strategies are available:

- **Grid** (default): rectangular grid clipped to the polygon
  boundary. Fast, simple, good for convex or mildly non-convex
  sections. Resolution controlled by ``mesh_size`` [mm].
- **Triangular**: Constrained Delaunay triangulation via the
  ``triangle`` library. Better for curved boundaries and complex
  holes. Requires ``triangle`` to be installed.

The class exposes the same attribute interface that
:class:`~gensec.solver.FiberSolver` consumes, so the solver,
capacity generator, and plotting modules need no changes.

Dependencies
------------
- ``shapely >= 2.0``
- ``triangle`` (optional, for triangular meshing)
- ``numpy``

Examples
--------
Build a rectangular section the quick way:

>>> from gensec.geometry.section import GenericSection
>>> from gensec.geometry.fiber import RebarLayer
>>> from gensec.materials.concrete import Concrete
>>> from gensec.materials.steel import Steel
>>> from shapely.geometry import box
>>> poly = box(0, 0, 300, 600)
>>> sec = GenericSection(poly, Concrete(fck=25), [], mesh_size=10)
>>> sec.n_fibers
1800

Build a circular section with a square hole:

>>> from shapely.geometry import Point, box
>>> outer = Point(0, 0).buffer(500, resolution=64)
>>> hole = box(-100, -100, 100, 100)
>>> poly = outer.difference(hole)
>>> sec = GenericSection(poly, Concrete(fck=30), [], mesh_size=20)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Literal

from shapely.geometry import Polygon, MultiPolygon, box as shapely_box
from shapely.ops import unary_union
from shapely import affinity

from .fiber import RebarLayer
from ..materials.base import Material


@dataclass
class GenericSection:
    r"""
    Cross-section defined by an arbitrary polygon, meshed into fibers.

    The section geometry is a :class:`shapely.geometry.Polygon` (which
    may contain holes). The bulk material is assigned to all fibers
    inside the polygon boundary. Point fibers (rebars, tendons, FRP
    strips) are placed independently.

    Coordinate system
    -----------------
    User-defined. Typically:

    - :math:`x` horizontal, :math:`y` vertical (upward).
    - Origin at bottom-left corner or at centroid, depending on the
      factory function used.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Section outline (may include holes).
    bulk_material : Material
        Constitutive law for the bulk (concrete, timber, ...).
    rebars : list of RebarLayer
        Point fibers with their own materials.
    mesh_size : float, optional
        Target fiber size [mm]. For grid meshing, this is the cell
        side length. For triangular meshing, this controls the
        maximum triangle area :math:`\approx 0.5 \cdot
        \text{mesh\_size}^2`. Default 10.0.
    mesh_method : ``'grid'`` or ``'triangle'``, optional
        Meshing strategy. Default ``'grid'``.
    bulk_materials : list of tuple, optional
        Additional bulk material zones. Each tuple is
        ``(Polygon, Material)``. Fibers inside each zone use
        that zone's material instead of ``bulk_material``.
        Zones are checked in order; first match wins.
        Default empty (single-material section).
    n_grid_x : int or None, optional
        Explicit number of grid columns for the ``'grid'`` mesher.
        When set, overrides the ``mesh_size``-based computation
        for the x-direction.  Default ``None`` (derive from
        ``mesh_size``).
    n_grid_y : int or None, optional
        Explicit number of grid rows.  Default ``None``.

    Attributes
    ----------
    x_fibers : numpy.ndarray
        x-coordinates of bulk fiber centroids [mm].
    y_fibers : numpy.ndarray
        y-coordinates [mm].
    A_fibers : numpy.ndarray
        Fiber areas [mm²].
    mat_indices : numpy.ndarray of int
        Material index for each bulk fiber. ``0`` = ``bulk_material``,
        ``1..N`` = zones from ``bulk_materials`` list.
    x_rebars : numpy.ndarray
        x-coordinates of rebar layers [mm].
    y_rebars : numpy.ndarray
        y-coordinates [mm].
    A_rebars : numpy.ndarray
        Areas [mm²].
    embedded_rebars : numpy.ndarray of bool
    n_fibers : int
        Total number of bulk fibers.
    B : float
        Bounding box width (x-direction) [mm].
    H : float
        Bounding box height (y-direction) [mm].
    n_fibers_x : int
        Number of grid columns (grid mesh only; triangular sets -1).
    n_fibers_y : int
        Number of grid rows (grid mesh only; triangular sets -1).
    ideal_gross_area : float
        ideal_gross polygon area [mm²].

    Raises
    ------
    ValueError
        If the polygon is empty, invalid, or has zero area.
    ImportError
        If ``mesh_method='triangle'`` but the ``triangle`` package
        is not installed.
    """

    polygon: Polygon
    bulk_material: Material
    rebars: List[RebarLayer]
    mesh_size: float = 10.0
    mesh_method: Literal["grid", "triangle"] = "grid"
    bulk_materials: List[tuple] = field(default_factory=list)
    n_grid_x: Optional[int] = None
    n_grid_y: Optional[int] = None

    def __post_init__(self):
        # ---- Validate polygon ----
        if self.polygon.is_empty or not self.polygon.is_valid:
            raise ValueError(
                "Section polygon is empty or invalid. "
                "Check vertex order and self-intersections."
            )
        if self.polygon.area < 1e-6:
            raise ValueError(
                f"Section polygon area is {self.polygon.area:.2e} mm² "
                f"— effectively zero."
            )

        # Ensure single Polygon (not MultiPolygon)
        if isinstance(self.polygon, MultiPolygon):
            # Take the largest component
            self.polygon = max(self.polygon.geoms, key=lambda g: g.area)

        self.ideal_gross_area = self.polygon.area

        # ---- Bounding box properties ----
        minx, miny, maxx, maxy = self.polygon.bounds
        self.B = maxx - minx
        self.H = maxy - miny
        self._bounds = (minx, miny, maxx, maxy)

        # ---- Mesh ----
        if self.mesh_method == "grid":
            self._mesh_grid()
        elif self.mesh_method == "triangle":
            self._mesh_triangular()
        else:
            raise ValueError(
                f"Unknown mesh_method '{self.mesh_method}'. "
                f"Use 'grid' or 'triangle'."
            )

        # ---- Rebars ----
        self._setup_rebars()

    # ------------------------------------------------------------------
    #  Grid meshing
    # ------------------------------------------------------------------

    def _mesh_grid(self):
        r"""
        Rectangular grid meshing clipped to the polygon boundary.

        Each grid cell is intersected with the polygon. If the
        intersection is non-empty, a fiber is created at the
        intersection centroid with the intersection area. This
        correctly handles partial cells at the boundary and cells
        spanning holes.

        When ``n_grid_x`` / ``n_grid_y`` are explicitly set, they
        override the ``mesh_size``-based computation.  Otherwise
        the grid resolution is:

        .. math::

            n_x = \max\!\left(1,\;\left\lceil B / s \right\rceil\right),
            \quad
            n_y = \max\!\left(1,\;\left\lceil H / s \right\rceil\right)

        where :math:`s` = ``mesh_size``.
        """
        minx, miny, maxx, maxy = self._bounds
        s = self.mesh_size

        if self.n_grid_x is not None:
            self.n_fibers_x = max(1, self.n_grid_x)
        else:
            self.n_fibers_x = max(1, int(np.ceil(self.B / s)))
        if self.n_grid_y is not None:
            self.n_fibers_y = max(1, self.n_grid_y)
        else:
            self.n_fibers_y = max(1, int(np.ceil(self.H / s)))

        dx = self.B / self.n_fibers_x
        dy = self.H / self.n_fibers_y
        self.dx = dx
        self.dy = dy

        xc_list = []
        yc_list = []
        area_list = []
        mat_list = []

        poly = self.polygon  # local reference for speed

        for iy in range(self.n_fibers_y):
            y0 = miny + iy * dy
            y1 = y0 + dy
            for ix in range(self.n_fibers_x):
                x0 = minx + ix * dx
                x1 = x0 + dx

                cell = shapely_box(x0, y0, x1, y1)
                clipped = poly.intersection(cell)

                if clipped.is_empty or clipped.area < 1e-10:
                    continue

                centroid = clipped.centroid
                xc_list.append(centroid.x)
                yc_list.append(centroid.y)
                area_list.append(clipped.area)
                mat_list.append(self._material_index(centroid.x,
                                                     centroid.y))

        self.x_fibers = np.array(xc_list, dtype=float)
        self.y_fibers = np.array(yc_list, dtype=float)
        self.A_fibers = np.array(area_list, dtype=float)
        self.mat_indices = np.array(mat_list, dtype=int)
        self.n_fibers = len(self.x_fibers)

    # ------------------------------------------------------------------
    #  Triangular meshing
    # ------------------------------------------------------------------

    def _mesh_triangular(self):
        r"""
        Constrained Delaunay triangulation using the ``triangle``
        library.

        The polygon boundary (exterior + holes) is triangulated with
        a maximum triangle area constraint:

        .. math::

            A_{\max} = 0.5 \cdot s^2

        where :math:`s` = ``mesh_size``. Each triangle becomes one
        fiber at its centroid.

        Raises
        ------
        ImportError
            If the ``triangle`` package is not installed.
        """
        try:
            import triangle as tr
        except ImportError:
            raise ImportError(
                "Triangular meshing requires the 'triangle' package. "
                "Install with: pip install triangle"
            )

        # Build PSLG (Planar Straight Line Graph) from Shapely polygon
        vertices, segments, holes = self._polygon_to_pslg()

        max_area = 0.5 * self.mesh_size ** 2
        pslg = {
            "vertices": vertices,
            "segments": segments,
        }
        if len(holes) > 0:
            pslg["holes"] = holes

        # 'p' = triangulate PSLG
        # 'q' = quality mesh (min angle 20°)
        # 'a' = area constraint
        tri = tr.triangulate(pslg, f"pq20a{max_area:.6f}")

        tri_verts = tri["vertices"]       # (n_verts, 2)
        tri_elems = tri["triangles"]      # (n_tri, 3)

        xc_list = []
        yc_list = []
        area_list = []
        mat_list = []

        for elem in tri_elems:
            v = tri_verts[elem]            # (3, 2)
            cx = v[:, 0].mean()
            cy = v[:, 1].mean()
            # Signed area via cross product
            area = 0.5 * abs(
                (v[1, 0] - v[0, 0]) * (v[2, 1] - v[0, 1])
                - (v[2, 0] - v[0, 0]) * (v[1, 1] - v[0, 1])
            )
            if area < 1e-10:
                continue

            xc_list.append(cx)
            yc_list.append(cy)
            area_list.append(area)
            mat_list.append(self._material_index(cx, cy))

        self.x_fibers = np.array(xc_list, dtype=float)
        self.y_fibers = np.array(yc_list, dtype=float)
        self.A_fibers = np.array(area_list, dtype=float)
        self.mat_indices = np.array(mat_list, dtype=int)
        self.n_fibers = len(self.x_fibers)
        self.n_fibers_x = -1  # not applicable
        self.n_fibers_y = -1
        self.dx = self.mesh_size
        self.dy = self.mesh_size

    def _polygon_to_pslg(self):
        r"""
        Convert the Shapely polygon to a PSLG for the ``triangle``
        library.

        Returns
        -------
        vertices : numpy.ndarray
            Shape ``(n, 2)``.
        segments : numpy.ndarray
            Shape ``(m, 2)`` — vertex index pairs.
        holes : numpy.ndarray
            Shape ``(h, 2)`` — one interior point per hole.
        """
        coords_all = []
        segs_all = []
        holes_pts = []
        offset = 0

        # Exterior ring
        ext_coords = np.array(self.polygon.exterior.coords[:-1])
        n_ext = len(ext_coords)
        coords_all.append(ext_coords)
        for i in range(n_ext):
            segs_all.append([offset + i, offset + (i + 1) % n_ext])
        offset += n_ext

        # Interior rings (holes)
        for interior in self.polygon.interiors:
            ring_coords = np.array(interior.coords[:-1])
            n_ring = len(ring_coords)
            coords_all.append(ring_coords)
            for i in range(n_ring):
                segs_all.append([offset + i,
                                 offset + (i + 1) % n_ring])
            offset += n_ring

            # A point inside the hole for triangle's hole marker.
            # Use the centroid of the hole ring (works for convex
            # holes; for non-convex, use representative_point).
            hole_poly = Polygon(interior.coords)
            rep = hole_poly.representative_point()
            holes_pts.append([rep.x, rep.y])

        vertices = np.vstack(coords_all)
        segments = np.array(segs_all, dtype=int)
        holes = np.array(holes_pts) if holes_pts else np.empty((0, 2))

        return vertices, segments, holes

    # ------------------------------------------------------------------
    #  Multi-material zone support
    # ------------------------------------------------------------------

    def _material_index(self, x, y):
        r"""
        Determine the material index for a fiber at ``(x, y)``.

        Checks ``bulk_materials`` zones in order. Returns 0 if no
        zone claims the point (falls back to ``bulk_material``).

        Parameters
        ----------
        x, y : float
            Fiber centroid coordinates [mm].

        Returns
        -------
        int
            ``0`` for ``bulk_material``, ``1..N`` for zones.
        """
        from shapely.geometry import Point
        pt = Point(x, y)
        for i, (zone_poly, _) in enumerate(self.bulk_materials):
            if zone_poly.contains(pt):
                return i + 1
        return 0

    def get_material_for_fiber(self, fiber_index):
        r"""
        Return the material object for a given bulk fiber index.

        Parameters
        ----------
        fiber_index : int

        Returns
        -------
        Material
        """
        mi = self.mat_indices[fiber_index]
        if mi == 0:
            return self.bulk_material
        return self.bulk_materials[mi - 1][1]

    def get_all_bulk_materials(self):
        r"""
        Return the list of all bulk materials (base + zones).

        Returns
        -------
        list of Material
            Index-aligned with ``mat_indices`` values.
        """
        mats = [self.bulk_material]
        for _, mat in self.bulk_materials:
            mats.append(mat)
        return mats

    # ------------------------------------------------------------------
    #  Rebar setup
    # ------------------------------------------------------------------

    def _setup_rebars(self):
        """
        Finalize rebar arrays. If a rebar has ``x=None``, default to
        the section x-centroid.
        """
        xc = self.x_centroid
        for r in self.rebars:
            if r.x is None:
                r.x = xc

        if self.rebars:
            self.x_rebars = np.array([r.x for r in self.rebars],
                                     dtype=float)
            self.y_rebars = np.array([r.y for r in self.rebars],
                                     dtype=float)
            self.A_rebars = np.array([r.As for r in self.rebars],
                                     dtype=float)
            self.embedded_rebars = np.array(
                [r.embedded for r in self.rebars], dtype=bool)
        else:
            self.x_rebars = np.empty(0, dtype=float)
            self.y_rebars = np.empty(0, dtype=float)
            self.A_rebars = np.empty(0, dtype=float)
            self.embedded_rebars = np.empty(0, dtype=bool)

    # ------------------------------------------------------------------
    #  Geometric properties
    # ------------------------------------------------------------------

    @property
    def x_centroid(self):
        r"""
        ideal_gross centroid x-coordinate [mm].

        Computed from the Shapely polygon centroid (exact for
        arbitrary polygons, unlike the :math:`B/2` approximation
        of the former ``RectSection``).
        """
        return self.polygon.centroid.x

    @property
    def y_centroid(self):
        r"""ideal_gross centroid y-coordinate [mm]."""
        return self.polygon.centroid.y

    @property
    def bbox(self):
        r"""
        Bounding box as ``(minx, miny, maxx, maxy)``.

        Returns
        -------
        tuple of float
        """
        return self._bounds

    ### TODO: we should add support for multi-staged sections,
    ### where the ideal_gross properties are dependent from the time
    ### of construction or load application. At the moment, the property
    ### is computed lazily and cached, but it assumes a immutable section. 
    ### If the section geometry or materials change, the cache should be invalidated.
    @property
    def ideal_gross_properties(self):
        """Lazy-computed homogenized section properties."""
        if getattr(self, '_ideal_gross_props_cache', None) is None:
            from .properties import (
                compute_section_properties, HomogenizedRebar,
            )
            homog = [
                ### TODO: generalize Es to another flag which is more generic. Es is only for steel, but
                ### how does it works with a generic rebar? Carbon? Fiberglass? 
                ### Maybe we can add a method to the RebarLayer class to return the appropriate 
                ### stiffness for homogenization, which by default returns Es but can be overridden 
                ### for different materials.
                ### At the moment, we stay on Es.
                HomogenizedRebar(r.x, r.y, r.As, r.material.Es)
                for r in self.rebars
                if r.embedded and r.x is not None
            ]
            ### TODO: add support for multi-material bulk zones here (currently ignored in homogenization)
            if len(self.bulk_materials) > 1:
                self._ideal_gross_props_cache = None  # placeholder to avoid repeated warnings
                raise NotImplementedError(
                    "Warning: ideal_gross_properties currently ignores "
                    "multi-material bulk zones."
                )
            else:
                ### TODO: generalize Ec to another flag which is more generic. Ec is only for steel, but
                ### how does it works with a generic bulk? Wood? Other? 
                ### At the moment, we stay on Ec.
                self._ideal_gross_props_cache = compute_section_properties(
                    self.polygon,
                    rebars=homog,
                    E_bulk=self.bulk_material.Ec,
                )
        return self._ideal_gross_props_cache

    # ------------------------------------------------------------------
    #  Mesh quality diagnostics
    # ------------------------------------------------------------------

    def mesh_summary(self):
        r"""
        Return a summary dict of mesh quality metrics.

        Returns
        -------
        dict
            Keys: ``n_fibers``, ``total_area``, ``ideal_gross_area``,
            ``area_error_pct``, ``min_fiber_area``,
            ``max_fiber_area``, ``mean_fiber_area``,
            ``mesh_method``, ``mesh_size``.
        """
        total = float(np.sum(self.A_fibers))
        return {
            "n_fibers": self.n_fibers,
            "total_area": total,
            "ideal_gross_area": self.ideal_gross_area,
            "area_error_pct": abs(total - self.ideal_gross_area)
                              / self.ideal_gross_area * 100
                              if self.ideal_gross_area > 0 else 0.0,
            "min_fiber_area": float(self.A_fibers.min())
                              if self.n_fibers > 0 else 0.0,
            "max_fiber_area": float(self.A_fibers.max())
                              if self.n_fibers > 0 else 0.0,
            "mean_fiber_area": float(self.A_fibers.mean())
                               if self.n_fibers > 0 else 0.0,
            "mesh_method": self.mesh_method,
            "mesh_size": self.mesh_size,
        }

    # ------------------------------------------------------------------
    #  Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"GenericSection(B={self.B:.1f}, H={self.H:.1f}, "
            f"n_fibers={self.n_fibers}, "
            f"ideal_gross_area={self.ideal_gross_area:.1f} mm², "
            f"mesh={self.mesh_method}@{self.mesh_size}mm, "
            f"n_rebars={len(self.rebars)})"
        )