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

from .fiber import RebarLayer, Tendon
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
        Additional bulk material zones.  Each entry is either
        ``(Polygon, Material)`` or ``(Polygon, Material, name)``.
        Fibers inside each zone use that zone's material instead of
        ``bulk_material``.  Zones are checked in order; first match
        wins.  Default empty (single-material section).

        Zone *names* (Phase 8) are the stable staging references of
        the ``section_ops`` ``activate_bulk`` schema.  Unnamed zones
        (2-tuples, or 3-tuples with ``name=None``) are auto-named
        ``zone_<k>`` with *k* the 1-based position in this list; the
        implicit zone ``0`` (``bulk_material``) is named ``base`` and
        is always active.  Names must be unique, non-numeric strings
        distinct from ``'base'`` (a numeric name would be ambiguous
        with the 1-based integer zone reference).  After construction
        the normalized 2-tuples are stored back on this attribute and
        the names are exposed on :attr:`zone_names`, index-aligned
        with ``mat_indices`` values.
    n_grid_x : int or None, optional
        Explicit number of grid columns for the ``'grid'`` mesher.
        When set, overrides the ``mesh_size``-based computation
        for the x-direction.  Default ``None`` (derive from
        ``mesh_size``).
    n_grid_y : int or None, optional
        Explicit number of grid rows.  Default ``None``.
    bulk_eps_init : float, optional
        Uniform locked-in pre-strain of the bulk material [-], tension
        positive (e.g. an imposed shrinkage/thermal field).  Default
        ``0.0``.  This is the bulk counterpart of a tendon's
        ``eps_pe``: a *resistance*-side imposed-strain offset (it
        belongs to the capacity hash, not to the demand path).  It is
        carried by :class:`~gensec.solver.section_state.SectionState`
        and quantized into
        :meth:`~gensec.solver.section_state.SectionState.capacity_hash`.

        .. note::
           Since the Phase-5 bulk-kernel patch the fiber integrator
           **consumes** this offset: the bulk constitutive law is
           evaluated at :math:`\varepsilon_{\text{sec}} +
           \varepsilon_{b,0}` at the scalar, tangent and batch sites
           (validated by ``run_bulk_prestrain_validation_new.py``), so
           a non-zero value moves the resistance domain, not only the
           cache identity.  As of Phase 8 the scalar is one term of
           the general per-fiber offset field: it is added on top of
           the per-zone locked-in datum planes carried by
           :attr:`~gensec.solver.section_state.SectionState.bulk_planes`.
           Sections that never set this field are unaffected.

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
    tendons: List["Tendon"] = field(default_factory=list)
    bulk_eps_init: float = 0.0

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

        # ---- Bulk zone normalization (Phase 8: named zones) ----
        # Must precede meshing: ``_material_index`` (called per fiber
        # during the mesh walk) unpacks the normalized 2-tuples.
        self._normalize_bulk_zones()

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

        # ---- Tendons (prestress, Phase 1: bonded) ----
        self._setup_tendons()

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

    def _normalize_bulk_zones(self):
        r"""
        Normalize ``bulk_materials`` entries and build the zone-name
        table.

        Accepts, per entry, either the legacy 2-tuple
        ``(Polygon, Material)`` or the Phase-8 3-tuple
        ``(Polygon, Material, name)``.  Entries are stored back as
        2-tuples (the internal contract every downstream consumer
        unpacks) and the names — auto-generated ``zone_<k>`` where not
        given — are collected on :attr:`zone_names`, index-aligned
        with ``mat_indices`` values (``zone_names[0] == 'base'``).

        Raises
        ------
        ValueError
            Malformed entry; non-string explicit name; the reserved
            name ``'base'``; a purely numeric name (ambiguous with the
            1-based integer zone reference of the staging schema); or
            a duplicate name.
        """
        norm, names = [], []
        for k, entry in enumerate(self.bulk_materials):
            entry_t = tuple(entry)
            if len(entry_t) == 2:
                zone_poly, zone_mat = entry_t
                name = None
            elif len(entry_t) == 3:
                zone_poly, zone_mat, name = entry_t
            else:
                raise ValueError(
                    f"bulk_materials[{k}]: expected (Polygon, Material)"
                    f" or (Polygon, Material, name), got a "
                    f"{len(entry_t)}-tuple."
                )
            if name is None:
                name = f"zone_{k + 1}"
            elif not isinstance(name, str):
                raise ValueError(
                    f"bulk_materials[{k}]: zone name must be a string, "
                    f"got {name!r}. Integer zone references are the "
                    f"1-based positions in this list and need no name."
                )
            if name == "base":
                raise ValueError(
                    f"bulk_materials[{k}]: 'base' is the reserved name "
                    f"of the implicit zone 0 (the primary "
                    f"bulk_material) and cannot name an explicit zone."
                )
            stripped = name.strip().lstrip("+-")
            if stripped.isdigit():
                raise ValueError(
                    f"bulk_materials[{k}]: zone name {name!r} is "
                    f"purely numeric and would be ambiguous with the "
                    f"1-based integer zone reference. Use a "
                    f"non-numeric name."
                )
            if name in names:
                raise ValueError(
                    f"bulk_materials[{k}]: duplicate zone name "
                    f"{name!r} — names used as staging references must "
                    f"be unique."
                )
            norm.append((zone_poly, zone_mat))
            names.append(name)
        self.bulk_materials = norm
        self.zone_names = ["base"] + names

    @property
    def n_zones(self):
        r"""
        Number of bulk zones, including the implicit base zone.

        Returns
        -------
        int
            ``1 + len(bulk_materials)`` — index-aligned with
            ``mat_indices`` values and :attr:`zone_names`.
        """
        return 1 + len(self.bulk_materials)

    def zone_index(self, ref):
        r"""
        Resolve a bulk-zone reference to its zone index.

        The staging schema references a zone either by *name*
        (:attr:`zone_names`; ``'base'`` is zone 0) or by its **1-based
        integer position** in the ``bulk_materials`` list (``0`` is
        the base zone).  Whether a given zone may be the target of an
        operation (e.g. zone 0 is never activatable) is enforced by
        the operation, not here.

        Parameters
        ----------
        ref : str or int
            Zone name or zone index.

        Returns
        -------
        int
            Zone index in ``[0, n_zones)``.

        Raises
        ------
        ValueError
            Unknown name, index out of range, or unsupported type.
            Booleans are rejected explicitly (YAML ``true`` is a
            :class:`bool`, an :class:`int` subclass — accepting it as
            zone 1 would mask an input error).
        """
        if isinstance(ref, bool):
            raise ValueError(
                f"zone reference must be a zone name (str) or a zone "
                f"index (int), got the boolean {ref!r}."
            )
        if isinstance(ref, str):
            try:
                return self.zone_names.index(ref)
            except ValueError:
                raise ValueError(
                    f"unknown bulk zone name {ref!r}. Known zones: "
                    f"{self.zone_names}."
                ) from None
        if isinstance(ref, (int, np.integer)):
            zi = int(ref)
            if 0 <= zi < self.n_zones:
                return zi
            raise ValueError(
                f"bulk zone index {zi} out of range: the section has "
                f"{self.n_zones} zone(s) (0 = 'base', 1..N = "
                f"material_zones order)."
            )
        raise ValueError(
            f"zone reference must be a zone name (str) or a zone "
            f"index (int), got {ref!r}."
        )

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
        r"""
        Finalize rebar arrays.

        - If a rebar has ``x=None``, default to the section
          x-centroid.
        - Compute ``mat_indices_rebar``: for each rebar, the index
          of the bulk-material zone containing the rebar centroid
          (``0`` for the primary ``bulk_material``, ``1..N`` for
          named zones).  This is used by the integrator to subtract
          the displaced bulk material from the correct zone,
          restoring the correct stress balance for multi-material
          sections (e.g. a confined core inside an unconfined
          cover).
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
            # Zone lookup: which bulk-material zone is physically
            # displaced by each rebar's volume.  Drives the
            # embedded-rebar bulk-stress subtraction in the
            # integrator.
            self.mat_indices_rebar = np.array(
                [self._material_index(r.x, r.y)
                 for r in self.rebars],
                dtype=int)
        else:
            self.x_rebars = np.empty(0, dtype=float)
            self.y_rebars = np.empty(0, dtype=float)
            self.A_rebars = np.empty(0, dtype=float)
            self.embedded_rebars = np.empty(0, dtype=bool)
            self.mat_indices_rebar = np.empty(0, dtype=int)

    def _setup_tendons(self):
        r"""
        Finalize tendon arrays (prestress, Phase 1).

        Builds the parallel point-fiber arrays the solver consumes for
        tendons, analogous to :meth:`_setup_rebars` but with the extra
        ``eps_init_tendons`` array carrying each tendon's locked-in
        initial strain.  As for rebars:

        - a tendon with ``x=None`` defaults to the section x-centroid;
        - ``mat_indices_tendon`` records which bulk-material zone each
          tendon physically displaces, so the integrator subtracts the
          correct zone's stress (evaluated at the **section** strain,
          per the hard correctness invariant).

        For a section with no tendons, all arrays are empty and the
        solver's tendon block is skipped.
        """
        xc = self.x_centroid
        for t in self.tendons:
            if t.x is None:
                t.x = xc

        if self.tendons:
            self.x_tendons = np.array([t.x for t in self.tendons],
                                      dtype=float)
            self.y_tendons = np.array([t.y for t in self.tendons],
                                      dtype=float)
            self.A_tendons = np.array([t.Ap for t in self.tendons],
                                      dtype=float)
            self.eps_init_tendons = np.array(
                [t.eps_pe   for t in self.tendons], dtype=float)
            self.embedded_tendons = np.array(
                [t.embedded for t in self.tendons], dtype=bool)
            self.mat_indices_tendon = np.array(
                [self._material_index(t.x, t.y)
                 for t in self.tendons],
                dtype=int)
            self.staging_parent_tendon = \
                self._resolve_tendon_parents()
        else:
            self.x_tendons = np.empty(0, dtype=float)
            self.y_tendons = np.empty(0, dtype=float)
            self.A_tendons = np.empty(0, dtype=float)
            self.eps_init_tendons = np.empty(0, dtype=float)
            self.embedded_tendons = np.empty(0, dtype=bool)
            self.mat_indices_tendon = np.empty(0, dtype=int)
            self.staging_parent_tendon = np.empty(0, dtype=int)

    def _resolve_tendon_parents(self):
        r"""
        Resolve each tendon's **staging parent** zone.

        The staging parent is the bulk zone whose activity gates the
        tendon in the per-stage containment invariant

        .. math::

            \mathrm{active}[i] \;\Rightarrow\;
            \mathrm{bulk\_active}[\,\mathrm{parent}(i)\,]

        enforced by
        :meth:`~gensec.solver.section_state.StagedDomainManager.resolve_stages`.
        By default it is the geometric containing zone
        (``mat_indices_tendon``).  A tendon may override it via
        :attr:`~gensec.geometry.fiber.Tendon.parent` **only** when it
        is not embedded: an embedded tendon physically displaces the
        zone that contains it, and the displaced-bulk subtraction —
        which always uses the geometric zone — would contradict a
        different staging parent.  The override exists for
        non-embedded elements whose structural anchorage belongs to a
        zone other than the one their coordinates happen to fall in
        (e.g. an external tendon routed across a void).

        Returns
        -------
        numpy.ndarray of int
            Per-tendon staging-parent zone index.

        Raises
        ------
        ValueError
            ``parent`` set on an embedded tendon (the message carries
            the coordinates and both zones), or an unresolvable zone
            reference (propagated from :meth:`zone_index`).
        """
        parents = self.mat_indices_tendon.copy()
        for j, t in enumerate(self.tendons):
            override = getattr(t, "parent", None)
            if override is None:
                continue
            zi = self.zone_index(override)
            if t.embedded:
                geo = int(parents[j])
                raise ValueError(
                    f"Tendon {j} ('{t.name}') at "
                    f"(x={self.x_tendons[j]:.1f}, "
                    f"y={self.y_tendons[j]:.1f}): staging 'parent' "
                    f"override ({override!r} -> zone "
                    f"'{self.zone_names[zi]}') is legal only with "
                    f"embedded=False. An embedded tendon displaces "
                    f"the zone that geometrically contains it (zone "
                    f"'{self.zone_names[geo]}'), and its staging "
                    f"parent must coincide with it."
                )
            parents[j] = zi
        return parents

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

    # ``GenericSection`` is immutable by contract: the bulk mesh and the
    # element set are fixed for the object's lifetime, so the lazily
    # cached homogenized properties below are valid for as long as the
    # object exists.  Section-state evolution (staged construction,
    # prestress losses, de-stressing) is handled *without* mutating the
    # base section: ``gensec.solver.section_state.materialize_view``
    # produces a shallow copy per state with the point-element arrays
    # re-sliced and ``eps_init`` overridden, and sets that copy's
    # ``_ideal_gross_props_cache`` to ``None`` so it recomputes for the
    # state.  The base section's cache is therefore never stale.
    @property
    def ideal_gross_properties(self):
        """Lazy, cached homogenized section properties (elastic only).

        Computed with ``compute_plastic=False``.  For the plastic
        section moduli use :meth:`compute_ideal_properties` with
        ``compute_plastic=True`` (not cached).
        """
        if getattr(self, '_ideal_gross_props_cache', None) is None:
            self._ideal_gross_props_cache = self.compute_ideal_properties(
                compute_plastic=False
            )
        return self._ideal_gross_props_cache

    def compute_ideal_properties(self, compute_plastic: bool = False):
        r"""Compute the homogenized (ideal) section properties.

        Single source of truth for the homogenization convention: every
        bulk area element contributes :math:`n_{\mathrm{bulk}}\,
        \mathrm{d}A` and every embedded rebar contributes
        :math:`(n_s - n_{\mathrm{bulk}})\,A_s`.  The cached
        :attr:`ideal_gross_properties` delegates here with
        ``compute_plastic=False``.

        Parameters
        ----------
        compute_plastic : bool, optional
            Also compute the plastic section moduli
            :math:`Z_x, Z_y, Z_\xi, Z_\eta` via neutral-axis bisection.
            Default ``False``.

        Returns
        -------
        gensec.properties.SectionProperties

        Raises
        ------
        NotImplementedError
            If the section has more than one bulk material zone
            (multi-material homogenization is not yet supported).

        Notes
        -----
        Tendons are **excluded** from the homogenized (ideal) section
        in this phase.  The existing-prestress domain and SLS stresses
        are obtained by ULS/SLS fiber integration with the locked-in
        prestrain (see the solver), not from the elastic ideal
        section; folding the tendon transformed area in here would
        double-count once the prestrain is applied at the fiber level.
        """
        from .properties import (
            compute_section_properties, HomogenizedRebar,
        )
        ### TODO: add support for multi-material bulk zones here
        ### (currently ignored in homogenization)
        if len(self.bulk_materials) > 1:
            raise NotImplementedError(
                "ideal_gross_properties currently ignores "
                "multi-material bulk zones."
            )
        homog = [
            HomogenizedRebar(r.x, r.y, r.As, r.material.E)
            for r in self.rebars
            if r.embedded and r.x is not None
        ]
        return compute_section_properties(
            self.polygon,
            rebars=homog,
            E_bulk=self.bulk_material.E,
            compute_plastic=compute_plastic,
        )

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