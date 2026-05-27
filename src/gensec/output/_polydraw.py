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
Canonical conversion of Shapely areal geometries into filled
Matplotlib patches with correct hole carve-out.

This is the single source of truth for rendering section outlines.
Both :mod:`gensec.output.plots` and :mod:`gensec.output.geometry_plot`
delegate here so that the (non-trivial) winding logic lives in one
place only.

Why ring orientation must be normalised
---------------------------------------
Matplotlib fills a compound :class:`~matplotlib.path.Path` using the
*non-zero winding* rule. A hole is therefore carved out only if its
ring is traversed in the *opposite* direction to the exterior ring.

Shapely does **not** guarantee any particular orientation for the
rings it produces; in particular the result of a boolean operation
such as :meth:`shapely.geometry.Polygon.difference` may come back
with a clockwise exterior. Relying on an assumed orientation (e.g.
"exteriors are always CCW") silently fails: two co-oriented rings
yield a winding number of :math:`\pm 2` inside the hole, which is
non-zero, so the hole is filled.

The functions below enforce the orientation explicitly:

.. math::

    A = \tfrac{1}{2} \sum_i \left( x_i\, y_{i+1} - x_{i+1}\, y_i \right)

is the shoelace signed area; :math:`A > 0` denotes a
counter-clockwise (CCW) ring. Exteriors are forced CCW and holes CW.
"""

import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path


__all__ = ["polygon_to_path", "draw_polygon"]


def _signed_area(coords):
    r"""
    Shoelace signed area of a ring.

    Parameters
    ----------
    coords : numpy.ndarray, shape (n, 2)
        Ring vertices. A duplicated closing vertex (first == last),
        as produced by Shapely, is tolerated and ignored.

    Returns
    -------
    float
        Signed area :math:`A`; positive for a counter-clockwise
        ring, negative for a clockwise one.
    """
    pts = coords
    if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _append_ring(verts, codes, coords, *, want_ccw):
    r"""
    Append one closed ring to compound-path buffers with a
    prescribed orientation.

    Parameters
    ----------
    verts, codes : list
        Accumulators for path vertices and path codes, mutated
        in place.
    coords : numpy.ndarray, shape (n, 2)
        Ring vertices (closing duplicate tolerated).
    want_ccw : bool
        Target orientation. Exteriors require ``True``, holes
        require ``False`` so that the non-zero winding rule carves
        the holes out.
    """
    if len(coords) >= 2 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    area = _signed_area(coords)
    if (want_ccw and area < 0.0) or (not want_ccw and area > 0.0):
        coords = coords[::-1]
    n = len(coords)
    verts.extend(coords.tolist())
    verts.append(coords[0].tolist())          # CLOSEPOLY anchor
    codes.append(Path.MOVETO)
    codes.extend([Path.LINETO] * (n - 1))
    codes.append(Path.CLOSEPOLY)


def _iter_polygons(geom):
    r"""
    Yield the simple-polygon parts of an areal Shapely geometry.

    Accepts :class:`~shapely.geometry.Polygon`,
    :class:`~shapely.geometry.MultiPolygon`, and
    :class:`~shapely.geometry.GeometryCollection` (the latter is
    filtered to its polygonal members). Non-areal geometries yield
    nothing.

    Yielding rather than assuming a single ``Polygon`` keeps the
    renderer correct when a boolean operation returns a disconnected
    result (several annular fragments, multi-cell box sections, …).
    """
    geom_type = getattr(geom, "geom_type", None)
    if geom_type == "Polygon":
        yield geom
    elif geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geom.geoms:
            if getattr(part, "geom_type", None) == "Polygon":
                yield part


def polygon_to_path(geom):
    r"""
    Build a Matplotlib compound path from an areal Shapely geometry,
    with every hole correctly carved out.

    Parameters
    ----------
    geom : shapely.geometry.Polygon or MultiPolygon
        Section geometry. Holes are taken from each polygon's
        interior rings.

    Returns
    -------
    matplotlib.path.Path
        Compound path with exteriors wound CCW and holes wound CW,
        ready to be filled with the non-zero winding rule.

    Notes
    -----
    Orientation is normalised explicitly; the input winding is not
    assumed. See the module docstring for the rationale.
    """
    verts, codes = [], []
    for poly in _iter_polygons(geom):
        _append_ring(verts, codes,
                     np.asarray(poly.exterior.coords), want_ccw=True)
        for interior in poly.interiors:
            _append_ring(verts, codes,
                         np.asarray(interior.coords), want_ccw=False)
    return Path(verts, codes)


def draw_polygon(ax, geom, *,
                 facecolor="#E8E8E8",
                 edgecolor="black",
                 linewidth=2.0,
                 zorder=1,
                 **patch_kwargs):
    r"""
    Render an areal Shapely geometry as a filled patch on an axes,
    carving out holes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    geom : shapely.geometry.Polygon or MultiPolygon
        Section geometry to draw.
    facecolor, edgecolor : color, optional
        Patch fill and outline colours.
    linewidth : float, optional
        Outline width in points.
    zorder : float, optional
        Drawing order.
    **patch_kwargs
        Forwarded to :class:`~matplotlib.patches.PathPatch`
        (e.g. ``alpha``, ``hatch``, ``label``).

    Returns
    -------
    matplotlib.patches.PathPatch
        The patch added to ``ax``.
    """
    patch = PathPatch(polygon_to_path(geom),
                      facecolor=facecolor, edgecolor=edgecolor,
                      linewidth=linewidth, zorder=zorder,
                      **patch_kwargs)
    ax.add_patch(patch)
    return patch