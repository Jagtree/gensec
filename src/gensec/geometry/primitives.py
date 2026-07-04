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
Parametric section primitives — factory functions.

Each function returns a :class:`shapely.geometry.Polygon` that can
be passed directly to :class:`GenericSection`. All dimensions are in
millimetres.

The coordinate origin convention is **bottom-left corner of the
bounding box**, consistent with the former ``RectSection``.

Available primitives
--------------------
- :func:`rect_poly` — Rectangle.
- :func:`circle_poly` — Circle (regular polygon approximation).
- :func:`annulus_poly` — Annular (hollow) circle.
- :func:`tee_poly` — T-section (flange on top).
- :func:`inv_tee_poly` — Inverted T-section (flange on bottom).
- :func:`h_poly` — H / I section (symmetric double-T).
- :func:`single_tee_slab_poly` — Single-tee precast slab (tegolo).
- :func:`double_tee_slab_poly` — Double-tee precast slab (tegolo).
- :func:`box_poly` — Hollow rectangular (box) section.

Examples
--------
>>> from gensec.geometry.primitives import rect_poly
>>> p = rect_poly(300, 600)
>>> p.area
180000.0

>>> from gensec.geometry.primitives import tee_poly
>>> p = tee_poly(bf=800, hf=150, bw=300, hw=450)
>>> p.area
255000.0
"""

from shapely.geometry import Polygon, Point, box as shapely_box
import numpy as np


def rect_poly(B, H):
    r"""
    Rectangular section polygon.

    .. code-block:: text

        (0,H) ┌─────────┐ (B,H)
              │         │
              │         │
        (0,0) └─────────┘ (B,0)

    Parameters
    ----------
    B : float
        Width [mm].
    H : float
        Height [mm].

    Returns
    -------
    shapely.geometry.Polygon
    """
    return shapely_box(0, 0, B, H)


def circle_poly(D, resolution=64):
    r"""
    Circular section polygon (regular N-gon approximation).

    The circle is centred at :math:`(D/2, D/2)` so that the bounding
    box starts at the origin.

    Parameters
    ----------
    D : float
        Diameter [mm].
    resolution : int, optional
        Number of vertices on the circumference. Default 64.

    Returns
    -------
    shapely.geometry.Polygon
    """
    return Point(D / 2, D / 2).buffer(D / 2, quad_segs=resolution)


def annulus_poly(D_ext, D_int, resolution=64):
    r"""
    Annular (hollow circular) section.

    Parameters
    ----------
    D_ext : float
        Outer diameter [mm].
    D_int : float
        Inner diameter [mm].
    resolution : int, optional

    Returns
    -------
    shapely.geometry.Polygon
        Polygon with an interior hole.

    Raises
    ------
    ValueError
        If ``D_int >= D_ext``.
    """
    if D_int >= D_ext:
        raise ValueError(
            f"Inner diameter ({D_int}) must be less than "
            f"outer diameter ({D_ext})."
        )
    cx, cy = D_ext / 2, D_ext / 2
    outer = Point(cx, cy).buffer(D_ext / 2, quad_segs=resolution)
    inner = Point(cx, cy).buffer(D_int / 2, quad_segs=resolution)
    return outer.difference(inner)


def tee_poly(bf, hf, bw, hw):
    r"""
    T-section with flange on top.

    .. code-block:: text

        ┌─────────────────┐  ← bf
        │     flange      │ hf
        └──┬─────────┬────┘
           │  web    │ hw
           │         │
           └─────────┘
              bw

    Origin at bottom-left of the bounding box.

    Parameters
    ----------
    bf : float
        Flange width [mm].
    hf : float
        Flange thickness [mm].
    bw : float
        Web width [mm].
    hw : float
        Web height [mm] (below the flange).

    Returns
    -------
    shapely.geometry.Polygon

    Raises
    ------
    ValueError
        If ``bw > bf``.
    """
    if bw > bf:
        raise ValueError(
            f"Web width ({bw}) cannot exceed flange width ({bf})."
        )
    H = hf + hw
    x_offset = (bf - bw) / 2
    # Build as a single polygon with 8 vertices (CCW)
    coords = [
        (x_offset, 0),           # bottom-left of web
        (x_offset + bw, 0),      # bottom-right of web
        (x_offset + bw, hw),     # web-flange junction right
        (bf, hw),                 # flange bottom-right
        (bf, H),                  # flange top-right
        (0, H),                   # flange top-left
        (0, hw),                  # flange bottom-left
        (x_offset, hw),          # web-flange junction left
    ]
    return Polygon(coords)


def inv_tee_poly(bf, hf, bw, hw):
    r"""
    Inverted T-section (flange on bottom).

    Same parameters as :func:`tee_poly` but the flange is at
    :math:`y = 0`.

    Parameters
    ----------
    bf, hf, bw, hw : float
        See :func:`tee_poly`.

    Returns
    -------
    shapely.geometry.Polygon
    """
    if bw > bf:
        raise ValueError(
            f"Web width ({bw}) cannot exceed flange width ({bf})."
        )
    H = hf + hw
    x_offset = (bf - bw) / 2
    coords = [
        (0, 0),                   # flange bottom-left
        (bf, 0),                  # flange bottom-right
        (bf, hf),                 # flange top-right
        (x_offset + bw, hf),     # web-flange junction right
        (x_offset + bw, H),      # web top-right
        (x_offset, H),           # web top-left
        (x_offset, hf),          # web-flange junction left
        (0, hf),                  # flange top-left
    ]
    return Polygon(coords)


def h_poly(bf, hf_top, hf_bot, bw, hw):
    r"""
    H / I section (double-T, symmetric about the vertical axis).

    .. code-block:: text

        ┌─────────────────┐  bf
        │   top flange    │ hf_top
        └──┬─────────┬────┘
           │  web    │ hw
        ┌──┴─────────┴────┐
        │  bottom flange  │ hf_bot
        └─────────────────┘  bf

    Parameters
    ----------
    bf : float
        Flange width [mm] (same for top and bottom).
    hf_top : float
        Top flange thickness [mm].
    hf_bot : float
        Bottom flange thickness [mm].
    bw : float
        Web width [mm].
    hw : float
        Web height [mm] (between flanges).

    Returns
    -------
    shapely.geometry.Polygon
    """
    if bw > bf:
        raise ValueError(
            f"Web width ({bw}) cannot exceed flange width ({bf})."
        )
    H = hf_bot + hw + hf_top
    xo = (bf - bw) / 2
    coords = [
        # Bottom flange (CCW)
        (0, 0),
        (bf, 0),
        (bf, hf_bot),
        (xo + bw, hf_bot),
        # Web
        (xo + bw, hf_bot + hw),
        # Top flange
        (bf, hf_bot + hw),
        (bf, H),
        (0, H),
        (0, hf_bot + hw),
        (xo, hf_bot + hw),
        # Web left
        (xo, hf_bot),
        (0, hf_bot),
    ]
    return Polygon(coords)


def box_poly(B, H, tw, tf_top, tf_bot=None):
    r"""
    Hollow rectangular (box) section.

    Parameters
    ----------
    B : float
        Outer width [mm].
    H : float
        Outer height [mm].
    tw : float
        Web (side wall) thickness [mm].
    tf_top : float
        Top flange thickness [mm].
    tf_bot : float, optional
        Bottom flange thickness [mm]. If ``None``, same as
        ``tf_top``.

    Returns
    -------
    shapely.geometry.Polygon
        Polygon with one rectangular hole.
    """
    if tf_bot is None:
        tf_bot = tf_top
    outer = shapely_box(0, 0, B, H)
    inner = shapely_box(tw, tf_bot, B - tw, H - tf_top)
    return outer.difference(inner)


def single_tee_slab_poly(b_top, h_top, bw, hw):
    r"""
    Single-tee precast slab section (tegolo singolo).

    A wide top slab with a single stem (rib) below.

    .. code-block:: text

        ┌───────────────────────┐  b_top
        │       top slab        │ h_top
        └────┬───────────┬──────┘
             │   stem    │ hw
             └───────────┘
                  bw

    Parameters
    ----------
    b_top : float
        Top slab width [mm].
    h_top : float
        Top slab thickness [mm].
    bw : float
        Stem width [mm].
    hw : float
        Stem depth [mm] (below the slab).

    Returns
    -------
    shapely.geometry.Polygon
    """
    return tee_poly(bf=b_top, hf=h_top, bw=bw, hw=hw)


def double_tee_slab_poly(b_top, h_top, bw, hw, stem_spacing):
    r"""
    Double-tee precast slab section (tegolo doppio).

    A wide top slab with two stems (ribs) below, symmetrically
    placed.

    .. code-block:: text

        ┌───────────────────────────────┐  b_top
        │          top slab             │ h_top
        └──┬────┬───────────────┬────┬──┘
           │ s1 │               │ s2 │ hw
           └────┘               └────┘
            bw      spacing      bw

    Parameters
    ----------
    b_top : float
        Top slab overall width [mm].
    h_top : float
        Top slab thickness [mm].
    bw : float
        Width of each stem [mm].
    hw : float
        Depth of each stem [mm].
    stem_spacing : float
        Centre-to-centre distance between the two stems [mm].

    Returns
    -------
    shapely.geometry.Polygon
    """
    H = h_top + hw

    # Slab
    slab = shapely_box(0, hw, b_top, H)

    # Stems (centred around the spacing)
    cx1 = (b_top - stem_spacing) / 2
    cx2 = (b_top + stem_spacing) / 2

    stem1 = shapely_box(cx1 - bw / 2, 0, cx1 + bw / 2, hw)
    stem2 = shapely_box(cx2 - bw / 2, 0, cx2 + bw / 2, hw)

    from shapely.ops import unary_union
    return unary_union([slab, stem1, stem2])


def custom_poly(exterior_coords, holes=None):
    r"""
    Arbitrary polygon from vertex coordinates.

    Parameters
    ----------
    exterior_coords : list of tuple
        ``[(x0, y0), (x1, y1), ...]``. The ring is automatically
        closed.
    holes : list of list of tuple, optional
        Each inner list defines a hole boundary.

    Returns
    -------
    shapely.geometry.Polygon
    """
    if holes:
        return Polygon(exterior_coords, holes)
    return Polygon(exterior_coords)
