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
Rectangular section factory — convenience wrapper.

:func:`RectSection` is a factory function (not a class) that builds
a :class:`GenericSection` with a rectangular polygon and grid
meshing.  It replaces the former ``RectSection`` dataclass, which
was a thin delegation wrapper that introduced a mesh-size bug on
anisotropic grids.

The returned object **is** a :class:`GenericSection`.  All
attributes expected by :class:`~gensec.solver.FiberSolver` are
available directly (``x_fibers``, ``y_fibers``, ``A_fibers``,
``B``, ``H``, ``n_fibers_x``, ``n_fibers_y``, ``dx``, ``dy``,
``polygon``, …).

Grid resolution
---------------
The mesh resolution in the y-direction is controlled by
``n_fibers_y``.  The x-direction follows one of two rules:

- **Isotropic** (default, ``n_fibers_x ≤ 1``): the x-resolution
  is derived from the same ``mesh_size = H / n_fibers_y``, giving
  a square-celled grid.  Example: B = 300, H = 600, n_fibers_y = 100
  → mesh_size = 6 → n_fibers_x = ceil(300/6) = 50.
- **Explicit** (``n_fibers_x > 1``): the user-specified value is
  passed as ``n_grid_x``.
"""

from typing import List

from .fiber import RebarLayer
from .geometry import GenericSection
from .primitives import rect_poly
from ..materials.base import Material


def RectSection(B, H, bulk_material, rebars,
                n_fibers_y=100, n_fibers_x=1):
    r"""
    Create a rectangular :class:`GenericSection`.

    Parameters
    ----------
    B : float
        Width (x-direction) [mm].
    H : float
        Height (y-direction) [mm].
    bulk_material : Material
        Bulk material (concrete, timber, …).
    rebars : list of RebarLayer
        Point fibers.
    n_fibers_y : int, optional
        Fiber rows.  Controls the isotropic mesh size:
        :math:`s = H / n_y`.  Default 100.
    n_fibers_x : int, optional
        Fiber columns.  When ``> 1``, overrides the x-resolution
        explicitly.  When ``≤ 1`` (default), the x-resolution is
        derived from the isotropic mesh size :math:`s`.

    Returns
    -------
    GenericSection
        Fully meshed rectangular section with all attributes
        expected by the solver (``x_fibers``, ``y_fibers``,
        ``A_fibers``, ``B``, ``H``, ``n_fibers_x``, ``n_fibers_y``,
        ``dx``, ``dy``, ``polygon``, …).
    """
    poly = rect_poly(B, H)
    dy = H / max(n_fibers_y, 1)

    kwargs = dict(
        polygon=poly,
        bulk_material=bulk_material,
        rebars=rebars,
        mesh_size=dy,
        mesh_method="grid",
        n_grid_y=n_fibers_y,
    )

    if n_fibers_x > 1:
        # User explicitly requests a specific x resolution.
        kwargs["n_grid_x"] = n_fibers_x

    # Otherwise: n_grid_x is None → GenericSection derives
    # n_fibers_x from mesh_size (isotropic square cells).

    return GenericSection(**kwargs)