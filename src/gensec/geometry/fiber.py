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
"""
Point fiber definitions (rebars, FRP strips, tendons).

Phase 2: each fiber has both x and y coordinates for biaxial bending.
"""

from dataclasses import dataclass
from typing import Optional
from ..materials.base import Material


@dataclass
class RebarLayer:
    r"""
    A point fiber (rebar, FRP strip, tendon, etc.).

    Each fiber carries its own :class:`Material` reference, allowing
    mixed-material sections. For biaxial bending, both ``x`` and ``y``
    coordinates are needed.

    The cross-sectional area ``As`` can be specified directly or
    computed automatically from ``diameter`` and ``n_bars``:

    .. math::

        A_s = n_{\text{bars}} \cdot \frac{\pi}{4} \, d^2

    If both ``As`` and ``diameter`` are given, ``As`` takes
    precedence.  If only ``diameter`` is given (with ``As`` omitted
    or set to 0), ``As`` is computed from the formula above.

    Parameters
    ----------
    y : float
        Vertical coordinate from bottom edge [mm].
    As : float, optional
        Cross-sectional area [mm²]. If 0 or omitted, computed
        from ``diameter`` and ``n_bars``.
    material : Material
        Constitutive law.
    x : float, optional
        Horizontal coordinate from left edge [mm]. If ``None``,
        defaults to the section centroid x-coordinate (set during
        section assembly). For uniaxial bending this is irrelevant.
    embedded : bool, optional
        If ``True`` (default), the fiber is embedded within the bulk
        material. The integrator will subtract the bulk material
        contribution at this location to avoid double-counting the
        area. Set to ``False`` for external elements (e.g. external
        FRP strips, steel truss chords outside the concrete).
    n_bars : int, optional
        Number of bars. Default 1.  Also used to compute ``As``
        when ``diameter`` is given.
    diameter : float, optional
        Bar diameter [mm]. Default 0.  When positive and ``As`` is
        0, ``As`` is computed as
        :math:`n_{\text{bars}} \cdot \pi/4 \cdot d^2`.
    """

    y: float
    As: float = 0.0
    material: Material = None
    x: Optional[float] = None
    embedded: bool = True
    n_bars: int = 1
    diameter: float = 0.0

    def __post_init__(self):
        """Compute As from diameter if not provided explicitly."""
        import math
        if self.As <= 0.0 and self.diameter > 0.0:
            self.As = self.n_bars * math.pi / 4.0 * self.diameter ** 2
        if self.As <= 0.0:
            raise ValueError(
                f"RebarLayer at y={self.y}: As must be positive. "
                f"Provide As directly or set diameter > 0."
            )
