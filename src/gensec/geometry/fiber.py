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


@dataclass
class Tendon:
    r"""
    A prestressing tendon as a point fiber with a locked-in prestrain.

    A tendon is distinct from a :class:`RebarLayer`: it carries an
    **effective prestrain** :math:`\varepsilon_{pe}` that is locked in
    independently of the section's strain field.  The total strain
    seen by the tendon's constitutive law is

    .. math::

        \varepsilon_{\text{tot},i}
            = \varepsilon_{\text{sec},i} + \varepsilon_{\text{init},i}

    where :math:`\varepsilon_{\text{sec},i}` is the section strain at
    the tendon location (from the strain plane) and
    :math:`\varepsilon_{\text{init},i}` is the locked-in initial
    strain.  In this phase (bonded, single instant, no losses) the
    initial strain equals the effective prestrain,
    :math:`\varepsilon_{\text{init}} = \varepsilon_{pe}`.

    Hard correctness invariant
    --------------------------
    The initial strain is **not** added to the section strain field
    globally.  Only the tendon evaluates its own law at the offset
    total strain; the displaced bulk (concrete) that the tendon
    occupies is evaluated at the **section** strain alone — the
    concrete never sees the tendon's locked-in strain.  This is
    enforced in the integrator, which evaluates the tendon and the
    displaced-bulk laws at two different strain arguments.

    Sign convention
    --------------
    :math:`\varepsilon_{pe} > 0` is a tensile prestrain, consistent
    with the symmetric prestressing-steel diagram.

    Parameters
    ----------
    y : float
        Vertical coordinate from bottom edge [mm].
    material : Material
        Constitutive law, typically
        :class:`~gensec.materials.steel.PrestressingSteel`.
    Ap : float, optional
        Tendon cross-sectional area [mm²].  If 0 or omitted, computed
        from ``n_strands`` and ``area_strand``.
    eps_pe : float, optional
        Effective prestrain (after losses, positive = tension).
        Default 0.0.  In Phase 1 this is taken as the locked-in
        initial strain directly.
    x : float, optional
        Horizontal coordinate from left edge [mm].  If ``None``,
        defaults to the section centroid x-coordinate during
        assembly.
    system : {'pre', 'post'}, optional
        Prestressing system: pre-tensioned or post-tensioned.  Stored
        for downstream phases (anchorage, bond, duct grouting); does
        not change the Phase 1 bonded computation.  Default ``'pre'``.
    bonded : bool, optional
        Whether the tendon is bonded to the surrounding concrete.
        Phase 1 supports **bonded only**; ``False`` raises
        :class:`NotImplementedError`.  Default ``True``.
    embedded : bool, optional
        If ``True`` (default), the tendon displaces bulk material and
        the integrator subtracts the bulk contribution at the tendon
        location (evaluated at the section strain).  Set ``False`` for
        external (unbonded-external, e.g. deviated) tendons — out of
        Phase 1 scope.
    n_strands : int, optional
        Number of strands.  Default 1.  Used to compute ``Ap`` when
        ``area_strand`` is given.
    area_strand : float, optional
        Single-strand area [mm²].  Default 0.  When positive and
        ``Ap`` is 0, ``Ap = n_strands * area_strand``.

    Attributes
    ----------
    eps_init : float
        Locked-in initial strain applied as an offset by the solver.
        In Phase 1, equals ``eps_pe``.
    """

    y: float
    material: Material = None
    Ap: float = 0.0
    eps_pe: float = 0.0
    x: Optional[float] = None
    system: str = "pre"
    bonded: bool = True
    embedded: bool = True
    n_strands: int = 1
    area_strand: float = 0.0

    def __post_init__(self):
        """Validate area/system and derive the initial strain."""
        if self.Ap <= 0.0 and self.area_strand > 0.0:
            self.Ap = self.n_strands * self.area_strand
        if self.Ap <= 0.0:
            raise ValueError(
                f"Tendon at y={self.y}: Ap must be positive. "
                f"Provide Ap directly or set area_strand > 0."
            )
        if self.system not in ("pre", "post"):
            raise ValueError(
                f"Tendon at y={self.y}: system must be 'pre' or "
                f"'post', got '{self.system}'."
            )
        if not self.bonded:
            raise NotImplementedError(
                "Phase 1 supports bonded tendons only "
                "(bonded=True). Unbonded/external tendons require "
                "the member-level elongation compatibility of a "
                "later phase."
            )
        # Phase 1: single instant, no losses → initial strain is the
        # effective prestrain directly.
        self.eps_init = self.eps_pe
