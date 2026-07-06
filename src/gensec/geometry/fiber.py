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

Element taxonomy
----------------
This module defines **section elements** — objects that are part of the
capacity state.  Two element kinds live here:

- :class:`RebarLayer` — a passive point fiber (mild steel, GFRP,
  stainless, ...).  The constitutive material distinguishes a "slow" bar
  from an ordinary one; there is deliberately no separate class.
- :class:`Tendon` — a **bonded** prestressing element, strain-compatible
  with the section, carrying a :class:`PrestressingSteel` law plus an
  effective prestrain :math:`\\varepsilon_{pe}`.

A prestressing *force* applied to hardened concrete (jacking, unbonded
or external tendons) is **not** an element — it is a load source modeled
by ``PrestressAction`` in the demand layer, never here.  The
element-vs-load split is the bonded-vs-unbonded boundary: bonded →
``Tendon`` element (in the capacity state); unbonded/external/during
jacking → ``PrestressAction`` load (in the demand).
"""

import math
from dataclasses import dataclass
from typing import Optional, Union
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
    name : str or None, optional
        Human-readable identifier (e.g. ``"B_top"``).  Default
        ``None``.  Symmetric to :attr:`Tendon.name`: a stable reference
        for the element across the I/O and reporting layers.  A YAML
        ``section_ops`` entry (``activate`` / ``deactivate`` /
        ``eps_override``) may target the element via its name instead of
        its union index; names used as references must be unique across
        the whole ``rebars + tendons`` union set.  Purely descriptive —
        no behavioural effect in the solver.
    """

    y: float
    As: float = 0.0
    material: Material = None
    x: Optional[float] = None
    embedded: bool = True
    n_bars: int = 1
    diameter: float = 0.0
    name: Optional[str] = None

    def __post_init__(self):
        """Compute As from diameter if not provided explicitly."""
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
    A bonded prestressing tendon as a strain-compatible section element.

    A :class:`Tendon` is the prestress counterpart of
    :class:`RebarLayer`: a point fiber carrying its own
    :class:`~gensec.materials.steel.PrestressingSteel` law, plus the one
    datum that distinguishes a tendon from a passive bar — the
    **effective prestrain** :math:`\varepsilon_{pe}`.

    The solver evaluates the tendon's constitutive law at the **offset
    total strain**

    .. math::

        \varepsilon_{\text{tot}}
            = \varepsilon_{\text{sec}} + \varepsilon_{pe},

    while the displaced bulk (for an embedded tendon) is evaluated at the
    section strain :math:`\varepsilon_{\text{sec}}` alone.  Equivalently,
    :math:`\varepsilon_{pe}` is a per-element imposed-strain offset; the
    same generic mechanism also carries shrinkage/thermal fields on the
    bulk (named ``eps_init`` there).

    Reference datum and elastic shortening
    --------------------------------------
    :math:`\varepsilon_{pe}` is referenced to the **unstrained-concrete**
    (casting) state, so that :math:`\varepsilon_{\text{sec}}=0` means the
    pre-transfer (jacking) configuration.  For a pre-tensioned strand,
    :math:`\varepsilon_{pe}` is initialised to the jacking strain
    :math:`\sigma_{p0}/E_p`.  The immediate **elastic-shortening loss is
    not stored** in :math:`\varepsilon_{pe}`: it *emerges* from section
    equilibrium as a non-zero :math:`\varepsilon_{\text{sec}}` at the
    tendon (see :mod:`gensec.solver.prestress_transfer`).  Only the
    later, time-dependent losses (relaxation, creep, shrinkage) reduce
    :math:`\varepsilon_{pe}` and thereby move the resistance domain.

    The area :math:`A_p` may be given directly or computed from a single
    strand area and a strand count:

    .. math::

        A_p = n_{\text{strands}} \cdot A_{\text{strand}}.

    Parameters
    ----------
    y : float
        Vertical coordinate from the bottom edge [mm].
    Ap : float, optional
        Tendon cross-sectional area [mm²].  If 0 or omitted, computed
        from ``area_strand`` and ``n_strands``.
    material : Material
        Prestressing-steel constitutive law (total-strain), e.g. a
        :class:`~gensec.materials.steel.PrestressingSteel`.
    x : float, optional
        Horizontal coordinate from the left edge [mm].  If ``None``,
        defaults to the section x-centroid during section assembly.
    eps_pe : float, optional
        Effective prestrain referenced to the unstrained-concrete state
        (tension positive).  Default 0.0.  For a pre-tensioned strand
        this is the jacking strain :math:`\sigma_{p0}/E_p` before
        time-dependent losses.
    embedded : bool, optional
        If ``True`` (default), the tendon is inside the bulk and the
        integrator subtracts the displaced-bulk stress at its location
        to avoid double-counting the area.  This is the integrator-level
        flag actually consumed by the solver.
    bonded : bool, optional
        Architectural flag marking the tendon as a strain-compatible
        **element**.  Default ``True``.  The current phase supports
        bonded tendons only; ``bonded=False`` (unbonded/external) is a
        *load*, not an element, and must be modeled as a
        ``PrestressAction`` in the demand layer — so it raises here.
    parent : int or str or None, optional
        **Staging-parent zone override** (Phase 8).  Default ``None``:
        the staging parent is the bulk zone that geometrically
        contains the tendon (``mat_indices_tendon``), which gates the
        tendon in the per-stage containment invariant
        :math:`\mathrm{active}[i] \Rightarrow
        \mathrm{bulk\_active}[\mathrm{parent}(i)]`.  A zone name or
        1-based zone index overrides the *staging* parent only — the
        displaced-bulk subtraction keeps using the geometric zone —
        and is legal **only** with ``embedded=False`` (enforced at
        section assembly, where the zone map exists).  The former
        ``system`` tag (``'pre'``/``'post'``) is retired: the
        construction system is derived from the staging timeline
        (ordering of stressing vs casting events), never declared.
    n_strands : int, optional
        Number of strands.  Default 1.  Used with ``area_strand`` to
        compute ``Ap``.
    area_strand : float, optional
        Single-strand area [mm²].  Default 0.  When positive and ``Ap``
        is 0, ``Ap = n_strands * area_strand``.
    name : str or None, optional
        Human-readable identifier (e.g. ``"T_bottom"``).  Default
        ``None``.  Used as a stable reference for the tendon across the
        I/O and reporting layers: a YAML ``prestress_actions`` entry may
        target a tendon's geometry via ``ref: <name>``, and per-tendon
        outputs (losses, stressing sequence) will report it.  Purely
        descriptive — no behavioural effect in the solver.

    Raises
    ------
    ValueError
        If ``Ap`` cannot be resolved to a positive value, or if
        ``bonded`` is ``False`` (use a ``PrestressAction`` load instead).

    Notes
    -----
    The geometry and integrator treat tendons and bars through the same
    point-fiber code paths; the prestrain offset is the only addition.
    At section assembly, :attr:`eps_pe` is mapped onto the solver's
    generic per-element offset array (``eps_init_tendons``).

    Examples
    --------
    A single 7-wire strand (Ap = 150 mm²) jacked to 1395 MPa
    (:math:`E_p = 195\,000` MPa):

    >>> from gensec.materials.steel import PrestressingSteel
    >>> ps = PrestressingSteel(f_p01d=1391.3, Ep=195000.0)
    >>> t = Tendon(y=60.0, Ap=150.0, material=ps,
    ...            eps_pe=1395.0 / 195000.0)
    >>> round(t.eps_pe, 5)
    0.00715
    """

    y: float
    Ap: float = 0.0
    material: Material = None
    x: Optional[float] = None
    eps_pe: float = 0.0
    embedded: bool = True
    bonded: bool = True
    parent: Optional[Union[int, str]] = None
    n_strands: int = 1
    area_strand: float = 0.0
    name: Optional[str] = None

    def __post_init__(self):
        """Resolve ``Ap`` and enforce the bonded-element invariant."""
        if self.Ap <= 0.0 and self.area_strand > 0.0:
            self.Ap = self.n_strands * self.area_strand
        if self.Ap <= 0.0:
            raise ValueError(
                f"Tendon at y={self.y}: Ap must be positive. "
                f"Provide Ap directly or set area_strand > 0."
            )
        if not self.bonded:
            raise ValueError(
                f"Tendon at y={self.y}: bonded=False is not a section "
                f"element. An unbonded/external prestressing force on "
                f"hardened concrete is a load — model it as a "
                f"PrestressAction in the demand layer, not as a Tendon."
            )