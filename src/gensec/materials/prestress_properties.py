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
Prestressing-steel properties as per EN 1992-1-1 §3.3.

Implements :class:`fbpren`, the prestressing-steel counterpart of the
concrete property class :class:`~gensec.materials.ec2_properties.fben2`.
It computes the design values of the idealized stress-strain diagram
(EN 1992-1-1 §3.3.6) from the characteristic strengths
:math:`f_{p0.1k}`, :math:`f_{pk}`, the characteristic ultimate strain
:math:`\varepsilon_{uk}`, and the modulus of elasticity :math:`E_p`,
for a chosen limit state and National Annex.

Design philosophy
-----------------
This class mirrors the conventions of :class:`fben2`:

- The partial factor :math:`\gamma_S` is resolved **procedurally**
  from the limit state, with the EC2 Table 2.1N recommended values
  as the default; a national annex that overrides them is a data
  change, not a code change.
- An unimplemented National Annex raises explicitly, so the current
  coverage is visible exactly where it matters.
- Properties are stored as plain attributes.  Time-dependent
  relaxation (EN 1992-1-1 §3.3.2) is **not** computed in this phase;
  the relaxation hooks are left as documented placeholders so that a
  later phase can populate them without restructuring.

This class does **not** build a GenSec material.  The bridge
:func:`~gensec.materials.ec2_bridge.prestress_from_ec2` consumes a
:class:`fbpren` instance and constructs a
:class:`~gensec.materials.steel.PrestressingSteel`, exactly as
:func:`~gensec.materials.ec2_bridge.concrete_from_ec2` does for
concrete.

References
----------
- EN 1992-1-1:2004 §3.3 (prestressing steel).
- EN 10138 (prestressing steel product standard) for the
  designations tabulated in :data:`PrestressClassData`.
"""

from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
#  Limit-state → partial-factor resolution
# ---------------------------------------------------------------------------

def gamma_s_prestress(ls='F', NA='EC2', gamma_s_override=None):
    r"""
    Resolve the prestressing-steel partial factor :math:`\gamma_S`.

    EN 1992-1-1 Table 2.1N gives :math:`\gamma_S = 1.15` for
    persistent and transient design situations and :math:`1.0` for
    accidental and serviceability situations.  These are the defaults;
    a National Annex may override them.

    Parameters
    ----------
    ls : str, optional
        Limit state: ``'F'`` (fundamental / persistent-transient),
        ``'A'`` (accidental), or ``'S'`` (serviceability).  Default
        ``'F'``.
    NA : str, optional
        National Annex selector.  Default ``'EC2'`` (recommended
        values).  Unknown values raise :class:`ValueError`.
    gamma_s_override : float or None, optional
        If given, bypasses the table and returns this value verbatim.
        Provided so a caller can inject an annex-specific factor
        without extending this function.  Default ``None``.

    Returns
    -------
    float
        The partial factor :math:`\gamma_S`.

    Raises
    ------
    ValueError
        If ``NA`` is not implemented.
    """
    if gamma_s_override is not None:
        return float(gamma_s_override)

    # Recommended EC2 values.  Additional annexes register their own
    # mapping here; the structure keeps the override a data change.
    _NA_TABLES = {
        'EC2': {'F': 1.15, 'A': 1.0, 'S': 1.0},
    }
    if NA not in _NA_TABLES:
        raise ValueError(
            f"National Annex '{NA}' not implemented for prestressing "
            f"steel. Available: {list(_NA_TABLES.keys())}. "
            f"Pass gamma_s_override to supply a value directly."
        )
    table = _NA_TABLES[NA]
    key = ls.upper() if isinstance(ls, str) else ls
    return table.get(key, table['S'])


# ---------------------------------------------------------------------------
#  Prestressing-steel property class
# ---------------------------------------------------------------------------

class fbpren:
    r"""
    Prestressing-steel design properties as per EN 1992-1-1 §3.3.

    Computes the design values of the idealized bilinear
    stress-strain diagram from the characteristic inputs, for a given
    limit state and National Annex.  It applies the formulas only and
    performs no class/range validity checks, exactly like
    :class:`~gensec.materials.ec2_properties.fben2`.

    Parameters
    ----------
    f_p01k : float
        Characteristic 0.1 % proof stress :math:`f_{p0.1k}` [MPa].
    f_pk : float
        Characteristic tensile strength :math:`f_{pk}` [MPa].
    eps_uk : float
        Characteristic strain at maximum force :math:`\varepsilon_{uk}`
        (positive, dimensionless).
    Ep : float, optional
        Modulus of elasticity [MPa].  Default 195000 (EN 1992-1-1
        §3.3.6(3) nominal value for strand; use 205000 for wires and
        bars).
    ls : str, optional
        Limit state: ``'F'``, ``'A'`` or ``'S'``.  Default ``'F'``.
    NA : str, optional
        National Annex.  Default ``'EC2'`` (recommended values).
    eps_ud_factor : float, optional
        Factor applied to :math:`\varepsilon_{uk}` to obtain the
        design ultimate strain :math:`\varepsilon_{ud}`.  EN 1992-1-1
        §3.3.6(7) recommends 0.9.  Default 0.9.
    gamma_s_override : float or None, optional
        Explicit :math:`\gamma_S`, bypassing the limit-state table.
        Default ``None``.

    Attributes
    ----------
    gamma_s : float
        Partial factor :math:`\gamma_S` for the chosen limit state.
    f_p01d : float
        Design proof stress :math:`f_{p0.1d} = f_{p0.1k}/\gamma_S`
        [MPa].
    f_pd : float
        Design tensile strength :math:`f_{pd} = f_{pk}/\gamma_S`
        [MPa].
    eps_ud : float
        Design ultimate strain
        :math:`\varepsilon_{ud} = k\,\varepsilon_{uk}` (default
        :math:`k = 0.9`).
    eps_el : float
        Strain at the end of the elastic branch,
        :math:`f_{p0.1d}/E_p`.
    Ep : float
        Modulus of elasticity [MPa].
    sigma_ud_inclined : float
        Stress at :math:`\varepsilon_{ud}` for the **inclined** design
        branch towards :math:`f_{pd}` (EN 1992-1-1 §3.3.6(7),
        second bullet): linear interpolation between
        :math:`(\varepsilon_{el}, f_{p0.1d})` and
        :math:`(\varepsilon_{uk}, f_{pd})` evaluated at
        :math:`\varepsilon_{ud}`.
    relaxation_class : int or None
        EN 1992-1-1 §3.3.2 relaxation class (1, 2 or 3).  Stored for a
        later time-dependent-loss phase; **not used** here.
    rho_1000 : float or None
        Relaxation loss at 1000 h.  Placeholder for the loss phase.

    References
    ----------
    - EN 1992-1-1 §3.3.
    - EN 1992-1-1 Table 2.1N (partial factors).

    Examples
    --------
    >>> ps = fbpren(f_p01k=1600.0, f_pk=1860.0, eps_uk=0.035)
    >>> round(ps.f_p01d, 1)
    1391.3
    >>> round(ps.f_pd, 1)
    1617.4
    >>> round(ps.eps_ud, 5)
    0.0315
    """

    def __init__(self, f_p01k, f_pk, eps_uk, Ep=195000.0,
                 ls='F', NA='EC2', eps_ud_factor=0.9,
                 gamma_s_override=None,
                 relaxation_class=None, rho_1000=None):
        self.f_p01k = float(f_p01k)
        self.f_pk = float(f_pk)
        self.eps_uk = float(eps_uk)
        self.Ep = float(Ep)
        self.ls = ls
        self.NA = NA

        # Partial factor from limit state / national annex.
        self.gamma_s = gamma_s_prestress(
            ls=ls, NA=NA, gamma_s_override=gamma_s_override)

        # Design strengths.
        self.f_p01d = self.f_p01k / self.gamma_s
        self.f_pd = self.f_pk / self.gamma_s

        # Strains.
        self.eps_el = self.f_p01d / self.Ep
        self.eps_ud = eps_ud_factor * self.eps_uk

        # Inclined design branch endpoint stress evaluated at eps_ud.
        # The EC2 inclined branch runs from (eps_el, f_p01d) to
        # (eps_uk, f_pd); we report its value at eps_ud so the bridge
        # can build either the horizontal or the inclined diagram.
        denom = self.eps_uk - self.eps_el
        if denom > 0.0:
            slope = (self.f_pd - self.f_p01d) / denom
            self.sigma_ud_inclined = (
                self.f_p01d + slope * (self.eps_ud - self.eps_el))
        else:
            self.sigma_ud_inclined = self.f_p01d

        # ---- Time-dependent relaxation (Phase 5 placeholders) ----
        # EN 1992-1-1 §3.3.2.  Not computed in this phase; stored so a
        # later loss module can populate the diagram/effective
        # prestrain without an interface change.
        self.relaxation_class = relaxation_class
        self.rho_1000 = rho_1000

    def __repr__(self):
        return (
            f"fbpren(f_p01k={self.f_p01k:.0f}, f_pk={self.f_pk:.0f}, "
            f"eps_uk={self.eps_uk:.4f}, gamma_s={self.gamma_s:.2f}, "
            f"f_p01d={self.f_p01d:.1f}, f_pd={self.f_pd:.1f}, "
            f"eps_ud={self.eps_ud:.4f})"
        )


# ---------------------------------------------------------------------------
#  Standard designations (EN 10138)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PSClass:
    r"""
    Characteristic data for one standard prestressing-steel
    designation.

    Attributes
    ----------
    f_p01k : float
        Characteristic 0.1 % proof stress [MPa].
    f_pk : float
        Characteristic tensile strength [MPa].
    eps_uk : float
        Characteristic strain at maximum force.
    Ep : float
        Nominal modulus of elasticity [MPa].
    kind : str
        ``'strand'``, ``'wire'`` or ``'bar'``.
    """
    f_p01k: float
    f_pk: float
    eps_uk: float
    Ep: float
    kind: str


#: Standard prestressing-steel designations (representative EN 10138
#: values).  The proof stress is taken as :math:`f_{p0.1k}\approx 0.88
#: f_{pk}` for strand/wire where a product value is not tabulated; these
#: are convenience defaults and a project may override every number via
#: explicit ``f_p01k``/``f_pk``/``eps_uk`` inputs.
PrestressClassData: Dict[str, _PSClass] = {
    # 7-wire strands (Ep ~ 195 GPa)
    "Y1770S7": _PSClass(f_p01k=1520.0, f_pk=1770.0, eps_uk=0.035,
                        Ep=195000.0, kind="strand"),
    "Y1860S7": _PSClass(f_p01k=1600.0, f_pk=1860.0, eps_uk=0.035,
                        Ep=195000.0, kind="strand"),
    "Y1960S7": _PSClass(f_p01k=1680.0, f_pk=1960.0, eps_uk=0.035,
                        Ep=195000.0, kind="strand"),
    # Cold-drawn wires (Ep ~ 205 GPa)
    "Y1670C":  _PSClass(f_p01k=1440.0, f_pk=1670.0, eps_uk=0.035,
                        Ep=205000.0, kind="wire"),
    "Y1770C":  _PSClass(f_p01k=1520.0, f_pk=1770.0, eps_uk=0.035,
                        Ep=205000.0, kind="wire"),
    # Hot-rolled / heat-treated bars (Ep ~ 205 GPa)
    "Y1030H":  _PSClass(f_p01k=835.0, f_pk=1030.0, eps_uk=0.035,
                        Ep=205000.0, kind="bar"),
    "Y1100H":  _PSClass(f_p01k=900.0, f_pk=1100.0, eps_uk=0.035,
                        Ep=205000.0, kind="bar"),
}
