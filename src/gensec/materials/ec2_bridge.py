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
Bridge between EC2 property classes and GenSec materials.

Provides factory functions that create :class:`Concrete` and
:class:`Steel` GenSec material objects from the EC2 / EN 10025
property classes, automatically extracting all constitutive parameters.

This avoids hard-coding EC2 Table 3.1 values in GenSec — they are
computed once by the EC2 property classes and passed through.

Examples
--------
Create a C30/37 concrete for fundamental ULS, slow loading, at 28 days:

>>> from gensec.materials.ec2_bridge import concrete_from_ec2
>>> c = concrete_from_ec2(fck=30, ls='F', loadtype='slow', TypeConc='R')
>>> c.fcd
17.0
>>> c.eps_cu2
-0.0035

Create an S355 structural steel plate, 20 mm thick:

>>> from gensec.materials.ec2_bridge import steel_from_en10025
>>> s = steel_from_en10025(grade='S355', t=20)
>>> s.fyk
345
"""

import numpy as np
from .concrete import Concrete
from .steel import Steel, PrestressingSteel
from .ec2_properties import fben2, ConcClassFck
from .en10025_properties import Steel_EN10025_2
from .prestress_properties import fbpren, PrestressClassData


def concrete_from_ec2(fck, ls='F', loadtype='slow', TypeConc='R',
                      NA='French', time=28, enable_tension=False,
                      tension_fct='fctd'):
    r"""
    Create a GenSec :class:`Concrete` from EC2 Table 3.1 properties.

    This function instantiates the EC2 :class:`fben2` class to
    compute all EC2 properties (including the correct
    :math:`\varepsilon_{c2}`, :math:`\varepsilon_{cu2}`, :math:`n`,
    :math:`\alpha_{cc}`, :math:`\gamma_c` for the chosen National Annex
    and limit state), then builds a GenSec ``Concrete`` with those values.

    Parameters
    ----------
    fck : float
        Characteristic cylinder strength [MPa].
    ls : str, optional
        Limit state: ``'F'`` (fundamental), ``'A'`` (accidental), or
        ``'S'`` (service). Default ``'F'``.
    loadtype : str, optional
        ``'slow'`` or ``'fast'``. Default ``'slow'``.
    TypeConc : str, optional
        Cement type: ``'R'``, ``'N'``, or ``'S'``. Default ``'R'``.
    NA : str, optional
        National Annex. Default ``'French'``.
    time : float, optional
        Concrete age [days]. Default 28.
    enable_tension : bool, optional
        If ``True``, the returned :class:`Concrete` includes a linear
        tension branch.  The tensile strength and elastic modulus are
        taken from the EC2 property object.  Default ``False``.
    tension_fct : str, optional
        Which tensile strength to use when ``enable_tension=True``:
        ``'fctd'`` for design value :math:`f_{ctd,0.05}` (default),
        ``'fctm'`` for mean value :math:`f_{ctm}`, or ``'fctk'`` for
        characteristic value :math:`f_{ctk,0.05}`.

    Returns
    -------
    Concrete
        GenSec material with all parameters from EC2.

    Notes
    -----
    The returned ``Concrete`` object also carries an ``ec2`` attribute
    holding the full :class:`fben2` instance for access to all
    EN 1992-1-1 properties (fcm, fctm, Ecm, etc.).
    """
    ec2 = fben2(fck=fck, ls=ls, loadtype=loadtype,
                TypeConc=TypeConc, NA=NA, time=time)

    # Resolve tensile strength from the requested source.
    fct_val = 0.0
    Ec_val = 0.0
    if enable_tension:
        _fct_map = {
            'fctd': ec2.fctd_005,
            'fctm': ec2.fctm,
            'fctk': ec2.fctk_005,
        }
        key = tension_fct.lower()
        if key not in _fct_map:
            raise ValueError(
                f"Unknown tension_fct='{tension_fct}'. "
                f"Valid: {list(_fct_map.keys())}"
            )
        fct_val = _fct_map[key]
        Ec_val = ec2.ecm

    # Note: fben2 stores eps_c2 and eps_cu2 as POSITIVE values.
    # GenSec uses NEGATIVE convention for compression.
    c = Concrete(
        fck=ec2.fck,
        gamma_c=ec2.gamma_c,
        alpha_cc=ec2.alpha_cc,
        n_parabola=ec2.n_exp,
        eps_c2=-ec2.eps_c2,      # convert to negative
        eps_cu2=-ec2.eps_cu2,     # convert to negative
        fct=fct_val,
        Ec=Ec_val,
    )
    # Attach the full EC2 object for downstream access
    c.ec2 = ec2
    return c


def concrete_from_class(conc_class, ls='F', loadtype='slow',
                        TypeConc='R', NA='French', time=28,
                        enable_tension=False, tension_fct='fctd'):
    r"""
    Create a GenSec :class:`Concrete` from an EC2 class name.

    Parameters
    ----------
    conc_class : str
        EC2 class name, e.g. ``'C25/30'``, ``'C30/37'``, etc.
    ls, loadtype, TypeConc, NA, time
        Passed through to :func:`concrete_from_ec2`.
    enable_tension : bool, optional
        Activate the linear tension branch. Default ``False``.
    tension_fct : str, optional
        Tensile strength source (``'fctd'``, ``'fctm'``, or
        ``'fctk'``). Default ``'fctd'``.

    Returns
    -------
    Concrete

    Raises
    ------
    ValueError
        If the class name is not recognized.
    """
    if conc_class not in ConcClassFck:
        raise ValueError(
            f"Unknown concrete class '{conc_class}'. "
            f"Valid: {list(ConcClassFck.keys())}"
        )
    return concrete_from_ec2(
        fck=ConcClassFck[conc_class], ls=ls, loadtype=loadtype,
        TypeConc=TypeConc, NA=NA, time=time,
        enable_tension=enable_tension, tension_fct=tension_fct,
    )


def steel_from_en10025(grade='S355', t=0, young=200000,
                       gamma_s=1.0, eps_su=0.05):
    r"""
    Create a GenSec :class:`Steel` from EN 10025-2 properties.

    This is for **structural steel** (plates, profiles), not
    reinforcing bars. The yield strength depends on thickness.

    Parameters
    ----------
    grade : str
        Steel grade: ``'S235'``, ``'S275'``, or ``'S355'``.
    t : float, optional
        Thickness [mm]. Default 0.
    young : float, optional
        Young's modulus [MPa]. Default 200000.
    gamma_s : float, optional
        Partial safety factor. Default 1.0 (no reduction for
        structural steel at ULS; use 1.0 or as per code).
    eps_su : float, optional
        Ultimate strain. Default 0.05 (5%).

    Returns
    -------
    Steel
        GenSec material. ``works_in_compression=True`` always for
        structural steel.

    Notes
    -----
    The returned ``Steel`` object also carries an ``en10025``
    attribute with the full :class:`Steel_EN10025_2` instance.
    """
    en = Steel_EN10025_2(grade=grade, t=t, young=young)
    s = Steel(
        fyk=en.f_yk,
        gamma_s=gamma_s,
        Es=young,
        k_hardening=en.f_uk / en.f_yk if en.f_yk > 0 else 1.0,
        eps_su=eps_su,
        works_in_compression=True,
    )
    s.en10025 = en
    return s


def prestress_from_ec2(f_p01k, f_pk, eps_uk, Ep=195000.0,
                       ls='F', NA='EC2', eps_ud_factor=0.9,
                       gamma_s_override=None, diagram='horizontal',
                       works_in_compression=True):
    r"""
    Create a GenSec :class:`PrestressingSteel` from EC2 §3.3 inputs.

    Instantiates the EC2 property class
    :class:`~gensec.materials.prestress_properties.fbpren` to resolve
    :math:`\gamma_S` (from the limit state / National Annex) and the
    design diagram values, then builds a GenSec
    :class:`~gensec.materials.steel.PrestressingSteel`.  This is the
    prestressing-steel analogue of :func:`concrete_from_ec2`.

    Parameters
    ----------
    f_p01k : float
        Characteristic 0.1 % proof stress [MPa].
    f_pk : float
        Characteristic tensile strength [MPa].
    eps_uk : float
        Characteristic strain at maximum force.
    Ep : float, optional
        Modulus of elasticity [MPa].  Default 195000.
    ls : str, optional
        Limit state: ``'F'``, ``'A'`` or ``'S'``.  Default ``'F'``.
    NA : str, optional
        National Annex.  Default ``'EC2'``.
    eps_ud_factor : float, optional
        Factor for :math:`\varepsilon_{ud} = k\,\varepsilon_{uk}`.
        Default 0.9 (EN 1992-1-1 §3.3.6(7)).
    gamma_s_override : float or None, optional
        Explicit :math:`\gamma_S`.  Default ``None`` (use table).
    diagram : {'horizontal', 'inclined'}, optional
        Idealization of the design diagram (EN 1992-1-1 §3.3.6(7)):

        - ``'horizontal'`` — top branch at :math:`f_{p0.1d}` with no
          strain limit need (the limit is still applied at
          :math:`\varepsilon_{ud}`).
        - ``'inclined'`` — branch rising to the interpolated stress at
          :math:`\varepsilon_{ud}` towards :math:`f_{pd}`.

        Default ``'horizontal'``.
    works_in_compression : bool, optional
        Symmetric diagram if ``True``.  Default ``True``.

    Returns
    -------
    PrestressingSteel
        GenSec material with design values from EC2.  Carries an
        ``ec2`` attribute holding the :class:`fbpren` instance.

    Notes
    -----
    The ``diagram`` choice maps onto the generic
    :class:`PrestressingSteel` purely through the second-branch
    endpoint stress ``sigma_ud``: ``'horizontal'`` passes
    :math:`f_{p0.1d}` (zero slope), ``'inclined'`` passes the
    interpolated :math:`\sigma_{ud}`.  No constitutive branching is
    needed, which is why the material itself stays idealization-free.
    """
    ec2 = fbpren(f_p01k=f_p01k, f_pk=f_pk, eps_uk=eps_uk, Ep=Ep,
                 ls=ls, NA=NA, eps_ud_factor=eps_ud_factor,
                 gamma_s_override=gamma_s_override)

    if diagram == 'inclined':
        sigma_ud = ec2.sigma_ud_inclined
    elif diagram == 'horizontal':
        sigma_ud = ec2.f_p01d
    else:
        raise ValueError(
            f"Unknown diagram='{diagram}'. Use 'horizontal' or "
            f"'inclined'."
        )

    ps = PrestressingSteel(
        f_p01d=ec2.f_p01d,
        sigma_ud=sigma_ud,
        eps_ud=ec2.eps_ud,
        Ep=ec2.Ep,
        works_in_compression=works_in_compression,
    )
    ps.ec2 = ec2
    return ps


def prestress_from_class(ps_class, ls='F', NA='EC2',
                         eps_ud_factor=0.9, gamma_s_override=None,
                         diagram='horizontal',
                         works_in_compression=True):
    r"""
    Create a GenSec :class:`PrestressingSteel` from a standard
    designation (EN 10138).

    Parameters
    ----------
    ps_class : str
        Designation key from
        :data:`~gensec.materials.prestress_properties.PrestressClassData`
        (e.g. ``'Y1860S7'``).
    ls, NA, eps_ud_factor, gamma_s_override, diagram,
    works_in_compression
        Passed through to :func:`prestress_from_ec2`.

    Returns
    -------
    PrestressingSteel

    Raises
    ------
    ValueError
        If the designation is not recognized.
    """
    if ps_class not in PrestressClassData:
        raise ValueError(
            f"Unknown prestressing-steel class '{ps_class}'. "
            f"Valid: {list(PrestressClassData.keys())}"
        )
    d = PrestressClassData[ps_class]
    return prestress_from_ec2(
        f_p01k=d.f_p01k, f_pk=d.f_pk, eps_uk=d.eps_uk, Ep=d.Ep,
        ls=ls, NA=NA, eps_ud_factor=eps_ud_factor,
        gamma_s_override=gamma_s_override, diagram=diagram,
        works_in_compression=works_in_compression,
    )
