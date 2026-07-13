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
#!/usr/bin/python
r'''
"""Concrete properties as per EN 1992-1-1.

Implements the :class:`fben2` class that computes all mechanical
properties of concrete from the characteristic compressive strength
:math:`f_{ck}`, following EN 1992-1-1 Table 3.1 and the French
National Annex (NF EN 1992-1-1/NA).

Also provides the Sargin and Parabola-Rectangle constitutive laws
(currently used only for completeness — the main verification modules
use the discrete property values).

.. note::

    Only the French National Annex is implemented at this time.
    
"""
'''
from math import sqrt
import numpy as np

#############################################
# List of functions
#############################################

# en2pr(strain,en2class) --> Parabole-Rectangle law
# en2sargin(strain,en2class) --> Sargin law
#   These functions are the main EN 1992 constitutive laws.

#############################################
# List of classes
#############################################

# fben2(fck=30, uls='F', loadtype='slow', t=28, TypeConc='r')
#   This class create the concrete properties as per EN 1992 + French National
#   Annex.

#############################################
#############################################
#############################################
#

# EN 1992-1-1: concrete properties as per Eurocode 2 (from fck)
# THE FRENCH ANNEX IS USED (see self.alphacc and self.alphact)


class fben2:
    r'''
    This class creates an object that describes as per the Eurocode 1992-1-1
    the concrete characteristics for design.
    The French National Annex is used.
    It doesn't include any limits or checks on the concrete classes,
    but it just applies the formulas that describes all the mechanical
    characteristics, at a given time.


    Args
    ----

    fck : float
        Characteristic resistance in compression of concrete.
        Usually is considered at 28 days, but we're not limited to.
        This value is the **nominal** value of concrete.
    ls : str
        Limit state which we consider ('F', 'A' or anything else).
    loadtype : str
        Speed of load application. Usually is 'slow'.
    TypeConc : str
        Ciment type ('R', 'S' or 'N') as per EN 1992-1-1 §3.1.2. The complete
        classification of cimentrs can be found in EN 197. See also the Notes
        section below.
    NA : str
        is the choosen National Annex. At the moment, only the French one
        (default value) is supported.
    time : float
        Concrete age. The default value is 28.


    Attributes
    ----------

    gamma_c : float
        Safety coefficient :math:`\gamma_{c}` (Table 2.1N
        EN 1992-1-1) associated to concrete for the input limit state
    alpha_cc : float
        Coefficient :math:`{\\alpha}_{cc}` (§3.1.6
        EN 1992-1-1) that accounts for long-term effects on concrete resistance
        in compression (see also :math:`k_{t}` coefficient at §3.1.2)
    alpha_ct : float
        Coefficient :math:`{\\alpha}_{ct}` (§3.1.6
        EN 1992-1-1) that accounts for long-term effects on concrete resistance
        in tension (see also :math:`k_{t}` coefficient at §3.1.2)
    ecm : float
        Young's module :math:`E_{cm}` at the specified concrete age
        as per the EN 1992-1-1, Table 3.1. Please pay attention to the fact
        that RCC-CW may introduce specific concrete Young's modules for
        different situations
    fcd : float
        Design resistance of concrete :math:`f_{cd}` in compression at the
        specificied concrete age
    fctd_005 : float
        Design resistance of concrete :math:`f_{ctd,0.05}` in tension at the
        specificied concrete age (fractile 5%)
    ecm_28 : float
        Young module :math:`E_{cm}` at the 28 days
        as per the EN 1992-1-1, Table 3.1. Please pay attention to the fact
        that RCC-CW may introduce specific concrete Young's modules for
        different situations
    fcd_28 : float
        Design resistance of concrete in compression at 28 days
    fctd_005_28 : float
        Design resistance of concrete in compression at 28 days (fractile 5%)
    fck : float
        Characteristic resistance of concrete in compression at the specified
        concrete age.
        Usually is considered at 28 days, but we're not limited to
    fcm : float
        Medium design resistance of concrete in compression at the specified
        concrete age.
        Usually is considered at 28 days, but we're not limited to
    fctk_005 : float
        Characteristic resistance of concrete in tension at the
        specificied concrete age (fractile 5%)
    fctk_095 : float
        Characteristic resistance of concrete in tension at the
        specificied concrete age (fractile 95%)
    fctm : float
        Medium design resistance of concrete in tension at the specified
        concrete age.
        Usually is considered at 28 days, but we're not limited to
    fck_28 : float
        Characteristic resistance of concrete in compression at 28 days
    fcm_28 : float
        Medium design resistance of concrete in compression at 28 days
    fctk_005_28 : float
        Characteristic resistance of concrete in tension at 28 days
        (fractile 5%)
    fctk_095_28 : float
        Characteristic resistance of concrete in tension at 28 days
        (fractile 95%)
    fctm_28 : float
        Medium design resistance of concrete in tension at 28 days
    eps_c1 : float
    eps_cu1 : float
    eps_c2 : float
    eps_cu2 : float
    n : float
    eps_c3 : float
    eps_cu3 : float
    eps_c1_28 : float
    eps_cu1_28 : float
    eps_c2_28 : float
    eps_cu2_28 : float
    n_28 : float
    eps_c3_28 : float
    eps_cu3_28 : float
    dilat : float
    s : float
    beta_cc : float
    eps_ct : float
        Mean cracking strain at time *t*:
        :math:`\varepsilon_{ct} = f_{ctm} / E_{cm}`.
    eps_ctd : float
        Design cracking strain at time *t*:
        :math:`\varepsilon_{ctd} = f_{ctd,0.05} / E_{cm}`.
    eps_ct_28 : float
        Mean cracking strain at 28 days.
    eps_ctd_28 : float
        Design cracking strain at 28 days.

    References
    ----------

    - EN 1992-1-1 (1st generation)
    - EN 197
    - RCC-CW 2021

        Here below we show the correspondence between EN 197 composition and
        ciment class as per EN 1992-1-1:

        +----------------+---------------------------+
        |*Ciment class*  |*Admissible composition*   |
        +================+===========================+
        |**R**           |CEM 42.5 R,                |
        |                |CEM 52.5 N,                |
        |                |CEM 52.5 R                 |
        +----------------+---------------------------+
        |**N**           |CEM 32.5 R,                |
        |                |CEM 42.5 N                 |
        +----------------+---------------------------+
        |**S**           |CEM 32.5 N                 |
        +----------------+---------------------------+

    '''

    def __init__(self,
                 fck: float,
                 ls: str,
                 loadtype: str,
                 TypeConc: str,
                 NA='French',
                 time=28,
                 ):
        """Initialise a concrete properties object from :math:`f_{ck}`.

        All EN 1992-1-1 Table 3.1 properties are computed: :math:`f_{cm}`,
        :math:`f_{ctm}`, :math:`E_{cm}`, :math:`\\varepsilon_{c1}`,
        :math:`\\varepsilon_{cu1}`, :math:`f_{cd}`, :math:`f_{ctd}`, and
        the time-dependent coefficient :math:`\\beta_{cc}(t)`.

        Parameters
        ----------
        fck : float
            Characteristic compressive strength [MPa].
        ls : str
            Limit state: ``'F'`` (fundamental), ``'A'`` (accidental), or
            ``'S'`` (service).
        loadtype : str, optional
            Speed of load application.  Default is ``'slow'``.
        TypeConc : str, optional
            Cement type (``'R'``, ``'S'`` or ``'N'``).  Default is ``'R'``.
        NA : str, optional
            National Annex code.  Default is ``'FR'`` (French).
        time : float, optional
            Concrete age [days].  Default is 28.
        """
        # fck en [MPa], t en [jours], TypeConc is a string value

        # Limit State
        if ls in ('f', 'F'):
            self.gamma_c = 1.5
        elif ls in ('a', 'A'):
            self.gamma_c = 1.2
        else:
            self.gamma_c = 1
        # Time of concrete properties evaluation
        self.time = time  # [days]
        # Load type --> SEE THE NATIONAL ANNEX
        if NA == 'French':
            if time > 28:
                kt = 1
            else:
                kt = 0.85
            if loadtype in ('slow', 'Slow', 'SLOW'):
                # LINTED == 'slow' or loadtype == 'Slow' or loadtype == 'SLOW':
                self.alpha_cc = 1*kt
                self.alpha_ct = 1*kt
            else:
                self.alpha_cc = 1*kt
                self.alpha_ct = 1*kt
        else:
            raise ValueError('National Annex not yet implemented.')
        # Definition of ciment type (CEM) in concrete
        if TypeConc in ('r', 'R'):  # Linted == 'r'or TypeConc == 'R':
            self.s_cem = 0.20
        elif TypeConc in ('n', 'N'):  # Linted == 'n' or TypeConc == 'N':
            self.s_cem = 0.25
        elif TypeConc in ('s', 'S'):  # Linted == 's' or TypeConc == 'S':
            self.s_cem = 0.38
        else:
            self.s_cem = 0
        self.beta_cc = np.exp(self.s_cem*(1-sqrt(28/self.time)))
        # Fundamental concrete compression properties evaluation in
        # function of time
        self.fcm_28 = fck + 8  # [MPa]
        self.fck_28 = fck  # [MPa]
        # F1 (Phase-5): the exponent alpha of EN 1992-1-1 eq. (3.4),
        #     f_ctm(t) = [beta_cc(t)]^alpha * f_ctm ,
        # is alpha = 1 for t < 28 d and alpha = 2/3 for t >= 28 d -- the
        # tensile strength drops FASTER than beta_cc when young and grows
        # SLOWER when mature.  It was inverted.  Inert at t = 28 d, where
        # beta_cc = 1; it bites only away from it, and Phase 5 is the first
        # consumer of fben2 at t != 28.
        # alpha belongs to f_ctm and to NOTHING ELSE -- see the modulus (F2).
        if time >= 28:
            self.fcm = fck + 8  # [MPa]
            self.fck = fck  # [MPa]
            alpha = 2/3
        elif time < 3:
            self.fcm = 0  # [MPa]
            self.fck = 0  # [MPa]
            alpha = 0
        else:
            self.fcm = self.beta_cc * self.fcm_28  # [MPa]
            self.fck = self.fcm - 8  # [MPa]
            alpha = 1
        # Other concrete properties evaluations
        self.eps_c1 = min(2.8, (0.7*(self.fcm**0.31))) / 10**3
        self.eps_c1_28 = min(2.8, (0.7*(self.fcm_28**0.31))) / 10**3
        # F1c: the strength CLASS is a 28-day property.  Branching on the
        # AGED self.fck made a C60 at 3 days (fck(3) = 32.7 MPa) take the
        # C<=50 formula -- ageing must not change which equation applies.
        if self.fck_28 <= 50:
            # with time influence:
            # F1b: EN 1992-1-1 (3.4) is  f_ctm(t) = beta_cc^alpha * f_ctm ,
            # with f_ctm the 28-DAY value.  Using self.fck (already aged,
            # fck(t) = beta_cc*fcm_28 - 8) double-counts beta_cc -- the same
            # vice the modulus had (F2).  Reference it to fck_28.
            self.fctm = ((self.beta_cc**alpha)
                         * 0.3*(self.fck_28**(2/3)))  # [MPa]
            self.eps_cu1 = 3.5 / 10**3  # [no unit] --> it is not in ‰
            self.eps_c2 = 2 / 10**3  # [no unit] --> it is not in ‰
            self.eps_cu2 = 3.5 / 10**3  # [no unit] --> it is not in ‰
            self.n_exp = 2
            self.eps_c3 = 1.75 / 10**3  # [no unit] --> it is not in ‰
            self.eps_cu3 = 3.5 / 10**3  # [no unit] --> it is not in ‰
            # without time influence:
            self.fctm_28 = 0.3*self.fck_28**(2/3)  # [MPa]
            self.eps_cu1_28 = 3.5 / 10**3  # [no unit] --> it is not in ‰
            self.eps_c2_28 = 2 / 10**3  # [no unit] --> it is not in ‰
            self.eps_cu2_28 = 3.5 / 10**3  # [no unit] --> it is not in ‰
            self.n_exp_28 = 2
            self.eps_c3_28 = 1.75 / 10**3  # [no unit] --> it is not in ‰
            self.eps_cu3_28 = 3.5 / 10**3  # [no unit] --> it is not in ‰
        else:
            # with time influence:
            # F1b (C55+ branch): same -- reference to the 28-day f_cm.
            self.fctm = ((self.beta_cc**alpha)
                         * 2.12*np.log(1+(self.fcm_28/10)))  # [MPa]
            # F1d: the strain limits of Table 3.1 are CLASS properties -- they
            # are not time-dependent.  Evaluating them on the AGED fck made a
            # C60 at 3 d (fck(3) = 32.7 MPa) raise a NEGATIVE base to the power
            # 0.53 and produce a silent **NaN**, which then propagated into the
            # constitutive law.  Reference them to the 28-day class.
            self.eps_cu1 = max((2.8+27*((98-self.fcm_28)/100)**4) / 10**3,
                               self.eps_c1)  # [no unit] --> it is not in ‰
            self.eps_c2 = (2 + 0.085*((self.fck_28-50)**0.53)) / 10**3 \
                # [no unit] --> it is not in ‰
            self.eps_cu2 = max((2.6 + 35*((90-self.fck_28)/100)**4) / 10**3,
                               self.eps_c2)  # [no unit] --> it is not in ‰
            self.n_exp = 1.4 + 23.4*((90-self.fck_28)/100)**4
            self.eps_c3 = (1.75 + 0.55*((self.fck_28-50)/40)) / 10**3 \
                # [no unit] --> it is not in ‰
            self.eps_cu3 = max((2.6 + 35*((90-self.fck_28)/100)**4) / 10**3,
                               self.eps_c3)  # [no unit] --> it is not in ‰
            # without time influence:
            self.fctm_28 = 2.12*np.log(1+(self.fcm_28/10))  # [MPa]
            self.eps_cu1_28 = max((2.8+27*((98-self.fcm_28)/100)**4) / 10**3,
                                  # [no unit] --> it is not in ‰
                                  self.eps_c1_28)
            self.eps_c2_28 = (2 + 0.085*((self.fck_28-50)**0.53)) / 10**3 \
                # [no unit] --> it is not in ‰
            self.eps_cu2_28 = max((2.6 + 35*((90-self.fck_28)/100)**4) / 10**3,
                                  # [no unit] --> it is not in ‰
                                  self.eps_c2_28)
            self.n_exp_28 = 1.4 + 23.4*((90-self.fck_28)/100)**4
            self.eps_c3_28 = (1.75 + 0.55*((self.fck_28-50)/40)) / 10**3 \
                # [no unit] --> it is not in ‰
            self.eps_cu3_28 = max((2.6 + 35*((90-self.fck_28)/100)**4) / 10**3,
                                  # [no unit] --> it is not in ‰
                                  self.eps_c3_28)
        self.fctk_005 = 0.7*self.fctm  # [MPa]
        self.fctk_095 = 1.3*self.fctm  # [MPa]
        # F2 (Phase-5): EN 1992-1-1 3.1.3(3) is
        #     E_cm(t) = [f_cm(t) / f_cm]^0.3 * E_cm = [beta_cc(t)]^0.3 * E_cm.
        # This line used to multiply by ((beta_cc ** alpha) ** 0.3) -- f_ctm's
        # exponent, which has no business in a modulus.  Below 28 d, where
        # self.fcm already carries a beta_cc, that DOUBLE-COUNTED it and gave
        # beta_cc^0.5 (~ -10% at 3 d on C45).  Merely inverting alpha (the F1
        # fix) does not repair it: it makes it beta_cc^0.6, AND it breaks the
        # t >= 28 branch, which was correct.  alpha must be REMOVED, not
        # flipped.
        self.ecm = ((1000)*22*(self.fcm_28/10)**(0.3)
                    * (self.beta_cc ** 0.3))  # [MPa]
        self.fctk_005_28 = 0.7*self.fctm_28  # [MPa]
        self.fctk_095_28 = 1.3*self.fctm_28  # [MPa]
        self.ecm_28 = (1000)*22*(self.fcm_28/10)**(0.3)  # [MPa]
        self.dilat = 10**-5  # [K^-1]
        self.fcd = self.alpha_cc * self.fck / self.gamma_c
        self.fcd28 = self.alpha_cc * self.fck_28 / self.gamma_c
        self.fctd_005 = self.alpha_ct * self.fctk_005 / self.gamma_c
        self.fctd_005_28 = self.alpha_ct * self.fctk_005_28 / self.gamma_c

        # Tensile strain limits (cracking strain).
        # These are used by GenSec to enable the optional linear
        # tension branch in the Concrete constitutive law (SLS).
        #   eps_ct  = fctm   / Ecm   (mean cracking strain at time t)
        #   eps_ctd = fctd   / Ecm   (design cracking strain at time t)
        if self.ecm > 0:
            self.eps_ct = self.fctm / self.ecm
            self.eps_ctd = self.fctd_005 / self.ecm
            self.eps_ct_28 = self.fctm_28 / self.ecm_28
            self.eps_ctd_28 = self.fctd_005_28 / self.ecm_28
        else:
            self.eps_ct = 0.0
            self.eps_ctd = 0.0
            self.eps_ct_28 = 0.0
            self.eps_ctd_28 = 0.0

# Parabole-Rectangle law


def en2pr(
        strain: float,
        en2class: fben2
) -> float:
    '''
    This function gives for a given strain and a concrete class (from fben2())
    the correspondent stress from the constitutive law 'Parabola-Rectangle'.
    
    '''
    if ((strain <= en2class.eps_cu2) & (strain >= 0)).any():
        if (strain >= en2class.eps_c2).any():
            stress = en2class.fcd
        else:
            stress = en2class.fcd * \
                (1-(1-(strain/en2class.eps_c2))**en2class.n_exp)
    else:
        stress = 0
    return stress

# Sargin law


def en2sargin(
        strain: float,
        en2class: fben2,
) -> float:
    '''
    This function gives for a given strain and a concrete class (from fben2())
    the correspondent stress from the constitutive law of Sargin.
    This constitutive law should be used only for non-linear applications.
    '''
    if abs(strain) <= abs(en2class.eps_cu1):
        k = 1.05*(en2class.ecm/1000)*abs(en2class.eps_c1) * \
            1000 / en2class.fcm_28
        eta = strain/en2class.eps_c1
        stress = en2class.fcm*(k*eta - eta**2)/(1+(k-2)*eta)
    else:
        stress = 0
    return stress

# Test of constitutive laws
# import Curve_Tracing as ct
# ct.checkcurvetracing(en2pr, 0, 0.0035, 0.0001, fben2(50))
# ct.checkcurvetracing(en2sargin, 0, 0.0035, 0.0001, fben2())
# a,x,y = ct.int_and_cog_curve(en2pr, 0.002, 0.0035, fben2(50))
# print(a,x,y)


ConcClassFck = {
    'C12/15': 12,
    'C16/20': 16,
    'C20/25': 20,
    'C25/30': 25,
    'C30/37': 30,
    'C35/45': 35,
    'C40/50': 40,
    'C45/55': 45,
    'C50/60': 50,
    'C55/67': 55,
    'C60/70': 60,
    'C70/85': 70,
    'C80/95': 80,
    'C90/105': 90,
    'C100/115': 100,
}
