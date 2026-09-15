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
            # F-M5 (Phase-16).  EN 1992-1-1 (3.1)/(3.2) give f_cm(t), and
            # (6) derives f_ck(t) = f_cm(t) - 8 MPa only for 3 d < t: below
            # three days that offset breaks down (it goes NEGATIVE on a
            # C30 at half a day), so the zero here is a *marker* for "this
            # code does not state it", guarded by the provider's declared
            # domain (``EC2RheologicalModel.DOMAIN``), which is the one
            # home of the number 3.
            #
            # ``alpha`` is NOT part of that marker.  It is the exponent of
            # expression (3.4),
            #     f_ctm(t) = [beta_cc(t)]^alpha * f_ctm ,
            # which is written on beta_cc(t) and the 28-DAY f_ctm and
            # never routes through f_cm(t); the clause states alpha = 1
            # for every t < 28 d and carves out no exception at three.
            # With alpha = 0 the factor collapsed to one and a concrete of
            # one day was handed the 28-day tensile strength: +192 % on a
            # C30/N against the value (3.4) prescribes at that age, and a
            # discontinuity at t = 3 in the NON-CONSERVATIVE direction --
            # the strength fell by a factor 1.672 as the concrete matured.
            # Registered in 12_2 §7 and repeated unchanged in 12_11,
            # 13_1 §13, 14_1 §10 and 15_0 §2.
            self.fcm = 0  # [MPa]  -- marker, see above
            self.fck = 0  # [MPa]  -- marker, see above
            alpha = 1
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
    r"""
    Stress of EN 1992-1-1 expression (3.14) at a given strain.

    The non-linear law for structural analysis --- Sargin's curve ---
    written on the three constants a concrete has at one age:

    .. math::

        \sigma_c = f_{cm}\,
            \frac{k\eta - \eta^{2}}{1 + (k-2)\eta},
        \qquad
        \eta = \frac{\varepsilon_c}{\varepsilon_{c1}},
        \qquad
        k = \frac{1.05\,E_{cm}\,\lvert\varepsilon_{c1}\rvert}{f_{cm}} .

    Sign convention is ``fben2``'s own and **not** GenSec's: here
    compression is positive, so a consumer working in the library's
    convention evaluates :math:`\sigma(\varepsilon) = -S(-\varepsilon)`.

    Age consistency (F-P26-0)
    -------------------------
    All three constants are read from the **same** ``en2class`` and
    therefore from the same age.  Until Phase 26 the shape
    parameter :math:`k` was divided by ``fcm_28`` while the stress
    was scaled by ``fcm``, i.e. :math:`f_{cm}(t)`: at 28 days the two
    coincide and nothing showed, and away from it the curve's
    initial tangent --- which is :math:`1.05 E_{cm}` by construction,
    because :math:`E_{cm}` is the secant at :math:`0.4 f_{cm}` and
    not the tangent at the origin --- drifted to :math:`0.8177 E_{cm}`
    at 7 days and :math:`0.6282 E_{cm}` at 3 days.  Those are the ages
    at which a tendon is stressed.  The peak is unaffected by the
    error --- at :math:`\eta = 1` the fraction is :math:`(k-1)/(k-1)`
    for any :math:`k` --- which is why a probe on the peak never
    saw it; see ``tests/test_phase26_service_law.py``.

    Parameters
    ----------
    strain : float
        Strain [-], compression **positive**.
    en2class : fben2
        The normative constants, at the age of interest.  This
        function derives none of them: it is handed :math:`f_{cm}`,
        :math:`E_{cm}`, :math:`\varepsilon_{c1}` and
        :math:`\varepsilon_{cu1}` and forms only the curve.

    Returns
    -------
    float
        Stress [MPa], compression positive; zero beyond
        :math:`\lvert\varepsilon_{cu1}\rvert`.

    Raises
    ------
    ValueError
        If ``en2class`` carries a zero strain at peak or a non-positive
        mean strength.  Below three days ``fben2`` reports
        :math:`f_{cm} = 0` as a marker of its own domain, and
        :math:`\varepsilon_{c1}` follows it to zero.  A curve through a
        zero strength is not a soft concrete, it is no concrete: the
        constants are outside the domain their source claims, and
        that is an undeclared input rather than unmodelled physics ---
        hence ``ValueError`` and not ``NotImplementedError``, and
        never the ``ZeroDivisionError`` of an arithmetic accident.

    Examples
    --------
    >>> props = fben2(fck=35.0, ls='F', loadtype='slow', TypeConc='N')
    >>> round(en2sargin(props.eps_c1, props), 9) == props.fcm
    True
    """
    if en2class.eps_c1 == 0 or en2class.fcm <= 0:
        raise ValueError(
            f"EN 1992-1-1 (3.14) needs a strain at peak and a"
            f" mean strength at the age it is asked about;"
            f" the constants handed in carry"
            f" eps_c1={en2class.eps_c1!r} and"
            f" fcm={en2class.fcm!r}, which fben2 reports as its"
            f" own out-of-domain marker (age"
            f" {en2class.time!r} d, below the"
            f" three-day floor).  Declare an age inside the"
            f" domain, or a law that covers it."
        )
    if abs(strain) <= abs(en2class.eps_cu1):
        k = 1.05*(en2class.ecm/1000)*abs(en2class.eps_c1) * \
            1000 / en2class.fcm
        eta = strain/en2class.eps_c1
        stress = en2class.fcm*(k*eta - eta**2)/(1+(k-2)*eta)
    else:
        stress = 0
    return stress

def en2sargin_array(strain, en2class):
    r"""
    Vectorised form of :func:`en2sargin`, on the same constants.

    Expression (3.14) evaluated on a whole array at once.  This is not
    a second law: the scalar function above and this one form the same
    curve from the same four constants of the same ``en2class``, and
    ``tests/test_phase26_service_law.py`` binds them element by element
    so neither can drift from the other.  It exists only because the
    fibre integrator asks for a strain plane, not for one fibre at a
    time, and a Python-level loop over ten thousand fibres inside a
    Newton iteration is a cost the sectional solver cannot carry.

    Parameters
    ----------
    strain : array_like
        Strain [-], compression **positive**, as in :func:`en2sargin`.
    en2class : fben2
        The normative constants, at the age of interest.

    Returns
    -------
    numpy.ndarray
        Stress [MPa], compression positive; zero beyond
        :math:`\lvert\varepsilon_{cu1}\rvert`, with the shape of
        ``strain``.

    Raises
    ------
    ValueError
        On the same out-of-domain constants as :func:`en2sargin`.
    """
    if en2class.eps_c1 == 0 or en2class.fcm <= 0:
        raise ValueError(
            f"EN 1992-1-1 (3.14) needs a strain at peak and a"
            f" mean strength at the age it is asked about;"
            f" the constants handed in carry"
            f" eps_c1={en2class.eps_c1!r} and"
            f" fcm={en2class.fcm!r}, which fben2 reports as its"
            f" own out-of-domain marker (age"
            f" {en2class.time!r} d, below the"
            f" three-day floor).  Declare an age inside the"
            f" domain, or a law that covers it."
        )
    e = np.asarray(strain, dtype=float)
    k = 1.05*(en2class.ecm/1000)*abs(en2class.eps_c1) * \
        1000 / en2class.fcm
    inside = np.abs(e) <= abs(en2class.eps_cu1)
    eta = np.where(inside, e/en2class.eps_c1, 0.0)
    den = 1.0 + (k - 2.0)*eta
    out = np.where(inside,
                   en2class.fcm*(k*eta - eta*eta)/den,
                   0.0)
    return out


def en2sargin_tangent_array(strain, en2class):
    r"""
    Derivative of expression (3.14) with respect to strain.

    Differentiating the quotient of :func:`en2sargin` gives

    .. math::

        \frac{d\sigma_c}{d\varepsilon_c} =
            \frac{f_{cm}}{\varepsilon_{c1}}\;
            \frac{(k - 2\eta)\,[1 + (k-2)\eta] - (k\eta - \eta^{2})(k-2)}
                 {[1 + (k-2)\eta]^{2}} ,

    which at :math:`\eta = 0` collapses to :math:`f_{cm}k/\varepsilon_{c1}
    = 1.05\,E_{cm}`: the initial tangent the shape parameter exists to
    produce.  Beyond :math:`\lvert\varepsilon_{cu1}\rvert` the stress is
    identically zero and so is its derivative.

    The closed form is checked against a central difference of
    :func:`en2sargin_array` in ``tests/test_phase26_service_law.py``, so
    it cannot drift away from the curve it claims to differentiate.

    Parameters
    ----------
    strain : array_like
        Strain [-], compression **positive**.
    en2class : fben2
        The normative constants, at the age of interest.

    Returns
    -------
    numpy.ndarray
        Slope [MPa], in the same sign convention as the stress.

    Raises
    ------
    ValueError
        On the same out-of-domain constants as :func:`en2sargin`.
    """
    if en2class.eps_c1 == 0 or en2class.fcm <= 0:
        raise ValueError(
            f"EN 1992-1-1 (3.14) needs a strain at peak and a"
            f" mean strength at the age it is asked about;"
            f" the constants handed in carry"
            f" eps_c1={en2class.eps_c1!r} and"
            f" fcm={en2class.fcm!r}, which fben2 reports as its"
            f" own out-of-domain marker (age"
            f" {en2class.time!r} d, below the"
            f" three-day floor).  Declare an age inside the"
            f" domain, or a law that covers it."
        )
    e = np.asarray(strain, dtype=float)
    k = 1.05*(en2class.ecm/1000)*abs(en2class.eps_c1) * \
        1000 / en2class.fcm
    inside = np.abs(e) <= abs(en2class.eps_cu1)
    eta = np.where(inside, e/en2class.eps_c1, 0.0)
    den = 1.0 + (k - 2.0)*eta
    num = k*eta - eta*eta
    d_num = (k - 2.0*eta)/en2class.eps_c1
    d_den = (k - 2.0)/en2class.eps_c1
    out = np.where(inside,
                   en2class.fcm*(d_num*den - num*d_den)/(den*den),
                   0.0)
    return out


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


# ===========================================================================
#  EN 1992-1-1 below three days: a declared scope boundary
# ===========================================================================
#
#  This block lives in the Eurocode module and nowhere else.  The three-day
#  limit is EN 1992-1-1 §3.1.2(5)'s own statement about its own expression,
#  so it belongs to the document that makes it, together with the class that
#  refuses on its behalf and the law that will one day replace the refusal.
#  Another document draws its own early-age boundary in its own place, and
#  this module has nothing to say about where.  Each code runs a closed,
#  watertight cycle: that is what normative agnosticism costs and what it
#  buys.  Nothing here reaches outside EN 1992-1-1, and nothing outside
#  reaches in.

#: Age [days] below which EN 1992-1-1 §3.1.2(5) declares its own strength
#: development expression insufficiently accurate.
EC2_EARLY_AGE_LIMIT_DAYS = 3.0


class EC2EarlyAgeConcrete:
    r"""
    Placeholder for the constitutive law of concrete below three days.

    Constructing this class always raises.  It is a class rather than a
    bare ``raise`` at the call site for three reasons, and the third is
    the one that matters:

    * the refusal names one address, so every consumer that hits the
      boundary reports the same thing;
    * the message can carry what is actually missing, rather than a
      symptom (``fcm=0``) that reads like a corrupted input;
    * when the law is written, its home already exists, and the future
      phase changes what this class *does* instead of hunting for the
      places that currently refuse.

    Parameters
    ----------
    t : float
        Age [days].
    fck : float, optional
        Characteristic strength [MPa], carried into the message when
        known so that the refusal identifies the material it refused
        for.
    name : str, optional

    Raises
    ------
    NotImplementedError
        Always, at construction.

    Notes
    -----
    What a future implementation needs, and why none of it is guessable
    from the tree as it stands:

    * a strength development valid below three days, expressed within
      EN 1992-1-1 or declared as an amendment to it.  What other
      documents write is their own business and is not admissible
      here;
    * a :math:`\varepsilon_{c1}(t)`.  EN 1992-1-1 Table 3.1 ties it to
      :math:`f_{cm}`, so it follows the strength once the strength
      exists;
    * a statement about whether the tension branch is still
      :math:`1.05\,E_{cm}` up to :math:`f_{ctm}` at an age where the
      concrete is barely set.  This is the one that cannot be inherited
      from the mature law by analogy, because early-age cracking is
      governed by restrained shrinkage rather than by load, and the
      relevant strength is not necessarily the one the mature law
      stops at.

    Examples
    --------
    >>> EC2EarlyAgeConcrete(t=2.0, fck=45.0)      # doctest: +SKIP
    Traceback (most recent call last):
        ...
    NotImplementedError: ...
    """

    def __init__(self, t, fck=None, name=None):
        which = f" for C{fck:g}" if fck is not None else ""
        label = f" ('{name}')" if name else ""
        raise NotImplementedError(
            f"no constitutive law is available for concrete at "
            f"t = {float(t):g} d{which}{label}: EN 1992-1-1 §3.1.2(5) "
            f"declares its strength development insufficiently accurate "
            f"below {EC2_EARLY_AGE_LIMIT_DAYS:g} days, and fben2 marks that "
            f"boundary by reporting f_cm = 0 and eps_c1 = 0.  A concrete "
            f"this young has a compressive strength; the Eurocode "
            f"declines to state it, which is a limit of the clause and "
            f"not of the material.  GenSec refuses rather than "
            f"extrapolate a normative formula outside the domain its own "
            f"document claims, and rather than carry on with a stiffness "
            f"and a tensile strength but no compressive strength -- a "
            f"state no concrete is in.  Move the instant to "
            f"t >= {EC2_EARLY_AGE_LIMIT_DAYS:g} d, or declare the zone's "
            f"service material explicitly."
        )
