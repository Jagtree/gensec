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
Service constitutive law for concrete: EN 1992-1-1 (3.14) with a
cracking tension branch.

The physics first, the clause afterwards
----------------------------------------
A concrete of a given age has one stiffness, one mean strength, one
strain at peak and one tensile strength.  Those four facts, and nothing
else, define how a fibre of it answers a strain.  This module writes
that answer down.

In **compression** the answer is Sargin's curve, EN 1992-1-1 expression
(3.14):

.. math::

    \sigma_c = f_{cm}\,
        \frac{k\eta - \eta^{2}}{1 + (k-2)\eta},
    \qquad
    \eta = \frac{\varepsilon_c}{\varepsilon_{c1}},
    \qquad
    k = \frac{1.05\,E_{cm}\,\lvert\varepsilon_{c1}\rvert}{f_{cm}} .

In **tension** the answer is a straight line up to the tensile strength
and nothing beyond it:

.. math::

    \sigma_c =
    \begin{cases}
      1.05\,E_{cm}\,\varepsilon_c ,
        & 0 \le \varepsilon_c \le \varepsilon_{ct} \\[4pt]
      0 , & \varepsilon_c > \varepsilon_{ct}
    \end{cases}
    \qquad
    \varepsilon_{ct} = \frac{f_{ct,\mathrm{eff}}}{1.05\,E_{cm}} .

Three consequences of that choice, each of which was measured before it
was written:

**The slope in tension is** :math:`1.05\,E_{cm}`, **not** :math:`E_{cm}`.
:math:`E_{cm}` is the *secant* modulus at :math:`0.4 f_{cm}`; the tangent
at the origin is 5 % steeper, and (3.14) produces exactly
:math:`1.05\,E_{cm}` there by construction of :math:`k`.  Joining a
tension branch at :math:`E_{cm}` would put a 5 % slope discontinuity at
:math:`\varepsilon_c = 0` — precisely where the strain plane of every
bent section crosses.  At :math:`1.05\,E_{cm}` the law is :math:`C^1` at
the origin, and on a lightly compressed section the choice was measured
to move the curvature by 1.46 percentage points, more than the
non-linearity of the compression branch itself.

**The tension branch cuts.**  Below :math:`\varepsilon_{ct}` a cut law
and an uncut one are the same law, to the last bit: the cut costs
nothing where it does not bite.  Above it, the cut is what makes the
section cracked *by arithmetic* rather than by a downstream verdict.  An
unbounded tension branch reports states that do not exist — a C35/45
carrying 6.46 MPa of tension — and, worse, was measured to converge on
demands beyond the section's strength, manufacturing equilibrium out of
concrete tension that no concrete has.

**What the cut does not give is tension stiffening**, and it cannot.
Tension stiffening is the concrete held in tension *between* cracks by
bond; it depends on the crack spacing and the bond stress, which are
facts about a length of member and not about a section.  At the crack
the concrete carries nothing, which is what this law returns.  The
member average — EN 1992-1-1 (7.18) — sits between this law and the
uncracked one, and was measured 30.2 % below this law just past
cracking and 16.0 % below it at four times the cracking moment.  That
gap is not a defect of the law: it is the boundary between a sectional
library and a member analysis.

Where the constants come from
-----------------------------
This class **derives nothing**.  It is handed :math:`f_{cm}`,
:math:`E_{cm}`, :math:`\varepsilon_{c1}`, :math:`\varepsilon_{cu1}` and
:math:`f_{ct,\mathrm{eff}}`, all at one declared age, and forms only the
curve through them.  The normative document that produces those numbers
is :mod:`gensec.materials.ec2_properties`, and it is the only place they
are allowed to be computed: two zones of nominally identical concrete
cast on different days are two different materials here, because
:math:`E_{cm}(t)`, :math:`f_{cm}(t)` and :math:`\varepsilon_{c1}(t)` all
age, and the class refuses to guess which day it is being asked about.

The compression expression itself lives in
:func:`gensec.materials.ec2_properties.en2sargin`.  What follows is a
vectorised transcription of the same expression — the fibre loop cannot
afford a Python call per fibre — and
``tests/test_phase26_service_law.py`` binds the two together to
:math:`10^{-12}` across the whole admissible range, so the transcription
cannot drift away from its source.

See Also
--------
gensec.materials.concrete.Concrete
    The **design** parabola-rectangle of EN 1992-1-1 §3.1.7.  It is not
    an alternative to this law and must never be used in its place: its
    slope at the origin is :math:`2 f_{cd}/\varepsilon_{c2}`, which
    carries :math:`\gamma_c` and :math:`\alpha_{cc}`, and was measured
    between 0.450 and 0.760 of :math:`E_{cm}` across the strength
    classes.  A stiffness that is a function of a partial factor is not
    a stiffness.
gensec.materials.elastic.LinearElastic
    A straight line at a declared modulus.  Still a legitimate material
    for whoever declares it; no longer something substituted for
    another material behind the model's back.
"""

from dataclasses import dataclass, field

import numpy as np

from .base import Material, ReferenceModulus

#: Ratio of the tangent modulus at the origin to the secant modulus
#: :math:`E_{cm}`.  Not a tuning constant: it is the value for which
#: EN 1992-1-1 (3.14) is written, and the tension branch takes it so
#: that the law is :math:`C^1` at zero strain.
TANGENT_OVER_SECANT = 1.05


@dataclass
class ServiceConcrete(Material):
    r"""
    EN 1992-1-1 (3.14) in compression, linear cracking branch in tension.

    Sign convention is GenSec's: compression is **negative**.  The
    normative expression is written with compression positive, so the
    stress is formed as :math:`\sigma(\varepsilon) = -S(-\varepsilon)`
    and the conversion happens once, here.

    Parameters
    ----------
    fcm : float
        Mean cylinder compressive strength at :attr:`reference_age`
        [MPa].  Strictly positive.
    Ecm : float
        Secant modulus of elasticity at :math:`0.4 f_{cm}`, at the same
        age [MPa].  Strictly positive.
    eps_c1 : float
        Strain magnitude at peak compressive stress, at the same age
        [-].  Strictly positive; the sign is supplied by the law, not
        by the caller.
    eps_cu1 : float
        Nominal ultimate strain magnitude in compression [-].  Strictly
        positive and not smaller than ``eps_c1``.
    fct_eff : float
        Effective tensile strength [MPa] at which the tension branch
        stops.  Strictly positive.  Which tensile strength this is —
        :math:`f_{ctm}`, :math:`f_{ctk,0.05}` or :math:`f_{ctm,fl}` —
        is a declaration with consequences and has no default here.
    reference_age : float
        Age [days] at which the four constants above were evaluated.
        Carried so that a material cannot be silently reused at an age
        it was not built for, and reported by
        :meth:`reference_modulus`.
    name : str, optional
        Human-readable identifier, e.g. ``"C35/45 service @28d"``.

    Attributes
    ----------
    k : float
        Shape parameter of (3.14), :math:`1.05 E_{cm}
        \lvert\varepsilon_{c1}\rvert / f_{cm}`.  Derived in
        :meth:`__post_init__` from the declared constants only.
    eps_ct : float
        Cracking strain, :math:`f_{ct,\mathrm{eff}} / (1.05 E_{cm})`.
        Equal to :attr:`eps_max`.

    Raises
    ------
    ValueError
        On any non-finite or non-positive constant, on
        ``eps_cu1 < eps_c1``, or on a non-positive ``reference_age``.
        Every one of these is an undeclared or impossible input rather
        than unmodelled physics, which is why the exception is
        ``ValueError`` and never ``NotImplementedError``.

    Examples
    --------
    >>> law = ServiceConcrete(fcm=43.0, Ecm=34077.146199189326,
    ...                       eps_c1=0.0022463284691964671,
    ...                       eps_cu1=0.0035, fct_eff=3.2099624416952379,
    ...                       reference_age=28.0, name="C35/45 @28d")
    >>> round(law.stress(-law.eps_c1), 9)
    -43.0
    >>> round(law.tangent(0.0) / law.Ecm, 6)
    1.05
    >>> round(law.eps_max, 9) == round(law.eps_ct, 9)
    True
    >>> law.stress(2.0 * law.eps_ct)
    0.0
    """

    fcm: float
    Ecm: float
    eps_c1: float
    eps_cu1: float
    fct_eff: float
    reference_age: float
    name: str = ""
    k: float = field(init=False, default=0.0)
    eps_ct: float = field(init=False, default=0.0)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def __post_init__(self):
        r"""
        Validate the declared constants and form the derived pair.

        Raises
        ------
        ValueError
            See the class docstring.
        """
        for label, value in (("fcm", self.fcm), ("Ecm", self.Ecm),
                             ("eps_c1", self.eps_c1),
                             ("eps_cu1", self.eps_cu1),
                             ("reference_age", self.reference_age)):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"ServiceConcrete requires a finite, strictly "
                    f"positive {label}; got {label}={value!r}.  Below "
                    f"three days fben2 reports fcm = 0 and eps_c1 = 0 "
                    f"as markers of its own domain: a curve through a "
                    f"zero strength is not a soft concrete, it is no "
                    f"concrete.  Declare an age inside the domain, or "
                    f"a law that covers it."
                )
        # fct_eff is the one constant of this law that may legitimately
        # be zero.  The other five are properties of the concrete, and a
        # zero there means they were never evaluated -- below three days
        # fben2 returns fcm = 0 as its own out-of-domain marker.
        # fct_eff is not a property but a declaration: the strength at
        # which the tension branch stops, and a model is entitled to say
        # that it stops immediately.  A crack-width check reads the
        # section *at* the crack, where the concrete carries nothing.
        # Zero width is a law; a negative width is not.
        if not np.isfinite(self.fct_eff) or self.fct_eff < 0.0:
            raise ValueError(
                f"ServiceConcrete requires a finite, non-negative "
                f"fct_eff; got fct_eff={self.fct_eff!r}.  Zero is "
                f"admissible and declares a law with no tension "
                f"branch; a negative tensile strength declares "
                f"nothing."
            )
        if self.eps_cu1 < self.eps_c1:
            raise ValueError(
                f"ServiceConcrete requires eps_cu1 >= eps_c1; got "
                f"eps_cu1={self.eps_cu1!r} and eps_c1={self.eps_c1!r}.  "
                f"The peak would sit outside the admissible range and "
                f"the law would never reach its own strength."
            )
        object.__setattr__(
            self, "k",
            float(TANGENT_OVER_SECANT * self.Ecm * abs(self.eps_c1)
                  / self.fcm))
        object.__setattr__(
            self, "eps_ct",
            float(self.fct_eff / (TANGENT_OVER_SECANT * self.Ecm)))

    # ------------------------------------------------------------------
    # admissible range
    # ------------------------------------------------------------------
    @property
    def initial_tangent(self):
        r"""
        Tangent modulus at zero strain, :math:`1.05\,E_{cm}` [MPa].

        Named because it is the quantity two branches must agree on:
        it is the slope (3.14) produces at the origin by construction
        of :math:`k`, and the slope the tension branch is given, so
        that the law has no corner at the strain every bent section
        contains.

        Returns
        -------
        float
        """
        return float(TANGENT_OVER_SECANT * self.Ecm)

    @property
    def eps_min(self):
        r"""
        Most compressive admissible strain, :math:`-\varepsilon_{cu1}`.

        Returns
        -------
        float
        """
        return -abs(self.eps_cu1)

    @property
    def eps_max(self):
        r"""
        Most tensile admissible strain, the cracking strain
        :math:`\varepsilon_{ct}`.

        This is a real limit and not a scan hint: beyond it the fibre
        carries nothing, so there is no state of the material further
        out to report.

        Returns
        -------
        float
        """
        return float(self.eps_ct)

    # ------------------------------------------------------------------
    # constitutive law
    # ------------------------------------------------------------------
    def stress(self, eps):
        r"""
        Stress at a single strain [MPa], compression negative.

        Parameters
        ----------
        eps : float

        Returns
        -------
        float
        """
        return float(self.stress_array(np.asarray([eps], dtype=float))[0])

    def stress_array(self, eps):
        r"""
        Stress at an array of strains [MPa], compression negative.

        Vectorised transcription of EN 1992-1-1 (3.14) in compression
        and of the linear cracking branch in tension.  The compression
        half is bound to
        :func:`~gensec.materials.ec2_properties.en2sargin` by
        ``tests/test_phase26_service_law.py`` to :math:`10^{-12}`.

        Parameters
        ----------
        eps : array_like

        Returns
        -------
        numpy.ndarray
        """
        e = np.asarray(eps, dtype=float)
        out = np.zeros(e.shape, dtype=float)

        comp = (e < 0.0) & (np.abs(e) <= abs(self.eps_cu1))
        if np.any(comp):
            eta = -e[comp] / self.eps_c1
            den = 1.0 + (self.k - 2.0) * eta
            out[comp] = -self.fcm * (self.k * eta - eta * eta) / den

        tens = (e > 0.0) & (e <= self.eps_ct)
        if np.any(tens):
            out[tens] = TANGENT_OVER_SECANT * self.Ecm * e[tens]

        return out

    def tangent(self, eps):
        r"""
        Tangent modulus at a single strain [MPa].

        Parameters
        ----------
        eps : float

        Returns
        -------
        float
        """
        return float(self.tangent_array(np.asarray([eps], dtype=float))[0])

    def tangent_array(self, eps):
        r"""
        Tangent modulus at an array of strains [MPa].

        Closed form, differentiating the expressions of
        :meth:`stress_array`.  In compression, with
        :math:`\eta = -\varepsilon/\varepsilon_{c1}` and
        :math:`D = 1 + (k-2)\eta`,

        .. math::

            \frac{d\sigma}{d\varepsilon}
              = \frac{f_{cm}}{\varepsilon_{c1}}\,
                \frac{(k - 2\eta)\,D - (k\eta - \eta^{2})(k-2)}{D^{2}} .

        At :math:`\eta \to 0` this returns :math:`1.05 E_{cm}`, which is
        also the slope of the tension branch: the law is :math:`C^1` at
        the origin, and the zero strain that every bent section contains
        is not a corner.

        Parameters
        ----------
        eps : array_like

        Returns
        -------
        numpy.ndarray
        """
        e = np.asarray(eps, dtype=float)
        out = np.zeros(e.shape, dtype=float)

        comp = (e <= 0.0) & (np.abs(e) <= abs(self.eps_cu1))
        if np.any(comp):
            eta = -e[comp] / self.eps_c1
            den = 1.0 + (self.k - 2.0) * eta
            num = self.k * eta - eta * eta
            out[comp] = (self.fcm / self.eps_c1
                         * ((self.k - 2.0 * eta) * den - num * (self.k - 2.0))
                         / (den * den))

        tens = (e > 0.0) & (e <= self.eps_ct)
        if np.any(tens):
            out[tens] = TANGENT_OVER_SECANT * self.Ecm

        return out

    def stress_breakpoints(self):
        r"""
        The three strains where this law turns over or drops.

        .. math::

            \{-\varepsilon_{cu1},\; -\varepsilon_{c1},\;
              +\varepsilon_{ct}\}

        * :math:`-\varepsilon_{c1}` is the peak of (3.14): the most
          compressive stress the law can return, and beyond it the
          expression softens, so the most compressed *vertex* of a
          region is no longer the most compressed *point* of it.
        * :math:`+\varepsilon_{ct}` is the cracking strain: the tension
          branch attains :math:`f_{ct,\mathrm{eff}}` there and returns
          zero immediately beyond, so a region straddling the crack
          front carries its maximum tension on that front and not at
          any vertex.
        * :math:`-\varepsilon_{cu1}` is where the compression branch
          stops being defined and the law returns zero, which is a jump
          for the same reason.

        Returns
        -------
        tuple of float
        """
        return (-abs(self.eps_cu1), -abs(self.eps_c1),
                float(self.eps_ct))

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def reference_modulus(self):
        r"""
        The declared secant modulus and the age it belongs to.

        The service law reports :math:`E_{cm}` and not its own initial
        tangent :math:`1.05 E_{cm}`: consumers that ask a material for
        "the" modulus are building homogenisation ratios and creep
        bases, and EN 1992-1-1 writes those on the secant.

        Returns
        -------
        ReferenceModulus
        """
        return ReferenceModulus(
            value=float(self.Ecm),
            reference_age=float(self.reference_age),
            source="declared on the EN 1992-1-1 (3.14) service law")
