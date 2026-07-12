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
Rheological providers — the normative specialisations of
:class:`~gensec.materials.base.RheologicalModel`.

This module is the **only** place where a code's creep, shrinkage and
relaxation formulae are written down.  The container
(:mod:`gensec.solver.losses`) consumes the four abstract functions and
never imports from here.

Three providers ship:

=============================  ==============================================
:class:`EC2RheologicalModel`   EN 1992-1-1 Annex B + §3.1.4 + §3.3.2 (also
                               NTC 2018, which adopts the same formulae).
                               The **default**.
:class:`ACIRheologicalModel`   ACI 209R-92.  A *structurally different* code
                               — hyperbolic time functions, volume/surface
                               ratio in inches, :math:`E_c=4700\sqrt{f'_c}`,
                               a compliance referenced to the modulus at
                               loading rather than at 28 days.  It exists to
                               **falsify** the claim of normative agnosticism.
:class:`TabulatedRheologicalModel`
                               User compliance / shrinkage / relaxation
                               tables.  A code that is *only data* passes
                               through the same door.
=============================  ==============================================

Geometry binding
----------------
Providers are constructed with their **material** parameters only.  The
drying geometry :math:`(A_c, u)` is bound **late**, by the container,
with :meth:`~RheologicalModel.with_geometry` — because in a composite
section every concrete zone has its own exposed perimeter, and hence its
own notional size.  A provider whose geometry is unbound raises when a
geometry-dependent function is called: fail loud, never a silent default
size.
"""

from __future__ import annotations

import copy
import math

import numpy as np

from .base import RheologicalModel

__all__ = [
    "EC2RheologicalModel",
    "ACIRheologicalModel",
    "TabulatedRheologicalModel",
]


# ==================================================================
#  Shared geometry binding
# ==================================================================

# ==================================================================
#  EN 1992-1-1 / NTC 2018
# ==================================================================

class EC2RheologicalModel(RheologicalModel):
    r"""
    EN 1992-1-1 (and NTC 2018) rheological provider.

    Creep
    -----
    Annex B.1.  The creep coefficient

    .. math::

        \varphi(t,t_0) = \varphi_0 \, \beta_c(t,t_0),
        \qquad
        \varphi_0 = \varphi_{RH}\, \beta(f_{cm})\, \beta(t_0)

    with (B.3a/b, B.4, B.5, B.7, B.8a-c)

    .. math::

        \varphi_{RH} =
        \begin{cases}
        1 + \dfrac{1 - RH/100}{0.1\,\sqrt[3]{h_0}}
            & f_{cm} \le 35\ \mathrm{MPa} \\[8pt]
        \left[1 + \dfrac{1 - RH/100}{0.1\,\sqrt[3]{h_0}}\,\alpha_1\right]
        \alpha_2
            & f_{cm} > 35\ \mathrm{MPa}
        \end{cases}

    .. math::

        \beta(f_{cm}) = \frac{16.8}{\sqrt{f_{cm}}},
        \qquad
        \beta(t_0) = \frac{1}{0.1 + t_0^{0.20}},
        \qquad
        \beta_c(t,t_0) = \left[
            \frac{t-t_0}{\beta_H + t - t_0}\right]^{0.3}

    .. math::

        \beta_H = 1.5\bigl[1 + (0.012\,RH)^{18}\bigr] h_0 + 250\,\alpha_3
        \;\le\; 1500\,\alpha_3 ,
        \qquad
        \alpha_i = \left(\frac{35}{f_{cm}}\right)^{\{0.7,\,0.2,\,0.5\}}

    (:math:`\alpha_3 = 1` when :math:`f_{cm} \le 35`), the notional size
    :math:`h_0 = 2A_c/u` [mm], and the cement-class correction of the
    loading age (B.9)

    .. math::

        t_0 = t_{0,T}\left[\frac{9}{2 + t_{0,T}^{1.2}} + 1\right]^{\alpha}
        \;\ge\; 0.5 ,
        \qquad \alpha = \{-1, 0, 1\}\ \text{for class}\ \{S, N, R\} ,

    applied to :math:`\beta(t_0)` only (the elapsed time
    :math:`t - t_0` uses the *physical* age).

    Compliance
    ----------
    .. math::

        J(t,t') = \frac{1}{E_{cm}(t')}
                  + \frac{\varphi(t,t')}{E_{cm}(28)}

    — the convention of §5.10.6 and §7.4.3, where the creep strain is
    referenced to the **28-day** modulus while the instantaneous strain
    uses the modulus *at loading*.  Setting ``creep_modulus='at_t0'``
    selects instead the collapsed effective-modulus form
    :math:`J = [1+\varphi]/E_{cm}(t')`.  The two coincide at
    :math:`t' = 28` d.

    Modulus
    -------
    §3.1.3(3):
    :math:`E_{cm}(t) = \bigl[\beta_{cc}(t)\bigr]^{0.3} E_{cm}(28)` with
    :math:`\beta_{cc}(t) = \exp\!\bigl[s\,(1 - \sqrt{28/t})\bigr]` and
    :math:`s = \{0.20, 0.25, 0.38\}` for cement class
    :math:`\{R, N, S\}`.

    .. warning::

        This is computed **here**, not read from
        :attr:`gensec.materials.ec2_properties.fben2.ecm`.  That
        attribute multiplies the modulus by
        :math:`(\beta_{cc}^{\,\alpha})^{0.3}` with the *tensile-strength*
        exponent :math:`\alpha` — giving :math:`\beta_{cc}^{0.5}` instead
        of :math:`\beta_{cc}^{0.3}` for :math:`t < 28` d (Phase-5 finding
        F2; inert at :math:`t = 28`, where :math:`\beta_{cc} = 1`).
        ``fben2`` is still the source of :math:`f_{cm}`, :math:`f_{ck}`
        and :math:`E_{cm}(28)`.

    Shrinkage
    ---------
    §3.1.4(6): :math:`\varepsilon_{cs} = \varepsilon_{cd} +
    \varepsilon_{ca}`, drying (3.9-3.10, B.11-B.12, Table 3.3) plus
    autogenous (3.11-3.13).  Returned **signed** — negative, a
    shortening.

    Relaxation
    ----------
    §3.3.2, expressions (3.28)-(3.30), with *t* in **hours** internally
    and :math:`\mu = \sigma_{pi}/f_{pk}`:

    .. math::

        \frac{\Delta\sigma_{pr}}{\sigma_{pi}} =
        k_1 \,\rho_{1000}\, e^{k_2 \mu}
        \left(\frac{t}{1000}\right)^{0.75(1-\mu)} 10^{-5}

    ==========  =======  =======  ====================
    class       k1       k2       :math:`\rho_{1000}`
    ==========  =======  =======  ====================
    1           5.39     6.7      8.0 %
    2           0.66     9.1      2.5 %
    3           1.98     8.0      4.0 %
    ==========  =======  =======  ====================

    Returned **signed** (negative) and **intrinsic**: the reduction due
    to the tendon shortening with the concrete (EC2's 0.8) belongs to
    the container, not here.  Long-term values are capped at
    :math:`t = 500\,000` h per §3.3.2(8) — the Eurocode does not
    extrapolate beyond it.

    Parameters
    ----------
    fck : float
        Characteristic cylinder strength at 28 d [MPa].
    cement_class : {'R', 'N', 'S'}, optional
        EN 197 cement class.  Default ``'N'``.
    RH : float, optional
        Ambient relative humidity [%].  Default 70.
    A_c, u : float, optional
        Drying geometry [mm², mm].  Usually bound later by the
        container (:meth:`~_GeometryBound.with_geometry`).
    relaxation_class : {1, 2, 3}, optional
        EN 1992-1-1 §3.3.2 class.  Default 2 (low relaxation).
    rho_1000 : float, optional
        Relaxation loss at 1000 h and 20 °C, at :math:`\mu = 0.7` [%].
        Default: the class value (8.0 / 2.5 / 4.0).
    creep_modulus : {'E28', 'at_t0'}, optional
        Reference modulus of the creep term of :meth:`J`.  Default
        ``'E28'`` (the §5.10.6 convention).
    name : str, optional
        Identifier.

    Notes
    -----
    Temperature adjustment of the concrete age (Annex B.10) is **not**
    applied: an isothermal history at 20 °C is assumed.  Pass an
    already-equivalent age if the history is not isothermal.

    Examples
    --------
    >>> m = EC2RheologicalModel(fck=35, cement_class='N', RH=70)
    >>> m = m.with_geometry(A_c=600 * 1400, u=2 * (600 + 1400))
    >>> round(m.phi(25550.0, 28.0), 3)              # doctest: +SKIP
    1.717
    """

    #: (k1, k2, rho_1000 [%]) per EN 1992-1-1 §3.3.2 relaxation class.
    _RELAX = {1: (5.39, 6.7, 8.0),
              2: (0.66, 9.1, 2.5),
              3: (1.98, 8.0, 4.0)}
    #: Cement-class coefficient *s* (§3.1.2(6), eq. 3.2).
    _S_CEM = {"R": 0.20, "N": 0.25, "S": 0.38}
    #: Cement-class exponent alpha of the loading-age correction (B.9).
    _ALPHA_CEM = {"R": 1.0, "N": 0.0, "S": -1.0}
    #: Shrinkage cement coefficients (alpha_ds1, alpha_ds2), eq. (B.11).
    _ALPHA_DS = {"R": (6.0, 0.11), "N": (4.0, 0.12), "S": (3.0, 0.13)}
    #: EN 1992-1-1 §3.3.2(8): the long-term relaxation reference time [h].
    T_RELAX_MAX_HOURS = 500000.0

    def __init__(self, fck, cement_class="N", RH=70.0, A_c=None, u=None,
                 relaxation_class=2, rho_1000=None, creep_modulus="E28",
                 name=""):
        cc = str(cement_class).upper()
        if cc not in self._S_CEM:
            raise ValueError(
                f"EC2RheologicalModel: cement_class must be one of "
                f"{sorted(self._S_CEM)}, got {cement_class!r}."
            )
        rc = int(relaxation_class)
        if rc not in self._RELAX:
            raise ValueError(
                f"EC2RheologicalModel: relaxation_class must be 1, 2 or "
                f"3 (EN 1992-1-1 §3.3.2), got {relaxation_class!r}."
            )
        if creep_modulus not in ("E28", "at_t0"):
            raise ValueError(
                f"EC2RheologicalModel: creep_modulus must be 'E28' (the "
                f"§5.10.6 convention, default) or 'at_t0' (the collapsed "
                f"effective-modulus form), got {creep_modulus!r}."
            )
        if not 0.0 < float(RH) <= 100.0:
            raise ValueError(
                f"EC2RheologicalModel: RH must be in (0, 100] %, got {RH}."
            )
        self.fck = float(fck)
        self.cement_class = cc
        self.RH = float(RH)
        self.relaxation_class = rc
        self.rho_1000 = (float(rho_1000) if rho_1000 is not None
                         else self._RELAX[rc][2])
        self.creep_modulus = creep_modulus
        self.name = name or f"ec2(fck={self.fck:g},{cc},RH={self.RH:g})"

        # Table 3.1 (28 d).  fcm = fck + 8; Ecm = 22000 (fcm/10)^0.3.
        self.fcm = self.fck + 8.0
        self.Ecm28 = 22000.0 * (self.fcm / 10.0) ** 0.3
        self.s_cem = self._S_CEM[cc]

        if A_c is not None and u is not None:
            bound = self.with_geometry(A_c, u)
            self.A_c = bound.A_c
            self.u = bound.u

    # -- geometry ---------------------------------------------------

    @property
    def h0(self):
        r"""Notional size :math:`h_0 = 2 A_c / u` [mm] (eq. B.6)."""
        self._require_geometry("h0")
        return 2.0 * self.A_c / self.u

    # -- ageing of the strength / modulus ---------------------------

    def beta_cc(self, t):
        r"""
        Strength-development function
        :math:`\beta_{cc}(t) = \exp[s(1 - \sqrt{28/t})]` (eq. 3.2).

        Parameters
        ----------
        t : float
            Age [days], ``> 0``.

        Returns
        -------
        float
        """
        t = float(t)
        if t <= 0.0:
            raise ValueError(
                f"EC2RheologicalModel.beta_cc: age t must be > 0, got {t}."
            )
        return math.exp(self.s_cem * (1.0 - math.sqrt(28.0 / t)))

    def E_c(self, t):
        r"""
        :math:`E_{cm}(t) = [\beta_{cc}(t)]^{0.3} E_{cm}(28)` [MPa]
        (§3.1.3(3)).  See the class-level warning on finding F2.
        """
        return self.Ecm28 * self.beta_cc(t) ** 0.3

    def linearity_limit(self, t):
        r"""
        :math:`0.45\, f_{ck}(t)` [MPa] — EN 1992-1-1 §3.1.4(4).  Above
        it the creep is **non-linear** and the stress-independent
        compliance of this provider no longer applies.
        """
        fck_t = self.fcm * self.beta_cc(t) - 8.0
        return 0.45 * max(fck_t, 0.0)

    # -- creep (Annex B.1) ------------------------------------------

    def _t0_adjusted(self, t0):
        r"""Cement-class-corrected loading age (eq. B.9) [days]."""
        a = self._ALPHA_CEM[self.cement_class]
        t0 = float(t0)
        return max(0.5, t0 * (9.0 / (2.0 + t0 ** 1.2) + 1.0) ** a)

    def phi_ec2(self, t, t0):
        r"""
        EN 1992-1-1 Annex B creep coefficient :math:`\varphi(t,t_0)` [-].

        This is the *Eurocode's own* coefficient, referenced to
        :math:`E_{cm}(28)`.  It is **not** the container's generalized
        :meth:`~gensec.materials.base.RheologicalModel.phi`, which is
        derived from :meth:`J` and coincides with this one only when
        ``creep_modulus='at_t0'`` or :math:`t_0 = 28` d.

        Parameters
        ----------
        t : float
            Age at observation [days].
        t0 : float
            Age at loading [days].

        Returns
        -------
        float
        """
        self._require_geometry("phi_ec2")
        t = float(t)
        t0 = float(t0)
        if t <= t0:
            return 0.0
        h0 = self.h0
        fcm = self.fcm
        rh = self.RH

        # phi_RH  (B.3a / B.3b)
        base = (1.0 - rh / 100.0) / (0.1 * h0 ** (1.0 / 3.0))
        if fcm <= 35.0:
            phi_rh = 1.0 + base
            a3 = 1.0
        else:
            a1 = (35.0 / fcm) ** 0.7
            a2 = (35.0 / fcm) ** 0.2
            a3 = (35.0 / fcm) ** 0.5
            phi_rh = (1.0 + base * a1) * a2

        beta_fcm = 16.8 / math.sqrt(fcm)                       # (B.4)
        beta_t0 = 1.0 / (0.1 + self._t0_adjusted(t0) ** 0.20)  # (B.5)
        phi_0 = phi_rh * beta_fcm * beta_t0                    # (B.2)

        # beta_H  (B.8a / B.8b), capped
        beta_H = min(1.5 * (1.0 + (0.012 * rh) ** 18) * h0 + 250.0 * a3,
                     1500.0 * a3)
        dt = t - t0
        beta_c = (dt / (beta_H + dt)) ** 0.3                   # (B.7)
        return phi_0 * beta_c                                  # (B.1)

    def J(self, t, t_prime):
        r"""
        Creep compliance [1/MPa].  See the class docstring for the two
        conventions selected by ``creep_modulus``.
        """
        t = float(t)
        tp = float(t_prime)
        if t < tp:
            raise ValueError(
                f"EC2RheologicalModel.J: t ({t}) must be >= t' ({tp})."
            )
        phi = self.phi_ec2(t, tp)
        if self.creep_modulus == "at_t0":
            return (1.0 + phi) / self.E_c(tp)
        return 1.0 / self.E_c(tp) + phi / self.Ecm28

    # -- shrinkage (§3.1.4(6) + Annex B.2) --------------------------

    @staticmethod
    def _k_h(h0):
        r"""Coefficient :math:`k_h` from Table 3.3, linearly
        interpolated between the tabulated notional sizes."""
        return float(np.interp(h0, [100.0, 200.0, 300.0, 500.0],
                               [1.00, 0.85, 0.75, 0.70]))

    def eps_cd(self, t, t_s):
        r"""
        Drying shrinkage :math:`\varepsilon_{cd}(t)` [-], **positive
        magnitude** (eq. 3.9-3.10, B.11-B.12).
        """
        self._require_geometry("eps_cd")
        t = float(t)
        t_s = float(t_s)
        if t <= t_s:
            return 0.0
        h0 = self.h0
        a_ds1, a_ds2 = self._ALPHA_DS[self.cement_class]
        beta_rh = 1.55 * (1.0 - (self.RH / 100.0) ** 3)              # (B.12)
        eps_cd0 = (0.85 * (220.0 + 110.0 * a_ds1)
                   * math.exp(-a_ds2 * self.fcm / 10.0)
                   * 1e-6 * beta_rh)                                 # (B.11)
        dt = t - t_s
        beta_ds = dt / (dt + 0.04 * math.sqrt(h0 ** 3))              # (3.10)
        return beta_ds * self._k_h(h0) * eps_cd0                     # (3.9)

    def eps_ca(self, t):
        r"""
        Autogenous shrinkage :math:`\varepsilon_{ca}(t)` [-],
        **positive magnitude** (eq. 3.11-3.13).
        """
        t = float(t)
        if t <= 0.0:
            return 0.0
        eps_ca_inf = 2.5 * (self.fck - 10.0) * 1e-6                  # (3.12)
        beta_as = 1.0 - math.exp(-0.2 * math.sqrt(t))                # (3.13)
        return max(0.0, beta_as * eps_ca_inf)                        # (3.11)

    def eps_imposed(self, t, t_s):
        r"""
        Total shrinkage :math:`\varepsilon_{cs} = \varepsilon_{cd} +
        \varepsilon_{ca}`, returned **signed** — negative (a
        shortening), per the GenSec convention.
        """
        return -(self.eps_cd(t, t_s) + self.eps_ca(t))

    # -- relaxation (§3.3.2) ----------------------------------------

    def relaxation(self, t, mu):
        r"""
        Intrinsic relaxation :math:`\Delta\sigma_{pr}` [MPa], **signed**
        (negative), per :math:`\sigma_{pi}` — see the class docstring
        for (3.28)-(3.30).

        Parameters
        ----------
        t : float
            Duration under load [days] (converted to hours internally
            and capped at 500 000 h, §3.3.2(8)).
        mu : float
            :math:`\sigma_{pi}/f_{pk}` [-].

        Returns
        -------
        float
            Stress decay **per unit initial stress** is *not* what is
            returned: the value is the decay [MPa] for the initial
            stress implied by ``mu`` and the tendon's ``f_pk``, which
            the container supplies through :attr:`f_pk`.

        Raises
        ------
        ValueError
            If ``f_pk`` has not been bound (see :meth:`with_steel`).
        """
        self._require_steel("relaxation")
        f_pk = self.f_pk
        t = float(t)
        mu = float(mu)
        if t <= 0.0 or mu <= 0.0:
            return 0.0
        t_h = min(24.0 * t, self.T_RELAX_MAX_HOURS)
        k1, k2, _ = self._RELAX[self.relaxation_class]
        ratio = (k1 * self.rho_1000 * math.exp(k2 * mu)
                 * (t_h / 1000.0) ** (0.75 * (1.0 - mu)) * 1e-5)
        return -ratio * mu * float(f_pk)

    def with_steel(self, f_pk, relaxation_class=None, rho_1000=None):
        r"""
        Return a copy bound to a prestressing steel.

        Parameters
        ----------
        f_pk : float
            Characteristic tensile strength of the tendon [MPa].
        relaxation_class : int, optional
            Override the provider's class.
        rho_1000 : float, optional
            Override :math:`\rho_{1000}` [%].

        Returns
        -------
        EC2RheologicalModel
        """
        out = copy.copy(self)
        out.f_pk = float(f_pk)
        if relaxation_class is not None:
            rc = int(relaxation_class)
            if rc not in self._RELAX:
                raise ValueError(
                    f"EC2RheologicalModel.with_steel: relaxation_class "
                    f"must be 1, 2 or 3, got {relaxation_class!r}."
                )
            out.relaxation_class = rc
            if rho_1000 is None:
                out.rho_1000 = self._RELAX[rc][2]
        if rho_1000 is not None:
            out.rho_1000 = float(rho_1000)
        return out


# ==================================================================
#  ACI 209R-92 — the agnosticism falsification
# ==================================================================

class ACIRheologicalModel(RheologicalModel):
    r"""
    ACI 209R-92 rheological provider — *the falsification test*.

    Structurally unlike EN 1992-1-1 at every point, yet it satisfies the
    same four-function interface:

    ================  =========================  =========================
    concept           EN 1992-1-1                ACI 209R-92
    ================  =========================  =========================
    time function     :math:`[\Delta t/(\beta_H+\Delta t)]^{0.3}`
                                                 :math:`\Delta t^{0.6}/
                                                 (10+\Delta t^{0.6})`
    geometry          :math:`h_0 = 2A_c/u` [mm]  :math:`V/S = A_c/u` [in]
    modulus           :math:`22000(f_{cm}/10)^{0.3}`
                                                 :math:`4700\sqrt{f'_c}`
    compliance        creep on :math:`E_{cm}(28)`
                                                 creep on :math:`E_c(t')`
    relaxation        power law on :math:`f_{pk}`
                                                 log law on :math:`f_{py}`
    ================  =========================  =========================

    Creep (§2.2)
    ------------
    .. math::

        \varphi(t,t_0) = \frac{(t-t_0)^{0.6}}{10 + (t-t_0)^{0.6}}\,
                         \varphi_u ,
        \qquad
        \varphi_u = 2.35\,\gamma_{la}\,\gamma_{\lambda}\,\gamma_{vs}\,
                    \gamma_{\mathrm{other}}

    with the loading-age, humidity and volume/surface corrections

    .. math::

        \gamma_{la} = 1.25\, t_0^{-0.118}\ \text{(moist)},\quad
        1.13\, t_0^{-0.094}\ \text{(steam)} ,

    .. math::

        \gamma_{\lambda} = 1.27 - 0.0067\,\lambda \quad (\lambda > 40) ,
        \qquad
        \gamma_{vs} = \tfrac{2}{3}
            \Bigl[1 + 1.13\,e^{-0.54\,(V/S)}\Bigr] ,

    :math:`V/S` **in inches** — the unit conversion is done here, inside
    the provider, exactly as the interface contract requires.

    Compliance (§2.2, definition of :math:`\varphi`)
    ------------------------------------------------
    ACI's creep coefficient multiplies the *initial* strain at loading,
    hence

    .. math::

        J(t,t') = \frac{1 + \varphi(t,t')}{E_c(t')} ,

    referenced to the modulus **at loading** — not to a 28-day modulus
    as in EC2.  This is the structural difference the container must
    absorb without noticing.

    Shrinkage (§2.3)
    ----------------
    .. math::

        \varepsilon_{sh}(t,t_s) =
        \frac{t - t_s}{f + (t - t_s)}\,\varepsilon_{shu} ,
        \qquad
        \varepsilon_{shu} = 780 \times 10^{-6}\,
                            \gamma_{\lambda,sh}\,\gamma_{vs,sh} ,

    :math:`f = 35` d (moist-cured) or 55 d (steam-cured), with

    .. math::

        \gamma_{\lambda,sh} =
        \begin{cases}
            1.40 - 0.0102\,\lambda & 40 \le \lambda \le 80 \\
            3.00 - 0.030\,\lambda  & 80 < \lambda \le 100
        \end{cases}
        \qquad
        \gamma_{vs,sh} = 1.2\, e^{-0.12\,(V/S)} .

    Returned **signed** (negative).

    Modulus (§2.1)
    --------------
    .. math::

        E_c(t) = 4700 \sqrt{f'_c(t)}\ [\mathrm{MPa}],
        \qquad
        f'_c(t) = \frac{t}{a + b\,t}\, f'_c(28) ,

    :math:`(a,b) = (4.0, 0.85)` moist-cured, :math:`(1.0, 0.95)`
    steam-cured, Type-I cement.

    Relaxation
    ----------
    The Magura–Sozen–Siess logarithmic law (the basis of the AASHTO /
    ACI treatment of strand relaxation):

    .. math::

        \Delta\sigma_{pr}(t) = -\,\sigma_{pi}\,
        \frac{\log_{10}(24\,t)}{C}
        \left(\frac{\sigma_{pi}}{f_{py}} - 0.55\right) ,
        \qquad
        \frac{\sigma_{pi}}{f_{py}} > 0.55 ,

    with :math:`C = 45` for low-relaxation and :math:`C = 10` for
    stress-relieved strand, and :math:`f_{py} = k_{py} f_{pk}`
    (:math:`k_{py} \approx 0.90` for low-relaxation strand).  Zero below
    the 0.55 threshold.

    Parameters
    ----------
    fc_28 : float
        Specified cylinder strength :math:`f'_c` at 28 d [MPa].
    RH : float, optional
        Relative humidity [%].  Default 70.
    curing : {'moist', 'steam'}, optional
        Default ``'moist'``.
    A_c, u : float, optional
        Drying geometry [mm², mm]; bound late by the container.
    gamma_other_creep, gamma_other_shrinkage : float, optional
        Product of the correction factors this provider does not model
        explicitly (slump, fine-aggregate content, air content, cement
        content).  Default 1.0 each — *standard conditions*.
    low_relaxation : bool, optional
        Default ``True`` (:math:`C = 45`).
    f_py_ratio : float, optional
        :math:`f_{py}/f_{pk}`.  Default 0.90.
    name : str, optional

    Warnings
    --------
    The ACI constants above were transcribed for the express purpose of
    **falsifying** the container's normative agnosticism, and the
    provider is validated against a single hand-computed point
    (``run_phase5_c5_validation.py``, assembly B).  Before using it for
    a real ACI design, check the coefficients against a copy of
    ACI 209R-92 and the strand-relaxation law against the governing
    AASHTO/ACI edition.  Its purpose here is structural, not normative.

    Notes
    -----
    ACI 209R-92's standard conditions are 40 % ≤ RH; below 40 % the
    humidity corrections are outside the fitted range and the provider
    raises rather than extrapolate.
    """

    def __init__(self, fc_28, RH=70.0, curing="moist", A_c=None, u=None,
                 gamma_other_creep=1.0, gamma_other_shrinkage=1.0,
                 low_relaxation=True, f_py_ratio=0.90, name=""):
        cur = str(curing).lower()
        if cur not in ("moist", "steam"):
            raise ValueError(
                f"ACIRheologicalModel: curing must be 'moist' or 'steam', "
                f"got {curing!r}."
            )
        if not 40.0 <= float(RH) <= 100.0:
            raise ValueError(
                f"ACIRheologicalModel: RH must be in [40, 100] % — the "
                f"ACI 209R-92 humidity corrections are fitted only above "
                f"40 % and this provider does not extrapolate.  Got {RH}."
            )
        self.fc_28 = float(fc_28)
        self.RH = float(RH)
        self.curing = cur
        self.gamma_other_creep = float(gamma_other_creep)
        self.gamma_other_shrinkage = float(gamma_other_shrinkage)
        self.low_relaxation = bool(low_relaxation)
        self.f_py_ratio = float(f_py_ratio)
        self.name = name or f"aci209(fc={self.fc_28:g},RH={self.RH:g})"
        if A_c is not None and u is not None:
            bound = self.with_geometry(A_c, u)
            self.A_c = bound.A_c
            self.u = bound.u

    # -- geometry ---------------------------------------------------

    @property
    def v_over_s_in(self):
        r"""Volume/surface ratio :math:`V/S` **in inches** (ACI's own
        measure; the mm → in conversion is internal, as the interface
        contract demands)."""
        self._require_geometry("v_over_s_in")
        return (self.A_c / self.u) / 25.4

    # -- modulus ----------------------------------------------------

    def fc(self, t):
        r"""Strength development :math:`f'_c(t) = t/(a+bt)\, f'_c(28)`
        [MPa] (§2.1)."""
        t = float(t)
        if t <= 0.0:
            raise ValueError(
                f"ACIRheologicalModel.fc: age t must be > 0, got {t}."
            )
        a, b = (4.0, 0.85) if self.curing == "moist" else (1.0, 0.95)
        return t / (a + b * t) * self.fc_28

    def E_c(self, t):
        r""":math:`E_c(t) = 4700\sqrt{f'_c(t)}` [MPa] (§2.1)."""
        return 4700.0 * math.sqrt(self.fc(t))

    def linearity_limit(self, t):
        r"""
        :math:`0.40\, f'_c(t)` [MPa] — the upper bound of ACI 209R-92's
        linear-creep assumption (§2.2).  Deliberately *not* EC2's 0.45:
        each code owns its own range, and the container asks rather than
        assumes.
        """
        return 0.40 * self.fc(t)

    # -- creep ------------------------------------------------------

    def phi_aci(self, t, t0):
        r"""ACI 209R-92 creep coefficient :math:`\varphi(t,t_0)` [-]."""
        self._require_geometry("phi_aci")
        t = float(t)
        t0 = float(t0)
        if t <= t0:
            return 0.0
        if self.curing == "moist":
            g_la = 1.25 * t0 ** -0.118
        else:
            g_la = 1.13 * t0 ** -0.094
        g_rh = 1.27 - 0.0067 * self.RH
        vs = self.v_over_s_in
        g_vs = (2.0 / 3.0) * (1.0 + 1.13 * math.exp(-0.54 * vs))
        phi_u = 2.35 * g_la * g_rh * g_vs * self.gamma_other_creep
        dt = t - t0
        return (dt ** 0.6) / (10.0 + dt ** 0.6) * phi_u

    def J(self, t, t_prime):
        r""":math:`J(t,t') = [1 + \varphi(t,t')] / E_c(t')` [1/MPa] —
        creep referenced to the modulus **at loading**."""
        t = float(t)
        tp = float(t_prime)
        if t < tp:
            raise ValueError(
                f"ACIRheologicalModel.J: t ({t}) must be >= t' ({tp})."
            )
        return (1.0 + self.phi_aci(t, tp)) / self.E_c(tp)

    # -- shrinkage --------------------------------------------------

    def eps_imposed(self, t, t_s):
        r"""Shrinkage :math:`\varepsilon_{sh}(t,t_s)` [-], **signed**
        (negative)."""
        self._require_geometry("eps_imposed")
        t = float(t)
        t_s = float(t_s)
        if t <= t_s:
            return 0.0
        lam = self.RH
        if lam <= 80.0:
            g_rh = 1.40 - 0.0102 * lam
        else:
            g_rh = 3.00 - 0.030 * lam
        g_vs = 1.2 * math.exp(-0.12 * self.v_over_s_in)
        eps_shu = 780e-6 * g_rh * g_vs * self.gamma_other_shrinkage
        f = 35.0 if self.curing == "moist" else 55.0
        dt = t - t_s
        return -(dt / (f + dt)) * eps_shu

    # -- relaxation -------------------------------------------------

    def relaxation(self, t, mu):
        r"""Magura–Sozen–Siess log law, **signed** (negative) [MPa]."""
        self._require_steel("relaxation")
        f_pk = self.f_pk
        t = float(t)
        mu = float(mu)
        if t <= 0.0 or mu <= 0.0:
            return 0.0
        sigma_pi = mu * float(f_pk)
        f_py = self.f_py_ratio * float(f_pk)
        over = sigma_pi / f_py - 0.55
        if over <= 0.0:
            return 0.0
        C = 45.0 if self.low_relaxation else 10.0
        t_h = max(24.0 * t, 1.0)          # log10(1 h) = 0: no decay yet
        return -sigma_pi * (math.log10(t_h) / C) * over

    def with_steel(self, f_pk, low_relaxation=None, f_py_ratio=None):
        r"""Return a copy bound to a prestressing steel (see
        :meth:`EC2RheologicalModel.with_steel`)."""
        out = copy.copy(self)
        out.f_pk = float(f_pk)
        if low_relaxation is not None:
            out.low_relaxation = bool(low_relaxation)
        if f_py_ratio is not None:
            out.f_py_ratio = float(f_py_ratio)
        return out


# ==================================================================
#  Tabulated — a code that is only data
# ==================================================================

class TabulatedRheologicalModel(RheologicalModel):
    r"""
    User-supplied compliance / shrinkage / relaxation tables.

    The cheapest possible proof of genericity: a norm that exists only
    as tabulated data — a national annex, a lab campaign, a B4
    pre-computation, a client's own curves — enters through the same
    four-function door as the Eurocode, and the container cannot tell
    the difference.

    Interpolation is **bilinear in** :math:`\log_{10}` **time** for the
    compliance (creep data are log-uniform by nature) and linear in the
    time axis for shrinkage; outside the tabulated range the *nearest
    edge* value is held, and a request beyond the range raises unless
    ``extrapolate=True``.

    Parameters
    ----------
    t_prime_grid : array_like
        Loading ages :math:`t'` of the compliance table [days], strictly
        increasing, positive.
    t_grid : array_like
        Observation ages :math:`t` [days], strictly increasing, positive.
    J_table : array_like
        Compliance :math:`J(t, t')` [1/MPa], shape
        ``(len(t_grid), len(t_prime_grid))``.
    shrinkage : callable or tuple, optional
        Either a callable ``f(t, t_s) -> eps`` (signed), or a pair
        ``(t_grid_sh, eps_sh)`` interpolated in ``t - t_s``.  Default:
        no imposed strain.
    relaxation : callable, optional
        ``f(t_days, mu) -> Delta sigma_pr [MPa]`` (signed).  Default: no
        relaxation.
    extrapolate : bool, optional
        Default ``False`` — a query outside the table raises rather than
        silently clamp.
    name : str, optional

    Raises
    ------
    ValueError
        Malformed grids or table shape; out-of-range query with
        ``extrapolate=False``.

    Notes
    -----
    :meth:`E_c` is **derived**: :math:`E_c(t) = 1/J(t,t)`, i.e. the
    diagonal of the table.  A table whose diagonal is not populated
    cannot supply a modulus, and says so.
    """

    def __init__(self, t_prime_grid, t_grid, J_table, shrinkage=None,
                 relaxation=None, extrapolate=False, name=""):
        tp = np.asarray(t_prime_grid, dtype=float).ravel()
        tt = np.asarray(t_grid, dtype=float).ravel()
        JT = np.asarray(J_table, dtype=float)
        if tp.size < 2 or tt.size < 2:
            raise ValueError(
                "TabulatedRheologicalModel: both grids need >= 2 nodes."
            )
        if np.any(np.diff(tp) <= 0) or np.any(np.diff(tt) <= 0):
            raise ValueError(
                "TabulatedRheologicalModel: t_prime_grid and t_grid must "
                "be strictly increasing."
            )
        if tp[0] <= 0.0 or tt[0] <= 0.0:
            raise ValueError(
                "TabulatedRheologicalModel: ages must be positive (the "
                "interpolation is in log-time)."
            )
        if JT.shape != (tt.size, tp.size):
            raise ValueError(
                f"TabulatedRheologicalModel: J_table has shape "
                f"{JT.shape}, expected (len(t_grid), len(t_prime_grid)) "
                f"= ({tt.size}, {tp.size})."
            )
        if np.any(JT <= 0.0):
            raise ValueError(
                "TabulatedRheologicalModel: every compliance entry must "
                "be positive (J = 1/E at the diagonal)."
            )
        self._tp = tp
        self._t = tt
        self._J = JT
        self._log_tp = np.log10(tp)
        self._log_t = np.log10(tt)
        self._shrinkage = shrinkage
        self._relaxation = relaxation
        self.extrapolate = bool(extrapolate)
        self.name = name or "tabulated"

    def _check_range(self, value, grid, what):
        if self.extrapolate:
            return
        if value < grid[0] - 1e-9 or value > grid[-1] + 1e-9:
            raise ValueError(
                f"TabulatedRheologicalModel.{what}: {value:g} d is "
                f"outside the tabulated range "
                f"[{grid[0]:g}, {grid[-1]:g}] d.  Extend the table, or "
                f"construct with extrapolate=True to hold the edge "
                f"value (and accept the flat extrapolation)."
            )

    def J(self, t, t_prime):
        r"""Bilinear interpolation of the table in :math:`\log_{10}`
        time [1/MPa]."""
        t = float(t)
        tp = float(t_prime)
        if t < tp:
            raise ValueError(
                f"TabulatedRheologicalModel.J: t ({t}) must be >= t' "
                f"({tp})."
            )
        self._check_range(t, self._t, "J")
        self._check_range(tp, self._tp, "J")
        lt = np.clip(math.log10(max(t, 1e-9)),
                     self._log_t[0], self._log_t[-1])
        ltp = np.clip(math.log10(max(tp, 1e-9)),
                      self._log_tp[0], self._log_tp[-1])
        # interpolate along t' for each bracketing t row, then along t
        rows = np.array([np.interp(ltp, self._log_tp, self._J[i, :])
                         for i in range(self._t.size)])
        return float(np.interp(lt, self._log_t, rows))

    def E_c(self, t):
        r""":math:`E_c(t) = 1/J(t,t)` — the table's diagonal [MPa]."""
        return 1.0 / self.J(t, t)

    def eps_imposed(self, t, t_s):
        r"""Imposed strain [-] from the user's shrinkage callable or
        table; **signed** — the user owns the sign."""
        if self._shrinkage is None:
            return 0.0
        if callable(self._shrinkage):
            return float(self._shrinkage(float(t), float(t_s)))
        grid, vals = self._shrinkage
        dt = max(0.0, float(t) - float(t_s))
        return float(np.interp(dt, np.asarray(grid, dtype=float),
                               np.asarray(vals, dtype=float)))

    def relaxation(self, t, mu):
        r"""Relaxation [MPa] from the user's callable; **signed**."""
        if self._relaxation is None:
            return 0.0
        return float(self._relaxation(float(t), float(mu)))

