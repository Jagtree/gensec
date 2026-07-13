# ---------------------------------------------------------------------------
# GenSec — Copyright (c) 2026 Andrea Albero
#
# This file is part of GenSec.  Licensed under the GNU Affero General
# Public License v3 or later.  See LICENSE.
# ---------------------------------------------------------------------------
r"""
ACI 209R-92 — the normative-agnosticism **falsification fixture**.

**This is not a library module and it does not ship with the package.**  It
sits at the repository root, beside the validators, deliberately.

An abstraction with a single implementation is not an abstraction, it is an
indirection: you cannot know whether
:class:`~gensec.materials.base.RheologicalModel` is genuinely
normative-agnostic until a **second, structurally different** code has gone
through it.  ACI 209R-92 is that code — hyperbolic time functions where EC2
has :math:`[\Delta t/(\beta_H + \Delta t)]^{0.3}`; a volume/surface ratio in
**inches** where EC2 has a notional size in millimetres;
:math:`E_c = 4700\sqrt{f'_c}` where EC2 has :math:`22000\,(f_{cm}/10)^{0.3}`;
a compliance referenced to the modulus **at loading** where EC2 references
28 days.  The same four functions.  The container cannot tell them apart,
and :func:`~gensec.materials.base.aging_coefficient` returns
:math:`\chi = 0.839` for ACI against :math:`0.804` for EC2 — *from the same
code path*.  That is the proof; without it the agnosticism claim is an
assertion.

But GenSec has **no ACI material**: no partial factors, no limit states, no
bridge.  You cannot run an ACI design with it.  Shipping this inside
``gensec.materials`` promised a capability that does not exist — and the tell
was the warning its own docstring had to carry: *"before using it for a real
ACI design, check the coefficients against a copy of ACI 209R-92."*  **Code
you must warn people not to use is not a product.**  It is a fixture, and a
fixture belongs with the tests that run it.

When an ACI *material* exists — bridge, partial factors, limit states — this
becomes a provider and moves into the package.  Not before.
"""

import copy
import math

import numpy as np  # noqa: F401  (kept for parity with the providers)

from gensec.materials.base import RheologicalModel

__all__ = ["ACIRheologicalModel"]


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
