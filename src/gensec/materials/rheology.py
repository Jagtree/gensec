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

Two providers ship.

An **ACI 209R-92** provider also exists, but it does **not** ship here: it
lives in ``aci209_falsification.py``, outside the package, because GenSec
has no ACI *material* -- no partial factors, no limit states, no bridge.  A
rheology suspended over a normative vacuum is not a provider, it is a test
fixture; putting it in ``src/`` would promise a design capability that does
not exist.  Its job is to **falsify** the claim this module makes, and a
falsification belongs with the tests that run it.

The two that do ship:

:class:`EC2RheologicalModel`
    EN 1992-1-1 Annex B + §3.1.4 + §3.3.2 (also NTC 2018, which adopts
    the same formulae).  Selected in YAML by ``provider: ec2`` or
    ``provider: ntc``; it is not a default, a ``losses_models`` entry
    names its provider.
:class:`TabulatedRheologicalModel`
    User compliance / shrinkage / relaxation tables
    (``provider: tabulated``).  A code that is *only data* passes
    through the same door.

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

from functools import lru_cache

from .base import ReferenceModulusRule, RheologicalModel
from .ec2_properties import fben2


@lru_cache(maxsize=2048)
def _ec2_at(fck, cement_class, t):
    r"""
    EN 1992-1-1's instantaneous properties at age *t*, from
    :class:`~gensec.materials.ec2_properties.fben2` — **the single place
    they live**.

    Module-level and keyed by its own inputs, so it cannot go stale:
    there is no instance state to fall out of sync with the material.
    Change ``fben2`` and *everything* downstream changes; there is nothing
    to keep in step by hand.

    Costs 4.6 us per distinct age.  A 100-step Volterra inversion touches
    ~100 distinct ages, so the whole ageing-coefficient computation pays
    under half a millisecond for it.  There is no reason to cache anything
    on the provider.

    ``ls='S'`` / ``loadtype='slow'`` select fben2's *design* factors
    (gamma_c, alpha_cc, f_cd), **none of which a rheological model may
    see**.  Naming them explicitly is the point: this reads material
    properties, never design ones.
    """
    if t <= 0.0:
        raise ValueError(f"EC2 rheology: age t must be > 0, got {t}.")
    return fben2(fck=fck, ls="S", loadtype="slow", TypeConc=cement_class,
                 time=t)

__all__ = [
    "EC2RheologicalModel",
    "EC2ReferenceModulus",
    "TabulatedRheologicalModel",
    "default_reference_modulus",
]


# ==================================================================
#  Shared geometry binding
# ==================================================================

class EC2ReferenceModulus(ReferenceModulusRule):
    r"""
    EN 1992-1-1 Table 3.1: :math:`E_{cm}` at the 28-day reference age.

    The conventional secant modulus a Eurocode analysis uses when the
    model declares no time axis:

    .. math::

        E_{cm} = 22\,000
                 \left(\frac{f_{cm}}{10}\right)^{0.3}
        \qquad [\mathrm{MPa}],\; f_{cm} = f_{ck} + 8

    read from the material's **own** EC2 bridge rather than
    recomputed here, so the number a datum is drawn on and the number
    the material's other clauses use are the same object's answer
    (rule 5).  Verified equal to
    ``EC2RheologicalModel(fck).E_c(28.0)`` to the last digit
    (35 220.462 MPa at ``fck = 40``), which is the consistency this
    reading buys.

    This class is a **default, not a wiring**.  A model under ACI 318,
    AS 3600 or a national annex supplies its own
    :class:`~gensec.materials.base.ReferenceModulusRule`, with its own
    reference age and its own citation, and nothing in the resolver
    changes.  The Eurocode is the shipped default because it is this
    project's shipped default everywhere else -- not because the
    mechanism knows it.

    Parameters
    ----------
    source : str, optional
        Citation.  Defaults to the Table 3.1 reference.

    See Also
    --------
    gensec.materials.base.ReferenceModulusRule : the contract.
    default_reference_modulus : the shipped instance.

    Examples
    --------
    >>> EC2ReferenceModulus().reference_age
    28.0
    """

    _DEFAULT_SOURCE = (
        "EN 1992-1-1:2004 §3.1.3 Table 3.1 -- E_cm, secant modulus of "
        "elasticity of concrete at the 28-day reference age.  Applied "
        "only where the model declares no time axis, i.e. where no "
        "other age is stated by the analysis."
    )

    def __init__(self, source=None):
        super().__init__(source=source or self._DEFAULT_SOURCE,
                         reference_age=28.0)

    def E_c(self, material):
        r"""
        :math:`E_{cm}` [MPa] of *material*, or ``None``.

        Parameters
        ----------
        material : Material
            Any material.  Only one that carries an EC2 bridge
            (``material.ec2.ecm``) is described by this rule.

        Returns
        -------
        float or None
            ``None`` for a material with no EC2 bridge -- a steel, a
            tabulated law, a user law.  Declining is the correct
            answer: the resolver, not this rule, owns the refusal.
        """
        # Ask the material first.  Every material class declares its
        # own conventional modulus, at its own reference age, with its
        # own citation, so the agnostic path needs no probe.  The EC2
        # branches below are the *override*: they exist for a material
        # that cannot state one (the legacy ``concrete`` alias) or for
        # an engineer deliberately applying this code's convention to a
        # material built under another.
        try:
            declared = material.reference_modulus()
        except (NotImplementedError, AttributeError):
            declared = None
        if declared is not None and float(declared.value) > 0.0:
            return float(declared.value)
        bridge = getattr(material, "ec2", None)
        ecm = getattr(bridge, "ecm", None) if bridge is not None else None
        if ecm is None:
            # A material built through the legacy ``concrete`` alias
            # carries no bridge object but does declare ``fck``, which
            # is the only input Table 3.1 needs.  It is evaluated
            # through the same ``fben2`` the bridge would have used --
            # one formula, one home -- and at the 28-day reference age
            # :math:`\beta_{cc} = 1`, so the cement class drops out
            # exactly here.  The conventional value therefore carries
            # no undeclared hardening choice, which is the property
            # that makes it usable without a time axis at all.
            fck = getattr(material, "fck", None)
            if fck is None:
                return None
            try:
                ecm = _ec2_at(float(fck), "N", 28.0).ecm
            except Exception:                            # noqa: BLE001
                return None
        ecm = float(ecm)
        return ecm if ecm > 0.0 else None


#: The shipped default rule.  Named so that a caller can replace it
#: rather than edit it: ``ConstructionTimeline(...,
#: reference_modulus=MyCodeReferenceModulus())``.
def default_reference_modulus():
    r"""
    The default :class:`ReferenceModulusRule` of this build.

    A function rather than a module constant so that the object is not
    shared mutable state across analyses, and so that a build for a
    different code family has one symbol to replace.

    Returns
    -------
    EC2ReferenceModulus
    """
    return EC2ReferenceModulus()


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

    — the convention of §5.10.6 and §7.4.3: the creep strain is
    referenced to the **28-day** modulus, the instantaneous strain to the
    modulus *at loading*.

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

    #: (k1, k2, rho_1000 [%]) — EN 1992-1-1 §3.3.2 (3.28)-(3.30).
    _RELAX = {1: (5.39, 6.7, 8.0),
              2: (0.66, 9.1, 2.5),
              3: (1.98, 8.0, 4.0)}
    #: Cement-class exponent alpha of the loading-age correction (B.9).
    _ALPHA_CEM = {"R": 1.0, "N": 0.0, "S": -1.0}
    #: Shrinkage cement coefficients (alpha_ds1, alpha_ds2), eq. (B.11).
    _ALPHA_DS = {"R": (6.0, 0.11), "N": (4.0, 0.12), "S": (3.0, 0.13)}
    #: EN 1992-1-1 §3.3.2(8): the long-term relaxation reference time [h].
    T_RELAX_MAX_HOURS = 500000.0

    #: Declared content: the Eurocode material parameters, plus the
    #: late-bound drying geometry and prestressing steel.  ``fcm``,
    #: ``Ecm28`` and every other Table-3.1 quantity are deliberately
    #: absent: this class stores none of them, it reads them from the
    #: normative source on demand, so they are functions of ``fck``
    #: and ``cement_class`` and add no discrimination.
    CONTENT_FIELDS: tuple = (
        "fck", "cement_class", "RH", "relaxation_class", "rho_1000",
        "A_c", "u", "f_pk",
    )

    def __init__(self, fck, cement_class="N", RH=70.0, A_c=None, u=None,
                 relaxation_class=2, rho_1000=None, name=""):
        cc = str(cement_class).upper()
        if cc not in self._ALPHA_CEM:
            raise ValueError(
                f"EC2RheologicalModel: cement_class must be one of "
                f"{sorted(self._ALPHA_CEM)}, got {cement_class!r}.  "
                f"(fben2 accepts an unknown class silently, so the check "
                f"has to live here.)"
            )
        rc = int(relaxation_class)
        if rc not in self._RELAX:
            raise ValueError(
                f"EC2RheologicalModel: relaxation_class must be 1, 2 or 3 "
                f"(EN 1992-1-1 §3.3.2), got {relaxation_class!r}."
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
        self.name = name or f"ec2(fck={self.fck:g},{cc},RH={self.RH:g})"
        if A_c is not None and u is not None:
            bound = self.with_geometry(A_c, u)
            self.A_c, self.u = bound.A_c, bound.u

    # -- everything instantaneous is READ from fben2, never stored ----
    #
    # Not one Eurocode value is copied onto this object.  A copy taken at
    # construction is a value frozen at construction: change the source and
    # it does not follow.  That is the unenforced redundancy of findings
    # F5/F6/F7, in miniature -- and an earlier revision of this class had
    # exactly that (fcm, Ecm28, s_cem, beta_cc, E_c all re-derived, because
    # fben2.ecm was then broken by F2).  Forking a normative computation to
    # route around a bug leaves two of them forever.  Fix the source.

    #: EN 1992-1-1 §3.1.2(5) gives f_cm(t) = beta_cc(t) f_cm and
    #: f_ck(t) = f_cm(t) - 8 MPa for 3 d <= t; below three days the
    #: clause does not define them and refers the question elsewhere.
    #: The floor is therefore *data taken from a document*, not a rule
    #: of GenSec's: another code places it elsewhere or nowhere at all
    #: (ACI 209R-92 states f'_c(t) = t/(a+bt) f'_c(28) from t -> 0+),
    #: and a National Annex may move this one.  Nothing outside this
    #: mapping knows the number 3.
    #:
    #: E_c(t), phi and eps_imposed are deliberately absent: §3.1.3(3)
    #: builds the modulus from beta_cc directly, not through f_cm, and
    #: is stated at every age.
    #: Closed vocabulary (N-8).  The container's four invariants come
    #: from :class:`~gensec.materials.base.RheologicalModel`; the four
    #: names added here are the EN 1992-1-1 quantities this provider
    #: declares an interval for, plus ``phi``, which it states at every
    #: age.
    #:
    #: Three of them -- ``f_ck``, ``f_cm``, ``f_ctm`` -- have **no
    #: method on this class**.  They are declared because the norm
    #: states where they hold and the dump publishes that, and they are
    #: kept in the vocabulary rather than deleted because deleting a
    #: declared interval a downstream consumer reads would widen the
    #: domain silently -- the same defect from the other side.
    #: :meth:`~gensec.materials.base.RheologicalModel.check_vocabulary`
    #: reports them under ``declared_no_surface`` so the gap is visible
    #: instead of silent.  Whether the interval itself is right is
    #: **open**: see ``17_0`` section N-8.
    QUANTITIES = RheologicalModel.QUANTITIES | frozenset({
        "f_ck", "f_cm", "f_ctm", "linearity_limit", "phi",
    })

    #: N-8, closed in Phase 17 **from the implementation**, which is
    #: where the route each quantity takes is written:
    #:
    #: * ``f_cm`` and ``f_ck`` are zeroed below three days by
    #:   :class:`~gensec.materials.ec2_properties.fben2` -- a deliberate
    #:   marker, because ``f_ck(t) = f_cm(t) - 8`` of §3.1.2(6) goes
    #:   negative there.  The floor describes the code.  Kept.
    #: * ``linearity_limit`` is :math:`0.45\,f_{ck}(t)` and inherits that
    #:   floor through ``f_ck``.  Kept.
    #: * ``f_ctm`` is **removed**.  Expression (3.4) is written on
    #:   :math:`\beta_{cc}(t)` and the 28-day :math:`f_{ctm}`, and
    #:   ``fben2`` computes it from ``fck_28`` and ``beta_cc`` at every
    #:   age, below three days included -- which is precisely what F-M5
    #:   established when it restored :math:`\alpha = 1` there.  The
    #:   entry declared a floor the implementation does not have, and
    #:   its citation ("through ``f_cm(t)``") named a route neither the
    #:   code nor (3.4) takes.  It was the last fossil of the belief
    #:   F-M5 corrected.
    #:
    #: Absent from this mapping means *stated at every age*: that is now
    #: a claim about ``f_ctm`` and no longer an accident.
    DOMAIN = {
        "f_cm": (3.0, float("inf")),
        "f_ck": (3.0, float("inf")),
        "linearity_limit": (3.0, float("inf")),
    }

    DOMAIN_SOURCE = {
        "f_cm": "EN 1992-1-1 §3.1.2(5)",
        "f_ck": "EN 1992-1-1 §3.1.2(5)",
        "linearity_limit": ("EN 1992-1-1 §3.1.4(4), through f_ck(t) of "
                            "§3.1.2(5)"),
    }

    def _p(self, t):
        r"""fben2 at age *t*.  The single source (:func:`_ec2_at`)."""
        return _ec2_at(self.fck, self.cement_class, float(t))

    @property
    def fcm(self):
        r"""Mean cylinder strength at 28 d [MPa] (Table 3.1)."""
        return float(self._p(28.0).fcm_28)

    @property
    def Ecm28(self):
        r"""Secant modulus at 28 d [MPa] (Table 3.1)."""
        return float(self._p(28.0).ecm_28)

    def E_c(self, t):
        r"""
        :math:`E_{cm}(t) = [\beta_{cc}(t)]^{0.3} E_{cm}(28)` [MPa]
        (§3.1.3(3)) — :attr:`fben2.ecm`.
        """
        return float(self._p(t).ecm)

    def service_law(self, t):
        r"""
        EN 1992-1-1 expression (3.14) at age *t*, with the Eurocode's
        own tensile strength.

        The five constants are read from the **one** ``fben2``
        evaluation this provider already uses for :meth:`E_c`, so the
        law and the modulus cannot drift apart: they are two readings
        of a single normative evaluation at a single age.

        .. math::

            f_{cm}(t),\quad E_{cm}(t),\quad
            \varepsilon_{c1}(t),\quad \varepsilon_{cu1},\quad
            f_{ct,\mathrm{eff}} = f_{ctm}(t) .

        The last line is the point.  :math:`f_{ct,\mathrm{eff}}` is not
        a parameter of this method and has no default: **EN 1992-1-1
        7.3.2(2) nominates** :math:`f_{ctm}` as the effective tensile
        strength at the moment cracking is expected, and this provider
        is entitled to apply that clause precisely because it *is* the
        Eurocode provider.  A provider that declares a different
        document answers from that document, inside that document's own
        module; neither may answer for the other.  A caller that wants
        :math:`f_{ctm,fl}` or :math:`f_{ctk,0.05}` is declaring a
        different rule and declares it by building the material
        itself -- which is a declaration, not an override of this one.

        Below three days ``fben2`` reports :math:`f_{cm}=0` as the
        marker of its own domain, and
        :class:`~gensec.materials.service_concrete.ServiceConcrete`
        refuses it with a ``ValueError``.  That refusal is propagated
        unchanged: an age outside the domain the source claims is an
        undeclared input, not unmodelled physics.

        Parameters
        ----------
        t : float
            Age [days].

        Returns
        -------
        gensec.materials.service_concrete.ServiceConcrete

        Raises
        ------
        ValueError
            If *t* is outside the domain of EN 1992-1-1 Table 3.1 as
            ``fben2`` reports it.
        """
        from .ec2_properties import (EC2_EARLY_AGE_LIMIT_DAYS,
                                     EC2EarlyAgeConcrete)
        from .service_concrete import ServiceConcrete

        t = float(t)
        if t < EC2_EARLY_AGE_LIMIT_DAYS:
            # Refused from its own address, not by a stray guard here.
            # Below three days fben2 zeroes f_cm and eps_c1 while still
            # reporting E_cm and f_ctm, so a law built from it would be
            # stiff, resist tension and resist no compression.  The
            # boundary is EN 1992-1-1 §3.1.2(5)'s, so it is declared
            # in the Eurocode module and refused from there: each code
            # runs a closed cycle and does not borrow another's edges.
            EC2EarlyAgeConcrete(t=t, fck=self.fck, name=self.name)
        p = self._p(t)
        return ServiceConcrete(
            fcm=float(p.fcm), Ecm=float(p.ecm),
            eps_c1=float(p.eps_c1), eps_cu1=float(p.eps_cu1),
            fct_eff=float(p.fctm), reference_age=t,
            name=f"{self.name} service @{t:g}d")

    def linearity_limit(self, t):
        r"""
        :math:`0.45\, f_{ck}(t)` [MPa] (§3.1.4(4)), above which the creep
        is **non-linear** and this provider's stress-independent compliance
        no longer applies.

        :math:`f_{ck}(t)` is fben2's, hence **capped at the 28-day value**
        for :math:`t \ge 28` d, exactly as §3.1.2(5) prescribes.

        Below three days §3.1.2(5) does not define :math:`f_{ck}(t)` and
        this provider therefore has no limit to give.  It raises rather
        than return the zero that fben2's out-of-range branch produces:
        a limit of zero is not a limit, it is a tripwire that any
        compression whatsoever crosses, and a guard that fires on a
        quantity the code never defined reports the wrong cause.  Supply
        a :class:`~gensec.materials.base.DeclaredRule` if the analysis
        must reach below three days.

        Parameters
        ----------
        t : float
            Age at loading [days].

        Returns
        -------
        float
            Limit stress [MPa, positive magnitude].

        Raises
        ------
        ~gensec.materials.base.OutOfDeclaredDomain
            For :math:`t < 3` d with no declared rule covering *t*.

        Examples
        --------
        >>> m = EC2RheologicalModel(fck=30, cement_class="N")
        >>> round(m.linearity_limit(28.0), 3)
        13.5
        >>> m.linearity_limit(1.0)               # doctest: +ELLIPSIS
        Traceback (most recent call last):
        gensec.materials.base.OutOfDeclaredDomain: ...
        """
        return self._stated(
            "linearity_limit", t,
            lambda tt: 0.45 * max(float(self._p(tt).fck), 0.0))

    # -- geometry ---------------------------------------------------

    @property
    def h0(self):
        r"""Notional size :math:`h_0 = 2 A_c / u` [mm] (eq. B.6)."""
        self._require_geometry("h0")
        return 2.0 * self.A_c / self.u

    # -- ageing of the strength / modulus ---------------------------

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
        :math:`t_0 = 28` d.

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
        convention (§5.10.6).
        """
        t = float(t)
        tp = float(t_prime)
        if t < tp:
            raise ValueError(
                f"EC2RheologicalModel.J: t ({t}) must be >= t' ({tp})."
            )
        return 1.0 / self.E_c(tp) + self.phi_ec2(t, tp) / self.Ecm28

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

    #: Declared content: the two grids, the compliance table, the two
    #: optional callables and the extrapolation policy, plus the
    #: late-bound geometry and steel.  The log-grids cached at
    #: construction are functions of the grids and are not listed.
    #:
    #: A callable enters the token by its importable name, which is
    #: the honest limit of the scheme: two different functions with
    #: the same qualified name would collide, and an anonymous
    #: function is rejected outright rather than tokenised into
    #: something that does not survive a process boundary.
    CONTENT_FIELDS: tuple = (
        "_tp", "_t", "_J", "_shrinkage", "_relaxation", "extrapolate",
        "A_c", "u", "f_pk",
    )

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
        self._declare_domains()

    def _declare_domains(self):
        r"""
        Publish the table's own span through
        :meth:`~gensec.materials.base.RheologicalModel.domain`.

        The grids differ per object, so the intervals are recorded on
        the instance rather than on the class.  ``extrapolate=True``
        widens them to every age, which is exactly what that flag means:
        the caller has accepted a flat extension of the data.
        """
        if self.extrapolate:
            self._domain = {}
            self._domain_source = {}
            return
        span_t = (float(self._t[0]), float(self._t[-1]))
        span_tp = (float(self._tp[0]), float(self._tp[-1]))
        src = f"tabulated data supplied to {self.name or 'this provider'}"
        self._domain = {"J": span_t, "E_c": span_tp, "phi": span_t}
        self._domain_source = {"J": src, "E_c": src, "phi": src}

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

    def service_law(self, t):
        r"""
        Refuse, by name: a compliance table nominates no strength.

        This provider is *a code that is only data*.  It declares
        :math:`J(t,t')` and nothing else, so it knows the concrete's
        compliance and does not know its strength, its strain at peak
        or its tensile strength.  A constitutive law cannot be built
        from a compliance table, and building one from **another**
        code's constants -- reaching for EN 1992-1-1 Table 3.1 because
        it happens to be in the same package -- would be precisely the
        silent normative choice this architecture exists to prevent.
        The Eurocode's :math:`f_{ctm}` is the Eurocode's answer and is
        not available to a provider that has not declared the
        Eurocode.

        The refusal is a ``NotImplementedError`` and not a
        ``ValueError``: the input is complete and well formed, and what
        is missing is a *model* -- a constitutive law this provider's
        declared content does not contain.  Supplying the missing
        physics is a declaration the caller makes by handing in a
        material of its own.

        Parameters
        ----------
        t : float
            Age [days].  Accepted and unused: the refusal does not
            depend on the age.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "TabulatedRheologicalModel declares a compliance table and "
            "no normative document: it knows J(t,t') and therefore E_c, "
            "but not f_cm, eps_c1 or a tensile strength, so it cannot "
            "produce a constitutive law.  Borrowing EN 1992-1-1 Table "
            "3.1 here would attribute to this model a code it has not "
            "declared.  Declare the service material explicitly, or use "
            "a provider whose code owns a constitutive law."
        )

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

