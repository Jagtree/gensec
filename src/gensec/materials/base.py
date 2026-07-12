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
Abstract base class for constitutive material laws.

All concrete material implementations must inherit from :class:`Material`
and provide scalar/vectorized stress evaluation as well as tangent
modulus evaluation for the analytical Jacobian.
"""

import numpy as np
import copy
from abc import ABC, abstractmethod


class Material(ABC):
    r"""
    Abstract base class for all constitutive material laws.

    Every material must expose:

    - ``stress(eps)`` — scalar stress evaluation.
    - ``stress_array(eps)`` — vectorized evaluation over an array
      of **arbitrary shape** (1-D, 2-D, …).
    - ``tangent(eps)`` — scalar tangent modulus
      :math:`E_t = \mathrm{d}\sigma/\mathrm{d}\varepsilon`.
    - ``tangent_array(eps)`` — vectorized tangent modulus over an
      array of arbitrary shape.
    - ``eps_min`` / ``eps_max`` — admissible strain range.

    The tangent modulus is used by
    :meth:`~gensec.solver.FiberSolver.integrate_with_tangent` to
    assemble the **analytical** tangent stiffness matrix, avoiding
    the cost of finite-difference Jacobians in Newton-Raphson solvers.

    The strain limits are consumed by the N-M diagram generator to
    automatically determine the scan range for ultimate configurations.

    Attributes
    ----------
    eps_min : float
        Most compressive admissible strain (typically negative).
    eps_max : float
        Most tensile admissible strain (typically positive).
    name : str
        Human-readable identifier set by the YAML loader.
    """

    name: str = ""

    @property
    @abstractmethod
    def eps_min(self):
        """Most compressive admissible strain."""
        ...

    @property
    @abstractmethod
    def eps_max(self):
        """Most tensile admissible strain."""
        ...

    @abstractmethod
    def stress(self, eps):
        """
        Compute stress for a single strain value.

        Parameters
        ----------
        eps : float
            Strain.

        Returns
        -------
        float
            Stress [MPa].
        """
        ...

    def stress_array(self, eps):
        r"""
        Vectorized stress computation.

        Default implementation loops over :meth:`stress`.  Subclasses
        should override for performance.  The input may have
        **any shape** (1-D, 2-D, …); the output must have the same
        shape.

        Parameters
        ----------
        eps : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return np.array([self.stress(e) for e in eps.ravel()]).reshape(
            eps.shape)

    # ------------------------------------------------------------------
    #  Tangent modulus — analytical Jacobian support
    # ------------------------------------------------------------------

    def tangent(self, eps):
        r"""
        Scalar tangent modulus :math:`E_t = d\sigma / d\varepsilon`.

        Default implementation uses a central finite difference with
        step :math:`h = 10^{-8}`.  Subclasses should override with the
        closed-form derivative for best accuracy and speed.

        Parameters
        ----------
        eps : float

        Returns
        -------
        float
            Tangent modulus [MPa].
        """
        h = 1e-8
        return (self.stress(eps + h) - self.stress(eps - h)) / (2.0 * h)

    def tangent_array(self, eps):
        r"""
        Vectorized tangent modulus.

        Default implementation uses a central finite difference.
        Subclasses should override with the closed-form derivative.
        Input may have **any shape**.

        Parameters
        ----------
        eps : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Same shape as *eps*.
        """
        h = 1e-8
        return (self.stress_array(eps + h) - self.stress_array(eps - h)) / (
            2.0 * h)


# ======================================================================
#  Rheology — time-dependent constitutive providers
#
#  APPEND TO gensec/materials/base.py (after the Material ABC).
#  Requires, at the top of base.py:  import copy, numpy as np
#                                    from abc import ABC, abstractmethod
# ======================================================================


class RheologicalModel(ABC):
    r"""
    Normative-agnostic time-dependent constitutive provider.

    Supplies the four material functions the age-adjusted viscoelastic
    section container (:mod:`gensec.solver.losses`) needs.  **All**
    normative content — EN 1992-1-1, NTC 2018, ACI 209R, *fib* MC2010,
    B4, a national annex, a lab campaign — lives behind this interface.
    The container consumes only these four functions and derives
    everything else, including the ageing coefficient :math:`\chi`,
    from them.

    The four functions
    ------------------
    ========================  ====================================
    :meth:`J`                 creep compliance [1/MPa]
    :meth:`eps_imposed`       stress-independent strain [-]
    :meth:`relaxation`        intrinsic steel relaxation [MPa]
    :meth:`E_c`               concrete modulus at age *t* [MPa]
    ========================  ====================================

    Units and signs, fixed at this boundary
    ---------------------------------------
    - Time in **days**, length in **mm**, stress in **MPa**, force in
      **N**.  A provider authored in other units (ACI 209R in psi and
      inches) converts *internally*: unit leakage is the **provider's**
      responsibility, never the container's.
    - Strain is **signed** in GenSec's structural convention:
      shortening is **negative**.  Shrinkage therefore returns a
      *negative* :meth:`eps_imposed` — not the positive magnitude that
      EN 1992-1-1 §3.1.4 tabulates.  This is the sign mapping the
      Phase-5 bulk kernel deliberately left open
      (``7_1-GENSEC_PHASE5_BULK_KERNEL_PATCH.md`` §9); it is fixed
      here, and the container flips it once, explicitly, when it forms
      the bulk offset (:math:`\varepsilon_{\text{bulk}} =
      -\varepsilon_{\mathrm{imp}}`, because the kernel *adds* the
      offset while the physics *subtracts* the eigenstrain).
    - :meth:`relaxation` likewise returns the **signed** stress change
      of the tendon: a decay, hence :math:`\le 0`.

    Geometry
    --------
    The container hands geometry as the pair :math:`(A_c, u)` — the
    concrete area [mm²] and the perimeter exposed to drying [mm] — and
    **never** as a code-specific nominal size.  Each provider forms its
    own measure: EN 1992-1-1 the notional size :math:`h_0 = 2A_c/u`,
    ACI 209R the volume/surface ratio :math:`V/S = A_c/u` (a factor of
    two apart, and in different units).  Binding is **late**
    (:meth:`with_geometry`) because in a composite section every
    concrete zone has its own exposed perimeter, hence its own notional
    size, while the *material* parameters are shared.

    Notes
    -----
    The creep coefficient the container works with is **derived** from
    the compliance, never asked of the provider:

    .. math::

        \varphi(t, t') \;=\; E_c(t')\, J(t, t') \;-\; 1

    (:meth:`phi`).  This is an identity for any provider.  For an EC2
    provider at :math:`t' = 28` d it reproduces the Eurocode
    :math:`\varphi(t,t_0)` exactly; for a B4 provider it returns
    whatever B4's multi-term compliance implies.  A norm that has no
    creep coefficient still has one *implied* by its compliance, and
    that is the number the AAEM equation needs.

    See Also
    --------
    aging_coefficient : :math:`\chi(t,t_0)` computed **from** ``J``.
    relaxation_function : :math:`R(t,t_0)` computed **from** ``J``.
    gensec.materials.rheology : the shipped providers.
    """

    #: Human-readable identifier.
    name: str = ""
    #: Concrete area of the bound zone [mm²] (``None`` until bound).
    A_c = None
    #: Perimeter of the bound zone exposed to drying [mm].
    u = None
    #: Characteristic strength of the bound prestressing steel [MPa].
    f_pk = None

    # ------------------------------------------------------------------
    #  The four material functions
    # ------------------------------------------------------------------

    @abstractmethod
    def J(self, t, t_prime):
        r"""
        Creep compliance :math:`J(t, t')` [1/MPa] — total
        (instantaneous + delayed) strain at age *t* per unit stress
        sustained since *t'*.

        The instantaneous response is *encoded here*, not supplied
        separately: :math:`E_c(t') = 1 / J(t', t')`.

        An EN 1992-1-1 provider returns
        :math:`1/E_{cm}(t') + \varphi(t,t')/E_{cm}(28)`; an ACI 209R
        provider :math:`[1 + \varphi(t,t')]/E_c(t')`; a B4 provider a
        genuine multi-term compliance.  **The container never assumes
        the** :math:`[1+\varphi]/E_c` **form.**

        Parameters
        ----------
        t : float
            Age at observation [days], ``>= t_prime``.
        t_prime : float
            Age at loading [days].

        Returns
        -------
        float
            Compliance [1/MPa].
        """

    @abstractmethod
    def eps_imposed(self, t, t_s):
        r"""
        Stress-independent imposed strain
        :math:`\varepsilon_{\mathrm{imp}}(t, t_s)` [-], **signed**
        (negative for shrinkage).

        The container sees only the *total*; the split — EC2's drying +
        autogenous, ACI 209R's single curve, B4's explicit autogenous
        term — is internal to the provider.

        Parameters
        ----------
        t : float
            Age at observation [days].
        t_s : float
            Age at the end of curing / start of drying [days].

        Returns
        -------
        float
            Imposed strain [-].
        """

    @abstractmethod
    def relaxation(self, t, mu):
        r"""
        **Intrinsic** steel relaxation :math:`\Delta\sigma_{pr}(t,\mu)`
        [MPa], **signed** (negative).

        Stress decay of a tendon held at **constant strain**, with
        :math:`\mu = \sigma_{pi}/f_{pk}` the initial stress ratio.

        *Intrinsic* is load-bearing: the reduction caused by the tendon
        shortening with the concrete — the lump factor 0.8 of
        EN 1992-1-1 (5.46) — is the **container's** coupled equilibrium
        and is applied there
        (:attr:`~gensec.solver.losses.LossModel.relaxation_reduction`).
        It must not be baked in here.

        Parameters
        ----------
        t : float
            Duration under load [days] — not an age: relaxation is
            measured from stressing.
        mu : float
            :math:`\sigma_{pi}/f_{pk}` [-].

        Returns
        -------
        float
            Stress change [MPa], negative.
        """

    @abstractmethod
    def E_c(self, t):
        r"""
        Concrete modulus at age *t* [MPa].

        Consistent with :meth:`J` by the identity
        :math:`E_c(t) = 1/J(t,t)`; implementations may return the
        closed form directly, and the validator asserts the identity.

        Parameters
        ----------
        t : float
            Age [days].

        Returns
        -------
        float
            Modulus [MPa].
        """

    # ------------------------------------------------------------------
    #  Derived — provider-agnostic, never overridden by a norm
    # ------------------------------------------------------------------

    def phi(self, t, t_prime):
        r"""
        Generalized creep coefficient
        :math:`\varphi(t,t') = E_c(t')\,J(t,t') - 1` [-], derived from
        :meth:`J`.

        Parameters
        ----------
        t, t_prime : float
            Ages [days].

        Returns
        -------
        float
        """
        return self.E_c(t_prime) * self.J(t, t_prime) - 1.0

    def linearity_limit(self, t):
        r"""
        Compressive stress magnitude [MPa, positive] above which the
        provider's compliance ceases to be stress-independent, or
        ``None`` if the provider does not declare one.

        Linear viscoelasticity — the entire :math:`J`/:math:`\chi`/AAEM
        machinery — is valid only below this stress.  EN 1992-1-1
        §3.1.4(4) puts it at :math:`0.45\, f_{ck}(t_0)`, beyond which a
        *non-linear* creep coefficient is required.  The container
        checks the concrete stress against this value and **raises**
        rather than apply linear creep outside its range: a silent
        extrapolation here is a silent mismodel.

        Parameters
        ----------
        t : float
            Age at loading [days].

        Returns
        -------
        float or None
            Limit stress [MPa, positive magnitude].
        """
        return None

    # ------------------------------------------------------------------
    #  Late binding — geometry (per zone) and steel (per tendon)
    # ------------------------------------------------------------------

    def with_geometry(self, A_c, u):
        r"""
        Return a **copy** bound to the drying geometry
        :math:`(A_c, u)`.

        The receiver is left unchanged: one provider instance is shared
        across the concrete zones that use the same mix, and each zone
        binds its own exposed perimeter.

        Parameters
        ----------
        A_c : float
            Concrete area of the zone [mm²], ``> 0``.
        u : float
            Perimeter of the zone exposed to drying [mm], ``> 0``.

        Returns
        -------
        RheologicalModel

        Raises
        ------
        ValueError
            Non-positive area or perimeter.
        """
        A_c = float(A_c)
        u = float(u)
        if A_c <= 0.0 or u <= 0.0:
            raise ValueError(
                f"{type(self).__name__}.with_geometry: A_c ({A_c}) and u "
                f"({u}) must both be positive.  A zone with no exposed "
                f"perimeter cannot dry — model it with an explicit "
                f"zero-shrinkage provider rather than u = 0."
            )
        out = copy.copy(self)
        out.A_c = A_c
        out.u = u
        return out

    def with_steel(self, f_pk, **kwargs):
        r"""
        Return a **copy** bound to a prestressing steel.

        Parameters
        ----------
        f_pk : float
            Characteristic tensile strength of the tendon [MPa].
        **kwargs
            Provider-specific overrides (relaxation class, ρ₁₀₀₀, …).

        Returns
        -------
        RheologicalModel
        """
        out = copy.copy(self)
        out.f_pk = float(f_pk)
        for k, v in kwargs.items():
            if v is not None:
                setattr(out, k, v)
        return out

    def _require_geometry(self, what):
        r"""Fail loud on an unbound drying geometry."""
        if self.A_c is None or self.u is None:
            raise ValueError(
                f"{type(self).__name__}.{what}: the drying geometry "
                f"(A_c, u) is unbound.  The container binds it per "
                f"concrete zone with with_geometry(A_c, u); a provider "
                f"used bare must be given the geometry at construction."
            )

    def _require_steel(self, what):
        r"""Fail loud on an unbound prestressing steel."""
        if self.f_pk is None:
            raise ValueError(
                f"{type(self).__name__}.{what}: f_pk is unbound.  The "
                f"container binds the prestressing steel per tendon with "
                f"with_steel(f_pk); a provider used bare must be given "
                f"it explicitly."
            )


def relaxation_function(model, t, t0, n_steps=100):
    r"""
    Relaxation function :math:`R(t, t_0)` [MPa] from the provider's
    compliance, by numerical inversion of the Volterra equation.

    :math:`R(t,t_0)` is the stress at age *t* produced by a **unit
    strain** imposed at :math:`t_0` and held constant — the solution of

    .. math::

        \int_{t_0}^{t} J(t, \tau)\,
        \frac{\partial R(\tau, t_0)}{\partial \tau}\,\mathrm{d}\tau
        \;+\; J(t, t_0)\, R(t_0, t_0) \;=\; 1 ,
        \qquad R(t_0,t_0) = \frac{1}{J(t_0,t_0)} = E_c(t_0) .

    Discretised with the classical step-by-step scheme (Bažant 1972):
    a **logarithmic** grid :math:`t_0 = \tau_0 < \dots < \tau_n = t`,
    each stress increment applied at the geometric mid-step
    :math:`\bar\tau_k = \sqrt{\tau_{k-1}\tau_k}`, the unit-strain
    condition enforced at every node,

    .. math::

        \sigma_0\, J(\tau_i, t_0)
        + \sum_{k=1}^{i} \Delta\sigma_k\, J(\tau_i, \bar\tau_k) = 1 ,

    which is lower-triangular and solves forward in a single pass:

    .. math::

        \Delta\sigma_i = \frac{1 - \sigma_0 J(\tau_i, t_0)
            - \sum_{k<i} \Delta\sigma_k\, J(\tau_i, \bar\tau_k)}
            {J(\tau_i, \bar\tau_i)} .

    Parameters
    ----------
    model : RheologicalModel
        The provider.  Only :meth:`~RheologicalModel.J` is used.
    t : float
        Age at observation [days].
    t0 : float
        Age at which the unit strain is imposed [days].
    n_steps : int, optional
        Log-grid resolution.  Default 100.

    Returns
    -------
    float
        :math:`R(t,t_0)` [MPa].

    Raises
    ------
    ValueError
        ``t < t0``, or ``n_steps < 1``.

    Notes
    -----
    Pure mechanics: **no normative constant appears in this function.**
    It is the machine that makes :func:`aging_coefficient` — hence the
    whole container — normative-agnostic.
    """
    t = float(t)
    t0 = float(t0)
    if t < t0:
        raise ValueError(
            f"relaxation_function: t ({t}) must be >= t0 ({t0})."
        )
    if int(n_steps) < 1:
        raise ValueError("relaxation_function: n_steps must be >= 1.")
    sigma_0 = 1.0 / model.J(t0, t0)
    if t == t0:
        return sigma_0

    n = int(n_steps)
    frac = np.logspace(-4.0, 0.0, n + 1)
    frac = (frac - frac[0]) / (frac[-1] - frac[0])
    nodes = t0 + frac * (t - t0)
    nodes[0] = t0

    d_sigma = np.zeros(n, dtype=float)
    tau_bar = np.sqrt(nodes[:-1] * nodes[1:])
    for i in range(1, n + 1):
        ti = nodes[i]
        acc = sigma_0 * model.J(ti, t0)
        for k in range(1, i):
            acc += d_sigma[k - 1] * model.J(ti, tau_bar[k - 1])
        d_sigma[i - 1] = (1.0 - acc) / model.J(ti, tau_bar[i - 1])
    return float(sigma_0 + d_sigma.sum())


def aging_coefficient(model, t, t0, n_steps=100):
    r"""
    Age-adjusted **ageing coefficient** :math:`\chi(t, t_0)` [-],
    computed *from* the provider's compliance — never supplied as an
    input.

    Obtained by relaxing a unit imposed strain on
    :meth:`~RheologicalModel.J` (:func:`relaxation_function`) and
    inverting the AAEM identity:

    .. math::

        \chi(t,t_0) \;=\; \frac{E_c(t_0)}{E_c(t_0) - R(t,t_0)}
                    \;-\; \frac{1}{\varphi(t,t_0)} ,

    with :math:`\varphi(t,t_0) = E_c(t_0) J(t,t_0) - 1`
    (:meth:`~RheologicalModel.phi`).

    **This is the cardinal test of normative agnosticism.**  The
    :math:`0.8` that EN 1992-1-1 (5.46) writes twice is *what EC2's own
    compliance produces*, not a datum: an EC2 provider at
    :math:`t_0 = 28` d, :math:`t \to \infty` returns
    :math:`\chi \approx 0.80` (0.76–0.86 over
    :math:`f_{ck} \in [25,60]`, :math:`h_0 \in [170,570]` mm).  An
    ACI 209R provider returns ACI's own value (:math:`\approx 0.84` on
    the same section); a B4 provider, B4's.  A container that took
    :math:`\chi` as an *argument* would have cemented the Eurocode
    approximation into what is meant to be pure mechanics.

    Parameters
    ----------
    model : RheologicalModel
        The provider.  Only ``J`` (through ``E_c`` and ``phi``) is used.
    t : float
        Age at observation [days].
    t0 : float
        Age at loading [days].
    n_steps : int, optional
        Log-grid resolution of the Volterra inversion.  Default 100,
        for which :math:`\chi` is converged to :math:`1.4\times10^{-4}`
        (verified by the grid-convergence check in
        ``run_phase5_c5_validation.py``).

    Returns
    -------
    float
        Ageing coefficient :math:`\chi` [-], typically in
        :math:`[0.5, 1.0]`.

    Raises
    ------
    ValueError
        ``t <= t0`` (:math:`\varphi \to 0` and the :math:`1/\varphi`
        term diverges), or a provider reporting no creep at all.
    """
    t = float(t)
    t0 = float(t0)
    if t <= t0:
        raise ValueError(
            f"aging_coefficient: t ({t}) must be > t0 ({t0}); chi is "
            f"undefined at zero elapsed time (phi -> 0, so the 1/phi "
            f"term diverges)."
        )
    E0 = model.E_c(t0)
    phi = model.phi(t, t0)
    if abs(phi) < 1e-12:
        raise ValueError(
            f"aging_coefficient: the provider reports a vanishing creep "
            f"coefficient phi({t}, {t0}) = {phi:.3e}.  chi is undefined "
            f"for a non-creeping material — use the elastic modulus "
            f"directly."
        )
    R = relaxation_function(model, t, t0, n_steps=n_steps)
    return float(E0 / (E0 - R) - 1.0 / phi)
