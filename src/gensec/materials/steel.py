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
Reinforcing steel constitutive law — elastic-plastic with optional hardening.

When `numba <https://numba.pydata.org>`_ is installed, the
element-wise stress and tangent kernels are JIT-compiled to native
code.  If *numba* is not available the pure-NumPy path is used
transparently.
"""

import numpy as np
from dataclasses import dataclass
from .base import Material

# ------------------------------------------------------------------
#  Optional Numba acceleration
# ------------------------------------------------------------------
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):           # noqa: E303
        """No-op decorator when Numba is absent."""
        def _passthrough(fn):
            return fn
        if args and callable(args[0]):
            return args[0]
        return _passthrough


@njit(cache=True)
def _steel_stress_kernel(eps_flat, Es, fyd, ftd, eps_yd, eps_su,
                         works_in_compression):
    r"""
    Numba-accelerated stress kernel for elastic-plastic steel.

    Parameters
    ----------
    eps_flat : numpy.ndarray
        1-D strain array.
    Es : float
        Young's modulus [MPa].
    fyd : float
        Design yield strength [MPa].
    ftd : float
        Design ultimate strength [MPa].
    eps_yd : float
        Yield strain (positive).
    eps_su : float
        Ultimate strain (positive).
    works_in_compression : bool
        If False, compressive stress is zero.

    Returns
    -------
    numpy.ndarray
        1-D stress array.
    """
    out = np.zeros(eps_flat.shape[0], dtype=np.float64)
    hard_range = eps_su - eps_yd
    for i in range(eps_flat.shape[0]):
        e = eps_flat[i]
        if e >= 0.0:
            if e <= eps_yd:
                out[i] = Es * e
            elif e <= eps_su:
                out[i] = fyd + (ftd - fyd) * (e - eps_yd) / hard_range
            # else: 0 (rupture)
        elif works_in_compression:
            ea = -e
            if ea <= eps_yd:
                out[i] = Es * e
            elif ea <= eps_su:
                out[i] = -(fyd + (ftd - fyd) * (ea - eps_yd) / hard_range)
            # else: 0
    return out


@njit(cache=True)
def _steel_tangent_kernel(eps_flat, Es, fyd, ftd, eps_yd, eps_su,
                          works_in_compression):
    r"""
    Numba-accelerated tangent-modulus kernel for steel.

    .. math::

        E_t(\varepsilon) =
        \begin{cases}
            E_s & |\varepsilon| \le \varepsilon_{yd} \\
            \dfrac{f_{td} - f_{yd}}
                  {\varepsilon_{su} - \varepsilon_{yd}}
                & \varepsilon_{yd} < |\varepsilon| \le \varepsilon_{su} \\
            0 & |\varepsilon| > \varepsilon_{su}
        \end{cases}

    Parameters
    ----------
    eps_flat : numpy.ndarray
        1-D strain array.
    Es, fyd, ftd, eps_yd, eps_su : float
    works_in_compression : bool

    Returns
    -------
    numpy.ndarray
    """
    out = np.zeros(eps_flat.shape[0], dtype=np.float64)
    E_hard = (ftd - fyd) / (eps_su - eps_yd) if eps_su > eps_yd else 0.0
    for i in range(eps_flat.shape[0]):
        e = eps_flat[i]
        ea = abs(e)
        if e >= 0.0:
            if ea <= eps_yd:
                out[i] = Es
            elif ea <= eps_su:
                out[i] = E_hard
        elif works_in_compression:
            if ea <= eps_yd:
                out[i] = Es
            elif ea <= eps_su:
                out[i] = E_hard
    return out


# ------------------------------------------------------------------
#  Steel dataclass
# ------------------------------------------------------------------

@dataclass
class Steel(Material):
    r"""
    Elastic-plastic steel with optional linear hardening.

    .. math::

        \sigma_s(\varepsilon) =
        \begin{cases}
            E_s\,\varepsilon
                & |\varepsilon| \le \varepsilon_{yd} \\
            \mathrm{sign}(\varepsilon)\!\left[f_{yd}
                + (f_{td}-f_{yd})
                \dfrac{|\varepsilon|-\varepsilon_{yd}}
                      {\varepsilon_{su}-\varepsilon_{yd}}\right]
                & \varepsilon_{yd} < |\varepsilon| \le \varepsilon_{su} \\
            0 & |\varepsilon| > \varepsilon_{su}
        \end{cases}

    Parameters
    ----------
    fyk : float
        Characteristic yield strength [MPa].
    gamma_s : float, optional
        Partial safety factor. Default 1.15.
    Es : float, optional
        Young's modulus [MPa]. Default 200000.
    k_hardening : float, optional
        :math:`f_t/f_y` ratio. Default 1.0 (perfectly plastic).
    eps_su : float, optional
        Ultimate strain. Default 0.01.
    works_in_compression : bool, optional
        If False, compressive stress is zero. Default True.

    Attributes
    ----------
    fyd : float
        Design yield strength [MPa].
    ftd : float
        Design ultimate strength [MPa].
    eps_yd : float
        Yield strain.
    """

    fyk: float = 450.0
    gamma_s: float = 1.15
    Es: float = 200000.0
    k_hardening: float = 1.0
    eps_su: float = 0.01
    works_in_compression: bool = True

    def __post_init__(self):
        self.fyd = self.fyk / self.gamma_s
        self.ftd = self.fyd * self.k_hardening
        self.eps_yd = self.fyd / self.Es

    @property
    def eps_min(self):
        return -self.eps_su if self.works_in_compression else 0.0

    @property
    def eps_max(self):
        return self.eps_su

    # ------------------------------------------------------------------
    #  Scalar interface
    # ------------------------------------------------------------------

    def stress(self, eps):
        if eps >= 0:
            if eps <= self.eps_yd:
                return self.Es * eps
            elif eps <= self.eps_su:
                return self.fyd + (self.ftd - self.fyd) * (
                    (eps - self.eps_yd) / (self.eps_su - self.eps_yd))
            return 0.0
        if not self.works_in_compression:
            return 0.0
        ea = abs(eps)
        if ea <= self.eps_yd:
            return self.Es * eps
        elif ea <= self.eps_su:
            return -(self.fyd + (self.ftd - self.fyd) * (
                (ea - self.eps_yd) / (self.eps_su - self.eps_yd)))
        return 0.0

    def tangent(self, eps):
        r"""
        Scalar tangent modulus :math:`E_t = d\sigma_s / d\varepsilon`.

        Returns
        -------
        float
        """
        ea = abs(eps)
        if eps >= 0.0 or self.works_in_compression:
            if ea <= self.eps_yd:
                return self.Es
            if ea <= self.eps_su:
                return ((self.ftd - self.fyd)
                        / (self.eps_su - self.eps_yd))
        return 0.0

    # ------------------------------------------------------------------
    #  Vectorized interface (any-shape, Numba-accelerated when available)
    # ------------------------------------------------------------------

    def stress_array(self, eps):
        r"""
        Vectorized stress computation.

        Accepts arrays of **any shape**.  When *numba* is installed,
        the inner loop is JIT-compiled.

        Parameters
        ----------
        eps : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        if _HAS_NUMBA:
            flat = np.ascontiguousarray(eps.ravel(), dtype=np.float64)
            return _steel_stress_kernel(
                flat, self.Es, self.fyd, self.ftd,
                self.eps_yd, self.eps_su, self.works_in_compression,
            ).reshape(eps.shape)

        # Pure-NumPy fallback
        sigma = np.zeros_like(eps, dtype=np.float64)
        m_te = (eps >= 0) & (eps <= self.eps_yd)
        sigma[m_te] = self.Es * eps[m_te]
        m_tp = (eps > self.eps_yd) & (eps <= self.eps_su)
        sigma[m_tp] = self.fyd + (self.ftd - self.fyd) * (
            (eps[m_tp] - self.eps_yd) / (self.eps_su - self.eps_yd))
        if self.works_in_compression:
            ea = np.abs(eps)
            m_ce = (eps < 0) & (ea <= self.eps_yd)
            sigma[m_ce] = self.Es * eps[m_ce]
            m_cp = (eps < 0) & (ea > self.eps_yd) & (ea <= self.eps_su)
            sigma[m_cp] = -(self.fyd + (self.ftd - self.fyd) * (
                (ea[m_cp] - self.eps_yd)
                / (self.eps_su - self.eps_yd)))
        return sigma

    def tangent_array(self, eps):
        r"""
        Vectorized tangent modulus :math:`E_t = d\sigma_s / d\varepsilon`.

        .. math::

            E_t(\varepsilon) =
            \begin{cases}
                E_s & |\varepsilon| \le \varepsilon_{yd} \\
                \dfrac{f_{td} - f_{yd}}
                      {\varepsilon_{su} - \varepsilon_{yd}}
                & \varepsilon_{yd} < |\varepsilon|
                  \le \varepsilon_{su} \\
                0 & \text{otherwise}
            \end{cases}

        Parameters
        ----------
        eps : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        if _HAS_NUMBA:
            flat = np.ascontiguousarray(eps.ravel(), dtype=np.float64)
            return _steel_tangent_kernel(
                flat, self.Es, self.fyd, self.ftd,
                self.eps_yd, self.eps_su, self.works_in_compression,
            ).reshape(eps.shape)

        # Pure-NumPy fallback
        Et = np.zeros_like(eps, dtype=np.float64)
        ea = np.abs(eps)
        E_hard = ((self.ftd - self.fyd)
                  / (self.eps_su - self.eps_yd)
                  if self.eps_su > self.eps_yd else 0.0)

        # Tension or symmetric compression
        if self.works_in_compression:
            m_e = ea <= self.eps_yd
            m_p = (ea > self.eps_yd) & (ea <= self.eps_su)
        else:
            m_e = (eps >= 0) & (ea <= self.eps_yd)
            m_p = (eps >= 0) & (ea > self.eps_yd) & (ea <= self.eps_su)

        Et[m_e] = self.Es
        Et[m_p] = E_hard
        return Et


# ------------------------------------------------------------------
#  Prestressing steel — EC2 §3.3 generic bilinear diagram
# ------------------------------------------------------------------

@njit(cache=True)
def _pssteel_stress_kernel(eps_flat, Ep, eps_el, sig_el, eps_ud,
                           E_hard, works_in_compression):
    r"""
    Numba-accelerated stress kernel for the generic bilinear
    prestressing-steel diagram (EC2 §3.3.6).

    The diagram is fully described by four numbers and a hardening
    slope, leaving the *idealization* (horizontal vs. inclined top
    branch) entirely to the caller:

    - first (elastic) branch of slope ``Ep`` up to the proof point
      :math:`(\varepsilon_{el}, \sigma_{el})`;
    - second branch of slope ``E_hard`` up to the design ultimate
      strain :math:`\varepsilon_{ud}`;
    - zero beyond :math:`\varepsilon_{ud}` (rupture).

    The law is **symmetric**: the same shape is mirrored in
    compression so that a generic locked-in initial strain remains
    meaningful in either sign, consistent with treating the
    prestrain as a material-agnostic offset applied by the solver.

    Parameters
    ----------
    eps_flat : numpy.ndarray
        1-D **total** strain array (section strain + initial
        prestrain already added by the caller).
    Ep : float
        Modulus of elasticity of the prestressing steel [MPa].
    eps_el : float
        Strain at the end of the elastic branch (positive),
        :math:`\varepsilon_{el} = \sigma_{el}/E_p`.
    sig_el : float
        Stress at the end of the elastic branch (positive), i.e. the
        design proof stress :math:`f_{p0.1d}`.
    eps_ud : float
        Design ultimate strain (positive).
    E_hard : float
        Slope of the second branch [MPa].  Zero reproduces the
        horizontal-top idealization; a positive value reproduces the
        inclined branch towards :math:`f_{pd}` (or :math:`f_{pk}`).
    works_in_compression : bool
        If ``False``, compressive stress is zero.

    Returns
    -------
    numpy.ndarray
        1-D stress array [MPa].
    """
    out = np.zeros(eps_flat.shape[0], dtype=np.float64)
    for i in range(eps_flat.shape[0]):
        e = eps_flat[i]
        if e >= 0.0:
            if e <= eps_el:
                out[i] = Ep * e
            elif e <= eps_ud:
                out[i] = sig_el + E_hard * (e - eps_el)
            # else: 0 (rupture)
        elif works_in_compression:
            ea = -e
            if ea <= eps_el:
                out[i] = Ep * e
            elif ea <= eps_ud:
                out[i] = -(sig_el + E_hard * (ea - eps_el))
            # else: 0
    return out


@njit(cache=True)
def _pssteel_tangent_kernel(eps_flat, Ep, eps_el, eps_ud,
                            E_hard, works_in_compression):
    r"""
    Numba-accelerated tangent-modulus kernel for prestressing steel.

    .. math::

        E_t(\varepsilon) =
        \begin{cases}
            E_p & |\varepsilon| \le \varepsilon_{el} \\
            E_{\text{hard}}
                & \varepsilon_{el} < |\varepsilon| \le \varepsilon_{ud} \\
            0 & |\varepsilon| > \varepsilon_{ud}
        \end{cases}

    Parameters
    ----------
    eps_flat : numpy.ndarray
        1-D total strain array.
    Ep, eps_el, eps_ud, E_hard : float
    works_in_compression : bool

    Returns
    -------
    numpy.ndarray
        1-D tangent-modulus array [MPa].
    """
    out = np.zeros(eps_flat.shape[0], dtype=np.float64)
    for i in range(eps_flat.shape[0]):
        e = eps_flat[i]
        ea = abs(e)
        if e >= 0.0 or works_in_compression:
            if ea <= eps_el:
                out[i] = Ep
            elif ea <= eps_ud:
                out[i] = E_hard
    return out


@dataclass
class PrestressingSteel(Material):
    r"""
    Prestressing steel — generic bilinear design diagram (EC2 §3.3.6).

    This material implements the EC2 idealized stress-strain diagram
    for prestressing steel as a **generic** bilinear law, deliberately
    not committing to a single national-annex idealization.  The
    diagram is:

    .. math::

        \sigma_p(\varepsilon) =
        \begin{cases}
            E_p\,\varepsilon
                & |\varepsilon| \le \varepsilon_{el} \\[4pt]
            \operatorname{sign}(\varepsilon)\!\left[
                f_{p0.1d} + E_{\text{hard}}
                \bigl(|\varepsilon| - \varepsilon_{el}\bigr)\right]
                & \varepsilon_{el} < |\varepsilon| \le \varepsilon_{ud} \\[4pt]
            0 & |\varepsilon| > \varepsilon_{ud}
        \end{cases}

    where :math:`\varepsilon_{el} = f_{p0.1d}/E_p` is the strain at
    the proof point and

    .. math::

        E_{\text{hard}} =
        \frac{\sigma_{ud} - f_{p0.1d}}
             {\varepsilon_{ud} - \varepsilon_{el}}

    is the second-branch slope.  Choosing the second-branch endpoint
    stress :math:`\sigma_{ud}` recovers every common idealization:

    - :math:`\sigma_{ud} = f_{p0.1d}` → **horizontal** top branch
      (:math:`E_{\text{hard}} = 0`);
    - :math:`\sigma_{ud} = f_{pd}` → **inclined** branch towards the
      design tensile strength;
    - :math:`\sigma_{ud} = f_{pk}` → inclined branch towards the
      characteristic tensile strength (a characteristic-level diagram,
      :math:`\gamma_S = 1`).

    This material is **constitutive-only**: it knows nothing about
    prestrain, prestressing systems, bond, or losses.  The locked-in
    prestrain of a tendon is supplied by the geometry/solver layer as
    a per-element initial-strain offset (see
    :class:`~gensec.geometry.fiber.Tendon`), so that the same diagram
    can also serve generic initial-strain problems (shrinkage,
    thermal) without modification.

    Design values are expected to be supplied **already factored**.
    The mapping from characteristic strengths and a chosen limit
    state / national annex to design values lives in
    :mod:`~gensec.materials.ec2_bridge`
    (see :func:`~gensec.materials.ec2_bridge.prestress_from_ec2`),
    mirroring the concrete and structural-steel bridges.

    Parameters
    ----------
    f_p01d : float
        Design proof stress :math:`f_{p0.1d}` [MPa] (the 0.1 % proof
        stress divided by :math:`\gamma_S`).  End of the elastic
        branch.
    sigma_ud : float, optional
        Stress [MPa] at the design ultimate strain
        :math:`\varepsilon_{ud}`.  Sets the second-branch slope.
        Default equals ``f_p01d`` (horizontal top branch).
    eps_ud : float, optional
        Design ultimate strain (positive).  Default 0.02.
    Ep : float, optional
        Modulus of elasticity [MPa].  Default 195000 (EC2 §3.3.6(3)
        nominal value for strand; use 205000 for wires and bars).
    works_in_compression : bool, optional
        If ``False``, the compressive branch returns zero.  Default
        ``True`` (symmetric diagram).

    Attributes
    ----------
    eps_el : float
        Strain at the end of the elastic branch,
        :math:`f_{p0.1d}/E_p`.
    E_hard : float
        Second-branch slope [MPa].

    Notes
    -----
    The EC2 §3.3.6 recommended design ultimate strain is
    :math:`\varepsilon_{ud} = 0.9\,\varepsilon_{uk}`; this material
    takes :math:`\varepsilon_{ud}` directly so that a national annex
    omitting the reduction is reachable by input.

    Examples
    --------
    Inclined-branch diagram for a Y1860S7 strand, fundamental ULS
    (:math:`\gamma_S = 1.15`), second branch towards :math:`f_{pk}`:

    >>> ps = PrestressingSteel(
    ...     f_p01d=1600.0 / 1.15,
    ...     sigma_ud=1860.0,            # towards f_pk (example)
    ...     eps_ud=0.9 * 0.035,
    ...     Ep=195000.0,
    ... )
    >>> round(ps.eps_el, 5)
    0.00714
    """

    f_p01d: float = 1391.3            # e.g. 1600/1.15
    sigma_ud: float = 0.0             # 0 → set to f_p01d in __post_init__
    eps_ud: float = 0.02
    Ep: float = 195000.0
    works_in_compression: bool = True

    def __post_init__(self):
        if self.sigma_ud <= 0.0:
            # Default to the horizontal-top idealization.
            self.sigma_ud = self.f_p01d
        self.eps_el = self.f_p01d / self.Ep
        denom = self.eps_ud - self.eps_el
        self.E_hard = ((self.sigma_ud - self.f_p01d) / denom
                       if denom > 0.0 else 0.0)

    @property
    def E(self):
        r"""Elastic modulus :math:`E_p` [MPa].

        Exposed under the common ``E`` name used by the homogenization
        machinery and by sections building lever-arm-weighted moduli.
        """
        return self.Ep

    @property
    def eps_min(self):
        """Most compressive admissible strain."""
        return -self.eps_ud if self.works_in_compression else 0.0

    @property
    def eps_max(self):
        """Most tensile admissible strain."""
        return self.eps_ud

    # ------------------------------------------------------------------
    #  Scalar interface
    # ------------------------------------------------------------------

    def stress(self, eps):
        r"""
        Evaluate stress for a single **total** strain value.

        Parameters
        ----------
        eps : float
            Total strain (section strain + locked-in prestrain).

        Returns
        -------
        float
            Stress [MPa].
        """
        if eps >= 0.0:
            if eps <= self.eps_el:
                return self.Ep * eps
            if eps <= self.eps_ud:
                return self.f_p01d + self.E_hard * (eps - self.eps_el)
            return 0.0
        if not self.works_in_compression:
            return 0.0
        ea = -eps
        if ea <= self.eps_el:
            return self.Ep * eps
        if ea <= self.eps_ud:
            return -(self.f_p01d + self.E_hard * (ea - self.eps_el))
        return 0.0

    def tangent(self, eps):
        r"""
        Scalar tangent modulus :math:`E_t = d\sigma_p / d\varepsilon`.

        Returns
        -------
        float
        """
        ea = abs(eps)
        if eps >= 0.0 or self.works_in_compression:
            if ea <= self.eps_el:
                return self.Ep
            if ea <= self.eps_ud:
                return self.E_hard
        return 0.0

    # ------------------------------------------------------------------
    #  Vectorized interface (any-shape, Numba-accelerated when available)
    # ------------------------------------------------------------------

    def stress_array(self, eps):
        r"""
        Vectorized stress computation over **total** strains.

        Accepts arrays of **any shape**.  When *numba* is installed,
        the inner loop is JIT-compiled.

        Parameters
        ----------
        eps : numpy.ndarray
            Total strain array.

        Returns
        -------
        numpy.ndarray
            Stress array [MPa], same shape as *eps*.
        """
        if _HAS_NUMBA:
            flat = np.ascontiguousarray(eps.ravel(), dtype=np.float64)
            return _pssteel_stress_kernel(
                flat, self.Ep, self.eps_el, self.f_p01d, self.eps_ud,
                self.E_hard, self.works_in_compression,
            ).reshape(eps.shape)

        sigma = np.zeros_like(eps, dtype=np.float64)
        ea = np.abs(eps)
        sign = np.sign(eps)

        if self.works_in_compression:
            m_e = ea <= self.eps_el
            m_p = (ea > self.eps_el) & (ea <= self.eps_ud)
        else:
            m_e = (eps >= 0) & (ea <= self.eps_el)
            m_p = (eps >= 0) & (ea > self.eps_el) & (ea <= self.eps_ud)

        sigma[m_e] = self.Ep * eps[m_e]
        sigma[m_p] = sign[m_p] * (
            self.f_p01d + self.E_hard * (ea[m_p] - self.eps_el))
        return sigma

    def tangent_array(self, eps):
        r"""
        Vectorized tangent modulus :math:`E_t = d\sigma_p/d\varepsilon`.

        Parameters
        ----------
        eps : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Same shape as *eps*.
        """
        if _HAS_NUMBA:
            flat = np.ascontiguousarray(eps.ravel(), dtype=np.float64)
            return _pssteel_tangent_kernel(
                flat, self.Ep, self.eps_el, self.eps_ud,
                self.E_hard, self.works_in_compression,
            ).reshape(eps.shape)

        Et = np.zeros_like(eps, dtype=np.float64)
        ea = np.abs(eps)
        if self.works_in_compression:
            m_e = ea <= self.eps_el
            m_p = (ea > self.eps_el) & (ea <= self.eps_ud)
        else:
            m_e = (eps >= 0) & (ea <= self.eps_el)
            m_p = (eps >= 0) & (ea > self.eps_el) & (ea <= self.eps_ud)
        Et[m_e] = self.Ep
        Et[m_p] = self.E_hard
        return Et
