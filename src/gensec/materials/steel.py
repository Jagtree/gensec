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
