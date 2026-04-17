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
Tabulated material — arbitrary stress-strain curve from data points.

Piecewise-linear interpolation between tabulated
:math:`(\varepsilon, \sigma)` pairs.  The tangent modulus is the
slope of each linear segment.
"""

import numpy as np
from .base import Material


class TabulatedMaterial(Material):
    r"""
    Material defined by a tabulated :math:`(\varepsilon, \sigma)` curve.

    Piecewise-linear interpolation between data points. Strains
    outside the table range return zero stress (material failure).

    Parameters
    ----------
    strains : array_like
        Strictly increasing strain values.
    stresses : array_like
        Corresponding stress values [MPa].
    name : str, optional
        Label. Default ``"Tabulated"``.

    Raises
    ------
    ValueError
        Bad dimensions or non-monotonic strains.

    Examples
    --------
    >>> mat = TabulatedMaterial([-0.01, -0.002, 0, 0.002, 0.01],
    ...                        [-400, -400, 0, 400, 400], "EPP")
    >>> mat.stress(0.001)
    200.0
    """

    def __init__(self, strains, stresses, name="Tabulated"):
        strains = np.asarray(strains, dtype=float)
        stresses = np.asarray(stresses, dtype=float)
        if len(strains) != len(stresses):
            raise ValueError("strains/stresses length mismatch")
        if len(strains) < 2:
            raise ValueError("Need >= 2 points")
        if not np.all(np.diff(strains) > 0):
            raise ValueError("strains must be strictly increasing")
        self._strains = strains
        self._stresses = stresses
        self._name = name

        # Pre-compute segment slopes for tangent evaluation
        ds = np.diff(stresses)
        de = np.diff(strains)
        self._slopes = ds / de           # shape (n_segments,)
        self._seg_bounds = strains       # breakpoints (n_points,)

    @property
    def eps_min(self):
        return float(self._strains[0])

    @property
    def eps_max(self):
        return float(self._strains[-1])

    @property
    def name(self):
        """Material label."""
        return self._name

    # ------------------------------------------------------------------
    #  Scalar interface
    # ------------------------------------------------------------------

    def stress(self, eps):
        if eps < self._strains[0] or eps > self._strains[-1]:
            return 0.0
        return float(np.interp(eps, self._strains, self._stresses))

    def tangent(self, eps):
        r"""
        Scalar tangent modulus from the piecewise-linear table.

        Returns the slope of the linear segment containing *eps*,
        or 0 outside the table range.

        Parameters
        ----------
        eps : float

        Returns
        -------
        float
        """
        if eps < self._strains[0] or eps > self._strains[-1]:
            return 0.0
        idx = np.searchsorted(self._strains, eps, side='right') - 1
        idx = min(idx, len(self._slopes) - 1)
        return float(self._slopes[idx])

    # ------------------------------------------------------------------
    #  Vectorized interface (any-shape)
    # ------------------------------------------------------------------

    def stress_array(self, eps):
        r"""
        Vectorized piecewise-linear interpolation.

        Accepts arrays of **any shape** (1-D, 2-D, …).

        Parameters
        ----------
        eps : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Same shape as *eps*.
        """
        shape = eps.shape
        flat = eps.ravel()
        sigma = np.interp(flat, self._strains, self._stresses)
        sigma[flat < self._strains[0]] = 0.0
        sigma[flat > self._strains[-1]] = 0.0
        return sigma.reshape(shape)

    def tangent_array(self, eps):
        r"""
        Vectorized tangent modulus from the piecewise-linear table.

        For each strain value, returns the slope of the containing
        linear segment, or 0 outside the table range.

        Parameters
        ----------
        eps : numpy.ndarray

        Returns
        -------
        numpy.ndarray
            Same shape as *eps*.
        """
        shape = eps.shape
        flat = eps.ravel()
        # searchsorted gives the insertion index; subtract 1 to get
        # the segment index.  Clip to valid range.
        idx = np.searchsorted(self._strains, flat, side='right') - 1
        idx = np.clip(idx, 0, len(self._slopes) - 1)
        Et = self._slopes[idx].copy()
        # Zero outside the admissible range
        Et[flat < self._strains[0]] = 0.0
        Et[flat > self._strains[-1]] = 0.0
        return Et.reshape(shape)
