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
    """

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
