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
Unbounded linear-elastic constitutive law.

This module provides :class:`LinearElastic`, the constitutive law used
by the SLS verification layer (:mod:`gensec.solver.sls`) to build the
*SLS view* of a section: every design-level law (parabola-rectangle
concrete, elastic-plastic steel, prestressing steel) is replaced by a
linear law

.. math::

    \sigma(\varepsilon) = E\,\varepsilon
    \qquad \forall\, \varepsilon \in \mathbb{R},

so that the fiber equilibrium solve on the substituted section is the
**uncracked, homogenized elastic** solution — the classical
transformed-section state — while reusing the exact same integration
machinery (initial-strain offsets, embedded-area subtraction,
multi-material zones) as the ULS path.

The law is *unbounded*: it never caps, never crushes, and carries
tension symmetrically.  This is deliberate — the uncracked SLS basis is
linear by definition, and its *validity* (tensile stress below the
effective tensile strength) is assessed downstream as a verification
verdict, never inside the constitutive law (see
``uncracked_basis_violated`` in :mod:`gensec.solver.sls`).

Notes
-----
:attr:`LinearElastic.eps_min` / :attr:`LinearElastic.eps_max` report a
**finite** admissible range (``±eps_lim``) even though the stress
function itself is unbounded.  The limits exist only to keep the
solver's scanning machinery well-posed (the pure-axial fast path of
:meth:`~gensec.solver.integrator.FiberSolver.solve_equilibrium` builds
its bracketing grid from the material strain range; an infinite range
would break it).  SLS strains are orders of magnitude below the default
``eps_lim``.
"""

from dataclasses import dataclass

import numpy as np

from .base import Material


@dataclass
class LinearElastic(Material):
    r"""
    Unbounded symmetric linear-elastic law :math:`\sigma = E\varepsilon`.

    Parameters
    ----------
    E : float
        Elastic modulus [MPa].  Must be finite and strictly positive
        (a zero or negative modulus would silently produce a
        zero-stiffness or non-physical fiber — fail loud instead).
    eps_lim : float, optional
        Reported admissible strain magnitude for :attr:`eps_min` /
        :attr:`eps_max`.  Default ``0.05``.  Purely a scan-range hint
        for the solver's bracketing machinery; the stress function is
        linear for **all** strains, with no cutoff at ``±eps_lim``.
    name : str, optional
        Human-readable identifier (e.g. ``"SLS(C30/37)"``).

    Raises
    ------
    ValueError
        If ``E`` is not finite and strictly positive, or if
        ``eps_lim`` is not strictly positive.

    Examples
    --------
    >>> law = LinearElastic(E=33000.0, name="SLS(C30/37)")
    >>> law.stress(-1.0e-4)
    -3.3
    >>> law.tangent(0.0)
    33000.0
    """

    E: float = 0.0
    eps_lim: float = 0.05
    name: str = ""

    def __post_init__(self):
        if not np.isfinite(self.E) or self.E <= 0.0:
            raise ValueError(
                f"LinearElastic requires a finite, strictly positive "
                f"modulus; got E={self.E!r}."
            )
        if not np.isfinite(self.eps_lim) or self.eps_lim <= 0.0:
            raise ValueError(
                f"LinearElastic requires a finite, strictly positive "
                f"eps_lim; got eps_lim={self.eps_lim!r}."
            )

    # ------------------------------------------------------------------
    #  Strain range (scan hints — see module notes)
    # ------------------------------------------------------------------

    @property
    def eps_min(self):
        r"""Most compressive reported strain, :math:`-\varepsilon_{\lim}`."""
        return -self.eps_lim

    @property
    def eps_max(self):
        r"""Most tensile reported strain, :math:`+\varepsilon_{\lim}`."""
        return self.eps_lim

    # ------------------------------------------------------------------
    #  Scalar interface
    # ------------------------------------------------------------------

    def stress(self, eps):
        r"""
        Scalar stress :math:`\sigma = E\,\varepsilon`.

        Parameters
        ----------
        eps : float
            Strain [-].

        Returns
        -------
        float
            Stress [MPa].
        """
        return self.E * eps

    def tangent(self, eps):
        r"""
        Scalar tangent modulus (constant, :math:`E_t = E`).

        Parameters
        ----------
        eps : float
            Strain [-] (unused; the tangent is constant).

        Returns
        -------
        float
            Tangent modulus [MPa].
        """
        return self.E

    # ------------------------------------------------------------------
    #  Vectorized interface
    # ------------------------------------------------------------------

    def stress_array(self, eps):
        r"""
        Vectorized stress: :math:`\sigma_i = E\,\varepsilon_i`.

        Parameters
        ----------
        eps : numpy.ndarray
            Strain array of arbitrary shape.

        Returns
        -------
        numpy.ndarray
            Stress array, same shape as ``eps`` [MPa].
        """
        return self.E * np.asarray(eps, dtype=float)

    def tangent_array(self, eps):
        r"""
        Vectorized tangent modulus (constant array).

        Parameters
        ----------
        eps : numpy.ndarray
            Strain array of arbitrary shape.

        Returns
        -------
        numpy.ndarray
            Array filled with ``E``, same shape as ``eps`` [MPa].
        """
        eps = np.asarray(eps, dtype=float)
        return np.full_like(eps, self.E)
