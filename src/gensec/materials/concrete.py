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
Concrete constitutive law — parabola-rectangle with optional tension.

The compression branch follows EC2 §3.1.7 (parabola-rectangle).
An optional linear-elastic tension branch can be activated for
serviceability checks (SLS) or nonlinear analyses by providing
the tensile strength :math:`f_{ct}` and the elastic modulus
:math:`E_c`.

When `numba <https://numba.pydata.org>`_ is installed, the
element-wise stress and tangent kernels are JIT-compiled to native
code, giving a ~2–3× speed-up on large fiber arrays.  If *numba* is
not available the pure-NumPy path is used transparently.
"""

import numpy as np
from dataclasses import dataclass
from .base import (Material, NormativeChoiceNotDeclared,
                   ReferenceModulus)

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
def _concrete_stress_kernel(eps_flat, fcd, eps_c2, eps_cu2, n,
                            tension_enabled, fct, Ec, eps_ct):
    r"""
    Numba-accelerated stress kernel for the parabola-rectangle law
    with optional linear tension branch.

    Operates on a **flat** (1-D) strain array and returns a flat
    stress array of the same length.

    Parameters
    ----------
    eps_flat : numpy.ndarray
        1-D strain array.
    fcd : float
        Design compressive strength [MPa] (positive).
    eps_c2 : float
        Peak-stress strain (negative).
    eps_cu2 : float
        Ultimate strain (negative,
        :math:`\varepsilon_{cu2} < \varepsilon_{c2}`).
    n : float
        Parabolic exponent.
    tension_enabled : bool
        Whether the linear tension branch is active.
    fct : float
        Tensile strength [MPa] (positive).  Ignored when
        ``tension_enabled`` is ``False``.
    Ec : float
        Elastic modulus [MPa] for the tension branch.
    eps_ct : float
        Cracking strain :math:`f_{ct}/E_c`.

    Returns
    -------
    numpy.ndarray
        1-D stress array.
    """
    out = np.zeros(eps_flat.shape[0], dtype=np.float64)
    for i in range(eps_flat.shape[0]):
        e = eps_flat[i]
        if e > 0.0:
            if tension_enabled and e <= eps_ct:
                out[i] = Ec * e
        elif e >= eps_c2:
            eta = e / eps_c2
            out[i] = -fcd * (1.0 - (1.0 - eta) ** n)
        elif e >= eps_cu2:
            out[i] = -fcd
        # else: 0 (beyond ultimate or above cracking) — already init
    return out


@njit(cache=True)
def _concrete_tangent_kernel(eps_flat, fcd, eps_c2, eps_cu2, n,
                             tension_enabled, Ec, eps_ct):
    r"""
    Numba-accelerated tangent-modulus kernel.

    .. math::

        E_t = \frac{d\sigma_c}{d\varepsilon} =
        \begin{cases}
            E_c
                & 0 < \varepsilon \le \varepsilon_{ct}
                  \;\text{(tension enabled)} \\
            -\dfrac{f_{cd}\,n}{\varepsilon_{c2}}
                \left(1 - \dfrac{\varepsilon}{\varepsilon_{c2}}\right)^{n-1}
                & \varepsilon_{c2} < \varepsilon \le 0 \\
            0 & \text{otherwise (plateau, beyond ultimate, post-cracking)}
        \end{cases}

    Parameters
    ----------
    eps_flat : numpy.ndarray
        1-D strain array.
    fcd, eps_c2, eps_cu2, n : float
        Material constants (same as stress kernel).
    tension_enabled : bool
    Ec : float
    eps_ct : float

    Returns
    -------
    numpy.ndarray
        1-D tangent modulus array [MPa].
    """
    out = np.zeros(eps_flat.shape[0], dtype=np.float64)
    for i in range(eps_flat.shape[0]):
        e = eps_flat[i]
        if e > 0.0:
            if tension_enabled and e <= eps_ct:
                out[i] = Ec
        elif e > eps_c2 and e <= 0.0:
            eta = e / eps_c2
            out[i] = -fcd * n / eps_c2 * (1.0 - eta) ** (n - 1.0)
        # Plateau (eps_c2 >= e >= eps_cu2): tangent = 0
        # Beyond ultimate or post-cracking: tangent = 0
    return out


# ------------------------------------------------------------------
#  Concrete dataclass
# ------------------------------------------------------------------

@dataclass
class Concrete(Material):
    r"""
    Parabola-rectangle concrete (EC2 3.1.7 / NTC 2018) with optional
    linear tension branch.

    **Compression** (:math:`\varepsilon \le 0`):

    .. math::

        \sigma_c(\varepsilon) =
        \begin{cases}
            -f_{cd}\!\left[1 - \left(1 - \dfrac{\varepsilon}
                {\varepsilon_{c2}}\right)^{\!n}\right]
                & \varepsilon_{c2} \le \varepsilon \le 0 \\[6pt]
            -f_{cd}
                & \varepsilon_{cu2} \le \varepsilon < \varepsilon_{c2} \\[6pt]
            0 & \varepsilon < \varepsilon_{cu2}
        \end{cases}

    **Tension** (:math:`\varepsilon > 0`), activated when
    :math:`f_{ct} > 0` **and** :math:`E_c > 0`:

    .. math::

        \sigma_c(\varepsilon) =
        \begin{cases}
            E_c \, \varepsilon
                & 0 < \varepsilon \le \varepsilon_{ct} \\[6pt]
            0
                & \varepsilon > \varepsilon_{ct}
        \end{cases}

    where :math:`\varepsilon_{ct} = f_{ct} / E_c` is the cracking
    strain.  When :math:`f_{ct} = 0` (default), the tension branch
    is suppressed and :math:`\sigma = 0` for all
    :math:`\varepsilon > 0`.

    Parameters
    ----------
    fck : float
        Characteristic cylinder strength [MPa].  **Required**: a
        concrete class is a declaration, and this container invents
        none.
    gamma_c : float
        Partial factor on the concrete.  **Required**, and supplied by
        the module that owns the clause -- for the Eurocode,
        :mod:`gensec.materials.ec2_properties` reading EN 1992-1-1
        §2.4.2.4.  This container carries it and does not
        choose it.
    long_term_reduction_factor : float
        Dimensionless factor multiplying :math:`f_{ck}` before the
        partial factor divides it, so that

        .. math::

            f_{cd} \;=\; k_{lt}\,\frac{f_{ck}}{\gamma_{c}} .

        **Required**, and deliberately *not* named after any clause:
        EN 1992-1-1 §3.1.6(1)P calls its own value
        :math:`\alpha_{cc}` and recommends 0.85, and that symbol lives
        in the Eurocode module where the clause lives.  A code that
        applies no material-side long-term reduction supplies 1.0 here
        as an explicit proxy rather than leaving a Eurocode-named
        field to be filled with a value the document never states.
        The neutral name also leaves room for the tension-branch
        counterpart :math:`\alpha_{ct}` without either factor being
        mistaken for the other.
    n_parabola : float, optional
        Parabolic exponent. Default 2.0.
    eps_c2 : float, optional
        Peak-stress strain (negative). Default -0.002.
    eps_cu2 : float, optional
        Ultimate compressive strain (negative). Default -0.0035.
    fct : float, optional
        Tensile strength [MPa] for the linear tension branch.
        Use :math:`f_{ctd}` (design) or :math:`f_{ctm}` (mean)
        depending on the verification context.  Default 0.0
        (no tension).
    Ec : float, optional
        Elastic modulus [MPa] for the tension branch.  Typically
        :math:`E_{cm}` from EC2 Table 3.1.  Default 0.0.

    Attributes
    ----------
    fcd : float
        Design compressive strength [MPa].
    eps_ct : float
        Cracking strain :math:`f_{ct}/E_c`.  Zero when the tension
        branch is disabled.
    tension_enabled : bool
        ``True`` when both ``fct > 0`` and ``Ec > 0``.
    """

    fck: float
    gamma_c: float
    long_term_reduction_factor: float
    n_parabola: float = 2.0
    eps_c2: float = -0.002
    eps_cu2: float = -0.0035
    fct: float = 0.0
    Ec: float = 0.0

    def reference_modulus(self):
        r"""
        :math:`E_{cm}` at the 28-day reference age [MPa].

        Read from this material's own EC2 bridge, not recomputed, so
        the number a datum is drawn on and the number the material's
        other clauses use are the same object's answer (rule 5).
        Verified equal to ``EC2RheologicalModel(fck).E_c(28.0)`` to the
        last digit: 35 220.462 MPa at :math:`f_{ck} = 40`.

        .. math::

            E_{cm} = 22\,000\left(\frac{f_{cm}}{10}\right)^{0.3},
            \qquad f_{cm} = f_{ck} + 8 \quad [\mathrm{MPa}]

        Returns
        -------
        ReferenceModulus

        Raises
        ------
        NotImplementedError
            If the bridge carries no ``ecm``.
        """
        from .codes import UNDECLARED, resolve_code
        code = getattr(self, "code", None)
        if code is None:
            # A material built outside the loader carries no
            # declaration.  It may still be governed by its own class:
            # a Concrete that carries an EC2 property object was built
            # through the EC2 bridge, and that IS a declaration.  A
            # bare Concrete(fck=...) is not, and gets UNDECLARED, which
            # declines and says why.
            code = (resolve_code(material_type="concrete_ec2_gen1")
                    if getattr(self, "ec2", None) is not None
                    else UNDECLARED)
        return code.reference_modulus(self)

    def __post_init__(self):
        r"""
        Validate the declared reduction and publish :math:`f_{cd}`.

        The three inputs are checked here rather than trusted, because
        a non-positive one propagates as a *number* -- a zero
        :math:`f_{cd}` yields zero capacity everywhere with nothing
        raised, which is the silent-failure shape this project
        refuses.  Each refusal names the field, the value seen and
        why the value cannot stand.

        Raises
        ------
        ValueError
            If any of :math:`f_{ck}`, :math:`\gamma_{c}` or
            :math:`k_{lt}` is not strictly positive.
        """
        for _name, _value in (("fck", self.fck),
                              ("gamma_c", self.gamma_c),
                              ("long_term_reduction_factor",
                               self.long_term_reduction_factor)):
            if not (_value > 0.0):
                raise ValueError(
                    f"Concrete.{_name} = {_value!r} is not positive. "
                    f"All three of fck, gamma_c and "
                    f"long_term_reduction_factor must be strictly "
                    f"positive: they multiply and divide into f_cd, so "
                    f"a zero or negative value would publish a design "
                    f"strength of zero or of the wrong sign and every "
                    f"capacity computed from it would come out silently "
                    f"wrong rather than raising.")
        self.fcd = self.long_term_reduction_factor * self.fck / self.gamma_c
        self.tension_enabled = (self.fct > 0.0) and (self.Ec > 0.0)
        self.eps_ct = self.fct / self.Ec if self.tension_enabled else 0.0

    @property
    def eps_min(self):
        """Most compressive admissible strain."""
        return self.eps_cu2

    @property
    def eps_max(self):
        r"""Most tensile admissible strain.

        Returns :math:`\varepsilon_{ct}` when the tension branch is
        active, 0.0 otherwise.
        """
        return self.eps_ct if self.tension_enabled else 0.0

    @property
    def E(self):
        r"""Elastic modulus for homogenization [MPa] — fail-loud.

        Returns ``Ec`` when a positive value was provided at
        construction.  When ``Ec`` is unset (the default: the
        design parabola-rectangle law needs no elastic modulus
        unless the tension branch is enabled), accessing ``E``
        **raises** instead of returning a silent zero stiffness
        (Phase-7 hygiene edit, finding F0).

        Note the role separation: ``Ec`` here is the modulus
        attached to this constitutive object (it also scales the
        tension branch).  The SLS verification layer never reads
        it implicitly — SLS moduli (:math:`E_{cm}`,
        :math:`E_{cm}(t)`, :math:`E_{c,\mathrm{eff}}`) are
        normative, time-dependent choices supplied explicitly
        through :func:`gensec.solver.sls.resolve_sls_moduli`
        (Phase-7 D3).

        Raises
        ------
        AttributeError
            If ``Ec`` was not set to a positive value.
        """
        if self.Ec > 0.0:
            return self.Ec
        raise AttributeError(
            "Concrete.E requested but Ec was not set at "
            "construction.  Provide Ec explicitly, or — for SLS "
            "verification — pass the modulus through the "
            "explicit 'moduli' mapping (gensec.solver.sls)."
        )

    # ------------------------------------------------------------------
    #  Scalar interface
    # ------------------------------------------------------------------

    def stress(self, eps):
        r"""
        Evaluate stress for a single strain value.

        Parameters
        ----------
        eps : float
            Strain (positive = tension, negative = compression).

        Returns
        -------
        float
            Stress [MPa].  Positive = tension, negative = compression.
        """
        # --- Tension ---
        if eps > 0:
            if self.tension_enabled and eps <= self.eps_ct:
                return self.Ec * eps
            return 0.0
        # --- Compression ---
        if eps < self.eps_cu2:
            return 0.0
        if eps >= self.eps_c2:
            eta = eps / self.eps_c2
            return -self.fcd * (1.0 - (1.0 - eta) ** self.n_parabola)
        return -self.fcd

    def tangent(self, eps):
        r"""
        Scalar tangent modulus :math:`E_t = d\sigma_c / d\varepsilon`.

        .. math::

            E_t(\varepsilon) =
            \begin{cases}
                E_c & 0 < \varepsilon \le \varepsilon_{ct}
                      \;\text{(tension enabled)} \\
                -\dfrac{f_{cd}\,n}{\varepsilon_{c2}}
                    \left(1 - \dfrac{\varepsilon}{\varepsilon_{c2}}
                    \right)^{n-1}
                    & \varepsilon_{c2} < \varepsilon \le 0 \\
                0 & \text{otherwise}
            \end{cases}

        Parameters
        ----------
        eps : float

        Returns
        -------
        float
        """
        if eps > 0.0:
            if self.tension_enabled and eps <= self.eps_ct:
                return self.Ec
            return 0.0
        if eps <= 0.0 and eps > self.eps_c2:
            eta = eps / self.eps_c2
            return (-self.fcd * self.n_parabola / self.eps_c2
                    * (1.0 - eta) ** (self.n_parabola - 1.0))
        return 0.0

    # ------------------------------------------------------------------
    #  Vectorized interface (any-shape, Numba-accelerated when available)
    # ------------------------------------------------------------------

    def stress_array(self, eps):
        r"""
        Vectorized stress evaluation.

        Accepts arrays of **any shape** (1-D, 2-D, …).  When *numba*
        is installed, the inner loop is JIT-compiled.

        Parameters
        ----------
        eps : numpy.ndarray
            Strain array.

        Returns
        -------
        numpy.ndarray
            Stress array [MPa], same shape as *eps*.
        """
        if _HAS_NUMBA:
            flat = np.ascontiguousarray(eps.ravel(), dtype=np.float64)
            return _concrete_stress_kernel(
                flat, self.fcd, self.eps_c2, self.eps_cu2,
                self.n_parabola, self.tension_enabled,
                self.fct, self.Ec, self.eps_ct,
            ).reshape(eps.shape)

        # Pure-NumPy fallback — works on any shape
        sigma = np.zeros_like(eps, dtype=np.float64)

        # --- Tension branch ---
        if self.tension_enabled:
            mt = (eps > 0) & (eps <= self.eps_ct)
            sigma[mt] = self.Ec * eps[mt]

        # --- Parabolic branch ---
        m1 = (eps <= 0) & (eps >= self.eps_c2)
        eta = eps[m1] / self.eps_c2
        sigma[m1] = -self.fcd * (1.0 - (1.0 - eta) ** self.n_parabola)

        # --- Plateau branch ---
        m2 = (eps < self.eps_c2) & (eps >= self.eps_cu2)
        sigma[m2] = -self.fcd

        return sigma

    def tangent_array(self, eps):
        r"""
        Vectorized tangent modulus :math:`E_t = d\sigma_c / d\varepsilon`.

        .. math::

            E_t(\varepsilon) =
            \begin{cases}
                E_c & 0 < \varepsilon \le \varepsilon_{ct}
                      \;\text{(tension enabled)} \\[4pt]
                -\dfrac{f_{cd} \, n}{\varepsilon_{c2}}
                    \left(1 - \dfrac{\varepsilon}{\varepsilon_{c2}}
                    \right)^{n-1}
                & \varepsilon_{c2} < \varepsilon \le 0 \\[4pt]
                0 & \text{otherwise}
            \end{cases}

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
            return _concrete_tangent_kernel(
                flat, self.fcd, self.eps_c2, self.eps_cu2,
                self.n_parabola, self.tension_enabled,
                self.Ec, self.eps_ct,
            ).reshape(eps.shape)

        # Pure-NumPy fallback
        Et = np.zeros_like(eps, dtype=np.float64)

        # --- Tension branch tangent ---
        if self.tension_enabled:
            mt = (eps > 0) & (eps <= self.eps_ct)
            Et[mt] = self.Ec

        # --- Parabolic branch tangent (strictly inside, not at eps_c2
        #     where we hit the plateau with Et = 0) ---
        m = (eps <= 0) & (eps > self.eps_c2)
        eta = eps[m] / self.eps_c2
        Et[m] = (-self.fcd * self.n_parabola / self.eps_c2
                 * (1.0 - eta) ** (self.n_parabola - 1.0))

        return Et