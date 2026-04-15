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
"""

import numpy as np
from dataclasses import dataclass
from .base import Material


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
        Characteristic cylinder strength [MPa].
    gamma_c : float, optional
        Partial safety factor. Default 1.5.
    alpha_cc : float, optional
        Long-term coefficient. Default 0.85.
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

    fck: float = 25.0
    gamma_c: float = 1.5
    alpha_cc: float = 0.85
    n_parabola: float = 2.0
    eps_c2: float = -0.002
    eps_cu2: float = -0.0035
    fct: float = 0.0
    Ec: float = 0.0

    def __post_init__(self):
        self.fcd = self.alpha_cc * self.fck / self.gamma_c
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

    def stress_array(self, eps):
        r"""
        Vectorized stress evaluation.

        Parameters
        ----------
        eps : numpy.ndarray
            Strain array.

        Returns
        -------
        numpy.ndarray
            Stress array [MPa].
        """
        sigma = np.zeros_like(eps)

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
