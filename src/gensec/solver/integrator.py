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
Fiber integrator and equilibrium solver — biaxial bending.

Supports both single-material and multi-material sections. When
the section has ``mat_indices`` (from :class:`GenericSection` with
``bulk_materials`` zones), the integrator groups fibers by material
to call each constitutive law on its own subset. For single-material
sections, the fast vectorized path is used unchanged.

The strain plane has 3 parameters
:math:`(\varepsilon_0, \chi_x, \chi_y)` and the internal forces
are :math:`(N, M_x, M_y)`.

Uniaxial bending is the special case :math:`\chi_y = 0`.

Performance features
--------------------
- **Batch integration** (:meth:`integrate_batch`): evaluate many
  strain configurations in a single vectorized call, eliminating
  Python-loop overhead in capacity-surface generators.
- **Analytical tangent stiffness** (:meth:`integrate_with_tangent`):
  compute internal forces *and* the 3×3 tangent matrix in one pass,
  halving the cost of each Newton-Raphson iteration compared to a
  finite-difference Jacobian.
"""

import numpy as np


class FiberSolver:
    r"""
    Material-agnostic fiber solver for biaxial bending + axial force.

    Strain plane:

    .. math::

        \varepsilon(x, y) = \varepsilon_0
            + \chi_x \, (y - y_{\text{ref}})
            + \chi_y \, (x - x_{\text{ref}})

    Internal forces:

    .. math::

        N   &= \sum_i \sigma_i \, A_i \\
        M_x &= \sum_i \sigma_i \, A_i \, (y_i - y_{\text{ref}}) \\
        M_y &= \sum_i \sigma_i \, A_i \, (x_i - x_{\text{ref}})

    Convention:

    - :math:`M_x > 0` compresses the bottom edge (:math:`y = 0`),
      tensions the top edge. Right-hand rule: thumb along +x,
      fingers curl from +y toward observer.
    - :math:`M_y > 0` compresses the right edge (:math:`x = x_{\max}`),
      tensions the left edge. Right-hand rule: thumb along +y,
      fingers curl from +x toward observer.

    Parameters
    ----------
    section : GenericSection or RectSection
        Any section exposing ``x_fibers``, ``y_fibers``, ``A_fibers``,
        ``x_rebars``, ``y_rebars``, ``A_rebars``, ``embedded_rebars``,
        ``bulk_material``, ``rebars``, ``x_centroid``, ``y_centroid``.
    x_ref : float or None, optional
        Reference x-coordinate [mm]. Default: x-centroid.
    y_ref : float or None, optional
        Reference y-coordinate [mm]. Default: y-centroid.
    """

    def __init__(self, section, x_ref=None, y_ref=None):
        self.sec = section
        self.x_ref = x_ref if x_ref is not None else section.x_centroid
        self.y_ref = y_ref if y_ref is not None else section.y_centroid
        self._rebar_groups = self._build_rebar_groups()

        # Pre-compute lever arms (constant for a given section)
        self._ly_bulk = self.sec.y_fibers - self.y_ref
        self._lx_bulk = self.sec.x_fibers - self.x_ref
        self._ly_rebar = self.sec.y_rebars - self.y_ref
        self._lx_rebar = self.sec.x_rebars - self.x_ref

        # Build bulk material groups for multi-material support
        self._bulk_groups = self._build_bulk_groups()
        self._is_multi_material = len(self._bulk_groups) > 1

    def _build_rebar_groups(self):
        r"""
        Group rebar layers by the pair
        ``(rebar material, bulk-zone material)``.

        For multi-material sections (a confined core inside an
        unconfined cover, for example), an embedded rebar must
        subtract the bulk stress evaluated with the constitutive
        law of the **zone the rebar physically occupies**, not the
        primary ``bulk_material``.  Splitting the rebar groups by
        this pair lets the integrator vectorise the lookup with no
        per-rebar branching at evaluation time.

        For single-material sections (no zones, or every rebar in
        zone ``0``) this collapses to the legacy "group by rebar
        material" behaviour, with the primary ``bulk_material``
        attached to every group.

        Returns
        -------
        list of tuple
            Each tuple is
            ``(rebar_material, bulk_zone_material,
            ndarray_of_rebar_indices)``.
        """
        sec = self.sec
        # Index-aligned list of candidate bulk-zone materials.
        all_bulk_mats = (sec.get_all_bulk_materials()
                         if hasattr(sec, 'get_all_bulk_materials')
                         else [sec.bulk_material])
        # Per-rebar zone index.  Sections without
        # ``mat_indices_rebar`` (legacy ``RectSection``) place every
        # rebar in zone 0.
        mat_idx_r = getattr(sec, 'mat_indices_rebar', None)
        if mat_idx_r is None:
            mat_idx_r = np.zeros(len(sec.rebars), dtype=int)

        groups = {}
        for i, r in enumerate(sec.rebars):
            key = (id(r.material), int(mat_idx_r[i]))
            if key not in groups:
                bulk_mat = all_bulk_mats[mat_idx_r[i]]
                groups[key] = (r.material, bulk_mat, [])
            groups[key][2].append(i)
        return [(rm, bm, np.array(ix))
                for rm, bm, ix in groups.values()]

    def _build_bulk_groups(self):
        r"""
        Group bulk fibers by material index.

        For single-material sections (no ``mat_indices`` attribute or
        all indices == 0), returns a single group covering all fibers.

        Returns
        -------
        list of tuple
            Each tuple is ``(Material, ndarray_of_fiber_indices)``.
        """
        sec = self.sec

        # Check if section has multi-material zones
        mat_indices = getattr(sec, 'mat_indices', None)
        if mat_indices is None or np.all(mat_indices == 0):
            # Single material: one group covering all fibers
            idx = np.arange(sec.n_fibers)
            return [(sec.bulk_material, idx)]

        # Multi-material: group by index
        all_mats = sec.get_all_bulk_materials()
        groups = []
        for mi in np.unique(mat_indices):
            idx = np.where(mat_indices == mi)[0]
            groups.append((all_mats[mi], idx))
        return groups

    # ==================================================================
    #  Single-configuration integration
    # ==================================================================

    def strain_field(self, eps0, chi_x, chi_y=0.0):
        r"""
        Compute strains at all fibers.

        .. math::

            \varepsilon_i = \varepsilon_0
                + \chi_x (y_i - y_{\text{ref}})
                - \chi_y (x_i - x_{\text{ref}})

        The minus sign on :math:`\chi_y` ensures that positive
        :math:`M_y` compresses the right edge (:math:`x = x_{\max}`),
        consistent with the right-hand rule around the y-axis.

        Parameters
        ----------
        eps0 : float
            Strain at the reference point.
        chi_x : float
            Curvature about the x-axis [1/mm].
        chi_y : float, optional
            Curvature about the y-axis [1/mm]. Default 0.

        Returns
        -------
        eps_bulk : numpy.ndarray
        eps_rebars : numpy.ndarray
        """
        eb = eps0 + chi_x * self._ly_bulk - chi_y * self._lx_bulk
        er = eps0 + chi_x * self._ly_rebar - chi_y * self._lx_rebar
        return eb, er

    def integrate(self, eps0, chi_x, chi_y=0.0):
        r"""
        Direct problem: strain parameters to internal forces.

        For embedded rebars, the bulk material contribution at the
        rebar location is subtracted to avoid double-counting:

        .. math::

            F_i = \bigl[\sigma_{\text{rebar},i}(\varepsilon_i)
                  - \sigma_{\text{bulk}}(\varepsilon_i)\bigr] \, A_{s,i}

        For multi-material sections, each fiber group is evaluated
        with its own constitutive law.

        Parameters
        ----------
        eps0 : float
        chi_x : float
            Curvature about x-axis [1/mm].
        chi_y : float, optional
            Curvature about y-axis [1/mm]. Default 0.

        Returns
        -------
        N : float
            Axial force [N]. Positive = tension.
        Mx : float
            Bending moment about x-axis [N*mm].
        My : float
            Bending moment about y-axis [N*mm].
        """
        eb, er = self.strain_field(eps0, chi_x, chi_y)

        # ---- Bulk contribution ----
        if not self._is_multi_material:
            # Fast path: single material on all fibers
            sb = self.sec.bulk_material.stress_array(eb)
        else:
            # Multi-material: evaluate each group separately
            sb = np.zeros_like(eb)
            for mat, idx in self._bulk_groups:
                sb[idx] = mat.stress_array(eb[idx])

        fA = sb * self.sec.A_fibers
        N = float(np.sum(fA))
        Mx = float(np.sum(fA * self._ly_bulk))
        My = -float(np.sum(fA * self._lx_bulk))

        # ---- Rebar contribution ----
        # For each (rebar material, bulk-zone material) group, subtract
        # the bulk stress evaluated with the **zone's** constitutive
        # law from the rebar stress for embedded rebars, then accumulate
        # forces and moments.  Splitting by (rebar mat, zone mat) is
        # what makes this correct for multi-material sections.
        embedded = self.sec.embedded_rebars

        for mat, bulk_mat, idx in self._rebar_groups:
            er_g = er[idx]
            s_rebar = mat.stress_array(er_g)
            sb_at_rebars = bulk_mat.stress_array(er_g)
            a = self.sec.A_rebars[idx]
            emb = embedded[idx]

            s_net = s_rebar.copy()
            s_net[emb] -= sb_at_rebars[emb]

            fa = s_net * a
            N += float(np.sum(fa))
            Mx += float(np.sum(fa * self._ly_rebar[idx]))
            My -= float(np.sum(fa * self._lx_rebar[idx]))

        return N, Mx, My

    # ==================================================================
    #  Analytical tangent stiffness
    # ==================================================================

    def integrate_with_tangent(self, eps0, chi_x, chi_y=0.0):
        r"""
        Internal forces **and** 3×3 tangent stiffness in one pass.

        The tangent stiffness matrix relates infinitesimal changes in
        the strain-plane parameters to changes in internal forces:

        .. math::

            \mathbf{K} =
            \frac{\partial (N,\,M_x,\,M_y)}
                 {\partial (\varepsilon_0,\,\chi_x,\,\chi_y)}
            = \sum_i E_{t,i} \, A_i \;
              \boldsymbol{\varphi}_i \, \boldsymbol{\varphi}_i^T

        where the shape-function vector for fiber *i* is

        .. math::

            \boldsymbol{\varphi}_i =
            \bigl[1,\; (y_i - y_{\text{ref}}),\;
                  -(x_i - x_{\text{ref}})\bigr]^T

        and :math:`E_{t,i} = d\sigma_i / d\varepsilon_i` is the
        tangent modulus at the current strain.

        Parameters
        ----------
        eps0 : float
        chi_x : float
        chi_y : float, optional

        Returns
        -------
        N, Mx, My : float
            Internal forces [N, N·mm].
        K : numpy.ndarray
            3×3 tangent stiffness matrix.
        """
        eb, er = self.strain_field(eps0, chi_x, chi_y)

        # ---- Bulk: stress and tangent ----
        if not self._is_multi_material:
            sb = self.sec.bulk_material.stress_array(eb)
            Et_b = self.sec.bulk_material.tangent_array(eb)
        else:
            sb = np.zeros_like(eb)
            Et_b = np.zeros_like(eb)
            for mat, idx in self._bulk_groups:
                sb[idx] = mat.stress_array(eb[idx])
                Et_b[idx] = mat.tangent_array(eb[idx])

        A = self.sec.A_fibers
        ly = self._ly_bulk
        lx = self._lx_bulk

        fA = sb * A
        N = float(np.sum(fA))
        Mx = float(np.sum(fA * ly))
        My = -float(np.sum(fA * lx))

        # Tangent stiffness from bulk fibers
        #   phi = [1, ly, -lx]
        #   K[i,j] = sum(Et * A * phi_i * phi_j)
        EtA = Et_b * A
        K = np.empty((3, 3))
        K[0, 0] = float(np.sum(EtA))
        K[0, 1] = float(np.sum(EtA * ly))
        K[0, 2] = -float(np.sum(EtA * lx))
        K[1, 0] = K[0, 1]
        K[1, 1] = float(np.sum(EtA * ly * ly))
        K[1, 2] = -float(np.sum(EtA * ly * lx))
        K[2, 0] = K[0, 2]
        K[2, 1] = K[1, 2]
        K[2, 2] = float(np.sum(EtA * lx * lx))

        # ---- Rebar contribution ----
        # See ``integrate``: each (rebar mat, zone mat) group subtracts
        # the displaced bulk stress and tangent using the zone's law.
        embedded = self.sec.embedded_rebars

        for mat, bulk_mat, idx in self._rebar_groups:
            er_g = er[idx]
            s_rebar = mat.stress_array(er_g)
            Et_rebar = mat.tangent_array(er_g)
            sb_at_rebars = bulk_mat.stress_array(er_g)
            Et_bulk_r = bulk_mat.tangent_array(er_g)
            a = self.sec.A_rebars[idx]
            emb = embedded[idx]
            ly_r = self._ly_rebar[idx]
            lx_r = self._lx_rebar[idx]

            s_net = s_rebar.copy()
            s_net[emb] -= sb_at_rebars[emb]

            Et_net = Et_rebar.copy()
            Et_net[emb] -= Et_bulk_r[emb]

            fa = s_net * a
            N += float(np.sum(fa))
            Mx += float(np.sum(fa * ly_r))
            My -= float(np.sum(fa * lx_r))

            EtA_r = Et_net * a
            K[0, 0] += float(np.sum(EtA_r))
            K[0, 1] += float(np.sum(EtA_r * ly_r))
            K[0, 2] -= float(np.sum(EtA_r * lx_r))
            K[1, 1] += float(np.sum(EtA_r * ly_r * ly_r))
            K[1, 2] -= float(np.sum(EtA_r * ly_r * lx_r))
            K[2, 2] += float(np.sum(EtA_r * lx_r * lx_r))

        # Symmetric
        K[1, 0] = K[0, 1]
        K[2, 0] = K[0, 2]
        K[2, 1] = K[1, 2]

        return N, Mx, My, K

    # ==================================================================
    #  Batch integration — many configurations in one NumPy call
    # ==================================================================

    def integrate_batch(self, eps0, chi_x, chi_y):
        r"""
        Evaluate internal forces for many strain configurations at once.

        All inputs are 1-D arrays of the same length *n*.  The
        computation is fully vectorized: strain fields are built as
        2-D arrays of shape ``(n, n_fibers)`` and passed through the
        constitutive laws in one call, eliminating Python-loop
        overhead.

        Parameters
        ----------
        eps0 : numpy.ndarray
            Shape ``(n,)``.  Strains at the reference point.
        chi_x : numpy.ndarray
            Shape ``(n,)``.  Curvatures about the x-axis [1/mm].
        chi_y : numpy.ndarray
            Shape ``(n,)``.  Curvatures about the y-axis [1/mm].

        Returns
        -------
        N : numpy.ndarray
            Shape ``(n,)``.  Axial forces [N].
        Mx : numpy.ndarray
            Shape ``(n,)``.  Bending moments about x [N·mm].
        My : numpy.ndarray
            Shape ``(n,)``.  Bending moments about y [N·mm].
        """
        eps0 = np.asarray(eps0, dtype=np.float64)
        chi_x = np.asarray(chi_x, dtype=np.float64)
        chi_y = np.asarray(chi_y, dtype=np.float64)

        # Bulk strains: (n, n_fibers)
        eb = (eps0[:, None]
              + chi_x[:, None] * self._ly_bulk[None, :]
              - chi_y[:, None] * self._lx_bulk[None, :])

        # Bulk stresses: (n, n_fibers)
        if not self._is_multi_material:
            sb = self.sec.bulk_material.stress_array(eb)
        else:
            sb = np.zeros_like(eb)
            for mat, idx in self._bulk_groups:
                sb[:, idx] = mat.stress_array(eb[:, idx])

        fA = sb * self.sec.A_fibers[None, :]        # (n, n_fibers)
        N = fA.sum(axis=1)                           # (n,)
        Mx = (fA * self._ly_bulk[None, :]).sum(axis=1)
        My = -(fA * self._lx_bulk[None, :]).sum(axis=1)

        # Rebar strains: (n, n_rebars)
        n_rebars = len(self.sec.y_rebars)
        if n_rebars == 0:
            return N, Mx, My

        er = (eps0[:, None]
              + chi_x[:, None] * self._ly_rebar[None, :]
              - chi_y[:, None] * self._lx_rebar[None, :])

        embedded = self.sec.embedded_rebars

        for mat, bulk_mat, idx in self._rebar_groups:
            er_g = er[:, idx]
            s_rebar = mat.stress_array(er_g)
            sb_at_rebars = bulk_mat.stress_array(er_g)
            a = self.sec.A_rebars[idx]
            emb = embedded[idx]
            ly_r = self._ly_rebar[idx]
            lx_r = self._lx_rebar[idx]

            s_net = s_rebar.copy()
            s_net[:, emb] -= sb_at_rebars[:, emb]

            fa = s_net * a[None, :]                  # (n, len(idx))
            N += fa.sum(axis=1)
            Mx += (fa * ly_r[None, :]).sum(axis=1)
            My -= (fa * lx_r[None, :]).sum(axis=1)

        return N, Mx, My

    # ==================================================================
    #  Numerical Jacobian (kept for validation / fallback)
    # ==================================================================

    def jacobian(self, eps0, chi_x, chi_y=0.0, deps=1e-8):
        r"""
        Numerical 3×3 Jacobian via forward finite differences.

        .. math::

            \mathbf{J} = \frac{\partial(N, M_x, M_y)}
                              {\partial(\varepsilon_0, \chi_x, \chi_y)}

        Parameters
        ----------
        eps0, chi_x, chi_y : float
        deps : float, optional
            Perturbation. Default 1e-8.

        Returns
        -------
        numpy.ndarray
            Shape (3, 3).
        """
        f0 = np.array(self.integrate(eps0, chi_x, chi_y))
        J = np.empty((3, 3))
        for j, (de, dx, dy) in enumerate([
            (deps, 0, 0), (0, deps, 0), (0, 0, deps)
        ]):
            f1 = np.array(self.integrate(
                eps0 + de, chi_x + dx, chi_y + dy))
            J[:, j] = (f1 - f0) / deps
        return J

    def _is_uniaxial(self, axis=None):
        r"""
        Detect if the section is effectively uniaxial.

        A section is uniaxial when one of its in-plane extents is
        degenerate, so the corresponding curvature has no
        integration support and the matching Jacobian column is
        singular.

        Two degenerate cases are recognised:

        - **vertical-degenerate**: every fiber shares the same
          :math:`x` coordinate.  :math:`\chi_y` has no effect, so
          bending is meaningful only about the :math:`x` axis.
        - **horizontal-degenerate**: every fiber shares the same
          :math:`y` coordinate.  :math:`\chi_x` has no effect, so
          bending is meaningful only about the :math:`y` axis.

        Parameters
        ----------
        axis : ``'x'``, ``'y'`` or ``None``, optional
            If specified, the test is restricted to the matching
            degeneracy: ``'x'`` returns ``True`` only for the
            vertical-degenerate case (chi_y singular), ``'y'`` only
            for the horizontal-degenerate case.  Without ``axis``,
            the test returns ``True`` for either degeneracy —
            preserving the legacy single-argument semantics.

        Returns
        -------
        bool
        """
        # Range of lever arms along x (sensitivity to chi_y).
        if len(self._lx_rebar) > 0:
            lx_range = (np.max(np.abs(self._lx_bulk))
                        + np.max(np.abs(self._lx_rebar)))
        else:
            lx_range = np.max(np.abs(self._lx_bulk))

        # Range of lever arms along y (sensitivity to chi_x).
        if len(self._ly_rebar) > 0:
            ly_range = (np.max(np.abs(self._ly_bulk))
                        + np.max(np.abs(self._ly_rebar)))
        else:
            ly_range = np.max(np.abs(self._ly_bulk))

        vertical_deg = lx_range < 1e-6     # bending about x
        horizontal_deg = ly_range < 1e-6   # bending about y

        if axis == 'x':
            return vertical_deg
        if axis == 'y':
            return horizontal_deg
        return vertical_deg or horizontal_deg

    # ==================================================================
    #  Main solver entry point
    # ==================================================================

    def solve_equilibrium(self, N_target, Mx_target, My_target=0.0,
                          eps0_init=0.0, chi_x_init=1e-6,
                          chi_y_init=0.0,
                          tol=1e-3, max_iter=50):
        r"""
        Inverse problem: find strain plane for target (N, Mx, My).

        Newton-Raphson with **analytical tangent stiffness** and
        backtracking line search.  Automatically reduces to a 2×2
        system when the section is uniaxial, and further to a
        **1×1 bisection** when both target moments are negligible
        (pure axial load).

        The pure-axial branch exploits the monotonicity of
        :math:`N(\varepsilon_0)` at :math:`\chi_x = \chi_y = 0`:

        .. math::

            \frac{\partial N}{\partial \varepsilon_0}\bigg|_{\chi=0}
            = \sum_i \frac{\partial \sigma_i}
              {\partial \varepsilon}\,A_i \;\ge\; 0

        which guarantees unique bracketing and convergence of the
        bisection search.

        Parameters
        ----------
        N_target : float
            Target axial force [N].
        Mx_target : float
            Target moment about x-axis [N*mm].
        My_target : float, optional
            Target moment about y-axis [N*mm]. Default 0.
        eps0_init, chi_x_init, chi_y_init : float, optional
            Initial guesses.
        tol : float, optional
            Force tolerance [N]; moment tolerance is ``tol * 1000``.
        max_iter : int, optional

        Returns
        -------
        dict
            Keys: ``eps0``, ``chi_x``, ``chi_y``, ``N``, ``Mx``,
            ``My``, ``converged``, ``iterations``.
        """
        M_tol = tol * 1000  # moment tolerance [N·mm]

        # ----------------------------------------------------------
        #  Fast path: pure axial load (both moments negligible)
        # ----------------------------------------------------------
        if abs(Mx_target) < M_tol and abs(My_target) < M_tol:
            sol = self._solve_pure_axial(N_target, tol, max_iter)
            if sol["converged"]:
                return sol
            # If pure-axial fails (shouldn't), fall through to 2×2

        # ----------------------------------------------------------
        #  Uniaxial or near-uniaxial
        # ----------------------------------------------------------
        # Vertically-degenerate section: only chi_x is meaningful,
        # solve about the x-axis.
        if self._is_uniaxial(axis='x'):
            return self._solve_uniaxial(
                N_target, Mx_target, eps0_init, chi_x_init,
                tol, max_iter)

        # Horizontally-degenerate section: only chi_y is meaningful,
        # solve about the y-axis.
        if self._is_uniaxial(axis='y'):
            return self._solve_uniaxial_y(
                N_target, My_target, eps0_init, chi_y_init,
                tol, max_iter)

        # Near-uniaxial fast paths.  When one of the target moments
        # is negligible, try the corresponding 2-unknown solve in
        # the dominant plane and accept it only if the spurious
        # moment stays inside tolerance.  Both branches are present
        # for symmetry — previously only the My-near-zero branch
        # existed, penalising demands dominated by My.
        if abs(My_target) < M_tol:
            sol = self._solve_uniaxial(
                N_target, Mx_target, eps0_init, chi_x_init,
                tol, max_iter)
            if sol["converged"] and abs(sol["My"]) < M_tol:
                return sol

        if abs(Mx_target) < M_tol:
            sol = self._solve_uniaxial_y(
                N_target, My_target, eps0_init, chi_y_init,
                tol, max_iter)
            if sol["converged"] and abs(sol["Mx"]) < M_tol:
                return sol

        return self._solve_biaxial(
            N_target, Mx_target, My_target,
            eps0_init, chi_x_init, chi_y_init,
            tol, max_iter)

    # ------------------------------------------------------------------
    #  Pure axial solver  (1-unknown bisection)
    # ------------------------------------------------------------------

    def _solve_pure_axial(self, N_target, tol, max_iter):
        r"""
        Find :math:`\varepsilon_0` for pure axial load
        (:math:`\chi_x = \chi_y = 0`).

        The function :math:`N(\varepsilon_0)` at zero curvature is
        **not globally monotone** because the concrete constitutive
        law returns zero stress for strains beyond
        :math:`\varepsilon_{cu2}`, creating a non-monotone drop at
        the compressive end.  However, :math:`N` *is* monotonically
        non-decreasing on the sub-interval
        :math:`[\varepsilon_{c2},\, \varepsilon_{su}]` where all
        constitutive laws are active, and the vast majority of
        practical targets lie in this range.

        Strategy:

        1.  Scan :math:`N(\varepsilon_0)` on a fine grid spanning
            the full material strain range — using
            :meth:`integrate_batch` for speed.
        2.  Find consecutive grid points where :math:`N` crosses
            :math:`N_{\text{target}}`.  Among all crossings, pick
            the one in the monotone working region (closest to the
            centroid strain).
        3.  Bisect within that local bracket.

        Parameters
        ----------
        N_target : float
            Target axial force [N].
        tol : float
            Convergence tolerance on N [N].
        max_iter : int
            Maximum bisection iterations.

        Returns
        -------
        dict
        """
        sec = self.sec

        # --- Build the full strain scan range ---
        eps_lo = sec.bulk_material.eps_min
        eps_hi = 0.0
        for r in sec.rebars:
            eps_lo = min(eps_lo, r.material.eps_min)
            eps_hi = max(eps_hi, r.material.eps_max)
        eps_lo *= 1.01
        eps_hi *= 1.01

        # --- Batch scan ---
        n_scan = 120
        eps_vals = np.linspace(eps_lo, eps_hi, n_scan)
        chi_zero = np.zeros(n_scan)
        N_vals, _, _ = self.integrate_batch(eps_vals, chi_zero, chi_zero)

        # --- Find all crossings of N_target ---
        crossings = []
        for k in range(n_scan - 1):
            if ((N_vals[k] - N_target) * (N_vals[k + 1] - N_target) <= 0
                    and abs(N_vals[k + 1] - N_vals[k]) > 0.1):
                crossings.append(k)

        if not crossings:
            return self._fail_result()

        # Pick the best crossing: prefer the one closest to eps0=0
        # (i.e. in the working monotone region, not in the
        # non-physical tail beyond eps_cu2).
        best_k = min(crossings,
                     key=lambda k: abs(eps_vals[k] + eps_vals[k + 1]))

        a = eps_vals[best_k]
        b = eps_vals[best_k + 1]
        Na = N_vals[best_k]

        # --- Bisection within the local bracket ---
        for i in range(max_iter):
            mid = 0.5 * (a + b)
            N_mid, Mx_mid, My_mid = self.integrate(mid, 0.0, 0.0)
            if abs(N_mid - N_target) < tol:
                return {"eps0": mid, "chi_x": 0.0, "chi_y": 0.0,
                        "N": N_mid, "Mx": Mx_mid, "My": My_mid,
                        "converged": True, "iterations": i + 1}
            if (N_mid - N_target) * (Na - N_target) <= 0:
                b = mid
            else:
                a = mid
                Na = N_mid

        mid = 0.5 * (a + b)
        N_f, Mx_f, My_f = self.integrate(mid, 0.0, 0.0)
        converged = abs(N_f - N_target) < tol
        return {"eps0": mid, "chi_x": 0.0, "chi_y": 0.0,
                "N": N_f, "Mx": Mx_f, "My": My_f,
                "converged": converged, "iterations": max_iter}

    @staticmethod
    def _fail_result():
        """Return a non-converged result placeholder."""
        return {"eps0": 0.0, "chi_x": 0.0, "chi_y": 0.0,
                "N": 0.0, "Mx": 0.0, "My": 0.0,
                "converged": False, "iterations": 0}

    # ------------------------------------------------------------------
    #  Internal: elastic initial guess
    # ------------------------------------------------------------------

    def _elastic_initial_guess(self, N_target, Mx_target, My_target=0.0):
        r"""
        Estimate initial (eps0, chi_x, chi_y) from linear-elastic theory.

        Uses the **analytical** tangent stiffness at the origin,
        which is the elastic stiffness matrix.

        Returns
        -------
        eps0, chi_x, chi_y : float
        
        """
        ### TODO: we should use ideal gross section properties here, 
        ### to get a more accurate initial guess. 
        try:
            _, _, _, K0 = self.integrate_with_tangent(0.0, 0.0, 0.0)
            target = np.array([N_target, Mx_target, My_target])
            x0 = np.linalg.solve(K0, target)
            x0[0] = np.clip(x0[0], -0.003, 0.003)
            x0[1] = np.clip(x0[1], -1e-4, 1e-4)
            x0[2] = np.clip(x0[2], -1e-4, 1e-4)
            return float(x0[0]), float(x0[1]), float(x0[2])
        except np.linalg.LinAlgError:
            sec = self.sec
            A_ideal_gross = getattr(sec, 'ideal_gross_area',
                                    sec.B * sec.H)
            # Independent moments of inertia for each bending axis.
            I_x_approx = (A_ideal_gross * sec.H ** 2 / 12
                          if sec.H > 0 else 0.0)
            I_y_approx = (A_ideal_gross * sec.B ** 2 / 12
                          if sec.B > 0 else 0.0)
            ### TODO: substitute 30000 with real bulk base modulus of the section
            E_eff = 30000.0
            eps0_est = N_target / (A_ideal_gross * E_eff)
            chi_x_est = (Mx_target / (E_eff * I_x_approx)
                         if I_x_approx > 0 else 1e-6)
            chi_y_est = (My_target / (E_eff * I_y_approx)
                         if I_y_approx > 0 else 0.0)
            return (float(eps0_est),
                    float(chi_x_est),
                    float(chi_y_est))

    # ------------------------------------------------------------------
    #  Uniaxial solver (analytical Jacobian)
    # ------------------------------------------------------------------

    def _solve_uniaxial(self, N_target, Mx_target,
                        eps0_init, chi_x_init, tol, max_iter):
        r"""
        2-unknown solver (:math:`\chi_y` fixed at 0).

        Strategy:

        1. Try elastic initial guess → Newton-Raphson.
        2. If that fails, warm-start from a pure-axial solve at
           :math:`N = N_{\text{target}}` with a small perturbation
           in :math:`\chi_x` to break the singularity.
        3. Multi-start grid of :math:`\chi_x` values.

        Parameters
        ----------
        N_target, Mx_target : float
        eps0_init, chi_x_init : float
        tol : float
        max_iter : int

        Returns
        -------
        dict
        """
        # --- Attempt 1: user-supplied or elastic guess ---
        if eps0_init == 0.0 and chi_x_init == 1e-6:
            e0, c0, _ = self._elastic_initial_guess(
                N_target, Mx_target, 0.0)
        else:
            e0, c0 = eps0_init, chi_x_init

        sol = self._nr_uniaxial(N_target, Mx_target, e0, c0,
                                tol, max_iter)
        if sol["converged"]:
            return sol

        # --- Attempt 2: warm-start from pure-axial bisection ---
        sol_axial = self._solve_pure_axial(N_target, tol, max_iter)
        if sol_axial["converged"]:
            e0_ax = sol_axial["eps0"]
            # Estimate chi_x from the sign of Mx_target
            chi_sign = 1.0 if Mx_target >= 0 else -1.0
            for chi_mag in [1e-6, 5e-6, 1e-5, 3e-5, 5e-5]:
                sol2 = self._nr_uniaxial(
                    N_target, Mx_target,
                    e0_ax, chi_sign * chi_mag,
                    tol, max_iter)
                if sol2["converged"]:
                    return sol2

        # --- Attempt 3: multi-start grid ---
        emb = self.sec.bulk_material.eps_min
        for chi_sign in [1.0, -1.0]:
            for chi_mag in [1e-5, 5e-6, 2e-5, 3e-5, 5e-5]:
                chi_try = chi_sign * chi_mag
                try:
                    N0, _, _ = self.integrate(0.0, chi_try, 0.0)
                    N1, _, _ = self.integrate(1e-6, chi_try, 0.0)
                    dNde = (N1 - N0) / 1e-6
                    if abs(dNde) > 1:
                        eps0_try = (N_target - N0) / dNde
                        eps0_try = np.clip(eps0_try, emb, -emb)
                    else:
                        eps0_try = emb / 2
                except Exception:
                    eps0_try = emb / 2

                sol = self._nr_uniaxial(N_target, Mx_target,
                                        eps0_try, chi_try,
                                        tol, max_iter // 2)
                if sol["converged"]:
                    return sol

        return self._nr_uniaxial(N_target, Mx_target, e0, c0,
                                 tol, max_iter)

    def _nr_uniaxial(self, N_target, Mx_target, eps0, chi_x,
                     tol, max_iter):
        r"""
        Core Newton-Raphson for uniaxial case.

        Uses the **analytical** 2×2 sub-block of the tangent
        stiffness matrix instead of finite-difference perturbations.
        """
        for i in range(max_iter):
            N, Mx, My, K = self.integrate_with_tangent(eps0, chi_x, 0.0)
            r = np.array([N - N_target, Mx - Mx_target])

            if abs(r[0]) < tol and abs(r[1]) < tol * 1000:
                return {
                    "eps0": eps0, "chi_x": chi_x, "chi_y": 0.0,
                    "N": N, "Mx": Mx, "My": My,
                    "converged": True, "iterations": i + 1,
                }

            # Extract 2×2 sub-block [dN/de, dN/dc; dMx/de, dMx/dc]
            J = K[:2, :2]
            det = abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
            if det < 1e-20:
                eps0 += 1e-5
                chi_x += 1e-7
                continue

            try:
                d = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                eps0 += 1e-5
                chi_x += 1e-7
                continue

            # Backtracking line search
            alpha = 1.0
            r_norm = np.linalg.norm(r)
            for _ in range(15):
                e_new = eps0 + alpha * d[0]
                c_new = chi_x + alpha * d[1]
                Nn, Mxn, _ = self.integrate(e_new, c_new, 0.0)
                rn = np.array([Nn - N_target, Mxn - Mx_target])
                if np.linalg.norm(rn) < r_norm:
                    break
                alpha *= 0.5
            eps0 += alpha * d[0]
            chi_x += alpha * d[1]

        N, Mx, My = self.integrate(eps0, chi_x, 0.0)
        return {
            "eps0": eps0, "chi_x": chi_x, "chi_y": 0.0,
            "N": N, "Mx": Mx, "My": My,
            "converged": False, "iterations": max_iter,
        }

    # ------------------------------------------------------------------
    #  Uniaxial solver about the y-axis  (mirror of _solve_uniaxial)
    # ------------------------------------------------------------------

    def _solve_uniaxial_y(self, N_target, My_target,
                          eps0_init, chi_y_init, tol, max_iter):
        r"""
        2-unknown solver with :math:`\chi_x` fixed at 0.

        Mirror image of :meth:`_solve_uniaxial`, applicable when
        the bending is dominant about the :math:`y` axis (target
        :math:`M_x \approx 0`) or when the section is degenerate
        horizontally (:meth:`_is_uniaxial(axis='y')`).

        Strategy is identical to the x-axis variant: elastic guess
        → Newton, axial warm-start with sign-aware curvature, and
        a multi-start grid as last resort.

        Parameters
        ----------
        N_target, My_target : float
        eps0_init, chi_y_init : float
        tol : float
        max_iter : int

        Returns
        -------
        dict
            ``eps0``, ``chi_x`` (always 0), ``chi_y``, ``N``, ``Mx``,
            ``My``, ``converged``, ``iterations``.
        """
        # --- Attempt 1: user-supplied or elastic guess ---
        if eps0_init == 0.0 and chi_y_init == 0.0:
            e0, _, c0 = self._elastic_initial_guess(
                N_target, 0.0, My_target)
            if c0 == 0.0:
                # Elastic-fallback path may return chi_y = 0; nudge.
                c0 = 1e-6 if My_target >= 0 else -1e-6
        else:
            e0, c0 = eps0_init, chi_y_init

        sol = self._nr_uniaxial_y(N_target, My_target, e0, c0,
                                  tol, max_iter)
        if sol["converged"]:
            return sol

        # --- Attempt 2: warm-start from pure-axial bisection ---
        sol_axial = self._solve_pure_axial(N_target, tol, max_iter)
        if sol_axial["converged"]:
            e0_ax = sol_axial["eps0"]
            chi_sign = 1.0 if My_target >= 0 else -1.0
            for chi_mag in [1e-6, 5e-6, 1e-5, 3e-5, 5e-5]:
                sol2 = self._nr_uniaxial_y(
                    N_target, My_target,
                    e0_ax, chi_sign * chi_mag,
                    tol, max_iter)
                if sol2["converged"]:
                    return sol2

        # --- Attempt 3: multi-start grid ---
        emb = self.sec.bulk_material.eps_min
        for chi_sign in [1.0, -1.0]:
            for chi_mag in [1e-5, 5e-6, 2e-5, 3e-5, 5e-5]:
                chi_try = chi_sign * chi_mag
                try:
                    N0, _, _ = self.integrate(0.0, 0.0, chi_try)
                    N1, _, _ = self.integrate(1e-6, 0.0, chi_try)
                    dNde = (N1 - N0) / 1e-6
                    if abs(dNde) > 1:
                        eps0_try = (N_target - N0) / dNde
                        eps0_try = np.clip(eps0_try, emb, -emb)
                    else:
                        eps0_try = emb / 2
                except Exception:
                    eps0_try = emb / 2

                sol = self._nr_uniaxial_y(N_target, My_target,
                                          eps0_try, chi_try,
                                          tol, max_iter // 2)
                if sol["converged"]:
                    return sol

        return self._nr_uniaxial_y(N_target, My_target, e0, c0,
                                   tol, max_iter)

    def _nr_uniaxial_y(self, N_target, My_target, eps0, chi_y,
                       tol, max_iter):
        r"""
        Core Newton-Raphson for uniaxial bending about the y-axis.

        Uses the **analytical** 2×2 sub-block of the tangent
        stiffness matrix corresponding to
        :math:`(\varepsilon_0, \chi_y)`, i.e. rows/cols 0 and 2 of
        the full 3×3 tangent.  Mirror image of :meth:`_nr_uniaxial`.
        """
        for i in range(max_iter):
            N, Mx, My, K = self.integrate_with_tangent(eps0, 0.0, chi_y)
            r = np.array([N - N_target, My - My_target])

            if abs(r[0]) < tol and abs(r[1]) < tol * 1000:
                return {
                    "eps0": eps0, "chi_x": 0.0, "chi_y": chi_y,
                    "N": N, "Mx": Mx, "My": My,
                    "converged": True, "iterations": i + 1,
                }

            # Extract 2×2 sub-block [dN/de, dN/dcy; dMy/de, dMy/dcy]
            J = np.array([[K[0, 0], K[0, 2]],
                          [K[2, 0], K[2, 2]]])
            det = abs(J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0])
            if det < 1e-20:
                eps0 += 1e-5
                chi_y += 1e-7
                continue

            try:
                d = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                eps0 += 1e-5
                chi_y += 1e-7
                continue

            # Backtracking line search
            alpha = 1.0
            r_norm = np.linalg.norm(r)
            for _ in range(15):
                e_new = eps0 + alpha * d[0]
                c_new = chi_y + alpha * d[1]
                Nn, _, Myn = self.integrate(e_new, 0.0, c_new)
                rn = np.array([Nn - N_target, Myn - My_target])
                if np.linalg.norm(rn) < r_norm:
                    break
                alpha *= 0.5
            eps0 += alpha * d[0]
            chi_y += alpha * d[1]

        N, Mx, My = self.integrate(eps0, 0.0, chi_y)
        return {
            "eps0": eps0, "chi_x": 0.0, "chi_y": chi_y,
            "N": N, "Mx": Mx, "My": My,
            "converged": False, "iterations": max_iter,
        }

    # ------------------------------------------------------------------
    #  Biaxial solver (analytical Jacobian)
    # ------------------------------------------------------------------

    def _solve_biaxial(self, N_target, Mx_target, My_target,
                       eps0_init, chi_x_init, chi_y_init,
                       tol, max_iter):
        """3-unknown solver with multi-start strategy."""
        if (eps0_init == 0.0 and chi_x_init == 1e-6
                and chi_y_init == 0.0):
            e0, cx0, cy0 = self._elastic_initial_guess(
                N_target, Mx_target, My_target)
        else:
            e0, cx0, cy0 = eps0_init, chi_x_init, chi_y_init

        sol = self._nr_biaxial(N_target, Mx_target, My_target,
                               e0, cx0, cy0, tol, max_iter)
        if sol["converged"]:
            return sol

        # Warm-start from uniaxial
        sol_uni = self._solve_uniaxial(
            N_target, Mx_target, 0.0, 1e-6, tol, max_iter // 2)
        if sol_uni["converged"]:
            sol2 = self._nr_biaxial(
                N_target, Mx_target, My_target,
                sol_uni["eps0"], sol_uni["chi_x"], 1e-7,
                tol, max_iter)
            if sol2["converged"]:
                return sol2

        # Grid of chi_y
        for cy_try in [1e-6, -1e-6, 5e-6, -5e-6, 1e-5, -1e-5]:
            sol3 = self._nr_biaxial(
                N_target, Mx_target, My_target,
                e0, cx0, cy_try, tol, max_iter // 2)
            if sol3["converged"]:
                return sol3

        return sol

    def _nr_biaxial(self, N_target, Mx_target, My_target,
                    eps0, chi_x, chi_y, tol, max_iter):
        r"""
        Core Newton-Raphson for biaxial case.

        Uses the **analytical** 3×3 tangent stiffness matrix.
        """
        x = np.array([eps0, chi_x, chi_y])
        target = np.array([N_target, Mx_target, My_target])
        tol_vec = np.array([tol, tol * 1000, tol * 1000])

        for i in range(max_iter):
            N, Mx, My, K = self.integrate_with_tangent(
                x[0], x[1], x[2])
            f = np.array([N, Mx, My])
            r = f - target

            if np.all(np.abs(r) < tol_vec):
                return {
                    "eps0": x[0], "chi_x": x[1], "chi_y": x[2],
                    "N": f[0], "Mx": f[1], "My": f[2],
                    "converged": True, "iterations": i + 1,
                }

            try:
                d = np.linalg.solve(K, -r)
            except np.linalg.LinAlgError:
                x += np.array([1e-5, 1e-7, 1e-7])
                continue

            # Backtracking line search
            alpha = 1.0
            r_norm = np.linalg.norm(r)
            for _ in range(15):
                x_new = x + alpha * d
                f_new = np.array(self.integrate(
                    x_new[0], x_new[1], x_new[2]))
                r_new = f_new - target
                if np.linalg.norm(r_new) < r_norm:
                    break
                alpha *= 0.5
            x = x + alpha * d

        f = np.array(self.integrate(x[0], x[1], x[2]))
        return {
            "eps0": x[0], "chi_x": x[1], "chi_y": x[2],
            "N": f[0], "Mx": f[1], "My": f[2],
            "converged": False, "iterations": max_iter,
        }

    # ------------------------------------------------------------------
    #  Post-processing
    # ------------------------------------------------------------------

    def get_fiber_results(self, eps0, chi_x, chi_y=0.0):
        r"""
        Full strain/stress state at every fiber.

        For multi-material sections, each bulk fiber is evaluated
        with its own constitutive law.

        Parameters
        ----------
        eps0 : float
        chi_x : float
        chi_y : float, optional

        Returns
        -------
        dict
            ``'bulk'``: sub-dict with ``x``, ``y``, ``eps``,
            ``sigma``, ``dA``.
            ``'rebars'``: sub-dict with ``x``, ``y``, ``eps``,
            ``sigma`` (ideal_gross), ``sigma_net`` (net after bulk
            subtraction), ``A``, ``embedded``.
        """
        eb, er = self.strain_field(eps0, chi_x, chi_y)

        # Bulk stresses
        if not self._is_multi_material:
            sb = self.sec.bulk_material.stress_array(eb)
        else:
            sb = np.zeros_like(eb)
            for mat, idx in self._bulk_groups:
                sb[idx] = mat.stress_array(eb[idx])

        # Rebar stresses (zone-aware grouping).
        sr_ideal_gross = np.zeros_like(er)
        sb_at_rebars = np.zeros_like(er)
        for mat, bulk_mat, idx in self._rebar_groups:
            er_g = er[idx]
            sr_ideal_gross[idx] = mat.stress_array(er_g)
            sb_at_rebars[idx] = bulk_mat.stress_array(er_g)

        # Net rebar stress (subtract zone-correct bulk at rebar location)
        sr_net = sr_ideal_gross.copy()
        emb = self.sec.embedded_rebars
        sr_net[emb] -= sb_at_rebars[emb]

        return {
            "bulk": {
                "x": self.sec.x_fibers.copy(),
                "y": self.sec.y_fibers.copy(),
                "eps": eb.copy(),
                "sigma": sb.copy(),
                "dA": self.sec.A_fibers.copy(),
            },
            "rebars": {
                "x": self.sec.x_rebars.copy(),
                "y": self.sec.y_rebars.copy(),
                "eps": er.copy(),
                "sigma": sr_ideal_gross.copy(),
                "sigma_net": sr_net.copy(),
                "A": self.sec.A_rebars.copy(),
                "embedded": self.sec.embedded_rebars.copy(),
            },
        }
