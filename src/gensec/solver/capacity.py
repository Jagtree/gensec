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
# along with GenSec. If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------

r"""
Resistance surface generator — N-Mx-My interaction domain.

Phase 2: generates a 3D point cloud in (N, Mx, My) space by
scanning ultimate strain configurations across all curvature
directions.

For uniaxial analysis (chi_y=0), produces the classic N-M diagram.

Performance notes
-----------------
All capacity generators use :meth:`FiberSolver.integrate_batch` to
evaluate hundreds or thousands of strain configurations in a single
vectorized NumPy call, eliminating Python-loop overhead.  The
moment-curvature scanner uses
:meth:`FiberSolver.integrate_with_tangent` for the inner Newton
iteration, halving the cost compared to a finite-difference Jacobian.
"""

import warnings
import numpy as np
from .integrator import FiberSolver

# --- Equilibrium feasibility guard ---------------------------------
# A (eps0, chi) state is physically valid only if the axial residual
#   |N(eps0, chi) - N_target|
# stays below this tolerance. Beyond it the target N is unreachable at
# that curvature: the solver returns the closest non-equilibrium state,
# whose moment is meaningless and must be discarded (set to NaN).
# Absolute + relative form: the 10 N floor dominates as N_target -> 0
# (near-pure bending); the relative term avoids false positives at high
# |N_target|, where 10 N is below the Newton noise.
_FEAS_ABS_TOL = 10.0      # [N]
_FEAS_REL_TOL = 1.0e-4    # [-]


def _feas_tol(N_target):
    """Axial-equilibrium feasibility tolerance [N] for a given target."""
    return max(_FEAS_ABS_TOL, _FEAS_REL_TOL * abs(N_target))


# --- Mx-My domain degeneracy ---------------------------------------
# The interaction domain is declared collapsed (axial limit, M_Rd ~ 0)
# when its outer radius falls below this fraction of a section plastic
# moment scale  M_ref = f_cd * B * H**2 / 4.  The 3+ order-of-magnitude
# gap between a real domain (~1e8 N*mm) and limit noise (<=1e3 N*mm)
# makes the exact coefficient non-critical.
_DOMAIN_DEGEN_FRAC = 1.0e-3


class NMDiagram:
    r"""
    Material-agnostic resistance domain generator.

    For **uniaxial** bending, generates the N-Mx interaction diagram
    (chi_y = 0). For **biaxial** bending, generates a 3D point cloud
    in (N, Mx, My) space by scanning curvature directions in the
    :math:`(\chi_x, \chi_y)` plane.

    The strain plane at any point is:

    .. math::

        \varepsilon(x, y) = \varepsilon_0
            + \chi_x\,(y - y_{\text{ref}})
            - \chi_y\,(x - x_{\text{ref}})

    For the 3D scan, the curvature direction angle :math:`\theta` is:

    .. math::

        \chi_x = \chi \cos\theta, \quad \chi_y = \chi \sin\theta

    and the curvature magnitude :math:`\chi` is determined by the
    strain limits at the extreme fibers for that direction.

    Parameters
    ----------
    solver : FiberSolver
        Configured fiber solver with section and materials.
    include_pivot_a : bool, optional
        If True, include **Pivot A** configurations (steel at
        :math:`\varepsilon_{ud}`, concrete below
        :math:`\varepsilon_{cu2}`) in the biaxial strain template.
        This adds the "bridge branch" that covers the
        axial-force gap between crush-limited and tension
        branches, producing a larger (less conservative) domain
        near :math:`N \approx 0`.

        Per EC2 §6.1 these configurations are valid ULS states.
        However, many commercial tools only consider **Pivot B**
        configurations (:math:`\varepsilon_c = \varepsilon_{cu2}`),
        giving a smaller domain.

        Default ``False`` (Pivot B only, matches most commercial
        tools).  Set to ``True`` for the full EC2 domain.

    Notes
    -----
    Strain limits and lever arms are computed once at construction
    and cached for the lifetime of the object.  This avoids
    redundant loops over materials and rebars in every method call.
    If the section or material properties change, a new
    :class:`NMDiagram` instance must be created.
    """

    def __init__(self, solver, include_pivot_a=True):
        self.solver = solver
        self.include_pivot_a = include_pivot_a

        # ----- cached strain limits (immutable during object life) -----
        sec = solver.sec
        b = sec.bulk_material
        emin_g, emax_g = b.eps_min, b.eps_max
        for r in sec.rebars:
            emin_g = min(emin_g, r.material.eps_min)
            emax_g = max(emax_g, r.material.eps_max)
        self._emg = emin_g        # eps_min_global
        self._exg = emax_g        # eps_max_global
        self._emb = b.eps_min     # eps_min_bulk (crush)
        self._exb = b.eps_max     # eps_max_bulk
        self._fcd_bulk = getattr(b, 'fcd', None)  # design strength [N/mm²]

        # ----- cached lever arms -----
        self._all_lx = np.concatenate([
            sec.x_fibers - solver.x_ref,
            sec.x_rebars - solver.x_ref,
        ])
        self._all_ly = np.concatenate([
            sec.y_fibers - solver.y_ref,
            sec.y_rebars - solver.y_ref,
        ])

        # ----- cached rebar yield strains -----
        self._rebar_eps_yd = []
        for rb in sec.rebars:
            if hasattr(rb.material, 'eps_yd'):
                self._rebar_eps_yd.append(rb.material.eps_yd)
        self._eps_yd_min = (min(self._rebar_eps_yd)
                           if self._rebar_eps_yd else None)

    # ------------------------------------------------------------------
    #  Strain limits accessor (backward-compatible tuple interface)
    # ------------------------------------------------------------------

    def _collect_strain_limits(self):
        """
        Global strain envelope from all materials.

        Returns the cached values computed at construction time.

        Returns
        -------
        eps_min_global : float
            Most compressive strain across all materials.
        eps_max_global : float
            Most tensile strain across all materials.
        eps_min_bulk : float
            Bulk material compressive limit.
        eps_max_bulk : float
            Bulk material tensile limit.
        """
        return self._emg, self._exg, self._emb, self._exb

    # ------------------------------------------------------------------
    #  Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_positive_int(value, name):
        """
        Raise ``ValueError`` if *value* is not a positive integer.

        Parameters
        ----------
        value : int
            Value to check.
        name : str
            Parameter name for error messages.

        Raises
        ------
        ValueError
        """
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(
                f"{name} must be a positive integer, got {value!r}")

    @staticmethod
    def _validate_finite_float(value, name):
        """
        Raise ``ValueError`` if *value* is not a finite float.

        Parameters
        ----------
        value : float
            Value to check.
        name : str
            Parameter name for error messages.

        Raises
        ------
        ValueError
        """
        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite, got {value!r}")

    # ==================================================================
    #  Configuration builders (return arrays, not lists of tuples)
    # ==================================================================

    def _ultimate_strain_configs_1d(self, direction='x', n_points=200):
        r"""
        Generate arrays of ``(eps0, chi)`` for single-axis bending.

        For ``direction='x'`` (bending about x-axis, curvature
        :math:`\chi_x`), the section depth is ``H`` and the reference
        lever arm is ``y_ref``.

        For ``direction='y'`` (bending about y-axis, curvature
        :math:`\chi_y`), the section depth is ``B`` and the reference
        lever arm is ``x_ref``.

        Four branches cover the full :math:`(N, M)` space:

        1. **Crush-limited** (positive and negative curvature):
           one edge at :math:`\varepsilon_{cu}`, the other sweeps
           from max tension to crush.
        2. **Near-uniform compression**: both edges near
           :math:`\varepsilon_{cu}`, with small perturbations.
        3. **Pure tension**: both edges in the tensile range.
           Avoids the trivial :math:`\varepsilon = 0` configuration
           to prevent degeneracies in the solver.
        4. **Bridge** (Pivot A): one edge at max steel strain
           :math:`\varepsilon_{ud}`, the other sweeps from zero to
           bulk crush.  Fills the axial-force gap between
           crush-limited and tension branches for sections with
           large :math:`A_c / A_s` ratios.

        Parameters
        ----------
        direction : ``'x'`` or ``'y'``
        n_points : int

        Returns
        -------
        eps0_arr : numpy.ndarray
        chi_arr : numpy.ndarray
            Curvature arrays ready for :meth:`integrate_batch`.
        """
        emg = self._emg
        exg = self._exg
        emb = self._emb
        sec = self.solver.sec

        # Bbox edges along the bending direction (``sec.bbox`` returns
        # ``(minx, miny, maxx, maxy)``).  We need the *physical* lever
        # arm from the bottom edge of the section to the integrator's
        # reference point (``y_ref`` for direction='x', ``x_ref`` for
        # direction='y'); using ``y_ref`` directly is correct only when
        # ``y_min == 0``, which holds for every standard primitive but
        # not for custom polygons defined on a centred frame.
        minx, miny, maxx, maxy = sec.bbox
        if direction == 'x':
            depth = sec.H
            lever = self.solver.y_ref - miny
        else:
            depth = sec.B
            lever = self.solver.x_ref - minx

        eps0_list = []
        chi_list = []

        def _append(ei, es):
            r"""
            Convert (bot, top) edge strains to (eps0, chi) and append.

            Strain field along the bending direction:

            .. math::

                \varepsilon(\xi)
                = \varepsilon_i + \chi\,(\xi - \xi_{\min})

            with :math:`\chi = (\varepsilon_s - \varepsilon_i) / H`.
            Evaluating at :math:`\xi = \xi_{\text{ref}}` gives:

            .. math::

                \varepsilon_0
                = \varepsilon_i + \chi\,(\xi_{\text{ref}} - \xi_{\min}).

            The lever arm is the *physical* distance from the bottom
            edge of the section to the reference point, which makes
            the conversion invariant under translations of the
            coordinate frame.  This is what guarantees a
            mirror-symmetric N-M cloud for a doubly symmetric section
            defined on a centred frame.
            """
            chi = (es - ei) / depth if depth > 0 else 0.0
            eps0 = ei + chi * lever
            eps0_list.append(eps0)
            chi_list.append(chi)

        # Branch 1: crush-limited
        # Positive curvature: "top" (max coord) compressed
        for ei in np.linspace(exg, emg, n_points):
            _append(ei, emb)
        for es in np.linspace(emb, emb * 0.5, n_points // 2):
            _append(emb, es)

        # Negative curvature: "bottom" (min coord) compressed
        for es in np.linspace(exg, emg, n_points):
            _append(emb, es)
        for ei in np.linspace(emb, emb * 0.5, n_points // 2):
            _append(ei, emb)

        # Branch 2: near-uniform compression
        for ev in np.linspace(emb * 0.5, emb, n_points // 4):
            _append(ev, ev)
            span = abs(emb) * 0.15
            for d in np.linspace(-span, span, 5):
                _append(ev + d, ev - d)

        # Branch 3: pure tension (no zero-strain point)
        eps_min_t = exg / max(n_points // 4, 1)
        for ev in np.linspace(eps_min_t, exg, n_points // 4):
            _append(ev, eps_min_t)
            _append(eps_min_t, ev)
            _append(exg, ev)
            _append(ev, exg)

        # Branch 4: bridge — variable compression at one edge,
        # max steel strain at the other.  Fills the N-gap between
        # crush-limited and tension branches for sections with
        # large Ac/As ratio.  Only included when include_pivot_a=True.
        if self.include_pivot_a:
            n_bridge = n_points // 2
            for comp in np.linspace(0, emb, n_bridge):
                _append(exg, comp)      # bottom tension, top crush
                _append(comp, exg)      # bottom crush, top tension

        return np.array(eps0_list), np.array(chi_list)

    # ==================================================================
    #  Uniaxial N-M diagram (batch)
    # ==================================================================

    def generate(self, n_points=200, direction='x'):
        r"""
        Generate the N-M interaction diagram for a single bending
        direction.

        For ``direction='x'``: N-Mx diagram (:math:`\chi_y = 0`).
        For ``direction='y'``: N-My diagram (:math:`\chi_x = 0`).

        All strain configurations are evaluated in a **single batch
        call**, avoiding per-configuration Python-loop overhead.

        Parameters
        ----------
        n_points : int, optional
            Base resolution. Default 200.
        direction : ``'x'`` or ``'y'``, optional
            Bending direction. Default ``'x'``.

        Returns
        -------
        dict
            ``N``, ``M`` [N, N*mm], ``N_kN``, ``M_kNm``.
            Also ``Mx`` or ``My`` alias depending on direction.

        Raises
        ------
        ValueError
            If *n_points* < 1 or *direction* not in ``{'x', 'y'}``.
        """
        self._validate_positive_int(n_points, "n_points")
        if direction not in ('x', 'y'):
            raise ValueError(
                f"direction must be 'x' or 'y', got {direction!r}")

        eps0_arr, chi_arr = self._ultimate_strain_configs_1d(
            direction, n_points)
        n = len(eps0_arr)

        if direction == 'x':
            Na, Mxa, Mya = self.solver.integrate_batch(
                eps0_arr, chi_arr, np.zeros(n))
            Ma = Mxa
        else:
            Na, Mxa, Mya = self.solver.integrate_batch(
                eps0_arr, np.zeros(n), chi_arr)
            Ma = Mya

        M_key = "Mx" if direction == 'x' else "My"
        return {
            "N": Na, "M": Ma, M_key: Ma,
            "N_kN": Na / 1e3, "M_kNm": Ma / 1e6,
            f"{M_key}_kNm": Ma / 1e6,
            "direction": direction,
        }

    # ==================================================================
    #  Shared helpers for biaxial generators
    # ==================================================================

    def _build_edge_template(self, n_points):
        r"""
        Pre-build the edge-strain template shared by all curvature
        directions.

        The pattern of ``(eps_bot, eps_top)`` pairs is identical for
        every angle — only the projection parameters differ.
        Building it once eliminates per-angle Python loops.

        Four (or five) branches cover the full :math:`(N, M)` space:

        1. **Top at crush**: bottom varies from max tension to crush.
        2. **Bottom at crush**: top varies from max tension to crush.
        3. **Near-uniform compression**: both edges near crush strain.
        4. **Pure tension**: both edges in the tensile range, with
           multiple gradient variants for thorough coverage of the
           tension-dominated region.
        5. **Bridge** (optional, ``include_pivot_a=True``): one edge
           at max steel strain, the other sweeps from zero to bulk
           crush.  Bridges the axial-force gap between crush-limited
           and tension branches — critical for sections with high
           :math:`A_c / A_s` ratios.

        Parameters
        ----------
        n_points : int
            Base resolution per branch.

        Returns
        -------
        ebot : numpy.ndarray
        etop : numpy.ndarray
        """
        emg = self._emg
        exg = self._exg
        emb = self._emb
        n = n_points

        ebot_parts = []
        etop_parts = []

        # Branch 1: top at bulk crush
        b1 = np.linspace(exg, emg, n)
        ebot_parts.append(b1)
        etop_parts.append(np.full(n, emb))

        # Branch 2: bottom at bulk crush
        b2 = np.linspace(exg, emg, n)
        ebot_parts.append(np.full(n, emb))
        etop_parts.append(b2)

        # Branch 3: near-uniform compression
        n3 = n // 4
        ev3 = np.linspace(emb * 0.5, emb, n3)
        sp = abs(emb) * 0.15
        d3 = np.linspace(-sp, sp, 3)
        for ev in ev3:
            ebot_parts.append(np.array([ev]))
            etop_parts.append(np.array([ev]))
            ebot_parts.append(ev + d3)
            etop_parts.append(ev - d3)

        # Branch 4: pure tension — full coverage with gradient
        # variants to ensure thorough sampling of the tension-
        # dominated region.
        n4 = n // 2
        ev4 = np.linspace(0, exg, n4)
        # Uniform tension
        ebot_parts.append(ev4)
        etop_parts.append(ev4.copy())
        # Gradient variants
        ebot_parts.append(ev4)
        etop_parts.append(np.zeros(n4))
        ebot_parts.append(np.zeros(n4))
        etop_parts.append(ev4)
        ebot_parts.append(ev4)
        etop_parts.append(ev4 * 0.5)
        ebot_parts.append(ev4 * 0.5)
        etop_parts.append(ev4)

        # Branch 5: bridge — variable compression at one edge,
        # max steel strain at the other.
        # Only included when include_pivot_a=True (full EC2 domain).
        if self.include_pivot_a:
            n5 = n // 2
            comp_sweep = np.linspace(0, emb, n5)
            ebot_parts.append(np.full(n5, exg))
            etop_parts.append(comp_sweep)
            ebot_parts.append(comp_sweep.copy())
            etop_parts.append(np.full(n5, exg))

        return np.concatenate(ebot_parts), np.concatenate(etop_parts)

    def _build_edge_template_mx_my(self, n_points):
        r"""
        **Asymmetric** edge-strain template for :meth:`generate_mx_my`.

        This template is deliberately **one-sided**: each branch
        produces curvatures of a single sign.  The opposite curvature
        sign is obtained naturally when the angle :math:`\theta`
        advances by :math:`\pi` — the projection axis
        :math:`p(\theta) = l_y \cos\theta - l_x \sin\theta` flips,
        swapping "bottom" and "top" fibers for the same edge-strain
        pair.

        A **symmetric** template (containing both ``(a, b)`` and
        ``(b, a)`` for every pair) makes :math:`\theta` and
        :math:`\theta + \pi` produce identical ``(N, M_x, M_y)``
        triplets.  The interpolated Mx-My contour then covers only
        one half of the moment plane — which is incorrect.

        Five branches:

        1. **Crush-limited**: bottom varies from max tension to
           crush; top fixed at :math:`\varepsilon_{cu}`.
           Covers the deep-compression N range.
        2. **Near-uniform compression**: both edges near
           :math:`\varepsilon_{cu}`, with small perturbations.
        3. **Tension**: bottom varies from 0 to max tension;
           top fixed at 0.  Covers the pure-tension N range.
        4. **Bridge**: bottom at :math:`\varepsilon_{ud}`;
           top sweeps from 0 to :math:`\varepsilon_{cu}`.
           Provides N-crossings near :math:`N \approx 0` with
           significant moments — essential for sections with
           large :math:`A_c / A_s` ratios.

        Parameters
        ----------
        n_points : int
            Base resolution per branch.

        Returns
        -------
        ebot : numpy.ndarray
        etop : numpy.ndarray
        """
        emg = self._emg
        exg = self._exg
        emb = self._emb
        n = n_points

        ebot_parts = []
        etop_parts = []

        # Branch 1: crush-limited — ONE-SIDED (no mirror).
        # Bottom edge varies from max tension to max compression;
        # top edge fixed at bulk crush.
        b1 = np.linspace(exg, emg, n)
        ebot_parts.append(b1)
        etop_parts.append(np.full(n, emb))

        # Branch 2: near-uniform compression.
        # Inherently near-symmetric (chi ≈ 0) — no issue.
        n3 = n // 4
        ev3 = np.linspace(emb * 0.5, emb, n3)
        sp = abs(emb) * 0.15
        d3 = np.linspace(-sp, sp, 5)
        for ev in ev3:
            ebot_parts.append(np.array([ev]))
            etop_parts.append(np.array([ev]))
            ebot_parts.append(ev + d3)
            etop_parts.append(ev - d3)

        # Branch 3: tension — ONE-SIDED.
        # Bottom from 0 to max tension; top at 0.
        n4 = n // 4
        ev4 = np.linspace(0, exg, n4)
        ebot_parts.append(ev4)
        etop_parts.append(np.zeros(n4))

        # Branch 4: bridge — ONE-SIDED.
        # Bottom at max steel strain; top sweeps from 0
        # to bulk crush.  Only included when include_pivot_a=True.
        if self.include_pivot_a:
            n5 = n // 2
            comp_sweep = np.linspace(0, emb, n5)
            ebot_parts.append(np.full(n5, exg))
            etop_parts.append(comp_sweep)

        return np.concatenate(ebot_parts), np.concatenate(etop_parts)

    def _chi_max_for_direction(self, cos_t, sin_t, all_lx, all_ly):
        r"""
        Maximum curvature magnitude for a given curvature direction.

        The maximum curvature is determined by the strain range that
        does not violate material limits divided by the projected
        section depth along that direction:

        .. math::

            \chi_{\max}(\theta) = \frac{\varepsilon_{\max,\text{global}}
                - \varepsilon_{\min,\text{bulk}}}
                {\text{span}(\theta)}

        where :math:`\text{span}(\theta)` is the distance between the
        extreme fibers projected onto direction :math:`\theta`.

        Parameters
        ----------
        cos_t, sin_t : float
            Direction cosines of the curvature direction angle.
        all_lx, all_ly : numpy.ndarray
            Lever arms of all fibers and rebars.

        Returns
        -------
        chi_max : float
            Maximum curvature magnitude [1/mm].
        span : float
            Projected depth [mm].
        p_min : float
            Minimum projection coordinate [mm].
        """
        proj = all_ly * cos_t - all_lx * sin_t
        p_min = proj.min()
        p_max = proj.max()
        span = p_max - p_min
        if span < 1e-10:
            return 0.0, span, p_min
        chi_max = (self._exg - self._emb) / span
        return chi_max, span, p_min

    def _compute_angle_params(self, thetas, all_lx, all_ly):
        r"""
        Compute projection parameters for each curvature direction.

        Parameters
        ----------
        thetas : numpy.ndarray
            Curvature direction angles [rad].
        all_lx, all_ly : numpy.ndarray
            Lever arms of all fibers + rebars.

        Returns
        -------
        list of tuple
            ``(cos_t, sin_t, p_min, span)`` for each valid angle.
            Degenerate angles (zero span) are skipped.
        """
        params = []
        for theta in thetas:
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            proj = all_ly * cos_t - all_lx * sin_t
            p_max = proj.max()
            p_min = proj.min()
            span = p_max - p_min
            if span < 1e-10:
                continue
            params.append((cos_t, sin_t, p_min, span))
        return params

    def _mega_batch_integrate(self, ebot_tmpl, etop_tmpl, angle_params):
        r"""
        Build configs for all angles and integrate in chunked
        mega-batches.

        Instead of calling :meth:`integrate_batch` once per angle
        (high per-call overhead), this method concatenates configs
        across multiple angles and processes them in large chunks.

        Parameters
        ----------
        ebot_tmpl, etop_tmpl : numpy.ndarray
            Edge-strain template arrays (identical for every angle).
        angle_params : list of tuple
            ``(cos_t, sin_t, p_min, span)`` per angle.

        Returns
        -------
        list of tuple
            ``(N_array, Mx_array, My_array)`` per angle, in the
            same order as *angle_params*.
        """
        n_per = len(ebot_tmpl)
        n_angles = len(angle_params)
        total = n_per * n_angles

        # Pre-allocate flat config arrays
        all_eps0 = np.empty(total)
        all_chi_x = np.empty(total)
        all_chi_y = np.empty(total)

        for i, (cos_t, sin_t, p_min, span) in enumerate(angle_params):
            s = i * n_per
            e = s + n_per
            chi = (etop_tmpl - ebot_tmpl) / span
            all_eps0[s:e] = ebot_tmpl - chi * p_min
            all_chi_x[s:e] = chi * cos_t
            all_chi_y[s:e] = chi * sin_t

        # Chunked integration to limit memory usage.
        # Peak per chunk: 4 live (n, n_fibers) matrices inside
        # integrate_batch (strain, stress, force, moment-temp).
        # At 12M elements ÷ n_fibers configs, each matrix is ~96 MB,
        # giving ~400 MB true peak per chunk.
        n_fibers = self.solver.sec.n_fibers
        max_configs = max(500, 12_000_000 // max(n_fibers, 1))

        all_N = np.empty(total)
        all_Mx = np.empty(total)
        all_My = np.empty(total)

        for cs in range(0, total, max_configs):
            ce = min(cs + max_configs, total)
            N_c, Mx_c, My_c = self.solver.integrate_batch(
                all_eps0[cs:ce], all_chi_x[cs:ce], all_chi_y[cs:ce])
            all_N[cs:ce] = N_c
            all_Mx[cs:ce] = Mx_c
            all_My[cs:ce] = My_c

        # Split by angle
        results = []
        for i in range(n_angles):
            s = i * n_per
            e = s + n_per
            results.append((all_N[s:e], all_Mx[s:e], all_My[s:e]))
        return results

    # ==================================================================
    #  Biaxial 3-D surface (mega-batch)
    # ==================================================================

    def generate_biaxial(self, n_angles=36, n_points_per_angle=80):
        r"""
        Generate the 3D resistance surface (N, Mx, My).

        Scans curvature directions :math:`\theta` in
        :math:`[0, 2\pi)` and, for each direction, scans curvature
        magnitudes through ultimate strain configurations.

        All directions are integrated in **chunked mega-batches**,
        reducing Python-loop overhead by an order of magnitude
        compared to per-angle batching.

        Parameters
        ----------
        n_angles : int, optional
            Number of curvature direction angles. Default 36
            (every 10°).
        n_points_per_angle : int, optional
            Strain configurations per angle. Default 80.

        Returns
        -------
        dict
            ``N`` [N], ``Mx``, ``My`` [N*mm], and ``_kN``/``_kNm``
            variants.

        Raises
        ------
        ValueError
            If *n_angles* or *n_points_per_angle* < 1.
        """
        self._validate_positive_int(n_angles, "n_angles")
        self._validate_positive_int(n_points_per_angle,
                                    "n_points_per_angle")

        thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        ebot, etop = self._build_edge_template(n_points_per_angle)
        angle_params = self._compute_angle_params(
            thetas, self._all_lx, self._all_ly)

        per_angle = self._mega_batch_integrate(
            ebot, etop, angle_params)

        Nl = [r[0] for r in per_angle]
        Mxl = [r[1] for r in per_angle]
        Myl = [r[2] for r in per_angle]

        Na = np.concatenate(Nl)
        Mxa = np.concatenate(Mxl)
        Mya = np.concatenate(Myl)

        return {
            "N": Na, "Mx": Mxa, "My": Mya,
            "N_kN": Na / 1e3,
            "Mx_kNm": Mxa / 1e6,
            "My_kNm": Mya / 1e6,
        }

    # ==================================================================
    #  Vectorized Newton solve for fixed N
    # ==================================================================

    def _vectorized_solve_N(self, N_fixed, chi_x_arr, chi_y_arr,
                            eps0_init=None, n_iter=15, tol=1e3,
                            delta=1e-7):
        r"""
        Solve for :math:`\varepsilon_0` at fixed :math:`N` for many
        curvature configurations simultaneously.

        Uses a fully vectorized Newton–Raphson iteration: each
        iteration consists of two :meth:`integrate_batch` calls
        (one for the residual, one for the numerical Jacobian
        :math:`dN/d\varepsilon_0`), with no Python loops over
        configurations.

        The iteration exits early when all residuals satisfy
        :math:`|N - N_{\text{fixed}}| < \text{tol}`, avoiding
        unnecessary batch calls when convergence is fast.

        Parameters
        ----------
        N_fixed : float
            Target axial force [N].
        chi_x_arr, chi_y_arr : numpy.ndarray
            Curvature components for each configuration [1/mm].
        eps0_init : numpy.ndarray or None
            Initial guesses.  If None, computed from elastic theory.
        n_iter : int, optional
            Maximum number of Newton iterations.  Default 15.
        tol : float, optional
            Convergence tolerance on :math:`|N - N_{\text{fixed}}|`
            [N].  Default 1000 N (= 1 kN).
        delta : float, optional
            Finite-difference step for :math:`dN/d\varepsilon_0`.
            Default 1e-7.

        Returns
        -------
        eps0 : numpy.ndarray
            Converged :math:`\varepsilon_0` for each configuration.
        N_arr : numpy.ndarray
            Axial force at convergence [N].
        Mx_arr, My_arr : numpy.ndarray
            Moments at convergence [N·mm].
        """
        sv = self.solver
        n = len(chi_x_arr)
        emb = self._emb

        # Initial guess
        if eps0_init is None:
            sec = sv.sec
            A_gross = getattr(sec, 'ideal_gross_area', sec.B * sec.H)
            E_est = 15000.0  # rough estimate for initial guess
            eps0 = np.full(n, N_fixed / max(A_gross * E_est, 1.0))
            eps0 = np.clip(eps0, emb * 1.2, -emb * 1.2)
        else:
            eps0 = eps0_init.copy()

        # Active-set Newton: only integrate non-converged configs.
        n = len(chi_x_arr)
        N_arr  = np.empty(n)
        Mx_arr = np.empty(n)
        My_arr = np.empty(n)
        active = np.ones(n, dtype=bool)

        for _ in range(n_iter):
            idx = np.nonzero(active)[0]
            if idx.size == 0:
                break
            Na, Mxa, Mya = sv.integrate_batch(
                eps0[idx], chi_x_arr[idx], chi_y_arr[idx])
            N_arr[idx], Mx_arr[idx], My_arr[idx] = Na, Mxa, Mya

            residual = Na - N_fixed
            conv = np.abs(residual) < tol
            todo = ~conv
            if todo.any():
                sub = idx[todo]
                N_pert, _, _ = sv.integrate_batch(
                    eps0[sub] + delta, chi_x_arr[sub], chi_y_arr[sub])
                dNde = (N_pert - Na[todo]) / delta
                good = np.abs(dNde) > 1.0
                step = np.zeros_like(dNde)
                step[good] = -residual[todo][good] / dNde[good]
                eps0[sub] += np.clip(step, -0.002, 0.002)
                active[sub[~good]] = False  # collapsed tangent → drop
            active[idx[conv]] = False       # converged → drop

        return eps0, N_arr, Mx_arr, My_arr

    # ==================================================================
    #  Mx-My contour at fixed N (mega-batch + interpolation)
    # ==================================================================

    def generate_mx_my(self, N_fixed, n_angles=72,
                       n_points_per_angle=200, n_chi=20, n_scan=None):
        # TODO: remove anything that concerns n_points_per_angle from the API, since it's not used internally.
        r"""
        Generate the Mx-My interaction contour at a fixed axial force.

        Algorithm:

        1. For each curvature direction :math:`\theta \in [0, 2\pi)`,
           sweeps curvature magnitude :math:`\chi` and solves
           :math:`\varepsilon_0` via vectorized Newton so that
           :math:`N = N_{\text{fixed}}`.
        2. Collects **all** converged :math:`(M_x, M_y)` points
           across all angles and curvatures.
        3. Builds the 2-D convex hull of the converged cloud.
        4. Resamples the hull boundary at ``n_angles`` evenly-spaced
           angular positions to produce a smooth, closed contour
           with exactly ``n_angles`` points.

        This approach guarantees that the contour is the true
        outer boundary of the capacity at :math:`N_{\text{fixed}}`,
        not limited to a single "best" point per angle.

        Parameters
        ----------
        N_fixed : float
            Fixed axial force [N].
        n_angles : int, optional
            Number of output contour points. Default 72.
        n_points_per_angle : int, optional
            Kept for API compatibility; not used internally.
        n_chi : int, optional
            Curvature magnitudes per direction. Default 20.

        Returns
        -------
        dict
            ``Mx`` [N*mm], ``My`` [N*mm], ``Mx_kNm``, ``My_kNm``,
            ``N_fixed_kN``.

        Raises
        ------
        ValueError
            If *n_angles* < 1 or *N_fixed* is not finite.

        Warnings
        --------
        RuntimeWarning
            If fewer than 3 points converge (cannot build hull).
        """
        from scipy.spatial import ConvexHull, QhullError

        self._validate_finite_float(N_fixed, "N_fixed")
        self._validate_positive_int(n_angles, "n_angles")

        all_lx = self._all_lx
        all_ly = self._all_ly

        if n_scan is None:
            n_scan = min(max(n_angles, 72), 120)
        thetas = np.linspace(0, 2 * np.pi, n_scan, endpoint=False)

        chi_x_parts = []
        chi_y_parts = []

        for theta in thetas:
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            chi_max, span, _ = self._chi_max_for_direction(
                cos_t, sin_t, all_lx, all_ly)
            if chi_max < 1e-15:
                continue
            frac = np.linspace(1.0 / n_chi, 1.0, n_chi)
            chis = chi_max * frac ** 0.7
            chi_x_parts.append(chis * cos_t)
            chi_y_parts.append(chis * sin_t)

        if len(chi_x_parts) == 0:
            Mxa = np.full(n_angles, np.nan)
            Mya = np.full(n_angles, np.nan)
            warnings.warn(
                "generate_mx_my: all curvature directions degenerate.",
                RuntimeWarning, stacklevel=2)
            return {
                "Mx": Mxa, "My": Mya,
                "Mx_kNm": Mxa / 1e6, "My_kNm": Mya / 1e6,
                "N_fixed_kN": N_fixed / 1e3,
            }

        all_chi_x = np.concatenate(chi_x_parts)
        all_chi_y = np.concatenate(chi_y_parts)

        # Vectorized Newton: solve eps0 for all configs at once.
        ftol = _feas_tol(N_fixed)
        _, N_arr, Mx_arr, My_arr = self._vectorized_solve_N(
            N_fixed, all_chi_x, all_chi_y, tol=ftol)

        # Filter converged points.
        conv = np.abs(N_arr - N_fixed) <= ftol
        Mx_conv = Mx_arr[conv]
        My_conv = My_arr[conv]

        if len(Mx_conv) < 3:
            Mxa = np.full(n_angles, np.nan)
            Mya = np.full(n_angles, np.nan)
            warnings.warn(
                f"generate_mx_my: only {len(Mx_conv)} converged "
                f"points at N={N_fixed / 1e3:.1f} kN (need >= 3).",
                RuntimeWarning, stacklevel=2)
            return {
                "Mx": Mxa, "My": Mya,
                "Mx_kNm": Mxa / 1e6, "My_kNm": Mya / 1e6,
                "N_fixed_kN": N_fixed / 1e3,
            }

        # Collapsed domain check: the domain degenerates at the axial limits.
        sec = self.solver.sec
        f_cd = abs(self._fcd_bulk) if self._fcd_bulk is not None else 0.0
        M_ref = f_cd * sec.B * sec.H ** 2 / 4.0
        R = float(np.hypot(Mx_conv, My_conv).max())
        if M_ref > 0 and R < _DOMAIN_DEGEN_FRAC * M_ref:
            Mxa = np.full(n_angles, np.nan)
            Mya = np.full(n_angles, np.nan)
            warnings.warn(
                f"generate_mx_my: collapsed domain at "
                f"N={N_fixed / 1e3:.1f} kN (R={R:.2e} N*mm "
                f"< {_DOMAIN_DEGEN_FRAC * M_ref:.2e}); "
                f"M_Rd ≈ 0 at axial limit.",
                RuntimeWarning, stacklevel=2)
            return {"Mx": Mxa, "My": Mya,
                    "Mx_kNm": Mxa / 1e6, "My_kNm": Mya / 1e6,
                    "N_fixed_kN": N_fixed / 1e3, "degenerate": True}

        # Build convex hull of converged cloud.
        pts = np.column_stack([Mx_conv, My_conv])
        try:
            hull = ConvexHull(pts)
        except QhullError:
            Mxa = np.full(n_angles, np.nan)
            Mya = np.full(n_angles, np.nan)
            warnings.warn(
                f"generate_mx_my: degenerate hull at "
                f"N={N_fixed / 1e3:.1f} kN.",
                RuntimeWarning, stacklevel=2)
            return {
                "Mx": Mxa, "My": Mya,
                "Mx_kNm": Mxa / 1e6, "My_kNm": Mya / 1e6,
                "N_fixed_kN": N_fixed / 1e3,
            }

        # Extract ordered hull boundary (closed polygon).
        hv = hull.vertices
        hull_Mx = pts[hv, 0]
        hull_My = pts[hv, 1]

        # Centroid for angular parameterization.
        cx = hull_Mx.mean()
        cy = hull_My.mean()

        # Compute angle of each hull vertex from centroid.
        angles_hull = np.arctan2(hull_My - cy, hull_Mx - cx)
        order = np.argsort(angles_hull)
        hull_Mx = hull_Mx[order]
        hull_My = hull_My[order]
        angles_hull = angles_hull[order]

        # Close the polygon.
        hull_Mx = np.append(hull_Mx, hull_Mx[0])
        hull_My = np.append(hull_My, hull_My[0])
        angles_hull = np.append(angles_hull,
                                angles_hull[0] + 2 * np.pi)

        # Resample at n_angles evenly-spaced angular positions.
        target_angles = np.linspace(
            angles_hull[0], angles_hull[0] + 2 * np.pi,
            n_angles, endpoint=False)

        Mxa = np.interp(target_angles, angles_hull, hull_Mx,
                         period=2 * np.pi)
        Mya = np.interp(target_angles, angles_hull, hull_My,
                         period=2 * np.pi)

        return {
            "Mx": Mxa, "My": Mya,
            "Mx_kNm": Mxa / 1e6, "My_kNm": Mya / 1e6,
            "N_fixed_kN": N_fixed / 1e3,
            "degenerate": False,
        }

    # ==================================================================
    #  Fast Mx-My slice from pre-computed 3D surface
    # ==================================================================

    @staticmethod
    def slice_mx_my_at_N(nm_3d, N_fixed_kN, n_angles=72,
                         band_kN=None):
        r"""
        Slice the 3D resistance surface at a fixed N level.

        Instead of re-running the full vectorised Newton solve
        (as :meth:`generate_mx_my` does), this method filters
        the existing ``(N, Mx, My)`` point cloud within an
        adaptive band around ``N_fixed_kN`` and builds a 2-D
        convex hull of the filtered ``(Mx, My)`` points.

        Typical cost: **< 1 ms** (array filter + 2-D ConvexHull),
        versus ~200 ms for a fresh ``generate_mx_my`` call.
        Designed for the GUI N-slider, where low latency matters
        more than sub-kN accuracy.

        Parameters
        ----------
        nm_3d : dict
            Output of :meth:`generate_biaxial`.  Must contain
            ``N_kN``, ``Mx_kNm``, ``My_kNm``.
        N_fixed_kN : float
            Target axial force [kN].
        n_angles : int, optional
            Angular resolution of the output contour.  Default 72.
        band_kN : float or None, optional
            Half-width of the N-band used for filtering [kN].
            If ``None``, set adaptively to 5 %% of the total N range
            (minimum 10 kN).

        Returns
        -------
        dict or None
            Same shape as :meth:`generate_mx_my` output
            (``Mx``, ``My``, ``Mx_kNm``, ``My_kNm``,
            ``N_fixed_kN``), or ``None`` if fewer than 3 points
            fall within the band.

        Notes
        -----
        The accuracy of the slice depends on the density of the
        underlying point cloud.  For demand verification (where
        sub-kN precision matters), use :meth:`generate_mx_my`
        with the full Newton solver instead.
        """
        from scipy.spatial import ConvexHull, QhullError

        N = nm_3d["N_kN"]
        Mx = nm_3d["Mx_kNm"]
        My = nm_3d["My_kNm"]

        if band_kN is None:
            band_kN = max(10.0, (N.max() - N.min()) * 0.05)

        mask = np.abs(N - N_fixed_kN) <= band_kN
        if mask.sum() < 3:
            return None

        mx_band = Mx[mask]
        my_band = My[mask]
        pts = np.column_stack([mx_band, my_band])

        try:
            hull = ConvexHull(pts)
        except QhullError:
            return None

        hv = hull.vertices
        hull_Mx = pts[hv, 0]
        hull_My = pts[hv, 1]

        # Centroid for angular parameterization.
        cx = hull_Mx.mean()
        cy = hull_My.mean()
        angles_hull = np.arctan2(hull_My - cy, hull_Mx - cx)
        order = np.argsort(angles_hull)
        hull_Mx = hull_Mx[order]
        hull_My = hull_My[order]
        angles_hull = angles_hull[order]

        # Close polygon.
        hull_Mx = np.append(hull_Mx, hull_Mx[0])
        hull_My = np.append(hull_My, hull_My[0])
        angles_hull = np.append(angles_hull,
                                angles_hull[0] + 2 * np.pi)

        # Resample at n_angles evenly-spaced positions.
        target = np.linspace(angles_hull[0],
                             angles_hull[0] + 2 * np.pi,
                             n_angles, endpoint=False)
        Mxa = np.interp(target, angles_hull, hull_Mx,
                         period=2 * np.pi)
        Mya = np.interp(target, angles_hull, hull_My,
                         period=2 * np.pi)

        # Return in Nmm + kNm for consistency with generate_mx_my.
        return {
            "Mx": Mxa * 1e6, "My": Mya * 1e6,
            "Mx_kNm": Mxa, "My_kNm": Mya,
            "N_fixed_kN": float(N_fixed_kN),
        }

    # ==================================================================
    #  Per-demand utilization ratio
    # ==================================================================

    def eta_demand(self, N_demand, Mx_demand, My_demand,
                   n_angles=72, n_points_per_angle=200):
        r"""
        Utilization ratio and ductility for a single demand point.

        Generates the Mx-My contour at :math:`N = N_d` via
        :meth:`generate_mx_my` (sweep + interpolation), builds a
        2-D convex hull, and computes the ray-cast intersection
        to find :math:`\eta_{2D}`.

        Using the same contour algorithm as the plotted contour
        guarantees that the :math:`\eta` value is **consistent**
        with the visual output.

        Parameters
        ----------
        N_demand : float
            Design axial force [N].
        Mx_demand, My_demand : float
            Design moments [N·mm].
        n_angles : int, optional
            Number of curvature directions for the contour.
            Default 72.
        n_points_per_angle : int, optional
            Strain scan resolution per direction.  Default 200.

        Returns
        -------
        dict
            ``eta`` — utilization ratio (:math:`\eta_{2D}` at
            :math:`N = N_d`; < 1 safe, > 1 unsafe).
            ``M_Rd_kNm`` — capacity on the contour boundary in
            the demand direction [kN·m].
            ``M_Ed_kNm`` — demand resultant [kN·m].
            ``alpha_deg`` — demand moment direction [°].
            ``ductility`` — :math:`\mu = \chi_u / \chi_y` at
            :math:`N_d` in the curvature direction closest to
            :math:`\alpha`.

        Raises
        ------
        ValueError
            If any input is non-finite.
        """
        from scipy.spatial import ConvexHull, QhullError

        self._validate_finite_float(N_demand, "N_demand")
        self._validate_finite_float(Mx_demand, "Mx_demand")
        self._validate_finite_float(My_demand, "My_demand")

        M_Ed = np.sqrt(Mx_demand**2 + My_demand**2)
        if M_Ed < 1e-6:
            return {"eta": 0.0, "M_Rd_kNm": 0.0,
                    "M_Ed_kNm": 0.0, "alpha_deg": 0.0,
                    "ductility": None}

        alpha = np.arctan2(My_demand, Mx_demand)

        # Generate Mx-My contour at N = N_demand using the same
        # sweep + interpolation algorithm used for plotting.
        mxmy = self.generate_mx_my(
            N_demand, n_angles=n_angles,
            n_points_per_angle=n_points_per_angle)
        Mxa = mxmy["Mx"]
        Mya = mxmy["My"]

        # Filter NaN points (directions with no N-crossing)
        valid = np.isfinite(Mxa) & np.isfinite(Mya)
        Mxa = Mxa[valid]
        Mya = Mya[valid]

        if len(Mxa) < 3:
            warnings.warn(
                f"eta_demand: fewer than 3 valid contour points at "
                f"N={N_demand / 1e3:.1f} kN. Cannot build convex "
                f"hull.",
                RuntimeWarning,
                stacklevel=2,
            )
            return {"eta": float('inf'), "M_Rd_kNm": 0.0,
                    "M_Ed_kNm": M_Ed / 1e6,
                    "alpha_deg": np.degrees(alpha),
                    "ductility": None}

        # 2-D convex hull and ray-cast.
        pts = np.column_stack([Mxa, Mya])
        try:
            hull = ConvexHull(pts)
        except QhullError:
            # Degenerate hull: all points are collinear or
            # coincident.  This can happen for very thin sections
            # or pathological N levels.
            warnings.warn(
                f"eta_demand: convex hull degenerate at "
                f"N={N_demand / 1e3:.1f} kN (points may be "
                f"collinear). Returning eta=inf.",
                RuntimeWarning,
                stacklevel=2,
            )
            return {"eta": float('inf'), "M_Rd_kNm": 0.0,
                    "M_Ed_kNm": M_Ed / 1e6,
                    "alpha_deg": np.degrees(alpha),
                    "ductility": None}

        d = np.array([Mx_demand, My_demand], dtype=float)
        eqs = hull.equations
        A = eqs[:, :-1]
        c = eqs[:, -1]
        denom = A @ d
        numer = -c
        valid_mask = np.abs(denom) > 1e-15
        t = np.full_like(denom, np.inf)
        t[valid_mask] = numer[valid_mask] / denom[valid_mask]
        t[t <= 1e-12] = np.inf
        t_bnd = float(np.min(t))

        if t_bnd == np.inf or t_bnd <= 0:
            eta = float('inf')
            M_Rd = 0.0
        else:
            eta = 1.0 / t_bnd
            M_Rd = M_Ed / eta

        # Ductility at the curvature direction closest to α.
        ductility = self._ductility_at_direction(N_demand, alpha)

        return {
            "eta": eta,
            "M_Rd_kNm": M_Rd / 1e6,
            "M_Ed_kNm": M_Ed / 1e6,
            "alpha_deg": np.degrees(alpha),
            "ductility": ductility,
        }

    def _ductility_at_direction(self, N_demand, theta, n_chi=30):
        r"""
        Compute ductility :math:`\mu = \chi_u / \chi_y` at a given
        curvature direction and axial force.

        Sweeps curvature from zero to the strain-limit-derived
        maximum along direction :math:`\theta`, solving for
        :math:`\varepsilon_0` at each step.  Detects the first
        rebar yield and the first material strain limit violation.

        Parameters
        ----------
        N_demand : float
            Axial force [N].
        theta : float
            Curvature direction [rad].
        n_chi : int, optional
            Number of curvature steps.  Default 30.

        Returns
        -------
        float or None
            Ductility ratio, or None if yield or ultimate cannot
            be detected.
        """
        sv = self.solver
        exg = self._exg
        emb = self._emb

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        proj = self._all_ly * cos_t - self._all_lx * sin_t
        span = proj.max() - proj.min()
        if span < 1e-10:
            return None
        chi_max = (exg - emb) / span
        chis = np.linspace(chi_max / n_chi, chi_max, n_chi)
        cx = chis * cos_t
        cy = chis * sin_t

        eps0_arr, N_arr, _, _ = self._vectorized_solve_N(
            N_demand, cx, cy, n_iter=15)
        conv = np.abs(N_arr - N_demand) < 1e3

        if self._eps_yd_min is None:
            return None

        chi_yield = None
        chi_ult = None
        for j in range(len(chis)):
            if not conv[j]:
                continue
            eb, er = sv.strain_field(eps0_arr[j], cx[j], cy[j])
            all_eps = (np.concatenate([eb, er])
                       if len(er) > 0 else eb)

            # Yield detection: single comparison against
            # pre-computed minimum yield strain.
            if chi_yield is None and len(er) > 0:
                if np.abs(er).max() >= self._eps_yd_min * 0.99:
                    chi_yield = chis[j]

            if chi_ult is None:
                if (all_eps.min() <= emb * 0.99
                        or all_eps.max() >= exg * 0.99):
                    chi_ult = chis[j]

        if chi_yield is not None and chi_ult is not None:
            return abs(chi_ult / chi_yield)
        return None

    # ==================================================================
    #  Moment-curvature diagram (analytical Jacobian in Newton)
    # ==================================================================

    def generate_moment_curvature(self, N_fixed, chi_max=None,
                                  n_chi=100, direction='x'):
        r"""
        Generate the moment-curvature diagram at fixed axial force.

        Scans curvature :math:`\chi` from 0 to the ultimate value
        (where a material strain limit is reached), tracking the
        moment response.  Key points (first yield, ultimate) are
        identified.

        The inner Newton iteration for :math:`\varepsilon_0` at each
        curvature step uses the **analytical tangent** from
        :meth:`FiberSolver.integrate_with_tangent`, halving the
        integrate calls compared to a finite-difference approach.

        Parameters
        ----------
        N_fixed : float
            Fixed axial force [N]. Negative = compression.
        chi_max : float, optional
            Maximum curvature to scan [1/mm]. If ``None``, computed
            automatically from strain limits.
        n_chi : int, optional
            Number of curvature steps. Default 100.
        direction : str, optional
            ``'x'`` for Mx-chi_x (default) or ``'y'`` for My-chi_y.

        Returns
        -------
        dict
            ``chi`` [1/mm], ``M`` [N*mm], ``M_kNm``, ``chi_km``
            [1/km], ``eps_min``, ``eps_max`` (extreme strains at each
            step), ``N_fixed_kN``, ``direction``,
            cracking/yield/ultimate points for both positive and
            negative curvature branches, and ductility ratios
            ``ductility_pos``, ``ductility_neg``.

        Raises
        ------
        ValueError
            If *N_fixed* is not finite, *n_chi* < 1, or
            *direction* not in ``{'x', 'y'}``.
        """
        self._validate_finite_float(N_fixed, "N_fixed")
        self._validate_positive_int(n_chi, "n_chi")
        if direction not in ('x', 'y'):
            raise ValueError(
                f"direction must be 'x' or 'y', got {direction!r}")

        sec = self.solver.sec

        # Determine chi_max from the strain-limit span bound.
        #
        # The ultimate curvature for *any* axial force is bounded above
        # by the curvature at which the strain difference between the
        # extreme fibres equals the full admissible strain range
        # (eps_max_global - eps_min_bulk):
        #
        #     chi_u(N) <= (eps_xg - eps_mb) / span(direction)   for all N
        #
        # This is the same bound used by :meth:`_chi_max_for_direction`
        # (and hence by :meth:`generate_mx_my` and the ductility
        # computation).  The previous fixed heuristic assumed a neutral
        # axis at ~0.2*depth and was calibrated on concrete crushing
        # only; it systematically underestimated chi_max in the
        # tension-controlled regime (low/zero/negative N, lightly
        # reinforced sections), so the ultimate point fell outside the
        # scanned window and was reported as ``None``.
        if chi_max is None:
            cos_t, sin_t = (1.0, 0.0) if direction == 'x' else (0.0, 1.0)
            chi_max, span, _ = self._chi_max_for_direction(
                cos_t, sin_t, self._all_lx, self._all_ly)
            if not np.isfinite(chi_max) or chi_max <= 0.0:
                # Degenerate projection (zero span): fall back to a
                # geometry-based estimate so the scan is never empty.
                d_max = sec.H if direction == 'x' else sec.B
                chi_max = (abs(self._emb) / (d_max * 0.3) * 1.5
                           if d_max > 0 else 1e-4)

        # Cracking strain from EC2 properties (if available).
        eps_cr = None
        bulk = sec.bulk_material
        ec2_obj = getattr(bulk, 'ec2', None)
        if ec2_obj is not None:
            fctm = getattr(ec2_obj, 'fctm', None)
            ecm = getattr(ec2_obj, 'ecm', None)
            if fctm is not None and ecm is not None and ecm > 0:
                eps_cr = fctm / ecm

        # Scan both positive and negative curvature
        results_pos = self._scan_chi_vectorized(
            N_fixed, 0, chi_max, n_chi, direction, eps_cr=eps_cr)
        results_neg = self._scan_chi_vectorized(
            N_fixed, 0, -chi_max, n_chi, direction, eps_cr=eps_cr)

        # Merge: negative reversed + positive
        chi_all = np.concatenate([
            results_neg["chi"][::-1], results_pos["chi"][1:]])
        M_all = np.concatenate([
            results_neg["M"][::-1], results_pos["M"][1:]])
        eps_min_all = np.concatenate([
            results_neg["eps_min"][::-1],
            results_pos["eps_min"][1:]])
        eps_max_all = np.concatenate([
            results_neg["eps_max"][::-1],
            results_pos["eps_max"][1:]])

        # Ductility ratios: μ = χ_ultimate / χ_yield
        mu_pos = None
        mu_neg = None
        y_pos = results_pos.get("yield_chi")
        u_pos = results_pos.get("ultimate_chi")
        y_neg = results_neg.get("yield_chi")
        u_neg = results_neg.get("ultimate_chi")
        if (y_pos is not None and u_pos is not None
                and abs(y_pos) > 0):
            mu_pos = abs(u_pos / y_pos)
        if (y_neg is not None and u_neg is not None
                and abs(y_neg) > 0):
            mu_neg = abs(u_neg / y_neg)

        return {
            "chi": chi_all,
            "M": M_all,
            "M_kNm": M_all / 1e6,
            "chi_km": chi_all * 1e6,  # 1/mm -> 1/km
            "eps_min": eps_min_all,
            "eps_max": eps_max_all,
            "N_fixed_kN": N_fixed / 1e3,
            "direction": direction,
            "cracking_chi_pos": results_pos.get("cracking_chi"),
            "cracking_M_pos": results_pos.get("cracking_M"),
            "cracking_chi_neg": results_neg.get("cracking_chi"),
            "cracking_M_neg": results_neg.get("cracking_M"),
            "yield_chi_pos": y_pos,
            "yield_M_pos": results_pos.get("yield_M"),
            "ultimate_chi_pos": u_pos,
            "ultimate_M_pos": results_pos.get("ultimate_M"),
            "yield_chi_neg": y_neg,
            "yield_M_neg": results_neg.get("yield_M"),
            "ultimate_chi_neg": u_neg,
            "ultimate_M_neg": results_neg.get("ultimate_M"),
            "ductility_pos": mu_pos,
            "ductility_neg": mu_neg,
            "diagnostics_pos": results_pos.get("diagnostics"),
            "diagnostics_neg": results_neg.get("diagnostics"),
        }

    # ------------------------------------------------------------------
    # METHOD 1 — _scan_chi  (fallback)
    # ------------------------------------------------------------------

    def _scan_chi(self, N_fixed, chi_start, chi_end, n_points,
                  direction, eps_cr=None):
        r"""
        Internal: scan curvature from chi_start to chi_end,
        solving for eps0 at each step to maintain N = N_fixed.

        Uses Newton iteration with warm-start from previous step,
        falling back to scan-and-bisect if Newton fails.  The
        Newton step uses the **analytical tangent** from
        :meth:`FiberSolver.integrate_with_tangent`.

        Yield detection uses the pre-computed minimum yield strain
        :math:`\varepsilon_{yd,\min}` and a single
        ``np.abs(er).max()`` comparison per step, avoiding inner
        loops over rebar groups.

        Parameters
        ----------
        N_fixed : float
            Target axial force [N].
        chi_start, chi_end : float
            Curvature range [1/mm].
        n_points : int
            Number of curvature steps.
        direction : str
            ``'x'`` or ``'y'``.
        eps_cr : float or None
            Cracking strain of concrete (positive, tensile).
            If provided, the first-cracking point is detected.

        Returns
        -------
        dict
            Curvature scan results with the following keys:

            ``chi`` : numpy.ndarray
                Scanned curvatures [1/mm], shape ``(n_points,)``.
            ``M`` : numpy.ndarray
                Bending moment about the requested direction at
                each step [N.mm].
            ``eps_min`` : numpy.ndarray
                Most compressive fibre strain at each step.
            ``eps_max`` : numpy.ndarray
                Most tensile fibre strain at each step.
            ``yield_chi`` : float or None
                Curvature at first rebar yield, or ``None`` if not
                reached within the scan.
            ``yield_M`` : float or None
                Moment at first rebar yield.
            ``ultimate_chi`` : float or None
                Curvature at which a material strain limit is first
                reached (concrete crushing or steel rupture), or
                ``None`` if not reached.
            ``ultimate_M`` : float or None
                Moment at the ultimate curvature.
            ``cracking_chi`` : float or None
                Curvature at first concrete cracking, or ``None``
                (requires ``eps_cr``).
            ``cracking_M`` : float or None
                Moment at first cracking.
        """
        sec = self.solver.sec
        exg = self._exg
        emb = self._emb
        sv = self.solver

        chis = np.linspace(chi_start, chi_end, n_points)
        Ms = np.zeros(n_points)
        eps_mins = np.zeros(n_points)
        eps_maxs = np.zeros(n_points)

        yield_chi = None
        yield_M = None
        ultimate_chi = None
        ultimate_M = None
        cracking_chi = None
        cracking_M = None

        # Initial eps0 estimate
        if abs(chi_start) < 1e-15:
            A_ideal_gross = getattr(sec, 'ideal_gross_area',
                                    sec.B * sec.H)
            eps0_guess = N_fixed / (A_ideal_gross * 15000)
            eps0_guess = np.clip(eps0_guess, emb, -emb)
        else:
            eps0_guess = 0.0

        for k, chi in enumerate(chis):
            if direction == 'x':
                chi_x, chi_y = chi, 0.0
            else:
                chi_x, chi_y = 0.0, chi

            # Solve for eps0 such that N(eps0, chi_x, chi_y) = N_fixed
            eps0 = self._solve_eps0_for_N(
                sv, N_fixed, chi_x, chi_y, eps0_guess, emb)

            N, Mx, My = sv.integrate(eps0, chi_x, chi_y)
            if abs(N - N_fixed) > _feas_tol(N_fixed):
                # Target N unreachable at this curvature: drop the point.
                Ms[k] = np.nan
                eps_mins[k] = np.nan
                eps_maxs[k] = np.nan
                continue          # keep previous eps0_guess; skip event detection
            M = Mx if direction == 'x' else My
            Ms[k] = M
            eps0_guess = eps0  # warm-start

            # Track extreme strains
            eb, er = sv.strain_field(eps0, chi_x, chi_y)
            all_eps = (np.concatenate([eb, er])
                       if len(er) > 0 else eb)
            eps_mins[k] = all_eps.min()
            eps_maxs[k] = all_eps.max()

            # Detect first cracking
            if (cracking_chi is None and eps_cr is not None
                    and abs(chi) > 0):
                if eb.max() >= eps_cr:
                    cracking_chi = chi
                    cracking_M = M

            # Detect first yield — single comparison against
            # pre-computed eps_yd_min (no inner loop over rebars).
            if (yield_chi is None
                    and self._eps_yd_min is not None
                    and len(er) > 0
                    and abs(chi) > 0):
                if np.abs(er).max() >= self._eps_yd_min * 0.99:
                    yield_chi = chi
                yield_M = M

            # Detect ultimate
            if ultimate_chi is None and abs(chi) > 0:
                if (all_eps.min() <= emb * 0.99
                        or all_eps.max() >= exg * 0.99):
                    ultimate_chi = chi
                    ultimate_M = M

        return {
            "chi": chis, "M": Ms,
            "eps_min": eps_mins, "eps_max": eps_maxs,
            "yield_chi": yield_chi, "yield_M": yield_M,
            "ultimate_chi": ultimate_chi, "ultimate_M": ultimate_M,
            "cracking_chi": cracking_chi, "cracking_M": cracking_M,
        }


    # ------------------------------------------------------------------
    # METHOD 1 — _scan_chi_vectorized   (replaces _scan_chi calls)
    # ------------------------------------------------------------------

    def _scan_chi_vectorized(self, N_fixed, chi_start, chi_end, n_points,
                            direction, eps_cr=None):
        r"""
        Vectorized curvature scan at fixed axial force.

        Functionally equivalent to :meth:`_scan_chi` but replaces the
        Python loop with a three-phase strategy:

        1.  **Batch Newton** via :meth:`_vectorized_solve_N` — solves
            :math:`\varepsilon_0` for *all* curvature steps in parallel.
        2.  **Scalar fallback** via :meth:`_solve_eps0_for_N` — mops up
            the (typically few) non-converged points that need the
            scan-and-bisect safety net.
        3.  **Batch strain field** — a single ``(n, n_\text{fibers})``
            matrix multiply replaces ``n`` individual
            :meth:`~FiberSolver.strain_field` calls, followed by
            vectorized event detection (cracking, yield, ultimate).

        The warm-start that the sequential scan enjoys is sacrificed in
        favour of eliminating Python-loop overhead.  For the
        :math:`N(\varepsilon_0)` problems encountered in moment-curvature
        analysis, the global elastic estimate used by
        :meth:`_vectorized_solve_N` is sufficient for convergence in
        :math:`\le 15` Newton iterations on the vast majority of
        curvature steps.

        Parameters
        ----------
        N_fixed : float
            Target axial force [N].
        chi_start, chi_end : float
            Curvature range [1/mm].
        n_points : int
            Number of curvature steps.
        direction : ``'x'`` or ``'y'``
            Bending direction.
        eps_cr : float or None
            Cracking strain of concrete (positive, tensile).
            If provided, the first-cracking point is detected.

        Returns
        -------
        dict
            The base keys and semantics of :meth:`_scan_chi`,
            plus a ``diagnostics`` sub-dict not produced by the
            scalar scan:

            ``chi_max_scanned`` : float
                Largest scanned curvature magnitude [1/mm].
            ``n_points`` : int
                Number of curvature steps.
            ``n_converged`` : int
                Steps whose axial-equilibrium solve converged
                (``|N - N_fixed| < fallback_tol``); event detection
                is restricted to these rows.
            ``cracking_reason`` : str or None
                ``None`` when cracking is detected, otherwise why it
                is not: ``'no_ec2_properties'`` (bulk lacks
                ``fctm``/``Ecm``), ``'no_tension_in_range'`` (the
                section stays in compression over the scanned
                range), ``'no_convergence'``, or
                ``'below_threshold'``.
            ``ultimate_reason`` : str or None
                ``None`` when the ultimate is detected, otherwise
                ``'no_convergence'`` or
                ``'limit_not_reached_in_range'``.

        Notes
        -----
        Memory footprint of the batch strain field is
        :math:`n_\text{points} \times n_\text{fibers} \times 8` bytes.
        For typical values (200 × 10 000) this is ≈ 16 MB, well within
        budget.
        """
        sv = self.solver
        exg = self._exg
        emb = self._emb

        chis = np.linspace(chi_start, chi_end, n_points)

        # --- 1. Build curvature arrays for the chosen direction ---
        if direction == 'x':
            chi_x_arr = chis.copy()
            chi_y_arr = np.zeros(n_points, dtype=np.float64)
        else:
            chi_x_arr = np.zeros(n_points, dtype=np.float64)
            chi_y_arr = chis.copy()

        # --- 2. Batch Newton solve for all ε₀ simultaneously ---
        #
        # Use tighter tolerance than the default 1 kN to match the
        # quality of the scalar solver (which converges to 1 N).
        # Points that remain above ``fallback_tol`` get the full
        # scalar treatment.
        eps0_arr, N_arr, Mx_arr, My_arr = self._vectorized_solve_N(
            N_fixed, chi_x_arr, chi_y_arr,
            n_iter=20, tol=10.0, delta=1e-7)

        # --- 3. Scalar fallback for non-converged points ---
        #
        # Typically < 5 % of points — often concentrated near χ ≈ 0
        # where dN/dε₀ is nearly flat (large, stiff cross-section
        # under small curvature).
        fallback_tol = 10.0   # N — same as vectorized tol
        residual = np.abs(N_arr - N_fixed)
        bad_mask = residual >= fallback_tol
        if np.any(bad_mask):
            bad_idx = np.where(bad_mask)[0]
            for j in bad_idx:
                eps0_j = self._solve_eps0_for_N(
                    sv, N_fixed,
                    float(chi_x_arr[j]), float(chi_y_arr[j]),
                    float(eps0_arr[j]), emb)
                eps0_arr[j] = eps0_j
                Nj, Mxj, Myj = sv.integrate(
                    eps0_j, float(chi_x_arr[j]), float(chi_y_arr[j]))
                N_arr[j] = Nj
                Mx_arr[j] = Mxj
                My_arr[j] = Myj

        # --- 3b. Convergence mask after the scalar fallback ---
        #
        # Event detection must only trust curvature steps whose axial
        # equilibrium actually converged.  A non-converged row carries a
        # meaningless strain field that would otherwise trigger or hide
        # a cracking / yield / ultimate event at the wrong curvature.
        converged = np.abs(N_arr - N_fixed) < fallback_tol

        # --- Feasibility mask: drop points failing axial equilibrium ---
        feasible = np.abs(N_arr - N_fixed) <= _feas_tol(N_fixed)
        Mx_arr = np.where(feasible, Mx_arr, np.nan)
        My_arr = np.where(feasible, My_arr, np.nan)

        # --- 4. Extract moments along the requested direction ---
        Ms = Mx_arr if direction == 'x' else My_arr

        # --- 5. Batch strain field for event detection ---
        #
        # Bulk: ε_b[k, i] = ε₀[k] + χ_x[k]·ly[i] − χ_y[k]·lx[i]
        # Shape: (n_points, n_bulk_fibers)
        eb = (eps0_arr[:, None]
            + chi_x_arr[:, None] * sv._ly_bulk[None, :]
            - chi_y_arr[:, None] * sv._lx_bulk[None, :])

        n_rebars = len(sv.sec.y_rebars)
        if n_rebars > 0:
            # Rebar: ε_r[k, i] = ε₀[k] + χ_x[k]·ly_r[i] − χ_y[k]·lx_r[i]
            er = (eps0_arr[:, None]
                + chi_x_arr[:, None] * sv._ly_rebar[None, :]
                - chi_y_arr[:, None] * sv._lx_rebar[None, :])
            # Concatenate for global min/max
            all_eps = np.concatenate([eb, er], axis=1)
        else:
            er = np.empty((n_points, 0), dtype=np.float64)
            all_eps = eb

        eps_mins = all_eps.min(axis=1)
        eps_maxs = all_eps.max(axis=1)

        # --- 6. Vectorized event detection ---
        #
        # Events are detected as the *first* curvature step (smallest
        # |chi|) where the condition holds, restricted to steps that
        # (a) have nonzero curvature and (b) converged in the
        # equilibrium solve.  ``np.argmax`` on a boolean array returns
        # the index of the first ``True``.
        cracking_chi = cracking_M = None
        yield_chi = yield_M = None
        ultimate_chi = ultimate_M = None

        nonzero = (np.abs(chis) > 0) & feasible
        valid = nonzero & converged

        diagnostics = {
            "chi_max_scanned": (float(np.abs(chis).max())
                                if chis.size else 0.0),
            "n_points": int(n_points),
            "n_converged": int(converged.sum()),
            "cracking_reason": None,
            "ultimate_reason": None,
        }

        eb_max = eb.max(axis=1)

        # 6a. Cracking: first step where max bulk tensile strain >= eps_cr.
        if eps_cr is None:
            diagnostics["cracking_reason"] = "no_ec2_properties"
        else:
            crack_mask = valid & (eb_max >= eps_cr)
            if np.any(crack_mask):
                k = int(np.argmax(crack_mask))
                cracking_chi = float(chis[k])
                cracking_M = float(Ms[k])
            else:
                # Distinguish a physically uncracked section (bulk never
                # reaches tension within the scanned range, e.g. high
                # compression) from a convergence / window failure.
                reached = eb_max[valid]
                if reached.size == 0:
                    diagnostics["cracking_reason"] = "no_convergence"
                elif reached.max() < eps_cr:
                    diagnostics["cracking_reason"] = "no_tension_in_range"
                else:
                    diagnostics["cracking_reason"] = "below_threshold"

        # 6b. Yield: first step where max |eps_rebar| >= eps_yd_min * 0.99.
        if self._eps_yd_min is not None and n_rebars > 0:
            er_abs_max = np.abs(er).max(axis=1)
            yield_mask = valid & (er_abs_max >= self._eps_yd_min * 0.99)
            if np.any(yield_mask):
                k = int(np.argmax(yield_mask))
                yield_chi = float(chis[k])
                yield_M = float(Ms[k])

        # 6c. Ultimate: first step where any strain limit is reached,
        #     eps_min <= eps_cu * 0.99  or  eps_max >= eps_max_global * 0.99.
        ult_mask = valid & (
            (eps_mins <= emb * 0.99) | (eps_maxs >= exg * 0.99))
        if np.any(ult_mask):
            k = int(np.argmax(ult_mask))
            ultimate_chi = float(chis[k])
            ultimate_M = float(Ms[k])
        elif not np.any(valid):
            diagnostics["ultimate_reason"] = "no_convergence"
        else:
            diagnostics["ultimate_reason"] = "limit_not_reached_in_range"

        return {
            "chi": chis, "M": Ms,
            "eps_min": eps_mins, "eps_max": eps_maxs,
            "yield_chi": yield_chi, "yield_M": yield_M,
            "ultimate_chi": ultimate_chi, "ultimate_M": ultimate_M,
            "cracking_chi": cracking_chi, "cracking_M": cracking_M,
            "diagnostics": diagnostics,
        }


    # ------------------------------------------------------------------
    # METHOD 2 — generate_polar_curvature   (new public method)
    # ------------------------------------------------------------------

    def generate_polar_curvature(self, N_fixed, n_angles=72, n_chi=100):
        r"""
        Compute the ultimate-curvature polar diagram at fixed axial force.

        For each curvature direction :math:`\theta \in [0, 2\pi)`, finds
        the ultimate curvature magnitude :math:`\chi_u(\theta)` — the
        first :math:`\chi` at which a material strain limit is reached.

        The computation is fully vectorized: all
        :math:`n_\text{angles} \times n_\chi` strain configurations are
        solved in a single pass through :meth:`_vectorized_solve_N`,
        then the strain field is evaluated per-angle to detect the
        first limit crossing.

        Performance model
        -----------------
        The total number of configurations is
        :math:`n_\text{angles} \times n_\chi`.  For the default values
        (72 × 100 = 7 200), the vectorized Newton requires
        :math:`\sim 20 \times 2 = 40` batch calls regardless of the
        configuration count, compared with the
        :math:`72 \times 400 \approx 28\,800` scalar solves of the
        sequential scan.  The strain-field evaluation is computed
        per-angle to keep memory at :math:`n_\chi \times n_\text{fibers}`
        per chunk.

        Parameters
        ----------
        N_fixed : float
            Axial force [N]. Negative = compression.
        n_angles : int, optional
            Angular resolution.  Default 72 (every 5°).
        n_chi : int, optional
            Curvature steps per direction.  Default 100.

        Returns
        -------
        dict
            ``thetas`` — angles [rad], shape ``(n_angles,)``.
            ``chi_u`` — ultimate curvature per angle [1/mm].
            ``chi_u_km`` — same in 1/km.
            ``eps0_u`` — :math:`\varepsilon_0` at ultimate, per angle.
            ``N_fixed_kN`` — axial force in kN (convenience copy).

        Raises
        ------
        ValueError
            If ``n_angles < 3`` or ``n_chi < 2``.

        Notes
        -----
        Memory budget: the Newton solve operates on flat arrays of
        length :math:`n_\text{angles} \times n_\chi`.  If this exceeds
        the internal chunking threshold of :meth:`integrate_batch`
        (:math:`\approx 50\,000\,000 / n_\text{fibers}`), the batch
        is automatically split by the solver.  The per-angle strain
        field is always :math:`n_\chi \times n_\text{fibers}`, safely
        below budget.
        """
        if n_angles < 3:
            raise ValueError(
                f"n_angles must be >= 3, got {n_angles}")
        if n_chi < 2:
            raise ValueError(
                f"n_chi must be >= 2, got {n_chi}")

        sec = self.solver.sec
        sv = self.solver
        exg = self._exg
        emb = self._emb

        thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        # ----------------------------------------------------------
        # 1. Compute chi_max for each angle from geometry + limits
        # ----------------------------------------------------------
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)

        chi_max_arr = np.empty(n_angles, dtype=np.float64)
        for i in range(n_angles):
            chi_max_arr[i], _, _ = self._chi_max_for_direction(
                cos_t[i], sin_t[i], self._all_lx, self._all_ly)

        # Fallback for degenerate directions
        chi_max_arr[chi_max_arr < 1e-15] = 1e-4

        # ----------------------------------------------------------
        # 2. Build flat (n_total,) arrays of (chi_x, chi_y) configs
        # ----------------------------------------------------------
        #
        # For angle i, chi_mag ranges in
        # linspace(chi_max[i]/n_chi, chi_max[i], n_chi).
        # We skip chi_mag = 0 — it is a trivial configuration where
        # no bending occurs and no strain limit can be crossed.
        n_total = n_angles * n_chi
        flat_chi_x = np.empty(n_total, dtype=np.float64)
        flat_chi_y = np.empty(n_total, dtype=np.float64)
        flat_chi_mag = np.empty(n_total, dtype=np.float64)

        for i in range(n_angles):
            s = i * n_chi
            e = s + n_chi
            mags = np.linspace(
                chi_max_arr[i] / n_chi, chi_max_arr[i], n_chi)
            flat_chi_mag[s:e] = mags
            flat_chi_x[s:e] = mags * cos_t[i]
            flat_chi_y[s:e] = mags * sin_t[i]

        # ----------------------------------------------------------
        # 3. Vectorized Newton solve for all ε₀
        # ----------------------------------------------------------
        #
        # Memory check: n_total × n_fibers floats for the strain
        # matrix inside integrate_batch.  If this exceeds ~400 MB the
        # solver's internal chunking handles it automatically.
        eps0_flat, N_flat, _, _ = self._vectorized_solve_N(
            N_fixed, flat_chi_x, flat_chi_y,
            n_iter=20, tol=100.0, delta=1e-7)

        # Scalar fallback for non-converged points
        residual = np.abs(N_flat - N_fixed)
        bad_mask = residual >= 100.0
        if np.any(bad_mask):
            bad_idx = np.where(bad_mask)[0]
            for j in bad_idx:
                eps0_j = self._solve_eps0_for_N(
                    sv, N_fixed,
                    float(flat_chi_x[j]), float(flat_chi_y[j]),
                    float(eps0_flat[j]), emb)
                eps0_flat[j] = eps0_j

        # ----------------------------------------------------------
        # 4. Per-angle strain-field scan for ultimate detection
        # ----------------------------------------------------------
        #
        # For each angle, build the (n_chi, n_fibers) strain matrix
        # and find the first chi_mag where a limit is crossed.
        # Processing per-angle keeps peak memory at
        # n_chi × n_fibers × 8 bytes (≈ 0.8 MB for 100 × 1000).
        chi_u = np.zeros(n_angles, dtype=np.float64)
        eps0_u = np.zeros(n_angles, dtype=np.float64)

        for i in range(n_angles):
            s = i * n_chi
            e = s + n_chi
            e0 = eps0_flat[s:e]
            cx = flat_chi_x[s:e]
            cy = flat_chi_y[s:e]
            mags = flat_chi_mag[s:e]

            # Bulk strains: (n_chi, n_bulk)
            eb_i = (e0[:, None]
                    + cx[:, None] * sv._ly_bulk[None, :]
                    - cy[:, None] * sv._lx_bulk[None, :])

            n_rebars = len(sv.sec.y_rebars)
            if n_rebars > 0:
                er_i = (e0[:, None]
                        + cx[:, None] * sv._ly_rebar[None, :]
                        - cy[:, None] * sv._lx_rebar[None, :])
                all_eps = np.concatenate([eb_i, er_i], axis=1)
            else:
                all_eps = eb_i

            eps_min_i = all_eps.min(axis=1)
            eps_max_i = all_eps.max(axis=1)

            # First index where a strain limit is reached
            ult_mask = ((eps_min_i <= emb * 0.99)
                        | (eps_max_i >= exg * 0.99))
            if np.any(ult_mask):
                k = int(np.argmax(ult_mask))
                chi_u[i] = mags[k]
                eps0_u[i] = e0[k]
            else:
                # No crossing found — use the last (largest) curvature
                chi_u[i] = mags[-1]
                eps0_u[i] = e0[-1]

        return {
            "thetas": thetas,
            "chi_u": chi_u,
            "chi_u_km": chi_u * 1e6,
            "eps0_u": eps0_u,
            "N_fixed_kN": N_fixed / 1e3,
        }

    @staticmethod
    def _solve_eps0_for_N(sv, N_target, chi_x, chi_y, eps0_init, emb):
        r"""
        Solve for eps0 at fixed (chi_x, chi_y) such that N = N_target.

        Two-phase strategy:

        1. **Newton with analytical tangent**:
           :math:`dN/d\varepsilon_0` from the tangent stiffness
           matrix (element ``K[0,0]``), avoiding an extra
           ``integrate()`` call for the finite-difference derivative.
           Step clamped to ±0.001.  Converges in 5–10 iterations
           for well-conditioned problems.

        2. **Scan-and-bisect fallback**:
           if Newton stalls (near-zero Jacobian or non-convergence
           after 25 iterations), a coarse scan over 30 uniformly
           spaced :math:`\varepsilon_0` values in
           :math:`[1.2\,\varepsilon_{cu},\; -1.2\,\varepsilon_{cu}]`
           identifies a sign-change bracket, then bisection refines
           to 1 N tolerance.

        Parameters
        ----------
        sv : FiberSolver
            Fiber solver instance.
        N_target : float
            Target axial force [N].
        chi_x, chi_y : float
            Curvature components [1/mm].
        eps0_init : float
            Initial guess for :math:`\varepsilon_0`.
        emb : float
            Bulk material ultimate strain (negative).

        Returns
        -------
        float
            Converged :math:`\varepsilon_0`.
        """
        eps0 = eps0_init

        # Phase 1: Newton with analytical tangent
        for _ in range(25):
            N, _, _, K = sv.integrate_with_tangent(eps0, chi_x, chi_y)
            r = N - N_target
            if abs(r) < 1.0:  # 1 N tolerance
                return eps0
            dNde = K[0, 0]      # analytical dN/deps0
            if abs(dNde) > 1:
                step = -r / dNde
                step = np.clip(step, -0.001, 0.001)
                eps0 += step
            else:
                # Near-zero Jacobian: skip to scan-and-bisect
                break

        # Phase 2: scan-and-bisect fallback.
        # Coarse scan to find a sign-change bracket.
        a_bound = emb * 1.2
        b_bound = -emb * 1.2
        n_scan = 30
        scan_eps = np.linspace(a_bound, b_bound, n_scan)
        scan_N = np.empty(n_scan)
        for i, e0 in enumerate(scan_eps):
            Ni, _, _ = sv.integrate(e0, chi_x, chi_y)
            scan_N[i] = Ni

        scan_r = scan_N - N_target

        # Find first sign change
        a, b = None, None
        for i in range(n_scan - 1):
            if scan_r[i] * scan_r[i + 1] <= 0:
                a = scan_eps[i]
                b = scan_eps[i + 1]
                break

        if a is None:
            # No bracket found — return the scan point closest
            # to the target.
            idx = np.argmin(np.abs(scan_r))
            return float(scan_eps[idx])

        # Bisection refinement
        Na, _, _ = sv.integrate(a, chi_x, chi_y)
        for _ in range(50):
            mid = (a + b) / 2
            Nm, _, _ = sv.integrate(mid, chi_x, chi_y)
            if abs(Nm - N_target) < 1.0:
                return mid
            if (Nm - N_target) * (Na - N_target) < 0:
                b = mid
            else:
                a = mid
                Na = Nm
        return mid
