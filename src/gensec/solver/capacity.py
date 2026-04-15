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
"""

import numpy as np
from .integrator import FiberSolver


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
    """

    def __init__(self, solver):
        self.solver = solver

    def _collect_strain_limits(self):
        """
        Global strain envelope from all materials.

        Returns
        -------
        eps_min_global, eps_max_global, eps_min_bulk, eps_max_bulk
        """
        sec = self.solver.sec
        b = sec.bulk_material
        emin_g, emax_g = b.eps_min, b.eps_max
        for r in sec.rebars:
            emin_g = min(emin_g, r.material.eps_min)
            emax_g = max(emax_g, r.material.eps_max)
        return emin_g, emax_g, b.eps_min, b.eps_max

    def _ultimate_strain_configs_1d(self, direction='x', n_points=200):
        r"""
        Generate ``(eps_inf, eps_sup)`` pairs for single-axis bending.

        For ``direction='x'`` (bending about x-axis, curvature
        :math:`\chi_x`), the section depth is ``H`` and the reference
        lever arm is ``y_ref``.

        For ``direction='y'`` (bending about y-axis, curvature
        :math:`\chi_y`), the section depth is ``B`` and the reference
        lever arm is ``x_ref``.

        Parameters
        ----------
        direction : ``'x'`` or ``'y'``
        n_points : int

        Returns
        -------
        list of (eps0, chi)
            Strain configurations as ``(eps0, curvature)`` pairs.
        """
        emg, exg, emb, _ = self._collect_strain_limits()
        sec = self.solver.sec

        if direction == 'x':
            depth = sec.H
            ref = self.solver.y_ref
        else:
            depth = sec.B
            ref = self.solver.x_ref

        configs = []

        def _to_eps0_chi(ei, es):
            """Convert edge strains to (eps0, chi)."""
            chi = (es - ei) / depth if depth > 0 else 0.0
            eps0 = ei + chi * ref
            return eps0, chi

        # Positive curvature: "top" (max coord) compressed
        for ei in np.linspace(exg, emg, n_points):
            configs.append(_to_eps0_chi(ei, emb))
        for es in np.linspace(emb, emb * 0.5, n_points // 2):
            configs.append(_to_eps0_chi(emb, es))

        # Negative curvature: "bottom" (min coord) compressed
        for es in np.linspace(exg, emg, n_points):
            configs.append(_to_eps0_chi(emb, es))
        for ei in np.linspace(emb, emb * 0.5, n_points // 2):
            configs.append(_to_eps0_chi(ei, emb))

        # Pure tension
        for ev in np.linspace(0, exg, n_points // 4):
            configs.append(_to_eps0_chi(ev, 0.0))
            configs.append(_to_eps0_chi(0.0, ev))
            configs.append(_to_eps0_chi(exg, ev))
            configs.append(_to_eps0_chi(ev, exg))

        # Near-uniform compression
        for ev in np.linspace(emb * 0.5, emb, n_points // 4):
            configs.append(_to_eps0_chi(ev, ev))
            span = abs(emb) * 0.15
            for d in np.linspace(-span, span, 5):
                configs.append(_to_eps0_chi(ev + d, ev - d))

        return configs

    def generate(self, n_points=300, direction='x'):
        r"""
        Generate the N-M interaction diagram for a single bending
        direction.

        For ``direction='x'``: N-Mx diagram (:math:`\chi_y = 0`).
        For ``direction='y'``: N-My diagram (:math:`\chi_x = 0`).

        Parameters
        ----------
        n_points : int, optional
            Base resolution. Default 300.
        direction : ``'x'`` or ``'y'``, optional
            Bending direction. Default ``'x'``.

        Returns
        -------
        dict
            ``N``, ``M`` [N, N*mm], ``N_kN``, ``M_kNm``.
            Also ``Mx`` or ``My`` alias depending on direction.
        """
        configs = self._ultimate_strain_configs_1d(direction, n_points)
        Nl, Ml = [], []
        for eps0, chi in configs:
            if direction == 'x':
                N, Mx, My = self.solver.integrate(eps0, chi, 0.0)
                Nl.append(N)
                Ml.append(Mx)
            else:
                N, Mx, My = self.solver.integrate(eps0, 0.0, chi)
                Nl.append(N)
                Ml.append(My)
        Na, Ma = np.array(Nl), np.array(Ml)
        M_key = "Mx" if direction == 'x' else "My"
        return {
            "N": Na, "M": Ma, M_key: Ma,
            "N_kN": Na / 1e3, "M_kNm": Ma / 1e6,
            f"{M_key}_kNm": Ma / 1e6,
            "direction": direction,
        }

    def generate_biaxial(self, n_angles=72, n_points_per_angle=200):
        r"""
        Generate the 3D resistance surface (N, Mx, My).

        Scans curvature directions :math:`\theta` in
        :math:`[0, 2\pi)` and, for each direction, scans curvature
        magnitudes through ultimate strain configurations.

        Parameters
        ----------
        n_angles : int, optional
            Number of curvature direction angles. Default 36 (every 10°).
        n_points_per_angle : int, optional
            Strain configurations per angle. Default 200.

        Returns
        -------
        dict
            ``N`` [N], ``Mx``, ``My`` [N*mm], and ``_kN``/``_kNm``
            variants.
        """
        sec = self.solver.sec
        emg, exg, emb, _ = self._collect_strain_limits()

        # All bulk fiber positions relative to reference
        lx = sec.x_fibers - self.solver.x_ref
        ly = sec.y_fibers - self.solver.y_ref

        # Also rebar positions
        lx_r = sec.x_rebars - self.solver.x_ref
        ly_r = sec.y_rebars - self.solver.y_ref

        # All lever arms combined
        all_lx = np.concatenate([lx, lx_r])
        all_ly = np.concatenate([ly, ly_r])

        thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        Nl, Mxl, Myl = [], [], []

        for theta in thetas:
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            # Project all fiber positions onto curvature direction.
            # Strain field: eps = eps0 + chi_x*ly - chi_y*lx
            # With chi_x = chi*cos(t), chi_y = chi*sin(t):
            # eps = eps0 + chi*(ly*cos_t - lx*sin_t)
            proj = all_ly * cos_t - all_lx * sin_t

            # Extreme projections
            p_max = proj.max()
            p_min = proj.min()

            if abs(p_max - p_min) < 1e-10:
                continue

            # Scan strain configurations: similar to uniaxial but
            # along the projected direction.
            # The "top" extreme (max projection) gets eps_top,
            # the "bottom" extreme (min projection) gets eps_bot.
            # eps0 + chi * p_max = eps_top
            # eps0 + chi * p_min = eps_bot
            # => chi = (eps_top - eps_bot) / (p_max - p_min)
            # => eps0 = eps_bot - chi * p_min

            span = p_max - p_min

            def _scan(eps_bot, eps_top):
                chi = (eps_top - eps_bot) / span
                eps0 = eps_bot - chi * p_min
                chi_x = chi * cos_t
                chi_y = chi * sin_t
                return self.solver.integrate(eps0, chi_x, chi_y)

            n = n_points_per_angle

            # Branch 1: "top" at bulk crush limit
            for eb in np.linspace(exg, emg, n):
                N, Mx, My = _scan(eb, emb)
                Nl.append(N); Mxl.append(Mx); Myl.append(My)

            # Branch 2: "bottom" at bulk crush limit
            for et in np.linspace(exg, emg, n):
                N, Mx, My = _scan(emb, et)
                Nl.append(N); Mxl.append(Mx); Myl.append(My)

            # Branch 3: near-uniform compression
            for ev in np.linspace(emb * 0.5, emb, n // 4):
                N, Mx, My = _scan(ev, ev)
                Nl.append(N); Mxl.append(Mx); Myl.append(My)
                sp = abs(emb) * 0.15
                for d in np.linspace(-sp, sp, 3):
                    N, Mx, My = _scan(ev + d, ev - d)
                    Nl.append(N); Mxl.append(Mx); Myl.append(My)

            # Branch 4: pure tension — denser scan for smooth hull cap.
            # Uniform tension at various levels.
            n_tens = n // 2
            for ev in np.linspace(0, exg, n_tens):
                N, Mx, My = _scan(ev, ev)
                Nl.append(N); Mxl.append(Mx); Myl.append(My)
            # Non-uniform tension with gradient.
            for ev in np.linspace(0, exg, n_tens):
                N, Mx, My = _scan(ev, 0.0)
                Nl.append(N); Mxl.append(Mx); Myl.append(My)
                N, Mx, My = _scan(0.0, ev)
                Nl.append(N); Mxl.append(Mx); Myl.append(My)
                # Intermediate gradient.
                N, Mx, My = _scan(ev, ev * 0.5)
                Nl.append(N); Mxl.append(Mx); Myl.append(My)
                N, Mx, My = _scan(ev * 0.5, ev)
                Nl.append(N); Mxl.append(Mx); Myl.append(My)

        Na = np.array(Nl)
        Mxa = np.array(Mxl)
        Mya = np.array(Myl)

        return {
            "N": Na, "Mx": Mxa, "My": Mya,
            "N_kN": Na / 1e3,
            "Mx_kNm": Mxa / 1e6,
            "My_kNm": Mya / 1e6,
        }

    def generate_mx_my(self, N_fixed, n_angles=72,
                       n_points_per_angle=200):
        r"""
        Generate the Mx-My interaction contour at a fixed axial force.

        For a given :math:`N`, scans curvature directions
        :math:`\theta \in [0, 2\pi)`. For each direction, sweeps
        through strain configurations, collects all (N, Mx, My)
        points, and **interpolates** at :math:`N = N_{\text{fixed}}`
        to find the moment point on the contour.

        Parameters
        ----------
        N_fixed : float
            Fixed axial force [N].
        n_angles : int, optional
            Number of curvature directions. Default 72 (every 5°).
        n_points_per_angle : int, optional
            Strain scan resolution per direction. Default 200.

        Returns
        -------
        dict
            ``Mx`` [N*mm], ``My`` [N*mm], ``Mx_kNm``, ``My_kNm``,
            ``N_fixed_kN``.
        """
        sec = self.solver.sec
        emg, exg, emb, _ = self._collect_strain_limits()

        lx = sec.x_fibers - self.solver.x_ref
        ly = sec.y_fibers - self.solver.y_ref
        lx_r = sec.x_rebars - self.solver.x_ref
        ly_r = sec.y_rebars - self.solver.y_ref
        all_lx = np.concatenate([lx, lx_r])
        all_ly = np.concatenate([ly, ly_r])

        thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        Mxl, Myl = [], []

        for theta in thetas:
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            proj = all_ly * cos_t - all_lx * sin_t
            p_max = proj.max()
            p_min = proj.min()
            if abs(p_max - p_min) < 1e-10:
                continue
            span = p_max - p_min

            # Collect all (N, Mx, My) along this curvature direction
            points_N = []
            points_Mx = []
            points_My = []

            def _collect(eps_bot, eps_top):
                chi = (eps_top - eps_bot) / span
                eps0 = eps_bot - chi * p_min
                chi_x = chi * cos_t
                chi_y = chi * sin_t
                N, Mx, My = self.solver.integrate(eps0, chi_x, chi_y)
                points_N.append(N)
                points_Mx.append(Mx)
                points_My.append(My)

            n = n_points_per_angle
            for eb in np.linspace(exg, emg, n):
                _collect(eb, emb)
            for et in np.linspace(exg, emg, n):
                _collect(emb, et)
            for ev in np.linspace(emb * 0.5, emb, n // 4):
                _collect(ev, ev)
                sp = abs(emb) * 0.15
                for d in np.linspace(-sp, sp, 3):
                    _collect(ev + d, ev - d)
            for ev in np.linspace(0, exg, n // 4):
                _collect(ev, 0.0)
                _collect(0.0, ev)

            pN = np.array(points_N)
            pMx = np.array(points_Mx)
            pMy = np.array(points_My)

            # Find crossings where N passes through N_fixed,
            # and among those pick the one with maximum |M|
            # in the direction theta.
            best_Mx = 0.0
            best_My = 0.0
            best_M_mag = -1.0

            for k in range(len(pN) - 1):
                N_a, N_b = pN[k], pN[k + 1]
                if (N_a - N_fixed) * (N_b - N_fixed) <= 0 and abs(N_b - N_a) > 1e-6:
                    # Linear interpolation
                    t = (N_fixed - N_a) / (N_b - N_a)
                    Mx_interp = pMx[k] + t * (pMx[k + 1] - pMx[k])
                    My_interp = pMy[k] + t * (pMy[k + 1] - pMy[k])
                    M_mag = np.sqrt(Mx_interp**2 + My_interp**2)
                    if M_mag > best_M_mag:
                        best_M_mag = M_mag
                        best_Mx = Mx_interp
                        best_My = My_interp

            # Fallback: closest point in N
            if best_M_mag < 0:
                idx = np.argmin(np.abs(pN - N_fixed))
                best_Mx = pMx[idx]
                best_My = pMy[idx]

            Mxl.append(best_Mx)
            Myl.append(best_My)

        Mxa = np.array(Mxl)
        Mya = np.array(Myl)

        return {
            "Mx": Mxa, "My": Mya,
            "Mx_kNm": Mxa / 1e6, "My_kNm": Mya / 1e6,
            "N_fixed_kN": N_fixed / 1e3,
        }

    def generate_moment_curvature(self, N_fixed, chi_max=None,
                                  n_points=200, direction='x'):
        r"""
        Generate the moment-curvature diagram at fixed axial force.

        Scans curvature :math:`\chi` from 0 to the ultimate value
        (where a material strain limit is reached), tracking the
        moment response. Key points (first yield, ultimate) are
        identified.

        Parameters
        ----------
        N_fixed : float
            Fixed axial force [N]. Negative = compression.
        chi_max : float, optional
            Maximum curvature to scan [1/mm]. If ``None``, computed
            automatically from strain limits.
        n_points : int, optional
            Number of curvature steps. Default 200.
        direction : str, optional
            ``'x'`` for Mx-chi_x (default) or ``'y'`` for My-chi_y.

        Returns
        -------
        dict
            ``chi`` [1/mm], ``M`` [N*mm], ``M_kNm``, ``chi_km``
            [1/km], ``eps_min``, ``eps_max`` (extreme strains at each
            step), ``yield_index`` (first yield step or None),
            ``ultimate_index`` (ultimate step or None),
            ``N_fixed_kN``, ``direction``.
        """
        sec = self.solver.sec
        emg, _, emb, _ = self._collect_strain_limits()

        # Determine chi_max from geometry and strain limits.
        # Use the full section height (not half) as lever arm to
        # ensure the ultimate strain is reached even when the
        # neutral axis is far from the centroid.
        if chi_max is None:
            if direction == 'x':
                d_max = sec.H
            else:
                d_max = sec.B
            if d_max > 0:
                chi_max = abs(emb) / (d_max * 0.3) * 1.5
            else:
                chi_max = 1e-4

        # Collect rebar yield strains for yield detection
        rebar_eps_yd = []
        for rb in sec.rebars:
            if hasattr(rb.material, 'eps_yd'):
                rebar_eps_yd.append(rb.material.eps_yd)

        # Cracking strain from EC2 properties (if available).
        # eps_cr = fctm / Ecm (positive, tensile).
        eps_cr = None
        bulk = sec.bulk_material
        ec2_obj = getattr(bulk, 'ec2', None)
        if ec2_obj is not None:
            fctm = getattr(ec2_obj, 'fctm', None)
            ecm = getattr(ec2_obj, 'ecm', None)
            if fctm is not None and ecm is not None and ecm > 0:
                eps_cr = fctm / ecm

        # Scan both positive and negative curvature
        results_pos = self._scan_chi(
            N_fixed, 0, chi_max, n_points, direction,
            rebar_eps_yd, eps_cr=eps_cr)
        results_neg = self._scan_chi(
            N_fixed, 0, -chi_max, n_points, direction,
            rebar_eps_yd, eps_cr=eps_cr)

        # Merge: negative reversed + positive
        chi_all = np.concatenate([
            results_neg["chi"][::-1], results_pos["chi"][1:]])
        M_all = np.concatenate([
            results_neg["M"][::-1], results_pos["M"][1:]])
        eps_min_all = np.concatenate([
            results_neg["eps_min"][::-1], results_pos["eps_min"][1:]])
        eps_max_all = np.concatenate([
            results_neg["eps_max"][::-1], results_pos["eps_max"][1:]])

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
            "yield_chi_pos": results_pos.get("yield_chi"),
            "yield_M_pos": results_pos.get("yield_M"),
            "ultimate_chi_pos": results_pos.get("ultimate_chi"),
            "ultimate_M_pos": results_pos.get("ultimate_M"),
            "yield_chi_neg": results_neg.get("yield_chi"),
            "yield_M_neg": results_neg.get("yield_M"),
            "ultimate_chi_neg": results_neg.get("ultimate_chi"),
            "ultimate_M_neg": results_neg.get("ultimate_M"),
        }

    def _scan_chi(self, N_fixed, chi_start, chi_end, n_points,
                  direction, rebar_eps_yd, eps_cr=None):
        """
        Internal: scan curvature from chi_start to chi_end,
        solving for eps0 at each step to maintain N = N_fixed.

        Uses Newton iteration with warm-start from previous step,
        falling back to bisection if Newton fails.

        Parameters
        ----------
        N_fixed : float
        chi_start, chi_end : float
        n_points : int
        direction : str
        rebar_eps_yd : list of float
        eps_cr : float or None
            Cracking strain of concrete (positive, tensile).
            If provided, the first-cracking point is detected.
        """
        sec = self.solver.sec
        emg, exg, emb, _ = self._collect_strain_limits()
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

        # Initial eps0 estimate from the equilibrium solver at chi=0
        if abs(chi_start) < 1e-15:
            # Use a rough estimate: N_fixed ~ sigma_mean * A
            A_gross = getattr(sec, 'gross_area', sec.B * sec.H)
            eps0_guess = N_fixed / (A_gross * 15000)  # rough Ec/2
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
            M = Mx if direction == 'x' else My
            Ms[k] = M
            eps0_guess = eps0  # warm-start

            # Track extreme strains
            eb, er = sv.strain_field(eps0, chi_x, chi_y)
            all_eps = np.concatenate([eb, er]) if len(er) > 0 else eb
            eps_mins[k] = all_eps.min()
            eps_maxs[k] = all_eps.max()

            # Detect first cracking (concrete tensile strain exceeds
            # eps_cr).  Only bulk fibers are checked since rebars do
            # not crack.
            if (cracking_chi is None and eps_cr is not None
                    and abs(chi) > 0):
                if eb.max() >= eps_cr:
                    cracking_chi = chi
                    cracking_M = M

            # Detect first yield
            if yield_chi is None and len(rebar_eps_yd) > 0 and abs(chi) > 0:
                for j, rb in enumerate(sec.rebars):
                    if hasattr(rb.material, 'eps_yd'):
                        if abs(er[j]) >= rb.material.eps_yd * 0.99:
                            yield_chi = chi
                            yield_M = M
                            break

            # Detect ultimate (any fiber at material limit).
            # emb is negative: emb * 0.99 is less negative, so the
            # check triggers when strain reaches (or just exceeds)
            # the limit, not 1% past it.
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

    @staticmethod
    def _solve_eps0_for_N(sv, N_target, chi_x, chi_y, eps0_init, emb):
        """
        Solve for eps0 at fixed (chi_x, chi_y) such that N = N_target.

        Newton with bisection fallback.
        """
        eps0 = eps0_init

        # Newton phase
        for _ in range(25):
            N, _, _ = sv.integrate(eps0, chi_x, chi_y)
            r = N - N_target
            if abs(r) < 1.0:  # 1 N tolerance
                return eps0
            de = max(abs(eps0) * 1e-7, 1e-10)
            N1, _, _ = sv.integrate(eps0 + de, chi_x, chi_y)
            dNde = (N1 - N) / de
            if abs(dNde) > 1:
                step = -r / dNde
                # Clamp step to avoid wild jumps
                step = np.clip(step, -0.001, 0.001)
                eps0 += step
            else:
                eps0 -= 1e-5
                continue

        # Bisection fallback: search eps0 in [emb, -emb]
        a, b = emb * 1.2, -emb * 1.2
        Na, _, _ = sv.integrate(a, chi_x, chi_y)
        Nb, _, _ = sv.integrate(b, chi_x, chi_y)
        if (Na - N_target) * (Nb - N_target) > 0:
            return eps0  # can't bracket — return best Newton

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
