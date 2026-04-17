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
Demand verification against the resistance domain.

Implements the GenSec v2.1 demand architecture with four utilization
ratio types, staged combinations, and envelopes.

Utilization ratios
------------------
All four :math:`\eta` types share the same geometric primitive —
a ray from base :math:`\mathbf{B}` through target :math:`\mathbf{T}`
intersecting a boundary at :math:`\mathbf{R}`:

.. math::

    \eta = \frac{|\mathbf{T} - \mathbf{B}|}
                {|\mathbf{R} - \mathbf{B}|}

The four variants differ in what :math:`\mathbf{B}`, :math:`\mathbf{T}`
and the boundary represent:

+------------------+---------------------------+-------------------+----------------------------+
| Type             | Base :math:`\mathbf{B}`   | Target            | Boundary                   |
+==================+===========================+===================+============================+
| ``eta_3D``       | Origin                    | Demand            | 3-D hull :math:`(N,Mx,My)` |
+------------------+---------------------------+-------------------+----------------------------+
| ``eta_2D``       | Origin in :math:`Mx`-     | :math:`(Mx, My)`  | :math:`Mx`-:math:`My`      |
|                  | :math:`My` plane          | of demand         | contour at :math:`N`       |
+------------------+---------------------------+-------------------+----------------------------+
| ``eta_path``     | :math:`\mathbf{S}_{k-1}` | :math:`\mathbf{S}_k` | 3-D hull              |
+------------------+---------------------------+-------------------+----------------------------+
| ``eta_path_2D``  | :math:`(Mx,My)_{k-1}`    | :math:`(Mx,My)_k` | :math:`Mx`-:math:`My`      |
|                  |                           |                   | contour at :math:`N_k`     |
+------------------+---------------------------+-------------------+----------------------------+

Classes
-------
DomainChecker
    3-D ConvexHull operations (``eta_3D``, ``eta_path``, ``is_inside``).
MxMyContour
    2-D contour at fixed *N* (``eta_2D``).
VerificationEngine
    Flag-driven orchestrator that resolves demands, combinations and
    envelopes, managing contour caches and producing structured results.
"""

import numpy as np
from scipy.spatial import ConvexHull


# ==================================================================
#  DomainChecker — 3-D hull operations
# ==================================================================

class DomainChecker:
    r"""
    3-D resistance domain checker using a ConvexHull in
    :math:`(N, M_x, M_y)` space.

    Parameters
    ----------
    nm_3d : dict
        Output of :meth:`NMDiagram.generate_biaxial` (keys ``N``,
        ``Mx``, ``My`` in Newton / N·mm) **or** of
        :meth:`NMDiagram.generate` for uniaxial mode (keys ``N``,
        ``M``).

    Attributes
    ----------
    ndim : int
        Dimensionality (2 for uniaxial, 3 for biaxial).
    hull : scipy.spatial.ConvexHull
    N_range : float
        :math:`N_{Rd,\max} - N_{Rd,\min}` [N]. Used by
        ``eta_path_2D`` to evaluate the :math:`\Delta N` tolerance.
    """

    def __init__(self, nm_3d):
        if "My" in nm_3d and nm_3d.get("My") is not None:
            pts = np.column_stack([nm_3d["N"], nm_3d["Mx"],
                                   nm_3d["My"]])
            self.ndim = 3
        else:
            M = nm_3d.get("M", nm_3d.get("Mx"))
            pts = np.column_stack([nm_3d["N"], M])
            self.ndim = 2

        self.hull = ConvexHull(pts)
        self._equations = self.hull.equations

        # Axial range for delta_N tolerance check.
        self.N_range = float(pts[:, 0].max() - pts[:, 0].min())

    # ------------------------------------------------------------------
    #  Point queries
    # ------------------------------------------------------------------

    def _make_point(self, N, Mx, My=0.0):
        """Build a point array consistent with hull dimensionality."""
        if self.ndim == 3:
            return np.array([N, Mx, My], dtype=float)
        return np.array([N, Mx], dtype=float)

    def is_inside(self, N, Mx, My=0.0):
        """
        Check whether a point lies inside the resistance domain.

        Parameters
        ----------
        N : float
            Axial force [N].
        Mx : float
            Bending moment about *x* [N·mm].
        My : float, optional
            Bending moment about *y* [N·mm]. Default 0.

        Returns
        -------
        bool
        """
        p = self._make_point(N, Mx, My)
        return bool(np.all(
            self._equations[:, :-1] @ p + self._equations[:, -1] <= 1e-6
        ))

    # ------------------------------------------------------------------
    #  Ray-cast engine (shared by eta_3D and eta_path)
    # ------------------------------------------------------------------

    def _ray_eta(self, base, target):
        r"""
        Cast a ray from *base* through *target* and return :math:`\eta`.

        .. math::

            \eta = \frac{|\mathbf{T} - \mathbf{B}|}
                        {|\mathbf{R} - \mathbf{B}|}

        where :math:`\mathbf{R}` is the **first** intersection of the
        ray with the hull boundary in the direction from *base* toward
        *target*.

        Parameters
        ----------
        base : numpy.ndarray
        target : numpy.ndarray

        Returns
        -------
        float
            :math:`\eta`.  ``0.0`` if *target* coincides with *base*
            and is inside; ``inf`` if outside and ray does not hit the
            hull.
        """
        direction = target - base
        length = np.linalg.norm(direction)
        if length < 1e-12:
            inside = bool(np.all(
                self._equations[:, :-1] @ base
                + self._equations[:, -1] <= 1e-6))
            return 0.0 if inside else float("inf")

        A = self._equations[:, :-1]
        c = self._equations[:, -1]
        denom = A @ direction
        numer = -(A @ base + c)

        valid = np.abs(denom) > 1e-15
        t = np.full_like(denom, np.inf)
        t[valid] = numer[valid] / denom[valid]
        # Only consider intersections ahead of the base (t > 0).
        t[t <= 1e-12] = np.inf
        t_bnd = float(np.min(t))

        if t_bnd == np.inf:
            inside = bool(np.all(
                self._equations[:, :-1] @ base
                + self._equations[:, -1] <= 1e-6))
            return 0.0 if inside else float("inf")

        return 1.0 / t_bnd

    # ------------------------------------------------------------------
    #  Public η methods
    # ------------------------------------------------------------------

    def eta_3D(self, N, Mx, My=0.0):
        r"""
        Utilization ratio via ray from the origin.

        .. math::

            \eta_{\text{3D}}
            = \frac{|\mathbf{S}|}{|\mathbf{R}|},
            \qquad
            \mathbf{S} = (N,\,M_x,\,M_y)

        Parameters
        ----------
        N, Mx, My : float
            Demand forces [N, N·mm].

        Returns
        -------
        float
        """
        origin = np.zeros(self.ndim)
        target = self._make_point(N, Mx, My)
        return self._ray_eta(origin, target)

    def eta_path(self, N_base, Mx_base, My_base,
                 N_target, Mx_target, My_target):
        r"""
        Utilization ratio via ray from an arbitrary base point.

        .. math::

            \eta_{\text{path}}
            = \frac{|\mathbf{T} - \mathbf{B}|}
                   {|\mathbf{R} - \mathbf{B}|}

        Parameters
        ----------
        N_base, Mx_base, My_base : float
            Base point [N, N·mm].
        N_target, Mx_target, My_target : float
            Target point [N, N·mm].

        Returns
        -------
        float
        """
        base = self._make_point(N_base, Mx_base, My_base)
        target = self._make_point(N_target, Mx_target, My_target)
        return self._ray_eta(base, target)


# ==================================================================
#  MxMyContour — 2-D contour at fixed N
# ==================================================================

class MxMyContour:
    r"""
    Mx-My interaction contour at a fixed axial force.

    Wraps a 2-D ConvexHull in :math:`(M_x, M_y)` space and provides
    a ray-cast method for :math:`\eta_{\text{2D}}`.

    Parameters
    ----------
    mx_my_data : dict
        Output of :meth:`NMDiagram.generate_mx_my`.  Must contain
        ``Mx`` [N·mm], ``My`` [N·mm].

    Attributes
    ----------
    N_fixed : float
        The axial force [N] at which this contour was generated.
    hull : scipy.spatial.ConvexHull
    """

    def __init__(self, mx_my_data):
        Mx = mx_my_data["Mx"]
        My = mx_my_data["My"]
        self.N_fixed = mx_my_data.get("N_fixed_kN", 0.0) * 1e3
        pts = np.column_stack([Mx, My])
        self.hull = ConvexHull(pts)
        self._equations = self.hull.equations

    def _ray_eta_2d(self, base_mx, base_my, target_mx, target_my):
        r"""
        2-D ray-cast from (base_mx, base_my) through (target_mx,
        target_my) to the Mx-My contour.

        Parameters
        ----------
        base_mx, base_my : float
            Base point in Mx-My space [N·mm].
        target_mx, target_my : float
            Target point in Mx-My space [N·mm].

        Returns
        -------
        float
            :math:`\eta`.
        """
        base = np.array([base_mx, base_my], dtype=float)
        target = np.array([target_mx, target_my], dtype=float)
        direction = target - base
        length = np.linalg.norm(direction)
        if length < 1e-6:
            inside = bool(np.all(
                self._equations[:, :-1] @ base
                + self._equations[:, -1] <= 1e-6))
            return 0.0 if inside else float("inf")

        A = self._equations[:, :-1]
        c = self._equations[:, -1]
        denom = A @ direction
        numer = -(A @ base + c)
        valid = np.abs(denom) > 1e-15
        t = np.full_like(denom, np.inf)
        t[valid] = numer[valid] / denom[valid]
        t[t <= 1e-12] = np.inf
        t_bnd = float(np.min(t))

        if t_bnd == np.inf:
            return 0.0 if self.is_inside(base[0], base[1]) else float("inf")
        return 1.0 / t_bnd

    def is_inside(self, Mx, My):
        """
        Check whether a point lies inside the Mx-My contour.

        Parameters
        ----------
        Mx, My : float
            Moment components [N·mm].

        Returns
        -------
        bool
        """
        p = np.array([Mx, My], dtype=float)
        return bool(np.all(
            self._equations[:, :-1] @ p + self._equations[:, -1] <= 1e-6
        ))

    def eta_2D(self, Mx, My):
        r"""
        :math:`\eta_{\text{2D}}`: ray from origin to demand in the
        :math:`M_x`-:math:`M_y` plane.

        Parameters
        ----------
        Mx, My : float
            Demand moments [N·mm].

        Returns
        -------
        float
        """
        return self._ray_eta_2d(0.0, 0.0, Mx, My)

    def eta_path_2D(self, Mx_base, My_base, Mx_target, My_target):
        r"""
        :math:`\eta_{\text{path,2D}}`: ray from base to target in the
        :math:`M_x`-:math:`M_y` plane.

        Parameters
        ----------
        Mx_base, My_base : float
            Base moments [N·mm].
        Mx_target, My_target : float
            Target moments [N·mm].

        Returns
        -------
        float
        """
        return self._ray_eta_2d(Mx_base, My_base, Mx_target, My_target)


# ==================================================================
#  VerificationEngine — flag-driven orchestrator
# ==================================================================

class VerificationEngine:
    r"""
    Top-level orchestrator for demand verification.

    Resolves demands, combinations and envelopes, computes all
    enabled :math:`\eta` types, and manages a cache of
    :class:`MxMyContour` instances for the N-levels actually needed.

    Parameters
    ----------
    nm_3d : dict
        3-D point cloud from :meth:`NMDiagram.generate_biaxial`
        (or uniaxial :meth:`~NMDiagram.generate` fallback).
    nm_gen : NMDiagram
        Generator instance for on-demand Mx-My contour production.
    output_flags : dict
        Parsed output block from YAML.  Expected keys:

        - ``eta_3D`` (bool, default ``True``)
        - ``eta_2D`` (bool, default ``False``)
        - ``eta_path`` (bool, default ``True``)
        - ``eta_path_2D`` (bool, default ``False``)
        - ``delta_N_tol`` (float, default ``0.03``)
        - ``n_angles_mx_my`` (int, default ``144``)
    n_points : int, optional
        Resolution passed to ``generate_mx_my``. Default 200.
    """

    def __init__(self, nm_3d, nm_gen, output_flags, n_points=200):
        self.domain = DomainChecker(nm_3d)
        self.nm_gen = nm_gen
        self.n_points = n_points

        # Parse flags with defaults.
        self.do_3D = output_flags.get("eta_3D", True)
        self.do_2D = output_flags.get("eta_2D", False)
        self.do_path = output_flags.get("eta_path", True)
        self.do_path_2D = output_flags.get("eta_path_2D", False)
        self.delta_N_tol = float(output_flags.get("delta_N_tol", 0.03))
        self.n_angles = int(output_flags.get("n_angles_mx_my", 144))

        # Auto-disable 2-D contour operations when the resistance
        # domain is 2-D (uniaxial N-M only).  In that case every
        # Mx-My contour is degenerate (a single point or segment)
        # and QHull cannot build a 2-D convex hull.
        if self.domain.ndim == 2 and (self.do_2D or self.do_path_2D):
            import sys
            print("  INFO: eta_2D / eta_path_2D disabled — resistance "
                  "domain is 2-D (no My data).  Consider using a "
                  "biaxial mesh (n_fibers_x > 1) or GenericSection.",
                  file=sys.stderr)
            self.do_2D = False
            self.do_path_2D = False

        # Contour cache: N_level (float, in N) → MxMyContour or None.
        self._contour_cache = {}

    # ------------------------------------------------------------------
    #  Contour cache management
    # ------------------------------------------------------------------

    def _get_contour(self, N_fixed):
        r"""
        Return the :class:`MxMyContour` at *N_fixed*, generating it
        on demand and caching the result.

        If the contour is degenerate (e.g. at an extreme N near
        pure compression/tension where the cross-section collapses
        to a point), ``None`` is returned and cached so the
        expensive generation is not retried.

        Parameters
        ----------
        N_fixed : float
            Axial force [N].

        Returns
        -------
        MxMyContour or None
            ``None`` when the contour cannot be built (degenerate
            geometry at this N level).
        """
        # Use a coarse key to avoid near-duplicate contours.
        key = round(N_fixed, 0)
        if key not in self._contour_cache:
            try:
                mx_my_data = self.nm_gen.generate_mx_my(
                    N_fixed,
                    n_angles=self.n_angles,
                    n_points_per_angle=self.n_points,
                )
                self._contour_cache[key] = MxMyContour(mx_my_data)
            except Exception:
                # Degenerate contour (collinear points, extreme N,
                # near-zero moment capacity, …).
                # Cache None to avoid retrying.
                self._contour_cache[key] = None
        return self._contour_cache[key]

    # ------------------------------------------------------------------
    #  Demand resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_ref(ref_name, demand_db, combination_results=None):
        r"""
        Look up a demand name and return the resolved force triple.

        Resolution order: ``demand_db`` first, then
        ``combination_results`` (using the combination resultant).

        Parameters
        ----------
        ref_name : str
        demand_db : dict
            ``{name: {"N": ..., "Mx": ..., "My": ...}}``.
        combination_results : dict or None
            ``{name: <combination result dict>}``.

        Returns
        -------
        tuple of float
            ``(N, Mx, My)`` in [N, N·mm, N·mm].

        Raises
        ------
        KeyError
            If *ref_name* is not found in either database.
        """
        if ref_name in demand_db:
            d = demand_db[ref_name]
            return d["N"], d["Mx"], d["My"]
        if combination_results and ref_name in combination_results:
            r = combination_results[ref_name]["resultant"]
            return r["N"], r["Mx"], r["My"]
        raise KeyError(
            f"Reference '{ref_name}' not found in demands or "
            f"combinations."
        )

    @staticmethod
    def resolve_components(components, demand_db):
        r"""
        Sum factored components into a single force triple.

        .. math::

            \mathbf{S} = \sum_i f_i \, \mathbf{d}_i

        Parameters
        ----------
        components : list of dict
            Each dict has ``ref`` (str) and optionally ``factor``
            (float, default 1.0).
        demand_db : dict
            ``{name: {"N": ..., "Mx": ..., "My": ...}}``.

        Returns
        -------
        dict
            ``{"N": float, "Mx": float, "My": float}`` in
            [N, N·mm, N·mm].
        """
        N_sum = 0.0
        Mx_sum = 0.0
        My_sum = 0.0
        for comp in components:
            ref = comp["ref"]
            f = float(comp.get("factor", 1.0))
            d = demand_db[ref]
            N_sum += f * d["N"]
            Mx_sum += f * d["Mx"]
            My_sum += f * d["My"]
        return {"N": N_sum, "Mx": Mx_sum, "My": My_sum}

    # ------------------------------------------------------------------
    #  Single-point η computation
    # ------------------------------------------------------------------

    def _compute_etas(self, N, Mx, My):
        r"""
        Compute all enabled :math:`\eta` for a single demand point
        (base = origin).

        Parameters
        ----------
        N, Mx, My : float
            Demand forces [N, N·mm].

        Returns
        -------
        dict
            Contains whichever of ``eta_3D``, ``eta_2D`` are enabled,
            plus ``inside`` and ``verified``.
        """
        result = {}
        etas = []

        if self.do_3D:
            e3 = self.domain.eta_3D(N, Mx, My)
            result["eta_3D"] = round(e3, 4)
            etas.append(e3)

        if self.do_2D:
            try:
                contour = self._get_contour(N)
                if contour is not None:
                    e2 = contour.eta_2D(Mx, My)
                    result["eta_2D"] = round(e2, 4)
                    etas.append(e2)
                else:
                    result["eta_2D"] = None
            except Exception as exc:
                import sys
                print(f"  WARNING: eta_2D failed at N={N/1e3:.1f} kN:"
                      f" {exc}", file=sys.stderr)
                result["eta_2D"] = None

        result["inside"] = self.domain.is_inside(N, Mx, My)
        result["verified"] = (all(e <= 1.0 for e in etas)
                              if etas else result["inside"])
        return result

    # ------------------------------------------------------------------
    #  Public: check a single demand
    # ------------------------------------------------------------------

    def check_demand(self, demand):
        r"""
        Verify a single demand point.

        Parameters
        ----------
        demand : dict
            ``{"name": str, "N": float, "Mx": float, "My": float}``.

        Returns
        -------
        dict
            Verification result with forces in kN/kN·m, enabled η
            values, ``inside``, ``verified``.
        """
        N, Mx, My = demand["N"], demand["Mx"], demand["My"]
        result = {
            "name": demand["name"],
            "N_kN": round(N / 1e3, 2),
            "Mx_kNm": round(Mx / 1e6, 4),
            "My_kNm": round(My / 1e6, 4),
        }
        result.update(self._compute_etas(N, Mx, My))
        return result

    def check_demands(self, demands):
        """
        Batch verification of demand points.

        Parameters
        ----------
        demands : list of dict

        Returns
        -------
        list of dict
        """
        return [self.check_demand(d) for d in demands]

    # ------------------------------------------------------------------
    #  Public: check a combination
    # ------------------------------------------------------------------

    def check_combination(self, combo, demand_db):
        r"""
        Verify a combination (simple or staged).

        For a **simple** combination (``components`` only), computes the
        factored resultant and returns :math:`\eta_{\text{3D}}` and
        :math:`\eta_{\text{2D}}` from the origin.

        For a **staged** combination, accumulates stages sequentially.
        Stage 0 is verified from the origin.  Stage :math:`k > 0` is
        verified from the cumulative point of stage :math:`k-1`:

        .. math::

            \eta_{\text{path},k}
            = \frac{|\mathbf{S}_k - \mathbf{S}_{k-1}|}
                   {|\mathbf{R} - \mathbf{S}_{k-1}|}

        :math:`\eta_{\text{path,2D}}` is computed only if the axial
        force jump satisfies:

        .. math::

            \frac{|N_{S_k} - N_{S_{k-1}}|}{N_{Rd,\max} - N_{Rd,\min}}
            < \delta_N

        Parameters
        ----------
        combo : dict
            Parsed combination spec with ``name`` and either
            ``components`` or ``stages``.
        demand_db : dict
            ``{name: {"N": ..., "Mx": ..., "My": ...}}``.

        Returns
        -------
        dict
            Structured result per §6 of the v2.1 spec.
        """
        name = combo["name"]

        if "stages" in combo:
            return self._check_staged(name, combo["stages"], demand_db)

        # Simple combination: flat factored sum.
        resultant = self.resolve_components(combo["components"],
                                            demand_db)
        N, Mx, My = resultant["N"], resultant["Mx"], resultant["My"]

        etas_dict = self._compute_etas(N, Mx, My)
        result = {
            "name": name,
            "type": "simple",
            "resultant": {
                "N_kN": round(N / 1e3, 2),
                "Mx_kNm": round(Mx / 1e6, 4),
                "My_kNm": round(My / 1e6, 4),
                "N": N, "Mx": Mx, "My": My,
            },
        }
        result.update(etas_dict)
        return result

    def _check_staged(self, name, stages, demand_db):
        r"""
        Internal: verify a staged combination.

        Parameters
        ----------
        name : str
        stages : list of dict
            Each stage has ``name`` and ``components``.
        demand_db : dict

        Returns
        -------
        dict
        """
        cum_N = 0.0
        cum_Mx = 0.0
        cum_My = 0.0

        stage_results = []
        all_etas = []

        for k, stage in enumerate(stages):
            inc = self.resolve_components(stage["components"], demand_db)
            dN, dMx, dMy = inc["N"], inc["Mx"], inc["My"]

            prev_N, prev_Mx, prev_My = cum_N, cum_Mx, cum_My
            cum_N += dN
            cum_Mx += dMx
            cum_My += dMy

            sr = {
                "name": stage["name"],
                "increment": {
                    "N_kN": round(dN / 1e3, 2),
                    "Mx_kNm": round(dMx / 1e6, 4),
                    "My_kNm": round(dMy / 1e6, 4),
                },
                "cumulative": {
                    "N_kN": round(cum_N / 1e3, 2),
                    "Mx_kNm": round(cum_Mx / 1e6, 4),
                    "My_kNm": round(cum_My / 1e6, 4),
                },
            }

            if k == 0:
                # Stage 0: base is origin → eta_3D / eta_2D.
                etas_dict = self._compute_etas(cum_N, cum_Mx, cum_My)
                sr.update(etas_dict)
                for key in ("eta_3D", "eta_2D"):
                    if key in sr:
                        all_etas.append(sr[key])
            else:
                # Stage k > 0: compute path-based η.
                sr["base"] = {
                    "N_kN": round(prev_N / 1e3, 2),
                    "Mx_kNm": round(prev_Mx / 1e6, 4),
                    "My_kNm": round(prev_My / 1e6, 4),
                }

                # eta_path (3D)
                if self.do_path:
                    ep = self.domain.eta_path(
                        prev_N, prev_Mx, prev_My,
                        cum_N, cum_Mx, cum_My)
                    sr["eta_path"] = round(ep, 4)
                    all_etas.append(ep)

                # eta_path_2D (conditional on delta_N)
                if self.do_path_2D:
                    delta_N_abs = abs(cum_N - prev_N)
                    N_range = self.domain.N_range
                    if N_range > 0:
                        delta_N_ratio = delta_N_abs / N_range
                    else:
                        delta_N_ratio = 0.0

                    if delta_N_ratio < self.delta_N_tol:
                        contour = self._get_contour(cum_N)
                        if contour is not None:
                            ep2 = contour.eta_path_2D(
                                prev_Mx, prev_My, cum_Mx, cum_My)
                            sr["eta_path_2D"] = round(ep2, 4)
                            all_etas.append(ep2)
                        else:
                            sr["eta_path_2D"] = None
                            sr["warning"] = (
                                "Mx-My contour degenerate at "
                                f"N={cum_N/1e3:.1f} kN, "
                                "eta_path_2D skipped"
                            )
                    else:
                        sr["eta_path_2D"] = None
                        sr["warning"] = (
                            f"delta_N={delta_N_ratio*100:.1f}% "
                            f"> tol={self.delta_N_tol*100:.1f}%, "
                            f"eta_path_2D skipped"
                        )

                # Also report eta_3D / eta_2D of cumulative point.
                cum_etas = self._compute_etas(cum_N, cum_Mx, cum_My)
                if "eta_3D" in cum_etas:
                    sr["eta_3D"] = cum_etas["eta_3D"]
                    all_etas.append(cum_etas["eta_3D"])
                if "eta_2D" in cum_etas:
                    sr["eta_2D"] = cum_etas["eta_2D"]
                    all_etas.append(cum_etas["eta_2D"])

            stage_results.append(sr)

        # Governing η.
        numeric_etas = [e for e in all_etas
                        if e is not None and np.isfinite(e)]
        eta_gov = max(numeric_etas) if numeric_etas else float("inf")

        inside = self.domain.is_inside(cum_N, cum_Mx, cum_My)
        return {
            "name": name,
            "type": "staged",
            "stages": stage_results,
            "resultant": {
                "N_kN": round(cum_N / 1e3, 2),
                "Mx_kNm": round(cum_Mx / 1e6, 4),
                "My_kNm": round(cum_My / 1e6, 4),
                "N": cum_N, "Mx": cum_Mx, "My": cum_My,
            },
            "eta_governing": round(eta_gov, 4),
            "inside": inside,
            "verified": eta_gov <= 1.0,
        }

    # ------------------------------------------------------------------
    #  Public: check an envelope
    # ------------------------------------------------------------------

    def check_envelope(self, envelope, demand_db,
                       combination_results=None):
        r"""
        Verify an envelope: report max :math:`\eta` among members.

        Each member can be a ``ref`` to a demand or combination, or an
        inline demand.  An optional ``factor`` scales the resolved
        resultant.

        Parameters
        ----------
        envelope : dict
            ``{"name": str, "members": [...]}``.
        demand_db : dict
            ``{name: {"N": ..., "Mx": ..., "My": ...}}``.
        combination_results : dict or None
            ``{name: <combination result dict>}``.

        Returns
        -------
        dict
            ``name``, ``eta_max``, ``governing_member``, ``verified``,
            ``members`` (list of per-member results).
        """
        env_name = envelope["name"]
        member_results = []
        all_etas = []

        for i, member in enumerate(envelope["members"]):
            factor = float(member.get("factor", 1.0))

            if "ref" in member:
                ref_name = member["ref"]
                N, Mx, My = self.resolve_ref(
                    ref_name, demand_db, combination_results)
                m_name = ref_name
            else:
                # Inline demand.
                N = float(member.get("N_kN", 0)) * 1e3
                Mx = float(member.get("Mx_kNm", 0)) * 1e6
                My = float(member.get("My_kNm", 0)) * 1e6
                m_name = member.get("name", f"{env_name}[{i}]")

            # Apply envelope-level factor.
            N *= factor
            Mx *= factor
            My *= factor

            mr = {
                "name": m_name,
                "N_kN": round(N / 1e3, 2),
                "Mx_kNm": round(Mx / 1e6, 4),
                "My_kNm": round(My / 1e6, 4),
            }
            mr.update(self._compute_etas(N, Mx, My))
            member_results.append(mr)

            # Collect numeric η for governing.
            for key in ("eta_3D", "eta_2D"):
                if key in mr and mr[key] is not None:
                    all_etas.append((mr[key], m_name))

        # Governing member.
        if all_etas:
            eta_max, gov_name = max(all_etas, key=lambda x: x[0])
        else:
            eta_max = 0.0
            gov_name = ""

        return {
            "name": env_name,
            "eta_max": round(eta_max, 4),
            "governing_member": gov_name,
            "verified": eta_max <= 1.0,
            "members": member_results,
        }