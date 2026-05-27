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
Lightweight analysis engine — force decomposition and on-demand
:math:`\eta`.

This module provides an alternative to the full
:class:`~gensec.solver.check.VerificationEngine` pipeline.  Instead
of pre-computing the entire resistance domain (which requires
hundreds of :meth:`~gensec.solver.FiberSolver.integrate` calls), it
operates *on demand*:

- **Force decomposition** (:meth:`AnalysisEngine.analyze`): solves
  equilibrium for a single load state and returns per-component
  force contributions (bulk zones, rebar groups, future element
  types).

- **On-demand** :math:`\eta` (:meth:`AnalysisEngine.compute_eta`):
  shoots a ray from a base point through the demand and finds the
  boundary via bisection on
  :meth:`~gensec.solver.FiberSolver.solve_equilibrium` +
  strain-limit check.  Cost: ~15–25 solves per demand instead of
  thousands for the full domain.

Usage
-----
.. code-block:: python

    from gensec.solver.integrator import FiberSolver
    from gensec.solver.analysis import AnalysisEngine

    solver = FiberSolver(section)
    engine = AnalysisEngine(solver)

    result = engine.analyze(N=-1500e3, Mx=200e6, My=0.0)
    for comp in result["components"]:
        print(comp["material_name"], comp["N"] / 1e3, "kN")

    eta = engine.compute_eta(N=-1500e3, Mx=200e6, My=0.0)
"""

import numpy as np
from .integrator import FiberSolver


# ==================================================================
#  Helper: build material-name reverse map
# ==================================================================

def _build_material_name_map(solver):
    r"""
    Build a mapping ``{id(Material): name}`` from the section
    attached to *solver*.

    Falls back to class-name labels when the ``name`` attribute is
    not set (materials created without the YAML loader).

    Parameters
    ----------
    solver : FiberSolver

    Returns
    -------
    dict
        ``{id(mat_object): str}``.
    """
    name_map = {}
    sec = solver.sec

    # Bulk materials.
    bm = sec.bulk_material
    name_map[id(bm)] = getattr(bm, "name", type(bm).__name__) or type(bm).__name__
    if hasattr(sec, "get_all_bulk_materials"):
        for i, mat in enumerate(sec.get_all_bulk_materials()):
            if id(mat) not in name_map:
                name_map[id(mat)] = getattr(
                    mat, "name", f"{type(mat).__name__}_{i}") or f"{type(mat).__name__}_{i}"

    # Rebar materials.
    for rb in sec.rebars:
        mid = id(rb.material)
        if mid not in name_map:
            name_map[mid] = getattr(
                rb.material, "name", type(rb.material).__name__) or type(rb.material).__name__

    return name_map


# ==================================================================
#  AnalysisEngine
# ==================================================================

class AnalysisEngine:
    r"""
    Lightweight analysis engine for force decomposition and
    on-demand :math:`\eta` computation.

    Unlike :class:`~gensec.solver.check.VerificationEngine`, this
    class does **not** require a pre-computed resistance domain.
    All operations are performed by calling
    :meth:`~gensec.solver.FiberSolver.solve_equilibrium` directly
    on each demand point.

    Parameters
    ----------
    solver : FiberSolver
        Configured fiber solver with section and materials.

    Notes
    -----
    Material names in the output are resolved from the
    ``Material.name`` attribute, which is set by the YAML loader.
    If the attribute is absent (API-constructed materials), the
    class name is used as a fallback.
    """

    def __init__(self, solver):
        self.solver = solver
        self._name_map = _build_material_name_map(solver)

    # ------------------------------------------------------------------
    #  Core: analyze a single load state
    # ------------------------------------------------------------------

    def analyze(self, N, Mx, My=0.0, tol=1e-3, max_iter=50):
        r"""
        Solve equilibrium and decompose forces by material component.

        Given a target load state :math:`(N,\,M_x,\,M_y)`, finds the
        strain plane :math:`(\varepsilon_0,\,\chi_x,\,\chi_y)` via
        Newton–Raphson and then aggregates internal forces by material:

        .. math::

            N_k   &= \sum_{i \in G_k} \sigma_i \, A_i \\
            M_{x,k} &= \sum_{i \in G_k} \sigma_i \, A_i
                        \,(y_i - y_{\text{ref}}) \\
            M_{y,k} &= -\sum_{i \in G_k} \sigma_i \, A_i
                        \,(x_i - x_{\text{ref}})

        where :math:`G_k` denotes fibers belonging to material
        group *k*.

        Parameters
        ----------
        N : float
            Axial force [N].  Positive = tension.
        Mx : float
            Bending moment about x-axis [N·mm].
        My : float, optional
            Bending moment about y-axis [N·mm].
        tol : float, optional
            Equilibrium tolerance [N].  Default 1e-3.
        max_iter : int, optional
            Maximum Newton iterations.  Default 50.

        Returns
        -------
        dict
            Top-level keys:

            - ``converged`` (bool)
            - ``strain_state`` — ``{eps0, chi_x, chi_y}``
            - ``total`` — ``{N, Mx, My}`` (confirmed by integration)
            - ``components`` — list of per-material dicts, each with
              ``{type, material_name, N, Mx, My, ...}``
            - ``strains_ok`` (bool) — whether all fiber strains lie
              within the material limits
        """
        sol = self.solver.solve_equilibrium(
            N, Mx, My, tol=tol, max_iter=max_iter)

        if not sol["converged"]:
            return {
                "converged": False,
                "strain_state": {
                    "eps0": sol["eps0"],
                    "chi_x": sol["chi_x"],
                    "chi_y": sol["chi_y"],
                },
                "total": {"N": sol["N"], "Mx": sol["Mx"],
                          "My": sol["My"]},
                "components": [],
                "strains_ok": False,
            }

        eps0 = sol["eps0"]
        chi_x = sol["chi_x"]
        chi_y = sol["chi_y"]

        components = self._decompose(eps0, chi_x, chi_y)
        strains_ok = self.solver.strains_within_limits(
            eps0, chi_x, chi_y)

        return {
            "converged": True,
            "strain_state": {
                "eps0": eps0,
                "chi_x": chi_x,
                "chi_y": chi_y,
            },
            "total": {"N": sol["N"], "Mx": sol["Mx"],
                      "My": sol["My"]},
            "components": components,
            "strains_ok": strains_ok,
        }

    # ------------------------------------------------------------------
    #  Batch: demands and combinations
    # ------------------------------------------------------------------

    def analyze_demands(self, demands, tol=1e-3, max_iter=50):
        r"""
        Analyze a list of demand dicts.

        Parameters
        ----------
        demands : list of dict
            Each dict must have ``name``, ``N``, ``Mx``, ``My``
            (forces in N / N·mm, as produced by
            :func:`~gensec.io_yaml.load_yaml`).
        tol : float, optional
        max_iter : int, optional

        Returns
        -------
        list of dict
            Each entry is the output of :meth:`analyze` enriched
            with ``name``, ``N_kN``, ``Mx_kNm``, ``My_kNm``.
        """
        results = []
        for d in demands:
            r = self.analyze(d["N"], d["Mx"], d["My"],
                             tol=tol, max_iter=max_iter)
            r["name"] = d["name"]
            r["N_kN"] = round(d["N"] / 1e3, 2)
            r["Mx_kNm"] = round(d["Mx"] / 1e6, 4)
            r["My_kNm"] = round(d["My"] / 1e6, 4)
            results.append(r)
        return results

    def analyze_combinations(self, combinations, demand_db,
                             tol=1e-3, max_iter=50):
        r"""
        Resolve and analyze combinations.

        For **simple** combinations (``components`` key), computes
        the factored resultant and analyzes it.  For **staged**
        combinations (``stages`` key), analyzes each cumulative stage
        sequentially and collects per-stage decompositions.

        Parameters
        ----------
        combinations : list of dict
            Parsed combination specs (from
            :func:`~gensec.io_yaml.load_yaml`).
        demand_db : dict
            ``{name: {"N": ..., "Mx": ..., "My": ...}}``.
        tol : float, optional
        max_iter : int, optional

        Returns
        -------
        list of dict
            One entry per combination, with ``name``, ``type``,
            ``resultant``, ``components`` (simple) or ``stages``
            (staged), each including the decomposition.
        """
        results = []
        for combo in combinations:
            name = combo["name"]

            if "stages" in combo:
                result = self._analyze_staged(
                    name, combo["stages"], demand_db,
                    tol, max_iter)
            else:
                res = _resolve_components(
                    combo["components"], demand_db)
                ar = self.analyze(
                    res["N"], res["Mx"], res["My"],
                    tol=tol, max_iter=max_iter)
                ar["name"] = name
                ar["type"] = "simple"
                ar["N_kN"] = round(res["N"] / 1e3, 2)
                ar["Mx_kNm"] = round(res["Mx"] / 1e6, 4)
                ar["My_kNm"] = round(res["My"] / 1e6, 4)
                result = ar

            results.append(result)
        return results

    # ------------------------------------------------------------------
    #  On-demand η (no pre-computed domain)
    # ------------------------------------------------------------------

    def compute_eta(self, N, Mx, My=0.0,
                    base_N=0.0, base_Mx=0.0, base_My=0.0,
                    tol=1e-3, max_iter=50, n_bisect=30):
        r"""
        On-demand utilization ratio via ray–bisection.

        Shoots a ray from ``(base_N, base_Mx, base_My)`` through
        ``(N, Mx, My)`` and finds the domain boundary by bisection
        on :meth:`~gensec.solver.FiberSolver.solve_equilibrium` +
        strain-limit check.

        .. math::

            \eta = \frac{|\mathbf{D} - \mathbf{B}|}
                        {|\mathbf{R} - \mathbf{B}|}

        where :math:`\mathbf{D}` is the demand, :math:`\mathbf{B}`
        the base point, and :math:`\mathbf{R}` the boundary point.

        The domain is convex; when the base point is inside (which
        is guaranteed for the origin or any previously verified
        stage), the ray crosses the boundary exactly once, so
        bisection converges monotonically.

        Parameters
        ----------
        N, Mx, My : float
            Demand point [N, N·mm].
        base_N, base_Mx, base_My : float, optional
            Base point of the ray.  Default (0, 0, 0).
        tol : float, optional
            Equilibrium solver tolerance [N].
        max_iter : int, optional
            Max Newton iterations per equilibrium solve.
        n_bisect : int, optional
            Number of bisection iterations.  Default 30
            (relative precision ~1e-9).

        Returns
        -------
        dict
            ``eta`` (float), ``boundary`` (dict with N, Mx, My of
            the boundary point), ``demand_inside`` (bool),
            ``converged`` (bool — whether the boundary search
            succeeded).
        """
        dN = N - base_N
        dMx = Mx - base_Mx
        dMy = My - base_My
        norm = np.sqrt(dN**2 + dMx**2 + dMy**2)
        if norm < 1e-12:
            return {"eta": 0.0,
                    "boundary": {"N": N, "Mx": Mx, "My": My},
                    "demand_inside": True, "converged": True}

        # Phase 1: exponential scan to bracket the boundary.
        t_lo = 0.0
        t_hi = None

        demand_inside = self._is_feasible(
            N, Mx, My, tol, max_iter)

        if not demand_inside:
            t_hi = 1.0
        else:
            t = 2.0
            for _ in range(20):
                pt_N = base_N + t * dN
                pt_Mx = base_Mx + t * dMx
                pt_My = base_My + t * dMy
                if self._is_feasible(pt_N, pt_Mx, pt_My,
                                     tol, max_iter):
                    t_lo = t
                    t *= 2.0
                else:
                    t_hi = t
                    break
            else:
                return {"eta": 0.0,
                        "boundary": {"N": 0.0, "Mx": 0.0,
                                     "My": 0.0},
                        "demand_inside": True,
                        "converged": False}

        # Phase 2: bisection on [t_lo, t_hi].
        for _ in range(n_bisect):
            t_mid = 0.5 * (t_lo + t_hi)
            pt_N = base_N + t_mid * dN
            pt_Mx = base_Mx + t_mid * dMx
            pt_My = base_My + t_mid * dMy
            if self._is_feasible(pt_N, pt_Mx, pt_My,
                                 tol, max_iter):
                t_lo = t_mid
            else:
                t_hi = t_mid

        t_bnd = t_lo
        bnd_N = base_N + t_bnd * dN
        bnd_Mx = base_Mx + t_bnd * dMx
        bnd_My = base_My + t_bnd * dMy

        eta = 1.0 / t_bnd if t_bnd > 1e-12 else float("inf")

        return {
            "eta": round(eta, 6),
            "boundary": {
                "N": bnd_N, "Mx": bnd_Mx, "My": bnd_My,
            },
            "demand_inside": demand_inside,
            "converged": True,
        }

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------

    def _is_feasible(self, N, Mx, My, tol, max_iter):
        r"""
        Check whether a load state is inside the resistance domain.

        A point is feasible if :meth:`solve_equilibrium` converges
        **and** all fiber strains lie within their material limits.
        """
        sol = self.solver.solve_equilibrium(
            N, Mx, My, tol=tol, max_iter=max_iter)
        if not sol["converged"]:
            return False
        return self.solver.strains_within_limits(
            sol["eps0"], sol["chi_x"], sol["chi_y"])

    def _decompose(self, eps0, chi_x, chi_y):
        r"""
        Aggregate internal forces by material component.

        Uses the same grouping structures as the integrator
        (``_bulk_groups``, ``_rebar_groups``) to ensure consistency
        with :meth:`~gensec.solver.FiberSolver.integrate`.
        """
        sv = self.solver
        sec = sv.sec
        eb, er = sv.strain_field(eps0, chi_x, chi_y)
        components = []

        # ---- Bulk zones ----
        for zone_idx, (mat, idx) in enumerate(sv._bulk_groups):
            sb = mat.stress_array(eb[idx])
            a = sec.A_fibers[idx]
            ly = sv._ly_bulk[idx]
            lx = sv._lx_bulk[idx]
            fA = sb * a

            N_c = float(np.sum(fA))
            Mx_c = float(np.sum(fA * ly))
            My_c = -float(np.sum(fA * lx))

            mat_name = self._name_map.get(
                id(mat), type(mat).__name__)
            components.append({
                "type": "bulk",
                "material_name": mat_name,
                "zone": zone_idx,
                "N": N_c,
                "Mx": Mx_c,
                "My": My_c,
                "N_kN": round(N_c / 1e3, 4),
                "Mx_kNm": round(Mx_c / 1e6, 6),
                "My_kNm": round(My_c / 1e6, 6),
            })

        # ---- Rebar groups ----
        embedded = sec.embedded_rebars
        for mat, bulk_mat, idx in sv._rebar_groups:
            er_g = er[idx]
            s_rebar = mat.stress_array(er_g)
            sb_at_rebars = bulk_mat.stress_array(er_g)
            a = sec.A_rebars[idx]
            emb = embedded[idx]
            ly_r = sv._ly_rebar[idx]
            lx_r = sv._lx_rebar[idx]

            s_net = s_rebar.copy()
            s_net[emb] -= sb_at_rebars[emb]
            fA = s_net * a

            N_s = float(np.sum(fA))
            Mx_s = float(np.sum(fA * ly_r))
            My_s = -float(np.sum(fA * lx_r))

            mat_name = self._name_map.get(
                id(mat), type(mat).__name__)

            layers = []
            for j, gi in enumerate(idx):
                layers.append({
                    "index": int(gi),
                    "x": float(sec.x_rebars[gi]),
                    "y": float(sec.y_rebars[gi]),
                    "A": float(a[j]),
                    "eps": float(er_g[j]),
                    "sigma_gross": float(s_rebar[j]),
                    "sigma_net": float(s_net[j]),
                    "F_net_kN": round(
                        float(s_net[j] * a[j]) / 1e3, 4),
                })

            components.append({
                "type": "rebar",
                "material_name": mat_name,
                "N": N_s,
                "Mx": Mx_s,
                "My": My_s,
                "N_kN": round(N_s / 1e3, 4),
                "Mx_kNm": round(Mx_s / 1e6, 6),
                "My_kNm": round(My_s / 1e6, 6),
                "layers": layers,
            })

        return components

    def _analyze_staged(self, name, stages, demand_db,
                        tol, max_iter):
        r"""
        Analyze a staged combination.

        Each stage is resolved cumulatively and analyzed
        independently, producing per-stage force decompositions.
        """
        cum_N = 0.0
        cum_Mx = 0.0
        cum_My = 0.0
        stage_results = []

        for stage in stages:
            res = _resolve_components(
                stage["components"], demand_db)
            cum_N += res["N"]
            cum_Mx += res["Mx"]
            cum_My += res["My"]

            ar = self.analyze(cum_N, cum_Mx, cum_My,
                              tol=tol, max_iter=max_iter)
            ar["name"] = stage["name"]
            ar["cumulative"] = {
                "N_kN": round(cum_N / 1e3, 2),
                "Mx_kNm": round(cum_Mx / 1e6, 4),
                "My_kNm": round(cum_My / 1e6, 4),
            }
            stage_results.append(ar)

        return {
            "name": name,
            "type": "staged",
            "resultant": {
                "N_kN": round(cum_N / 1e3, 2),
                "Mx_kNm": round(cum_Mx / 1e6, 4),
                "My_kNm": round(cum_My / 1e6, 4),
                "N": cum_N, "Mx": cum_Mx, "My": cum_My,
            },
            "stages": stage_results,
        }


# ==================================================================
#  Standalone helpers
# ==================================================================

def _resolve_components(components, demand_db):
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
        ``{"N": float, "Mx": float, "My": float}``
        in [N, N·mm, N·mm].

    Raises
    ------
    KeyError
        If a referenced demand name is not found.
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
