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

Implements the GenSec v0.3 demand architecture with seven
utilization ratio types, staged combinations, and envelopes.

Utilization ratios
------------------
GenSec computes up to seven utilization metrics, organised in two
geometric families.

**3-D family** — operates in *anisotropy-corrected normalised
space*: the axial axis is rescaled by
:math:`u_x = \Delta M_x / \Delta N` and (in 3-D) the :math:`M_y`
axis by :math:`v_y = \Delta M_x / \Delta M_y`, so that the
bounding box of the resistance domain has the same numeric extent
on every axis.  This makes euclidean distances physically
meaningful even for strongly anisotropic sections (walls, slabs)
and makes the metrics scale-invariant under any consistent change
of force / moment units.

Point metrics:

- ``eta_norm`` — linear distance from the demand to the boundary,
  expressed as a fraction of the Chebyshev radius
  :math:`D_{\max}` of the normalised domain:
  :math:`\eta = 1 - d_{\min}/D_{\max}`.  A true geometric
  distance.  Monotone in proximity to the boundary; reaches 0
  at the Chebyshev centre and 1 on the boundary.
- ``eta_norm_beta`` — composite ratio
  :math:`F_{SU}/(F_{SU}+d_{\min})`.  Mixes the demand norm
  :math:`F_{SU}` with the face-distance :math:`d_{\min}`.
  Geometrically not a distance: a ratio that grows with both
  proximity to the boundary and lontananza dall'origine.  The
  semantic reading is "sensitivity to perturbation in proportion
  to the demand magnitude".  Useful for cross-software validation.
- ``eta_norm_ray`` — ray-cast from the origin to the demand in
  normalised space.  Answers "if I scale all three force
  components proportionally, when does the demand exit?"

Path metrics (staged combinations):

- ``eta_path_norm_ray`` — ray-cast from the previous stage's
  cumulative demand to the current one.  Path-direction-aware.
- ``eta_path_norm_beta`` — composite-ratio analogue of
  ``eta_norm_beta`` along the stage segment :math:`B \to T`:
  :math:`L/(L+d_{\text{seg}})`, where :math:`L = |T-B|` and
  :math:`d_{\text{seg}}` is the minimum signed distance of the
  segment from the boundary.

**2-D family** — operates in the :math:`(M_x, M_y)` plane at
fixed :math:`N`.  Both axes already share the same physical units;
no normalisation is needed.

- ``eta_2D`` — ray-cast from the origin in the
  :math:`(M_x, M_y)` plane to the contour at :math:`N`.
- ``eta_path_2D`` — staged variant on the contour at the
  cumulative :math:`N`.  Skipped when the axial-force change
  between stages exceeds a configurable tolerance.

+----------------------------+--------------------------+--------------------------+
| Metric                     | Geometric primitive      | Domain frame             |
+============================+==========================+==========================+
| ``eta_norm``               | linear distance fraction | normalised 3-D hull      |
+----------------------------+--------------------------+--------------------------+
| ``eta_norm_beta``          | composite ratio          | normalised 3-D hull      |
+----------------------------+--------------------------+--------------------------+
| ``eta_norm_ray``           | ray from origin          | normalised 3-D hull      |
+----------------------------+--------------------------+--------------------------+
| ``eta_path_norm_ray``      | ray from base (staged)   | normalised 3-D hull      |
+----------------------------+--------------------------+--------------------------+
| ``eta_path_norm_beta``     | composite ratio (segment)| normalised 3-D hull      |
+----------------------------+--------------------------+--------------------------+
| ``eta_2D``                 | ray from origin          | 2-D slice at :math:`N`   |
+----------------------------+--------------------------+--------------------------+
| ``eta_path_2D``            | ray from base (staged)   | 2-D slice at cumulative N|
+----------------------------+--------------------------+--------------------------+

Each metric answers a distinct geometric question.  No strict
ordering between them holds in general — they are complementary,
not redundant.  For verification, the governing :math:`\eta` is
the maximum across all enabled metrics.

Classes
-------
DomainChecker
    3-D ConvexHull operations (``eta_norm``, ``eta_norm_beta``,
    ``eta_norm_ray``, ``eta_path_norm_ray``,
    ``eta_path_norm_beta``, ``is_inside``).
MxMyContour
    2-D contour at fixed *N* (``eta_2D``, ``eta_path_2D``).
VerificationEngine
    Flag-driven orchestrator that resolves demands, combinations and
    envelopes, managing contour caches and producing structured results.
"""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog


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
        Convex hull of the resistance domain in **normalised
        coordinates** (see :meth:`eta_norm` for the construction).
        All ray-cast queries (:meth:`eta_norm`, :meth:`eta_path`,
        :meth:`is_inside`) operate on this hull.
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

        # Axial range — needed both for normalisation and for the
        # delta_N tolerance check used by eta_path_2D.
        self.N_range = float(pts[:, 0].max() - pts[:, 0].min())

        # ----------------------------------------------------------
        #  Anisotropic normalisation factors.
        # ----------------------------------------------------------
        # Goal: rescale the resistance domain so that the bounding
        # box becomes numerically isotropic — every axis has the
        # same span.  Picking ``Mx`` as the reference span:
        #
        #     u_x = ΔMx / ΔN          (rescales N → Mx-equivalent)
        #     v_y = ΔMx / ΔMy         (rescales My → Mx-equivalent)
        #
        # In the resulting space, ΔN·u_x = ΔMx, ΔMy·v_y = ΔMx, so
        # all three dimensions span the same numeric extent.  This
        # makes euclidean distances physically meaningful even for
        # strongly anisotropic sections (e.g. walls 187x30 cm,
        # where ΔMx/ΔMy can exceed 30).
        Mx_span = float(pts[:, 1].max() - pts[:, 1].min())
        if self.ndim == 3:
            My_span = float(pts[:, 2].max() - pts[:, 2].min())
        else:
            My_span = 0.0

        if self.N_range > 1e-9:
            self._u_x = Mx_span / self.N_range
        else:
            # Degenerate axial range: skip N rescaling.
            self._u_x = 1.0

        if self.ndim == 3 and My_span > 1e-9:
            self._v_y = Mx_span / My_span
        else:
            self._v_y = 1.0

        # ----------------------------------------------------------
        #  Build the (single) hull in normalised coordinates.
        # ----------------------------------------------------------
        pts_norm = pts.copy()
        pts_norm[:, 0] *= self._u_x
        if self.ndim == 3:
            pts_norm[:, 2] *= self._v_y

        self.hull = ConvexHull(pts_norm)
        self._equations = self.hull.equations
        self._pts_norm = pts_norm

        # ----------------------------------------------------------
        #  D_max: radius of the largest sphere inscribed in the
        #  normalised hull (Chebyshev radius).  Used as the
        #  reference scale for ``eta_norm`` (alpha formulation):
        #  the demand's d_min is normalised against D_max so that
        #  eta_norm = 0 corresponds to the geometric "centre" of
        #  the domain (the Chebyshev centre) and eta_norm = 1 to
        #  the boundary itself.
        #
        #  D_max is intrinsic to the domain — it does not depend on
        #  the demand — so it is computed once at construction.
        # ----------------------------------------------------------
        self._D_max = self._compute_chebyshev_radius()

    # ------------------------------------------------------------------
    #  Point queries
    # ------------------------------------------------------------------

    def _make_point(self, N, Mx, My=0.0):
        """Build a point array consistent with hull dimensionality."""
        if self.ndim == 3:
            return np.array([N, Mx, My], dtype=float)
        return np.array([N, Mx], dtype=float)

    def _to_norm(self, p):
        """
        Map a raw point ``(N, Mx, [My])`` into the normalised space
        used by the convex hull.

        The mapping rescales the axial coordinate by ``u_x`` and
        (in 3-D) the ``My`` coordinate by ``v_y`` so that the
        resulting bounding box is numerically isotropic.
        """
        q = p.astype(float, copy=True)
        q[0] *= self._u_x
        if self.ndim == 3:
            q[2] *= self._v_y
        return q

    def _compute_chebyshev_radius(self):
        r"""
        Compute the radius of the largest sphere inscribed in the
        normalised hull (Chebyshev radius :math:`D_{\max}`).

        Notes
        -----
        Given face equations :math:`A x + b \le 0` from
        :class:`scipy.spatial.ConvexHull`, where each face normal
        ``a_i`` is unit-length, the Chebyshev centre and radius
        satisfy the linear program

        .. math::

            \max_{x,\, r} r \quad \text{s.t.} \quad
            a_i^\top x + b_i + r \le 0 \quad \forall i,
            \quad r \ge 0.

        Solved here with :func:`scipy.optimize.linprog`.

        Returns
        -------
        float
            The radius :math:`D_{\max}`, in the units of the
            normalised space.  A degenerate hull (LP infeasible or
            unbounded) returns the bounding-box semi-diagonal as a
            conservative fallback.
        """
        A = self._equations[:, :-1]      # (n_faces, ndim), unit normals
        b = self._equations[:, -1]       # (n_faces,)
        n = self.ndim

        # Decision variables: [x_0, x_1, ..., x_{n-1}, r].
        # Objective: maximise r → minimise -r.
        c = np.zeros(n + 1)
        c[-1] = -1.0

        # Constraints: a_i^T x + r <= -b_i  (since ||a_i|| = 1).
        A_ub = np.hstack([A, np.ones((A.shape[0], 1))])
        b_ub = -b

        # Bounds: x_i free, r >= 0.
        bounds = [(None, None)] * n + [(0.0, None)]

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                          method="highs")
            if res.success and res.x[-1] > 0:
                return float(res.x[-1])
        except Exception:
            pass

        # Fallback: bbox semi-diagonal in normalised space.
        spans = (self._pts_norm.max(axis=0)
                 - self._pts_norm.min(axis=0)) / 2
        return float(np.linalg.norm(spans))

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
        # Hull is in normalised coordinates → map the query point
        # through the same transform before testing the half-spaces.
        p = self._to_norm(self._make_point(N, Mx, My))
        return bool(np.all(
            self._equations[:, :-1] @ p + self._equations[:, -1] <= 1e-6
        ))

    # ------------------------------------------------------------------
    #  Ray-cast engine (shared by eta_norm and eta_path)
    #
    #  Operates on ``self._equations``, which describes the hull in
    #  the *normalised* coordinate system.  Callers must pass points
    #  already mapped through :meth:`_to_norm`.
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
    #
    #  All 3-D metrics operate in the anisotropy-corrected normalised
    #  coordinate system (axis-isotropic bounding box).  Five
    #  geometric primitives, five independent answers:
    #
    #    eta_norm          — alpha formulation: linear distance from
    #                        the demand to the boundary, normalised
    #                        by D_max.  A true distance.
    #
    #    eta_norm_beta     — composite ratio F_SU / (F_SU + d_min).
    #                        Sensitivity of the demand to a small
    #                        perturbation, weighted by its magnitude.
    #
    #    eta_norm_ray      — ray-cast from the origin to the demand:
    #                        proportional growth of all three force
    #                        components.
    #
    #    eta_path_norm_ray — ray-cast from the previous stage's
    #                        cumulative demand to the current one:
    #                        directional consumption along the
    #                        actual load path.
    #
    #    eta_path_norm_beta — composite ratio L / (L + d_seg) along
    #                        the stage segment B->T.  Path analogue
    #                        of eta_norm_beta: stage sensitivity in
    #                        proportion to its own magnitude.
    # ------------------------------------------------------------------

    def eta_norm(self, N, Mx, My=0.0):
        r"""
        Minimum-distance utilization ratio (alpha formulation).

        Computed in anisotropy-corrected normalised space, where the
        resistance domain has an isotropic bounding box.  The metric
        is the euclidean distance from the demand to the nearest
        face of the boundary, expressed as a fraction of the
        Chebyshev radius :math:`D_{\max}` of the domain (the radius
        of the largest sphere that fits inside the domain):

        .. math::

            \mathbf{S}_u
              &= (N\cdot u_x,\; M_x,\; M_y\cdot v_y),

            d_{\min}
              &= \min_{\text{faces}}
                 \big| \mathbf{n}^\top \mathbf{S}_u + d \big|,

            \eta_{\text{norm}}
              &= 1 - \frac{d_{\min}}{D_{\max}}
                 \quad \text{(if interior)},

            \eta_{\text{norm}}
              &= 1 + \frac{d_{\min}}{D_{\max}}
                 \quad \text{(if exterior)}.

        Geometrically, :math:`\eta_{\text{norm}}` answers
        "what fraction of the available reserve has the demand
        consumed in the *worst* direction?"  Reading:

        - :math:`\eta_{\text{norm}} = 0` — demand at the Chebyshev
          centre of the domain (the geometric "deepest" point).
        - :math:`\eta_{\text{norm}} = 1` — demand exactly on the
          boundary.
        - :math:`\eta_{\text{norm}} > 1` — demand outside, with the
          excess proportional to the penetration depth.

        Linear and monotone in distance to the boundary, with no
        dependence on :math:`|OS|`: this is the sense in which the
        metric is a "true distance" rather than a composite ratio.

        Parameters
        ----------
        N, Mx, My : float
            Demand forces [N, N·mm].

        Returns
        -------
        float
        """
        p = self._to_norm(self._make_point(N, Mx, My))

        # Signed distance to each face in normalised space.
        # ConvexHull.equations convention: A x + b <= 0 inside.
        A = self._equations[:, :-1]
        b = self._equations[:, -1]
        signed = A @ p + b
        is_inside = bool(np.all(signed <= 1e-6))
        d_min = float(np.min(np.abs(signed)))

        D_max = self._D_max
        if D_max <= 0:
            return float("inf")

        if is_inside:
            return 1.0 - d_min / D_max
        return 1.0 + d_min / D_max

    def eta_norm_beta(self, N, Mx, My=0.0):
        r"""
        Composite-ratio utilization (beta formulation).

        Same anisotropy-corrected normalised space as
        :meth:`eta_norm`, but combines the demand norm and the
        face-distance into a single ratio:

        .. math::

            \mathbf{S}_u
              &= (N\cdot u_x,\; M_x,\; M_y\cdot v_y),

            F_{SU}
              &= \lVert \mathbf{S}_u \rVert,

            d_{\min}
              &= \min_{\text{faces}}
                 \big| \mathbf{n}^\top \mathbf{S}_u + d \big|,

            \eta_{\text{norm,\beta}}
              &= \frac{F_{SU}}{F_{SU} + d_{\min}}
                 \quad \text{(if interior)},

            \eta_{\text{norm,\beta}}
              &= \frac{F_{SU}}{F_{SU} - d_{\min}}
                 \quad \text{(if exterior)}.

        Notes
        -----
        This metric is **not** a distance.  It is a composite ratio
        whose value depends both on the demand's position relative
        to the boundary and on its position relative to the
        coordinate origin.  For a fixed face-distance :math:`d_{\min}`,
        the metric grows with :math:`F_{SU}` — so demands far from
        the origin get larger values than demands close to the
        origin even at the same boundary distance.

        The semantic reading is "load-amplification sensitivity":
        if you perturb the demand by a small absolute amount, the
        metric estimates how big that perturbation is *in proportion
        to the magnitude of the demand itself*.  This is not the
        same question as "how close is the demand to failure", which
        is what :meth:`eta_norm` answers.  Both metrics are
        legitimate; they answer different questions and a strict
        ordering between them does not exist in general.

        Equivalent or near-equivalent formulations are documented in
        commercial reinforced-concrete analysis tools, so this
        metric is also useful for cross-software validation.

        Parameters
        ----------
        N, Mx, My : float
            Demand forces [N, N·mm].

        Returns
        -------
        float
            ``0.0`` if the demand coincides with the origin.
        """
        p = self._to_norm(self._make_point(N, Mx, My))
        F_SU = float(np.linalg.norm(p))
        if F_SU < 1e-12:
            return 0.0

        A = self._equations[:, :-1]
        b = self._equations[:, -1]
        signed = A @ p + b
        is_inside = bool(np.all(signed <= 1e-6))
        d_min = float(np.min(np.abs(signed)))

        if is_inside:
            return F_SU / (F_SU + d_min)
        denom = F_SU - d_min
        if denom <= 0:
            return float("inf")
        return F_SU / denom

    def eta_norm_ray(self, N, Mx, My=0.0):
        r"""
        Radial-growth utilization ratio: ray-cast from the origin.

        Same anisotropy-corrected normalised space as
        :meth:`eta_norm`, but uses the ray-cast primitive instead of
        the sphere-based distance:

        .. math::

            \eta_{\text{norm,ray}}
              = \frac{|\mathbf{S}_u|}{|\mathbf{R}_u|}

        where :math:`\mathbf{R}_u` is the first hit of the ray
        :math:`O \to \mathbf{S}_u` with the hull boundary.

        Geometrically, this metric answers "if I scale the demand
        proportionally in all three components, when does it exit
        the domain?".  This is the right question for proportional
        load amplification (overall load factor analyses).

        Note
        ----
        Unlike :meth:`eta_norm`, this metric **does not** satisfy
        :math:`\eta \le \eta_{\text{2D}}` in general.  The two
        metrics measure different geometric quantities and neither
        dominates the other.  For a worst-case verification, the
        governing :math:`\eta` is the maximum across all enabled
        metrics — :meth:`eta_norm` and :meth:`eta_norm_ray` are
        complementary, not redundant.

        Parameters
        ----------
        N, Mx, My : float
            Demand forces [N, N·mm].

        Returns
        -------
        float
        """
        origin = np.zeros(self.ndim)
        target = self._to_norm(self._make_point(N, Mx, My))
        return self._ray_eta(origin, target)

    def eta_path_norm_ray(self, N_base, Mx_base, My_base,
                          N_target, Mx_target, My_target):
        r"""
        Path-based utilization ratio: ray-cast from an arbitrary
        base point to the target, in anisotropy-corrected
        normalised space.

        .. math::

            \eta_{\text{path,norm,ray}}
              = \frac{|\mathbf{T}_u - \mathbf{B}_u|}
                     {|\mathbf{R}_u - \mathbf{B}_u|}

        where :math:`\mathbf{B}_u` and :math:`\mathbf{T}_u` are the
        base and target mapped through the same anisotropy
        correction as :meth:`eta_norm`, and :math:`\mathbf{R}_u` is
        the first hit of the segment with the boundary.

        Used by staged combinations: the base is the cumulative
        demand at the previous stage, the target the cumulative
        demand at the current stage.  The ray direction is the
        actual load path between the two stages — taking the ray
        from the origin would not represent the physical history
        of the structure.

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
        base = self._to_norm(
            self._make_point(N_base, Mx_base, My_base))
        target = self._to_norm(
            self._make_point(N_target, Mx_target, My_target))
        return self._ray_eta(base, target)

    def _segment_to_hull_min_distance(self, base, target):
        r"""
        Minimum (signed) distance between the segment
        :math:`B \to T` and the boundary of the normalised hull.

        For each face :math:`a_i^\top x + b_i \le 0` of the hull
        (with :math:`\|a_i\| = 1`), the signed distance from a
        point :math:`x` to the face is :math:`-(a_i^\top x + b_i)`
        (positive = inside, negative = outside).  Along the segment
        :math:`x(t) = B + t(T - B)`, :math:`t \in [0,1]`, the signed
        distance is **affine** in :math:`t`, so it attains its
        minimum at one of the endpoints.

        The metric returned is

        .. math::

            d_{\min,\text{seg}} =
            \min_{i,\,t\in\{0,1\}}
            \big(-(a_i^\top x(t) + b_i)\big),

        i.e. the minimum signed distance over both endpoints and
        all faces.  When :math:`d_{\min,\text{seg}} > 0` the entire
        segment is interior to the hull; when :math:`< 0` at least
        one endpoint is outside.

        Cost is :math:`O(n_{\text{faces}})`.

        Parameters
        ----------
        base, target : numpy.ndarray
            Segment endpoints in normalised coordinates.

        Returns
        -------
        float
            Minimum signed distance of the segment from the
            boundary, in the units of the normalised space.
        """
        A = self._equations[:, :-1]      # unit normals
        b = self._equations[:, -1]
        # Signed distance at each endpoint, for each face.
        s_base = A @ base + b            # negative = inside
        s_target = A @ target + b
        # Inside-distance (positive when inside): -s.
        # The minimum along the segment for a given face is the min
        # of the two endpoint values, since the function is affine.
        d_along_face = np.minimum(-s_base, -s_target)
        # Across all faces, take the minimum (= worst clearance).
        return float(np.min(d_along_face))

    def eta_path_norm_beta(self, N_base, Mx_base, My_base,
                            N_target, Mx_target, My_target):
        r"""
        Path-based composite-ratio utilization (staged beta).

        The staged analogue of :meth:`eta_norm_beta`: combines the
        magnitude of the path increment with the path's clearance
        from the boundary, both in anisotropy-corrected normalised
        space.

        .. math::

            \mathbf{B}_u, \mathbf{T}_u
              &= \text{base, target in normalised coords},

            L
              &= \lVert \mathbf{T}_u - \mathbf{B}_u \rVert,

            d_{\min,\text{seg}}
              &= \text{min signed distance of segment } B_u \to T_u
                 \text{ from the boundary},

            \eta_{\text{path,norm,\beta}}
              &= \frac{L}{L + d_{\min,\text{seg}}}
                 \quad \text{(segment interior)},

            \eta_{\text{path,norm,\beta}}
              &= \frac{L}{L - |d_{\min,\text{seg}}|}
                 \quad \text{(segment crosses or beyond boundary)}.

        Geometric reading: for a *staged* increment of magnitude
        :math:`L`, the metric reports how close the path approaches
        the boundary in proportion to its own size.

        - Small increment in a region already near the boundary
          (gravity stage already saturated, mild seismic
          increment): :math:`L` small, :math:`d_{\min,\text{seg}}`
          small → metric close to 1.  Correctly flags that the
          structure is critical despite the small new load.
        - Large increment far from the boundary: :math:`L` large,
          :math:`d_{\min,\text{seg}}` large → metric moderate.
          Significant load consumption in safe zone.
        - Large increment that approaches the boundary: both
          :math:`L` and the closeness contribute → metric close
          to 1.  Stage near saturation.

        Useful in seismic verification when the gravity stage is
        applied first and the seismic increment is the second
        stage: this metric distinguishes "structure already at
        the limit for gravity" from "increment large in the
        seismic direction".

        Parameters
        ----------
        N_base, Mx_base, My_base : float
            Base (cumulative previous stage) in [N, N·mm].
        N_target, Mx_target, My_target : float
            Target (cumulative current stage) in [N, N·mm].

        Returns
        -------
        float
        """
        base = self._to_norm(
            self._make_point(N_base, Mx_base, My_base))
        target = self._to_norm(
            self._make_point(N_target, Mx_target, My_target))

        L = float(np.linalg.norm(target - base))
        if L < 1e-12:
            # Zero-length segment: collapses to point β.
            return self.eta_norm_beta(N_target, Mx_target, My_target)

        d_seg = self._segment_to_hull_min_distance(base, target)
        if d_seg >= 0.0:
            return L / (L + d_seg)
        denom = L - abs(d_seg)
        if denom <= 0:
            return float("inf")
        return L / denom


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

        Point metrics (4):

        - ``eta_norm`` (bool, default ``True``).  Linear distance
          to the boundary, expressed as a fraction of the Chebyshev
          radius of the normalised domain.  The principal 3-D
          metric: a true geometric distance, monotone in the
          demand's proximity to the boundary.
        - ``eta_norm_beta`` (bool, default ``True``).  Composite
          ratio :math:`F_{SU}/(F_{SU}+d_{\min})`.  Sensitivity of
          the demand to small perturbations, weighted by its
          magnitude.  Useful for cross-software validation.
        - ``eta_norm_ray`` (bool, default ``False``).  Ray-cast
          from the origin in the same normalised space.  Answers
          "proportional load amplification": when does the demand
          exit if all components scale equally?
        - ``eta_2D`` (bool, default ``False``).  Ray-cast in the
          :math:`(M_x, M_y)` plane at the demand's :math:`N`.
          Flexural verification at fixed axial force.  Requires a
          biaxial section.

        Path metrics (3, for staged combinations):

        - ``eta_path`` (bool, default ``False``).  3-D ray-cast in
          normalised space, from the cumulative previous-stage
          demand to the cumulative current-stage demand.  Stored
          under the key ``eta_path_norm_ray`` in per-stage results.
        - ``eta_path_norm_beta`` (bool, default ``False``).
          Composite-ratio analogue of ``eta_norm_beta`` along the
          stage segment :math:`B \to T`: :math:`L/(L+d_{seg})`,
          where :math:`L = |T-B|` and :math:`d_{seg}` is the
          minimum signed distance of the segment from the boundary.
          Useful in seismic verification with staged gravity:
          flags "structure already at the limit before the
          increment" distinctly from ``eta_path_norm_ray``.
        - ``eta_path_2D`` (bool, default ``False``).  2-D variant
          on the :math:`M_x`-:math:`M_y` contour at cumulative
          :math:`N`.  Skipped when the axial-force change between
          stages exceeds ``delta_N_tol``.

        Other:

        - ``delta_N_tol`` (float, default ``0.03``)
        - ``n_angles_mx_my`` (int, default ``144``)

        Each metric answers a distinct geometric question; no strict
        ordering between them holds in general.  All enabled metrics
        contribute to ``verified``, ``eta_max`` and
        ``eta_governing``: pass/fail is decided by the worst of the
        enabled metrics.

        Any other key in ``output_flags`` is silently ignored,
        including the legacy ``eta_3D`` flag which has been removed
        in v0.3.
    n_points : int, optional
        Resolution passed to ``generate_mx_my``. Default 200.
    """

    def __init__(self, nm_3d, nm_gen, output_flags, n_points=200):
        self.domain = DomainChecker(nm_3d)
        self.nm_gen = nm_gen
        self.n_points = n_points

        # Parse flags with defaults.
        self.do_norm = output_flags.get("eta_norm", True)
        self.do_norm_beta = output_flags.get("eta_norm_beta", True)
        self.do_norm_ray = output_flags.get("eta_norm_ray", False)
        self.do_2D = output_flags.get("eta_2D", False)
        self.do_path = output_flags.get("eta_path_norm_ray", False)
        self.do_path_norm_beta = output_flags.get(
            "eta_path_norm_beta", False)
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
                  "domain is 2-D (no My data).  Use a biaxial mesh "
                  "(n_fibers_x > 1 AND n_fibers_y > 1) or a "
                  "GenericSection with non-degenerate extent in both "
                  "in-plane directions.",
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
            Contains whichever of ``eta_norm``, ``eta_norm_beta``,
            ``eta_norm_ray``, ``eta_2D`` are enabled, plus
            ``inside`` and ``verified``.

        Notes
        -----
        Pass/fail (``verified``) is decided by the worst of the
        enabled metrics.  The four single-point metrics answer
        different geometric questions and are not redundant:

        - ``eta_norm`` (alpha) — fraction of the available reserve
          (Chebyshev radius) consumed by the demand in the worst
          direction.  Linear distance to the boundary.
        - ``eta_norm_beta`` — composite ratio
          :math:`F_{SU}/(F_{SU}+d_{\min})`.  Sensitivity of the
          demand to perturbations in proportion to its magnitude.
        - ``eta_norm_ray`` — ray-cast from the origin in the same
          normalised space.  Proportional load amplification.
        - ``eta_2D`` — ray-cast in the :math:`(M_x,M_y)` plane at
          fixed :math:`N`.  Flexural growth at fixed axial force.
        """
        result = {}
        etas = []          # used for ``verified``

        if self.do_norm:
            en = self.domain.eta_norm(N, Mx, My)
            result["eta_norm"] = round(en, 4)
            etas.append(en)

        if self.do_norm_beta:
            enb = self.domain.eta_norm_beta(N, Mx, My)
            result["eta_norm_beta"] = round(enb, 4)
            etas.append(enb)

        if self.do_norm_ray:
            enr = self.domain.eta_norm_ray(N, Mx, My)
            result["eta_norm_ray"] = round(enr, 4)
            etas.append(enr)

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
        # Warn the user when a multi-stage combination is processed
        # with no path-aware metric enabled.  Without one of
        # ``eta_path``, ``eta_path_norm_beta`` or ``eta_path_2D``,
        # the report contains only the four point metrics evaluated
        # at each cumulative state — which is information-poor for
        # staged loading, since the loading *history* is not
        # captured.  Print once per staged combination, on stderr.
        if (len(stages) > 1
                and not (self.do_path
                         or self.do_path_norm_beta
                         or self.do_path_2D)):
            import sys
            print(
                f"  INFO: staged combination '{name}' has "
                f"{len(stages)} stages but no path-aware metric is "
                "enabled (eta_path, eta_path_norm_beta, "
                "eta_path_2D).  The report will contain only "
                "point metrics at each cumulative state; the load "
                "history is not analysed.  Enable at least one "
                "eta_path_* flag to characterise the path.",
                file=sys.stderr,
            )

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
                # Stage 0: base is origin → all enabled ray-cast
                # metrics contribute to the governing eta_max.
                etas_dict = self._compute_etas(cum_N, cum_Mx, cum_My)
                sr.update(etas_dict)
                for key in ("eta_norm", "eta_norm_beta", "eta_norm_ray", "eta_2D"):
                    if key in sr and sr[key] is not None:
                        all_etas.append(sr[key])
            else:
                # Stage k > 0: compute path-based η.
                sr["base"] = {
                    "N_kN": round(prev_N / 1e3, 2),
                    "Mx_kNm": round(prev_Mx / 1e6, 4),
                    "My_kNm": round(prev_My / 1e6, 4),
                }

                # eta_path_norm_ray (3D ray-cast in normalised space)
                if self.do_path:
                    ep = self.domain.eta_path_norm_ray(
                        prev_N, prev_Mx, prev_My,
                        cum_N, cum_Mx, cum_My)
                    sr["eta_path_norm_ray"] = round(ep, 4)
                    all_etas.append(ep)

                # eta_path_norm_beta (composite-ratio along segment)
                if self.do_path_norm_beta:
                    epb = self.domain.eta_path_norm_beta(
                        prev_N, prev_Mx, prev_My,
                        cum_N, cum_Mx, cum_My)
                    sr["eta_path_norm_beta"] = round(epb, 4)
                    all_etas.append(epb)

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

                # Also report the four point metrics of the
                # cumulative point.  All enabled metrics contribute
                # to the governing eta_max.
                cum_etas = self._compute_etas(cum_N, cum_Mx, cum_My)
                if "eta_norm" in cum_etas:
                    sr["eta_norm"] = cum_etas["eta_norm"]
                    all_etas.append(cum_etas["eta_norm"])
                if "eta_norm_beta" in cum_etas:
                    sr["eta_norm_beta"] = cum_etas["eta_norm_beta"]
                    all_etas.append(cum_etas["eta_norm_beta"])
                if "eta_norm_ray" in cum_etas:
                    sr["eta_norm_ray"] = cum_etas["eta_norm_ray"]
                    all_etas.append(cum_etas["eta_norm_ray"])
                if "eta_2D" in cum_etas and cum_etas["eta_2D"] is not None:
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

            # Collect numeric η for governing.  All enabled ray-cast
            # metrics contribute; the governing member is the one
            # with the worst (largest) eta across all enabled types.
            for key in ("eta_norm", "eta_norm_beta", "eta_norm_ray", "eta_2D"):
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