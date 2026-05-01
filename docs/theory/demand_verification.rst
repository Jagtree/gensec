=====================
Demand verification
=====================

Once the resistance domain is computed, GenSec verifies load demands
by measuring how close each demand point is to the domain boundary.
The verification engine supports seven utilization ratio types,
staged combinations, and envelopes.


Utilization ratios
------------------

GenSec computes up to seven utilization metrics, organised in two
geometric families plus path-aware variants for staged loading.
Each metric answers a *different* geometric question; they are
**complementary, not redundant**, and no strict ordering between
them holds in general.

Interpretation (uniform across all metrics):

+---------------------+----------------------------------------------+
| :math:`\eta < 1`    | Demand inside the domain — **verified**.     |
+---------------------+----------------------------------------------+
| :math:`\eta = 1`    | Demand exactly on the boundary.              |
+---------------------+----------------------------------------------+
| :math:`\eta > 1`    | Demand outside the domain — **not verified**.|
+---------------------+----------------------------------------------+

Pass/fail (``verified``) is decided by the worst of the enabled
metrics.


Anisotropy-corrected normalised space
-------------------------------------

The 3-D metrics operate in a normalised coordinate system designed
to make euclidean geometry physically meaningful for *any* section
shape, not just doubly symmetric ones.

Let :math:`\Delta N = N_{R,\max} - N_{R,\min}`,
:math:`\Delta M_x = M_{x,R,\max} - M_{x,R,\min}` and
:math:`\Delta M_y = M_{y,R,\max} - M_{y,R,\min}` be the bounding-box
extents of the resistance domain in raw :math:`(N, M_x, M_y)`
coordinates.  Define

.. math::

    u_x = \frac{\Delta M_x}{\Delta N},
    \qquad
    v_y = \frac{\Delta M_x}{\Delta M_y}.

The normalising map is

.. math::

    (N, M_x, M_y) \;\to\;
    (N \cdot u_x,\; M_x,\; M_y \cdot v_y).

In the resulting space, the bounding box of the resistance domain
has the same numeric extent on every axis (each side equals
:math:`\Delta M_x`).  This is what we mean by *anisotropy-corrected*:
the normalisation is per-axis, not a single global factor.

The two consequences are:

- **Scale-invariance**.  Changing the unit of :math:`N` (kN vs N)
  rescales :math:`u_x` accordingly, and the metric value is
  preserved.
- **Physical isotropy of distance**.  For sections with strong
  geometric anisotropy (e.g. walls with :math:`H/B \approx 6`,
  giving :math:`\Delta M_x / \Delta M_y \approx 36`), euclidean
  distance in raw :math:`M_x`-:math:`M_y` would weight :math:`M_x`
  thirty-six times more than :math:`M_y`.  In normalised space the
  two are equally weighted.

The convex hull of the resistance domain is built once, in
normalised coordinates, at construction of
:class:`gensec.solver.DomainChecker`.


3-D family — point metrics
---------------------------

:math:`\eta_{\text{norm}}` (alpha) — linear distance to boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Map the demand to normalised coordinates
:math:`\mathbf{S}_u = (N\cdot u_x, M_x, M_y\cdot v_y)`.  Compute the
signed distance from :math:`\mathbf{S}_u` to each face of the
normalised hull, take the absolute value of the minimum, call it
:math:`d_{\min}`.  Let :math:`D_{\max}` be the **Chebyshev radius**
of the normalised domain (the radius of the largest sphere
inscribed in the hull, computed at construction by linear
programming).  Then

.. math::

    \eta_{\text{norm}}
      = \begin{cases}
          1 - \dfrac{d_{\min}}{D_{\max}} & \text{(interior)} \\[2pt]
          1 + \dfrac{d_{\min}}{D_{\max}} & \text{(exterior).}
        \end{cases}

This is a **true geometric distance**: linear and monotone in
:math:`d_{\min}`.  Reading:

- :math:`\eta_{\text{norm}} = 0` at the Chebyshev centre (the
  geometrically deepest point of the domain).
- :math:`\eta_{\text{norm}} = 1` exactly on the boundary.
- :math:`\eta_{\text{norm}} > 1` outside, with the excess
  proportional to the penetration depth.

Geometrically, :math:`\eta_{\text{norm}}` answers:
**"what fraction of the available reserve has the demand consumed,
in the worst direction?"**

The recommended principal 3-D metric.  Enabled by
``eta_norm: true`` (default).


:math:`\eta_{\text{norm,\beta}}` — composite-ratio metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the same normalised space, compute :math:`F_{SU} = |\mathbf{S}_u|`
(the demand norm) and :math:`d_{\min}` (the face distance).  Then

.. math::

    \eta_{\text{norm,\beta}}
      = \begin{cases}
          \dfrac{F_{SU}}{F_{SU} + d_{\min}} & \text{(interior)} \\[2pt]
          \dfrac{F_{SU}}{F_{SU} - d_{\min}} & \text{(exterior).}
        \end{cases}

This is **not** a distance.  The numerator and denominator both
contain :math:`F_{SU}`, so the value depends on the demand's
position relative to the *coordinate origin*, not just its
distance to the boundary.  For a fixed :math:`d_{\min}`, demands
far from the origin produce larger values than demands close to
the origin.

The semantic reading is **"sensitivity to perturbation in
proportion to the demand magnitude"**: how big is a small absolute
perturbation of the demand, measured against the demand's own
size?  A demand that is large and close to the boundary scores
high; a demand that is small but equally close to the boundary
scores lower, because the perturbation is large relative to the
demand.

This formulation is also useful for **cross-software validation**:
equivalent expressions are reported by some commercial RC software
(under various names), so the metric value can be compared
directly across implementations of the same domain.

Enabled by ``eta_norm_beta: true`` (default).


:math:`\eta_{\text{norm,ray}}` — radial-growth metric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cast a ray from the origin to :math:`\mathbf{S}_u` in normalised
space, and compute

.. math::

    \eta_{\text{norm,ray}}
      = \frac{|\mathbf{S}_u|}{|\mathbf{R}_u|},

where :math:`\mathbf{R}_u` is the first hit of the ray with the
boundary.  Geometrically: **"if I scale all three force components
proportionally, when does the demand exit the domain?"**

Linear in demand magnitude along any fixed ray.  Useful for
proportional load-amplification analyses.

Off by default (``eta_norm_ray: false``); opt-in for specific
analyses.


2-D family — point metric
--------------------------

:math:`\eta_{\text{2D}}` — flexural ray at fixed N
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the demand's axial force :math:`N`, the Mx-My interaction
contour is generated.  A ray from the origin :math:`(0, 0)` to
:math:`(M_x, M_y)` is cast in this 2D plane.

This is equivalent to the :math:`\rho_M` of VCASLU: it measures
how close the moment vector is to the flexural boundary at the
given axial force level.

**Use case**: fixed axial force, variable bending direction and
magnitude (typical static design check).

Enabled by ``eta_2D: true`` (default ``false``).  Requires
generating Mx-My contours on demand — slower than the 3-D metrics.


Path metrics — staged combinations
----------------------------------

For staged loading sequences (gravity + seismic, prestress +
service, construction sequences), the *path* between consecutive
states matters in addition to the cumulative state.  Three
path-aware metrics are available, all defined on the segment
:math:`B \to T` from the previous-stage cumulative demand
:math:`B = \mathbf{S}_{k-1}` to the current-stage cumulative demand
:math:`T = \mathbf{S}_k`.


:math:`\eta_{\text{path}}` — staged 3-D ray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cast a ray from :math:`B` to :math:`T` in anisotropy-corrected
normalised space:

.. math::

    \eta_{\text{path}}
      = \frac{|\mathbf{T}_u - \mathbf{B}_u|}
             {|\mathbf{R}_u - \mathbf{B}_u|},

where :math:`\mathbf{R}_u` is where the ray exits the boundary.
Stored under the key ``eta_path_norm_ray`` in per-stage results.

**Use case**: directional consumption analysis along the actual
load path.  The ray direction is the physical direction of the
stage increment, not a hypothetical radial growth from the origin.

Stage 0 has no predecessor — the metric is undefined at stage 0
and only point metrics are reported at the cumulative state.

Enabled by ``eta_path_norm_ray: true`` (default ``false``).


:math:`\eta_{\text{path,norm,\beta}}` — staged composite ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Composite-ratio analogue of :math:`\eta_{\text{norm,\beta}}` along
the stage segment.  Let :math:`L = |\mathbf{T}_u - \mathbf{B}_u|`
be the magnitude of the stage increment in normalised space, and
let :math:`d_{\text{seg}}` be the **minimum signed distance** of
the segment :math:`B_u \to T_u` from the boundary (the most
critical point along the entire path).  Then

.. math::

    \eta_{\text{path,norm,\beta}}
      = \begin{cases}
          \dfrac{L}{L + d_{\text{seg}}} & \text{(segment interior)} \\[2pt]
          \dfrac{L}{L - |d_{\text{seg}}|} & \text{(segment crosses bdy).}
        \end{cases}

Because the signed distance is affine along a segment in a convex
polyhedron, :math:`d_{\text{seg}}` is the minimum over both
endpoints (and over all faces) of the inside-distance.  Cost is
:math:`O(n_{\text{faces}})`.

**Use case**: seismic verification with staged gravity.  Imagine
gravity is applied in stage 0 and the seismic increment in stage 1.

- Small seismic increment, gravity already near the boundary:
  :math:`L` small, :math:`d_{\text{seg}}` small, metric close to 1
  — correctly flags that the structure is critical regardless of
  the seismic action's size.
- Large seismic increment, structure far from the boundary:
  :math:`L` large, :math:`d_{\text{seg}}` large, metric moderate.
- Large seismic increment that approaches the boundary: both
  contributions push the metric toward 1.

Distinguished from :math:`\eta_{\text{path}}` (the ray) because the
ray sees only the *target* clearance and the *direction* of
growth, while :math:`\eta_{\text{path,norm,\beta}}` sees the
worst-case clearance over the *entire path*, weighted by the path
size.

Enabled by ``eta_path_norm_beta: true`` (default ``false``).


:math:`\eta_{\text{path,2D}}` — staged 2-D ray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same concept as :math:`\eta_{\text{path}}`, but in the
:math:`M_x`-:math:`M_y` plane at the target stage's axial force.

**Applicability condition**: the axial force jump between stages
must be small relative to the domain range:

.. math::

   \frac{|N_{\mathbf{S}_k} - N_{\mathbf{S}_{k-1}}|}
        {N_{R,\max} - N_{R,\min}} < \delta_N

where :math:`\delta_N` is set by ``delta_N_tol`` (default 0.03,
i.e. 3%).  When the condition is not satisfied,
:math:`\eta_{\text{path,2D}}` is reported as ``null`` and a
warning is emitted.

**Rationale**: if :math:`N` changes significantly between stages,
the base and target sit on different Mx-My contours, and a 2D ray
in a single contour has no physical meaning.

Enabled by ``eta_path_2D: true`` (default ``false``).


Combinations
------------

A **simple combination** produces a single resultant from factored
demands:

.. math::

   \mathbf{S} = \sum_i f_i \, \mathbf{d}_i

It is verified with all enabled point metrics.

A **staged combination** accumulates stages sequentially:

.. math::

   \mathbf{S}_k = \sum_{j=0}^{k} \Delta\mathbf{S}_j,
   \qquad
   \Delta\mathbf{S}_j = \sum_i f_{j,i} \, \mathbf{d}_{j,i}.

Each stage reports both the point metrics evaluated at the
cumulative state :math:`\mathbf{S}_k` and (for :math:`k > 0`) the
path metrics evaluated on the segment :math:`\mathbf{S}_{k-1} \to
\mathbf{S}_k`.  The ``eta_governing`` of the combination is the
maximum across all stages and all enabled metrics.

If a staged combination is processed without any
``eta_path_*`` flag enabled, GenSec emits an informational warning:
without a path-aware metric the report would only contain point
metrics at each cumulative state and would not characterise the
load history.


Envelopes
---------

An **envelope** collects demands and/or combinations and reports
the worst-case utilization:

.. math::

   \eta_{\text{envelope}} = \max_{\text{members}} \eta

where the maximum runs over both members and enabled metrics.
Members can be:

- References to named demands or combinations (``ref``).
- Inline demands with direct ``N_kN``, ``Mx_kNm``, ``My_kNm``.
- Optionally scaled with a ``factor`` on the resultant.


Convex hull
-----------

The 3D domain boundary is represented as a
:class:`scipy.spatial.ConvexHull` in **normalised**
:math:`(N \cdot u_x, M_x, M_y \cdot v_y)` space.  This is exact
for convex domains (the typical case for RC sections at ULS under
the parabola-rectangle law).  The Chebyshev radius
:math:`D_{\max}` is computed at construction by solving the linear
program

.. math::

    \max_{x, r} r \quad \text{s.t.}
    \quad \mathbf{a}_i^\top x + b_i + r \le 0 \;\;\forall i,
    \quad r \ge 0,

where :math:`\mathbf{a}_i, b_i` are the (unit-length) face
equations of the hull.  This is solved once per
:class:`DomainChecker` and cached.

For 2D Mx-My contours, the domain at a given *N* is obtained by
slicing the 3D hull and wrapping the result in a 2D ConvexHull.
Contours are cached by the :class:`VerificationEngine` to avoid
redundant generation.


Output structure
----------------

The verification engine produces structured JSON results:

- ``demand_summary.json``: per-demand with all enabled :math:`\eta`.
- ``combination_summary.json``: per-combination with staged detail.
- ``envelope_summary.json``: per-envelope with governing member.
- ``verification_summary.json``: unified export of all three.

Each :math:`\eta` field is present in the output only if the
corresponding flag is enabled in the ``output`` block.  The
``verified`` flag is ``true`` if all enabled :math:`\eta \le 1`.


Per-fiber post-processing
--------------------------

For any demand point, GenSec solves the inverse problem (find the
strain plane that equilibrates the demand) and extracts the
stress/strain state at every fiber and rebar:

- :meth:`FiberSolver.solve_equilibrium` — inverse solver.
- :meth:`FiberSolver.get_fiber_results` — stress/strain extraction.

Results are exported to CSV and plotted on the section geometry.
