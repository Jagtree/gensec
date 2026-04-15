.. _demand_verification:

=====================
Demand verification
=====================

Once the resistance domain is computed, GenSec verifies load demands
by measuring how close each demand point is to the domain boundary.
The verification engine supports four utilization ratio types,
staged combinations, and envelopes.

.. contents:: On this page
   :local:
   :depth: 2


Utilization ratios
------------------

All four :math:`\eta` types share the same geometric primitive — a
ray from base :math:`\mathbf{B}` through target :math:`\mathbf{T}`
intersecting a boundary at :math:`\mathbf{R}`:

.. math::

   \eta =
       \frac{|\mathbf{T} - \mathbf{B}|}
            {|\mathbf{R} - \mathbf{B}|}

Interpretation:

+--------------------+---------------------------------------------+
| :math:`\eta < 1`  | Demand inside the domain — **verified**.     |
+--------------------+---------------------------------------------+
| :math:`\eta = 1`  | Demand exactly on the boundary.              |
+--------------------+---------------------------------------------+
| :math:`\eta > 1`  | Demand outside the domain — **not verified**.|
+--------------------+---------------------------------------------+


:math:`\eta_{\text{3D}}` — 3D hull ray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray from the origin :math:`(0, 0, 0)` through the demand
:math:`(N, M_x, M_y)` to the 3D ConvexHull boundary.

**Use case**: all force components scale proportionally (overall
load factor).  Always fast — uses the 3D hull which is generated
for any biaxial section.

Enabled by ``eta_3D: true`` (default).


:math:`\eta_{\text{2D}}` — Mx-My plane ray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the demand's axial force :math:`N`, the Mx-My interaction
contour is generated.  A ray from :math:`(0, 0)` to
:math:`(M_x, M_y)` is cast in this 2D plane.

This is equivalent to the :math:`\rho_M` of VCASLU: it measures
how close the moment vector is to the flexural boundary at the
given axial force level.

**Use case**: fixed axial force, variable bending direction and
magnitude (typical static design check).

Enabled by ``eta_2D: true`` (default ``false``).  Requires
generating Mx-My contours on demand — slower than
:math:`\eta_{\text{3D}}`.


:math:`\eta_{\text{path}}` — staged 3D ray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For staged combinations, ray from the cumulative point of the
previous stage :math:`\mathbf{S}_{k-1}` through the current
cumulative :math:`\mathbf{S}_k` to the 3D hull boundary:

.. math::

   \eta_{\text{path},k}
   = \frac{|\mathbf{S}_k - \mathbf{S}_{k-1}|}
          {|\mathbf{R} - \mathbf{S}_{k-1}|}

**Use case**: measure the margin of a **load increment** relative
to a known base state.  The base is assumed certain; only the
increment might exhaust capacity.  Typical for seismic design
(gravity base + seismic increment) and staged construction
(prestress → gravity → variable).

Stage 0 has no predecessor — its :math:`\eta_{\text{path}}` falls
back to :math:`\eta_{\text{3D}}`.

Enabled by ``eta_path: true`` (default).


:math:`\eta_{\text{path,2D}}` — staged 2D ray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same concept as :math:`\eta_{\text{path}}`, but in the
:math:`M_x`-:math:`M_y` plane at the target stage's axial force.

**Applicability condition**: the axial force jump between stages
must be small relative to the domain range:

.. math::

   \frac{|N_{S_k} - N_{S_{k-1}}|}{N_{Rd,\max} - N_{Rd,\min}}
   < \delta_N

where :math:`\delta_N` is set by ``delta_N_tol`` (default 0.03,
i.e. 3%).  When the condition is not satisfied,
:math:`\eta_{\text{path,2D}}` is reported as ``null`` and a warning
is emitted.

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

It is verified with :math:`\eta_{\text{3D}}` and
:math:`\eta_{\text{2D}}` (if enabled).

A **staged combination** accumulates stages sequentially:

.. math::

   \mathbf{S}_k = \sum_{j=0}^{k} \Delta\mathbf{S}_j,
   \qquad
   \Delta\mathbf{S}_j = \sum_i f_{j,i} \, \mathbf{d}_{j,i}

Each stage reports its own :math:`\eta` values.  The
``eta_governing`` of the combination is the maximum across all
stages and all enabled :math:`\eta` types.


Envelopes
---------

An **envelope** collects demands and/or combinations and reports
the worst-case utilization:

.. math::

   \eta_{\text{envelope}} = \max_{\text{members}} \eta

Members can be:

- References to named demands or combinations (``ref``).
- Inline demands with direct ``N_kN``, ``Mx_kNm``, ``My_kNm``.
- Optionally scaled with a ``factor`` on the resultant.


Convex hull
-----------

The 3D domain boundary is represented as a
:class:`scipy.spatial.ConvexHull` in :math:`(N, M_x, M_y)` space.
This is exact for convex domains (the typical case for RC sections
at ULS under the parabola-rectangle law).

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
