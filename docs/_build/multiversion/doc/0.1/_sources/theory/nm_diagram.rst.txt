.. _nm_diagram:

=======================================
N-M diagram and resistance surface
=======================================

The resistance domain of a cross-section is the set of all internal
force states :math:`(N, M_x, M_y)` that the section can carry at the
ultimate limit state.  GenSec computes this domain numerically by
scanning strain configurations that satisfy the ultimate strain limits
of the constituent materials.


Uniaxial N-M diagram
---------------------

For uniaxial bending (:math:`\chi_y = 0`), the resistance domain
reduces to a 2-D curve in the :math:`(N, M_x)` plane.

GenSec generates the :math:`N\text{-}M` diagram by:

1. Identifying the **pivot points** — the extreme fibers of the section
   and the rebar positions.  At ultimate, at least one material is at
   its strain limit.

2. For each pivot, scanning the curvature :math:`\chi_x` from a
   large negative value to a large positive value (controlling the
   neutral axis depth).  At each step, the axial strain
   :math:`\varepsilon_0` is computed to satisfy the ultimate strain
   constraint at the pivot fiber.

3. Evaluating the **direct problem**
   :math:`(\varepsilon_0, \chi_x) \to (N, M_x)` at each configuration.

4. Collecting all :math:`(N, M_x)` points to form the domain boundary.

.. important::

   The curvature scan must cover **both positive and negative
   directions** to produce a symmetric, complete diagram.  Scanning only
   one direction is a known source of incorrect asymmetric results.

The number of points along the boundary is controlled by ``n_points``
(default 400).


Biaxial resistance surface
----------------------------

For biaxial bending, the resistance domain is a 3-D surface in
:math:`(N, M_x, M_y)` space.  GenSec generates it by:

1. Discretizing the curvature direction angle :math:`\theta` in the
   :math:`(\chi_x, \chi_y)` plane into ``n_angles`` equally-spaced
   values from :math:`0` to :math:`2\pi`.

2. For each angle :math:`\theta`, performing the same pivot-based scan
   as in the uniaxial case, but with:

   .. math::

      \chi_x = \chi \cos\theta, \qquad
      \chi_y = \chi \sin\theta

   where :math:`\chi` is the curvature magnitude being scanned.

3. Collecting the resulting :math:`(N, M_x, M_y)` point cloud.

The surface is represented as a point cloud.  For demand verification,
it is wrapped in a :class:`scipy.spatial.ConvexHull` (see
:doc:`demand_verification`).


:math:`M_x\text{-}M_y` contour at constant :math:`N`
------------------------------------------------------

A useful 2-D slice of the 3-D surface: at a specified axial force
:math:`N_{\text{fixed}}`, GenSec extracts the
:math:`(M_x, M_y)` interaction contour.  This is obtained by
intersecting the 3-D convex hull with the plane :math:`N = N_{\text{fixed}}`
and projecting onto the :math:`(M_x, M_y)` plane.

This is particularly useful for verifying biaxial bending demands
at a known axial load level (e.g. seismic columns).


Convergence and resolution
---------------------------

The accuracy of the domain depends on:

- **Fiber mesh density**: controls the accuracy of the direct problem.
  A finer mesh gives better stress integration but increases
  computation time.  Typical values: ``mesh_size = 10–20 mm`` for
  parametric sections, ``n_fibers_y = 60–120`` for rectangular sections.

- **Number of scan points** (``n_points``): controls the density of
  the domain boundary.  400 points are typically sufficient for smooth
  :math:`N\text{-}M` curves.

- **Number of curvature angles** (``n_angles``): controls the angular
  resolution of the biaxial surface.  36 angles give a reasonable
  approximation; 72 or 144 for publication-quality surfaces.
