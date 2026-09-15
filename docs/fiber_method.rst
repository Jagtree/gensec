.. _fiber_method:

==================
The fiber method
==================

GenSec analyses cross-sections using the **fiber method**, a numerical
approach that discretizes the section into small elements (fibers) and
computes internal forces by integrating stresses over the fiber
assembly.

This page describes the mathematical formulation implemented in
:class:`~gensec.solver.FiberSolver`.


Hypotheses
----------

The fiber method rests on three fundamental assumptions:

1. **Plane sections remain plane** (Bernoulli–Navier hypothesis): the
   strain distribution over the section is linear.
2. **Perfect bond** between all constituents: each fiber experiences
   the strain dictated by its position in the strain plane.
3. **Uniaxial stress state**: each fiber carries only axial stress
   :math:`\sigma(\varepsilon)`, governed by its own constitutive law.
   Shear stress and confinement effects are neglected at the section level.


Strain plane
------------

Given three scalar parameters :math:`(\varepsilon_0,\,\chi_x,\,\chi_y)`,
the strain at fiber :math:`i` located at :math:`(x_i,\,y_i)` is:

.. math::

   \varepsilon_i = \varepsilon_0
       + \chi_x \, (y_i - y_{\text{ref}})
       + \chi_y \, (x_i - x_{\text{ref}})

where :math:`(x_{\text{ref}},\,y_{\text{ref}})` is the reference point
(the geometric centroid by default).

- :math:`\varepsilon_0` — axial strain at the reference point.
- :math:`\chi_x` — curvature about the :math:`x`-axis (bending in the
  :math:`y`-direction).
- :math:`\chi_y` — curvature about the :math:`y`-axis (bending in the
  :math:`x`-direction).

Uniaxial bending is the special case :math:`\chi_y = 0`.


Internal forces
----------------

Once every fiber stress :math:`\sigma_i` is computed through the
material constitutive law, the resultant internal forces are obtained
by summation over all :math:`n_f` bulk fibers and :math:`n_r` rebar
fibers:

.. math::

   N   &= \sum_{i=1}^{n_f} \sigma_i^{(b)} \, A_i^{(b)}
        + \sum_{j=1}^{n_r} F_{\text{net},j} \\[6pt]
   M_x &= \sum_{i=1}^{n_f} \sigma_i^{(b)} \, A_i^{(b)} \,
           (y_i - y_{\text{ref}})
        + \sum_{j=1}^{n_r} F_{\text{net},j} \,
           (y_j - y_{\text{ref}}) \\[6pt]
   M_y &= \sum_{i=1}^{n_f} \sigma_i^{(b)} \, A_i^{(b)} \,
           (x_i - x_{\text{ref}})
        + \sum_{j=1}^{n_r} F_{\text{net},j} \,
           (x_j - x_{\text{ref}})

where :math:`F_{\text{net},j}` accounts for embedded bar correction
(see below).


Embedded bar correction
------------------------

For rebars lying *inside* the bulk material, the concrete area that the
bar displaces is already counted in the bulk fiber sum.  To avoid
double-counting, the net rebar contribution is:

.. math::

   F_{\text{net},j} =
       \bigl[\sigma_{\text{rebar},j}(\varepsilon_j)
       - \sigma_{\text{bulk}}(\varepsilon_j)\bigr] \, A_{s,j}

For external elements (``embedded = false``):

.. math::

   F_{\text{net},j} = \sigma_{\text{rebar},j}(\varepsilon_j) \, A_{s,j}


Direct and inverse problems
-----------------------------

**Direct problem** — given :math:`(\varepsilon_0,\,\chi_x,\,\chi_y)`,
compute :math:`(N,\,M_x,\,M_y)`.  This is a straightforward
evaluation of the equations above.

**Inverse problem** — given a target :math:`(N_{\text{target}},\,
M_{x,\text{target}})` (uniaxial) or :math:`(N_{\text{target}},\,
M_{x,\text{target}},\,M_{y,\text{target}})` (biaxial), find the strain
plane parameters that produce the specified internal forces.

GenSec solves the inverse problem with a **Newton–Raphson** iteration
using the **analytical tangent stiffness matrix**.  A **backtracking
line search** ensures global convergence even for strongly nonlinear
constitutive laws (e.g. softening concrete at ultimate).


Analytical tangent stiffness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 3×3 tangent stiffness matrix relates infinitesimal changes in the
strain-plane parameters to changes in internal forces:

.. math::

   \mathbf{K} =
   \frac{\partial (N,\,M_x,\,M_y)}
        {\partial (\varepsilon_0,\,\chi_x,\,\chi_y)}
   = \sum_i E_{t,i} \, A_i \;
     \boldsymbol{\varphi}_i \, \boldsymbol{\varphi}_i^T

where the shape-function vector for fiber :math:`i` is

.. math::

   \boldsymbol{\varphi}_i =
   \bigl[1,\; (y_i - y_{\text{ref}}),\;
         -(x_i - x_{\text{ref}})\bigr]^T

and :math:`E_{t,i} = d\sigma_i / d\varepsilon_i` is the tangent
modulus at the current strain, computed by the material's
``tangent_array`` method.

This analytical approach replaces the former finite-difference
Jacobian (which required 3–4 extra ``integrate()`` calls per
Newton iteration), halving the cost of each iteration.


Batch integration
~~~~~~~~~~~~~~~~~~

The :meth:`~gensec.solver.FiberSolver.integrate_batch` method
evaluates internal forces for many strain configurations at once.
All inputs are 1-D arrays of length :math:`n`; the computation
builds a 2-D strain matrix of shape :math:`(n, n_{\text{fibers}})`,
passes it through the constitutive law in a single vectorized call,
and sums forces and moments along the fiber axis.

This is used by the capacity generators
(:meth:`~gensec.solver.NMDiagram.generate`,
:meth:`~gensec.solver.NMDiagram.generate_biaxial`,
:meth:`~gensec.solver.NMDiagram.generate_mx_my`) to eliminate the
Python-loop overhead of calling ``integrate()`` once per
configuration.


Multi-material bulk
--------------------

When the section contains multiple bulk material zones (e.g. a
composite section with regions of different concrete grades), the
integrator groups fibers by material index and evaluates each
constitutive law on its own subset.  The summation then runs over all
groups:

.. math::

   N = \sum_{k=1}^{n_{\text{mat}}} \sum_{i \in G_k}
       \sigma^{(k)}(\varepsilon_i) \, A_i
     + \sum_{j=1}^{n_r} F_{\text{net},j}

This is handled transparently by the solver when the section exposes
a ``mat_indices`` array.

Accuracy and comparison with polygonal integration
---------------------------------------------------

The fiber method is GenSec's canonical tool for integrating
non-polynomial integrands — i.e. anything involving the
constitutive law :math:`\sigma(\varepsilon)`.  For purely
geometric quantities (area, moments of inertia, section moduli)
GenSec uses instead a closed-form polygonal integration via
Green's theorem.  A detailed comparison of the two methods —
their accuracy rates, their cost, and the specific
implementation choices in GenSec (Shapely-based cell clipping,
midpoint vs higher-order quadrature) — is given in
:ref:`integration_methods`.
