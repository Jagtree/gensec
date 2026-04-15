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

GenSec solves the inverse problem with a **Newton–Raphson** iteration.
The Jacobian is computed by finite differences.  A **backtracking line
search** ensures global convergence even for strongly nonlinear
constitutive laws (e.g. softening concrete at ultimate).


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
