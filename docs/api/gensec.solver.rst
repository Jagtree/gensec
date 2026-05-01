.. _api_solver:

=========================
:mod:`gensec.solver`
=========================

Fiber integration, equilibrium solving, interaction diagram generation,
and demand verification.

This is the computational core of GenSec.  The solver chain is:

1. :class:`~gensec.solver.integrator.FiberSolver` — evaluates the direct
   problem :math:`(\varepsilon_0, \chi_x, \chi_y) \to (N, M_x, M_y)`
   and solves the inverse problem via Newton–Raphson.
2. :class:`~gensec.solver.capacity.NMDiagram` — generates the
   :math:`N\text{-}M` diagram (uniaxial) or :math:`N\text{-}M_x\text{-}M_y`
   surface (biaxial) by scanning ultimate strain configurations.
3. :class:`~gensec.solver.check.DemandChecker` — verifies load demands
   against the resistance domain via convex-hull ray-casting.


Fiber integrator and equilibrium solver
----------------------------------------

.. automodule:: gensec.solver.integrator
   :members:
   :show-inheritance:


Resistance domain generator
-----------------------------

.. automodule:: gensec.solver.capacity
   :members:
   :show-inheritance:


Demand checker
---------------

.. automodule:: gensec.solver.check
   :members:
   :show-inheritance:


Package-level exports
----------------------

.. automodule:: gensec.solver
   :members:
   :noindex:
   :show-inheritance:
