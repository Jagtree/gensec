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
3. :class:`~gensec.solver.check.VerificationEngine` — verifies load
   demands against the resistance domain
   (:class:`~gensec.solver.check.DomainChecker`, convex-hull
   ray-casting).

The prestress chain sits on top of it: the section state and the
staged domains (:mod:`~gensec.solver.section_state`), the transfer and
stressing drivers (:mod:`~gensec.solver.prestress_transfer`,
:mod:`~gensec.solver.posttension`), the construction timeline
(:mod:`~gensec.solver.timeline`), the time-dependent losses
(:mod:`~gensec.solver.losses`) and the serviceability stresses
(:mod:`~gensec.solver.sls`).  What that chain models and where it stops
is stated in :doc:`/theory/prestress_scope`.


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


Analysis engine
----------------

.. automodule:: gensec.solver.analysis
   :members:
   :show-inheritance:


Section state
-------------

.. automodule:: gensec.solver.section_state
   :members:
   :show-inheritance:

Prestress transfer
-------------------

.. automodule:: gensec.solver.prestress_transfer
   :members:
   :show-inheritance:

Post-Tension sequence solver
-----------------------------

.. automodule:: gensec.solver.posttension
   :members:
   :show-inheritance:

Construction timeline
---------------------

.. automodule:: gensec.solver.timeline
   :members:
   :show-inheritance:

Timeline and SLS drivers
------------------------

.. automodule:: gensec.solver.timeline_run
   :members:
   :show-inheritance:

Time-dependent losses (AAEM)
----------------------------

.. automodule:: gensec.solver.losses
   :members:
   :show-inheritance:

Prestress ledger
----------------

.. automodule:: gensec.solver.prestress_ledger
   :members:
   :show-inheritance:

SLS stress verification
-----------------------

.. automodule:: gensec.solver.sls
   :members:
   :show-inheritance:

Package-level exports
----------------------

.. automodule:: gensec.solver
   :members:
   :noindex:
   :show-inheritance:
