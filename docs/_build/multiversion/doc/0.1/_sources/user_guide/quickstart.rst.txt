.. _quickstart:

===========
Quick start
===========

This page walks through the two main ways to use GenSec: the
**command-line interface** (YAML-driven) and the **Python API**.


Command-line workflow
---------------------

1. Write a YAML input file describing materials, section geometry, and
   load demands (see :doc:`yaml_reference` for the full specification).

2. Run the analysis:

   .. code-block:: bash

      uv run gensec examples/example_input.yaml --output-dir results

   Or equivalently:

   .. code-block:: bash

      uv run python -m gensec examples/example_input.yaml --output-dir results

3. GenSec produces the following outputs in ``results/``:

   +-------------------------------+------------------------------------------+
   | File                          | Content                                  |
   +===============================+==========================================+
   | ``nm_diagram.png``            | N-M interaction diagram with demands     |
   +-------------------------------+------------------------------------------+
   | ``nm_domain.csv`` / ``.json`` | Point cloud of the resistance domain     |
   +-------------------------------+------------------------------------------+
   | ``demand_summary.csv``        | Verification table with :math:`\eta`     |
   +-------------------------------+------------------------------------------+
   | ``fibers_<n>.csv``            | Per-fiber strain/stress for demand *n*   |
   +-------------------------------+------------------------------------------+
   | ``stress_<n>.png``            | Strain and stress profile plots          |
   +-------------------------------+------------------------------------------+
   | ``mx_my_diagram.png``         | :math:`M_x\text{-}M_y` contour (if      |
   |                               | ``generate_mx_my: true``)                |
   +-------------------------------+------------------------------------------+
   | ``surface_3d.png``            | 3-D resistance surface (if               |
   |                               | ``generate_3d_surface: true``)           |
   +-------------------------------+------------------------------------------+


Python API workflow
-------------------

Minimal example: rectangular 300×600 mm RC column, 3Φ20 top + 3Φ20 bottom.

.. code-block:: python

   import numpy as np
   from gensec.materials import Concrete, Steel
   from gensec.geometry import RebarLayer, RectSection
   from gensec.solver import FiberSolver, NMDiagram, DemandChecker

   # --- Materials ---
   concrete = Concrete(fck=25.0, gamma_c=1.5, alpha_cc=0.85)
   steel = Steel(fyk=450.0, gamma_s=1.15)

   # --- Section geometry ---
   A20 = np.pi / 4 * 20**2                     # area of one Φ20 bar
   rebars = [
       RebarLayer(y=40,  As=3 * A20, material=steel, n_bars=3, diameter=20),
       RebarLayer(y=560, As=3 * A20, material=steel, n_bars=3, diameter=20),
   ]
   section = RectSection(B=300, H=600, bulk_material=concrete, rebars=rebars)

   # --- Solver ---
   solver = FiberSolver(section)

   # Generate N-M interaction diagram
   nm = NMDiagram(solver).generate(n_points=400)

   # Verify a single demand
   checker = DemandChecker(nm)
   eta = checker.utilization_ratio(N=-1500e3, Mx_or_M=200e6)
   print(f"Utilization ratio η = {eta:.3f}")     # η < 1.0 → verified

   # Per-fiber stress state at a specific demand
   sol = solver.solve_equilibrium(N_target=-1500e3, Mx_target=200e6)
   results = solver.get_fiber_results(sol["eps0"], sol["chi_x"])


Using EC2 properties from :math:`f_{ck}`
-----------------------------------------

GenSec can compute all EN 1992-1-1 Table 3.1 parameters automatically:

.. code-block:: python

   from gensec.materials import concrete_from_ec2, concrete_from_class

   # From fck (French National Annex)
   c30 = concrete_from_ec2(fck=30, ls='F', loadtype='slow', TypeConc='R')
   print(c30.fcd)       # 17.0 MPa
   print(c30.eps_cu2)   # -0.0035

   # From class name
   c50 = concrete_from_class('C50/60', ls='F')

   # Access the full EC2 property object
   print(c30.ec2.fcm)   # 38.0 MPa
   print(c30.ec2.fctm)  # 2.9 MPa
   print(c30.ec2.ecm)   # 32837 MPa


Structural steel (EN 10025-2)
------------------------------

.. code-block:: python

   from gensec.materials import steel_from_en10025

   s355 = steel_from_en10025('S355', t=20)   # 20 mm plate
   # s355.fyk = 345 (thickness-dependent)
   # s355.k_hardening = 470/345 (fu/fy ratio)


Biaxial bending
----------------

For biaxial analysis, assign explicit ``x`` coordinates to rebars and use a
2-D fiber grid (``n_fibers_x > 1``):

.. code-block:: python

   rebars = [
       RebarLayer(y=40,  x=40,  As=A20, material=steel),
       RebarLayer(y=40,  x=260, As=A20, material=steel),
       RebarLayer(y=560, x=40,  As=A20, material=steel),
       RebarLayer(y=560, x=260, As=A20, material=steel),
   ]
   section = RectSection(
       B=300, H=600, bulk_material=concrete,
       rebars=rebars, n_fibers_y=60, n_fibers_x=30,
   )

   solver = FiberSolver(section)
   nm_3d = NMDiagram(solver).generate_biaxial(
       n_angles=36, n_points_per_angle=200,
   )

   checker = DemandChecker(nm_3d)
   eta = checker.utilization_ratio(N=-1000e3, Mx_or_M=100e6, My=50e6)


Mixed materials (e.g. RC + CFRP)
---------------------------------

Use :class:`~gensec.materials.TabulatedMaterial` for any arbitrary
:math:`\sigma\text{-}\varepsilon` curve:

.. code-block:: python

   from gensec.materials import TabulatedMaterial

   cfrp = TabulatedMaterial(
       strains=[0.0, 0.017],
       stresses=[0.0, 2800.0],
       name="CFRP",
   )

   rebars = [
       RebarLayer(y=40,  As=942, material=steel, embedded=True),
       RebarLayer(y=560, As=942, material=steel, embedded=True),
       RebarLayer(y=5,   As=150, material=cfrp,  embedded=False),  # external strip
   ]

Set ``embedded=False`` for elements external to the bulk material so that
GenSec does not subtract the displaced concrete (see
:doc:`sign_conventions`).
