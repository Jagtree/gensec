.. _developer:

===============
Developer guide
===============

This page covers the project layout, testing practices, and
contribution guidelines.


Project layout
--------------

::

    gensec/
    ├── src/gensec/              # package source
    │   ├── materials/           # constitutive laws and property tables
    │   │   ├── base.py
    │   │   ├── concrete.py
    │   │   ├── steel.py
    │   │   ├── tabulated.py
    │   │   ├── ec2_properties.py
    │   │   ├── en10025_properties.py
    │   │   └── ec2_bridge.py
    │   ├── geometry/            # section definition and meshing
    │   │   ├── fiber.py
    │   │   ├── geometry.py
    │   │   ├── primitives.py
    │   │   └── section.py
    │   ├── solver/              # numerical core
    │   │   ├── integrator.py
    │   │   ├── capacity.py
    │   │   └── check.py
    │   ├── output/              # plotting, export, reporting
    │   │   ├── plots.py
    │   │   ├── export.py
    │   │   └── report.py
    │   ├── io_yaml.py           # YAML input loader
    │   ├── cli.py               # command-line interface
    │   └── __main__.py          # python -m gensec
    ├── tests/                   # test suite
    ├── docs/                    # Sphinx documentation
    ├── examples/                # YAML input files
    └── pyproject.toml


Running the test suite
-----------------------

.. code-block:: bash

   # With pytest (recommended)
   uv run python -m pytest tests/ -v

   # With unittest
   uv run python -m unittest discover -s tests -v

The suite contains **106 tests** organized in four files:

+-------------------------------+----------------------------------------------+
| File                          | Coverage                                     |
+===============================+==============================================+
| ``test_materials.py``         | Constitutive laws: pointwise stress           |
|                               | evaluation, strain limits, edge cases.       |
+-------------------------------+----------------------------------------------+
| ``test_infrastructure.py``    | YAML loading, section construction, export   |
|                               | round-trips, sign conventions, embedded bars.|
+-------------------------------+----------------------------------------------+
| ``test_solver_uniaxial.py``   | Uniaxial solver: analytical comparisons      |
|                               | (pure compression, tension, bending),        |
|                               | systematic grid test, N-M diagram symmetry.  |
+-------------------------------+----------------------------------------------+
| ``test_solver_biaxial.py``    | Biaxial solver: analytical rebar-only cases, |
|                               | Mx-My symmetry, 3-D surface generation,      |
|                               | combination loading, solver dispatch.        |
+-------------------------------+----------------------------------------------+


Writing tests
--------------

Each development phase must include a validation suite.  Follow these
principles:

- **Analytical reference cases**: whenever possible, compare against
  hand-calculated or textbook results, not just against GenSec's own
  output.  Typical cases: pure axial compression/tension (closed-form),
  symmetric sections (symmetry checks), rebar-only sections (steel
  stress × area).

- **Tolerances**: use relative tolerances of 1–2 % for integration-based
  results (fiber discretization introduces mesh-dependent error).
  Use tight tolerances (< 0.01 %) for material law evaluations.

- **Round-trip consistency**: for the inverse solver, verify that
  ``solve_equilibrium`` followed by ``compute_forces`` recovers the
  original targets within tolerance.

- **Edge cases**: zero curvature, zero axial force, single-fiber
  sections, sections with only rebars.


Documentation standards
------------------------

All code is documented in **English**, in **numpydoc style**, ready for
Sphinx with LaTeX math formulas.

- Every public class, method, and function must have a complete
  docstring with ``Parameters``, ``Returns``, and ``Raises`` sections
  as appropriate.
- Use raw docstrings (``r"""``) when LaTeX math is present.
- Mathematical formulas use the ``:math:`` role for inline expressions
  and ``.. math::`` directives for display equations.
- Cross-reference other classes and modules using Sphinx roles:
  ``:class:`~gensec.materials.Concrete```,
  ``:func:`~gensec.materials.concrete_from_ec2```, etc.


Adding a new material
----------------------

1. Create a new module in ``materials/`` that subclasses
   :class:`~gensec.materials.base.Material`.
2. Implement ``stress(eps)``, ``stress_array(eps)``, ``eps_min``,
   and ``eps_max``.  The ``stress_array`` method must accept arrays
   of **any shape** (1-D, 2-D, …).
3. **Recommended**: implement ``tangent(eps)`` and ``tangent_array(eps)``
   with the closed-form derivative of the constitutive law.  If omitted,
   the base class provides a finite-difference fallback, but the
   analytical form is faster and more accurate.
4. **Optional**: if the constitutive law is computationally intensive,
   consider adding a `Numba <https://numba.pydata.org>`_ JIT kernel
   guarded by a ``try: import numba`` block.  See ``concrete.py`` for
   the pattern.
5. Register the new ``type`` string in ``io_yaml.py`` so that YAML
   files can reference it.
6. Add pointwise tests in ``test_materials.py``, including tests for
   ``tangent_array`` consistency with finite differences.
7. Update the API docs in ``docs/api/gensec.materials.rst``.


Adding a new parametric shape
------------------------------

1. Write a factory function in
   :mod:`~gensec.geometry.primitives` that returns a
   :class:`shapely.geometry.Polygon`.
2. Register the ``shape`` name in ``io_yaml.py``.
3. Add a test and an example YAML file.
4. Update :doc:`/user_guide/yaml_reference` with the new shape
   parameters.
