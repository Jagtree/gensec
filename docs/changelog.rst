.. _changelog:

=========
Changelog
=========

This page tracks notable changes across GenSec releases.

v0.3.0 (current)
-----------------

**Added**

- New module `gensec.geometry.properties` computing homogenized
  geometric properties on any polygon + rebar configuration:
  area, centroid, principal axes, extreme-fiber distances,
  elastic moduli (W), plastic moduli (Z via PNA bisection),
  central inertia ellipse, and kern.  Torsional constant `I_t`
  kept as placeholder for a future St.-Venant FEM solver.
- `GenericSection.ideal_gross_properties` lazy property for
  user-facing access.
- New plot/report functions `plot_section_properties`,
  `print_section_properties`, `write_section_report`
  (`gensec.output.geometry_plot`); legacy aliases
  `plot_ideal_gross_section` etc. preserved.
- Documentation: `docs/theory/ideal_gross_properties.rst` covering
  the EC2 homogenization convention; `docs/theory/
  integration_methods.rst` comparing polygonal vs fiber
  integration with convergence benchmarks on a disc and on a
  realistic RC cross-section.

**Tests**

- 10 new analytical tests in `tests/test_properties.py` covering
  mono-material degeneration, RC homogenization (symmetric and
  asymmetric), plastic modulus of rectangle / circle / I-section
  / rotated rectangle, and alternative-reference homogenization.

v0.2.0
-------

Performance overhaul and architecture refinements.

**Performance**

- **Batch integration** (:meth:`~gensec.solver.FiberSolver.integrate_batch`):
  evaluate thousands of strain configurations in a single vectorized
  NumPy call.  Eliminates Python-loop overhead in all capacity
  generators.
- **Mega-batch with chunking**: biaxial generators
  (``generate_biaxial``, ``generate_mx_my``) concatenate all curvature
  directions into a single flat array and integrate in large chunks,
  reducing per-call overhead by 10–70×.
- **Analytical tangent stiffness**
  (:meth:`~gensec.solver.FiberSolver.integrate_with_tangent`):
  compute internal forces and the 3×3 tangent matrix in one pass,
  halving the cost of each Newton-Raphson iteration.
- **Optional Numba JIT**: when ``numba`` is installed
  (``pip install gensec[fast]``), stress and tangent kernels for
  ``Concrete`` and ``Steel`` are JIT-compiled to native code
  (~2–3× speed-up on large fiber arrays).
- Measured end-to-end improvement: **15–40× faster** on typical
  biaxial analyses.

**Materials**

- ``tangent(eps)`` and ``tangent_array(eps)`` added to the abstract
  :class:`~gensec.materials.Material` interface.  Closed-form
  implementations for ``Concrete``, ``Steel``, and
  ``TabulatedMaterial``.  Finite-difference fallback in the base class
  for custom materials.
- ``stress_array`` now accepts arrays of **any shape** (1-D, 2-D, …)
  across all built-in materials.

**Geometry**

- ``RectSection`` is now a **factory function** returning a
  :class:`~gensec.geometry.GenericSection` directly, eliminating the
  former wrapper class with its 15 delegated properties.
- ``GenericSection`` accepts optional ``n_grid_x`` / ``n_grid_y``
  parameters for explicit grid control.
- ``GenericSection`` exposes ``dx`` and ``dy`` attributes.
- Isotropic grid by default: when ``n_fibers_x`` is omitted or
  set to 1, the grid cell size is derived from ``n_fibers_y``
  (approximately square cells).

**Verification**

- ``VerificationEngine`` auto-disables ``eta_2D`` / ``eta_path_2D``
  when the resistance domain is 2-D (no My data), preventing QHull
  errors on degenerate contours.
- ``_get_contour`` caches failures (degenerate contours at extreme N
  levels) as ``None`` to avoid expensive retries.

**Output flags**

- New YAML flags: ``generate_moment_curvature``,
  ``generate_polar_ductility``, ``generate_3d_moment_curvature``
  (all default ``true`` for backward compatibility).
- Setting all three to ``false`` skips the computationally expensive
  moment-curvature pipeline entirely.


v0.1.0
-------

Phase 2 complete: biaxial bending and generic sections.

- **Generic section** with arbitrary Shapely polygons and automatic
  fiber meshing (grid and triangular).
- **Parametric primitives**: rectangle, circle, annulus, T, inverted T,
  H, box, single-tee slab, double-tee slab.
- **Biaxial bending**: full :math:`(N, M_x, M_y)` resistance surface
  generation via curvature-angle scanning.
- :math:`M_x\text{-}M_y` **contour diagrams** at constant :math:`N`.
- **3-D resistance surface** visualization.
- **Convex-hull demand verification** in both 2-D and 3-D modes.
- **Combinations**: named groups of demand triples with envelope
  :math:`\eta_{\max}`.
- **EN 10025-2 structural steel** with thickness-dependent properties.
- **Multi-material bulk support** for composite sections.
- **Per-fiber post-processing**: strain/stress extraction and section
  state plots.
- **106 tests** covering materials, solvers, infrastructure, and biaxial
  analysis.


v0.0.1
------

Phase 0 + Phase 1: uniaxial bending for rectangular sections.

- Parabola-rectangle concrete (EC2 / NTC 2018).
- Elastic-plastic steel with optional hardening.
- Fiber integrator with Newton–Raphson inverse solver.
- :math:`N\text{-}M` interaction diagram via EC2 pivot method.
- YAML-driven input.
- CLI and Python API.
- EC2 Table 3.1 property computation (French National Annex).
- Six analytical validation test cases, all passing within 1 %.
