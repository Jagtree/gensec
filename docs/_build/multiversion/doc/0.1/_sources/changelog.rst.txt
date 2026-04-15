.. _changelog:

=========
Changelog
=========

This page tracks notable changes across GenSec releases.


v0.1.0 (current)
-----------------

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
