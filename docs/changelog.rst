.. _changelog:

=========
Changelog
=========

This page tracks notable changes across GenSec releases.
For complete patch-level detail see the ``CHANGELOG_v0_X_Y.md`` files
in the repository root.

.. note::

   **Version scheme.**  GenSec follows `Semantic Versioning`_.
   The working tree version is derived automatically from the latest Git
   tag by ``setuptools_scm``.

.. _Semantic Versioning: https://semver.org


v0.3.3
------

Bug fixes, performance improvements, new lightweight analysis pipeline,
API hardening, and tiered verification reporting.

**Bug fixes**

- **Moment–curvature key-point detection** (:meth:`~gensec.solver.NMDiagram.generate_moment_curvature`,
  :meth:`~gensec.solver.NMDiagram._scan_chi_vectorized`).
  The curvature window

  .. math::

     \chi_{\max}^{\text{old}} = \frac{|\varepsilon_{cu}|}{0.3\,d_{\max}} \cdot 1.5

  was calibrated for concrete crushing only.  In the tension-controlled
  regime the window was up to 4× too narrow, causing ``chi_u`` (and
  occasionally ``chi_cr``) to be reported as ``None``.

  Replaced by the span-bound

  .. math::

     \chi_{\max} = \frac{\varepsilon_{xg} - \varepsilon_{mb}}{\operatorname{span}(\theta)}

  which is a rigorous upper bound: at this curvature the strain difference
  between the two extreme fibres exhausts the entire admissible range.
  A user-supplied ``chi_max`` is still honoured.

  Event detection is now restricted to rows where the axial equilibrium
  solve converged, preventing spurious detections from the softening
  branch.  ``_scan_chi_vectorized`` returns a ``diagnostics`` sub-dict
  with fields ``chi_max_scanned``, ``n_converged``, ``cracking_reason``,
  and ``ultimate_reason`` (``None`` = success).

- **Section outline: interior voids rendered solid**
  (:mod:`gensec.output.plots`, :mod:`gensec.output.geometry_plot`).
  Sections with holes (e.g. ``annulus_poly``) showed the void filled grey
  in all section-outline plots.  Root cause: Shapely does not guarantee
  CCW exterior orientation after boolean operations; the legacy code
  assumed it.  For the standard annulus the exterior ring was CW, leaving
  both rings co-oriented under Matplotlib's non-zero winding rule.

  Fix: new module ``output/_polydraw.py`` with a single authoritative
  :func:`polygon_to_path` that enforces exterior-CCW / hole-CW
  unconditionally and supports ``MultiPolygon`` /
  ``GeometryCollection``.  Both legacy helpers delegate to it.

- **T-section resistance domain: pivot-A branch inactive by default**
  (:class:`~gensec.solver.NMDiagram`).
  ``include_pivot_a`` defaulted to ``False``, truncating the resistance
  domain on the tension side and producing overconservative
  :math:`\eta_{3D}` for demands near :math:`N \approx 0`.
  Default changed to ``True``.  The bridge branch in ``generate``
  (uniaxial) is now guarded by ``if self.include_pivot_a:``.

- **Polar ductility CLI: ``TypeError`` on ``n_points`` keyword**
  (:mod:`gensec.cli`).
  The call site passed ``n_points=args.n_points`` to
  ``plot_polar_ductility_refactored``, which expects ``n_chi``.
  The polar-ductility output was entirely inaccessible from the CLI.
  Fixed by adding a dedicated ``--n-chi`` argument (default 100).

- **Server import failure: ``api.InspectResult`` missing**
  (:mod:`gensec.server`, :mod:`gensec.api`).
  ``response_model=api.InspectResult`` was evaluated at ``create_app()``
  time; the missing attribute caused an ``AttributeError`` at import,
  breaking test collection.  Added ``InspectResult``, ``inspect()``, and
  ``SectionPropertiesPayload`` to ``gensec.api``.  Incidental fix:
  ``_Session.get_nm_3d`` was a module-level function with a ``self``
  parameter; re-indented into the class.

**Performance**

- **Active-set masking in** ``_vectorized_solve_N``: a boolean ``active``
  mask deactivates converged configs and configs with collapsed Newton
  tangent after each iteration, instead of iterating all configs to
  ``n_iter=15``.  Effective for every caller: ``generate_mx_my``,
  ``generate_moment_curvature``, ``eta_demand``, 3-D moment-curvature.

- **Decouple** ``n_scan`` **from** ``n_angles`` **in**
  ``generate_mx_my``: new optional ``n_scan`` parameter capped at 120,
  independent of the output resampling resolution ``n_angles``.  The
  verification branch drops from 360 to 120 internal scans (≈ 3×) with
  no loss of boundary accuracy.  YAML flag: ``n_scan_mx_my``.

- **Reduce default** ``n_chi`` **and bias curvature spacing**: default
  ``n_chi`` reduced from 36 to 14; steps distributed as

  .. math::

     \chi_i = \chi_{\max} \cdot \left(\frac{i}{n_\chi}\right)^{0.7}

  biasing toward :math:`\chi_{\max}` where the hull boundary lies.
  YAML flag: ``n_chi_mx_my``.

- **Memory budget correction in** ``_mega_batch_integrate``: the stated
  peak of ~400 MB per chunk was 4× too low (four simultaneous
  :math:`(n, n_\text{fibers})` arrays are live at peak).  Constant
  reduced from ``50_000_000`` to ``~13_000_000``; structural fix
  reformulates integrals as :math:`\boldsymbol{\sigma} \cdot
  \mathbf{w}` matrix–vector products, eliminating ``fA`` and moment
  temporaries and reducing peak from ~1.6 GB to ~0.5 GB per chunk.

**New features**

- **Analysis pipeline** (``gensec analyze``): new
  :class:`~gensec.solver.AnalysisEngine` computes per-material force
  decomposition and optional on-demand :math:`\eta` via exponential
  scan + bisection (30 iterations, precision ≈ 10⁻⁹) without generating
  the full resistance domain.

- **Tiered verification reporting** (new :mod:`gensec.output.summary`):
  three-level output — summary block (always), configurable Top-K console
  table, full ``verification_statistics.json`` export.  Scales to
  hundreds of combinations.  Heatmap colour bug fixed (verified bars now
  use per-η-type colours; exceeded bars turn red with ``///`` hatch).

**Tests**

- ``TestMomentCurvatureUltimateRegression``,
  ``TestMomentCurvatureDiagnostics`` (``tests/test_capacity_v2.py``).
- ``TestOutlineHoleCarving`` (``tests/test_v030_regressions.py``).
- End-to-end CLI test for polar ductility.
- 15 new tests in ``tests/test_analysis.py``.

**Deprecations**

- ``plot_polar_ductility(..., n_points=...)`` — deprecated alias for
  ``n_chi``; emits ``DeprecationWarning``.  Will be removed in v0.4.0.
- ``_scan_chi`` — retained as reference; scheduled for removal in v0.4.0
  after regression suite confirms equivalence with
  ``_scan_chi_vectorized``.


v0.3.2
------

Performance-focused release: vectorised moment–curvature scanner and
polar ultimate-curvature computation.

**Performance**

- **Vectorised** ``_scan_chi_vectorized``
  (:meth:`~gensec.solver.NMDiagram._scan_chi_vectorized`): replaces the
  sequential Python loop in ``generate_moment_curvature`` with a
  batch-vectorised solve using the ``_vectorized_solve_N`` +
  ``integrate_batch`` pattern already proven in ``generate_mx_my``.
  Estimated speed-up: 10–30× on typical sections.

- **New public method** ``generate_polar_curvature``
  (:meth:`~gensec.solver.NMDiagram.generate_polar_curvature`): moves
  the computational logic of the polar ultimate-curvature diagram from
  ``plots.py`` into the solver layer, matching the data/plot separation
  pattern used elsewhere.  Returns
  ``{"thetas", "chi_u", "chi_u_km", "eps0_u", "N_fixed_kN"}``.
  Speed-up over the sequential nested loop: 20–60×.

- ``plot_polar_ductility`` refactored to delegate to
  ``generate_polar_curvature``; ``n_points`` renamed to ``n_chi``
  (deprecated shim retained).

- Default ``--n-points`` CLI option reduced from 200 to 50; reflects the
  vectorised regime where a single batch call dominates over config count.

**New output**

- ``export_polar_curvature_json``, ``export_polar_curvature_csv``
  (:mod:`gensec.output.export`).


v0.3.1
------

Section properties and documentation.

**New**

- :mod:`gensec.geometry.properties`: homogenized geometric properties
  on any polygon + rebar configuration — area, centroid, principal axes,
  extreme-fibre distances, elastic moduli (:math:`W`), plastic moduli
  (:math:`Z` via PNA bisection), central inertia ellipse, kern.
  Torsional constant :math:`I_t` kept as placeholder.
- ``GenericSection.ideal_gross_properties`` lazy property.
- EC2 homogenization convention: net rebar contribution
  :math:`(n_s - n_\text{bulk}) \cdot A_s`.
- New plot/report functions ``plot_section_properties``,
  ``print_section_properties``, ``write_section_report``
  (:mod:`gensec.output.geometry_plot`); legacy aliases preserved.

**Documentation**

- ``docs/theory/ideal_gross_properties.rst``: EC2 homogenization
  convention with worked example.
- ``docs/theory/integration_methods.rst``: polygonal vs fibre
  integration comparison, convergence benchmarks on disc and RC section.

**Tests**

- 10 new analytical tests in ``tests/test_properties.py``.


v0.3.0
------

Tier-1 fixes for symmetry and asymmetric-section robustness; complete
redesign of the demand-verification metric architecture.

**Bug fixes**

- **A1 — N-M cloud symmetry for centred sections**
  (:meth:`~gensec.solver.NMDiagram._ultimate_strain_configs_1d`).
  The conversion from edge strains
  :math:`(\varepsilon_i, \varepsilon_s)` to
  :math:`(\varepsilon_0, \chi)` used :math:`y_\text{ref}` as the lever
  arm, correct only for sections anchored at :math:`y_\text{min} = 0`.
  Fixed to use :math:`y_\text{ref} - y_\text{min}`.  For sections with
  :math:`y_\text{min} = 0` (all standard primitives) the output is
  bit-for-bit identical.

- **B1 — Symmetric fast-path for** :math:`M_x \approx 0`
  (:meth:`~gensec.solver.FiberSolver.solve_equilibrium`):
  mirror of the existing :math:`M_y \approx 0` fast-path; new methods
  ``_solve_uniaxial_y``, ``_nr_uniaxial_y``.

- **B2 — Uniaxial detection in both axes**
  (:meth:`~gensec.solver.FiberSolver._is_uniaxial`): extended to detect
  both vertical-degenerate (all fibres at same :math:`x`) and
  horizontal-degenerate (all fibres at same :math:`y`) sections.
  New optional ``axis`` parameter.

- **B3 — Newton convergence on softening branch:** ``_vectorized_solve_N``
  skips the Newton step when ``|dN/dε₀| < 1.0`` (collapsed tangent)
  instead of dividing by near-zero; behaviour unchanged for converged
  configs.

- **F1 — Multi-material rebar force**
  (:meth:`~gensec.solver.FiberSolver._setup_rebars`): ``mat_indices_rebar``
  now stores the bulk-material index for each rebar group; the net force
  convention :math:`F_\text{net} = [\sigma_s(\varepsilon) -
  \sigma_\text{bulk}(\varepsilon)] \cdot A_s` is applied per group.

**Architecture**

- **Seven-metric verification redesign** (:mod:`gensec.solver.check`):
  three 3-D metrics in anisotropy-corrected normalised space
  (``eta_norm``, ``eta_norm_beta``, ``eta_norm_ray``), one 2-D metric
  (``eta_2D``), and three staged-path metrics
  (``eta_path_norm_ray``, ``eta_path_norm_beta``, ``eta_path_2D``).
  Chebyshev LP for bounding-ball radius; segment-to-hull distance helper.
  Previous ``eta_3D`` replaced.

- New YAML flags for all seven metrics; ``eta_norm`` and
  ``eta_norm_beta`` are ``true`` by default.

**Tests**

- ``test_check.py`` updated for v0.3 architecture (82 tests).
- ``test_v030_regressions.py``: 32 regression tests.


v0.2.0
------

Performance overhaul and architecture refinements.

**Performance**

- **Batch integration**
  (:meth:`~gensec.solver.FiberSolver.integrate_batch`): vectorised NumPy
  evaluation of thousands of strain configurations in a single call.
- **Mega-batch with chunking**: ``generate_biaxial`` and ``generate_mx_my``
  concatenate all curvature directions into a flat array and integrate in
  large chunks; 10–70× reduction in per-call overhead.  Memory cap:
  ``max_configs = 50_000_000 // n_fibers`` elements per chunk.
- **Analytical tangent**
  (:meth:`~gensec.solver.FiberSolver.integrate_with_tangent`): internal
  forces and the :math:`3 \times 3` tangent matrix in one pass.
- **Optional Numba JIT** (``pip install gensec[fast]``): JIT-compiled
  stress and tangent kernels for ``Concrete`` and ``Steel``.
- Measured end-to-end improvement: **15–40×** on typical biaxial analyses.

**Materials**

- ``tangent(eps)`` and ``tangent_array(eps)`` added to the abstract
  :class:`~gensec.materials.Material` interface; closed-form
  implementations for ``Concrete``, ``Steel``, ``TabulatedMaterial``;
  finite-difference fallback in base class.
- ``stress_array`` accepts arrays of any shape.

**Geometry**

- ``RectSection`` is now a factory function returning a
  :class:`~gensec.geometry.GenericSection` directly.
- ``GenericSection`` accepts ``n_grid_x`` / ``n_grid_y`` for explicit
  grid control; exposes ``dx``, ``dy``; default isotropic grid.

**Verification**

- ``VerificationEngine`` auto-disables ``eta_2D`` / ``eta_path_2D`` on
  2-D resistance domains.
- ``_get_contour`` caches degenerate-contour failures as ``None``.

**Output flags**

- New YAML flags: ``generate_moment_curvature``,
  ``generate_polar_ductility``, ``generate_3d_moment_curvature``
  (default ``true``).


v0.1.0
------

Phase 2 complete: biaxial bending and generic sections.

- Generic section with arbitrary Shapely polygons and automatic fibre
  meshing (grid and triangular).
- Parametric primitives: rectangle, circle, annulus, T, inverted T, H,
  box, single-tee slab, double-tee slab.
- Biaxial bending: full :math:`(N, M_x, M_y)` resistance surface via
  curvature-angle scanning.
- :math:`M_x`–:math:`M_y` contour diagrams at constant :math:`N`.
- 3-D resistance surface visualisation.
- Convex-hull demand verification in 2-D and 3-D modes.
- Combinations: named groups with envelope :math:`\eta_{\max}`.
- EN 10025-2 structural steel with thickness-dependent properties.
- Multi-material bulk support.
- Per-fibre post-processing: strain/stress extraction and section-state
  plots.
- 106 tests.


v0.0.1
------

Phase 0 + Phase 1: uniaxial bending for rectangular sections.

- Parabola-rectangle concrete (EC2 / NTC 2018).
- Elastic-plastic steel with optional hardening.
- Fibre integrator with Newton–Raphson inverse solver.
- :math:`N`–:math:`M` interaction diagram via EC2 pivot method.
- YAML-driven input.
- CLI and Python API.
- EC2 Table 3.1 property computation (French National Annex).
- Six analytical validation test cases, all passing within 1 %.
