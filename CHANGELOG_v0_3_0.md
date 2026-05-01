# CHANGELOG — GenSec v0.3.0

## Tier-1 fixes for symmetry and asymmetric-section robustness, complete redesign of the demand-verification metric architecture, anisotropy-corrected normalised space.

### Bug fixes

#### A1 — Symmetry of the uniaxial N-M cloud for centred sections
`capacity.py::NMDiagram._ultimate_strain_configs_1d`

The conversion from edge strains :math:`(\varepsilon_i, \varepsilon_s)`
to the integrator's parametrisation :math:`(\varepsilon_0, \chi)` used
:math:`\varepsilon_0 = \varepsilon_i + \chi \cdot y_{\text{ref}}`,
which is correct only when the section's bottom edge sits at
:math:`y = 0`.  All standard primitives (`rect_poly`, `circle_poly`,
`tee_poly`, …) anchor the bbox to the origin, so the bug was silent
for them.  For sections defined on a centred frame
(:math:`y \in [-H/2, +H/2]`, e.g. custom polygons) the missing
translation term broke the symmetry of the produced configuration set
and the resulting N-M cloud.

The fix uses the physical lever arm
:math:`y_{\text{ref}} - y_{\min}` instead of :math:`y_{\text{ref}}`.
The new formula is bit-for-bit identical to the previous behaviour
for every section with :math:`y_{\min} = 0`, and produces a
mirror-symmetric cloud for centred sections.

Verified on a doubly symmetric octagonal section
(:math:`y \in [-505, +505]`): post-patch, `M_max + M_min = 0` to
9e-13 kN·m, and the maximum mirror distance in the `(N, M)` cloud
drops from ~3000 kN·m·equiv to 9e-13.

#### B1 — Symmetric fast-path for `Mx_target ≈ 0`
`integrator.py::FiberSolver.solve_equilibrium`

`solve_equilibrium` had a fast-path that fell back to the 2-unknown
uniaxial solver when `My_target` was negligible, but no symmetric
counterpart for `Mx_target ≈ 0`.  Demands dominated by `My`
therefore went straight to the 3×3 biaxial Newton, with worse
convergence and higher cost.

Added the mirror branch: when `abs(Mx_target) < M_tol` and
`Mx_target` is the small one, the solver now tries `_solve_uniaxial_y`
first and accepts it only if the spurious `Mx` stays inside
tolerance.  Two new methods: `_solve_uniaxial_y` and `_nr_uniaxial_y`,
mirror images of the x-axis variants.  The `### TODO` previously
marking this asymmetry has been removed.

#### B2 — Uniaxial detection works in both axes
`integrator.py::FiberSolver._is_uniaxial`

`_is_uniaxial` only recognised "vertical-degenerate" sections (all
fibers at the same `x`).  A "horizontal-degenerate" section (all
fibers at the same `y`) would slip through and reach the biaxial
Newton with a singular `chi_x` column.

Extended to detect both cases.  New optional `axis` parameter: when
specified, the test is restricted to the matching degeneracy
(`'x'` ⇒ vertical-deg, `'y'` ⇒ horizontal-deg).  Without `axis` the
function returns `True` for either degeneracy, preserving
backward-compatible behaviour at every existing call site that does
not pass the parameter.

#### B3 — Symmetric elastic initial guess
`integrator.py::FiberSolver._elastic_initial_guess`

The fallback path (taken when the elastic 3×3 system is singular)
estimated `chi_x` from `Mx_target` using `I_approx = A * H^2 / 12`,
but always returned `chi_y = 0` even when `My_target` was the
dominant moment.  For demands with `Mx ≈ 0, My ≠ 0`, the Newton
then had to find `chi_y` from zero.

Added a parallel estimate for `chi_y` using `I_approx_y = A * B^2 / 12`,
and consolidated the magic number `30000` (effective concrete
modulus) into a named local with a TODO pointing to the future
material-aware fix.  Bit-for-bit identical when `My_target = 0`.

#### F1 — Embedded-rebar bulk subtraction is zone-aware
`integrator.py` + `geometry.py`

For multi-material sections (e.g. a confined core inside an
unconfined cover), embedded rebars must subtract the displaced bulk
stress evaluated with the **constitutive law of the zone the rebar
physically occupies**, not the primary `bulk_material`.  Pre-fix,
all rebars were subtracted with the primary bulk law, biasing the
section's force balance for any multi-bulk configuration.

Fix:

- `geometry.py::_setup_rebars` now also computes
  `mat_indices_rebar`: a per-rebar zone index, looked up via
  `_material_index(r.x, r.y)`.
- `integrator.py::_build_rebar_groups` splits rebars by the pair
  `(rebar material, bulk-zone material)`, so the embedded subtraction
  is vectorised per group.  The four integration routines
  (`integrate`, `integrate_with_tangent`, `integrate_batch`, and
  the diagnostics method `state_at`) were all updated.
- Sections without `mat_indices_rebar` (legacy `RectSection`)
  default to zone 0 transparently.

#### I1 — `is_biaxial` requires extent in both directions
`cli.py` (and matching info message in `check.py`)

`is_biaxial = section.n_fibers_x > 1` misclassified narrow tall
walls (1 fiber column, many rows) as biaxial — they have real
extent in `y` but zero extent in `x`, and the resulting Mx-My
contour was degenerate.  Now requires
`n_fibers_x > 1 AND n_fibers_y > 1`.

### New features and major redesign

#### K — Demand-verification metric architecture (complete redesign)

`check.py` is restructured around **seven utilization metrics** in
two geometric families, all operating in *anisotropy-corrected
normalised space*.  The previous single 3-D ray-cast (`eta_3D`)
is **removed** — it was scale-dependent and physically misleading
for sections with strong geometric anisotropy (walls, slabs).

##### Anisotropy-corrected normalised space

The 3-D metrics operate in a coordinate system where the axial
axis is rescaled by `u_x = ΔMx/ΔN` and the My axis by
`v_y = ΔMx/ΔMy`, so that the bounding box of the resistance
domain has the same numeric extent on every axis.  This makes
euclidean distances physically meaningful for any section shape
and makes the metrics **scale-invariant** under any consistent
change of force / moment units.

For a 187×30 cm wall (ΔMx/ΔMy ≈ 36), the previous ray-cast metric
weighted Mx thirty-six times more than My in distance computations,
producing values that did not correspond to the engineering
intuition of "consumed reserve".  The new normalisation eliminates
this asymmetry.

##### Point metrics (3-D family)

- **`eta_norm`** (alpha) — Linear distance from the demand to the
  nearest boundary face, expressed as a fraction of the **Chebyshev
  radius** D_max of the normalised domain (the radius of the
  largest inscribed sphere, computed at construction by linear
  programming):

    `η_norm = 1 - d_min / D_max` (interior)
    `η_norm = 1 + d_min / D_max` (exterior)

  A true geometric distance: linear and monotone in proximity to
  the boundary.  Reaches 0 at the Chebyshev centre (the
  geometrically deepest point), 1 on the boundary, > 1 outside.
  **The recommended principal 3-D metric.**  Default ON.

- **`eta_norm_beta`** — Composite ratio
  `F_SU / (F_SU + d_min)`, mixing the demand norm `F_SU` with the
  face-distance `d_min`.  Not a pure distance: a ratio that grows
  both with proximity to the boundary and with the demand
  magnitude.  Reads as "sensitivity to perturbation in proportion
  to the demand magnitude".  Useful for cross-software validation
  against commercial RC tools that report equivalent quantities.
  Default ON.

- **`eta_norm_ray`** — Ray-cast from the origin to the demand in
  normalised space.  Linear in demand magnitude along any fixed
  ray.  Useful for proportional load-amplification analyses
  (overall load factor).  Default OFF (opt-in).

##### Point metric (2-D family)

- **`eta_2D`** — Ray-cast in the (Mx, My) plane at the demand's N.
  Useful for fixed-axial-force flexural verifications.  Requires a
  biaxial section.  Default OFF (more expensive than 3-D metrics).

##### Path metrics (staged combinations)

- **`eta_path`** — 3-D ray-cast in normalised space, from the
  previous-stage cumulative demand B to the current-stage
  cumulative demand T.  Stored under the key `eta_path_norm_ray`
  in per-stage results.  Path-direction-aware analogue of
  `eta_norm_ray`.  Default OFF (opt-in for staged loading).

- **`eta_path_norm_beta`** — Composite ratio along the stage
  segment B→T:

    `η = L / (L + d_seg)`

  with `L = |T-B|` and `d_seg` the minimum signed distance of the
  segment from the boundary.  Path analogue of `eta_norm_beta`.

  *Particularly informative in seismic verification with staged
  gravity*: when gravity is applied as a first stage and seismic
  action as a second stage, this metric flags the "structure
  already at the limit before the seismic increment" case
  distinctly from `eta_path` (the ray sees only the target
  clearance and the direction of growth, while `eta_path_norm_beta`
  sees the worst-case clearance over the entire path, weighted
  by the path size).  Default OFF (opt-in).

- **`eta_path_2D`** — 2-D variant on the Mx-My contour at
  cumulative N.  Skipped when the axial-force change between
  stages exceeds `delta_N_tol`.  Default OFF.

##### Verdict and governing

All enabled metrics contribute to `verified`, `eta_max` (envelopes)
and `eta_governing` (combinations).  Pass/fail is decided by the
**worst** of the enabled metrics.  Each metric answers a different
geometric question; no strict ordering between them holds in
general, so they are complementary, not redundant.

##### Staged warning

When a staged combination is processed without any `eta_path_*`
flag enabled, an informational warning is printed to stderr:
without a path-aware metric the report would only contain point
metrics at each cumulative state and would not characterise the
load history.

##### Verification

Geometric properties verified by the regression test suite
(`tests/test_v030_regressions.py`):

- `eta_norm` (alpha): zero at the Chebyshev centre,
  monotone toward the boundary, < 1 interior / > 1 exterior.
- `eta_norm_beta`: zero at the origin, < 1 interior / > 1 exterior,
  monotone along radial.
- `eta_norm_ray`: zero at the origin, **linear** along any radial
  (doubling the demand doubles η).
- `eta_path_norm_beta`: collapses to point β for zero-length
  segments, path-dependent (different bases ⇒ different values).

### Documentation

- `yaml_reference.rst`: rewritten "Utilization metrics" section,
  documenting the seven metrics in two families with use cases.
- `demand_verification.rst`: full rewrite covering the
  anisotropy-corrected normalised space construction, the seven
  metrics with formulas and geometric readings, the path-metric
  semantics, and the staged-warning behaviour.
- Updated reference YAML (`yaml_reference_example.yaml`) with
  inline comments explaining when to use each metric.

### Breaking changes

- **`eta_3D` removed**: the legacy 3-D ray-cast in raw coordinates
  is no longer available.  Setting `eta_3D: true` in YAML has no
  effect (the flag is silently ignored, like any unknown key).
  Replacements:
  - For "fraction of available reserve consumed":
    use `eta_norm` (alpha).
  - For "proportional load amplification":
    use `eta_norm_ray` (ray-cast in normalised space).
  - For cross-software comparison with composite-ratio
    formulations: use `eta_norm_beta`.

- **CSV / JSON column changes**: the column `eta_3D` is removed
  from `verification_summary.csv` and `verification_summary.json`.
  New columns: `eta_norm`, `eta_norm_beta`, `eta_norm_ray`.  Per-
  stage results gain `eta_path_norm_ray` (replacing the previous
  `eta_path` key) and `eta_path_norm_beta`.

- **Default flags**: previously `eta_3D` was the default-on
  principal 3-D metric.  Now `eta_norm` and `eta_norm_beta` are
  default-on, while path-aware metrics (`eta_path_norm_ray`,
  `eta_path_norm_beta`, `eta_path_2D`) are default-off.  Combined
  with the new staged warning, this ensures users actively choose
  which path metric is appropriate for their analysis.

- **`DomainChecker.eta_path` renamed**: the public method is now
  `DomainChecker.eta_path_norm_ray`.  The result-dict key in
  per-stage output is `eta_path_norm_ray` (was `eta_path`).

### Tier-3 deferrals (not in this release)

- **C1 / C2** (`plot_polar_ductility` chi_max asymmetry +
  cross-angle warm-start): C1 alone is cosmetic without C2.  Marked
  with an explicit code comment in `plots.py` pointing at the joint
  fix.  Tracked for v0.4.0.
- **B4 / B5** (concrete-specific magic numbers in Newton clamp and
  in initial-guess effective modulus): material-aware refactor
  beyond the local cleanup of B3.
- **G2** (linear interpolation in N-crossing for Mx-My contour):
  PCHIP upgrade with benchmark.

### Migration notes

The Tier-1 fixes are backward-compatible by construction:

- All standard primitives (the A1 fix is a no-op for sections with
  `y_min = 0`).
- All single-bulk-material sections (the F1 fix is a no-op when
  every rebar sits in zone 0).

The metric architecture redesign **is not** backward-compatible:

- YAML files that explicitly set `eta_3D: true` will keep working
  but the flag is now silently ignored (no `eta_3D` column in
  output).  No error is raised — `eta_3D` is treated like any
  unknown key, exactly as YAML files written for the new
  architecture would treat e.g. `eta_pippo: true`.

- Code that consumed `eta_3D` from result dicts or CSV columns
  must migrate.  Choose the replacement based on intent:

  | Old `eta_3D` usage           | New replacement              |
  |------------------------------|------------------------------|
  | "consumed reserve fraction"  | `eta_norm` (alpha)           |
  | "proportional amplification" | `eta_norm_ray`               |
  | cross-software validation    | `eta_norm_beta`              |
  | staged ray-cast (3D)         | `eta_path_norm_ray`          |

- The numerical *values* of the new metrics are not comparable to
  the old `eta_3D`: they live in a different coordinate system
  (anisotropy-corrected normalised) and use different geometric
  primitives (sphere distance, ray-cast, composite ratio).

GenSec is in early development; the active redesign of the
verification metrics is intentional and will continue to be
iterated until v1.0.

### Verification

- 32 regression tests in `tests/test_v030_regressions.py`,
  covering A1 (3), B1 (2), B2 (1), B3 (1), F1 (3),
  K alpha (4), K beta (5), K ray (3), K path-beta (5),
  staged warning (2), and `VerificationEngine` integration (2).
  All 32 pass.
- 82 tests in `tests/test_check.py` (updated for the v0.3
  architecture).  All 82 pass.
- End-to-end smoke test on `test_exotic.yaml` confirms:
  - N-Mx cloud symmetric to ~1e-12 (was ~3000 kN·m before A1).
  - All four point metrics produced and exported correctly.
  - All three path metrics activated by combination produce
    coherent values on staged inputs.
  - Staged warning correctly emitted when no `eta_path_*` flag
    is enabled but a multi-stage combination is processed.
