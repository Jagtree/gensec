# CHANGELOG — GenSec v0.3.3

**Release scope:** bug fixes, performance improvements, new analysis pipeline,
API and output hardening.  All changes are backward-compatible unless
explicitly noted.

---

## 1. Bug fixes

### 1.1 Moment–curvature: ultimate / cracking key-points not detected

**Component:** `solver/capacity.py` — `NMDiagram.generate_moment_curvature`,
`_scan_chi_vectorized`

The curvature window `[0, chi_max]` was set with a fixed heuristic calibrated
on concrete crushing only:

```
chi_max = |eps_cu| / (0.3 * d_max) * 1.5
```

In the tension-controlled regime (low or negative *N*, lightly-reinforced
sections) the steel rupture strain governs and the true ultimate curvature can
be up to **4× larger** than the heuristic, falling silently outside the
window.  As a result, `chi_u` (and occasionally `chi_cr`) was reported as
`None` even for valid sections.

**Fix.** The span-bound

```
chi_max = (eps_xg - eps_mb) / span(theta)
```

replaces the heuristic.  This is a rigorous upper bound: at this curvature the
strain difference between the two extreme fibres exhausts the entire admissible
range, so at least one material limit is always reached within the window.
A user-supplied `chi_max` is still honoured; the geometric estimate is retained
only as a degenerate-span fallback.

Additionally, event detection is now restricted to rows where the axial
equilibrium solve converged (`|N_solved - N_fixed| < tol`), preventing
spurious detections from the softening branch.

**Diagnostics.** `_scan_chi_vectorized` now returns a `diagnostics` sub-dict,
surfaced by `generate_moment_curvature` as `diagnostics_pos` / `diagnostics_neg`.
The `reason` field is `None` on success and a string otherwise:

| Field | Values |
|---|---|
| `chi_max_scanned` | float — largest scanned curvature |
| `n_points` | int |
| `n_converged` | int |
| `cracking_reason` | `None` · `no_ec2_properties` · `no_tension_in_range` · `no_convergence` · `below_threshold` |
| `ultimate_reason` | `None` · `no_convergence` · `limit_not_reached_in_range` |

**Regression tests** (`tests/test_capacity_v2.py`):
`TestMomentCurvatureUltimateRegression`, `TestMomentCurvatureDiagnostics`.

**API impact:** additive only; all existing return keys unchanged.

---

### 1.2 Section outline rendering: interior voids filled solid

**Component:** `output/plots.py` — `_draw_polygon_outline`;
`output/geometry_plot.py` — `_draw_polygon_with_holes`

Sections with interior voids (e.g. `annulus_poly`) rendered with the void
filled grey in every plot that draws the section outline.  The root cause was
that Shapely does **not** guarantee any particular ring orientation after
boolean operations such as `Polygon.difference`; the legacy code trusted the
exterior ring to be CCW.  For the standard annulus the exterior was CW,
producing two co-oriented rings; under Matplotlib's non-zero winding rule the
winding number inside the void was ±2 ≠ 0 and the hole was filled.

**Fix.** New module `output/_polydraw.py` with a single authoritative
`polygon_to_path(geom)` function that enforces exterior-CCW / hole-CW
regardless of input orientation, and supports `MultiPolygon` and
`GeometryCollection`.  Both legacy helpers (`_draw_polygon_outline` and
`_draw_polygon_with_holes`) are reduced to thin delegates.

The `PathPatch` / `Path` imports are removed from both edited modules.

**Robustness gained:** the shared primitive handles disconnected section
geometries (`MultiPolygon`) which both legacy copies would have raised on —
a real case for custom multi-region sections and for future Class 4 steel
multi-plate paths.

**Regression test** (`tests/test_v030_regressions.py`):
`TestOutlineHoleCarving` — asserts exterior signed area > 0 (CCW) and each
hole signed area < 0 (CW) via Matplotlib `Path` decomposition.

---

### 1.3 T-section resistance domain: `include_pivot_a` default corrected

**Component:** `solver/capacity.py` — `NMDiagram.__init__`

Default `include_pivot_a` was `False`, omitting the EC2 pivot-A branch
(pure tension end of the interaction diagram) from the default pipeline.
This caused the resistance domain to be truncated on the tension side and
produced an overconservative `eta_3D` for demands near *N* ≈ 0.

**Fix.** Default changed to `True`.  No changes to call sites (`cli.py`,
`api.py`): the full EC2 domain is now generated automatically.  The bridge
branch in `generate` (uniaxial) is guarded with `if self.include_pivot_a:`,
removing a previously hardcoded unconditional activation.

**Validation.** Verified on the T-section fixture (`example_tee.yaml`,
800×150/300×450, C25/30): resistance domain N-Mx is physically asymmetric
(correct for a non-symmetric section in x) and symmetric in N-My (correct
for a section with vertical symmetry axis).  Mirror residual on My axis:
≤ 5.2 × 10⁻³ of domain diagonal (numerical noise); on Mx axis: ~ 4.6 × 10⁻²
(physical asymmetry, not a bug).

---

### 1.4 Polar ductility CLI: `n_points` keyword argument mismatch

**Component:** `cli.py` line ~674

`cli.py` passed `n_points=args.n_points` to `plot_polar_ductility_refactored`,
but the function parameter is named `n_chi`.  The call raised a `TypeError` at
runtime, making the polar-ductility output entirely inaccessible from the CLI.

**Fix (recommended — Option 3).** A dedicated `--n-chi` CLI argument (default
100) is added; the call site passes `n_chi=args.n_chi`.  This preserves the
semantic distinction between `--n-points` (geometric resolution of the N-M
contour) and `n_chi` (curvature-axis resolution along a single direction in
the polar scan).  The ductility post-processing pipeline (median filter +
smoothing) is calibrated for `n_chi=100`; coupling it to `--n-points` (default
50) would under-resolve `chi_u` before filtering.

A deprecation shim `n_points=None` is retained in `plot_polar_ductility` for
backward compatibility until API callers are migrated.

**Regression test:** end-to-end CLI test on a biaxial section with
`generate_polar_ductility: true` — must complete without `TypeError`.

---

### 1.5 Server import failure: `api.InspectResult` missing

**Component:** `server.py`, `api.py`

`server.py` declared `response_model=api.InspectResult` (evaluated at
`create_app()` time, i.e. at module import), but `InspectResult` and
`inspect()` did not exist in `gensec.api`.  This caused an
`AttributeError` at import, breaking `tests/test_server.py` at collection
time — not just at runtime.

**Fix.** Added to `gensec/api.py`:

- `SectionPropertiesPayload` — JSON-safe Pydantic mirror of
  `gensec.properties.SectionProperties` (44 fields, all `Optional`).
  Non-finite floats (`NaN`, `±inf` from plastic moduli and zero-span
  elastic moduli) are mapped to `null`; `json.dumps(..., allow_nan=False)`
  succeeds.
- `InspectResult` — `materials`, `section`, `properties` (optional),
  `meta`.  `properties` is `None` with a warning in `meta` when
  homogenization cannot be computed (e.g. multi-material bulk), rather
  than raising a 500.
- `inspect(yaml_text=None, yaml_path=None, compute_plastic=True)` —
  parses YAML and computes homogenized properties only.  No fiber solver,
  no resistance domain, no verification engine.  Target latency: < 200 ms.
- `_load_normalised_yaml` — shared helper for the temp-file dance
  required by `load_yaml`; removes duplication between `_Session.build`
  and the new parse path.
- `clear_cache()` now also evicts the new parse cache.

**Incidental fix:** `_Session.get_nm_3d` was defined at module level
(column 0) instead of as a `_Session` method, causing `AttributeError`
at runtime on biaxial sections via `session.get_nm_3d()`.  Re-indented
into the class.

**Follow-up (deferred):** add a real `/api/inspect` integration test;
remove the stale `@pytest.mark.xfail` on `test_analyze_biaxial_column_returns_valid_model`
(now XPASS).

---

## 2. Performance

### 2.1 Active-set masking in `_vectorized_solve_N`

**Component:** `solver/capacity.py` — `NMDiagram._vectorized_solve_N`

The Newton loop ran two `integrate_batch` calls over the **full** config array
at every iteration, with global exit only when all residuals converged.  A
handful of configs in the softening branch (where `|dN/dε₀| → 0`, filtered by
the `safe` mask but never removed) were enough to force all configs to iterate
to `n_iter=15`.  This affected every caller: `generate_mx_my`,
`generate_moment_curvature`, `eta_demand`, 3D moment-curvature.

**Fix.** An `active` boolean mask of size *n* is maintained.  Converged configs
and configs with collapsed tangent are deactivated immediately; each iteration
integrates only `np.nonzero(active)` rows.  After 3–4 iterations the active
set is typically < 5 % of the original size.

**Equivalence.** Converged configs are stored at the converging iteration (not
overwritten).  Tangent-collapsed configs retain their last value — identical
to the previous behaviour (`step=0`, then filtered by the post-solve
`|N - N_fixed| < 1e3` guard).

**Validation required:** full test suite (`test_solver_uniaxial`,
`test_solver_biaxial`, `test_check`, `test_capacity_v2`) before closing.

---

### 2.2 Decouple `n_scan` from `n_angles` in `generate_mx_my`

**Component:** `solver/capacity.py` — `generate_mx_my`

The internal scan direction count was set equal to `n_angles` (the output
resampling resolution).  The ConvexHull captures the exact boundary from the
raw point cloud; `n_angles` only controls post-hull resampling.  For a convex
domain these are independent.

**Fix.** A new optional `n_scan` parameter is added:

```python
def generate_mx_my(self, N_fixed, n_angles=72, n_scan=None,
                   n_points_per_angle=200, n_chi=14):
    if n_scan is None:
        n_scan = min(max(n_angles, 72), 120)
```

The default cap of 120 directions reduces the verification branch from 360 to
120 internal scans (≈ 3× on that call) with no loss of boundary accuracy.
YAML-configurable via `n_scan_mx_my` in `output:`.

---

### 2.3 Reduce default `n_chi` and improve curvature spacing

**Component:** `solver/capacity.py` — `generate_mx_my`

Default `n_chi` reduced from 36 to 14.  Curvature steps are distributed with a
power < 1 toward `chi_max` (where the hull boundary lies):

```python
frac = np.linspace(1.0 / n_chi, 1.0, n_chi)
chis = chi_max * frac**0.7
```

Points at low *χ* are mostly interior to the point cloud and discarded by the
hull; biasing toward `chi_max` preserves boundary accuracy at ≈ 2× lower cost.
YAML-configurable via `n_chi_mx_my` in `output:`.

---

### 2.4 Memory budget correction in `_mega_batch_integrate`

**Component:** `solver/capacity.py` — `_mega_batch_integrate`

The comment *"~400 MB peak per chunk"* was incorrect.  `integrate_batch` keeps
four simultaneous `(n, n_fibers)` arrays alive at peak (strain `eb`, stress
`sb`, force `fA`, and a temporary for moment computation), giving a true peak
of ≈ 4 × 400 MB = 1.6 GB per chunk with the old `max_configs` constant.

**Fix.** The constant `50_000_000` is reduced to `~13_000_000`, capping a
single array at ~100 MB and the peak at ~400 MB as originally intended.
Comment corrected accordingly.

**Structural fix (recommended, independent).** Reformulate the integrals as
matrix–vector products against pre-cached weight vectors:

```python
# Cached once in __init__:
self._A    = sec.A_fibers
self._A_ly = sec.A_fibers * self._ly_bulk
self._A_lx = sec.A_fibers * self._lx_bulk

# Per chunk:
eb = eps0[:, None] + chi_x[:, None] * ly - chi_y[:, None] * lx
sb = bulk_material.stress_array(eb)
del eb
N  = sb @ self._A     # (n,)
Mx = sb @ self._A_ly
My = -(sb @ self._A_lx)
```

Eliminates `fA` and the two moment temporaries (−1.2 GB of transient peak);
BLAS `@` is faster than `*` + `.sum(axis=1)`.  Peak drops from ~1.6 GB to
~0.5 GB at equal chunk size.

**Chunk-size floor hardened.** The `max(2000, …)` floor is replaced with an
explicit byte-budget derivation so the comment is true by construction and
high-fiber-count sections cannot bypass the cap.

---

## 3. New features

### 3.1 Analysis pipeline — `gensec analyze`

**New files:** `solver/analysis.py`, `tests/test_analysis.py`

A lightweight pipeline that computes **per-material force decomposition** and
optional **on-demand η** for every demand or combination, without generating
the resistance domain (no `NMDiagram`, no `ConvexHull`, no
`VerificationEngine`).

CLI:

```bash
gensec analyze input.yaml --output-dir results          # force decomposition
gensec analyze input.yaml --output-dir results --eta    # + on-demand eta
```

`AnalysisEngine.analyze()` output structure:

```python
{
    "converged": bool,
    "strains_ok": bool,
    "strain_state": {"eps0": ..., "chi_x": ..., "chi_y": ...},
    "total": {"N_kN": ..., "Mx_kNm": ..., "My_kNm": ...},
    "components": [
        {"type": "bulk",  "material_name": "C25/30", ...},
        {"type": "rebar", "material_name": "B450C",  ...,
         "layers": [{"index": 0, "x": ..., "y": ..., "A": ...,
                     "eps": ..., "sigma_net": ..., "F_net_kN": ...}]},
    ],
}
```

**On-demand η** (equivalent to `eta_norm_ray`) via exponential scan +
bisection (30 iterations, precision ≈ 10⁻⁹):

1. Ray from base *B* through demand *D*: `P(t) = B + t·(D − B)`.
2. Exponential bracket: *t* = 1, 2, 4, 8, … until `_is_feasible` fails.
3. Bisection on `[t_lo, t_hi]`.
4. `η = 1 / t_boundary`.

A point is **feasible** if `solve_equilibrium` converges *and* all fibre
strains lie within `[eps_min, eps_max]` of their material.  Domain convexity
guarantees monotone convergence from any interior base point.

**Note:** `eta_norm` and `eta_norm_beta` require the full ConvexHull and
are not available in `gensec analyze`.  Envelopes are not supported
(demands and combinations only).

**Ancillary changes:**

- `materials/base.py`: `name = ""` class attribute on `Material`.
- `io_yaml.py`: `mat.name = mat_name` after `_build_material`.
- `solver/integrator.py`: new method
  `FiberSolver.strains_within_limits(eps0, chi_x, chi_y)`.
- `cli.py`: `analyze` subparser and `_analyze()` handler.
- `output/export.py`: `export_analysis_json()`, `export_analysis_csv()`.

**Documentation RST** (6 files updated): `architecture_solver.rst`,
`demand_verification.rst`, `quickstart.rst`, `yaml_reference.rst`,
`gensec_solver.rst`, `gensec_cli.rst`.

---

### 3.2 Tiered verification reporting

**Component:** new `output/summary.py`; edits to `cli.py`, `io_yaml.py`,
`output/__init__.py`, `output/plots.py`

The previous verification output (single integral table + bar chart) did not
scale beyond a few dozen demands: the table became unreadable and the heatmap
a wall of pixels.

**Three-level reporting:**

| Level | Content | Where |
|---|---|---|
| 1 — Summary block | n_total, n_fail, η_max, η_mean, η_p95, η_p99, governing name | Console (always) |
| 2 — Top-K table | K demands with highest η, sorted descending | Console (configurable K) |
| 3 — Full export | All results ranked + aggregate statistics | `verification_statistics.json` |

YAML configuration:

```yaml
output:
  verification_top_k: 10     # rows printed in console (0 = disable table)
  fiber_details_top_k: 5     # fiber CSV/plot only for top N demands
```

The heatmap auto-limits to `max(verification_top_k, 30)` bars with an
explicit `"(top N of M)"` title.

**Heatmap colour bug fixed.** Each verified bar previously received a fixed
`#4CAF50` colour, ignoring the per-η-type base colours (`eta_norm` → blue,
`eta_norm_beta` → violet, etc.).  Fix: verified bars use their type colour;
exceeded bars (`η > 1`) turn red with `///` hatch, irrespective of type.

**Public API** (`output/summary.py`):

| Function | Purpose |
|---|---|
| `governing_eta(result, result_type)` | Extract governing η from any result dict |
| `rank_results(results, result_type)` | Sort by η descending, inject `_rank` |
| `top_k_results(results, k, result_type)` | Top-K + tail count |
| `compute_summary_stats(results, result_type)` | n_total, n_fail, η_max, η_mean, η_p95, η_p99, η_median |
| `print_demand_summary`, `print_combination_summary`, `print_envelope_summary` | Tier-1 + Tier-2 console output |
| `build_verification_summary(d, c, e)` | Unified dict for API/GUI |
| `select_demands_for_fiber_details(results, top_k)` | Names of demands receiving post-processing |

---

## 4. Documentation maintenance

- `changelog.rst`: unified history through v0.3.3, replacing the partial
  record that stopped at v0.3.0.
- `README.md`: rewritten as a proper project README (replaces the v0.3.0
  patch-bundle instructions that had accumulated as the root README).
- `demand_verification.rst`: new section *"On-demand η without domain"*
  documenting `AnalysisEngine` and the when-to-use comparison with the
  full domain pipeline.
- `gensec_cli.rst`: updated for three subcommands: `run`, `analyze`, `plot`.

---

## 5. Repository hygiene

- `PERF_TUNING_v0_3_2.md` (root-level technical analysis document) is
  superseded by the structured entries in §2 above.  The file is retained
  for reference but should be moved to `docs/dev/` or removed before the
  next tagged release.
- `profile_gensec.py` harness: two stale comments corrected — the
  `"uses internal Mx-My cache"` annotation (false: cache is regenerated)
  and the `"NON batched: loop Python"` annotation on moment-curvature
  (stale since v0.3.2 vectorisation).  The dead `n_points_per_angle`
  parameter is removed from all callers.

---

## 6. Upgrade notes

### Breaking changes

None.  All public API additions are additive.

### Deprecations

- `plot_polar_ductility(..., n_points=...)` — the `n_points` keyword is a
  deprecated alias for `n_chi`; it emits a `DeprecationWarning` and will
  be removed in v0.4.0.  Migrate to `n_chi=`.
- `_scan_chi` — internal; retained as reference implementation until the
  regression suite confirms full equivalence with `_scan_chi_vectorized`.
  Scheduled for removal in v0.4.0.

### Known deferred items

| Item | Notes | Target |
|---|---|---|
| First-cracking precision on wide curvature window | Span-bound widens window ~4×; with fixed `n_points`, `chi_cr` may be overestimated by one step.  Two-segment scan (dense near *χ*=0) is the clean remedy. | v0.4.0 |
| `/api/inspect` integration test | Import succeeds; a real assertion on `properties.area` and `W_*` is missing. | v0.3.4 |
| Stale `@pytest.mark.xfail` on `test_analyze_biaxial_column_returns_valid_model` | Now XPASS; decorator should be removed. | v0.3.4 |
| `integrate_batch_with_tangent` | Analytical tangent in batch mode → halves Newton cost in `_vectorized_solve_N`. | v0.4.0 |
| Warm-start propagation in `generate_polar_curvature` | Pass `eps0_init` from angle *i* to *i*+1; useful only if fallback rate is high. | v0.4.0 |
