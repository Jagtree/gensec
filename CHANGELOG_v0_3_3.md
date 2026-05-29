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


## Axial-limit robustness: equilibrium feasibility guard & Mx-My domain degeneracy

Correctness-focused release.  At the extremes of the resistance domain
the bending capacity is *identically* zero: at `N_Rd,min` (pure tension)
every bar yields uniformly and the section is fully cracked; at
`N_Rd,max` (squash) the strain field is uniform compression.  Two
defects made the analyses near these limits produce physically
meaningless, unstable output:

1. **Sampling on the extremes.**  `n_levels_mode: auto` used
   `linspace(N_min, N_max, n)`, so the first and last sampled axial
   levels coincided *exactly* with `N_Rd,max` / `N_Rd,min`, where
   `M ≡ 0`.

2. **Silent non-equilibrium states.**  Beyond a small curvature the
   target `N` becomes physically unreachable.  In that case
   `_solve_eps0_for_N` finds no sign-change bracket and returns
   `argmin |N(ε₀) − N_target|` — a state that does **not** satisfy axial
   equilibrium.  Both moment-curvature scanners and `generate_mx_my`
   recorded the moment of that state without checking the residual.

Symptoms (release-driving): saw-toothed M-χ curves at `N` near the
limits (moment ≈ `1e-7…1e-3` kN·m, i.e. floating-point zero amplified by
matplotlib autoscaling) and a spuriously **asymmetric** Mx-My contour at
`N_Rd,min`, formed by the convex hull of a noise cloud around the origin.

Two orthogonal fixes:

- **(A)** the `auto` sampler now excludes the domain extremes;
- **(B)** an **axial-equilibrium feasibility guard** masks unreachable
  states as `NaN` (M-χ) and flags collapsed interaction domains
  (Mx-My).  (B) is normative-agnostic — it only enforces that no
  reported result violates `ΣN = N_target`.

The Mx-My asymmetry is resolved **not** by symmetrising the noise but by
recognising that the domain has collapsed to a point and refusing to
draw a hull through floating-point dust.

| Defect (pre-patch)                                   | Location                          | Fix |
|------------------------------------------------------|-----------------------------------|-----|
| `auto` levels land on `N_Rd,min` / `N_Rd,max`        | `cli.py` ~569                     | A   |
| M-χ records non-equilibrium moments                  | `_scan_chi` ~1562                 | B   |
| M-χ (vectorized) records non-equilibrium moments     | `_scan_chi_vectorized` ~1708      | B   |
| Mx-My convergence filter too loose (`1e3` N)         | `generate_mx_my` ~966             | B   |
| Mx-My hull built on collapsed-domain noise           | `generate_mx_my` ~985             | B   |
| Mx-My legend overlaps the x-axis label               | `plot_mx_my_diagram` ~581         | cosmetic |

---

## Files modified

### 1. `src/gensec/solver/capacity.py`

**Two module constants, one helper, three guarded call-sites.**

#### A. Module-level feasibility constants + helper

Insert near the top of the module (above the `NMDiagram` class).

```python
# --- Equilibrium feasibility guard ---------------------------------
# A (eps0, chi) state is physically valid only if the axial residual
#   |N(eps0, chi) - N_target|
# stays below this tolerance. Beyond it the target N is unreachable at
# that curvature: the solver returns the closest non-equilibrium state,
# whose moment is meaningless and must be discarded (set to NaN).
# Absolute + relative form: the 10 N floor dominates as N_target -> 0
# (near-pure bending); the relative term avoids false positives at high
# |N_target|, where 10 N is below the Newton noise.
_FEAS_ABS_TOL = 10.0      # [N]
_FEAS_REL_TOL = 1.0e-4    # [-]

def _feas_tol(N_target):
    """Axial-equilibrium feasibility tolerance [N] for a given target."""
    return max(_FEAS_ABS_TOL, _FEAS_REL_TOL * abs(N_target))

# --- Mx-My domain degeneracy ---------------------------------------
# The interaction domain is declared collapsed (axial limit, M_Rd ~ 0)
# when its outer radius falls below this fraction of a section plastic
# moment scale  M_ref = f_cd * B * H**2 / 4.  The 3+ order-of-magnitude
# gap between a real domain (~1e8 N*mm) and limit noise (<=1e3 N*mm)
# makes the exact coefficient non-critical.
_DOMAIN_DEGEN_FRAC = 1.0e-3
```

`_solve_eps0_for_N` is **unchanged**: its `argmin` fallback is still the
right behaviour for genuinely converging problems; the callers now
decide whether the returned state is admissible.

#### B. `_scan_chi` — scalar guard

After `N, Mx, My = sv.integrate(eps0, chi_x, chi_y)` (~line 1562),
**before** `M = ...`:

```python
if abs(N - N_fixed) > _feas_tol(N_fixed):
    # Target N unreachable at this curvature: drop the point.
    Ms[k] = np.nan
    eps_mins[k] = np.nan
    eps_maxs[k] = np.nan
    continue          # keep previous eps0_guess; skip event detection
```

The `continue` preserves the previous warm-start (the infeasible ε₀ is
discarded) and prevents cracking/yield/ultimate detection from running
on a non-equilibrium state.

#### C. `_scan_chi_vectorized` — vectorized guard

After the scalar fallback loop (~line 1708), **before** the moment
extraction:

```python
# --- Feasibility mask: drop points failing axial equilibrium ---
feasible = np.abs(N_arr - N_fixed) <= _feas_tol(N_fixed)
Mx_arr = np.where(feasible, Mx_arr, np.nan)
My_arr = np.where(feasible, My_arr, np.nan)
```

Then fold the mask into the event-detection guard:

```python
nonzero = (np.abs(chis) > 0) & feasible
```

(masks 6a/6b/6c already key off `nonzero`).  The `NaN` moments break the
M-χ polyline at the last feasible χ automatically in matplotlib.

#### D. `generate_mx_my` — tolerance + domain degeneracy

Tighten the solve/filter tolerance (~lines 966–972):

```python
ftol = _feas_tol(N_fixed)
_, N_arr, Mx_arr, My_arr = self._vectorized_solve_N(
    N_fixed, all_chi_x, all_chi_y, tol=ftol)

conv = np.abs(N_arr - N_fixed) <= ftol
Mx_conv = Mx_arr[conv]
My_conv = My_arr[conv]
```

Add collapsed-domain detection immediately after the existing
`len(Mx_conv) < 3` guard (~line 985):

```python
# Section plastic moment scale; f_cd from the bulk constitutive law.
sec = self.solver.sec
f_cd = abs(self._fcd_bulk)
M_ref = f_cd * sec.B * sec.H**2 / 4.0
R = float(np.hypot(Mx_conv, My_conv).max())
if R < _DOMAIN_DEGEN_FRAC * M_ref:
    Mxa = np.full(n_angles, np.nan)
    Mya = np.full(n_angles, np.nan)
    warnings.warn(
        f"generate_mx_my: collapsed domain at "
        f"N={N_fixed / 1e3:.1f} kN (R={R:.2e} N*mm "
        f"< {_DOMAIN_DEGEN_FRAC * M_ref:.2e}); "
        f"M_Rd ≈ 0 at axial limit.",
        RuntimeWarning, stacklevel=2)
    return {"Mx": Mxa, "My": Mya,
            "Mx_kNm": Mxa / 1e6, "My_kNm": Mya / 1e6,
            "N_fixed_kN": N_fixed / 1e3, "degenerate": True}
```

Add `"degenerate": False` to the normal `return` for a stable schema.

**New attribute required:** `self._fcd_bulk` — the bulk material design
compressive strength (positive), cached in `NMDiagram.__init__` next to
`self._emb`.  If a material type exposes no `f_cd`, fall back to a
documented elastic-based estimate (does not affect the order-of-magnitude
degeneracy test).

---

### 2. `src/gensec/cli.py`

**One call-site update (behavioural change in `auto` mode).**

`n_levels_mode: auto` now samples the *open* interval — the domain
extremes `N_Rd,min` / `N_Rd,max` (where `M ≡ 0`) are excluded.  Users
who want the limits request them explicitly via `n_levels_mode:
explicit`.

Lines ~569–572:

```python
# BEFORE
N_levels_analysis = sorted(
    np.linspace(N_min_d, N_max_d, n_levels_count).tolist())

# AFTER
# n_levels_count interior values; endpoints (N_Rd,min / N_Rd,max,
# where M ≡ 0) deliberately excluded.
N_levels_analysis = sorted(
    np.linspace(N_min_d, N_max_d, n_levels_count + 2)[1:-1].tolist())
```

Update the adjacent `print` to state that the values are interior.

---

### 3. `src/gensec/output/plots.py`

**One degenerate-case branch, one cosmetic fix.**

#### A. `plot_mx_my_diagram` — collapsed-domain annotation

At the top of the function, after `N_fixed = mx_my_data.get(...)`
(~line 558):

```python
if mx_my_data.get("degenerate") or np.all(np.isnan(Mx)):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.text(0.5, 0.5,
            f"Domain collapsed: $M_{{Rd}} \\approx 0$\n"
            f"(axial limit, N = {N_fixed:.0f} kN)",
            ha='center', va='center', transform=ax.transAxes,
            fontsize=12)
    ax.set_axis_off()
    return fig
```

#### B. `plot_mx_my_diagram` — legend no longer overlaps the x-label

The single-entry legend was anchored below the axes (`bbox_to_anchor=
(0.5, -0.08)`), overlapping the `Mx [kN·m]` label.  Replace the block
(~lines 581–586):

```python
# BEFORE
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
          fontsize=9, ncol=3, borderaxespad=0)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')
fig.tight_layout()
fig.subplots_adjust(bottom=0.15)
return fig

# AFTER
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')
fig.tight_layout()
return fig
```

---

### 4. `tests/test_v030_regressions.py`

**One new test class: `TestB4_AxialLimitRobustness`** (5 tests).

- `test_mx_my_degenerate_at_squash` / `_at_tension` — `generate_mx_my`
  at the two axial limits must return `degenerate=True`.
- `test_mx_my_nondegenerate_interior` — interior `N` yields a finite,
  non-degenerate contour with non-zero outer radius.
- `test_mc_no_nan_before_ultimate_interior` — feasible pre-ultimate M-χ
  branch contains no `NaN` (defensive regression: the guard must not
  over-mask valid interior steps).
- `test_mc_guard_fires_near_squash` — at `0.995·N_Rd,max` the guard must
  `NaN`-mask the unreachable portion of the curve.

Discriminators (fail pre-patch, pass post-patch): the two `degenerate`
tests and `test_mc_guard_fires_near_squash`.

---

## Files NOT modified

| File              | Reason                                                      |
|-------------------|-------------------------------------------------------------|
| `integrator.py`   | `integrate`, `integrate_batch`, `strain_field` reused as-is |
| `check.py`        | η metrics consume capacity output; degenerate domains now flagged upstream |
| `io_yaml.py`      | No new YAML keys (constants are hardcoded by design)        |
| `geometry.py`     | Not involved                                                |
| `materials/*`     | Not involved (apart from reading `f_cd`, already exposed)   |
| `api.py`          | Inherits the fixes through the solver/plot layers           |

---

## Testing plan

### New tests (`test_v030_regressions.py::TestB4_AxialLimitRobustness`)

```
test_mx_my_degenerate_at_squash
test_mx_my_degenerate_at_tension
test_mx_my_nondegenerate_interior
test_mc_no_nan_before_ultimate_interior
test_mc_guard_fires_near_squash
```

### Manual verification

Re-run `example_biaxial_column.yaml`:

- `auto` levels no longer include `+1260` / `−5285` kN; the three
  interior levels produce clean M-χ curves and finite Mx-My contours.
- Forcing `n_levels_mode: explicit` with `[1260]` reproduces the
  collapsed-domain annotation instead of the asymmetric blob.

---

## Follow-up items (not in v0.3.3)

| Item | Description | Target |
|------|-------------|--------|
| Expose `_fcd_bulk` uniformly | Guarantee every material type provides a design compressive strength, or a documented surrogate | v0.4.0 |
| Feasibility guard in `generate_biaxial` | Same residual check on the 3-D surface cloud, for consistency with `generate_mx_my` | v0.3.4 |
| Inherited from v0.3.2 | Remove `_scan_chi`; drop the `n_points` alias in `plot_polar_ductility` after CLI/API migration | v0.3.3/0.3.4 |