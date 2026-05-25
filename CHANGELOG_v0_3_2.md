# CHANGELOG — GenSec v0.3.2

## Vectorized moment-curvature and polar ultimate-curvature computation

Performance-focused release: replaces the sequential Python loop in the
moment-curvature scanner and the double-nested loop in the polar
ultimate-curvature diagram with batch-vectorized solves, reusing the
`_vectorized_solve_N` + `integrate_batch` pattern already proven in
`generate_mx_my` and `generate_biaxial`.

Expected speed-ups (vs v0.3.1):

| Method                         | Before (scalar ops)         | After (batch calls)        | Est. speed-up |
|--------------------------------|-----------------------------|----------------------------|---------------|
| `generate_moment_curvature`    | ~4 800 per diagram          | ~60 batch + fallback       | 10–30×        |
| `plot_polar_ductility`         | ~28 800 per diagram         | ~40 batch + per-angle scan | 20–60×        |

---

## Files modified

### 1. `src/gensec/solver/capacity.py`

**Three additions, one two-line edit.**

#### A. New method: `NMDiagram._scan_chi_vectorized`

Insert **after** `_scan_chi` (after line ~1490).

Drop-in replacement for `_scan_chi` — same signature, same return dict.
Three-phase algorithm:

1. `_vectorized_solve_N` solves all `n_points` ε₀ in parallel
   (`n_iter=20`, `tol=10.0`).
2. Scalar `_solve_eps0_for_N` fallback on non-converged points
   (typically < 5%, near χ ≈ 0).
3. Batch strain-field matrix `(n_points, n_fibers)` + vectorized
   event detection via `np.argmax` on boolean masks.

No new dependencies.  Memory: `n_points × n_fibers × 8 B` for the
strain matrix (≈ 16 MB at 200 × 10 000 — well within budget).

#### B. New method: `NMDiagram.generate_polar_curvature`

Insert **after** `generate_moment_curvature` (or after
`_scan_chi_vectorized`).

New **public** method.  Moves the computational logic of the polar
ultimate-curvature diagram from `plots.py` into the solver layer,
matching the data/plot separation pattern used everywhere else
(`generate_biaxial` → `plot_3d_surface`, `generate_mx_my` →
`plot_mx_my_diagram`, etc.).

Signature:

```python
def generate_polar_curvature(self, N_fixed, n_angles=72, n_chi=100):
    ...
    return {
        "thetas": ...,     # (n_angles,)  rad
        "chi_u": ...,      # (n_angles,)  1/mm
        "chi_u_km": ...,   # (n_angles,)  1/km
        "eps0_u": ...,     # (n_angles,)  ε₀ at ultimate
        "N_fixed_kN": ..., # float
    }
```

Algorithm:

1. `_chi_max_for_direction` per angle (reuses existing method).
2. Flatten all `(n_angles × n_chi)` configs into 1-D arrays.
3. Single `_vectorized_solve_N` call + scalar fallback.
4. Per-angle strain-field loop for ultimate detection (memory:
   `n_chi × n_fibers` per chunk, not `n_total × n_fibers`).

#### C. Edit: `NMDiagram.generate_moment_curvature`

Lines ~1326–1329 — replace two calls:

```python
# BEFORE
results_pos = self._scan_chi(
    N_fixed, 0, chi_max, n_points, direction, eps_cr=eps_cr)
results_neg = self._scan_chi(
    N_fixed, 0, -chi_max, n_points, direction, eps_cr=eps_cr)

# AFTER
results_pos = self._scan_chi_vectorized(
    N_fixed, 0, chi_max, n_points, direction, eps_cr=eps_cr)
results_neg = self._scan_chi_vectorized(
    N_fixed, 0, -chi_max, n_points, direction, eps_cr=eps_cr)
```

No changes to the return dict, docstring, or public API.

#### D. Preserve: `_scan_chi`, `_solve_eps0_for_N`

Retained as-is.  `_scan_chi` is no longer called internally but
remains available as a reference/fallback until regression tests
confirm equivalence.  `_solve_eps0_for_N` is still used by the
scalar fallback in both new methods.

---

### 2. `src/gensec/output/plots.py`

**One function refactored.**

#### `plot_polar_ductility` — delegate to `generate_polar_curvature`

Replace the entire function body (lines ~1172–1305).  The new version:

1. Calls `nm_gen.generate_polar_curvature(N_fixed, n_angles, n_chi)`.
2. Applies the same outlier rejection + smoothing on `chi_u_km`.
3. Renders the polar plot (unchanged matplotlib code).

**Signature change:**

```python
# BEFORE
def plot_polar_ductility(nm_gen, N_fixed, n_angles=72,
                         n_points=400, title=""):

# AFTER
def plot_polar_ductility(nm_gen, N_fixed, n_angles=72,
                         n_chi=100, title="",
                         n_points=None):  # deprecated alias
```

`n_points` kept as a deprecated keyword that maps to `n_chi` with a
`DeprecationWarning`, so existing callers (`cli.py`, `api.py`) work
without changes until migrated.

---

### 3. `src/gensec/cli.py`

**One call-site update (non-breaking, deferred).**

Line ~674:

```python
# BEFORE
fig = plot_polar_ductility(
    nm_gen, N_fixed, n_angles=n_angles_mx_my,
    n_points=args.n_points)

# AFTER (when ready to drop the n_points alias)
fig = plot_polar_ductility(
    nm_gen, N_fixed, n_angles=n_angles_mx_my,
    n_chi=args.n_points)
```

Works unchanged via the `n_points` deprecation shim in `plots.py`.

---

### 4. `src/gensec/api.py`

**One call-site update (non-breaking, deferred).**

Line ~590:

```python
# BEFORE
fig = plot_polar_ductility(
    nm_gen, N_fixed=N_kN * 1e3,
    n_angles=int(kwargs.get("n_angles", 144)),
    n_points=int(kwargs.get("n_points", 400)))

# AFTER (when ready to drop the n_points alias)
fig = plot_polar_ductility(
    nm_gen, N_fixed=N_kN * 1e3,
    n_angles=int(kwargs.get("n_angles", 144)),
    n_chi=int(kwargs.get("n_chi", 100)))
```

Works unchanged via the `n_points` deprecation shim.

---

### 5. `src/gensec/output/export.py`

**One new function (optional, recommended).**

```python
def export_polar_curvature_json(polar_data, filepath):
    """Export polar ultimate-curvature data to JSON."""
    ...

def export_polar_curvature_csv(polar_data, filepath):
    """Export polar ultimate-curvature data to CSV."""
    ...
```

Follows the same pattern as `export_moment_curvature_json` /
`export_moment_curvature_csv`.  The CLI can then export polar
data before the plot, consistent with the data-first pipeline.

---

## Files NOT modified

| File               | Reason                                                    |
|--------------------|-----------------------------------------------------------|
| `integrator.py`    | No changes — `integrate_batch`, `strain_field` reused as-is |
| `check.py`         | No interaction with moment-curvature                      |
| `io_yaml.py`       | `generate_polar_ductility` flag already exists            |
| `geometry.py`      | Not involved                                              |
| `materials/*`      | Not involved                                              |

---

## Testing plan

### Regression: `_scan_chi_vectorized` vs `_scan_chi`

Run both on the same section/N/direction and verify:

- `chi` arrays are identical (same `np.linspace`).
- `M` arrays agree within 1 N·mm (tolerance from scalar Newton).
- Event points (cracking, yield, ultimate) are at the same index or
  ±1 step (discretisation-limited).
- `eps_min`, `eps_max` arrays agree within `1e-8` (float precision
  of the batch strain field vs scalar `strain_field`).

### Regression: `generate_polar_curvature` vs old `plot_polar_ductility`

Extract `chi_ultimates` from the old sequential scan and compare
against `polar["chi_u"]`.  Tolerance: 5% relative (different `n_chi`
default, no warm-start).

### New tests

```
test_scan_chi_vectorized_matches_scalar
test_scan_chi_vectorized_cracking_detection
test_scan_chi_vectorized_yield_detection
test_scan_chi_vectorized_ultimate_detection
test_generate_polar_curvature_keys
test_generate_polar_curvature_nonzero
test_generate_polar_curvature_symmetric_section
test_polar_curvature_n_chi_validation
```

### Performance benchmark

Add a case to `profile_gensec.py`:

```python
# Moment-curvature: old vs new
timings["mc_old"] = time_stage(
    "M-chi (scalar _scan_chi)", ...)
timings["mc_new"] = time_stage(
    "M-chi (vectorized)", ...)

# Polar: old vs new
timings["polar_old"] = time_stage(
    "Polar (sequential)", ...)
timings["polar_new"] = time_stage(
    "Polar (vectorized)", ...)
```

---

## Follow-up items (not in v0.3.2)

| Item | Description | Target |
|------|-------------|--------|
| `integrate_batch_with_tangent` | Analytical tangent in batch mode → halves Newton cost in `_vectorized_solve_N` | v0.4.0 |
| Remove `_scan_chi` | After regression suite confirms equivalence | v0.3.3 |
| Remove `n_points` alias in `plot_polar_ductility` | After CLI + API migrated to `n_chi` | v0.3.3 |
| Warm-start propagation for polar | Pass `eps0_init` from angle i to angle i+1 in `generate_polar_curvature` — useful only if fallback rate is high | v0.4.0 |
| JSON/CSV export for polar data | `export_polar_curvature_json`, `_csv` + CLI integration | v0.3.2 or v0.3.3 |
