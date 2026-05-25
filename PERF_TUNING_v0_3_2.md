# GenSec — Performance tuning & memory optimisation

**Date**: 2026-05-17  
**Scope**: Default parameter tuning, memory budget correction, GUI N-slider fast path, computation presets  
**Status**: Ready for integration  

---

## 1. Context

GenSec v0.3.x introduced vectorised solvers (`_mega_batch_integrate`,
`_vectorized_solve_N`, vectorised M-χ scan) that reduced Python-loop
overhead by 14–43×.  However, the *default parameter counts* were
never re-calibrated after vectorisation — they still reflected the
pre-vectorisation regime where each configuration was expensive.

Additionally, the web GUI (`server.py` + React frontend) exposed a
severe memory issue: the N-slider called `generate_mx_my` on every
slider position, allocating ~350 MB of temporary arrays per call.

Numba JIT on constitutive kernels (`_concrete_stress_kernel`,
`_steel_stress_kernel`) was benchmarked and found to provide
negligible end-to-end speedup (~0.6% of wall-clock), because the
kernels already operate on large flat arrays where NumPy is efficient,
and the kernel evaluation is a small fraction of total run time.

---

## 2. Problem analysis

### 2.1 Edge template explosion

`_build_edge_template(n_points)` generates ~6.5 × n_points configs
per angle (branches 1–5 combined).  With the old defaults:

| Call site | n_angles | n_points | Configs/angle | Total configs |
|-----------|----------|----------|---------------|---------------|
| `generate_biaxial` (CLI) | 72 | 200 | ~1 300 | **~93 600** |
| `generate_biaxial` (API) | 36 | 50 | ~325 | ~11 700 |

The CLI default produced ~94k configs for a surface that, after
ConvexHull + lofting, used ~1 440 structured grid points.

### 2.2 Memory budget undercount

The comment in `_mega_batch_integrate` stated "~400 MB peak per
chunk", but `integrate_batch` keeps 4 simultaneous `(n, n_fibers)`
matrices alive at peak (strain `eb`, stress `sb`, force `fA`, and a
temporary for the weighted sum during moment computation):

```
Peak per chunk = 4 × n_configs × n_fibers × 8 bytes
```

With `max_configs = 50 000 000 / 2 000 = 25 000`:

```
4 × 25 000 × 2 000 × 8 = 1.6 GB   (not 400 MB)
```

### 2.3 N-slider memory spikes

`contour_at_N` called `generate_mx_my(n_angles=144, ...)` per slider
position.  Each call ran 144 × 50 = 7 200 vectorised Newton solves,
with a ~350 MB intermediate array peak — on every slider move.

### 2.4 Cache over-sizing

`SECTION_CACHE_SIZE = 32` held up to 32 parsed sessions in memory.
Each session with a computed 3D surface adds ~5 MB of persistent
arrays.  For a single-user GUI, 32 sessions is 160 MB of idle cache.

---

## 3. Changes

### 3.1 New file: `_presets.py`

Computation presets — `"rapid"`, `"standard"`, `"accurate"` — as a
simple dict constant with a `resolve_preset()` helper.

| Key | rapid | standard | accurate |
|-----|-------|----------|----------|
| `n_points` | 100 | **200** | 400 |
| `n_angles_3d` | 24 | **36** | 72 |
| `n_points_per_angle` | 50 | **80** | 200 |
| `n_chi` | 30 | **36** | 50 |
| `n_angles_mx_my` | 36 | **72** | 144 |
| `n_levels_3d` | 10 | **15** | 20 |
| `n_angles_polar` | 24 | **36** | 72 |
| `n_chi_polar` | 30 | **50** | 100 |

Bold = new default (previously the "accurate" column was the default).

YAML integration:

```yaml
output:
  preset: standard           # rapid | standard | accurate
  n_angles_mx_my: 144        # per-parameter override wins
```

The preset is resolved in the IO layer (`io_yaml.py`, `cli.py`,
`api.py`); solver methods keep individual keyword arguments.

### 3.2 `capacity.py` — default parameter changes

| Method | Parameter | Before | After |
|--------|-----------|--------|-------|
| `generate` | `n_points` | 300 | **200** |
| `generate_biaxial` | `n_angles` | 72 | **36** |
| `generate_biaxial` | `n_points_per_angle` | 200 | **80** |
| `generate_mx_my` | `n_chi` | 50 | **36** |
| `generate_moment_curvature` | `n_points` | 200 | **100** |

### 3.3 `capacity.py` — memory budget correction

```python
# BEFORE
max_configs = max(2000, 50_000_000 // max(n_fibers, 1))

# AFTER
max_configs = max(500, 12_000_000 // max(n_fibers, 1))
```

For 2 000 fibers: 25 000 → 6 000 configs/chunk.  True peak: ~400 MB
(4 matrices × ~96 MB each) instead of the previous ~1.6 GB.

### 3.4 `capacity.py` — new `NMDiagram.slice_mx_my_at_N()`

Static method that slices the pre-computed 3D point cloud at a fixed
N level using band filtering + 2D ConvexHull + angular resampling.

- **Cost**: < 1 ms (vs ~200 ms for `generate_mx_my`)
- **Memory**: zero solver allocations (operates on cached arrays)
- **Accuracy**: depends on point cloud density; sufficient for
  interactive visualisation, not for demand verification
- **Fallback**: returns `None` if fewer than 3 points in band;
  caller falls back to full Newton solve

### 3.5 `integrator.py` — `einsum` for moment summation

Replaced:
```python
Mx = (fA * self._ly_bulk[None, :]).sum(axis=1)
My = -(fA * self._lx_bulk[None, :]).sum(axis=1)
```

With:
```python
Mx = np.einsum('ij,j->i', fA, self._ly_bulk)
My = -np.einsum('ij,j->i', fA, self._lx_bulk)
```

Eliminates one `(n, n_fibers)` temporary per moment component.
Reduces peak from 4 to 3 simultaneously live matrices inside
`integrate_batch`.

### 3.6 `api.py` — cache reduction

```python
SECTION_CACHE_SIZE: int = 4    # was 32
```

### 3.7 `api.py` — `contour_at_N` fast path

Rewritten to first attempt `NMDiagram.slice_mx_my_at_N()` on the
cached 3D surface.  Falls back to full `generate_mx_my` only when
no 3D surface is available (uniaxial sections) or the slice yields
too few points.

Default `n_angles` reduced from 144 to 72.

### 3.8 `cli.py` — default changes

| Parameter | Before | After |
|-----------|--------|-------|
| `--n-points` | 400 | **200** |
| `n_angles` in `generate_biaxial` call | 72 | **36** |
| `n_points_per_angle` in biaxial call | `n_points // 2` | `n_points` |
| `n_angles_mx_my` default | 144 | **72** |
| `VerificationEngine` `n_points` | `n_points // 2` | `n_points` |

Removed dead `n_points_per_angle` argument from `generate_mx_my`
call.

### 3.9 `check.py` — `VerificationEngine` default

```python
self.n_angles = int(output_flags.get("n_angles_mx_my", 72))  # was 144
```

---

## 4. Impact summary

### 4.1 Computation volume (CLI, full biaxial run)

| Diagram | Before (configs) | After (configs) | Ratio |
|---------|-------------------|-----------------|-------|
| N-M ×2 | 2 × 400 = 800 | 2 × 200 = 400 | 2× |
| 3D surface | ~93 600 | ~18 700 | **5×** |
| Mx-My / demand | 7 200 | 2 592 | 2.8× |
| M-χ / N-level | 2 × 200 = 400 | 2 × 100 = 200 | 2× |
| **Total (typical)** | **~110 000** | **~24 000** | **~4.5×** |

### 4.2 Peak memory per chunk

| | Before | After |
|---|--------|-------|
| Chunk size (2k fibers) | 25 000 configs | 6 000 configs |
| True peak | ~1.6 GB | ~400 MB |
| With einsum | ~1.6 GB | ~300 MB |

### 4.3 N-slider (GUI)

| | Before | After |
|---|--------|-------|
| Method | `generate_mx_my` (full Newton) | `slice_mx_my_at_N` (hull slice) |
| Time | ~200 ms | **< 1 ms** |
| Temp memory | ~350 MB | **~0 MB** |

### 4.4 Idle cache

| | Before | After |
|---|--------|-------|
| Max sessions | 32 | 4 |
| Persistent memory | ~160 MB | ~20 MB |

---

## 5. Verification checklist

Before merging, the following should be verified:

- [ ] **3D surface visual quality at 36 angles / 80 pts**: run the
  critical elongated section (high Ac/As ratio) with both old and new
  defaults and compare lofted surfaces.  Watch for angular faceting.

- [ ] **Mx-My contour at n_chi=36**: test with N near domain
  boundaries (N ≈ N_Rd,c and N ≈ N_Rd,t) where few curvature
  magnitudes produce N-crossings.

- [ ] **M-χ event detection at n_points=100**: cracking/yielding
  points may shift by ±1–2 steps vs n_points=200.  Update regression
  test tolerances accordingly.

- [ ] **`slice_mx_my_at_N` vs `generate_mx_my`**: compare contour
  shapes at several N levels.  The slice will be slightly less precise
  (band interpolation vs exact Newton), which is acceptable for
  visualisation but not for η verification.

- [ ] **einsum equivalence**: run existing unit tests to confirm
  `N`, `Mx`, `My` outputs are bitwise identical.

- [ ] **Preset integration**: wire `resolve_preset()` into
  `io_yaml.py` YAML parser and `cli.py` argparse (add `--preset`
  flag).

---

## 6. Deferred items

- **Numba JIT on full `integrate_batch`**: would require rewriting
  as a free function with pre-extracted arrays.  Potential 5–10× on
  the hot path, but invasive and not justified until interactive
  real-time use cases (GUI with live slider updating the 3D surface)
  demand it.

- **Multiprocessing on mega-batch chunks**: chunks are independent;
  trivial to parallelise with `ProcessPoolExecutor`.  Deferred until
  wall-clock time exceeds interactive thresholds (~5 s).

- **Client-side debounce on N-slider**: with `slice_mx_my_at_N`
  costing < 1 ms, debouncing is no longer necessary for performance.
  Still recommended for network round-trip if the server is remote.

- **Boundary-tracking algorithm for Mx-My**: would replace the
  generate-then-hull approach with direct contour marching.  More
  efficient but significantly more complex.  Not justified at current
  scale.
