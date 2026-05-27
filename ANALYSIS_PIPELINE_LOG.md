# GenSec — Analysis Pipeline Implementation Log

> **Chat session**: 2026-05-27
> **Scope**: Force decomposition mode + on-demand η + documentation
> **Status**: Fully implemented — code, tests, docs all applied

---

## 1. Motivation

GenSec's existing workflow (`gensec run`) always generates the full
resistance domain (NMDiagram → ConvexHull → VerificationEngine)
before checking demands.  This is wasteful when the goal is simply:

- **Force decomposition**: given (N, Mx, My), how much force does
  each material carry?
- **Quick η check**: is a demand inside the domain? (without
  producing the entire domain surface)

The new analysis pipeline bypasses domain generation entirely,
operating directly on `FiberSolver.solve_equilibrium`.

---

## 2. Architecture

```
                    ┌─────────────┐
                    │ FiberSolver │  (Layer 1 — unchanged)
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
     ┌────────┴────────┐     ┌──────────┴──────────┐
     │   NMDiagram     │     │  AnalysisEngine     │  ← NEW
     │   (capacity.py) │     │  (analysis.py)      │
     └────────┬────────┘     └──────────┬──────────┘
              │                         │
     ┌────────┴────────┐     outputs:   │
     │ VerificationEng │     • force decomposition per material
     │   (check.py)    │     • on-demand η (ray-bisection)
     └─────────────────┘     • staged analysis
```

---

## 3. Files created (new)

### `src/gensec/solver/analysis.py`

Main new module. Contains:

- **`AnalysisEngine`** class:
  - `__init__(solver)` → wraps FiberSolver, builds material name map
  - `analyze(N, Mx, My)` → solve + decompose for a single demand
  - `analyze_demands(demands)` → batch processing
  - `analyze_combinations(combinations, demand_db)` → simple + staged
  - `compute_eta(N, Mx, My, base_N, base_Mx, base_My)` → on-demand η
  - `_is_feasible(N, Mx, My)` → convergence + strain-limit check
  - `_decompose(eps0, chi_x, chi_y)` → per-material force aggregation
  - `_analyze_staged(name, stages, demand_db)` → staged support

- **`_resolve_components(components, demand_db)`** — standalone helper
  (same logic as `VerificationEngine.resolve_components`, duplicated
  to avoid import dependency on `check.py`)

- **`_build_material_name_map(solver)`** — builds `{id(mat): name}`
  from section materials using `Material.name` attribute

**Output structure of `analyze()`**:

```python
{
    "converged": True,
    "strain_state": {"eps0": ..., "chi_x": ..., "chi_y": ...},
    "total": {"N": ..., "Mx": ..., "My": ...},
    "strains_ok": True,
    "components": [
        {
            "type": "bulk",
            "material_name": "C25/30",
            "zone": 0,
            "N": ..., "Mx": ..., "My": ...,
            "N_kN": ..., "Mx_kNm": ..., "My_kNm": ...,
        },
        {
            "type": "rebar",
            "material_name": "B450C",
            "N": ..., "Mx": ..., "My": ...,
            "N_kN": ..., "Mx_kNm": ..., "My_kNm": ...,
            "layers": [
                {"index": 0, "x": ..., "y": ..., "A": ...,
                 "eps": ..., "sigma_gross": ..., "sigma_net": ...,
                 "F_net_kN": ...},
                ...
            ],
        },
    ],
}
```

### `tests/test_analysis.py`

15 tests across 5 test classes:

| Class | Tests | Covers |
|---|---|---|
| `TestDecomposeForces` | 8 | Sum consistency, integrate match, types, names, layers, pure axial, unconverged, strains_ok |
| `TestStrainsWithinLimits` | 3 | Zero strain, extreme strain, moderate strain |
| `TestOnDemandEta` | 5 | Origin η=0, inside <1, outside >1, monotonicity, staged base |
| `TestAnalyzeDemandsBatch` | 3 | Count, names, units |
| `TestAnalyzeCombinations` | 3 | Simple, staged, missing ref |

---

## 4. Files modified (patches applied)

### `src/gensec/materials/base.py`

**Change**: Added class attribute `name: str = ""` to `Material`
(after docstring, before `@property eps_min`).

**Reason**: Materials need human-readable names for decomposition
output. The YAML loader sets this after construction.

### `src/gensec/io_yaml.py`

**Change**: Line ~293, replaced:
```python
materials[mat_name] = _build_material(mat_name, mat_spec)
```
with:
```python
mat = _build_material(mat_name, mat_spec)
mat.name = mat_name
materials[mat_name] = mat
```

**Reason**: Propagates the YAML key to `Material.name`. Works for
all material types including early-return paths in
`_build_material` (concrete_ec2_gen1), since name is set *after*
the function returns.

### `src/gensec/solver/integrator.py`

**Change**: Added `strains_within_limits(eps0, chi_x, chi_y)`
method to `FiberSolver`, after `get_fiber_results()` (end of file).

**Reason**: The on-demand η bisection needs to check if a strain
field is admissible (all fibers within their material's
`[eps_min, eps_max]`). This is what distinguishes interior from
exterior points of the resistance domain.

**Logic**: Iterates `_bulk_groups` and `_rebar_groups`, checks
`np.any(e < mat.eps_min)` or `np.any(e > mat.eps_max)`.

### `src/gensec/solver/__init__.py`

**Change**: Added `from .analysis import AnalysisEngine` and
`"AnalysisEngine"` to `__all__`.

### `src/gensec/cli.py`

**Change**:
- Added `gensec analyze` subparser (after `gensec plot`) with
  `input_file`, `--output-dir`, `--eta` arguments.
- Added `elif args.command == "analyze": _analyze(args)` dispatch.
- Added `_analyze()`, `_print_analysis_table()`, and
  `_print_staged_analysis()` functions at end of file.

### `src/gensec/output/export.py`

**Change**: Appended `export_analysis_json()` and
`export_analysis_csv()` functions at end of file.

**JSON structure**: `{"demands": [...], "combinations": [...]}` with
full decomposition including nested components and layers.

---

## 5. YAML interface changes

### No new YAML keys required

The `gensec analyze` command uses the **same input YAML** as
`gensec run`. It reads `materials`, `section`, `demands`, and
`combinations`, and ignores the `output` block entirely.

### Existing output flags (reference)

These flags exist in `_parse_output_flags()` and control what
`gensec run` generates. They are **irrelevant** for
`gensec analyze`.

| Flag | Default | Controls |
|---|---|---|
| `eta_norm` | `true` | α distance metric |
| `eta_norm_beta` | `true` | Composite ratio metric |
| `eta_norm_ray` | `false` | Ray from origin |
| `eta_2D` | `false` | 2D ray at fixed N |
| `eta_path_norm_ray` | `false` | Staged 3D ray |
| `eta_path_norm_beta` | `false` | Staged composite ratio |
| `eta_path_2D` | `false` | Staged 2D ray |
| `delta_N_tol` | `0.03` | ΔN threshold for path_2D |
| `generate_mx_my` | `false` | Mx-My contour diagrams |
| `generate_3d_surface` | `false` | 3D CSV/JSON export |
| `n_angles_mx_my` | `144` | Angular resolution |
| `generate_moment_curvature` | **`true`** | **M-χ diagrams (heavy!)** |
| `generate_polar_ductility` | **`true`** | **Polar ductility (heavy!)** |
| `generate_3d_moment_curvature` | **`true`** | **3D M-χ surface (heavy!)** |

### What `gensec run` always computes (not flagged)

- N-Mx diagram → needed by VerificationEngine
- N-My diagram (biaxial) → needed by VerificationEngine
- 3D surface (biaxial) → ConvexHull for all η types

These cannot be skipped in `gensec run`. To skip them entirely,
use `gensec analyze`.

### Minimal fast `gensec run` config

```yaml
output:
  generate_moment_curvature: false
  generate_polar_ductility: false
  generate_3d_moment_curvature: false
```

Estimated speedup: **3–10×** vs default.

---

## 6. Documentation changes (RST — all applied)

### `docs/user_guide/yaml_reference.rst`

- **New note** at top: explains that `gensec analyze` ignores the
  `output` block.
- **New subsection "Mandatory vs optional generation"**: table
  showing what can/cannot be skipped in `gensec run`.
- **New subsection "Minimal fast verification"**: YAML example
  disabling heavy generators.
- **New subsection "Force decomposition only"**: CLI example for
  `gensec analyze`.

### `docs/architecture/architecture_solver.rst`

- **"Overview: three layers"** → **"Overview: two pipelines"**:
  updated Mermaid diagram showing domain pipeline vs analysis
  pipeline side by side, both building on FiberSolver.
- **New section "Analysis pipeline: AnalysisEngine"** after
  "Staged combinations", covering force decomposition, on-demand η,
  material naming.

### `docs/theory/demand_verification.rst`

- **New section "On-demand η (without domain generation)"** at the
  end: explains ray-bisection, feasibility criterion, convexity
  guarantee, cost comparison, when-to-use table.

### `docs/user_guide/quickstart.rst`

- **New intro paragraph**: mentions both CLI commands.
- **New subsection "Lightweight analysis (force decomposition)"**
  in CLI workflow: shows `gensec analyze` with example output.
- **New subsection "Force decomposition (Python API)"** at the
  end: 6-line example with AnalysisEngine.

### `docs/api/gensec.solver.rst`

- **New section "Analysis engine"** with `automodule` directive
  for `gensec.solver.analysis`.

### `docs/api/gensec.cli.rst`

- **Rewritten** to document all three subcommands: `run`,
  `analyze`, `plot`.

---

## 7. Design decisions and rationale

### Why `_resolve_components` is duplicated (not imported from check.py)

`analysis.py` intentionally does not import from `check.py` to
avoid pulling in `scipy.spatial.ConvexHull` and `scipy.optimize`
when only force decomposition is needed. The function is 15 lines
— the duplication cost is negligible vs. the dependency cost.

### Why `strains_within_limits` is on FiberSolver (not AnalysisEngine)

It's a pure property of the strain field + materials — no
analysis-level logic. Other code (e.g., future capacity.py
optimizations) may need it independently.

### Why `decompose_forces` is on AnalysisEngine (not FiberSolver)

The decomposition accesses `_bulk_groups` and `_rebar_groups`
(private attributes of FiberSolver) and needs the material name
map. Putting it on FiberSolver would either expose the name map
as a FiberSolver concern or make the output less informative.
AnalysisEngine is the natural owner.

### On-demand η is equivalent to `eta_norm_ray` (not `eta_norm`)

The ray-bisection produces a scale factor along a ray from the
origin — this is geometrically identical to `eta_norm_ray` in the
domain pipeline. The distance-based metrics (`eta_norm`,
`eta_norm_beta`) require the full ConvexHull and are not available
in the analysis pipeline. This is documented.

### Exponential scan before bisection

The boundary point can be at any distance from the demand. Linear
scan is too slow for large domains; pure bisection needs a bracket.
Exponential scan (t = 1, 2, 4, 8, ...) finds the bracket in
O(log t_max) steps, then bisection refines in 30 iterations.
Total: ~35–45 equilibrium solves worst case.

---

## 8. Known limitations / future work

- **`_resolve_components` duplication**: could be refactored to a
  shared `solver/_resolve.py` if more functions need it.
- **No normalised η in analysis pipeline**: `eta_norm` and
  `eta_norm_beta` require the hull. If needed, the user should
  use `gensec run`.
- **Envelopes not yet supported in analyze**: the analysis pipeline
  handles demands and combinations but not envelopes. Low priority
  — envelopes are a reporting convenience, not a computation.
- **No analysis-specific plots**: the analysis pipeline exports
  JSON/CSV but doesn't produce plots. Could add stress-field
  plots per combination in a future pass.
- **Material.name on dataclass subclasses**: `Concrete` and `Steel`
  are `@dataclass`. Setting `name` post-construction works because
  the dataclasses are not frozen, but `name` is not visible in
  `repr()` or `__eq__()`. If this becomes an issue, add
  `name: str = ""` as a dataclass field (would change constructor
  signature — backward-compatible if it's the last field with a
  default).

---

## 9. Testing checklist

- [ ] `test_analysis.py` — all 15 tests pass
- [ ] Existing `test_solver_biaxial.py` — no regressions
- [ ] Existing `test_check.py` — no regressions
- [ ] `test_infrastructure.py` — YAML loader still works, materials
      now have `.name` attribute
- [ ] CLI: `gensec analyze examples/example_tee.yaml --eta`
      produces `analysis_results.json`
- [ ] CLI: `gensec run examples/example_tee.yaml` unchanged

Run with: `uv run pytest tests/`

---

## 10. Integration steps (all completed 2026-05-27)

1. ✅ Added `name: str = ""` to `Material` in `materials/base.py`
2. ✅ Set `mat.name = mat_name` in `io_yaml.py` (line ~293)
3. ✅ Added `strains_within_limits()` to `FiberSolver` in
   `solver/integrator.py`
4. ✅ Created `analysis.py` at `solver/analysis.py`
5. ✅ Added `AnalysisEngine` to `solver/__init__.py` exports
6. ✅ Added `_analyze()` and printers to `cli.py`
7. ✅ Added `export_analysis_json/csv` to `output/export.py`
8. ✅ Created `test_analysis.py` in `tests/`
9. ✅ Applied RST patches to documentation (all 6 files)
