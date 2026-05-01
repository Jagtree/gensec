# GenSec v0.3.0 — Patch Bundle (v2, rebuilt)

This bundle contains all the source, documentation, and test changes
for v0.3.0.  The fixes are documented in detail in
``CHANGELOG_v0_3_0.md``.

> **Note on this bundle.**  This is the **v2 rebuild** of the v0.3.0
> patch bundle.  The first attempt accidentally truncated
> ``capacity.py`` by ~210 lines (lost ``include_pivot_a``, the cache
> attributes, ``eta_demand``, ``_ductility_at_direction``, and the
> two input-validation helpers).  The rebuild used your *current*
> ``capacity.py`` — the one you re-attached as a document — as the
> canonical baseline, and applied **only** the targeted A1 fix to it
> via a single surgical ``str_replace``.  Pre-/post-patch line-count
> deltas are documented under "Verification" below; no method or
> cache was lost on any file.

## File map

```
gensec_v030_patch/
├── CHANGELOG_v0_3_0.md           Release notes
├── yaml_reference_example.yaml   Updated reference YAML
├── gensec/                       Patched source files
│   ├── capacity.py               A1 (cloud symmetry for centred sections)
│   ├── integrator.py             B1, B2, B3, F1
│   ├── geometry.py               F1 (mat_indices_rebar)
│   ├── check.py                  K (7-metric architecture redesign) + I1 message
│   ├── io_yaml.py                K flag defaults (new metric set)
│   ├── export.py                 K (CSV + JSON columns for 7 metrics)
│   ├── cli.py                    I1 (is_biaxial requires both axes)
│   └── plots.py                  C1 (TODO comment, deferred to Tier 3)
├── docs/                         Patched documentation files
│   ├── yaml_reference.rst        Rewritten utilization-metrics section
│   └── demand_verification.rst   Full rewrite for 7-metric architecture
└── tests/
    ├── test_check.py             Updated for v0.3 (82 tests, all passing)
    ├── test_infrastructure.py    Updated for v0.3 (mechanical renames)
    └── test_v030_regressions.py  32 regression tests (all passing)
```

## Where each file goes in your tree

| Bundle path                       | Your tree path                                   |
|-----------------------------------|--------------------------------------------------|
| `gensec/capacity.py`              | `gensec/capacity.py`                             |
| `gensec/integrator.py`            | `gensec/integrator.py`                           |
| `gensec/geometry.py`              | `gensec/geometry/geometry.py` *(subpackage)*     |
| `gensec/check.py`                 | `gensec/check.py`                                |
| `gensec/io_yaml.py`               | `gensec/io_yaml.py`                              |
| `gensec/export.py`                | `gensec/export.py`                               |
| `gensec/cli.py`                   | `gensec/cli.py`                                  |
| `gensec/plots.py`                 | `gensec/plots.py`                                |
| `docs/yaml_reference.rst`         | `docs/yaml_reference.rst`                        |
| `docs/demand_verification.rst`    | `docs/demand_verification.rst`                   |
| `tests/test_v030_regressions.py`  | `gensec/test_v030_regressions.py`                |
| `yaml_reference_example.yaml`     | `examples/yaml_reference_example.yaml` *(or wherever you keep examples)* |

## Recommended apply procedure

1. **Create a feature branch.**

   ```bash
   git checkout -b v0.3.0-tier1-fixes
   ```

2. **Diff each file before overwriting.**  Crucially after the first
   bundle's truncation incident, you should diff *before* trusting
   any of these files:

   ```bash
   diff -u gensec/capacity.py /path/to/gensec_v030_patch/gensec/capacity.py
   diff -u gensec/integrator.py /path/to/gensec_v030_patch/gensec/integrator.py
   # ... etc.
   ```

   What you should expect to see (approximate):
   - `capacity.py`: only `_ultimate_strain_configs_1d` changed
     (about +32 lines: a longer docstring + the lever-arm fix).
     **Nothing else** should be different.
   - `integrator.py`: B2 (+~40), B1 dispatch (+~25), two new methods
     `_solve_uniaxial_y` + `_nr_uniaxial_y` (+~150), B3 (+~10), F1
     (+~30).  Total ≈ +254 lines.
   - `check.py`: complete redesign of the metric architecture (7
     metrics, anisotropy-corrected normalised space, Chebyshev
     LP, segment-to-hull distance helper).  Total ≈ +656 lines.
   - `geometry.py`: only `_setup_rebars` changed (+19).
   - `cli.py`: only the `is_biaxial` line + comment (+5).
   - `plots.py`: only a comment block in `plot_polar_ductility`
     (+12).  No logic touched.
   - `io_yaml.py`: new flag defaults for the 7-metric set (+3).
   - `export.py`: tuple updates to include the new metric columns
     (in-line edits, no significant size change).

   If any diff shows changes outside these areas, **stop and tell
   me**.

3. **Copy the files over** once you have approved every diff.

4. **Run the existing test suite** first.  Note that
   `check.py`'s API changed: pre-existing tests that referred to
   `eta_3D` need to be updated.  The `tests/test_check.py` and
   `tests/test_infrastructure.py` files in this bundle are the
   updated versions — drop them in over the originals.

   The Tier-1 fixes (A1, B1, B2, B3, F1) are backward-compatible
   by construction:

   - A1 produces bit-for-bit identical results for sections with
     `y_min = 0` (every standard primitive).
   - F1 produces bit-for-bit identical forces for single-bulk
     sections (the new `mat_indices_rebar` is all zeros, so every
     rebar group has `bulk_mat = primary bulk_material`).
   - B1 introduces a new fast-path branch that none of the existing
     tests exercise (they cover Mx-dominated demands).
   - B2 is backward-compatible for callers that do not pass the
     new `axis=` kwarg.

5. **Run the new regression tests.**

   ```bash
   python -m pytest tests/test_v030_regressions.py -v
   ```

   I have already run these in my sandbox: **32 tests, 32 pass.**
   If any fails on your side, that's likely an environment
   difference and I want to hear about it.

6. **Sanity-check on `test_exotic.yaml`** with the fine mesh
   (3220 fibres):

   - The N-Mx and N-My diagrams must be visibly mirror-symmetric.
   - The `verification_summary.csv` will now include four columns
     `eta_norm`, `eta_norm_beta`, `eta_norm_ray`, `eta_2D` (the
     latter two only if their flags are enabled).  Each metric
     answers a different geometric question — see
     `docs/demand_verification.rst`.

7. **Tag the release** when satisfied.

   ```bash
   git tag -a v0.3.0 -m "v0.3.0: Tier-1 fixes + 7-metric verification redesign"
   git push origin v0.3.0
   ```

   `setuptools_scm` picks up the version automatically.

## What's not in this bundle

By design.  See "Tier-3 deferrals" in the CHANGELOG for the items
left for v0.4.0.  Highlights:

- **C1 / C2** (polar ductility chi_max asymmetry + warm-start fix):
  marked with a code comment in `plots.py` pointing at the joint
  fix.  C1 alone would be cosmetic.
- **B4 / B5** (concrete-specific magic numbers in initial guess and
  Newton clamp): material-aware refactor.
- **G2** (linear interpolation in N-crossing for Mx-My contour):
  PCHIP upgrade with benchmark.
- **D1 / D2 / D3** (centred primitives, mechanical centroid,
  mandatory `r.x` for biaxial sections): API decisions that need
  your input before a patch makes sense.

## Verification record (what I ran in the sandbox)

End-to-end smoke test on `test_exotic_coarse.yaml` (496 fibres) —
all checks pass:

```
A1 - N-Mx range: [-2720.50, 2720.50]   M_max + M_min = 9.09e-13
B2 - _is_uniaxial(axis=x): False  _is_uniaxial(axis=y): False
B1 - _solve_uniaxial_y present: True  _nr_uniaxial_y present: True
F1 - mat_indices_rebar: (16,)
F1 - groups (rebar_mat, bulk_mat, n_idx): [('Steel', 'Concrete', 16)]
K  - do_norm=True, do_norm_beta=True, do_norm_ray=True, do_2D=True

 #    eta_norm  eta_norm_beta  eta_norm_ray  eta_2D
   1   0.5500   0.3507         0.1400        0.0901
   2   0.5431   0.3433         0.1345        0.0759
   3   0.2436   0.4396         0.3180        0.0790
   4   0.2354   0.4360         0.3136        0.0681
   5   0.5480   0.3723         0.1604        0.1259
```

The four point metrics give independent geometric perspectives on
the same demand.  No strict ordering between them holds: each is
the right answer to a different question (see
`docs/demand_verification.rst`).

CSV/JSON exports include all enabled metrics as columns.  Header
example:

```
name,N_kN,Mx_kNm,My_kNm,eta_norm,eta_norm_beta,eta_norm_ray,inside,verified
```

Regression test suite (`test_v030_regressions.py`):

```
Ran 32 tests in 4.08s
OK
```

Updated `test_check.py` (pre-existing tests, adapted to v0.3
architecture):

```
Ran 82 tests in 30.61s
OK
```

Final pre-/post-patch line counts:

```
capacity.py     1557 -> 1589   (+32)    [A1: only docstring + lever fix]
integrator.py   1024 -> 1278   (+254)   [B1+B2+B3+F1]
check.py         928 -> 1584   (+656)   [K: 7-metric redesign + I1 message]
geometry.py      621 ->  640   (+19)    [F1: mat_indices_rebar]
io_yaml.py       631 ->  634   (+3)     [K flag defaults]
export.py        488 ->  489   (+1)     [tuple updates for new metrics]
cli.py           709 ->  714   (+5)     [I1]
plots.py        1600 -> 1612   (+12)    [C1 comment only]
```
