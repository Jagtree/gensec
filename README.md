# GenSec

**Fiber-based cross-section analysis for reinforced concrete and composite
structural members under combined axial force and biaxial bending.**

GenSec computes *N*–*M*x–*M*y resistance domains, moment–curvature diagrams,
polar ductility, and demand-utilisation ratios (η) for arbitrary cross-sections
defined in YAML.  The engine is EC2 / NTC 2018-oriented but designed to accept
any constitutive law.

---

## Features

- **Arbitrary section geometry** — rectangle, circle, annulus, T, inverted T,
  H, box, single- and double-tee slab, fully custom polygon with holes.
  Meshing via uniform grid (default) or constrained Delaunay triangulation.
- **Multi-material bulk** — concrete, structural steel (EN 10025-2,
  thickness-dependent), user-defined piecewise-linear σ-ε curves.
- **Full biaxial analysis** — vectorised Newton solver over the complete
  (N, χx, χy) space; resistance surface via ConvexHull on a mega-batch
  point cloud.
- **Seven demand-utilisation metrics** — anisotropy-corrected normalised
  space (η_norm, η_norm_beta, η_norm_ray), 2-D Mx-My contour at fixed N
  (η_2D), and three staged-path metrics for seismic combinations
  (η_path_norm_ray, η_path_norm_beta, η_path_2D).
- **Tiered verification reporting** — summary block, configurable Top-K
  console table, full JSON/CSV export; scales to hundreds of combinations.
- **Lightweight analysis mode** — `gensec analyze` decomposes per-material
  forces without computing the full resistance domain; η on-demand via
  exponential scan + bisection (≈ 15–25 solves per demand).
- **Moment–curvature** with robust key-point detection (cracking, yield,
  ultimate) using a span-bound curvature window valid for both
  compression- and tension-controlled failures.
- **Polar ductility** diagram at fixed N.
- **Section properties** — homogenized area, centroid, principal axes,
  elastic moduli (W), plastic moduli (Z via PNA bisection), kern.
- **REST API** — FastAPI server with `/api/run`, `/api/analyze`,
  `/api/inspect` (section properties only, < 200 ms), `/api/plot`.
- **Optional Numba JIT** acceleration on constitutive kernels.

---

## Quick start

```bash
# Install (requires Python ≥ 3.10)
pip install gensec

# Optional: Numba acceleration
pip install gensec[fast]

# Run a full analysis
gensec run input.yaml --output-dir results/

# Force decomposition without resistance domain
gensec analyze input.yaml --output-dir results/

# With on-demand eta
gensec analyze input.yaml --output-dir results/ --eta
```

A minimal YAML input:

```yaml
materials:
  concrete:
    type: concrete_ec2_gen1
    class: C25/30
  steel:
    type: steel
    fyk: 450

section:
  shape: rect
  params: {B: 300, H: 600}
  bulk_material: concrete
  mesh_size: 15
  rebars:
    - {y: 40,  x: 150, As: 628, material: steel}
    - {y: 560, x: 150, As: 628, material: steel}

demands:
  - name: SLU_1
    N_kN: -1500
    Mx_kNm: 200
    My_kNm: 0
```

See `examples/yaml_reference_example.yaml` for the full reference with
every supported option documented inline.

---

## Installation

```bash
# From PyPI
pip install gensec

# From source (editable)
git clone https://github.com/jagtree/gensec.git
cd gensec
pip install -e ".[dev]"

# Run the test suite
pytest
```

**Dependencies:** Python ≥ 3.10, NumPy, SciPy, Shapely, Matplotlib.
Optional: `numba` (JIT), `fastapi` + `uvicorn` (REST server),
`triangle` (Delaunay meshing).

---

## Section geometry

### Parametric primitives

| `shape` | Key `params` |
|---|---|
| `rect` | `B`, `H` |
| `circle` | `D`, `resolution` |
| `annulus` | `D_ext`, `D_int`, `resolution` |
| `tee` | `bf`, `hf`, `bw`, `hw` |
| `inv_tee` | `bf`, `hf`, `bw`, `hw` |
| `h` | `bf`, `hf_top`, `hf_bot`, `bw`, `hw` |
| `box` | `B`, `H`, `tw`, `tf_top`, `tf_bot` |
| `single_tee` | `b_top`, `h_top`, `bw`, `hw` |
| `double_tee` | `b_top`, `h_top`, `bw`, `hw`, `stem_spacing` |
| `custom` | `exterior` (list of `[x, y]`), `holes` (optional) |

### Rebar definition

```yaml
rebars:
  - y: 40         # vertical coordinate [mm] (required)
    x: 150        # horizontal coordinate [mm] (required for biaxial)
    As: 628.3     # area [mm²] (or: diameter + n_bars)
    material: steel
    embedded: true  # subtract displaced bulk (default: true)
```

---

## Demand verification

### Metric families

**3-D metrics** (anisotropy-corrected normalised space):

| Flag | Default | Description |
|---|---|---|
| `eta_norm` | `true` | Distance from demand to nearest face, fraction of Chebyshev radius. Primary metric. |
| `eta_norm_beta` | `true` | Composite ratio mixing demand norm and face distance. |
| `eta_norm_ray` | `false` | Ray-cast from origin: proportional scale-to-boundary. |

**2-D metric** (Mx-My plane at fixed N):

| Flag | Default | Description |
|---|---|---|
| `eta_2D` | `false` | Ray-cast from origin in the Mx-My contour at the demand's N. |

**Path metrics** (staged combinations only):

| Flag | Default | Description |
|---|---|---|
| `eta_path_norm_ray` | `false` | 3-D path-aware ray-cast. |
| `eta_path_norm_beta` | `false` | Composite path metric. |
| `eta_path_2D` | `false` | 2-D path metric on Mx-My contour. |

η ≤ 1 → verified.  Each metric answers a geometrically distinct question;
they are not ordered against each other.

### Tiered output

```
================================================================================
  DEMAND VERIFICATION
================================================================================
  Items: 500    Verified: 498    Failed: 2
  eta_max: 1.0312    eta_mean: 0.4217    eta_p95: 0.8934    eta_p99: 0.9812
  Governing: SLU_347  (eta = 1.0312)
  --> *** 2 item(s) NOT VERIFIED ***
--------------------------------------------------------------------------------
     #   Name       N[kN]  Mx[kNm]   eta_norm   eta_norm_beta   Status
  1      SLU_347  -2100.0   312.40     1.0312          1.0156     FAIL
  ...
  10     SLU_055   -900.0   267.30     0.9201          0.9087       OK
  ... and 490 more (490 verified)
================================================================================
```

Configure with `verification_top_k` (console rows) and
`fiber_details_top_k` (fiber post-processing) in the `output:` block.

---

## Analysis pipeline (lightweight)

`gensec analyze` uses only the fiber integrator.  No resistance domain is
computed; on-demand η is computed via bisection on the ray from the
equilibrium base point through the demand:

```
η = 1 / t_boundary    where  P(t) = B + t·(D − B)
```

Output includes per-material force decomposition for each demand:

```
Material        N [kN]    Mx [kNm]   My [kNm]
C25/30         -1248.3      163.5       0.0
B450C (bot)      -48.2        5.8       0.0
B450C (top)     -203.5       30.7       0.0
─────────────────────────────────────────────
Total          -1500.0      200.0       0.0
```

---

## Output flags reference

```yaml
output:
  # Metrics
  eta_norm: true
  eta_norm_beta: true
  eta_norm_ray: false
  eta_2D: false
  eta_path_norm_ray: false
  eta_path_norm_beta: false
  eta_path_2D: false
  delta_N_tol: 0.03

  # Generation
  generate_mx_my: false
  generate_3d_surface: false
  generate_moment_curvature: true   # expensive
  generate_polar_ductility: true    # expensive
  generate_3d_moment_curvature: true  # expensive
  n_angles_mx_my: 144

  # Reporting
  verification_top_k: 10
  fiber_details_top_k: 5

  # N levels for M-chi / polar
  n_levels_mode: demands   # demands | auto | explicit
  n_levels_count: 10
  # n_levels_values: [0, -500, -1000]

  # Performance knobs
  n_scan_mx_my: 120   # internal scan directions (independent of n_angles_mx_my)
  n_chi_mx_my: 14     # curvature points per direction in Mx-My
```

**Fast configuration** (skip all expensive diagrams):

```yaml
output:
  generate_moment_curvature: false
  generate_polar_ductility: false
  generate_3d_moment_curvature: false
```

Estimated speedup: 3–10× on typical sections.

---

## Sign conventions

- *N* > 0: tension; *N* < 0: compression.
- Strains: `ε = ε₀ + χx·ly − χy·lx` (right-hand rule, fibres at levers
  `lx = x − xG`, `ly = y − yG`).
- Moments: `Mx = Σσ·A·ly`, `My = −Σσ·A·lx`.
- `Mx > 0` → bottom-edge compressed.
- Curvatures stored internally in 1/mm; displayed as χ × 10⁶ (1/km).

---

## Licensing

GenSec is released under the **GNU Affero General Public License v3.0
or later (AGPL-3.0-or-later)** for open-source use.

Commercial licensing (for use in proprietary software or SaaS) is
available on request.  See `COMMERCIAL_LICENSE.md`.

Contributors must sign the Contributor License Agreement (`CLA.md`)
before pull requests are accepted.

---

## Development

```bash
# Editable install with all dev extras
pip install -e ".[dev,fast,gui]"

# Run tests
pytest

# Build docs (requires Sphinx)
cd docs && make html
```

See `CONTRIBUTING.md` for the contribution guide and `CLA.md` for the
Contributor License Agreement.

---

## Changelog

See [CHANGELOG_v0_3_3.md](CHANGELOG_v0_3_3.md) for the current release.
Full history in `docs/changelog.rst`.
