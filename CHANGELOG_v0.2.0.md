# GenSec v0.2.0 — Changelog

## Demand architecture v2.1

### New features

**Four utilization ratio types** controlled by output flags:
- `eta_3D` — ray from origin on 3-D N-Mx-My ConvexHull (always fast).
- `eta_2D` — ray from origin on Mx-My contour at fixed N (VCASLU-style ρ_M).
- `eta_path` — ray from arbitrary base point on 3-D hull (staged loading).
- `eta_path_2D` — ray from base on Mx-My contour at target N, subject to `delta_N_tol` (default 3%).

**Combinations** — factored sums of base demands:
- Simple: flat `components` list with `ref` + `factor`.
- Staged: sequential `stages` with per-stage η_path. Measures increment margin relative to base state (seismic, prestress, staged construction).

**Envelopes** — collection of demands and/or combinations:
- Members via `ref` or inline demands with direct N/Mx/My.
- Optional `factor` on each member.
- Reports max η and governing member.

**Structured JSON output**:
- `demand_summary.json`, `combination_summary.json`, `envelope_summary.json`.
- `verification_summary.json` (unified).

### Output flag defaults

```yaml
output:
  eta_3D: true
  eta_2D: false
  eta_path: true
  eta_path_2D: false
  delta_N_tol: 0.03
```

### Breaking changes

- `DemandChecker` → `DomainChecker` + `MxMyContour` + `VerificationEngine`.
- JSON key `utilization` → `eta_3D` / `eta_2D`.
- `combinations` YAML format: `demands` list → `components` / `stages`.
- Old combination groups → use `envelopes`.

---

## CLI: subcommands and data-first pipeline

### `gensec run input.yaml`

Full analysis pipeline (replaces the old bare `gensec input.yaml`). Backward-compatible: passing a `.yaml` file without subcommand is auto-detected and routed to `run`.

### `gensec plot data_file.json`

Regenerate any plot from a previously exported JSON data file. Options:

```bash
gensec plot mx_chi_N0.json                  # default: saves mx_chi_N0.png
gensec plot mx_chi_N0.json -o custom.png    # custom output path
gensec plot mx_my_Nm10.json --dpi 300       # higher resolution
```

Supported JSON types (auto-detected from `"type"` key or key names):
- `moment_curvature` → M-χ diagram with cracking/yield/ultimate markers.
- `mx_my_contour` → Mx-My interaction contour.
- N-M domain, 3-D surface (inferred from keys).

### Data export before every plot

All analysis steps now export JSON+CSV data **before** generating the plot:
- M-χ curves: `mx_chi_N{label}.json` / `.csv`
- Mx-My contours: `mx_my_N{label}.json` / `.csv`
- N-M domains, 3-D surface, verification results: already exported.

---

## Moment-curvature: first cracking detection

- Cracking strain ε_cr = f_ctm / E_cm from `ec2` properties.
- Detected when any bulk fiber's tensile strain exceeds ε_cr.
- Orange diamond marker on M-χ curve.
- **Numerical legend box** in bottom-right corner showing M and χ values for cracking, yield, and ultimate (both curvature signs).
- Output keys: `cracking_chi_pos`, `cracking_M_pos`, `cracking_chi_neg`, `cracking_M_neg`.

---

## 3D resistance surface

### Lofted surface from contour slices

The 3D plot no longer uses the raw ConvexHull triangulation (which
produced spurious facets and artifacts especially in the tension zone).
Instead, the surface is **lofted from Mx-My contour slices**:

1. The 3D hull is sliced at 20 regular N levels.
2. Each slice is cleaned via a 2D ConvexHull (removes interior points).
3. Contours are resampled to 72 equi-spaced points using **arc-length
   parameterised Cartesian interpolation** (not polar — avoids artifacts
   on non-convex contours and degenerate contours near the poles).
4. Degenerate contours (near pure tension/compression) are filtered out.
5. The surface is rendered with `plot_surface` from the structured grid.

**Contour rings** (parallels at N = const) and **meridians** (longitude
lines) are drawn on the surface to aid shape comprehension.

### Two perspective views

Two side-by-side panels: compression side (elev=30) and tension side
(elev=−30).  A **dedicated vertical colorbar** is placed between the
panels (no overlap with the plots).

Linear `Normalize` colorbar (no more asymmetric TwoSlopeNorm).

---

## Plot improvements

## Plot improvements

**Mx-My diagram**: Mx on X-axis, My on Y-axis. Legend placed outside the plot area (``bbox_to_anchor``) to avoid overlap with domain contour.

**M-chi numerical legend**: monospace box in bottom-right with M and chi values for cracking, yield, and ultimate (both curvature signs).

**Demand utilization heatmap**: grouped bar chart showing all enabled eta types side by side. Red bars exceed 1, green within limit.

**Rebar numbering**: offset black text with semi-transparent white bbox in all section plots.

**``gensec plot`` subcommand**: regenerate any plot from exported JSON without re-running analysis.

---

## Material compatibility note

**Kent-Park / softening laws**: fully compatible. Any `Material` subclass with `stress(eps)` / `stress_array(eps)` and correct `eps_min` / `eps_max` works. `TabulatedMaterial` supports descending branches already.

---

## Modules changed

| Module         | Summary                                                     |
|----------------|-------------------------------------------------------------|
| `check.py`     | Full rewrite: DomainChecker, MxMyContour, VerificationEngine |
| `io_yaml.py`   | New parsers for combinations, envelopes, output flags       |
| `export.py`    | M-chi/Mx-My JSON+CSV exports, combination/envelope JSON      |
| `cli.py`       | Subcommands run/plot, data-first pipeline                   |
| `capacity.py`  | Cracking detection, denser tension zone scan                |
| `plots.py`     | Lofted 3D surface, M-chi legend, grouped heatmap, plot_from_json |
