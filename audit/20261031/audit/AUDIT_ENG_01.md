# AUDIT_ENG_01 — Independent structural-engineering audit of GenSec

Read-only audit.  No file of the audited tree was modified.  Every number
below was reproduced in this session; the probe scripts, YAML cases and raw
logs are in the delivered `cases/`, `probes/` and `out/` folders.  Exit
codes were read from the process (`cmd > log; echo EXIT=$?`), never through
a pipe.

Tag `[D3]` marks a verdict or finding whose only evidence is internal to
GenSec (GenSec runs, GenSec-vs-GenSec, or an analytic oracle written in this
session).  Hand calculations are *internal analytic oracles*: they are
independent of the code but do not close D3.

---

## 1. Provenance and measured baseline

| Item | Measured |
|---|---|
| Archive supplied | `gensec_audit_20261028.zip`, SHA256 `29d18eff54c2cf0ee23cc5f4632981fab7a573a0828824a7d7caecad211d36fd`, 269 entries. **Not** a `.tar.gz` and **contains no git bundle.** |
| Zip vs tree | 269/269 entries byte-identical to the working tree (CRLF-normalised); 228 tracked entries identical to `HEAD`; 41 untracked (`tools/`, logos), `tools/` is git-ignored (`tools/.gitignore: *`). |
| Bundle on disk | `gensec_phase38.bundle`, SHA256 `8170bfc40146416d28af5a60252f07c242916ea12968efaeae2c4c3be2fe93d1`, HEAD `a5a5dba`; `git cat-file -t a5a5dba` → *not a valid object*: the bundle describes a different history from the audited repository. |
| `archive_currency.py --gate` | **EXIT=1**, `archive_unreadable` (zip is not a tar) and `bundle HEAD (none)`. |
| Working tree | branch `claude_remerge`, HEAD `02aca33f4f50fffe3cf10e1c60a832877990f4e5` ("gensec update before internal audit"); `git status --porcelain` empty, `git diff --stat` empty. |
| Current phase | latest ledger `handoff/38_0-GENSEC_LEDGER_38.md` → Phase 38. |
| Environment | uv 0.11.6, CPython 3.13.13, numpy 2.4.4, scipy 1.17.1, shapely 2.1.2, numba 0.65.0, matplotlib 3.10.8, fastapi 0.136.1, uvicorn 0.46.0, pytest 9.0.3, PyYAML 6.0.3, gensec 0.4.1.dev19; `PYTHONPATH=src`, `MPLBACKEND=Agg`, `uv run --no-sync`. |

Gate tools (baseline only — not evidence of engineering correctness):

| Tool | EXIT | Note |
|---|---|---|
| `getattr_inventory --gate` | 0 | 35 sites read, 0 unaccounted |
| `element_class_inventory --gate` | **1** | drift +6 / −3 (`cli._run`, `io_yaml._parse_output_flags`, `plots.plot_section`, state plots) |
| `selfcheck_inventory --gate` | 0 | |
| `material_assumption_inventory --gate` | 0 | 436 sites, no unexplained drift (see AUD-I-2) |
| `completion_marker --gate` | **1** | 5 `stale_digest` (cli.py, io_yaml.py, sls.py, timeline.py, timeline_run.py) |
| `datum_fingerprint capture` (PRE) | 0 | 5 examples |
| `domain_fingerprint capture` (PRE) | 0 | 31 declarations |
| `numeric_fingerprint capture` (PRE) | 0 | |

Provenance verdict: the supplied archive cannot be verified with the
project's own currency gate (wrong format, no bundle); its content is,
however, byte-identical to the audited tree, so the audit object is
unambiguous (`HEAD 02aca33` + clean tree).

---

## 2. Executive summary

| Severity | Count | IDs |
|---|---|---|
| **S1** | 10 | A-1, B-1, B-2, B-3, D-1, D-2, E-1, F-1, F-2, I-1 |
| S2 | 13 | A-2, B-4, B-5, C-2, C-3, E-2, E-3, G-1, I-2, J-1, J-2, J-4, K-1 |
| S3 | 2 | I-3, J-5 |
| OSS | 9 | A-3, B-6, C-1, D-3, E-4, F-3, G-2, H-1, J-3 |

By area: A 3 · B 6 · C 3 · D 3 · E 4 · F 3 · G 2 · H 1 · I 3 · J 5 · K 1.

**S1 findings, in order of engineering consequence**

1. **AUD-D-2** — the convex hull over-states the resistance of a section with a brittle tension branch by up to **+46 %** in moment (non-conservative).
2. **AUD-A-1** — demand components written under an unrecognised key are silently set to zero; the shipped `example_aci318.yaml` verifies **all-zero demands**.
3. **AUD-E-1** — the API/GUI traffic light reports `ok` for a demand at 2.9 × N_Rd.
4. **AUD-F-1** — a prestress *action* on a bonded tendon that already carries `eps_pe` is accepted: prestress counted twice (η_ray 0.359 instead of 0.546).
5. **AUD-F-2** — `grout_selfconsistent` with zero post-grout demand freezes the plane at zero: tendon reads back 1232.3 MPa instead of 1400 MPa (−12.0 %).
6. **AUD-B-1** — EC2 bridge: f_cd jumps **+17.6 %** between t = 28 d and t = 28.001 d (α_cc 0.85 → 1.0).
7. **AUD-B-2** — `ls` is not validated: `ls: ULS` gives γ_c = 1.0, f_cd **+50 %**.
8. **AUD-B-3** — the "neutral" containers carry EC2 defaults (n = 2, ε_c2, ε_cu2; f_yk = 450, γ_s = 1.15, E_s, ε_su = 0.01); a C70 declared through the custom type is **+23.7 %** in M_Rd.
9. **AUD-I-1** — under `code: aci318-19` the EC2 parabola-rectangle with ε_cu2 = 3.5 ‰ and γ_c/γ_s factors is accepted without comment.
10. **AUD-D-1** — the sweep integrates strain planes that violate the pivot-C rule and element limits, and they sit on the hull: up to **+2.3 %** radial excess in the high-compression zone.

---

## 3. Verdicts by area

| Area | Verdict | Basis |
|---|---|---|
| A — Hypotheses, conventions, units | RILIEVI [D3] | A-1 (S1), A-2 (S2), A-3 |
| B — Constitutive laws | RILIEVI [D3] | B-1…B-3 (S1), B-4, B-5 (S2), B-6 |
| C — Fibre integration | RILIEVI [D3] | C-1 conforming (area subtraction, convergence); C-2, C-3 (S2) |
| D — Resistance domain | RILIEVI [D3] | D-1, D-2 (S1), D-3 |
| E — Utilisation ratios η | RILIEVI [D3] | E-1 (S1), E-2, E-3 (S2), E-4 |
| F — Prestress and life stages | RILIEVI [D3] | F-1, F-2 (S1), F-3 conforming ULS |
| G — SLS | RILIEVI [D3] | G-1 (S2), G-2; cracked-section stresses not measured (§6) |
| H — Geometric and ideal properties | CONFORME [D3] | H-1: four sections, deviations ≤ 7·10⁻¹⁶ |
| I — Normative agnosticism | RILIEVI [D3] | I-1 (S1), I-2 (S2), I-3 (S3); plus B-1…B-3 |
| J — Output as a calculation document | RILIEVI [D3] | J-1, J-2, J-4 (S2), J-5 (S3), J-3 |
| K — Applicability limits | RILIEVI [D3] | K-1 (S2) |

---

## 4. Detailed findings

Material constants used in the hand calculations unless stated:
$f_{cd} = k_{lt} f_{ck}/\gamma_c$ with $k_{lt}=0.85$, $\gamma_c = 1.5$;
$f_{yd} = 450/1.15 = 391.304$ MPa, $E_s = 200\,000$ MPa;
parabola-rectangle with $n=2$, $\varepsilon_{c2}=2$ ‰, $\varepsilon_{cu2}=3.5$ ‰, whose
stress-block factors are

$$
\alpha = 1-\frac{\varepsilon_{c2}}{3\varepsilon_{cu2}} = 0.809524,\qquad
\beta  = 1-\frac{\varepsilon_{cu2}^2/2-\varepsilon_{c2}^2/12}{\varepsilon_{cu2}(\varepsilon_{cu2}-\varepsilon_{c2}/3)} = 0.415966 .
$$

### Area A

#### AUD-A-1 — Demand components under an unrecognised key are silently zero [D3]

| Field | Content |
|---|---|
| Severity / direction | **S1** / non-conservative |
| Evidence | `A1_sign_right_rebar.yaml` demand `{N_kN: 0, My_kNm: 150}` (no `Mx_kNm`) → parsed `My = 0` (`pA_sign.py`, EXIT 0) and reported `verified: True` in `out/A1/demand_summary.csv`. `E3_degenerate.yaml` demand `{N_kN: -1000, Mx_kNM: 900}` (case typo) → `N=-1000, Mx=0, My=0`, `verified: True`. Shipped `examples/example_aci318.yaml` (copy `I1_example_aci318_copy.yaml`) uses keys `N`, `Mx`, `My` → both demands parsed `(0, 0, 0)` (`pI_aci.py`, EXIT 0). Cause: `io_yaml._parse_demand` reads `My_kNm` only inside the `Mx_kNm` branch and refuses no unknown key. |
| Expected | Every declared action component reaches the verification, or the document is refused (principle: no silent fallback). |
| Deviation | 100 % of the dropped component (150 kNm; 900 kNm; all of the ACI example). |
| Proposal | Apply `_refuse_unknown_keys` to demands/envelope members and read each component independently. |

#### AUD-A-2 — `My` sign convention: documentation contradicts the code [D3]

| Field | Content |
|---|---|
| Severity / direction | S2 / indeterminate |
| Evidence | Section 400×400 with a single bar at the **right** edge (`A1_sign_right_rebar.yaml`). N–My sweep near N = 0: $M_y\in[-236.81,\;+5.29]$ kNm; `strain_field(0,0,+1e-5)` gives −1.98·10⁻³ at the max-x fibre. Hence **positive My compresses the right edge**. `docs/user_guide/sign_conventions.rst` states "M_y positive: **left** edge in compression" and $\varepsilon = \varepsilon_0 + \chi_x y + \chi_y x$, whereas `FiberSolver.strain_field` uses $-\chi_y x$ and `integrate` returns $M_y = -\sum \sigma A (x-x_{ref})$. Mx is consistent with the documentation (A2 case: $M_x\in[-5.29,\;+236.81]$). |
| Expected | Bar in tension, left edge compressed: $x = A_s f_{yd}/(\alpha\, b f_{cd}) = 142.17$ mm, $M = A_s f_{yd}(d-\beta x) = 782.6\cdot(360-59.14)\cdot10^{-3} = 235.4$ kNm on the side the documentation calls positive; GenSec puts it on the negative side. GenSec 236.81 vs 235.4 (+0.6 %, N-window ±20 kN). |
| Deviation | Sign inversion between user documentation and computation. For asymmetric sections a user following the documentation obtains the capacity of the opposite face. |
| Proposal | Make the documentation and the integrator state the same convention, and print it in every output. |

#### AUD-A-3 — Unit conversion and Mx convention (OSS) [D3]

kN → N and kN·m → N·mm conversion verified on all probes; N–Mx extremes of
A1 equal the closed form $N_{max} = A_s f_{yd} = 782.61$ kN and
$N_{min} = -(A_c f_{cd} + A_s(f_{yd}-f_{cd})) = -3468.6$ kN (CSV and log
agree). No unit path found other than the key problem of A-1.

### Area B

#### AUD-B-1 — EC2 bridge: α_cc switches from 0.85 to 1.0 just after 28 days [D3]

| Field | Content |
|---|---|
| Severity / direction | **S1** / non-conservative for t > 28 d |
| Evidence | `pB_concrete.py` (EXIT 0), `concrete_from_ec2(30, ls='F', NA='French', TypeConc='N', time=t)`: t = 27 → f_cd 16.901; t = 28 → **17.000** (α_cc 0.85); t = 28.001, 29, 365 → **20.000** (α_cc 1). f_ck used is 30.000 at every t ≥ 28. Code: `fben2.__init__`: `if time > 28: kt = 1 else: kt = 0.85`. |
| Expected | Physics: with f_ck(t) unchanged, f_cd cannot jump by 1/0.85 between two instants. EN 1992-1-1 §3.1.2(4): the reduction $k_t$ applies to α_cc **when the strength is determined at t > 28 d** — the opposite branch. For NTC 2018 §4.1.2.1.1.1, α_cc = 0.85 at every age. |
| Deviation | +3.000 MPa, **+17.6 %** on f_cd (and on α_ct). |
| Proposal | Separate α_cc (annex value) from k_t (applies only for t > 28 d), and refuse a time with no stated strength basis. |

#### AUD-B-2 — `ls` not validated: any unrecognised value gives γ_c = 1 [D3]

| Field | Content |
|---|---|
| Severity / direction | **S1** / non-conservative |
| Evidence | `B2_ls_ULS.yaml` (`ls: ULS`) loads without error: γ_c = 1, f_cd = 25.5 MPa; `B2_ls_F.yaml` gives 17.0 (`pB_loader.py`, EXIT 0). `fben2` with `ls='xyz'` → γ_c = 1. `_declared(..., admissible=...)` lists the admissible set only in the message, it does not test membership; `fben2` maps every non-F/A value to 1. |
| Expected | Refusal (admissible set F, A, QP, C) — EN 1992-1-1 Table 2.1N, γ_c = 1.5 for persistent/transient. |
| Deviation | f_cd +8.5 MPa, **+50 %**. |
| Proposal | Test membership in `_declared` and make `fben2` raise on an unknown limit state. |

#### AUD-B-3 — Normative defaults inside the neutral containers [D3]

| Field | Content |
|---|---|
| Severity / direction | **S1** / non-conservative (C ≥ 55) or indeterminate |
| Evidence | `B3_steel_nodecl.yaml`: `s: {type: steel}` with **no constants** loads as f_yk = 450, γ_s = 1.15, E_s = 200 000, k = 1, ε_su = 0.01; `c: concrete_ec2_gen1_custom, fck: 70` loads with n = 2, ε_c2 = −0.002, ε_cu2 = −0.0035 (`pB_loader.py`). Shipped `example_prestress.yaml` declares `rebar_steel` without `gamma_s` (silently 1.15). `pB_c70.py` (EXIT 0) compares the same section with the EN 1992 Table 3.1 C70 values (`B5_c70_explicit.yaml`): M_Rd at N = −4000 kN 490.25 vs 396.19 kNm. |
| Expected | EN 1992-1-1 Table 3.1, f_ck = 70: $n = 1.4+23.4\,(0.2)^4 = 1.437$, $\varepsilon_{c2} = 2.0+0.085\,(20)^{0.53} = 2.416$ ‰, $\varepsilon_{cu2} = 2.6+35\,(0.2)^4 = 2.656$ ‰ — or a refusal, since GenSec declares itself normatively agnostic. |
| Deviation | M_Rd +1.2 % (N = −1000), +5.7 % (−2000), +20.7 % (−3000), **+23.7 %** (−4000 kN); max M +10.4 %. N_Rd identical (−7004.91 kN, steel yielded in both). |
| Proposal | Remove dataclass defaults from `Concrete`/`Steel`; require every constant or a code bridge. |

#### AUD-B-4 — Physically impossible parameter sets accepted [D3]

S2 / indeterminate.  `B4_steel_impossible.yaml` loads without error:
ε_su = 0.001 < ε_yd = f_yd/E_s = 0.001957; k_hardening = 0.5 (f_td < f_yd,
softening "hardening" branch); concrete f_ct = 45 MPa > f_cm = 38 MPa.
Expected: refusal on $\varepsilon_{su} \le \varepsilon_{yd}$, $k<1$,
$f_{ct} \ge f_{cm}$ (physical consistency).  Proposal: consistency checks in
`__post_init__`.

#### AUD-B-5 — Defaults left in the EC2 service-law bridge [D3]

S2 / indeterminate.  Measured signature
`service_concrete_from_ec2(fck, fct_eff, ls='F', loadtype='slow', TypeConc='R', time=28.0)`;
`fben2` is called without `NA` (its default). `TypeConc` drives
$\beta_{cc}(t)$ and therefore f_cm(t), E_cm(t) of the service law at any
age ≠ 28 d. Expected: declared or refused (principle 2). Proposal: make
them keyword-only with no default.

#### AUD-B-6 — Table 3.1 values (OSS) [D3]

Six classes (C30, C45, C50, C55, C70, C90) checked against Table 3.1
expressions for n, ε_c2, ε_cu2, f_ctm, E_cm, ε_c1, ε_cu1: all equal to
machine precision except ε_cu1 at C50 (3.500 vs $2.8+27(0.4)^4 = 3.491$ ‰,
+0.25 %, branch boundary `fck <= 50`) and ε_cu2 at C90 (2.6005 vs 2.6 ‰ via
`max(·, ε_c2)`). Service law C30: σ(−ε_c1) = −38.0 = −f_cm, tangent(0)/E_cm =
1.05, cut at f_ct,eff = f_ctm = 2.896 MPa, σ = 0 beyond ε_cu1 — as declared.

### Area C

#### AUD-C-1 — Area subtraction and convergence (OSS, conforming) [D3]

Bars displace concrete exactly once: rect 300×500, 1500 mm² → GenSec
N_Rd,c = −3111.4565 kN vs $-(bh f_{cd} + A_s(f_{yd}-f_{cd}))=-3111.4565$
kN; ρ ≈ 2 % C70 case −7004.91 kN equal to the closed form; circle −4278.3365 kN
equal to the polygon formula. M_Rd(N = 0), rect (closed form 229.4192 kNm):
mesh 50 / 25 / 12.5 / 6.25 / 3.125 mm → −3.75·10⁻³, −6.5·10⁻⁴, −6.6·10⁻⁵,
−4.4·10⁻⁵, +1.2·10⁻⁷ (conservative side). Circle D500, 8 bars: independent
oracle (exact chord integration, pivot B, bottom bar 9.638 ‰) **175.618 kNm**;
GenSec 175.415 / 174.924 / 175.189 / 175.255 / 175.247 (−0.11 % … −0.40 %).
Convergence is not monotone (sampling and mesh errors interact), so no
observed order can be quoted.

#### AUD-C-2 — Hidden discretisation defaults [D3]

S2 / conservative in what was measured.  Defaults that are not declared in
the output: `mesh_size = 10`, `mesh_method = grid`, legacy `n_fibers_y = 100`,
CLI `--n-points 50`, `generate_biaxial(n_angles=36)` hard-coded in `cli.py`,
`n_angles_mx_my = 72`. Effect measured in D-3 (≤ 0.99 % chord loss). This is
inconsistent with the project's own refusal of an undeclared
`segments_per_quadrant` ("GenSec does not choose this for you").
Proposal: declare resolution in the input and print it with the results.

#### AUD-C-3 — Bars without `x` lumped at the centroid [D3]

S2 / conservative.  `examples/example_input.yaml` writes layers
`{y: 40, n_bars: 3, diameter: 20}` without `x`; `_setup_rebars` puts all the
area at x = 150. `pC_lumped.py` (EXIT 0) vs the same bars at x = 40/150/260:
My_max at N = −1500 / −1200 / −900 kN = 93.2 / 91.4 / 88.1 vs 149.1 / 160.3 /
157.8 kNm (**−37 % … −44 %**); Mx_max within ±0.8 %. The example verifies
biaxial envelopes with My up to 180 kNm against this section. Proposal:
refuse a missing `x` when the section is analysed biaxially.

### Area D

#### AUD-D-1 — Inadmissible strain planes on the hull; the pivot domain is not enforced [D3]

| Field | Content |
|---|---|
| Severity / direction | **S1** / non-conservative (high-compression zone) |
| Evidence | `pD_admiss.py` (EXIT 0) re-evaluates every plane of `NMDiagram._ultimate_strain_configs_1d` (2600 planes) and flags (i) any element beyond its own limit (concrete below ε_cu2, steel beyond ±ε_su) and (ii) fully compressed planes violating EN 1992-1-1 §6.1(5) (strain at 3/7 h > ε_c2). Branch 1 drives the free edge down to the *global* minimum (−ε_su of steel); branch "near-uniform compression" goes to uniform −3.5 ‰. No admissibility filter is applied to the domain (`_ultimate_by_group` is used only by the curvature scans). |
| Expected | Hull of admissible planes only. |
| Deviation | Radial excess of the reported hull over the admissible hull: **+2.28 %** (`C2_rect_h12.5`, ρ = 1 %, at N = −2818 kN), +0.74 % (ρ = 4 %, plastic), +0.64 % and N_Rd,min **+0.17 %** (ρ = 4 %, k = 1.15, ε_su = 6.75 %). 25–27 of the hull vertices are inadmissible. |
| Note | `docs/architecture/architecture_solver.rst` states the scan "covers the same configuration space" as the EC2 pivot method — measured false. |
| Proposal | Filter planes per element group (and declare the pivot-C rule as a code-supplied constraint) before the hull. |

#### AUD-D-2 — Convex hull over a non-convex domain (brittle tension) [D3]

| Field | Content |
|---|---|
| Severity / direction | **S1** / non-conservative |
| Evidence | `D2_brittle_tension.yaml`: rect 300×500 C30 with the linear tension branch (f_ct = 2.9, E_c = 32 837, ε_ct = 8.83·10⁻⁵), 2 × 100 mm². `pD_convex.py` (EXIT 0): exhaustive enumeration of 176 024 admissible planes (grid on edge strains, concrete ≥ ε_cu2, steel ≤ ε_su), integrated by GenSec, gives the **upper bound** of the moment attainable at each N; compared with the hull of `NMDiagram.generate`. |
| Expected | Hull ≤ attainable upper bound (otherwise the hull contains loads no admissible state carries). |
| Deviation | N = 50 kN: hull 39.931 vs attainable 27.312 kNm (**+46.2 %**); N = 100: +37.2 %; N = 150: +18.0 %. A demand at 0.98 × hull is `inside` and exceeds the attainable bound by +43 %. Also the hull misses the uncracked peak (N_max 261 vs 427 kN). |
| Note | `docs/theory/demand_verification.rst` asserts convexity "under the parabola-rectangle law"; the tension branch is offered on the ULS law (`enable_tension`) with no restriction. |
| Proposal | Refuse the hull representation for laws with softening/brittle branches, or build a non-convex representation. |

#### AUD-D-3 — Sampling chord and symmetry (OSS) [D3]

`pD_chord.py` (EXIT 0): with the default domain (36 angles × 50) the
η_ray of points on the dense boundary (144 × 400) is ≤ **1.0099** (rect) and
≤ 1.0094 (circle) and never < 1.0000: chord error conservative, ≈ 1 %, but
not declared anywhere. Doubly symmetric rect: Mx extremes −603.435 / +604.939
kNm (asymmetry 2.5·10⁻³), My exactly symmetric; circle symmetric to 10⁻¹⁵.
No symmetry tolerance is declared.

### Area E

#### AUD-E-1 — API / GUI status is `ok` for failing demands [D3]

| Field | Content |
|---|---|
| Severity / direction | **S1** / non-conservative |
| Evidence | `E1_api_status_no2D.yaml` (η_2D disabled, defaults otherwise), `pE_api.py` (EXIT 0): `verify_point(0,0,150)` (≈ 30 × the My capacity on that side) → `status ok`; `verify_point(-10000,0,0)` (N_Rd,min = −3469 kN) → `status ok`; `analyze()` rows → `ok`. `api.py` reads `eta_3D`, which `VerificationEngine` no longer produces, falls back to `eta_2D`, and `_status_from_eta(None)` returns `"ok"`. |
| Expected | `fail` (point outside the domain; the CLI itself reports `inside False`). |
| Proposal | Drive the status from `verified`/`inside`, never from an absent metric. |

#### AUD-E-2 — The default η metrics are not demand/capacity ratios [D3]

| Field | Content |
|---|---|
| Severity / direction | S2 / non-conservative in the reading of the reserve |
| Evidence | Defaults `eta_norm`, `eta_norm_beta` (ray metric off). `pE_metrics.py` (EXIT 0) on `D3_sym_rect` with demands at a known radial fraction λ = η_ray: λ = 0.10 → η_norm 0.463–0.616; λ = 0.50 → **0.218**–0.709; λ = 0.90 → 0.839–0.905. Zero demand: η_norm = 0.595 (D3), **0.986** (A1). Outside: `G_out` η_ray 2.425, η_2D 2.045, but η_norm **1.0001**, η_β 1.0000; A1 `My_pos` η_2D 53.4, η_norm 1.0003. Pass/fail equals the hull `inside` test, so the verdict itself is consistent. |
| Expected | A utilisation printed as η in a calculation document is read as demand/capacity; $\eta_{norm}=1-d_{min}/D_{max}$ measures distance to the nearest face relative to the Chebyshev radius, is 0 at the Chebyshev centre (not at zero load), and for exterior points $\min_i|a_i^\top p+b_i|$ is the distance to a *plane*, not to the hull. |
| Proposal | Print η_ray (or η_2D) as the governing ratio; label the others as distance indices; compute exterior distance to the hull. |

#### AUD-E-3 — Degenerate cases are not fail-loud [D3]

S2 / indeterminate.  `E3_degenerate.yaml` (CLI EXIT 0): η_2D at N beyond the
domain is `None` and silently dropped from `verified` (only a
`RuntimeWarning` on stderr); a staged combination whose first stage is
already outside the domain produces η_path_norm_ray 2.104, η_path_2D 2.974
and **`"eta_path_norm_beta": Infinity`** written into
`combination_summary.json` (not valid JSON, RFC 8259), with no statement
that the base is outside. Path-dependence of the capacity is represented
only through locked-in strains and section states; for an unbonded/external
history it is a declared input (`prestress_scope.rst`), not a computed one.

#### AUD-E-4 — Metric coincidences (OSS) [D3]

At N = 0, η_ray / η_2D − 1 ∈ [+0.08 %, +0.30 %] over 12 directions (two
different chord discretisations); η_path from the origin equals η_ray
exactly (0.0e+00).

### Area F

#### AUD-F-1 — Prestress counted twice (action + prestrain) [D3]

| Field | Content |
|---|---|
| Severity / direction | **S1** / non-conservative in the measured case |
| Evidence | `F1_pretensioned_uls.yaml`: bonded tendon `T1`, `eps_pe = 0.006` (so the domain already contains the prestrain: zero-plane resultant N = +1170 kN, Mx = −234 kNm, as $A_pE_p\varepsilon_{pe}$ at y = 100), plus a stage `prestress_actions: [{ref: T1, sigma_p0: 1170, Ap: 1000}]`. Accepted by the loader and by the engine (CLI EXIT 0): cumulative demand N = −1170 kN, Mx = −300 + 234 = **−66 kNm**, η_ray 0.3588, `verified: True`. The same demand without the action: η_ray 0.5462. |
| Expected | Prestress enters either as an action or as a pre-strain, never both — refusal when a `prestress_actions` ref targets a bonded tendon with non-zero `eps_pe`. |
| Deviation | Demand moment −78 %; η_ray −34 %. |
| Proposal | Refuse the combination at resolution time. |

#### AUD-F-2 — `grout_selfconsistent` with zero post-grout demand [D3]

| Field | Content |
|---|---|
| Severity / direction | **S1** / indeterminate (lower tendon stress: unsafe for σ_p / σ_c limits, safe for decompression) |
| Evidence | `pF_grout.py` (EXIT 0) on the section of `example_prestress.yaml`, one tendon jacked to 1400 MPa, then grouted. `demand_after = (0,0,0)` → loop takes `eps0 = chi = 0` (shortcut in `grout_selfconsistent`), converges in 2 iterations to ε_init = ε_pe,eff = 7.1795·10⁻³; the bonded section, solved at zero load, reads back **1232.34 MPa**. `demand_after = (−1 N, 0, 0)` → ε_init = 8.2130·10⁻³, read-back **1400.0000 MPa**. The timeline passes exactly `(0,0,0)` for a stress→grout sequence with no base demand (`timeline.py` ≈ l. 2270). |
| Expected | Grouting is not a mechanical event: $\varepsilon_{init}=\varepsilon_{pe,eff}-\varepsilon_{sec,grout}$, continuous in the demand; read-back = 1400 MPa. A self-equilibrated prestressed section is not at the zero plane. |
| Deviation | −167.7 MPa (−12.0 %) in tendon stress; ε_init jumps by −1.0335·10⁻³ (−12.6 %) between 0 N and 1 N. |
| Note | Test `run_phase8_task2_validation.py::axis_A6` asserts byte-equality between timeline and driver at `demand_after=(0,0,0)`: it certifies the defect on both sides (structural blindness). This contradicts the docstring and the ratified closure of the read-back residue; per §4 of the brief it is reported in chat as a stop-and-report item. |
| Proposal | Remove the zero-demand shortcut; always solve the equilibrium of the bonded view. |

#### AUD-F-3 — Bonded pretensioned ULS (OSS, conforming) [D3]

`F1_pretensioned_uls.yaml`, Y1860S7 horizontal branch: f_p01d = 1391.304,
ε_ud = 0.0315. Hand calculation, tendon yielded:
$x = 1391.3\cdot10^3/(0.809524\cdot300\cdot22.667) = 252.75$ mm;
$\Delta\varepsilon_p = 3.5‰\,(500-252.75)/252.75 = 3.42‰$, total 9.42 ‰ > ε_el = 7.13 ‰;
$M_{Rd} = 1391.3\,(500-0.415966\cdot252.75) = 549.38$ kNm.
GenSec **549.376** kNm (< 0.01 %).

### Area G

#### AUD-G-1 — SLS limit set silently disables the steel and tendon checks [D3]

S2 / non-conservative.  `pG_limits.py` (EXIT 0): `limits: {kind: characteristic}`
without `fyk`/`fpk` gives `sigma_s_tens: None`, `sigma_p_max: None` (checks
off) with no refusal; with them, 360.0 and 1395.0 MPa ($0.8 f_{yk}$,
$0.75 f_{pk}$, EN 1992-1-1 §7.2(5)). The k-coefficients come from the EC2
provider (`EC2_RECOMMENDED`) unless overridden: recommended values, not a
declaration of the National Annex. Proposal: refuse a characteristic set
without the strengths it needs.

#### AUD-G-2 — Moduli on the combination (OSS) [D3]

Empty `sls.moduli` and a zero modulus are refused with a reason; no
fallback to E_cm(28) observed on this path.

### Area H

#### AUD-H-1 — Section properties against closed form (OSS, conforming) [D3]

`pH_props.py` (EXIT 0), oracle by composite-area formulae
$A=\sum A_i$, $y_g=\sum A_i y_i/A$, $I_x=\sum (I_{x,i}+A_i(y_i-y_g)^2)$,
$\tan 2\alpha = -2I_{xy}/(I_x-I_y)$:
box 400×700 with 240×540 hole ($I_x = 8.284053\cdot10^9$), T 800/150/300/550
($y_g = 422.3684$, $I_x = 1.289490\cdot10^{10}$), L-section
($I_{xy} = -4.921875\cdot10^8$, $I_\xi = 1.859117\cdot10^9$, α = 20.8017°),
homogenised RC rect ($n = 6.06$, $A_{id} = 160\,627.3$, $I_x = 3.544925\cdot10^9$).
Maximum relative deviation 6.9·10⁻¹⁶. Multi-zone homogenisation is refused
with `NotImplementedError` (fail-loud).

### Area I

#### AUD-I-1 — EC2 physics under an ACI 318 declaration [D3]

| Field | Content |
|---|---|
| Severity / direction | **S1** / non-conservative |
| Evidence | `I1_example_aci318_copy.yaml` (shipped example, `code: aci318-19`): concrete loads with ε_cu2 = −0.0035, n = 2, ε_c2 = −0.002, γ_c = 1.5, k_lt = 0.85 (f_cd = 15.867); steel γ_s = 1.15. Nothing refuses or flags EC2-shaped partial factors and strain limits under an ACI declaration; ε_cu2 comes from the container default (B-3). |
| Expected | ACI 318-19 §22.2.2.1: ε_cu = 0.003, Whitney block (§22.2.2.4), strength reduction φ on nominal strength (§21.2) — or an explicit refusal of a law the declared code does not own. |
| Proposal | Bind the constitutive shape and its constants to the declared code, refuse cross-code mixtures. |

#### AUD-I-2 — Inventory gate blind to container defaults [D3]

S2 / indeterminate.  `material_assumption_inventory --gate` EXIT 0 ("no
unexplained drift") while B-3 shows f_yk/γ_s/ε_su and n/ε_c2/ε_cu2 defaults
reaching analyses: the tool counts sites against a baseline, so a
baselined default is invisible. Structural blindness, not proof of absence.

#### AUD-I-3 — EC2 module names ACI [D3]

S3.  `materials/verification_limits.py` (which hosts `ec2_stress_limits` and
`EC2_RECOMMENDED`) names "ACI" at lines 36 and 285. Principle 5 requires
hermetic isolation. Also: a generic container and an EC2 provider share one
module.

### Area J

#### AUD-J-1 — Utilisation chart: wrong legend, skipped ≡ zero [D3]

S2.  `out/E3/demand_utilization.png` (inspected): legend shows η_norm, η_ray,
η_2D all as red hatched (the failure style), only η_β distinguishable; a
metric not computed (η_2D at N beyond the domain, `tension_beyond_Nmax`)
and a metric equal to zero (`origin`) are both drawn as no bar; failing
values 1.0001 / 1.0004 are labelled "1.000"; the "Showing: …" note is
covered by the legend. **Andrea's report confirmed.**

#### AUD-J-2 — Bulk zones of different material not distinguished [D3]

S2.  `J2_two_zones.yaml` (C25 with a C60 core 200×300): `section_geometry.png`
draws one grey rectangle — the C60 zone has neither outline nor colour, and
no material legend exists. **Andrea's report confirmed (stronger: the zone is invisible).**

#### AUD-J-3 — Section + neutral-axis images (OSS) [D3]

Generated for every demand (`section_<demand>.png`, A1, E3, F1, J2); A1
`section_My_neg.png` shows the neutral axis at x ≈ 167 mm with the left edge
compressed, consistent with A-2. Not generated for staged-combination
stages (F1, E3 combinations: none). **Andrea's report not confirmed for
demands; confirmed for combinations.**

#### AUD-J-4 — Verification outputs are not traceable [D3]

S2.  `verification_summary.json` / `demand_summary.csv` carry only name,
forces, η values, `inside`, `verified`: no code, material laws and their
constants, partial factors, domain resolution or GenSec version. The
summary distinguishes verified / not verified but not "not computed"
(J-1, E-3). A calculation cannot be signed from the output alone.

#### AUD-J-5 — Mislabelled export column [D3]

S3.  `out/A1/n_my_domain.csv` header `N_kN,Mx_kNm`, values = My (range
−257.17 / 255.78, equal to the logged My range).

### Area K

#### AUD-K-1 — Applicability limits only in the documentation [D3]

S2.  No output of the runs in this audit (A1, E3, F1, J2: logs, CSV, JSON)
mentions shear, torsion, second-order effects, local buckling, interface
slip or the declared prestress exclusions (grep: zero hits). They are
stated in `docs/theory/prestress_scope.rst` and the README only, not where
the result is used.

---

## 5. D3 recapitulation

All entries below rest on GenSec runs and/or analytic oracles written in
this session; none is closed by an external reference.

| Tag | Closing external check (type) | Quantity to compare |
|---|---|---|
| V-A | Textbook RC example with asymmetric reinforcement, independent software | sign of My for given compressed face; M_Rd |
| V-B | Published EN 1992-1-1 worked examples; independent section software | f_cd(t), Table 3.1 constants, M_Rd(N) at f_ck > 50 |
| V-C | Independent fibre/section software; closed-form literature | N_Rd, M_Rd vs mesh size |
| V-D | Independent interaction-diagram software; experimental N–M tests | N–M boundary in high compression; brittle-tension members |
| V-E | Independent software with a declared utilisation definition | proportional load factor for the same demands |
| V-F | Literature worked examples of pre-/post-tensioned sections; independent software | M_Rd, tendon stress after grouting |
| V-G | Hand/independent cracked-section SLS examples | σ_c, σ_s, σ_p under SLS |
| V-H | CAD / independent section-property tool | A, y_g, I_x, I_y, I_xy, α |
| V-I | ACI 318 design example | M_n, φM_n with ε_cu = 0.003 |
| V-J | Review by a signing engineer of the produced document | completeness of traceability |
| V-K | Review against the project's scope statement | presence of limitations at point of use |
| AUD-A-1 | Same input in independent software | demands actually verified |
| AUD-A-2 | Independent software, asymmetric section | sign of My vs compressed face |
| AUD-A-3 | Hand/independent N_Rd | N_Rd,max/min |
| AUD-B-1 | EN 1992-1-1 / annex worked example at t > 28 d | f_cd(t) |
| AUD-B-2 | EN 1992-1-1 Table 2.1N worked example | γ_c, f_cd |
| AUD-B-3 | Independent software with C70 Table 3.1 law | M_Rd(N) |
| AUD-B-4 | Material test data / supplier data | admissible parameter sets |
| AUD-B-5 | EN 1992-1-1 §3.1.2 worked example | f_cm(t), E_cm(t) |
| AUD-B-6 | Published Table 3.1 | constants |
| AUD-C-1 | Independent section software | M_Rd(N = 0), N_Rd |
| AUD-C-2 | Independent software at stated resolution | domain points |
| AUD-C-3 | Independent software with distributed bars | My_Rd |
| AUD-D-1 | Independent software applying Fig. 6.1 pivots; column tests | N–M in high compression |
| AUD-D-2 | Tests on lightly reinforced members in tension-flexure; non-linear section software | M_Rd at small tension |
| AUD-D-3 | Independent dense reference | boundary distance |
| AUD-E-1 | Independent software verdict | pass/fail of same demands |
| AUD-E-2 | Independent software (declared η definition) | η for same demands |
| AUD-E-3 | Independent software, staged loading | η for path from exterior base |
| AUD-E-4 | Independent software | η_2D vs η_ray at N = 0 |
| AUD-F-1 | Literature prestressed-section example | demand with prestress counted once |
| AUD-F-2 | Literature post-tensioning example, or measured tendon force | σ_p after grouting |
| AUD-F-3 | Literature ULS example of a pretensioned beam | M_Rd |
| AUD-G-1 | EN 1992-1-1 §7.2 worked example | σ_s, σ_p limits applied |
| AUD-G-2 | Independent SLS tool | moduli used |
| AUD-H-1 | CAD / independent tool | section properties |
| AUD-I-1 | ACI 318 design example | φM_n |
| AUD-I-2 | External code review | inventory coverage |
| AUD-I-3 | External code review | module isolation |
| AUD-J-1 | Reviewer reading the chart | metric identification |
| AUD-J-2 | Reviewer reading the figure | zone identification |
| AUD-J-3 | Reviewer reading the figure | neutral-axis position |
| AUD-J-4 | Signing engineer | traceability |
| AUD-J-5 | Independent re-plot of export | column meaning |
| AUD-K-1 | Signing engineer | limitations stated in output |

Count check (G3): 11 verdict tags + 34 finding tags = 45 tags in §3–§4;
45 rows above (measured with `grep`, see chat).

---

## 6. Limits of this audit

- **No external reference** (D3): all conclusions are internal; analytic oracles were written by the auditor.
- **SLS cracked-section stresses** (area G) were not computed against a hand cracked-section analysis; only the parsing of `moduli`/`limits` was measured.
- **Time-dependent losses, `PreexistingState` and the two clocks** (area F) were read in the scope documentation but not measured; the `losses_models` defaults it lists were not re-measured.
- **Mixed-material generality** (timber, CFRP, delamination, interface slip): only the `tabulated` and steel rupture branches were seen; no measurement was made, so no finding is raised beyond K-1.
- **NTC 2018 path** (`code: ntc18`): not run; B-1 was measured under `code: ec2`, NA `French`.
- **Biaxial domain admissibility** (D-1) was measured on the uniaxial sweep only; the biaxial template has the same branches by reading.
- **OOM on `example_prestress.yaml`** was not encountered: that file was used only to build a section for F-2; its full run was not attempted.
- Probe scripts call some private functions (`_ultimate_strain_configs_1d`, `_bonded_state`, `_parse_stress_limits`); the corresponding public paths were exercised by the CLI where stated.
- The numeric fingerprint capture spanned a machine suspension (wall time ≈ 3 days, 903 s CPU); results unaffected.

---

## 7. Closing gates

| Gate | Result |
|---|---|
| G1 — verdict for every area A–K | 11/11 (§3) |
| G2 — no finding without in-session evidence | every finding cites a case/probe/log in `cases/`, `probes/`, `out/` |
| G3 — `[D3]` tags vs §5 table | 45 tags in §3–§4, 45 rows in §5; 34 finding IDs identical in both (one further literal `[D3]` in the preamble defines the tag) |
| G4 — tree untouched | `git diff --exit-code` EXIT 0; `git status --porcelain` empty; HEAD `02aca33` |
| G5 — fingerprints POST = PRE | `datum`/`domain`/`numeric_fingerprint compare` EXIT 0; `cmp` PRE vs POST files EXIT 0 (×3) |
| G6 — report SHA256 | given in the delivery message (a hash cannot be contained in the file it hashes) |
