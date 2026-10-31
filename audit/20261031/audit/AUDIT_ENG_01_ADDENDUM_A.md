# AUDIT_ENG_01 — Addendum A: convexity of prestressed domains

Requested after the report (decision D0-3: "is the non-convex domain needed
for prestress?").  Same rules as AUDIT_ENG_01: read-only, tree at HEAD
`02aca33`, every number reproduced in session, exit codes read from the
process.  `AUDIT_ENG_01.md` is unchanged (SHA256
`f06f4852efb5e97d15c724f85d6f87f331a965f8780c334026b9d843e85524fd`).

## A.1 Method

`probes/pD_convex_general.py` (uniaxial N–Mx):

1. Enumerate strain planes on an $n_g \times n_g$ grid of (bottom, top) edge
   strains over GenSec's own global strain envelope.
2. Keep a plane only if **every element group respects its own law
   limits**: bulk compression limit (at $\varepsilon + $ bulk offset),
   rebars both limits, tendons both limits at
   $\varepsilon_{sec} + \varepsilon_{init}$.  Linear strain ⇒ evaluated
   exactly on the extreme lever arms of each group.
3. Per N slice (band 0.5 % of the N range, 300 slices) compare:
   - **true-domain concavity** — convex hull of the admissible cloud vs the
     largest moment the cloud actually attains in the slice;
   - **GenSec hull excess** — hull of `NMDiagram.generate(n_points=400)` vs
     the same attained moment.
4. Excess normalised on $M_{ref} = \max|M|$ of the section (a ratio to the
   local moment is unstable where that moment is small).
5. Repeat with $n_g = 720$ (4× planes): a concavity that does not shrink is
   real; one that shrinks is grid noise.

The oracle integrates with GenSec's own direct problem, so its conclusions
are internal **[D3]**.

## A.2 Controls

| Case | $n_g$ = 360 | $n_g$ = 720 | Reading |
|---|---|---|---|
| `C2_rect_h12.5` (RC, plastic steel) | 0.00 % | 0.00 % | convex |
| `D1_hardening_rho4` (RC, hardening steel) | 1.08 % | 0.61 % | shrinks → grid noise, convex |
| `D2_brittle_tension` (positive control) | 15.89 % | 16.33 % | non-convex, stable |

## A.3 Prestressed sections

| Case | true-domain concavity, 360 → 720 | GenSec hull excess (720) |
|---|---|---|
| `F1_pretensioned_uls` (audit case) | 3.16 % → 3.08 % (17.7 kNm, N = −3488 kN) | 20.66 % (119.0 kNm, N = −1390 kN) |
| `F1b_no_compression` (strand not in compression) | 3.16 % (n_g 360) | 7.05 % (n_g 360) |
| `example_prestress.yaml` | 2.17 % → 2.05 % (27.4 kNm) | 10.95 % (146.4 kNm) |
| `example_strand_only_slab.yaml` | 1.87 % → 1.82 % (3.8 kNm) | 7.68 % (15.9 kNm) |
| `example_external_tendon.yaml` | 3.88 % → 5.58 % (45.1 kNm) | 4.24 % (34.2 kNm) |
| `example_staged_injection.yaml` | 4.01 % → 3.97 % (147.6 kNm) | 5.84 % (217.0 kNm) |
| `example_composite_losses.yaml` | 0.63 % → 0.47 % | 3.39 % (155.9 kNm) |
| `example_composite_losses_fb_gap.yaml` | 0.63 % (n_g 360) | 4.19 % (n_g 360) |
| `yaml_reference_example.yaml` (no tendon) | 0.56 % (n_g 360) | 3.57 % (n_g 360) |

Reading:

- Sections with **bonded prestrained tendons** have a genuinely non-convex
  N–M domain: 1.8–5.6 % of $M_{ref}$, stable or growing with grid
  refinement, always on the side where the tendon lies in the compressed
  zone.  The concavity is unchanged when the strand is forbidden to work in
  compression (F1b), so it is a property of the prestressed domain, not of
  an unphysical branch.
- The composite and non-prestressed examples sit at the noise level of the
  hardening control.

## A.4 New findings

### AUD-D-4 — Prestressed domains are not convex [D3]

| Field | Content |
|---|---|
| Severity / direction | **S1** / non-conservative |
| Evidence | §A.3; logs `out/D2bis_controls_360.log`, `_720.log`, `out/D2bis_examples_360.log`, `_720.log`, `out/D2bis_F1b.log` (all EXIT 0). |
| Expected | Hull ≤ attained moment (the hull may only contain loads some admissible state carries). |
| Deviation | Up to 5.58 % of $M_{ref}$ (45.1 kNm, `example_external_tendon`) and 3.97 % (147.6 kNm, `example_staged_injection`) in the concavity alone, before AUD-D-1. |
| Proposal | Bring the non-convex domain representation (D0-3 option (c)) into the prestress scope; until then, measure and flag (option (b)). |

### AUD-F-4 — Prestressing steel in compression and over-crushed concrete on the hull [D3]

| Field | Content |
|---|---|
| Severity / direction | **S1** / non-conservative |
| Evidence | `probes/pF_hullvertex.py` on `F1_pretensioned_uls.yaml` (EXIT 0, `out/F4.log`): the sweep envelope is $\varepsilon \in [-37.5‰,\,+25.5‰]$, the strand limit shifted by $\varepsilon_{init}$. A hull vertex at N = −1391.3 kN, M = 278.3 kNm has bottom concrete at **−15.08 ‰** (limit −3.5 ‰) and the strand at −7.15 ‰, **σ_p = −1391.3 MPa** (yielded in compression). `prestressing_steel_ec2` defaults to `works_in_compression: true` in the loader (`spec.get("works_in_compression", True)`); F1 does not declare it. With `works_in_compression: false` (F1b, `out/F4b.log`) the envelope becomes [−6.0 ‰, +25.5 ‰] and the GenSec excess falls from 20.66 % to 7.05 % of $M_{ref}$; the residue is AUD-D-1 (edge at −3.6 ‰). |
| Expected | A strand carries no compression in a section; no plane with concrete beyond $\varepsilon_{cu2}$ enters the domain. |
| Deviation | +119.0 kNm (+20.7 % of $M_{ref}$) on F1; +146.4 kNm (+11.0 %) on `example_prestress.yaml`. |
| Proposal | No default for `works_in_compression` on prestressing steel (declared, physically `false`); the admissibility filter of AUD-D-1 applied per group. |

## A.5 D3 table (addendum)

| Tag | Closing external check (type) | Quantity to compare |
|---|---|---|
| AUD-D-4 | Non-linear section software with exact (non-hull) domain; tests on prestressed members under high axial load | M_Rd(N) on the tendon-compressed side |
| AUD-F-4 | Literature prestressed-section interaction diagram; independent software | N–M boundary of a pretensioned section |

Tag count: 2 `[D3]` in §A.4, 2 rows above.
