# ---------------------------------------------------------------------------
# GenSec — Copyright (c) 2026 Andrea Albero
#
# This file is part of GenSec.
#
# GenSec is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# GenSec is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public
# License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GenSec.  If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------
r"""
Runnable both-treatments post-tension toy (prestress Phase 4 / Step C).

Two independent purposes, kept separate:

1. **Engine self-validation (exact).**  A *linear-elastic* transfer
   section runs through :func:`solve_posttension_sequence` for both
   ``one_pass`` and ``coupled`` schemes; the per-tendon debit profile is
   compared to the closed-form reference
   :func:`sequential_posttension_loss` (meshed ``Ac, Sc, Ic``).  The two
   code paths must coincide to machine precision.

2. **Engineer's independent validation (VCASLU).**  A *realistic*
   section (real concrete + prestressing steel, same geometry/tendons)
   is stressed sequentially, then grouted, and the effective per-tendon
   prestress + reconciled ``eps_init`` are printed — the inputs an
   external tool (VCASLU) needs to reproduce the existing-prestress N-M
   domain.  The two grouting treatments — **rigorous** (grout: tendon
   enters the domain) vs **conservative** (never grout: tendon stays a
   load for life) — are both shown, so the demand/resistance split is
   explicit.

Independent validation against VCASLU is the engineer's, not the
engine's: this script produces the numbers to feed it, not a pass/fail.

Run from the project root::

    python run_posttension_sequence_validation.py
"""

import numpy as np

from gensec.geometry.section import RectSection
from gensec.geometry.fiber import Tendon
from gensec.solver.prestress_transfer import sequential_posttension_loss
from gensec.solver.posttension import solve_posttension_sequence, grout


# ---------------------------------------------------------------------------
#  Geometry / tendon layout (eccentric, sequential) — shared by both parts
# ---------------------------------------------------------------------------

B, H, N_Y = 350.0, 900.0, 360
EP = 195000.0
AP = 150.0 * 7          # 7-strand tendon, 150 mm^2/strand
SP0 = 1395.0            # jacking stress [MPa], net of friction + slip
TENDON_YS = [120.0, 200.0, 280.0]     # three tendons, low in the section
ORDER = [0, 1, 2]


# ---------------------------------------------------------------------------
#  Minimal linear-elastic material for the exact self-validation
# ---------------------------------------------------------------------------

class LinearElastic:
    r"""Constant-modulus law :math:`\sigma = E\,\varepsilon`."""

    def __init__(self, E, eps_lim=1.0):
        self.E, self._lim = E, eps_lim

    @property
    def eps_min(self):
        return -self._lim

    @property
    def eps_max(self):
        return self._lim

    def stress(self, eps):
        return self.E * eps

    def stress_array(self, eps):
        return self.E * np.asarray(eps, float)

    def tangent(self, eps):
        return self.E

    def tangent_array(self, eps):
        return np.full_like(np.asarray(eps, float), self.E)


def _meshed_props(section, y_ref):
    A = np.asarray(section.A_fibers, float)
    d = np.asarray(section.y_fibers, float) - y_ref
    return float(A.sum()), float((A * d).sum()), float((A * d * d).sum())


# ===========================================================================
#  PART 1 — engine self-validation (exact, linear-elastic)
# ===========================================================================

def part1_self_validation():
    print("=" * 70)
    print("PART 1 — engine self-validation vs closed form (linear-elastic)")
    print("=" * 70)
    EC = 31000.0
    bulk = LinearElastic(EC)
    strand = LinearElastic(EP)
    tendons = [Tendon(y=y, x=B / 2, material=strand, Ap=AP, eps_pe=0.0)
               for y in TENDON_YS]
    sec = RectSection(B=B, H=H, bulk_material=bulk, rebars=[],
                      n_fibers_y=N_Y, tendons=tendons)
    y_ref = sec.y_centroid
    Ac, Sc, Ic = _meshed_props(sec, y_ref)
    cf_tendons = [{"y": y, "Ap": AP, "sigma_p0": SP0} for y in TENDON_YS]

    for scheme in ("one_pass", "coupled"):
        res = solve_posttension_sequence(sec, sigma_p0=[SP0] * 3, Ep=EP,
                                         order=ORDER, scheme=scheme)
        ref = sequential_posttension_loss(
            EC, 0.0, EP, Ac=Ac, Sc=Sc, Ic=Ic,
            tendons=[dict(t) for t in cf_tendons], order=ORDER,
            y_ref=y_ref, scheme=scheme)
        dmax = float(np.max(np.abs(res.loss - ref["loss"])))
        print(f"\n  scheme = {scheme}")
        print(f"    driver loss   [MPa] = {np.round(res.loss, 4)}")
        print(f"    closed-form   [MPa] = {np.round(ref['loss'], 4)}")
        print(f"    max|Δ|        [MPa] = {dmax:.2e}   "
              f"(converged={res.converged}, "
              f"coupled_iters={res.coupled_iterations})")
        assert dmax < 1e-4, dmax
    print("\n  -> exact agreement (engine reproduces the closed form).")


# ===========================================================================
#  PART 2 — realistic section, both grouting treatments (VCASLU-pointable)
# ===========================================================================

def part2_realistic_both_treatments():
    print("\n" + "=" * 70)
    print("PART 2 — realistic section, rigorous vs conservative grouting")
    print("=" * 70)

    # Real materials.  Imported here so PART 1 stays dependency-light.
    from gensec.materials.concrete import Concrete
    from gensec.materials.steel import PrestressingSteel

    conc = Concrete(fck=45.0)
    strand = PrestressingSteel(f_p01d=1600.0 / 1.15, sigma_ud=1860.0,
                               eps_ud=0.0315, Ep=EP)
    tendons = [Tendon(y=y, x=B / 2, material=strand, Ap=AP, eps_pe=0.0)
               for y in TENDON_YS]
    sec = RectSection(B=B, H=H, bulk_material=conc, rebars=[],
                      n_fibers_y=N_Y, tendons=tendons)

    # Self-weight sollecitazione present at transfer (illustrative).
    base_Mx = 80.0e6      # N·mm (sagging) — adjust to your member

    res = solve_posttension_sequence(
        sec, sigma_p0=[SP0] * 3, Ep=EP, order=ORDER,
        base_Mx=base_Mx, scheme="one_pass")

    print(f"\n  jacking sigma_p0 [MPa] = {np.round(res.sigma_p0, 1)}")
    print(f"  loss  Δσ_p       [MPa] = {np.round(res.loss, 2)}")
    print(f"  effective σ_p    [MPa] = {np.round(res.sigma_p_after, 1)}")
    print(f"  eps_pe_eff       [-]   = {np.round(res.eps_pe_eff, 6)}")
    print(f"  eps_ref_grout    [-]   = {np.round(res.eps_ref_grout, 6)}")

    # --- Rigorous treatment: grout all, atomic action->element ----------
    g = grout(sec, res, indices=ORDER)
    print("\n  RIGOROUS (grout all): tendons enter the resistance domain.")
    print(f"    grouted tendons        = {g.grouted}")
    print(f"    dropped demand loads   = {len(g.dropped_actions)} "
          f"(one per grouted tendon)")
    print(f"    residual demand loads  = {len(g.residual_actions)}")
    for r in g.report["reconciliation"]:
        print(f"    tendon {r['tendon']}: eps_init = {r['eps_init']:.6e}, "
              f"σ_p target = {r['sigma_p_target']:.1f} MPa")
    print("    -> feed g.state into NMDiagram / `gensec run` to obtain the "
          "existing-prestress\n       N-M domain for the VCASLU cross-check.")

    # --- Conservative treatment: never grout ----------------------------
    eff = res.effective_actions()
    tot = res.total_effective_action()
    print("\n  CONSERVATIVE (never grout): each tendon stays a load for life.")
    for j, a in enumerate(eff):
        print(f"    tendon {j}: (N, Mx, My) = "
              f"({a.N:.3e}, {a.Mx:.3e}, {a.My:.3e}) N, N·mm")
    print(f"    total prestress action  = "
          f"(N, Mx, My) = ({tot.N:.3e}, {tot.Mx:.3e}, {tot.My:.3e})")
    print("    -> this triple is the demand-side prestress; the resistance "
          "domain is the\n       bare (tendon-free) section.")

    # --- What VCASLU receives -------------------------------------------
    print("\n  VCASLU cross-check inputs (independent validation is yours):")
    print(f"    section: {B:.0f} x {H:.0f} mm, fck = 45 MPa")
    for j, y in enumerate(TENDON_YS):
        print(f"    tendon {j}: y = {y:.0f} mm, Ap = {AP:.0f} mm^2, "
              f"effective σ_p = {res.sigma_p_after[j]:.1f} MPa "
              f"(P = {res.sigma_p_after[j]*AP/1e3:.1f} kN)")


if __name__ == "__main__":
    part1_self_validation()
    part2_realistic_both_treatments()
    print("\nDONE.")
