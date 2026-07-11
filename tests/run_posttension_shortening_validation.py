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
Validation of the closed-form sequential post-tension elastic-shortening
reference (:func:`gensec.solver.prestress_transfer.sequential_posttension_loss`).

This is the **independent safety net** for the future Phase-4 fiber-method
driver ``solve_posttension_sequence``, exactly as
:func:`elastic_shortening_loss` was the net for pre-tensioning in Phase 2:
a pure transformed-section computation, no fiber model, against which the
fiber driver will be regression-checked.

Run from the project root:

.. code-block:: console

    python run_posttension_shortening_validation.py

Checks (all closed-form or self-referential — no external tool needed):

1. **Degenerate collapse** — n identical centroidal tendons, sequential,
   symmetric section: the general routine must reproduce the textbook
   per-tendon debit profile and the (n-1)/2 mean-debit rule **exactly**.
2. **Order dependence** — reversing the stressing order mirrors the debit
   profile (the last tendon stressed never loses; the first loses most).
3. **Eccentric two-tendon** — closed hand calc, including the physically
   correct case where stressing an eccentric tendon *raises* the stress of
   an opposite-side tendon (negative debit) — which an axial textbook
   formula cannot represent.
4. **Coupled vs one-pass** — the self-consistent scheme yields a strictly
   smaller (or equal) loss than the single-pass scheme, by the second-order
   feedback amount (~0.3 %), confirming why EC2 stops at the explicit form.
5. **Frame invariance** — the same physical problem expressed about a
   non-centroidal reference (Sc != 0, hence the full coupled 2x2 with
   ES != 0) gives identical debits, to machine precision.
"""

import numpy as np

from gensec.solver.prestress_transfer import sequential_posttension_loss

# Common elastic data
EC = 31000.0      # concrete modulus at transfer [MPa]
EP = 195000.0     # tendon modulus [MPa]
B, H = 300.0, 700.0
AC = B * H
IC_CENT = B * H ** 3 / 12.0
SP0 = 1395.0      # jacking stress [MPa]
AP = 150.0        # strand area [mm2]


def test_degenerate_collapse():
    print("[1] degenerate: n identical centroidal tendons -> textbook")
    n = 4
    tendons = [{"y": 0.0, "Ap": AP, "sigma_p0": SP0} for _ in range(n)]
    res = sequential_posttension_loss(
        EC, 0.0, EP, Ac=AC, Sc=0.0, Ic=IC_CENT,
        tendons=tendons, y_ref=0.0, scheme="one_pass")

    unit = EP * SP0 * AP / (EC * AC)             # debit per later event
    expected = np.array([(n - 1 - i) * unit for i in range(n)])
    assert np.max(np.abs(res["loss"] - expected)) < 1e-9, \
        (res["loss"], expected)
    # mean debit = (n-1)/2 * unit
    assert abs(res["loss"].mean() - (n - 1) / 2 * unit) < 1e-9
    print(f"    per-tendon debit {np.round(res['loss'], 3)} MPa; "
          f"mean = (n-1)/2*unit: exact")


def test_order_dependence():
    print("[2] order dependence: reverse mirrors the debit profile")
    n = 4
    tendons = [{"y": 0.0, "Ap": AP, "sigma_p0": SP0} for _ in range(n)]
    fwd = sequential_posttension_loss(
        EC, 0.0, EP, Ac=AC, Sc=0.0, Ic=IC_CENT,
        tendons=tendons, scheme="one_pass")
    rev = sequential_posttension_loss(
        EC, 0.0, EP, Ac=AC, Sc=0.0, Ic=IC_CENT,
        tendons=tendons, order=[3, 2, 1, 0], scheme="one_pass")
    assert np.allclose(fwd["loss"], rev["loss"][::-1])
    print(f"    forward {np.round(fwd['loss'], 2)} | "
          f"reverse {np.round(rev['loss'], 2)}: mirrored")


def test_eccentric_two_tendon():
    print("[3] eccentric two-tendon: hand calc incl. negative debit")
    e = 250.0
    tendons = [{"y": -e, "Ap": AP, "sigma_p0": SP0},
               {"y": +e, "Ap": AP, "sigma_p0": SP0}]
    res = sequential_posttension_loss(
        EC, 0.0, EP, Ac=AC, Sc=0.0, Ic=IC_CENT,
        tendons=tendons, y_ref=0.0, scheme="one_pass")
    # stressing tendon@+e: N=-P, M=-P*e; concrete strain at y=-e:
    P = SP0 * AP
    de0 = -P / (EC * AC)
    dchi = -P * e / (EC * IC_CENT)
    deps_first = de0 + dchi * (-e)
    expected0 = -EP * deps_first
    assert abs(res["loss"][0] - expected0) < 1e-9
    assert abs(res["loss"][1]) < 1e-12      # last stressed, no loss
    sign = "raises" if expected0 < 0 else "lowers"
    print(f"    tendon0 debit {res['loss'][0]:.4f} MPa "
          f"({sign} its stress) == hand calc; tendon1 = 0: exact")


def test_coupled_vs_one_pass():
    print("[4] coupled <= one_pass (second-order feedback)")
    n = 4
    tendons = [{"y": 0.0, "Ap": AP, "sigma_p0": SP0} for _ in range(n)]
    op = sequential_posttension_loss(
        EC, 0.0, EP, Ac=AC, Sc=0.0, Ic=IC_CENT,
        tendons=tendons, scheme="one_pass")
    cp = sequential_posttension_loss(
        EC, 0.0, EP, Ac=AC, Sc=0.0, Ic=IC_CENT,
        tendons=tendons, scheme="coupled")
    assert cp["loss"].mean() <= op["loss"].mean() + 1e-12
    rel = (op["loss"].mean() - cp["loss"].mean()) / op["loss"].mean()
    assert 0.0 < rel < 0.02, rel
    print(f"    one_pass mean {op['loss'].mean():.4f} | "
          f"coupled mean {cp['loss'].mean():.4f} "
          f"(smaller by {rel*100:.3f} %)")


def test_frame_invariance():
    print("[5] frame invariance: non-centroidal reference (Sc!=0)")
    yc = H / 2
    tendons = [{"y": 350.0, "Ap": AP, "sigma_p0": SP0},
               {"y": 150.0, "Ap": AP, "sigma_p0": SP0}]
    # reference at the bottom fibre: Sc != 0, Ic by parallel axis
    Sc_bot = AC * (yc - 0.0)
    Ic_bot = IC_CENT + AC * (yc - 0.0) ** 2
    r_bot = sequential_posttension_loss(
        EC, 0.0, EP, Ac=AC, Sc=Sc_bot, Ic=Ic_bot,
        tendons=[dict(d) for d in tendons], y_ref=0.0, scheme="one_pass")
    # reference at the centroid
    r_cen = sequential_posttension_loss(
        EC, 0.0, EP, Ac=AC, Sc=0.0, Ic=IC_CENT,
        tendons=[dict(d) for d in tendons], y_ref=yc, scheme="one_pass")
    diff = np.max(np.abs(r_bot["loss"] - r_cen["loss"]))
    assert diff < 1e-9, diff
    print(f"    debit identical across frames to {diff:.1e} MPa "
          f"(ES!=0 coupled 2x2 exercised)")


if __name__ == "__main__":
    test_degenerate_collapse()
    test_order_dependence()
    test_eccentric_two_tendon()
    test_coupled_vs_one_pass()
    test_frame_invariance()
    print("\nALL POST-TENSION SHORTENING VALIDATIONS PASSED")
