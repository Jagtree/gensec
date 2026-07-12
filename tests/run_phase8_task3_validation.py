# ---------------------------------------------------------------------------
# GenSec — Copyright (c) 2026 Andrea Albero
#
# This file is part of GenSec.  AGPL-3.0-or-later.
# ---------------------------------------------------------------------------
r"""
Phase-8 Task-3 closed-form validation: composite staged SLS (fork C4).

Each check compares :func:`gensec.solver.sls.verify_sls_staged` on a
masked composite against an independent 2-DOF analytic model (discrete
fibre sums *and* the exact-rectangle continuum), the Phase-7 idiom lifted
to two bulk zones with a per-zone locked-in datum.

Uniaxial, x-symmetric; plane :math:`(\varepsilon_0, \chi_x)` about the
pinned reference ``YR``; module convention
:math:`\varepsilon = \varepsilon_0 + \chi_x\,(y - y_R)`.

SLS moduli are supplied explicitly (Phase-7 D3: concrete moduli are
normative and never derived silently); ``StressLimits`` thresholds are
positive magnitudes.
"""
import numpy as np
from shapely.geometry import box

from gensec.materials.elastic import LinearElastic
from gensec.materials.steel import PrestressingSteel
from gensec.geometry.fiber import Tendon
from gensec.geometry.geometry import GenericSection
from gensec.geometry.primitives import rect_poly
from gensec.solver.section_state import SectionState
from gensec.solver.sls import verify_sls_staged
from gensec.materials.verification_limits import StressLimits

_N_PASS = 0
_N_FAIL = 0


def check(label, ok, detail=""):
    global _N_PASS, _N_FAIL
    if ok:
        _N_PASS += 1
        print(f"  [PASS] {label}")
    else:
        _N_FAIL += 1
        print(f"  [FAIL] {label}  {detail}")


def check_raises(label, exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        check(label, True)
    except Exception as e:                                    # noqa: BLE001
        check(label, False, f"raised {type(e).__name__}: {e}")
    else:
        check(label, False, "did not raise")


def banner(title):
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


# ==================================================================
#  Geometry / materials
# ==================================================================
E1, E2 = 35000.0, 31000.0
EP = 195000.0
B = 400.0
H1, H2 = 800.0, 200.0
H = H1 + H2
YR = H / 2.0                       # pinned reference (any consistent value)
AP = 1500.0
Y_T = 80.0
EPS_PE = -6.0e-3

MOD_C = {"precast": E1, "topping": E2}   # composite: both zones explicit
MOD_1 = {"precast": E1}                   # single zone: only what exists


def _ps():
    ps = PrestressingSteel(f_p01d=1391.3, Ep=EP)
    ps.name = "Y1860"
    return ps


def composite():
    r"""Prestressed composite: precast web + topping + one bottom tendon."""
    return GenericSection(
        polygon=rect_poly(B, H),
        bulk_material=LinearElastic(E=E1, name="precast"), rebars=[],
        bulk_materials=[(box(0, H1, B, H),
                         LinearElastic(E=E2, name="topping"), "topping")],
        tendons=[Tendon(y=Y_T, Ap=AP, material=_ps(), x=B / 2.0,
                        eps_pe=EPS_PE, name="T1")],
        mesh_size=50.0, n_grid_x=8, n_grid_y=40)


def composite_plain():
    r"""Composite without prestress — unambiguous extremes for T2."""
    return GenericSection(
        polygon=rect_poly(B, H),
        bulk_material=LinearElastic(E=E1, name="precast"), rebars=[],
        bulk_materials=[(box(0, H1, B, H),
                         LinearElastic(E=E2, name="topping"), "topping")],
        mesh_size=50.0, n_grid_x=8, n_grid_y=20)


def single_zone():
    r"""Precast web alone (the trivial / A-B path)."""
    return GenericSection(
        polygon=rect_poly(B, H1),
        bulk_material=LinearElastic(E=E1, name="precast"), rebars=[],
        tendons=[Tendon(y=Y_T, Ap=AP, material=_ps(), x=B / 2.0,
                        eps_pe=EPS_PE, name="T1")],
        mesh_size=50.0, n_grid_x=8, n_grid_y=16)


# ==================================================================
#  Independent 2-DOF model
# ==================================================================
def _fiber_stiffness(sec, active_zone_mask):
    mi = np.asarray(sec.mat_indices)
    fib_active = np.asarray(active_zone_mask, dtype=bool)[mi]
    Emap = np.where(mi == 0, E1, E2)[fib_active]
    A = sec.A_fibers[fib_active]
    ly = sec.y_fibers[fib_active] - YR
    return np.array([[np.sum(Emap * A), np.sum(Emap * A * ly)],
                     [np.sum(Emap * A * ly), np.sum(Emap * A * ly * ly)]])


def _continuum_stiffness(zones):
    K = np.zeros((2, 2))
    for y0, y1, E in zones:
        A = B * (y1 - y0)
        yc = 0.5 * (y0 + y1) - YR
        I = B * (y1 - y0) ** 3 / 12.0 + A * yc ** 2
        S = A * yc
        K += E * np.array([[A, S], [S, I]])
    return K


def _with_tendon(K, tendon_in):
    if not tendon_in:
        return K
    d = Y_T - YR
    dE = (EP - E1) * AP
    return K + dE * np.array([[1.0, d], [d, d * d]])


def _F0(tendon_in):
    if not tendon_in:
        return np.zeros(2)
    d = Y_T - YR
    return EP * AP * EPS_PE * np.array([1.0, d])


def _solve(K, F_ext, F0):
    return np.linalg.solve(K, np.asarray(F_ext, float) - F0)


def _sig_c(u, y, E):
    return E * (u[0] + u[1] * (y - YR))


def _state(sec, *, active, bonded, eps_t, bulk_active=None,
           bulk_planes=None, idx=0, label=""):
    n = int(sec.x_rebars.size) + int(getattr(sec, "x_tendons",
                                             np.empty(0)).size)
    eps = np.zeros(n, dtype=float)
    if n:
        eps[-1] = eps_t
    return SectionState(
        idx, np.asarray(active, bool), np.asarray(bonded, bool), eps,
        bulk_active=(None if bulk_active is None
                     else np.asarray(bulk_active, bool)),
        bulk_planes=(None if bulk_planes is None
                     else np.asarray(bulk_planes, float)),
        label=label)


# ==================================================================
def t1_transfer(sec):
    banner("T1 — composite transfer (precast alone, topping inactive)")
    M = -1.20e8
    s = _state(sec, active=[True], bonded=[True], eps_t=EPS_PE,
               bulk_active=[True, False], label="transfer")
    lim = StressLimits(name="tr", sigma_c_comp=30.0)
    res = verify_sls_staged(
        sec, [{"name": "transfer", "state": s, "increment": (0.0, M, 0.0),
               "limits": lim}],
        moduli=MOD_C, x_ref=B / 2.0, y_ref=YR)
    active = np.array([True, False])
    for tag, K in (("discrete", _with_tendon(_fiber_stiffness(sec, active),
                                             True)),
                   ("continuum", _with_tendon(
                       _continuum_stiffness([(0.0, H1, E1)]), True))):
        u = _solve(K, (0.0, M), _F0(True))
        s_bot, s_top = _sig_c(u, 0.0, E1), _sig_c(u, H1, E1)
        rc = res["stages"][0]["concrete"]
        exp_min, exp_max = min(s_bot, s_top), max(s_bot, s_top)
        tol = 5e-2 if tag == "continuum" else 1e-3
        check(f"T1 concrete extremes ({tag})",
              abs(rc["sigma_min_MPa"] - exp_min) < tol
              and abs(rc["sigma_max_MPa"] - exp_max) < tol,
              f"got=({rc['sigma_min_MPa']:.4f},{rc['sigma_max_MPa']:.4f}) "
              f"exp=({exp_min:.4f},{exp_max:.4f})")


def t2_service(sec):
    banner("T2 — service, two-stage superposition (§3), plain composite")
    Mg = 1.20e8            # sagging: bottom compressed
    dM = 2.60e8
    K_prec = _fiber_stiffness(sec, [True, False])
    u_sub = _solve(K_prec, (0.0, Mg), np.zeros(2))
    K_comp = _fiber_stiffness(sec, [True, True])
    u_inc = _solve(K_comp, (0.0, dM), np.zeros(2))
    # F7: the fibre kernel ADDS the bulk offset (integrator.py), which is
    # why ConstructionTimeline._auto_datum stores the NEGATED plane.  This
    # validator was written -- and, per 10_6 §Verification-honesty, NOT RUN
    # -- against the opposite assumption, and flagged it inline as
    # "# DATUM-CONVENTION".  The first run found it.  The datum of a zone
    # cast onto a substrate standing at u_sub is -u_sub.
    datum_top = [-u_sub[0], -u_sub[1], 0.0]
    planes = np.array([[0.0, 0.0, 0.0], datum_top], float)

    s0 = _state(sec, active=[], bonded=[], eps_t=0.0,
                bulk_active=[True, False], idx=0, label="precast")
    s1 = _state(sec, active=[], bonded=[], eps_t=0.0,
                bulk_active=[True, True], bulk_planes=planes,
                idx=1, label="service")
    res = verify_sls_staged(
        sec,
        [{"name": "precast", "state": s0, "increment": (0.0, Mg, 0.0)},
         {"name": "service", "state": s1, "increment": (0.0, dM, 0.0)}],
        moduli=MOD_C, x_ref=B / 2.0, y_ref=YR)
    rc = res["stages"][1]["concrete"]
    sub_bot = _sig_c(u_sub, 0.0, E1) + _sig_c(u_inc, 0.0, E1)   # robust
    top_top = _sig_c(u_inc, H, E2)                              # datum-dep.
    check("T2 substrate bottom (datum-robust superposition)",
          abs(rc["sigma_min_MPa"] - sub_bot) < 1e-3,
          f"got={rc['sigma_min_MPa']:.4f} exp={sub_bot:.4f}")
    # The GLOBAL maximum sits in the SUBSTRATE here (E1 = 35000 > E2 =
    # 31000, so the substrate's top out-stresses the topping's), and this
    # check used to compare it against the TOPPING's top -- two different
    # quantities.  Read the topping's own extreme.  Each zone has its own
    # f_ck anyway: a check on the global extreme alone is checking one
    # zone against another zone's limit.
    top = rc["by_zone"]["topping"]
    check("T2 topping top (its own extreme fibre)",
          abs(top["sigma_max_MPa"] - top_top) < 1e-3,
          f"got={top['sigma_max_MPa']:.4f} exp={top_top:.4f}")
    check("T2 the global max is in the SUBSTRATE, not the topping",
          rc["at_max"][1] <= H1 + 1e-9,
          f"global max {rc['sigma_max_MPa']:.4f} @ y={rc['at_max'][1]:.0f} "
          f"vs topping top {top['sigma_max_MPa']:.4f}")


def t3_decompression(sec):
    banner("T3 — decompression probe on the composite")
    Mg = -1.20e8
    s = _state(sec, active=[True], bonded=[True], eps_t=EPS_PE,
               bulk_active=[True, False], label="transfer")
    lim = StressLimits(name="dec", decompression=True, c_dec=40.0)
    res = verify_sls_staged(
        sec, [{"name": "transfer", "state": s, "increment": (0.0, Mg, 0.0),
               "limits": lim}],
        moduli=MOD_C, x_ref=B / 2.0, y_ref=YR)
    dec = [c for c in res["stages"][0]["checks"]
           if c["name"].startswith("decompression")]
    check("T3 one decompression check present", len(dec) == 1,
          f"found {len(dec)}")
    if dec:
        u = _solve(_with_tendon(_fiber_stiffness(sec, [True, False]), True),
                   (0.0, Mg), _F0(True))
        yp = dec[0]["probe_at"][1]                 # evaluate at real probe
        check("T3 probe stress matches closed form at probe_at",
              abs(dec[0]["sigma_probe_MPa"] - _sig_c(u, yp, E1)) < 1e-2,
              f"probe={dec[0]['sigma_probe_MPa']:.4f} "
              f"exp={_sig_c(u, yp, E1):.4f} @y={yp:.1f}")


def t4_present_mask(sec):
    banner("T4 — present-mask (topping fibres absent at transfer)")
    s = _state(sec, active=[True], bonded=[True], eps_t=EPS_PE,
               bulk_active=[True, False], label="transfer")
    res = verify_sls_staged(
        sec, [{"name": "transfer", "state": s,
               "increment": (0.0, -1.2e8, 0.0)}],
        moduli=MOD_C, x_ref=B / 2.0, y_ref=YR)
    rc = res["stages"][0]["concrete"]
    check("T4 extremes in precast zone (y < H1)",
          rc["at_min"][1] < H1 + 1e-6 and rc["at_max"][1] < H1 + 1e-6,
          f"at_min.y={rc['at_min'][1]:.1f} at_max.y={rc['at_max'][1]:.1f}")


def t5_ab_identity(sec1):
    banner("T5 — single-zone path reproduces the closed form")
    s = _state(sec1, active=[True], bonded=[True], eps_t=EPS_PE,
               label="one")
    res = verify_sls_staged(
        sec1, [{"name": "one", "state": s,
                "increment": (0.0, -1.2e8, 0.0)}],
        moduli=MOD_1, x_ref=B / 2.0, y_ref=YR)
    u = _solve(_with_tendon(_fiber_stiffness(sec1, [True]), True),
               (0.0, -1.2e8), _F0(True))
    exp_bot, exp_top = _sig_c(u, 0.0, E1), _sig_c(u, H1, E1)
    rc = res["stages"][0]["concrete"]
    check("T5 single-zone concrete extremes match closed form",
          abs(rc["sigma_min_MPa"] - min(exp_bot, exp_top)) < 1e-3
          and abs(rc["sigma_max_MPa"] - max(exp_bot, exp_top)) < 1e-3,
          f"got=({rc['sigma_min_MPa']:.5f},{rc['sigma_max_MPa']:.5f}) "
          f"exp=({min(exp_bot, exp_top):.5f},{max(exp_bot, exp_top):.5f})")


def t6_invariant_guard(sec):
    banner("T6 — INV2: persisting-zone datum change raises")
    p0 = np.array([[0.10, 0.0, 0.0], [0.0, 0.0, 0.0]], float)
    p1 = np.array([[0.20, 0.0, 0.0], [0.0, 0.0, 0.0]], float)
    s0 = _state(sec, active=[True], bonded=[True], eps_t=EPS_PE,
                bulk_active=[True, True], bulk_planes=p0, idx=0, label="a")
    s1 = _state(sec, active=[True], bonded=[True], eps_t=EPS_PE,
                bulk_active=[True, True], bulk_planes=p1, idx=1, label="b")
    check_raises(
        "T6 datum change on persisting zone raises", NotImplementedError,
        verify_sls_staged, sec,
        [{"name": "a", "state": s0, "increment": (0.0, 0.0, 0.0)},
         {"name": "b", "state": s1, "increment": (0.0, -1e8, 0.0)}],
        moduli=MOD_C, x_ref=B / 2.0, y_ref=YR)


def main():
    banner("GenSec — Phase-8 Task-3 (C4) composite-SLS validation")
    t1_transfer(composite())
    t2_service(composite_plain())
    t3_decompression(composite())
    t4_present_mask(composite())
    t5_ab_identity(single_zone())
    t6_invariant_guard(composite())
    banner(f"RESULT: {_N_PASS} passed, {_N_FAIL} failed")
    return 0 if _N_FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
