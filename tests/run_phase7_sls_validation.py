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
Phase-7 closed-form validation: SLS stress verification engine.

Every check compares :func:`gensec.solver.sls.verify_sls_staged`
against an **independent analytic model**: the uniaxial linear
2-DOF equilibrium

.. math::

    \mathbf{K}\,\mathbf{u} = \mathbf{F}_{\mathrm{ext}}
        - \mathbf{F}_0, \qquad
    \mathbf{u} = (\varepsilon_0, \chi_x),

with the stiffness assembled two independent ways:

- **discrete** — from the section's own fiber arrays (same quadrature
  as the solver), so agreement is expected to solver tolerance;
- **continuum** — from exact rectangle formulas plus differential
  point areas, so agreement is expected to mesh-quadrature accuracy
  (loose tolerance): this validates the *modeling*, not just the
  linear algebra.

Checked (V1–V10):

V1   transfer stresses + elastic shortening (discrete & continuum)
V2   service moment increment on an unchanged state
V3   loss stage (eps-only transition): redistribution at constant
     demand + telescoping identity ``S == total solve``
V4   decompression probe: geometry and pass/fail flip across the
     analytic decompression boundary
V5   fail-loud guards (missing modulus, dead override, leaving
     elements, compound transition, invalid LinearElastic)
V6   entering elements: staged attribution vs. total read
V7   uncracked-basis flag semantics (D4)
V8   affine/fiber accumulator consistency (debug flag, all runs)
V9   superposition sanity: two half increments == one increment
V10  per-stage transformed properties (exact, tendon differential
     area included)

Run:  ``python run_phase7_sls_validation.py``
Exit code 0 iff every check passes.
"""

import sys

import numpy as np

from gensec.materials.concrete import Concrete
from gensec.materials.elastic import LinearElastic
from gensec.materials.steel import PrestressingSteel, Steel
from gensec.materials.verification_limits import (
    StressLimits, ec2_stress_limits,
)
from gensec.geometry.fiber import RebarLayer, Tendon
from gensec.geometry.section import RectSection
from gensec.solver.section_state import SectionState
from gensec.solver.sls import (
    resolve_sls_moduli, sls_transformed_properties, sls_view,
    verify_sls_staged,
)


# ==================================================================
#  Model data
# ==================================================================

B, H = 300.0, 500.0
XR, YR = 150.0, 250.0            # demand reference point
EC, ES, EP = 32000.0, 200000.0, 195000.0
AS1 = AS2 = 628.0                # rebar layers [mm^2]
Y_S1, Y_S2 = 50.0, 450.0
AP, Y_T = 1500.0, 100.0          # tendon
EPS_PE = 0.006

N_FIB_Y = 200

_failures = []


def _check(label, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        _failures.append(label)


def _build_section():
    conc = Concrete(fck=35.0)
    conc.name = "C35"
    st = Steel(fyk=450.0)
    st.name = "B450C"
    ps = PrestressingSteel(f_p01d=1391.3, Ep=EP)
    ps.name = "Y1860"
    reb = [RebarLayer(y=Y_S1, As=AS1, material=st, x=XR),
           RebarLayer(y=Y_S2, As=AS2, material=st, x=XR)]
    ten = [Tendon(y=Y_T, Ap=AP, material=ps, x=XR,
                  eps_pe=EPS_PE, name="T1")]
    return RectSection(B, H, conc, reb, n_fibers_y=N_FIB_Y,
                       tendons=ten)


def _state(active, bonded, eps_t, idx=0, label="", beps=0.0):
    return SectionState(
        idx,
        np.asarray(active, dtype=bool),
        np.asarray(bonded, dtype=bool),
        np.array([0.0, 0.0, eps_t], dtype=float),
        bulk_eps_init=beps, label=label)


# ==================================================================
#  Analytic 2-DOF model (uniaxial: x-symmetric section)
# ==================================================================

def _assemble(sec, *, tendon_in, eps_pe, beps=0.0, discrete=True):
    r"""
    Assemble :math:`\mathbf{K}` (2x2) and the internal constant load
    :math:`\mathbf{F}_0` about ``(XR, YR)``.

    Internal actions of the substituted linear view:

    .. math::

        N_{\mathrm{int}} &= \sum_f E_c(\varepsilon_f
            + \varepsilon_b)\,A_f
          + \sum_e \big[E_e(\varepsilon_e + \varepsilon_{0,e})
            - E_c(\varepsilon_e + \varepsilon_b)\big]A_e ,

    where the bracket is the embedded net contribution and
    :math:`\varepsilon = \varepsilon_0 + \chi_x (y - y_R)`.
    Collecting linear terms gives the stiffness with differential
    element areas :math:`(E_e - E_c)A_e`; collecting constants gives

    .. math::

        \mathbf{F}_0 = E_p A_p \varepsilon_{pe}
            \begin{bmatrix} 1 \\ y_t - y_R \end{bmatrix}
          + E_c\,\varepsilon_b
            \begin{bmatrix} A_{\mathrm{net}} \\
                            S_{\mathrm{net}} \end{bmatrix},

    with :math:`A_{\mathrm{net}} = A_c - \sum_e A_e` (the bulk offset
    also enters the displaced-bulk subtraction of every embedded
    element).

    Parameters
    ----------
    sec : GenericSection
    tendon_in : bool
        Whether the tendon is in the resistance set.
    eps_pe : float
        Tendon prestrain.
    beps : float, optional
        Bulk imposed strain.
    discrete : bool, optional
        ``True``: fiber-array sums (solver quadrature).
        ``False``: exact continuum rectangle.

    Returns
    -------
    K : numpy.ndarray, shape (2, 2)
    F0 : numpy.ndarray, shape (2,)
    """
    if discrete:
        Af = sec.A_fibers
        yf = sec.y_fibers - YR
        A_c, S_c, I_c = Af.sum(), (Af * yf).sum(), (Af * yf**2).sum()
    else:
        A_c = B * H
        S_c = 0.0
        I_c = B * H**3 / 12.0

    elems = [(ES, AS1, Y_S1 - YR), (ES, AS2, Y_S2 - YR)]
    if tendon_in:
        elems.append((EP, AP, Y_T - YR))

    EA = EC * A_c + sum((E - EC) * A for E, A, _ in elems)
    ESm = EC * S_c + sum((E - EC) * A * d for E, A, d in elems)
    EI = EC * I_c + sum((E - EC) * A * d**2 for E, A, d in elems)
    K = np.array([[EA, ESm], [ESm, EI]])

    F0 = np.zeros(2)
    if tendon_in:
        F0 += EP * AP * eps_pe * np.array([1.0, Y_T - YR])
    if beps:
        A_net = A_c - sum(A for _, A, _ in elems)
        S_net = S_c - sum(A * d for _, A, d in elems)
        F0 += EC * beps * np.array([A_net, S_net])
    return K, F0


def _solve_u(sec, F_ext, *, tendon_in, eps_pe, beps=0.0,
             discrete=True):
    """Analytic plane ``(eps0, chi_x)`` for demand ``(N, Mx)``."""
    K, F0 = _assemble(sec, tendon_in=tendon_in, eps_pe=eps_pe,
                      beps=beps, discrete=discrete)
    return np.linalg.solve(K, np.asarray(F_ext, dtype=float) - F0)


def _sig_c(u, y, beps=0.0):
    """Concrete stress at height ``y`` for plane ``u``."""
    return EC * (u[0] + u[1] * (y - YR) + beps)


def _sig_el(u, y, E, eps0_el=0.0):
    """Element (material) stress at height ``y`` for plane ``u``."""
    return E * (u[0] + u[1] * (y - YR) + eps0_el)


def _elem_sigma(res, k, union_index):
    """Accumulated element stress from a stage record."""
    for e in res["stages"][k]["elements"]:
        if e["union_index"] == union_index:
            return e["sigma_MPa"]
    raise KeyError(union_index)


# ==================================================================
#  V1 — transfer
# ==================================================================

def v1_transfer(sec, moduli):
    M0 = 40.0e6            # self-weight moment at transfer [N*mm]
    s0 = _state([1, 1, 1], [1, 1, 1], EPS_PE, 0, "transfer")
    res = verify_sls_staged(
        sec, [{"name": "transfer", "state": s0,
               "increment": (0.0, M0, 0.0)}],
        moduli=moduli, x_ref=XR, y_ref=YR,
        debug_check_affine=True)

    # F6 / Option A: EN 1992-1-1 7.2 limits sigma_c at the EXTREME FIBRE
    # of the section, not at the centroid of the outermost row of a
    # discretisation.  The strain plane is exactly linear, so the face
    # value is a closed form -- evaluating the reference at
    # y_fibers.min()/.max() shares the implementation's half-fibre inset
    # and therefore cannot ever catch it.
    y_bot, y_top = 0.0, H
    c = res["stages"][0]["concrete"]

    for tag, discrete, atol in (("discrete", True, 1e-3),
                                ("continuum", False, None)):
        u = _solve_u(sec, (0.0, M0), tendon_in=True, eps_pe=EPS_PE,
                     discrete=discrete)
        pairs = [
            ("sigma_c bottom", c["sigma_min_MPa"], _sig_c(u, y_bot)),
            ("sigma_c top", c["sigma_max_MPa"], _sig_c(u, y_top)),
            ("sigma_p", _elem_sigma(res, 0, 2),
             _sig_el(u, Y_T, EP, EPS_PE)),
            ("sigma_s bottom", _elem_sigma(res, 0, 0),
             _sig_el(u, Y_S1, ES)),
        ]
        for name, got, ref in pairs:
            if atol is not None:
                ok = abs(got - ref) < atol
                det = f"got {got:.4f}, ref {ref:.4f} MPa"
            else:
                ok = abs(got - ref) < 5e-3 * max(1.0, abs(ref))
                det = f"got {got:.4f}, ref {ref:.4f} MPa (mesh tol)"
            _check(f"V1 {tag}: {name}", ok, det)

    # Elastic shortening emerges: sigma_p below the jacking stress by
    # exactly Ep * eps_sec(y_t).
    u = _solve_u(sec, (0.0, M0), tendon_in=True, eps_pe=EPS_PE)
    sp = _elem_sigma(res, 0, 2)
    _check("V1 elastic shortening sign", sp < EP * EPS_PE,
           f"sigma_p {sp:.2f} < jacking {EP * EPS_PE:.2f} MPa")
    _check("V1 elastic shortening value",
           abs((EP * EPS_PE - sp)
               - (-EP * (u[0] + u[1] * (Y_T - YR)))) < 1e-3)
    return res


# ==================================================================
#  V2 / V3 — service increment, loss stage, telescoping
# ==================================================================

def v2_v3_service_and_loss(sec, moduli):
    M0, dM = 40.0e6, 80.0e6
    EPS_PE2 = 0.0055        # after time-dependent losses
    s0 = _state([1, 1, 1], [1, 1, 1], EPS_PE, 0, "transfer")
    s1 = _state([1, 1, 1], [1, 1, 1], EPS_PE, 1, "service")
    s2 = _state([1, 1, 1], [1, 1, 1], EPS_PE2, 2, "losses")
    res = verify_sls_staged(
        sec,
        [{"name": "transfer", "state": s0, "increment": (0, M0, 0)},
         {"name": "service", "state": s1, "increment": (0, dM, 0)},
         {"name": "losses", "state": s2}],
        moduli=moduli, x_ref=XR, y_ref=YR,
        debug_check_affine=True)

    y_bot = 0.0

    # V2: increment on an unchanged state — accumulated equals the
    # total solve at (0, M0 + dM) (telescoping, same view).
    u01 = _solve_u(sec, (0.0, M0 + dM), tendon_in=True,
                   eps_pe=EPS_PE)
    c1 = res["stages"][1]["concrete"]
    _check("V2 service: sigma_c bottom",
           abs(c1["sigma_min_MPa"] - _sig_c(u01, y_bot)) < 1e-3,
           f"got {c1['sigma_min_MPa']:.4f}")
    _check("V2 service: sigma_p",
           abs(_elem_sigma(res, 1, 2)
               - _sig_el(u01, Y_T, EP, EPS_PE)) < 1e-3)

    # V3: eps-only transition. Constant masks and moduli make the
    # scheme telescope to the total solve on the final state.
    u2 = _solve_u(sec, (0.0, M0 + dM), tendon_in=True,
                  eps_pe=EPS_PE2)
    c2 = res["stages"][2]["concrete"]
    _check("V3 losses: sigma_c bottom (telescoped total)",
           abs(c2["sigma_min_MPa"] - _sig_c(u2, y_bot)) < 1e-3,
           f"got {c2['sigma_min_MPa']:.4f}, "
           f"ref {_sig_c(u2, y_bot):.4f}")
    _check("V3 losses: sigma_p (telescoped total)",
           abs(_elem_sigma(res, 2, 2)
               - _sig_el(u2, Y_T, EP, EPS_PE2)) < 1e-3)

    # Redistribution at constant demand: the tendon stress drop is
    # smaller than the free drop Ep*d_eps_pe (the section springs
    # back), and the concrete decompresses at the tendon level.
    dsp = _elem_sigma(res, 2, 2) - _elem_sigma(res, 1, 2)
    _check("V3 redistribution: |d sigma_p| < Ep*|d eps_pe|",
           0.0 < -dsp < EP * (EPS_PE - EPS_PE2),
           f"d sigma_p {dsp:.3f} MPa, free "
           f"{-EP * (EPS_PE - EPS_PE2):.3f} MPa")
    return res


# ==================================================================
#  V4 — decompression boundary
# ==================================================================

def v4_decompression(sec, moduli):
    r"""
    Find the analytic moment :math:`M^\*` at which the accumulated
    concrete stress at the tendon's probe point is zero, then check
    the engine verdict flips across it.

    Both the probe side and the pass/fail direction are **derived**
    from the analytic model, not hardcoded: the probe sits at
    :math:`y_t + c_{\mathrm{dec}}\,\operatorname{sign}(c_y)` (the
    tensile side, :math:`c_y = E_c\,\chi_x` being the field
    gradient), and since :math:`\sigma_{\mathrm{probe}}(M)` is affine
    in :math:`M`, the violated side of :math:`M^\*` is the one where
    the slope drives the probe stress tensile.
    """
    c_dec = 25.0
    s0 = _state([1, 1, 1], [1, 1, 1], EPS_PE, 0, "t")

    def chi(M):
        return _solve_u(sec, (0.0, M),
                        tendon_in=True, eps_pe=EPS_PE)[1]

    def sig_at(y, M):
        u = _solve_u(sec, (0.0, M), tendon_in=True, eps_pe=EPS_PE)
        return _sig_c(u, y)

    # The engine probes at y_t + c_dec * sign(c_y), c_y = Ec*chi_x
    # of the accumulated field.  Near the decompression boundary the
    # valid side is the FIXED POINT of that rule: side s such that
    # the root M*_s of sigma(y_t + s*c_dec; M) = 0 has
    # sign(chi(M*_s)) = s.  Both sigma and chi are affine in M, so
    # two samples give exact roots.
    M_a, M_b = 100.0e6, 300.0e6
    M_star = y_probe = side = slope = None
    for s in (+1.0, -1.0):
        y_s = Y_T + s * c_dec
        sa, sb = sig_at(y_s, M_a), sig_at(y_s, M_b)
        sl = (sb - sa) / (M_b - M_a)
        root = M_a - sa / sl
        if np.sign(chi(root)) == s:
            M_star, y_probe, side, slope = root, y_s, s, sl
            break
    _check("V4 probe side fixed point exists", M_star is not None,
           f"M* = {M_star / 1e6 if M_star else float('nan'):.1f} "
           f"kNm, probe y = {y_probe}")
    if M_star is None:
        return
    ok_chi = (np.sign(chi(M_star - 2.0e6))
              == np.sign(chi(M_star + 2.0e6)) == side)
    _check("V4 probe side well-defined across M*", ok_chi)

    lim = StressLimits(name="dec", decompression=True, c_dec=c_dec)
    verdicts = {}
    for tag, M in (("below", M_star - 2.0e6),
                   ("above", M_star + 2.0e6)):
        res = verify_sls_staged(
            sec, [{"name": "t", "state": s0,
                   "increment": (0.0, M, 0.0), "limits": lim}],
            moduli=moduli, x_ref=XR, y_ref=YR,
            debug_check_affine=True)
        dec = [c for c in res["stages"][0]["checks"]
               if c["name"].startswith("decompression")][0]
        verdicts[tag] = dec

    # slope < 0: probe stress decreases with M -> passes above M*.
    pass_side, fail_side = (("above", "below") if slope < 0
                            else ("below", "above"))
    _check(f"V4 decompression passes {pass_side} M*",
           verdicts[pass_side]["passed"],
           f"sigma_probe "
           f"{verdicts[pass_side]['sigma_probe_MPa']:.4f}")
    _check(f"V4 decompression fails {fail_side} M*",
           not verdicts[fail_side]["passed"],
           f"sigma_probe "
           f"{verdicts[fail_side]['sigma_probe_MPa']:.4f}")
    _check("V4 probe geometry (tensile side of the tendon)",
           verdicts[fail_side]["probe_at"] == (XR, y_probe),
           f"probe at {verdicts[fail_side]['probe_at']}, "
           f"expected {(XR, y_probe)}")
    _check("V4 probe stress magnitude at boundary",
           abs(verdicts["above"]["sigma_probe_MPa"]) < 0.2
           and abs(verdicts["below"]["sigma_probe_MPa"]) < 0.2)


# ==================================================================
#  V5 — fail-loud guards
# ==================================================================

def v5_guards(sec, moduli):
    s_all = _state([1, 1, 1], [1, 1, 1], EPS_PE, 0)
    s_noT = _state([1, 1, 0], [1, 1, 1], EPS_PE, 1)
    s_loss = _state([1, 1, 1], [1, 1, 1], 0.0055, 1)

    def _raises(exc, fn, label):
        try:
            fn()
        except exc:
            _check(label, True)
        except Exception as e:               # noqa: BLE001
            _check(label, False, f"wrong exception {type(e).__name__}")
        else:
            _check(label, False, "no exception")

    _raises(ValueError,
            lambda: verify_sls_staged(
                sec, [{"name": "s", "state": s_all}], moduli={}),
            "V5 missing concrete modulus raises")
    _raises(ValueError,
            lambda: verify_sls_staged(
                sec, [{"name": "s", "state": s_all}],
                moduli={**moduli, "Ghost": 1.0e4}),
            "V5 dead override raises")
    _raises(NotImplementedError,
            lambda: verify_sls_staged(
                sec, [{"name": "a", "state": s_all},
                      {"name": "b", "state": s_noT}],
                moduli=moduli),
            "V5 leaving element raises")

    s_enterT = _state([1, 1, 0], [1, 1, 1], EPS_PE, 0)
    s_comp = _state([1, 1, 1], [1, 1, 1], EPS_PE, 1, beps=-1e-4)
    _raises(ValueError,
            lambda: verify_sls_staged(
                sec, [{"name": "a", "state": s_enterT},
                      {"name": "b", "state": s_comp}],
                moduli=moduli),
            "V5 compound transition raises")
    _raises(ValueError, lambda: LinearElastic(E=0.0),
            "V5 LinearElastic(E=0) raises")
    _raises(ValueError, lambda: LinearElastic(E=-1.0),
            "V5 LinearElastic(E<0) raises")
    _raises(ValueError,
            lambda: StressLimits(sigma_c_comp=-3.0),
            "V5 StressLimits negative threshold raises")
    _raises(ValueError,
            lambda: ec2_stress_limits("transfer"),
            "V5 ec2_stress_limits without fck raises")
    _raises(ValueError,
            lambda: ec2_stress_limits("transfer", fck=30.0, k9=1.0),
            "V5 ec2_stress_limits unknown override raises")

    # eps-only + loss path still healthy after the guard batch.
    res = verify_sls_staged(
        sec, [{"name": "a", "state": s_all},
              {"name": "b", "state": s_loss}],
        moduli=moduli, x_ref=XR, y_ref=YR)
    _check("V5 loss transition runs", res["stages"][1] is not None)


# ==================================================================
#  V6 — entering elements: staged attribution
# ==================================================================

def v6_entering(sec, moduli):
    r"""
    Tendon enters the resistance set at stage 1 (activation
    semantics).  Expected:

    - the tendon initialises from the stage-1 **total read**
      :math:`\sigma(\mathcal{V}_1, F_1)` (engine convention:
      strain-compatible from entry; physical for grouted tendons via
      the reconciled prestrain);
    - a persisting rebar accumulates
      :math:`\sigma(\mathcal{V}_0, F_0) +
      [\sigma(\mathcal{V}_1, F_1) - \sigma(\mathcal{V}_1, F_0)]` —
      staged attribution, *different* from the total solve on the
      final view.
    """
    M0, M1 = 40.0e6, 120.0e6
    s0 = _state([1, 1, 0], [1, 1, 1], EPS_PE, 0, "no tendon")
    s1 = _state([1, 1, 1], [1, 1, 1], EPS_PE, 1, "tendon in")
    res = verify_sls_staged(
        sec,
        [{"name": "a", "state": s0, "increment": (0, M0, 0)},
         {"name": "b", "state": s1, "increment": (0, M1 - M0, 0)}],
        moduli=moduli, x_ref=XR, y_ref=YR,
        debug_check_affine=True)

    u0 = _solve_u(sec, (0.0, M0), tendon_in=False, eps_pe=0.0)
    u1_hi = _solve_u(sec, (0.0, M1), tendon_in=True, eps_pe=EPS_PE)
    u1_lo = _solve_u(sec, (0.0, M0), tendon_in=True, eps_pe=EPS_PE)

    sp_ref = _sig_el(u1_hi, Y_T, EP, EPS_PE)
    _check("V6 entering tendon = total read",
           abs(_elem_sigma(res, 1, 2) - sp_ref) < 1e-3,
           f"got {_elem_sigma(res, 1, 2):.4f}, ref {sp_ref:.4f}")

    ss_ref = (_sig_el(u0, Y_S1, ES)
              + _sig_el(u1_hi, Y_S1, ES) - _sig_el(u1_lo, Y_S1, ES))
    _check("V6 persisting rebar = staged attribution",
           abs(_elem_sigma(res, 1, 0) - ss_ref) < 1e-3,
           f"got {_elem_sigma(res, 1, 0):.4f}, ref {ss_ref:.4f}")

    ss_total = _sig_el(u1_hi, Y_S1, ES)
    _check("V6 staged != total on final view (attribution matters)",
           abs(ss_ref - ss_total) > 1.0,
           f"staged {ss_ref:.3f} vs total {ss_total:.3f} MPa")


# ==================================================================
#  V7 — uncracked-basis flag (D4)
# ==================================================================

def v7_basis_flag(sec, moduli):
    s0 = _state([1, 1, 1], [1, 1, 1], EPS_PE, 0)
    lim = ec2_stress_limits("characteristic", fck=35.0, fyk=450.0,
                            fpk=1860.0, fct_eff=3.2)
    res = verify_sls_staged(
        sec, [{"name": "big M", "state": s0,
               "increment": (0.0, 600.0e6, 0.0), "limits": lim}],
        moduli=moduli, x_ref=XR, y_ref=YR)
    st = res["stages"][0]
    _check("V7 basis violated flag", st["uncracked_basis_violated"])
    _check("V7 stage not verified", st["verified"] is False)
    dep = [c for c in st["checks"] if not c.get("skipped")
           and c["basis_dependent"]]
    _check("V7 basis-dependent checks marked invalid",
           all(c.get("basis_valid") is False for c in dep))
    _check("V7 informative values retained",
           all("value_MPa" in c or "sigma_probe_MPa" in c
               for c in dep))

    # Member-level unbonded check is basis-independent.
    lim2 = ec2_stress_limits("characteristic", fck=35.0,
                             fpk=1860.0, fct_eff=3.2)
    res2 = verify_sls_staged(
        sec, [{"name": "u", "state": s0,
               "increment": (0.0, 600.0e6, 0.0), "limits": lim2,
               "unbonded_sigma_p": {"T_ext": 1200.0}}],
        moduli=moduli, x_ref=XR, y_ref=YR)
    ub = [c for c in res2["stages"][0]["checks"]
          if c["name"] == "tendon_stress[T_ext]"][0]
    _check("V7 unbonded member-level check present",
           ub["provenance"] == "member_level" and ub["passed"]
           and "basis_valid" not in ub)


# ==================================================================
#  V9 — superposition sanity
# ==================================================================

def v9_superposition(sec, moduli):
    M = 120.0e6
    mk = lambda i: _state([1, 1, 1], [1, 1, 1], EPS_PE, i)  # noqa: E731
    res_one = verify_sls_staged(
        sec, [{"name": "a", "state": mk(0),
               "increment": (0.0, M, 0.0)}],
        moduli=moduli, x_ref=XR, y_ref=YR)
    res_two = verify_sls_staged(
        sec, [{"name": "a", "state": mk(0),
               "increment": (0.0, M / 2, 0.0)},
              {"name": "b", "state": mk(1),
               "increment": (0.0, M / 2, 0.0)}],
        moduli=moduli, x_ref=XR, y_ref=YR)
    c1 = res_one["stages"][0]["concrete"]
    c2 = res_two["stages"][1]["concrete"]
    ok = (abs(c1["sigma_min_MPa"] - c2["sigma_min_MPa"]) < 1e-6
          and abs(c1["sigma_max_MPa"] - c2["sigma_max_MPa"]) < 1e-6
          and abs(_elem_sigma(res_one, 0, 2)
                  - _elem_sigma(res_two, 1, 2)) < 1e-6)
    _check("V9 two half increments == one increment", ok)


# ==================================================================
#  V10 — transformed properties
# ==================================================================

def v10_transformed(sec, moduli):
    s0 = _state([1, 1, 1], [1, 1, 1], EPS_PE, 0)
    p = sls_transformed_properties(sec, s0, moduli=moduli)
    A_ref = (B * H + (ES / EC - 1.0) * (AS1 + AS2)
             + (EP / EC - 1.0) * AP)
    _check("V10 transformed area (tendon differential included)",
           abs(p.area - A_ref) < 1e-6 * A_ref,
           f"got {p.area:.2f}, ref {A_ref:.2f} mm^2")

    s_noT = _state([1, 1, 0], [1, 1, 1], EPS_PE, 0)
    p2 = sls_transformed_properties(sec, s_noT, moduli=moduli)
    A_ref2 = B * H + (ES / EC - 1.0) * (AS1 + AS2)
    _check("V10 transformed area follows the state",
           abs(p2.area - A_ref2) < 1e-6 * A_ref2,
           f"got {p2.area:.2f}, ref {A_ref2:.2f} mm^2")

    # The SLS view swaps materials without touching the base section.
    modmap = resolve_sls_moduli(sec, moduli)
    from gensec.solver.section_state import materialize_view
    vw = sls_view(materialize_view(sec, s0), modmap)
    _check("V10 sls_view leaves base materials untouched",
           isinstance(sec.bulk_material, Concrete)
           and isinstance(vw.bulk_material, LinearElastic)
           and sec.rebars[0].material is not vw.rebars[0].material)


# ==================================================================
#  Main
# ==================================================================

def main():
    sec = _build_section()
    moduli = {"C35": EC}

    print("=" * 64)
    print("Phase 7 — SLS stress verification: closed-form validation")
    print("=" * 64)
    v1_transfer(sec, moduli)
    v2_v3_service_and_loss(sec, moduli)
    v4_decompression(sec, moduli)
    v5_guards(sec, moduli)
    v6_entering(sec, moduli)
    v7_basis_flag(sec, moduli)
    v9_superposition(sec, moduli)
    v10_transformed(sec, moduli)

    print("-" * 64)
    if _failures:
        print(f"{len(_failures)} FAILURE(S):")
        for f in _failures:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
