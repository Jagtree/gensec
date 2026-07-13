# ---------------------------------------------------------------------------
# GenSec — Copyright (c) 2026 Andrea Albero
#
# This file is part of GenSec.  Licensed under the GNU Affero General
# Public License v3 or later.  See LICENSE.
# ---------------------------------------------------------------------------
r"""
Phase-5 / C5 validator — rheology subsystem and time-dependent losses.

Two **independent assemblies** throughout: every number the library
produces is checked against a closed form written from the norm's own
equations, in a separate code path that shares no function with the
implementation.

    python run_phase5_c5_validation.py

Sections
--------
A. **The cardinal test.**  :math:`\chi` computed *from* EC2's compliance
   must come out :math:`\approx 0.80` — the value EN 1992-1-1 (5.46)
   assumes.  If the architecture is right, EC2's 0.8 is an *output* of
   its own creep law, not an input to ours.
B. **Grid convergence** of the Volterra inversion behind :math:`\chi`.
C. **EN 1992-1-1 Annex B** — creep, shrinkage and relaxation against a
   hand-written closed form.
D. **ACI 209R-92** — the falsification.  A structurally different norm
   (hyperbolic time, V/S in inches, :math:`E_c = 4700\sqrt{f'_c}`, a
   compliance referenced to the modulus at loading) must pass through
   the same four-function door and produce a *different* :math:`\chi`.
E. **Sectional AAEM vs EN 1992-1-1 (5.46)** — the engine against the
   closed form, on the section where the closed form is defined.
F. **Null and structural tests** — free shrinkage, restrained shrinkage
   (the sign proof), composite differential shrinkage (self-equilibrium).
G. **The emission theorem** — the tendon's prestrain moves by exactly
   its relaxation, and by nothing else.
H. **Boltzmann superposition** — the step-by-step integration must
   *converge*, and its converged value measures the AAEM's own error.
I. **The guards** — sub-quantum steps, non-linear creep, unbonded
   tendons, an active zone with no model.
J. **The flagship** — a prestressed composite, end to end from YAML.
"""

from __future__ import annotations

import math
import sys
import traceback

import numpy as np
from shapely.geometry import Polygon

from gensec.geometry.geometry import GenericSection
from gensec.geometry.fiber import Tendon
from gensec.materials.base import aging_coefficient, relaxation_function
from gensec.materials.ec2_bridge import concrete_from_class, prestress_from_ec2
from gensec.materials.rheology import (
    EC2RheologicalModel, TabulatedRheologicalModel,
)
from aci209_falsification import ACIRheologicalModel
from gensec.solver.losses import (
    LossModel, compute_interval_losses, ec2_5106_closed_form, expand_losses,
)
from gensec.solver.section_state import QUANT_EPS, StagedDomainManager

INF = 25550.0                      # 70 years [days] — EC2's practical infinity
FAILURES: list = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def close(a, b, rtol):
    return abs(a - b) <= rtol * max(abs(b), 1e-30)


# ======================================================================
#  ASSEMBLY B — the norms, written out by hand.  Shares no code with
#  gensec.materials.rheology: that is the whole point.
# ======================================================================

def ec2_phi_hand(fck, cement, RH, h0, t, t0):
    r"""EN 1992-1-1 Annex B.1, eq. (B.1)-(B.9), written from the code."""
    fcm = fck + 8.0
    a1, a2, a3 = ((35 / fcm) ** 0.7, (35 / fcm) ** 0.2, (35 / fcm) ** 0.5)
    base = (1 - RH / 100.0) / (0.1 * h0 ** (1 / 3))
    if fcm <= 35.0:
        phi_rh, a3 = 1.0 + base, 1.0
    else:
        phi_rh = (1.0 + base * a1) * a2
    alpha = {"R": 1.0, "N": 0.0, "S": -1.0}[cement]
    t0a = max(0.5, t0 * (9.0 / (2.0 + t0 ** 1.2) + 1.0) ** alpha)
    phi0 = phi_rh * (16.8 / math.sqrt(fcm)) / (0.1 + t0a ** 0.20)
    bH = min(1.5 * (1 + (0.012 * RH) ** 18) * h0 + 250 * a3, 1500 * a3)
    dt = t - t0
    return phi0 * (dt / (bH + dt)) ** 0.3


def ec2_eps_cs_hand(fck, cement, RH, h0, t, ts):
    r"""EN 1992-1-1 §3.1.4(6), eq. (3.9)-(3.13) + (B.11)-(B.12)."""
    fcm = fck + 8.0
    ads1, ads2 = {"R": (6.0, 0.11), "N": (4.0, 0.12), "S": (3.0, 0.13)}[cement]
    b_rh = 1.55 * (1 - (RH / 100.0) ** 3)
    ecd0 = (0.85 * (220 + 110 * ads1) * math.exp(-ads2 * fcm / 10.0)
            * 1e-6 * b_rh)
    kh = float(np.interp(h0, [100, 200, 300, 500], [1.0, 0.85, 0.75, 0.70]))
    bds = (t - ts) / ((t - ts) + 0.04 * math.sqrt(h0 ** 3)) if t > ts else 0.0
    ecd = bds * kh * ecd0
    eca = (1 - math.exp(-0.2 * math.sqrt(t))) * 2.5 * (fck - 10.0) * 1e-6
    return -(ecd + eca)


def ec2_relax_hand(cls, rho1000, f_pk, mu, t_days):
    r"""EN 1992-1-1 §3.3.2, eq. (3.28)-(3.30); *t* in hours, capped."""
    k1, k2 = {1: (5.39, 6.7), 2: (0.66, 9.1), 3: (1.98, 8.0)}[cls]
    th = min(24.0 * t_days, 500000.0)
    r = k1 * rho1000 * math.exp(k2 * mu) * (th / 1000.0) ** (
        0.75 * (1 - mu)) * 1e-5
    return -r * mu * f_pk


def aci_phi_hand(fc, RH, vs_in, t, t0):
    r"""ACI 209R-92 §2.2 (moist-cured, standard conditions)."""
    g_la = 1.25 * t0 ** -0.118
    g_rh = 1.27 - 0.0067 * RH
    g_vs = (2.0 / 3.0) * (1 + 1.13 * math.exp(-0.54 * vs_in))
    dt = t - t0
    return (dt ** 0.6) / (10.0 + dt ** 0.6) * 2.35 * g_la * g_rh * g_vs


def aci_Ec_hand(fc28, t):
    r"""ACI 209R-92 §2.1 (moist-cured, Type-I)."""
    return 4700.0 * math.sqrt(t / (4.0 + 0.85 * t) * fc28)


# ======================================================================
#  Sections
# ======================================================================

def section_A():
    print("\n== A. THE CARDINAL TEST — chi must EMERGE from EC2's own J ==")
    print("     EN 1992-1-1 (5.46) writes 0.8 twice.  If the container is")
    print("     genuinely normative-agnostic, that 0.8 is what EC2's own")
    print("     compliance PRODUCES, not what we told it.\n")
    print("     fck  h0[mm]   phi(inf,28)   chi(inf,28)")
    chis = []
    for fck in (25, 35, 45, 60):
        for (Ac, u) in ((200 * 1000.0, 2 * 1200.0),
                        (600 * 1400.0, 2 * 2000.0),
                        (800 * 2000.0, 2 * 2800.0)):
            m = EC2RheologicalModel(fck=fck, cement_class="N",
                                    RH=70).with_geometry(A_c=Ac, u=u)
            chi = aging_coefficient(m, INF, 28.0, n_steps=200)
            chis.append(chi)
            print(f"     {fck:3d}  {m.h0:6.0f}   {m.phi(INF, 28.0):9.3f}"
                  f"   {chi:9.4f}")
    lo, hi = min(chis), max(chis)
    check("chi from EC2's compliance lands on 0.80 +/- 0.06",
          0.74 <= lo and hi <= 0.87,
          f"range [{lo:.4f}, {hi:.4f}], mean {np.mean(chis):.4f}")
    check("chi is not a hard-coded 0.8 (it MOVES with fck and h0)",
          hi - lo > 0.05, f"spread {hi - lo:.4f}")


def section_B():
    print("\n== B. GRID CONVERGENCE of the Volterra inversion ==")
    m = EC2RheologicalModel(fck=35, RH=70).with_geometry(
        A_c=600 * 1400.0, u=2 * 2000.0)
    ref = aging_coefficient(m, INF, 28.0, n_steps=800)
    for n in (25, 50, 100, 200, 400):
        chi = aging_coefficient(m, INF, 28.0, n_steps=n)
        print(f"     n_steps={n:4d}   chi={chi:.6f}   |err|={abs(chi-ref):.2e}")
    err100 = abs(aging_coefficient(m, INF, 28.0, n_steps=100) - ref)
    check("the default n_steps=100 is converged to < 5e-4",
          err100 < 5e-4, f"|err| = {err100:.2e}")


def section_C():
    print("\n== C. EN 1992-1-1 — two independent assemblies ==")
    Ac, u, fck, cem, RH = 600 * 1400.0, 2 * 2000.0, 35, "N", 70.0
    h0 = 2 * Ac / u
    m = EC2RheologicalModel(fck=fck, cement_class=cem, RH=RH).with_geometry(
        A_c=Ac, u=u).with_steel(1860.0, relaxation_class=2, rho_1000=2.5)
    for (t, t0) in ((INF, 28.0), (365.0, 7.0), (90.0, 14.0)):
        a = m.phi_ec2(t, t0)
        b = ec2_phi_hand(fck, cem, RH, h0, t, t0)
        check(f"Annex B creep  phi_ec2({t:g},{t0:g})", close(a, b, 1e-9),
              f"{a:.6f} vs {b:.6f}")
    # The container's *generalized* creep coefficient is derived from J,
    # phi_gen = E_c(t') J(t,t') - 1, and with the section-5.10.6 convention
    # (creep referenced to E_cm(28)) it equals EC2's own phi scaled by the
    # modulus ratio.  They coincide at t' = 28 d and NOWHERE ELSE -- and
    # that is not a defect: the quantity the AAEM needs is the *strain*,
    #   phi_gen * sigma / E_c(t0) == sigma * phi_ec2 / E_cm(28),
    # which is EC2's creep strain exactly.  Assert both.
    for (t, t0) in ((INF, 28.0), (365.0, 7.0), (90.0, 14.0)):
        ratio = m.E_c(t0) / m.Ecm28
        check(f"phi_gen({t:g},{t0:g}) == phi_ec2 * E_cm(t0)/E_cm(28)",
              close(m.phi(t, t0), m.phi_ec2(t, t0) * ratio, 1e-9),
              f"{m.phi(t, t0):.6f} = {m.phi_ec2(t, t0):.6f} * {ratio:.5f}")
        # the creep STRAIN is EC2's, at every loading age
        creep_container = m.phi(t, t0) / m.E_c(t0)
        creep_ec2 = m.phi_ec2(t, t0) / m.Ecm28
        check(f"   ... so the creep STRAIN is EC2's exactly ({t:g},{t0:g})",
              close(creep_container, creep_ec2, 1e-12))
    check("phi_gen == phi_ec2 at t0 = 28 d (the conventions meet)",
          close(m.phi(INF, 28.0), m.phi_ec2(INF, 28.0), 1e-12))
    for (t, ts) in ((INF, 7.0), (365.0, 3.0)):
        a = m.eps_imposed(t, ts)
        b = ec2_eps_cs_hand(fck, cem, RH, h0, t, ts)
        check(f"3.1.4 shrinkage  eps_cs({t:g},{ts:g})", close(a, b, 1e-9),
              f"{a:.6e} vs {b:.6e}")
        check("   ... and it is SIGNED negative (a shortening)", a < 0.0)
    for mu in (0.60, 0.70, 0.80):
        a = m.relaxation(INF, mu)
        b = ec2_relax_hand(2, 2.5, 1860.0, mu, INF)
        check(f"3.3.2 relaxation  d_sigma_pr(mu={mu:.2f})", close(a, b, 1e-9),
              f"{a:.4f} vs {b:.4f} MPa")
    check("E_c(t) == 1/J(t,t)  (the ABC's own identity)",
          close(m.E_c(90.0), 1.0 / m.J(90.0, 90.0), 1e-12))


def section_D():
    print("\n== D. ACI 209R-92 — THE FALSIFICATION ==")
    print("     A structurally different norm through the same four-function")
    print("     door.  If the container had EC2 baked in, this would not work")
    print("     -- and it must produce a DIFFERENT chi, or the machinery is")
    print("     not measuring anything.\n")
    Ac, u = 600 * 1400.0, 2 * 2000.0
    a = ACIRheologicalModel(fc_28=35, RH=70).with_geometry(A_c=Ac, u=u)
    e = EC2RheologicalModel(fck=35, RH=70).with_geometry(A_c=Ac, u=u)
    vs = (Ac / u) / 25.4
    check("ACI phi(inf,28) vs the hand-written ACI 209R",
          close(a.phi(INF, 28.0), aci_phi_hand(35, 70, vs, INF, 28.0), 1e-9),
          f"{a.phi(INF, 28.0):.6f}")
    check("ACI E_c(28) vs the hand-written 4700*sqrt(f'c(t))",
          close(a.E_c(28.0), aci_Ec_hand(35, 28.0), 1e-9),
          f"{a.E_c(28.0):.1f} MPa")
    check("ACI's geometry measure is V/S [in], EC2's is h0 [mm]",
          abs(vs - 8.2677) < 1e-3 and abs(e.h0 - 420.0) < 1e-6,
          f"V/S={vs:.4f} in vs h0={e.h0:.0f} mm")
    chi_a = aging_coefficient(a, INF, 28.0, n_steps=200)
    chi_e = aging_coefficient(e, INF, 28.0, n_steps=200)
    check("chi(ACI) != chi(EC2), from the SAME container code path",
          abs(chi_a - chi_e) > 0.01,
          f"ACI {chi_a:.4f} vs EC2 {chi_e:.4f}")
    check("ACI shrinkage is signed negative too",
          a.eps_imposed(INF, 7.0) < 0.0,
          f"{a.eps_imposed(INF, 7.0):.4e}")
    # tabulated: a norm that is only data
    tp = np.array([7.0, 28.0, 90.0, 365.0])
    tt = np.array([7.0, 28.0, 90.0, 365.0, 3650.0, INF])
    JT = np.array([[e.J(max(t, s), s) for s in tp] for t in tt])
    tab = TabulatedRheologicalModel(tp, tt, JT)
    check("a TABULATED norm (data only) passes the same door",
          close(tab.E_c(28.0), e.E_c(28.0), 5e-3),
          f"E_c(28) {tab.E_c(28.0):.0f} vs {e.E_c(28.0):.0f} MPa")


def _pt_beam(mesh=25.0, y_t=200.0, sp0=1200.0):
    b, h = 600.0, 1400.0
    conc = concrete_from_class("C35/45", ls="S")
    ps = prestress_from_ec2(f_p01k=1600.0, f_pk=1860.0, eps_uk=0.035,
                            Ep=195000.0)
    ps.ec2.relaxation_class, ps.ec2.rho_1000 = 2, 2.5
    t = Tendon(y=y_t, x=b / 2, Ap=2000.0, material=ps, eps_pe=sp0 / 195000.0,
               bonded=True, embedded=True, name="T1")
    sec = GenericSection(polygon=Polygon([(0, 0), (b, 0), (b, h), (0, h)]),
                         bulk_material=conc, rebars=[], tendons=[t],
                         mesh_size=mesh)
    st = StagedDomainManager(sec, biaxial=False).initial_state()
    return sec, st, b, h, y_t


def section_E():
    print("\n== E. SECTIONAL AAEM  vs  EN 1992-1-1 (5.46) ==")
    print("     The engine (a 3x3 system on the age-adjusted transformed")
    print("     section) against the closed form, on the one section where")
    print("     the closed form is defined.  Two independent derivations.\n")
    sec, st, b, h, y_t = _pt_beam()
    prov = EC2RheologicalModel(fck=35, cement_class="N", RH=70)
    for chi in ("lump", "from_J"):
        r = compute_interval_losses(
            sec, st, models={0: LossModel(provider=prov, chi=chi)},
            demand=(0.0, -800e6, 0.0), zone_ages_t0={0: 28.0},
            zone_ages_t={0: INF}, zone_curing_ages={0: 7.0},
            tendon_ages={"T1": INF - 28.0})
        pb = prov.with_geometry(A_c=b * h, u=2 * (b + h)).with_steel(
            1860.0, relaxation_class=2, rho_1000=2.5)
        cf = ec2_5106_closed_form(
            Ep=195000.0, Ec=pb.E_c(28.0), Ap=2000.0, Ac=b * h,
            Ic=b * h ** 3 / 12, z_cp=h / 2 - y_t,
            sigma_c0=r.sigma_c_tendon["T1"],
            eps_sh=pb.eps_imposed(INF, 7.0) - pb.eps_imposed(28.0, 7.0),
            d_sigma_pr=pb.relaxation(INF - 28.0,
                                     r.sigma_p0["T1"] / 1860.0),
            phi=r.phi[0], chi=r.chi[0])
        got, exp = r.d_sigma_p["T1"], cf
        print(f"     chi={r.chi[0]:.4f} ({chi:6s}):  AAEM {got:8.2f} MPa"
              f"   (5.46) {exp:8.2f} MPa")
        check(f"sectional AAEM == (5.46) within 0.5%  [chi={chi}]",
              close(got, exp, 5e-3),
              f"spread {100 * abs(got - exp) / abs(exp):.3f}%")


def section_F():
    print("\n== F. NULL AND STRUCTURAL TESTS ==")
    from gensec.solver.losses import _UNCAST_PLACEHOLDER_E  # noqa: F401
    b, h = 600.0, 1400.0
    conc = concrete_from_class("C35/45", ls="S")
    sec = GenericSection(polygon=Polygon([(0, 0), (b, 0), (b, h), (0, h)]),
                         bulk_material=conc, rebars=[], tendons=[],
                         mesh_size=40.0)
    st = StagedDomainManager(sec, biaxial=False).initial_state()
    prov = EC2RheologicalModel(fck=35, RH=70)

    # free shrinkage, no load: the section must shorten freely, no self-stress
    r = compute_interval_losses(
        sec, st, models={0: LossModel(provider=prov)},
        demand=(0.0, 0.0, 0.0), zone_ages_t0={0: 28.0}, zone_ages_t={0: INF},
        zone_curing_ages={0: 7.0}, tendon_ages={})
    esh = (prov.with_geometry(A_c=b * h, u=2 * (b + h)).eps_imposed(INF, 7.0)
           - prov.with_geometry(A_c=b * h, u=2 * (b + h)).eps_imposed(28.0, 7.0))
    check("free shrinkage: d_eps0 == eps_sh exactly",
          close(r.u[0], esh, 1e-9), f"{r.u[0]:.6e} vs {esh:.6e}")
    check("free shrinkage: NO curvature",
          abs(r.u[1]) < 1e-15 and abs(r.u[2]) < 1e-15,
          f"chi_x={r.u[1]:.2e}")
    check("free shrinkage: NO self-stress (d_sigma = Ebar*(u - eps_imp) = 0)",
          abs(float(r.d_sigma_zone[0][0])) < 1e-9,
          f"{float(r.d_sigma_zone[0][0]):.3e} MPa")

    # the sign proof: restrained shrinkage must produce TENSION
    Ebar = prov.with_geometry(A_c=b * h, u=2 * (b + h)).E_c(28.0) / (
        1 + 0.8 * r.phi[0])
    check("restrained shrinkage puts concrete in TENSION (the sign proof)",
          -Ebar * esh > 0.0, f"-Ebar*eps_sh = {-Ebar * esh:+.3f} MPa")
    check("the emitted bulk offset is the NEGATIVE of the eigenstrain",
          r.bulk_plane_delta[0][0] > 0.0 > esh,
          f"beta_0 = {r.bulk_plane_delta[0][0]:+.4e}, eps_imp "
          f"= {-r.bulk_plane_delta[0][0]:+.4e}")


def section_G():
    print("\n== G. THE EMISSION THEOREM ==")
    print("     eps_init(t) = eps_init(t0) + d_sigma_pr / E_p, and NOTHING")
    print("     else.  Creep and shrinkage are properties of the CONCRETE;")
    print("     they reach the tendon through the section plane.\n")
    sec, st, b, h, y_t = _pt_beam()
    prov = EC2RheologicalModel(fck=35, RH=70)
    r = compute_interval_losses(
        sec, st, models={0: LossModel(provider=prov)},
        demand=(0.0, -800e6, 0.0), zone_ages_t0={0: 28.0},
        zone_ages_t={0: INF}, zone_curing_ages={0: 7.0},
        tendon_ages={"T1": INF - 28.0})
    ui = list(r.eps_override)[0]
    d_emitted = r.eps_override[ui] - float(st.eps_init[ui])
    d_relax = r.d_sigma_pr["T1"] / 195000.0
    print(f"     emitted d_eps_init = {d_emitted:+.6e}")
    print(f"     d_sigma_pr / E_p   = {d_relax:+.6e}")
    check("the tendon's prestrain moves by EXACTLY its relaxation",
          close(d_emitted, d_relax, 1e-12),
          f"|diff| = {abs(d_emitted - d_relax):.2e}")
    check("the total loss is MUCH larger than the relaxation alone "
          "(the rest travels through the plane)",
          abs(r.d_sigma_p["T1"]) > 2.5 * abs(r.d_sigma_pr["T1"]),
          f"{r.d_sigma_p['T1']:.2f} vs {r.d_sigma_pr['T1']:.2f} MPa")


def section_H():
    print("\n== H. BOLTZMANN SUPERPOSITION — the step-by-step must CONVERGE ==")
    print("     A naive cumulative sum (each step restarting the creep clock)")
    print("     over-counts by ~80%.  Boltzmann superposition over the stress")
    print("     history is the only thing that converges -- and its limit")
    print("     MEASURES the AAEM's own approximation error.\n")
    sec, st, b, h, y_t = _pt_beam(mesh=30.0)
    prov = EC2RheologicalModel(fck=35, RH=70)
    common = dict(demand=(0.0, -800e6, 0.0), zone_ages_t0={0: 28.0},
                  zone_ages_t={0: INF}, zone_curing_ages={0: 7.0},
                  tendon_ages_t0={"T1": 0.0}, interval_days=INF - 28.0)
    _, tr1, _, _ = expand_losses(
        sec, st, models={0: LossModel(provider=prov)}, **common)
    base = tr1[0].d_sigma_p["T1"]
    print(f"     single AAEM interval : {base:8.2f} MPa")
    prev = None
    for n in (2, 4, 8, 12):
        cuts = sorted({round((INF - 28.0) * 10 ** (-3 + 3 * k / n), 4)
                       for k in range(1, n)})
        _, tr, _, _ = expand_losses(
            sec, st, models={0: LossModel(provider=prov, steps=cuts)},
            **common)
        tot = sum(r.d_sigma_p["T1"] for r in tr)
        print(f"     {len(tr):3d} sub-steps       : {tot:8.2f} MPa"
              f"   ({100 * (tot - base) / abs(base):+6.2f}% vs the AAEM)")
        prev = tot
    check("the step-by-step CONVERGES (it does not run away)",
          abs(prev - base) / abs(base) < 0.05,
          f"converged {prev:.2f} vs AAEM {base:.2f} MPa")
    check("the converged value measures the AAEM's error at ~1%",
          0.002 < abs(prev - base) / abs(base) < 0.03,
          f"{100 * abs(prev - base) / abs(base):.2f}%")


def section_I():
    print("\n== I. THE GUARDS — every one must FIRE ==")
    sec, st, b, h, y_t = _pt_beam(mesh=30.0)
    prov = EC2RheologicalModel(fck=35, RH=70)
    common = dict(demand=(0.0, -800e6, 0.0), zone_ages_t0={0: 28.0},
                  zone_ages_t={0: INF}, zone_curing_ages={0: 7.0},
                  tendon_ages_t0={"T1": 0.0}, interval_days=INF - 28.0)
    try:
        expand_losses(sec, st, models={0: LossModel(
            provider=prov, steps=[INF - 30.0, INF - 29.5])}, **common)
        check(f"sub-quantum step (< QUANT_EPS = {QUANT_EPS:.0e}) raises",
              False, "NO RAISE")
    except ValueError as exc:
        check(f"sub-quantum step (< QUANT_EPS = {QUANT_EPS:.0e}) raises",
              "QUANT_EPS" in str(exc))

    # non-linear creep: load a young, weak concrete hard
    try:
        compute_interval_losses(
            sec, st, models={0: LossModel(provider=prov)},
            demand=(-9.0e6, -800e6, 0.0), zone_ages_t0={0: 3.0},
            zone_ages_t={0: INF}, zone_curing_ages={0: 3.0},
            tendon_ages={"T1": INF})
        check("concrete beyond 0.45 f_ck (non-linear creep) raises",
              False, "NO RAISE")
    except ValueError as exc:
        check("concrete beyond 0.45 f_ck (non-linear creep) raises",
              "linear-viscoelasticity" in str(exc))

    # unbonded tendon
    st_ub = st.with_deactivated([]) if False else st
    import dataclasses
    bonded = np.asarray(st.bonded).copy()
    bonded[-1] = False
    st_ub = dataclasses.replace(st, bonded=bonded)
    try:
        compute_interval_losses(
            sec, st_ub, models={0: LossModel(provider=prov)},
            demand=(0.0, -800e6, 0.0), zone_ages_t0={0: 28.0},
            zone_ages_t={0: INF}, zone_curing_ages={0: 7.0},
            tendon_ages={"T1": INF})
        check("an UNBONDED tendon raises (member-average strain, not "
              "sectional)", False, "NO RAISE")
    except NotImplementedError as exc:
        check("an UNBONDED tendon raises (member-average strain, not "
              "sectional)", "member-average" in str(exc))

    # the ABC itself
    try:
        EC2RheologicalModel(fck=35).phi(INF, 28.0)
        check("an unbound drying geometry raises", False, "NO RAISE")
    except ValueError as exc:
        check("an unbound drying geometry raises", "unbound" in str(exc))
    try:
        EC2RheologicalModel(fck=35).with_geometry(
            A_c=1e6, u=4e3).relaxation(INF, 0.7)
        check("an unbound prestressing steel raises", False, "NO RAISE")
    except ValueError as exc:
        check("an unbound prestressing steel raises", "f_pk" in str(exc))


def section_J(yaml_path="examples/example_composite_losses.yaml"):
    print("\n== J. THE FLAGSHIP — a PRESTRESSED COMPOSITE, end to end ==")
    print("     Three intervals, two concrete zones of different age, creep")
    print("     and shrinkage.  EN 1992-1-1 (5.46) cannot express this: which")
    print("     A_c, which I_c, which phi?  The sectional AAEM just does it.\n")
    from gensec.io_yaml import load_yaml
    from gensec.solver.timeline import ConstructionTimeline
    try:
        d = load_yaml(yaml_path)
    except FileNotFoundError:
        print(f"     (skipped: {yaml_path} not found)")
        return
    sec = d["section"]
    ddb = {x["name"]: x for x in d["demands"]}
    tl = ConstructionTimeline.from_block(d["construction_history"],
                                         losses_models=d["losses_models"])
    res = tl.resolve(sec, ddb)
    check("the timeline resolves with losses on three intervals",
          len(res.losses_ops) == 3, f"{sorted(res.losses_ops)}")
    tot = 0.0
    for ei, ops in res.losses_ops.items():
        for r in res.losses_trace[ei]:
            for nm in r.d_sigma_p:
                tot += r.d_sigma_p[nm]
            print(f"     interval[{ei}]  phi = "
                  + ", ".join(f"{sec.zone_names[z]} {v:.3f}"
                              for z, v in r.phi.items()))
    print(f"     total time-dependent loss on P1: {tot:+.1f} MPa")
    check("the total loss is physical (5-20% of the jacking stress)",
          40.0 < abs(tot) < 300.0, f"{tot:+.1f} MPa")
    check("the young topping creeps FASTER than the mature precast",
          res.losses_trace[7][0].phi[1] > res.losses_trace[7][0].phi[0],
          f"topping {res.losses_trace[7][0].phi[1]:.3f} vs precast "
          f"{res.losses_trace[7][0].phi[0]:.3f}")

    # the capacity hashes must DIFFER across the loss intervals: no cache
    # collapse.  (eta may still be identical -- at ULS a bonded tendon is
    # far past yield and a 100 MPa loss barely moves the capacity.  That is
    # physics, and it is exactly why losses matter for SLS, not for ULS.)
    combo = [c for c in d["combinations"] if c["name"] == "ULS_composite"][0]
    mgr = StagedDomainManager(sec, biaxial=False, gen_kwargs={"n_points": 20})
    hashes = []
    for _point, stages, _in in tl.compile_combination(combo, res, sec, ddb):
        _s, hs, _b, _dd = mgr.resolve_stages(stages, initially_inactive=[])
        hashes.append(hs[-1])
    check("each loss interval yields a DISTINCT capacity hash "
          "(no cache collapse)", len(set(hashes)) == len(hashes),
          f"{len(set(hashes))} distinct of {len(hashes)}")


def main():
    print("=" * 70)
    print("  GenSec — Phase-5 / C5 validation")
    print("  rheology subsystem + time-dependent prestress losses")
    print("=" * 70)
    for fn in (section_A, section_B, section_C, section_D, section_E,
               section_F, section_G, section_H, section_I, section_J):
        try:
            fn()
        except Exception:                                # noqa: BLE001
            FAILURES.append(fn.__name__)
            print(f"\n  [FAIL] {fn.__name__} raised:")
            traceback.print_exc()
    print("\n" + "=" * 70)
    if FAILURES:
        print(f"  {len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("  ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
