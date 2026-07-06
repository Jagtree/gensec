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
Phase 8, Task 1 validation — the six axes of the primer's §5.

V1  **Incremental ≡ one-shot identity** (the G-D2 linear-equivalence
    theorem).  Elastic composite (precast 600×1200 + topping
    600×200): a one-shot solve on the composite with the topping's
    datum plane set to the *negated* substrate plane at casting must
    reproduce the incremental staged solution **exactly** in the
    elastic range,

    .. math::

       \mathbf{u}_{\text{one-shot}}
       \;=\; \mathbf{u}_0 + \Delta\mathbf{u},
       \qquad
       \mathbf{K}_{\text{comp}}\,\Delta\mathbf{u} = \mathbf{L}_1 ,

    with stresses
    :math:`\sigma = E_z\,(\varepsilon(\mathbf{u}) +
    \varepsilon^{\text{off}}_z)` checked at the extreme fibers and on
    both sides of the interface, for both curvature signs and for a
    biaxial (3-DOF) variant.  The independent model is assembled two
    ways: **discrete** (fiber sums — solver tolerance, 1e-14) and
    **continuum** (exact rectangle formulas — mesh tolerance: the
    grid mesh misses the per-cell parallel-axis term,
    :math:`(\Delta y/H)^2 = (100/1400)^2 = 5.1\cdot 10^{-3}`
    relative, so the bound is 8e-3).

V2  **Masked view ≡ standalone section.**  With the interface on a
    mesh grid line, the stage-0 masked view of the composite carries
    fiber-for-fiber the same discretization as a standalone precast
    section; with the standalone solver's reference forced onto the
    composite full centroid (the pinned reference), the N–Mx domains
    and a biaxial cloud must be **identical floating-point
    computations** — exact equality is demanded, not a tolerance.
    This also proves the phantom-pivot fix: without the
    ``_bounds``/``H`` override the uniaxial edge-strain templates
    would pivot on fibers of the inactive zone and the masked domain
    would not close onto the standalone one.

V3  **Single-bulk A/B regression.**  A deterministic pipeline (solved
    plane + domain checksums) over shipped single-bulk / prestress
    example YAMLs against reference values captured from the
    **pristine** (pre-Task-1) package.  In the development sandbox
    the A/B was run as two package trees with a byte-level ``diff``
    of full-precision reprs (exact equality); this shipped re-run
    asserts ``rtol = 1e-12``, because bit-exactness across BLAS
    builds / platforms is not a portable contract while 1e-12 on a
    strain is.

V4  **Invariant enforcement** (fail-loud): containment invariant,
    empty active bulk, re-activation, ``deactivate_bulk``
    reservation, zone-0 activation, datum atomicity.

V5  **Hash discipline**: mask flip and beyond-quantum plane changes
    rebuild the domain; within-quantum changes reuse it; the
    curvature quantum is
    :math:`q_\chi = \texttt{QUANT\_EPS} / \max(H, B)`;
    ``path_schedule`` resets at the casting stage and reuses the
    cache on the unchanged stage.

V6  **Tendon-in-topping composite**: pretensioned strand in the
    precast active from stage 0; a topping tendon deactivated at
    stage 0, its zone cast at stage 1, the tendon entered at stage 2
    with its stressing prestrain — the containment chain passes,
    hashes change at each section event, the walk's equilibria
    converge; activating the topping tendon before its zone raises.

Exit code 0 iff every check passes.  Run from anywhere inside the
repository::

    python run_phase8_task1_validation.py
"""

import sys
import numpy as np
from pathlib import Path
from shapely.geometry import box

from gensec.materials.elastic import LinearElastic
from gensec.materials.concrete import Concrete
from gensec.materials.steel import Steel, PrestressingSteel
from gensec.geometry.fiber import RebarLayer, Tendon
from gensec.geometry.geometry import GenericSection
from gensec.geometry.primitives import rect_poly
from gensec.solver.section_state import (
    StagedDomainManager, materialize_view, path_schedule, QUANT_EPS,
)
from gensec.solver.integrator import FiberSolver
from gensec.solver.capacity import NMDiagram

# ------------------------------------------------------------------
#  Harness
# ------------------------------------------------------------------

_N_PASS = 0
_N_FAIL = 0


def check(label, ok, detail=""):
    r"""Record one boolean check with a banner line."""
    global _N_PASS, _N_FAIL
    if ok:
        _N_PASS += 1
        print(f"  [PASS] {label}")
    else:
        _N_FAIL += 1
        print(f"  [FAIL] {label}  {detail}")


def check_raises(label, exc, fn, *args, **kw):
    r"""Assert that ``fn(*args, **kw)`` raises exactly *exc*."""
    try:
        fn(*args, **kw)
    except exc:
        check(label, True)
        return
    except Exception as e:                                # noqa: BLE001
        check(label, False, f"raised {type(e).__name__}: {e}")
        return
    check(label, False, "did not raise")


def banner(title):
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")


# ------------------------------------------------------------------
#  Shared fixtures
# ------------------------------------------------------------------

E1, E2 = 35000.0, 31000.0     # precast / topping elastic moduli [MPa]
B, H1, H2 = 600.0, 1200.0, 200.0
H = H1 + H2


def composite_linear():
    r"""600×1400 grid-meshed composite; interface on a grid line."""
    c1 = LinearElastic(E=E1, name="precast")
    c2 = LinearElastic(E=E2, name="topping")
    return GenericSection(
        polygon=rect_poly(B, H), bulk_material=c1, rebars=[],
        bulk_materials=[(box(0, H1, B, H), c2, "topping")],
        mesh_size=100.0, n_grid_x=6, n_grid_y=14)


def hand_stiffness(sec, dofs3=False):
    r"""
    Discrete elastic stiffness from fiber sums about the solver
    reference (the full-polygon centroid) — the *first* independent
    assembly.

    With the strain-plane convention
    :math:`\varepsilon = \varepsilon_0 + \chi_x\,l_y - \chi_y\,l_x`
    and resultants :math:`N = \sum F`, :math:`M_x = \sum F\,l_y`,
    :math:`M_y = -\sum F\,l_x`, the 3-DOF stiffness is symmetric with
    the sign pattern encoded below.

    Parameters
    ----------
    sec : GenericSection
    dofs3 : bool, optional
        ``True`` for the biaxial 3×3; default 2×2 (``eps0, chi_x``).

    Returns
    -------
    numpy.ndarray
    """
    yr, xr = float(sec.y_centroid), float(sec.x_centroid)
    E = np.where(np.asarray(sec.mat_indices) == 0, E1, E2)
    A = sec.A_fibers
    ly = sec.y_fibers - yr
    lx = sec.x_fibers - xr
    if not dofs3:
        return np.array([[np.sum(E*A),      np.sum(E*A*ly)],
                         [np.sum(E*A*ly),   np.sum(E*A*ly*ly)]])
    return np.array([
        [np.sum(E*A),      np.sum(E*A*ly),     -np.sum(E*A*lx)],
        [np.sum(E*A*ly),   np.sum(E*A*ly*ly),  -np.sum(E*A*ly*lx)],
        [-np.sum(E*A*lx),  -np.sum(E*A*ly*lx),  np.sum(E*A*lx*lx)],
    ])


def continuum_stiffness_2dof():
    r"""
    Continuum (exact rectangle) elastic stiffness — the *second*
    independent assembly, about the composite full centroid
    :math:`y_c = H/2 = 700`:

    .. math::

       K_{00} = \sum_z E_z A_z,\qquad
       K_{01} = \sum_z E_z S_z,\qquad
       K_{11} = \sum_z E_z I_z ,

    with :math:`S_z, I_z` the first/second moments of each rectangle
    about the reference axis (exact closed forms).

    Returns
    -------
    numpy.ndarray
        Shape ``(2, 2)``.
    """
    yc = H / 2.0

    def rect_terms(E, y0, y1):
        A = B * (y1 - y0)
        S = B * ((y1 - yc) ** 2 - (y0 - yc) ** 2) / 2.0
        I = B * ((y1 - yc) ** 3 - (y0 - yc) ** 3) / 3.0
        return E * A, E * S, E * I

    a1, s1, i1 = rect_terms(E1, 0.0, H1)
    a2, s2, i2 = rect_terms(E2, H1, H)
    return np.array([[a1 + a2, s1 + s2],
                     [s1 + s2, i1 + i2]])


# ------------------------------------------------------------------
#  V1 — incremental ≡ one-shot identity
# ------------------------------------------------------------------

def v1_identity():
    banner("V1  Incremental == one-shot identity (G-D2)")
    sec = composite_linear()
    mgr = StagedDomainManager(sec, biaxial=False,
                              gen_kwargs={"n_points": 30})
    yr = float(sec.y_centroid)

    for sgn, tag in ((+1.0, "sagging-negative Mx"), (-1.0, "+Mx")):
        L0 = (-3.0e5, sgn * -2.5e8, 0.0)
        L1 = (0.0, sgn * -3.2e8, 0.0)

        # Stage 0: precast alone (activation-declarative pre-scan).
        stages = [{"section_ops": {}},
                  {"section_ops": {"activate_bulk":
                                   {1: (0.0, 0.0, 0.0)}}}]
        _, _, bundles, _ = mgr.resolve_stages(stages)
        sol0 = bundles[0].solver.solve_equilibrium(*L0)
        u0 = np.array([sol0["eps0"], sol0["chi_x"], sol0["chi_y"]])
        check(f"[{tag}] stage-0 solve converged", sol0["converged"])

        # One-shot composite solve with the negated substrate datum.
        stages = [{"section_ops": {}},
                  {"section_ops": {"activate_bulk":
                                   {1: tuple(-u0)}}}]
        _, _, bundles, _ = mgr.resolve_stages(stages)
        sv1 = bundles[1].solver
        Lt = tuple(a + b for a, b in zip(L0, L1))
        sol1 = sv1.solve_equilibrium(*Lt)
        uf = np.array([sol1["eps0"], sol1["chi_x"], sol1["chi_y"]])
        check(f"[{tag}] composite solve converged", sol1["converged"])

        # Independent assembly 1: discrete fiber sums.
        Kd = hand_stiffness(sec)
        du_d = np.linalg.solve(Kd, [L1[0], L1[1]])
        err_d = np.abs(uf[:2] - (u0[:2] + du_d))
        check(f"[{tag}] identity vs discrete model "
              f"(err={err_d.max():.2e})", err_d.max() < 1e-14)

        # Independent assembly 2: continuum rectangle formulas.  The
        # grid mesh concentrates each cell's area at its centroid, so
        # the discrete EI misses the per-cell parallel-axis term: the
        # relative defect is (dy^2/12) / (H^2/12) = (dy/H)^2 =
        # (100/1400)^2 = 5.1e-3.  Mesh tolerance, not solver
        # tolerance: assert < 8e-3 (1.5x the estimate).
        Kc = continuum_stiffness_2dof()
        du_c = np.linalg.solve(Kc, [L1[0], L1[1]])
        rel = np.abs((uf[:2] - u0[:2]) - du_c) / np.abs(du_c)
        check(f"[{tag}] identity vs continuum model "
              f"(rel={rel.max():.2e}, mesh bound 8e-3)",
              rel.max() < 8e-3)

        # Stress identities: precast on the total plane, topping on
        # the incremental plane (its datum cancels the substrate).
        fr = sv1.get_fiber_results(*uf)
        yb = np.asarray(fr["bulk"]["y"])
        sb = np.asarray(fr["bulk"]["sigma"])
        mi_v = np.asarray(sv1.sec.mat_indices)

        def plane(u, y):
            return u[0] + u[1] * (y - yr)

        du = uf - u0
        pts = [
            ("precast bottom  y=50",   50.0, 0, E1, uf),
            ("precast interf. y=1150", 1150.0, 0, E1, uf),
            ("topping interf. y=1250", 1250.0, 1, E2, du),
            ("topping top     y=1350", 1350.0, 1, E2, du),
        ]
        for label, y, zone, Ez, u in pts:
            i = np.nonzero((np.abs(yb - y) < 1e-9)
                           & (mi_v == zone))[0]
            got = float(sb[i[0]])
            want = Ez * plane(u, y)
            check(f"[{tag}] sigma {label} "
                  f"({got:+.6f} vs {want:+.6f} MPa)",
                  abs(got - want) < 1e-9)

    # Biaxial 3-DOF variant: chi_y engaged by a My increment.
    L0 = (-3.0e5, -2.0e8, 5.0e7)
    L1 = (0.0, -1.5e8, -8.0e7)
    stages = [{"section_ops": {}},
              {"section_ops": {"activate_bulk": {1: (0.0, 0.0, 0.0)}}}]
    _, _, bundles, _ = mgr.resolve_stages(stages)
    sol0 = bundles[0].solver.solve_equilibrium(*L0)
    u0 = np.array([sol0["eps0"], sol0["chi_x"], sol0["chi_y"]])
    stages = [{"section_ops": {}},
              {"section_ops": {"activate_bulk": {1: tuple(-u0)}}}]
    _, _, bundles, _ = mgr.resolve_stages(stages)
    sol1 = bundles[1].solver.solve_equilibrium(
        L0[0] + L1[0], L0[1] + L1[1], L0[2] + L1[2])
    uf = np.array([sol1["eps0"], sol1["chi_x"], sol1["chi_y"]])
    K3 = hand_stiffness(sec, dofs3=True)
    du3 = np.linalg.solve(K3, L1)
    err3 = np.abs(uf - (u0 + du3))
    check(f"[biaxial 3-DOF] identity (err={err3.max():.2e})",
          err3.max() < 1e-14)


# ------------------------------------------------------------------
#  V2 — masked view ≡ standalone section
# ------------------------------------------------------------------

def v2_masked_vs_standalone():
    banner("V2  Masked view == standalone section (domains)")
    st = Steel(fyk=450.0, gamma_s=1.15)
    conc = Concrete(fck=45.0)
    rebars = [RebarLayer(x=60.0, y=60.0, As=800.0, material=st),
              RebarLayer(x=540.0, y=60.0, As=800.0, material=st)]

    comp = GenericSection(
        polygon=rect_poly(B, H), bulk_material=conc,
        rebars=list(rebars),
        bulk_materials=[(box(0, H1, B, H), Concrete(fck=30.0),
                         "topping")],
        mesh_size=100.0, n_grid_x=6, n_grid_y=14)
    stand = GenericSection(
        polygon=rect_poly(B, H1), bulk_material=conc,
        rebars=list(rebars), mesh_size=100.0, n_grid_x=6,
        n_grid_y=12)

    mgr = StagedDomainManager(comp, biaxial=False,
                              gen_kwargs={"n_points": 40})
    s = mgr.initial_state().copy_advanced(0)
    s.bulk_active[1] = False
    view = materialize_view(comp, s)
    check("fiber counts match", view.n_fibers == stand.n_fibers,
          f"{view.n_fibers} vs {stand.n_fibers}")
    check("fiber coordinates match",
          np.array_equal(np.sort(view.y_fibers),
                         np.sort(stand.y_fibers)))

    # Same reference point on both sides: force the standalone solver
    # onto the composite full centroid (the pinned reference).
    sv_msk = FiberSolver(view)
    sv_ref = FiberSolver(stand, x_ref=float(comp.x_centroid),
                         y_ref=float(comp.y_centroid))
    check("masked reference is the composite centroid",
          sv_msk.y_ref == float(comp.y_centroid))

    dom_m = NMDiagram(sv_msk).generate(n_points=40)
    dom_r = NMDiagram(sv_ref).generate(n_points=40)
    dN = np.abs(np.sort(np.asarray(dom_m["N"]))
                - np.sort(np.asarray(dom_r["N"]))).max()
    dM = np.abs(np.sort(np.asarray(dom_m["Mx"]))
                - np.sort(np.asarray(dom_r["Mx"]))).max()
    # Same fibers, same reference, same templates: the two runs are
    # the same floating-point computation — demand exact equality.
    check(f"uniaxial N-Mx domains identical (dN={dN}, dMx={dM})",
          dN == 0.0 and dM == 0.0)

    # Biaxial cloud (coarse resolution — cost control).
    c_m = NMDiagram(FiberSolver(view)).generate_biaxial(
        n_angles=8, n_points_per_angle=12)
    c_r = NMDiagram(FiberSolver(stand,
                                x_ref=float(comp.x_centroid),
                                y_ref=float(comp.y_centroid))
                    ).generate_biaxial(n_angles=8,
                                       n_points_per_angle=12)
    dd = max(
        np.abs(np.sort(np.asarray(c_m["N"]))
               - np.sort(np.asarray(c_r["N"]))).max(),
        np.abs(np.sort(np.asarray(c_m["Mx"]))
               - np.sort(np.asarray(c_r["Mx"]))).max(),
        np.abs(np.sort(np.asarray(c_m["My"]))
               - np.sort(np.asarray(c_r["My"]))).max())
    check(f"biaxial clouds identical (d={dd})", dd == 0.0)


# ------------------------------------------------------------------
#  V3 — single-bulk A/B regression
# ------------------------------------------------------------------

#: Reference values captured from the PRISTINE (pre-Task-1) package:
#: ``(eps0, chi_x, sum N over the uniaxial cloud, sum Mx)`` for a
#: fixed deterministic pipeline (initial state, ``n_points=60``,
#: demand N=-200 kN / Mx=+50 kNm).  The sandbox A/B was
#: byte-identical (``diff`` of full-precision reprs over two package
#: trees); this re-run asserts rtol=1e-12 — bit-exactness across
#: BLAS builds is not a portable contract.
_V3_REFS = {
    "example_input.yaml": (
        -4.1593708096457384e-05, 6.995360452591111e-07,
        -602983473.4004353, 1.6689300537109375e-06),
    "example_tee.yaml": (
        -3.952335865666795e-05, 4.4907087353668365e-07,
        -808573842.0096023, -4050220485.100918),
    "vcaslu_1.yaml": (
        2.838982425116848e-05, 1.921290488436274e-06,
        -464204715.00370455, -3.2782554626464844e-07),
    "example_prestress.yaml": (
        0.00014167202634475092, 3.5467171585641617e-06,
        -912772316.9084029, -135255742964.064),
    "biaxial_column.yaml": (
        -1.1873085585995818e-05, 5.840807851662742e-07,
        -885006210.3702846, 1.1920928955078125e-06),
    "example_staged_construction.yaml": (
        -2.3295977264924258e-05, 1.3790253279603822e-06,
        -493601637.3038286, -2.0265579223632812e-06),
}


def _find_examples_dir():
    r"""Walk upward from the validator to the repo ``examples/``."""
    here = Path(__file__).resolve().parent
    for root in (here, *here.parents):
        cand = root / "examples"
        if (cand / "example_input.yaml").exists():
            return cand
    raise FileNotFoundError(
        "examples/ directory not found above the validator.")


def v3_ab_regression():
    banner("V3  Single-bulk A/B regression (pristine references)")
    from gensec.io_yaml import load_yaml
    ex = _find_examples_dir()
    for name, refs in _V3_REFS.items():
        d = load_yaml(str(ex / name))
        mgr = StagedDomainManager(d["section"], biaxial=False,
                                  gen_kwargs={"n_points": 60})
        st = mgr.initial_state()
        _, bundle, _ = mgr.get_bundle(st)
        sol = bundle.solver.solve_equilibrium(-2e5, 5e7, 0.0)
        got = (sol["eps0"], sol["chi_x"],
               float(np.sum(np.asarray(bundle.cloud["N"]))),
               float(np.sum(np.asarray(bundle.cloud["Mx"]))))
        ok = all(abs(g - r) <= 1e-12 * max(1.0, abs(r))
                 for g, r in zip(got, refs))
        check(f"{name}: plane + domain checksums", ok,
              "" if ok else f"got {got!r}")


# ------------------------------------------------------------------
#  V4 — invariants
# ------------------------------------------------------------------

def v4_invariants():
    banner("V4  Invariant enforcement (fail-loud)")
    st = Steel(fyk=450.0, gamma_s=1.15)
    reb_top = RebarLayer(x=60.0, y=1340.0, As=400.0, material=st)
    sec = GenericSection(
        polygon=rect_poly(B, H), bulk_material=Concrete(fck=45.0),
        rebars=[reb_top],
        bulk_materials=[(box(0, H1, B, H), Concrete(fck=30.0),
                         "topping")],
        mesh_size=100.0, n_grid_x=6, n_grid_y=14)
    mgr = StagedDomainManager(sec, biaxial=False,
                              gen_kwargs={"n_points": 20})

    check_raises(
        "containment invariant (active element, uncast parent zone)",
        ValueError, mgr.resolve_stages,
        [{"section_ops": {}},
         {"section_ops": {"activate_bulk": {1: (0.0, 0.0, 0.0)}}}])

    check_raises(
        "re-activation of an active zone",
        ValueError, mgr.resolve_stages,
        [{"section_ops": {"deactivate": [0], "release": False,
                          "activate_bulk": {1: (0.0, 0.0, 0.0)}}},
         {"section_ops": {"activate_bulk": {1: (0.0, 0.0, 0.0)}}}])

    check_raises(
        "deactivate_bulk reserved (NotImplementedError)",
        NotImplementedError, mgr.resolve_stages,
        [{"section_ops": {"deactivate_bulk": [1]}}])

    check_raises(
        "zone index out of range",
        ValueError, mgr.resolve_stages,
        [{"section_ops": {"activate_bulk": {7: (0.0, 0.0, 0.0)}}}])

    s0 = mgr.initial_state()
    check_raises(
        "zone 0 not activatable (state op)",
        ValueError, s0.with_bulk_activated, [0], {0: (0, 0, 0)})

    s = s0.copy_advanced(0)
    s.bulk_active[1] = False
    check_raises(
        "missing datum plane raises KeyError (atomicity)",
        KeyError, s.with_bulk_activated, [1], {})

    # Empty active bulk (zones cover the full polygon).
    c = LinearElastic(E=30000.0, name="c")
    sec2 = GenericSection(
        polygon=rect_poly(B, H), bulk_material=c, rebars=[],
        bulk_materials=[(box(0, 0, B, H1), c, "web"),
                        (box(0, H1, B, H), c, "top")],
        mesh_size=100.0, n_grid_x=6, n_grid_y=14)
    mgr2 = StagedDomainManager(sec2, biaxial=False,
                               gen_kwargs={"n_points": 20})
    check_raises(
        "empty active bulk at stage 0",
        ValueError, mgr2.resolve_stages,
        [{"section_ops": {}},
         {"section_ops": {"activate_bulk": {1: (0, 0, 0),
                                            2: (0, 0, 0)}}}])


# ------------------------------------------------------------------
#  V5 — hash discipline + path schedule
# ------------------------------------------------------------------

def v5_hash():
    banner("V5  Capacity hash: mask, quantized planes, path resets")
    sec = composite_linear()
    mgr = StagedDomainManager(sec, biaxial=False,
                              gen_kwargs={"n_points": 20})
    s0 = mgr.initial_state()
    h_all = mgr.hash_of(s0)

    s_m = s0.copy_advanced(0)
    s_m.bulk_active[1] = False
    h_msk = mgr.hash_of(s_m)
    check("mask flip changes the hash", h_msk != h_all)

    base = s_m.with_bulk_activated([1], {1: (2e-4, -3e-7, 0.0)})
    h_b = mgr.hash_of(base)
    check("casting (mask + plane) changes the hash", h_b != h_msk)

    qchi = mgr._chi_quantum
    check(f"curvature quantum = QUANT_EPS / max(H, B) "
          f"({qchi:.3e})", abs(qchi - QUANT_EPS / H) < 1e-20)

    same = s_m.with_bulk_activated(
        [1], {1: (2e-4 + 0.4 * QUANT_EPS, -3e-7 + 0.4 * qchi, 0.0)})
    check("within-quantum plane change keeps the hash",
          mgr.hash_of(same) == h_b)
    diff = s_m.with_bulk_activated(
        [1], {1: (2e-4, -3e-7 + 1.1 * qchi, 0.0)})
    check("beyond-quantum curvature change breaks the hash",
          mgr.hash_of(diff) != h_b)

    # Path resets across a casting walk: reset at stage 0 (origin),
    # reset at the casting stage, no reset on an unchanged stage.
    stages = [{"section_ops": {}},
              {"section_ops": {"activate_bulk":
                               {1: (2e-4, 0.0, 0.0)}}},
              {}]
    _, hashes, _, _ = mgr.resolve_stages(stages)
    sched = path_schedule(hashes)
    resets = [bool(rec["reset"]) for rec in sched]
    check("path_schedule resets = [True, True, False]",
          resets == [True, True, False], f"got {resets}")
    check("cache reuse on the unchanged stage",
          hashes[1] == hashes[2] and hashes[0] != hashes[1])


# ------------------------------------------------------------------
#  V6 — tendon-in-topping composite walk
# ------------------------------------------------------------------

def v6_tendon_in_topping():
    banner("V6  Tendon-in-topping composite (containment chain)")
    ps = PrestressingSteel()
    t_pre = Tendon(y=100.0, x=300.0, material=ps, Ap=1050.0,
                   eps_pe=6.5e-3)                    # precast strand
    t_top = Tendon(y=1300.0, x=300.0, material=ps, Ap=700.0,
                   eps_pe=0.0)                       # topping tendon
    sec = GenericSection(
        polygon=rect_poly(B, H), bulk_material=Concrete(fck=45.0),
        rebars=[], tendons=[t_pre, t_top],
        bulk_materials=[(box(0, H1, B, H), Concrete(fck=30.0),
                         "topping")],
        mesh_size=100.0, n_grid_x=6, n_grid_y=14)
    check("geometric parents: precast strand -> 0, topping -> 1",
          list(sec.staging_parent_tendon) == [0, 1])
    mgr = StagedDomainManager(sec, biaxial=False,
                              gen_kwargs={"n_points": 25})

    # Premature activation: the topping tendon (union index 1) left
    # active at stage 0 while its parent zone is uncast.
    check_raises(
        "topping tendon active before its zone is cast",
        ValueError, mgr.resolve_stages,
        [{"section_ops": {}},
         {"section_ops": {"activate_bulk": {1: (0.0, 0.0, 0.0)}}}])

    # Correct chain: deactivate at 0; cast at 1; enter at 2.
    stages = [
        {"section_ops": {"deactivate": [1], "release": False}},
        {"section_ops": {"activate_bulk": {1: (1.5e-4, 2e-7, 0.0)}}},
        {"section_ops": {"activate": [1],
                         "eps_override": {1: 5.5e-3}}},
    ]
    states, hashes, bundles, _ = mgr.resolve_stages(stages)
    check("containment chain resolves", len(states) == 3)
    check("stage-0 topping inactive, tendon inactive",
          not bool(states[0].bulk_active[1])
          and not bool(states[0].active[1]))
    check("stage-1 topping cast, tendon still inactive",
          bool(states[1].bulk_active[1])
          and not bool(states[1].active[1]))
    check("stage-2 tendon in with its stressing prestrain",
          bool(states[2].active[1])
          and float(states[2].eps_init[1]) == 5.5e-3)
    check("three distinct domain hashes (rebuild at each event)",
          len(set(hashes)) == 3)
    for k, bnd in enumerate(bundles):
        sol = bnd.solver.solve_equilibrium(-2.0e5, -1.0e8, 0.0)
        check(f"stage-{k} equilibrium converges", sol["converged"])


# ------------------------------------------------------------------

def _preflight():
    r"""
    Verify that the imported ``gensec`` package carries the Task-1
    patch before running any axis.

    The validator itself only imports pre-Task-1 symbols, so on an
    unpatched tree it would start and then die on a cryptic
    3-tuple-unpack error at the first named-zone fixture.  This probe
    turns that into an explicit message, and prints the imported
    package path to expose the other classic failure (patched repo,
    stale installed copy shadowing it).
    """
    import dataclasses
    from gensec.geometry import geometry as _geo
    from gensec.solver.section_state import SectionState as _SS
    fields = {f.name for f in dataclasses.fields(_SS)}
    patched = (hasattr(_geo.GenericSection, "_normalize_bulk_zones")
               and {"bulk_active", "bulk_planes"} <= fields)
    print(f"  gensec imported from: {_geo.__file__}")
    if not patched:
        print(
            "\nPREFLIGHT FAILED: this gensec package does not carry "
            "the Task-1 patch\n(missing GenericSection."
            "_normalize_bulk_zones and/or SectionState zone\n"
            "arrays).  Run `python apply_phase8_task1.py` from the "
            "repository root\nfirst; it must end with '... 0 failed' "
            "and exit code 0 — on any anchor\nfailure it writes "
            "NOTHING (atomic abort).  Then rerun this validator."
        )
        sys.exit(2)


def main():
    r"""Run the six axes; return a process exit code."""
    print("Phase 8 / Task 1 validation — engine bulk staging")
    _preflight()
    v1_identity()
    v2_masked_vs_standalone()
    v3_ab_regression()
    v4_invariants()
    v5_hash()
    v6_tendon_in_topping()
    print(f"\n{'=' * 66}")
    if _N_FAIL:
        print(f"RESULT: {_N_FAIL} FAILED / {_N_PASS + _N_FAIL} checks")
        return 1
    print(f"RESULT: ALL {_N_PASS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
