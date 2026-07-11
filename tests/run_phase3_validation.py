"""
Phase-3 validation runner.  Run in the real gensec tree:

    python run_phase3_validation.py

Drives the Phase-3 infrastructure **programmatically** (the YAML stage-op
schema and its loader are a separate task — io_yaml.py is not in the Phase-3
"Touches" list).  Two cases:

  (1) no-prestress staged construction — proves capacity staging in isolation:
      a rebar group declared inactive at stage 0, activated at stage 1.  The
      two stages must have different capacity hashes (two domain builds), and
      the stage-1 domain (more steel) must be strictly larger.

  (2) seismic demand staging on a fixed section — proves demand staging is a
      correct special case: single hash across all stages, so the manager-
      driven staged path must equal the legacy single-domain staged path,
      number for number.  Self-checking (no external baseline needed).

These assertions are written to fail loudly.  This script is delivered
unrun (the authoring workspace is a flattened copy and cannot import gensec).
"""
import numpy as np
from shapely.geometry import box

from gensec.geometry.section import GenericSection
from gensec.geometry.fiber import RebarLayer
from gensec.materials.concrete import Concrete
from gensec.materials.steel import Steel
from gensec.solver.integrator import FiberSolver
from gensec.solver.capacity import NMDiagram
from gensec.solver.check import DomainChecker, VerificationEngine
from gensec.solver.section_state import (
    SectionState, StagedDomainManager, materialize_view,
)

GEN_KW = dict(n_angles=24, n_points_per_angle=40)   # small/fast


def _domain_extent(bundle):
    """Size proxy: bounding-box volume of the (N,Mx,My) cloud."""
    c = bundle.cloud
    P = np.column_stack([np.asarray(c["N"]),
                         np.asarray(c["Mx"]),
                         np.asarray(c["My"])])
    return float(np.prod(P.max(axis=0) - P.min(axis=0)))


def build_section():
    """300x500 RC section. Group A (bottom) + group B (top)."""
    poly = box(0, 0, 300, 500)
    conc = Concrete(fck=25)
    steel = Steel(fyk=450, gamma_s=1.15)
    A = [RebarLayer(x=40, y=40, As=400.0, material=steel),
         RebarLayer(x=260, y=40, As=400.0, material=steel)]
    B = [RebarLayer(x=40, y=460, As=400.0, material=steel),
         RebarLayer(x=260, y=460, As=400.0, material=steel)]
    sec = GenericSection(poly, conc, A + B, mesh_size=20)
    return sec, len(A), len(B)


# ==================================================================
#  Case 1: staged construction
# ==================================================================

def case1_staged_construction():
    print("\n[Case 1] no-prestress staged construction")
    sec, nA, nB = build_section()
    n_union = nA + nB                       # rebars only, no tendons

    mgr = StagedDomainManager(sec, biaxial=True, gen_kwargs=GEN_KW)

    # Stage 0: only group A active.
    active0 = np.array([True] * nA + [False] * nB)
    bonded0 = np.ones(n_union, bool)
    eps0 = np.zeros(n_union)
    s0 = SectionState(0, active0, bonded0, eps0, label="A only")

    # Stage 1: activate group B.
    active1 = np.ones(n_union, bool)
    s1 = SectionState(1, active1, bonded0.copy(), eps0.copy(),
                      label="A+B")

    h0, b0, built0 = mgr.get_bundle(s0)
    h1, b1, built1 = mgr.get_bundle(s1)

    assert h0 != h1, "stage 0 and 1 must differ (element activated)"
    assert built0 and built1, "both stages must build (cache miss)"
    e0, e1 = _domain_extent(b0), _domain_extent(b1)
    assert e1 > e0, f"stage-1 domain must be larger (e0={e0:.3e}, e1={e1:.3e})"

    # Re-requesting stage 0 must hit the cache (no rebuild).
    _h0, _b0, built0b = mgr.get_bundle(s0)
    assert not built0b and _b0 is b0, "stage-0 re-request must reuse cache"
    print(f"   hashes differ, 2 builds, extent {e0:.3e} -> {e1:.3e}, "
          "cache reuse OK")


# ==================================================================
#  Case 2: seismic demand staging == legacy (reduces to identity)
# ==================================================================

def case2_seismic_reduces_to_identity():
    print("\n[Case 2] seismic demand staging reproduces legacy path")
    sec, nA, nB = build_section()
    n_union = nA + nB
    solver = FiberSolver(sec)
    nm_gen = NMDiagram(solver)
    cloud = nm_gen.generate_biaxial(**GEN_KW)

    flags = {"eta_norm": True, "eta_norm_beta": True,
             "eta_path_norm_ray": True, "eta_path_norm_beta": True}

    # Staged combination: gravity then two seismic increments.
    demands = {
        "G":  {"N": -800e3, "Mx": 30e6,  "My": 0.0},
        "Ex": {"N": 0.0,    "Mx": 120e6, "My": 40e6},
    }
    stages = [
        {"name": "gravity", "components": [{"ref": "G", "factor": 1.0}]},
        {"name": "seismic1", "components": [{"ref": "Ex", "factor": 1.0}]},
        {"name": "seismic2", "components": [{"ref": "Ex", "factor": 0.5}]},
    ]

    # Legacy: single fixed domain.
    eng_legacy = VerificationEngine(cloud, nm_gen, flags)
    res_legacy = eng_legacy._check_staged("seis", stages, demands)

    # Phase-3: manager with an all-active fixed state for every stage,
    # so every stage hashes identically -> single domain -> must match.
    mgr = StagedDomainManager(sec, biaxial=True, gen_kwargs=GEN_KW)
    eng_v3 = VerificationEngine(cloud, nm_gen, flags, staged_manager=mgr)
    res_v3 = eng_v3._check_staged("seis", stages, demands)

    def path_etas(res):
        out = []
        for sr in res["stages"]:
            out.append((sr.get("eta_path_norm_ray"),
                        sr.get("eta_path_norm_beta")))
        return out

    pe_l, pe_v = path_etas(res_legacy), path_etas(res_v3)
    assert pe_l == pe_v, f"path etas diverge:\n  legacy={pe_l}\n  v3={pe_v}"
    assert res_legacy["eta_governing"] == res_v3["eta_governing"], \
        "governing eta diverges"
    print(f"   single hash, path etas identical: {pe_v}")
    print(f"   eta_governing legacy==v3: {res_v3['eta_governing']}")


# ==================================================================
#  Case 3: staged construction THROUGH _check_staged (headline path)
# ==================================================================

def case3_staged_construction_through_check():
    print("\n[Case 3] staged construction through _check_staged")
    sec, nA, nB = build_section()
    mgr = StagedDomainManager(sec, biaxial=True, gen_kwargs=GEN_KW)
    solver = FiberSolver(sec)
    cloud = NMDiagram(solver).generate_biaxial(**GEN_KW)
    flags = {"eta_norm": True, "eta_norm_beta": True,
             "eta_path_norm_ray": True}
    eng = VerificationEngine(cloud, NMDiagram(solver), flags,
                             staged_manager=mgr)

    # Group B = union indices [nA, nA+nB).  Inactive at stage 0, added
    # at stage 1, section unchanged at stage 2.
    B_idx = list(range(nA, nA + nB))
    demands = {"G": {"N": -500e3, "Mx": 20e6, "My": 0.0},
               "E": {"N": 0.0, "Mx": 70e6, "My": 0.0}}
    stages = [
        {"name": "build-A", "components": [{"ref": "G", "factor": 1.0}],
         "section_ops": {"deactivate": B_idx}},
        {"name": "add-B", "components": [{"ref": "E", "factor": 1.0}],
         "section_ops": {"activate": B_idx}},
        {"name": "service", "components": [{"ref": "E", "factor": 0.5}]},
    ]
    r = eng._check_staged("staged-build", stages, demands)
    st = r["stages"]
    assert st[0]["domain_hash"] != st[1]["domain_hash"], \
        "activation must change the hash"
    assert st[1]["domain_hash"] == st[2]["domain_hash"], \
        "unchanged section must keep the hash"
    assert [s["domain_reset"] for s in st] == [True, True, False], \
        [s["domain_reset"] for s in st]
    assert np.isfinite(r["eta_governing"])

    # Activation enlarges the domain.
    _s, _h, bundles, _d = mgr.resolve_stages(stages)
    assert _domain_extent(bundles[1]) > _domain_extent(bundles[0])
    print(f"   hash change at activation, resets "
          f"{[s['domain_reset'] for s in st]}, "
          f"eta_gov={r['eta_governing']}, domain enlarged: OK")


# ==================================================================
#  Case 4: staged analysis reduces to identity (single hash)
# ==================================================================

def case4_analysis_staged_reduces_to_identity():
    print("\n[Case 4] staged analysis reduces to identity")
    from gensec.solver import AnalysisEngine
    sec, nA, nB = build_section()
    solver = FiberSolver(sec)
    mgr = StagedDomainManager(sec, biaxial=True, gen_kwargs=GEN_KW)
    demands = {"G": {"N": -600e3, "Mx": 25e6, "My": 0.0},
               "E": {"N": 0.0, "Mx": 80e6, "My": 0.0}}
    combo = {"name": "seis", "stages": [
        {"name": "g",  "components": [{"ref": "G", "factor": 1.0}]},
        {"name": "s1", "components": [{"ref": "E", "factor": 1.0}]},
    ]}
    r_static = AnalysisEngine(solver).analyze_combinations(
        [combo], demands)[0]
    r_staged = AnalysisEngine(solver, staged_manager=mgr
                              ).analyze_combinations([combo], demands)[0]
    for a, b in zip(r_static["stages"], r_staged["stages"]):
        assert a["cumulative"] == b["cumulative"], (a["cumulative"],
                                                    b["cumulative"])
    print("   single-hash staged analysis == static analysis: OK")


if __name__ == "__main__":
    case1_staged_construction()
    case2_seismic_reduces_to_identity()
    case3_staged_construction_through_check()
    case4_analysis_staged_reduces_to_identity()
    print("\nPHASE-3 VALIDATION OK")
