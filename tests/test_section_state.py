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
# along with GenSec. If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------

"""
Unit tests for the prestress Phase-3 infrastructure (section_state.py).

Covers:
- capacity state hash: sensitivity to eps_init (quantized), bonded and
  active masks; insensitivity within a QUANT_EPS bucket; the
  active-and-unbonded == inactive equivalence for resistance.
- path_schedule: reduces-to-identity (constant hash == legacy schedule)
  and the staged-construction / seismic / de-stressing patterns.
- SectionState transition helpers: immutable-style with_* operations.
- force release: numeric correctness on a linear material (isolated via
  lightweight stubs, since closed-form release force is impractical with
  the nonlinear constitutive laws) and wiring on a real section.
- VerificationEngine._check_staged reduces-to-identity: a single-hash
  staged manager reproduces the static (manager-less) engine.
"""

import unittest
import numpy as np

from gensec.materials import Concrete, Steel
from gensec.geometry import RebarLayer, RectSection
from gensec.solver import FiberSolver, NMDiagram, VerificationEngine
from gensec.solver.section_state import (
    QUANT_EPS,
    SectionState,
    StagedDomainManager,
    DomainBundle,
    PrestressAction,
    path_schedule,
    geometry_signature,
    materialize_view,
    released_force_action,
    _element_net_force,
)


# ------------------------------------------------------------------
#  Shared fixtures
# ------------------------------------------------------------------

def _rc_section(n_fibers_y=20):
    """300x500 RC section, 4 corner bars (no tendons)."""
    concrete = Concrete(fck=25.0)
    steel = Steel(fyk=450.0)
    As = np.pi / 4 * 16 ** 2
    rebars = [
        RebarLayer(y=45,  x=45,  As=As, material=steel),
        RebarLayer(y=45,  x=255, As=As, material=steel),
        RebarLayer(y=455, x=45,  As=As, material=steel),
        RebarLayer(y=455, x=255, As=As, material=steel),
    ]
    return RectSection(300, 500, concrete, rebars, n_fibers_y=n_fibers_y)


# Small generation budget — these tests check infrastructure, not
# domain accuracy.
_GEN_KW = dict(n_angles=12, n_points_per_angle=24)


# ==================================================================
#  Capacity state hash
# ==================================================================

class TestCapacityHash(unittest.TestCase):
    """Hash discriminates exactly what changes the resistance domain."""

    def setUp(self):
        self.sec = _rc_section()
        # A manager gives us a consistent geom signature + union
        # material ids without building any domain (we only call
        # hash_of, never get_bundle).
        self.mgr = StagedDomainManager.__new__(StagedDomainManager)
        self.mgr.base_section = self.sec
        self.mgr._geom_sig = geometry_signature(self.sec)
        self.mgr._union_materials = \
            StagedDomainManager._collect_union_materials(self.sec)
        self.base = self._all_active_state(self.sec)

    @staticmethod
    def _all_active_state(sec):
        n = int(sec.x_rebars.size) + int(getattr(sec, "x_tendons",
                                                 np.empty(0)).size)
        return SectionState(0, np.ones(n, bool), np.ones(n, bool),
                            np.zeros(n))

    def test_geometry_signature_stable_and_hashable(self):
        g1 = geometry_signature(self.sec)
        g2 = geometry_signature(self.sec)
        self.assertEqual(hash(g1), hash(g2))

    def test_eps_within_quantum_same_hash(self):
        h0 = self.mgr.hash_of(self.base)
        s = self.base.with_eps_override({0: 0.4 * QUANT_EPS})
        self.assertEqual(self.mgr.hash_of(s), h0)

    def test_eps_across_quantum_changes_hash(self):
        h0 = self.mgr.hash_of(self.base)
        s = self.base.with_eps_override({0: 2.0 * QUANT_EPS})
        self.assertNotEqual(self.mgr.hash_of(s), h0)

    def test_bonded_flip_changes_hash(self):
        h0 = self.mgr.hash_of(self.base)
        s = self.base.copy_advanced(0)
        s.bonded[0] = False
        self.assertNotEqual(self.mgr.hash_of(s), h0)

    def test_active_flip_changes_hash(self):
        h0 = self.mgr.hash_of(self.base)
        s = self.base.with_deactivated([0])
        self.assertNotEqual(self.mgr.hash_of(s), h0)

    def test_active_unbonded_equals_inactive(self):
        # An active-but-unbonded element is not in the resistance
        # domain, so it must hash identically to a removed element.
        s_unbonded = self.base.copy_advanced(0)
        s_unbonded.bonded[0] = False
        s_inactive = self.base.with_deactivated([0])
        self.assertEqual(self.mgr.hash_of(s_unbonded),
                         self.mgr.hash_of(s_inactive))

    def test_prestress_action_not_in_hash(self):
        # PrestressAction never reaches the hash; constructing one and
        # leaving the state untouched must not change the hash.
        h0 = self.mgr.hash_of(self.base)
        _ = PrestressAction(500e3, 20e6, 0.0, origin="transfer")
        self.assertEqual(self.mgr.hash_of(self.base), h0)


# ==================================================================
#  Path schedule (reduces-to-identity)
# ==================================================================

class TestPathSchedule(unittest.TestCase):
    """The path-reset scheduler and its legacy-equivalence guarantee."""

    @staticmethod
    def _legacy(n):
        return [{"stage": k, "domain_hash": "H", "reset": (k == 0),
                 "path_base": "origin" if k == 0 else "prev_cum"}
                for k in range(n)]

    def test_reduces_to_identity_constant_hash(self):
        for n in (1, 2, 5):
            self.assertEqual(path_schedule(["H"] * n), self._legacy(n))

    def test_staged_construction_reset_at_activation(self):
        s = path_schedule(["A", "B", "B", "B"])
        self.assertEqual([e["reset"] for e in s],
                         [True, True, False, False])
        self.assertEqual([e["path_base"] for e in s],
                         ["origin", "carry", "prev_cum", "prev_cum"])

    def test_seismic_single_hash_no_mid_resets(self):
        s = path_schedule(["S"] * 6)
        self.assertTrue(all(not e["reset"] for e in s[1:]))

    def test_destressing_reset_at_cut(self):
        s = path_schedule(["A", "A", "C", "C"])
        self.assertEqual([e["reset"] for e in s],
                         [True, False, True, False])


# ==================================================================
#  SectionState transitions (immutable-style)
# ==================================================================

class TestStateTransitions(unittest.TestCase):

    def setUp(self):
        self.s = SectionState(0, np.ones(3, bool), np.ones(3, bool),
                              np.zeros(3))

    def test_with_deactivated_does_not_mutate_parent(self):
        s2 = self.s.with_deactivated([2])
        self.assertEqual(s2.active.tolist(), [True, True, False])
        self.assertTrue(self.s.active.all())          # parent untouched

    def test_with_activated(self):
        s2 = self.s.with_deactivated([2]).with_activated([2])
        self.assertTrue(s2.active.all())

    def test_with_eps_override_isolated(self):
        s2 = self.s.with_eps_override({2: 0.006})
        self.assertEqual(s2.eps_init[2], 0.006)
        self.assertEqual(self.s.eps_init[2], 0.0)     # parent untouched

    def test_with_bulk_eps(self):
        s2 = self.s.with_bulk_eps(-2e-4)
        self.assertEqual(s2.bulk_eps_init, -2e-4)
        self.assertEqual(self.s.bulk_eps_init, 0.0)


# ==================================================================
#  Force release — numeric (linear material, isolated)
# ==================================================================

class _LinMat:
    """sigma = E * eps; minimal interface used by _element_net_force."""

    def __init__(self, E):
        self.E = E
        self.eps_min = -1.0
        self.eps_max = 1.0
        self.name = "lin"

    def stress_array(self, eps):
        return self.E * np.asarray(eps, dtype=float)


class _FakeSec:
    pass


class _FakeReb:
    def __init__(self, material):
        self.material = material


class _FakeSolver:
    """Returns a fixed converged strain plane, mimicking FiberSolver."""

    def __init__(self, sec, eps0, chi_x, chi_y):
        self.sec = sec
        self.x_ref = 150.0
        self.y_ref = 250.0
        self._plane = (eps0, chi_x, chi_y)

    def solve_equilibrium(self, N, Mx, My, **kw):
        e0, cx, cy = self._plane
        return {"converged": True, "eps0": e0, "chi_x": cx, "chi_y": cy}


class TestForceReleaseNumeric(unittest.TestCase):
    """
    Isolated numeric check of the released force on a linear material.

    Closed-form release force is impractical with the real nonlinear
    constitutive laws, so the formula is verified here against stubs;
    the wiring on a real section is checked separately below.
    """

    def setUp(self):
        self.steel = _LinMat(200000.0)
        self.bulk = _LinMat(30000.0)
        sec = _FakeSec()
        sec.x_rebars = np.array([40.0, 260.0])
        sec.y_rebars = np.array([40.0, 460.0])
        sec.A_rebars = np.array([400.0, 400.0])
        sec.embedded_rebars = np.array([False, True])
        sec.rebars = [_FakeReb(self.steel), _FakeReb(self.steel)]
        sec.x_tendons = np.array([150.0])
        sec.y_tendons = np.array([100.0])
        sec.A_tendons = np.array([150.0])
        sec.embedded_tendons = np.array([True])
        sec.eps_init_tendons = np.array([0.006])
        sec.tendons = [_FakeReb(self.steel)]
        sec.bulk_material = self.bulk
        sec._union_index = [0, 1, 2]
        self.sec = sec
        self.eps0 = 1e-3
        self.solver = _FakeSolver(sec, self.eps0, 0.0, 0.0)
        self.bundle = DomainBundle(solver=self.solver, nm_gen=None,
                                   domain=None)
        self.mgr = StagedDomainManager.__new__(StagedDomainManager)

    def test_rebar_non_embedded_force(self):
        # F = E * eps * A = 200000 * 1e-3 * 400 = 80000 N
        F, Mx, My = _element_net_force(self.solver, 0, self.eps0, 0, 0)
        self.assertAlmostEqual(F, 80000.0, places=3)
        ly = 40.0 - self.solver.y_ref
        lx = 40.0 - self.solver.x_ref
        self.assertAlmostEqual(Mx, F * ly, places=3)
        self.assertAlmostEqual(My, -F * lx, places=3)

    def test_rebar_embedded_net_subtraction(self):
        # net = (E_s - E_b) * eps * A = (200000-30000)*1e-3*400 = 68000
        F, _, _ = _element_net_force(self.solver, 1, self.eps0, 0, 0)
        self.assertAlmostEqual(F, 68000.0, places=3)

    def test_tendon_embedded_with_prestrain(self):
        # sigma = E_s*(eps+eps_init) - E_b*eps
        #       = 200000*(1e-3+6e-3) - 30000*1e-3 = 1370 MPa
        # F = 1370 * 150 = 205500 N
        F, _, _ = _element_net_force(self.solver, 2, self.eps0, 0, 0)
        self.assertAlmostEqual(F, 205500.0, places=3)

    def test_deactivation_action_sign(self):
        F, Mx, My = _element_net_force(self.solver, 0, self.eps0, 0, 0)
        acts = StagedDomainManager.deactivation_actions(
            self.mgr, self.bundle, [0], (0.0, 0.0, 0.0), release=True)
        self.assertEqual(len(acts), 1)
        self.assertEqual(acts[0].origin, "release")
        self.assertAlmostEqual(acts[0].N, -F, places=3)
        self.assertAlmostEqual(acts[0].Mx, -Mx, places=3)
        self.assertAlmostEqual(acts[0].My, -My, places=3)

    def test_clean_removal_no_action(self):
        acts = StagedDomainManager.deactivation_actions(
            self.mgr, self.bundle, [0], (0.0, 0.0, 0.0), release=False)
        self.assertEqual(acts, [])

    def test_absent_index_skipped(self):
        self.sec._union_index = [1, 2]   # element 0 not in this view
        acts = StagedDomainManager.deactivation_actions(
            self.mgr, self.bundle, [0], (0.0, 0.0, 0.0), release=True)
        self.assertEqual(acts, [])

    def test_released_force_action_matches(self):
        F, Mx, My = _element_net_force(self.solver, 0, self.eps0, 0, 0)
        act = released_force_action(
            self.bundle, 0, self.solver.x_ref, self.solver.y_ref,
            0.0, 0.0, 0.0)
        self.assertAlmostEqual(act.N, -F, places=3)


# ==================================================================
#  _check_staged reduces-to-identity on a real section
# ==================================================================

class TestCheckStagedReducesToIdentity(unittest.TestCase):
    """
    A single-hash staged manager must reproduce the static engine.

    Both run with the Phase-3 code; the manager-driven engine uses the
    full (all-active) section at every stage, so every stage shares one
    hash and the path metrics must match the manager-less engine.
    """

    def setUp(self):
        self.sec = _rc_section(n_fibers_y=20)
        self.solver = FiberSolver(self.sec)
        self.nm_gen = NMDiagram(self.solver)
        self.cloud = self.nm_gen.generate_biaxial(**_GEN_KW)
        self.flags = {
            "eta_norm": True, "eta_norm_beta": True,
            "eta_path_norm_ray": True, "eta_path_norm_beta": True,
        }
        self.demands = {
            "G":  {"N": -600e3, "Mx": 25e6, "My": 5e6},
            "Ex": {"N": 0.0,    "Mx": 90e6, "My": 30e6},
        }
        self.stages = [
            {"name": "gravity",
             "components": [{"ref": "G", "factor": 1.0}]},
            {"name": "seismic1",
             "components": [{"ref": "Ex", "factor": 1.0}]},
            {"name": "seismic2",
             "components": [{"ref": "Ex", "factor": 0.5}]},
        ]

    def test_single_hash_manager_matches_static(self):
        eng_static = VerificationEngine(self.cloud, self.nm_gen,
                                        self.flags)
        mgr = StagedDomainManager(self.sec, biaxial=True,
                                  gen_kwargs=_GEN_KW)
        eng_staged = VerificationEngine(self.cloud, self.nm_gen,
                                        self.flags, staged_manager=mgr)

        r_static = eng_static._check_staged("seis", self.stages,
                                            self.demands)
        r_staged = eng_staged._check_staged("seis", self.stages,
                                            self.demands)

        self.assertEqual(r_static["verified"], r_staged["verified"])
        self.assertAlmostEqual(r_static["eta_governing"],
                               r_staged["eta_governing"], places=6)
        for a, b in zip(r_static["stages"], r_staged["stages"]):
            for key in ("eta_path_norm_ray", "eta_path_norm_beta",
                        "eta_norm", "eta_norm_beta"):
                if a.get(key) is not None:
                    self.assertAlmostEqual(a[key], b.get(key), places=6,
                                           msg=f"stage metric {key}")


class TestCheckStagedConstructionE2E(unittest.TestCase):
    """
    End-to-end staged construction through ``_check_staged`` with real
    ``section_ops`` on a real section — the headline Phase-3 capability:
    the section changes between stages, demand accumulates, and the path
    resets at the activation boundary, all inside one staged check.

    Section: 4 corner bars; the two top bars (union indices 2, 3) are
    declared inactive at stage 0 and activated at stage 1.
    """

    def setUp(self):
        self.sec = _rc_section(n_fibers_y=20)
        self.mgr = StagedDomainManager(self.sec, biaxial=True,
                                       gen_kwargs=_GEN_KW)
        # A throwaway domain just to satisfy VerificationEngine.__init__;
        # the staged run uses the manager's per-stage domains.
        solver = FiberSolver(self.sec)
        cloud = NMDiagram(solver).generate_biaxial(**_GEN_KW)
        self.flags = {"eta_norm": True, "eta_norm_beta": True,
                      "eta_path_norm_ray": True}
        self.eng = VerificationEngine(cloud, NMDiagram(solver),
                                      self.flags, staged_manager=self.mgr)
        self.demands = {"G": {"N": -500e3, "Mx": 20e6, "My": 0.0},
                        "E": {"N": 0.0, "Mx": 70e6, "My": 0.0}}
        self.stages = [
            # Stage 0: only the bottom bars are built.
            {"name": "build-bottom",
             "components": [{"ref": "G", "factor": 1.0}],
             "section_ops": {"deactivate": [2, 3]}},
            # Stage 1: top bars added (the "new element").
            {"name": "add-top",
             "components": [{"ref": "E", "factor": 1.0}],
             "section_ops": {"activate": [2, 3]}},
            # Stage 2: more load, section unchanged → continuing path.
            {"name": "service",
             "components": [{"ref": "E", "factor": 0.5}]},
        ]

    def test_staged_construction_through_check(self):
        r = self.eng._check_staged("staged-build", self.stages,
                                   self.demands)
        st = r["stages"]
        # Hash changes when the top bars are activated.
        self.assertNotEqual(st[0]["domain_hash"], st[1]["domain_hash"])
        # Stage 2 keeps stage 1's section.
        self.assertEqual(st[1]["domain_hash"], st[2]["domain_hash"])
        # Resets: stage 0 (origin), stage 1 (hash change), not stage 2.
        self.assertEqual([s["domain_reset"] for s in st],
                         [True, True, False])
        # The run produces a finite governing utilisation.
        self.assertTrue(np.isfinite(r["eta_governing"]))

    def test_activation_enlarges_the_domain(self):
        # Two distinct section states must build two distinct, ordered
        # domains: activating reinforcement can only enlarge the cloud.
        states, hashes, bundles, _ = self.mgr.resolve_stages(self.stages)
        self.assertNotEqual(hashes[0], hashes[1])

        def extent(b):
            P = np.column_stack([b.cloud["N"], b.cloud["Mx"],
                                 b.cloud["My"]])
            return float(np.prod(P.max(axis=0) - P.min(axis=0)))

        self.assertGreater(extent(bundles[1]), extent(bundles[0]))


class TestAnalyzeStagedReducesToIdentity(unittest.TestCase):
    """
    A single-hash staged manager must reproduce the static analysis
    engine (regression guard for the analysis.py Phase-3 upgrade).
    """

    def setUp(self):
        from gensec.solver import AnalysisEngine
        self.AnalysisEngine = AnalysisEngine
        self.sec = _rc_section(n_fibers_y=20)
        self.solver = FiberSolver(self.sec)
        self.demands = {"G": {"N": -600e3, "Mx": 25e6, "My": 0.0},
                        "E": {"N": 0.0, "Mx": 80e6, "My": 0.0}}
        self.combo = {
            "name": "seis",
            "stages": [
                {"name": "g",  "components": [{"ref": "G", "factor": 1.0}]},
                {"name": "s1", "components": [{"ref": "E", "factor": 1.0}]},
            ],
        }

    def test_single_hash_manager_matches_static(self):
        eng_static = self.AnalysisEngine(self.solver)
        mgr = StagedDomainManager(self.sec, biaxial=True,
                                  gen_kwargs=_GEN_KW)
        eng_staged = self.AnalysisEngine(self.solver, staged_manager=mgr)

        r_static = eng_static.analyze_combinations(
            [self.combo], self.demands)[0]
        r_staged = eng_staged.analyze_combinations(
            [self.combo], self.demands)[0]

        for a, b in zip(r_static["stages"], r_staged["stages"]):
            b = dict(b)
            b.pop("domain_hash", None)   # provenance only in staged path
            self.assertEqual(a["cumulative"], b["cumulative"])


if __name__ == "__main__":
    unittest.main()
