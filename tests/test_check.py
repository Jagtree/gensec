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
Unit tests for the demand verification module (check.py) — v0.2.0.

Covers:
- DomainChecker 3D hull: is_inside, eta_3D, eta_path
- DomainChecker 2D fallback (uniaxial domain)
- MxMyContour: is_inside, eta_2D, eta_path_2D, symmetry
- VerificationEngine: simple demands, simple combinations,
  staged combinations, envelopes, output flags, contour cache
- Static helpers: resolve_ref, resolve_components
"""

import unittest
import numpy as np
from gensec.materials import Concrete, Steel
from gensec.geometry import RebarLayer, RectSection
from gensec.solver import (
    FiberSolver, NMDiagram, DomainChecker, MxMyContour, VerificationEngine,
)


# ------------------------------------------------------------------
#  Shared fixture factories
# ------------------------------------------------------------------

def _biaxial_section():
    """300×500 column with 4 corner bars — biaxial fixture."""
    concrete = Concrete(fck=25.0)
    steel = Steel(fyk=450.0)
    As = np.pi / 4 * 16 ** 2
    rb = [
        RebarLayer(y=45,  x=45,  As=As, material=steel),
        RebarLayer(y=45,  x=255, As=As, material=steel),
        RebarLayer(y=455, x=45,  As=As, material=steel),
        RebarLayer(y=455, x=255, As=As, material=steel),
    ]
    return RectSection(B=300, H=500, bulk_material=concrete,
                       rebars=rb, n_fibers_y=30, n_fibers_x=15)


def _uniaxial_section():
    """300×600 beam/column with 2 rebar layers — uniaxial fixture."""
    concrete = Concrete(fck=25.0)
    steel = Steel(fyk=450.0)
    rb = [
        RebarLayer(y=40,  As=942.5, material=steel),
        RebarLayer(y=560, As=942.5, material=steel),
    ]
    return RectSection(B=300, H=600, bulk_material=concrete,
                       rebars=rb, n_fibers_y=120, n_fibers_x=1)


# Shared demand database reused across several test classes
_DEMAND_DB = {
    "G":  {"N": -500e3, "Mx":  20e6, "My":  5e6},
    "Q1": {"N": -100e3, "Mx":  40e6, "My": 10e6},
    "Ex": {"N":   50e3, "Mx":  80e6, "My": 60e6},
    "Ey": {"N":   20e3, "Mx":  30e6, "My": 90e6},
}


# ==================================================================
#  DomainChecker — 3D biaxial hull
# ==================================================================

class TestDomainChecker3D(unittest.TestCase):
    """DomainChecker built from a biaxial 3D point cloud."""

    @classmethod
    def setUpClass(cls):
        sec = _biaxial_section()
        sv = FiberSolver(sec)
        cls.nm_gen = NMDiagram(sv)
        nm_3d = cls.nm_gen.generate_biaxial(n_angles=24, n_points_per_angle=80)
        cls.checker = DomainChecker(nm_3d)

    def test_ndim_is_3(self):
        self.assertEqual(self.checker.ndim, 3)

    def test_N_range_positive(self):
        self.assertGreater(self.checker.N_range, 0)

    def test_origin_inside(self):
        self.assertTrue(self.checker.is_inside(0.0, 0.0, 0.0))

    def test_small_demand_inside(self):
        self.assertTrue(self.checker.is_inside(-200e3, 10e6, 5e6))

    def test_huge_compression_outside(self):
        self.assertFalse(self.checker.is_inside(-5000e3, 0.0, 0.0))

    def test_huge_moment_outside(self):
        self.assertFalse(self.checker.is_inside(-300e3, 500e6, 300e6))

    # --- eta_3D ---

    def test_eta_3D_origin_is_zero(self):
        eta = self.checker.eta_3D(0.0, 0.0, 0.0)
        self.assertEqual(eta, 0.0)

    def test_eta_3D_inside_less_than_1(self):
        eta = self.checker.eta_3D(-200e3, 10e6, 5e6)
        self.assertGreater(eta, 0.0)
        self.assertLess(eta, 1.0)

    def test_eta_3D_outside_greater_than_1(self):
        eta = self.checker.eta_3D(-300e3, 600e6, 300e6)
        self.assertGreater(eta, 1.0)

    def test_eta_3D_scales_monotonically(self):
        # Double the demand → double eta (ray from origin is linear)
        Mx = 10e6
        eta1 = self.checker.eta_3D(-100e3, Mx, 0.0)
        eta2 = self.checker.eta_3D(-100e3, 2 * Mx, 0.0)
        # Not exactly 2× due to hull approximation, but eta2 > eta1
        self.assertGreater(eta2, eta1)

    def test_is_inside_consistent_with_eta(self):
        N, Mx, My = -200e3, 10e6, 4e6
        eta = self.checker.eta_3D(N, Mx, My)
        inside = self.checker.is_inside(N, Mx, My)
        self.assertLess(eta, 1.0)
        self.assertTrue(inside)

    # --- eta_path ---

    def test_eta_path_both_inside(self):
        eta = self.checker.eta_path(
            -100e3, 5e6, 2e6,
            -200e3, 10e6, 5e6,
        )
        self.assertGreater(eta, 0.0)
        self.assertLess(eta, 1.0)

    def test_eta_path_target_outside(self):
        eta = self.checker.eta_path(
            -100e3, 5e6, 0.0,
            -400e3, 600e6, 300e6,
        )
        self.assertGreater(eta, 1.0)

    def test_eta_path_coincident_base_target_inside(self):
        # base == target and inside → 0.0
        eta = self.checker.eta_path(
            -100e3, 5e6, 2e6,
            -100e3, 5e6, 2e6,
        )
        self.assertEqual(eta, 0.0)

    def test_eta_path_base_is_origin_equals_eta_3D(self):
        # eta_path from origin == eta_3D (by definition)
        N, Mx, My = -200e3, 10e6, 5e6
        eta_path = self.checker.eta_path(0.0, 0.0, 0.0, N, Mx, My)
        eta_3d   = self.checker.eta_3D(N, Mx, My)
        self.assertAlmostEqual(eta_path, eta_3d, places=4)


# ==================================================================
#  DomainChecker — 2D uniaxial fallback
# ==================================================================

class TestDomainChecker2DFallback(unittest.TestCase):
    """DomainChecker built from a uniaxial N-M domain (ndim=2)."""

    @classmethod
    def setUpClass(cls):
        sec = _uniaxial_section()
        nm = NMDiagram(FiberSolver(sec)).generate(400)
        cls.checker = DomainChecker(nm)

    def test_ndim_is_2(self):
        self.assertEqual(self.checker.ndim, 2)

    def test_origin_inside(self):
        self.assertTrue(self.checker.is_inside(0.0, 0.0))

    def test_inside_point(self):
        self.assertTrue(self.checker.is_inside(-500e3, 50e6))

    def test_outside_point(self):
        self.assertFalse(self.checker.is_inside(-4000e3, 0.0))

    def test_eta_3D_inside(self):
        eta = self.checker.eta_3D(-500e3, 50e6)
        self.assertLess(eta, 1.0)

    def test_eta_3D_outside(self):
        eta = self.checker.eta_3D(-500e3, 500e6)
        self.assertGreater(eta, 1.0)

    def test_eta_path_from_origin_equals_eta_3D(self):
        N, Mx = -400e3, 60e6
        ep = self.checker.eta_path(0.0, 0.0, 0.0, N, Mx, 0.0)
        e3 = self.checker.eta_3D(N, Mx)
        self.assertAlmostEqual(ep, e3, places=4)


# ==================================================================
#  MxMyContour — 2D contour at fixed N
# ==================================================================

class TestMxMyContour(unittest.TestCase):
    """MxMyContour: is_inside, eta_2D, eta_path_2D, symmetry."""

    @classmethod
    def setUpClass(cls):
        sec = _biaxial_section()
        sv = FiberSolver(sec)
        cls.nm_gen = NMDiagram(sv)
        # Contour at -300 kN
        mx_my_data = cls.nm_gen.generate_mx_my(
            -300e3, n_angles=36, n_points_per_angle=80)
        cls.contour = MxMyContour(mx_my_data)

    def test_origin_inside(self):
        self.assertTrue(self.contour.is_inside(0.0, 0.0))

    def test_small_moment_inside(self):
        self.assertTrue(self.contour.is_inside(10e6, 5e6))

    def test_huge_moment_outside(self):
        self.assertFalse(self.contour.is_inside(1000e6, 1000e6))

    # --- eta_2D ---

    def test_eta_2D_origin_is_zero(self):
        eta = self.contour.eta_2D(0.0, 0.0)
        self.assertEqual(eta, 0.0)

    def test_eta_2D_inside_less_than_1(self):
        eta = self.contour.eta_2D(10e6, 5e6)
        self.assertGreater(eta, 0.0)
        self.assertLess(eta, 1.0)

    def test_eta_2D_outside_greater_than_1(self):
        eta = self.contour.eta_2D(1000e6, 1000e6)
        self.assertGreater(eta, 1.0)

    def test_eta_2D_consistent_with_is_inside(self):
        Mx, My = 10e6, 5e6
        eta = self.contour.eta_2D(Mx, My)
        inside = self.contour.is_inside(Mx, My)
        self.assertLess(eta, 1.0)
        self.assertTrue(inside)

    # --- eta_path_2D ---

    def test_eta_path_2D_within(self):
        eta = self.contour.eta_path_2D(5e6, 2e6, 10e6, 5e6)
        self.assertGreater(eta, 0.0)
        self.assertLess(eta, 1.0)

    def test_eta_path_2D_exceeds_boundary(self):
        eta = self.contour.eta_path_2D(5e6, 0.0, 1000e6, 1000e6)
        self.assertGreater(eta, 1.0)

    def test_eta_path_2D_from_origin_equals_eta_2D(self):
        Mx, My = 10e6, 5e6
        ep = self.contour.eta_path_2D(0.0, 0.0, Mx, My)
        e2 = self.contour.eta_2D(Mx, My)
        self.assertAlmostEqual(ep, e2, places=4)

    # --- symmetry ---

    def test_symmetry_mx_axis(self):
        eta_pos = self.contour.eta_2D(30e6, 0.0)
        eta_neg = self.contour.eta_2D(-30e6, 0.0)
        self.assertAlmostEqual(eta_pos, eta_neg, delta=0.05)

    def test_symmetry_my_axis(self):
        eta_pos = self.contour.eta_2D(0.0, 20e6)
        eta_neg = self.contour.eta_2D(0.0, -20e6)
        self.assertAlmostEqual(eta_pos, eta_neg, delta=0.05)

    def test_contour_near_pure_compression_is_small(self):
        # Very close to NRd,max (≈ -2100 kN for a 300×500 fck=25 section),
        # the Mx-My capacity approaches zero: the maximum total moment
        # must be much smaller than at moderate N=-300 kN.
        data_high = self.nm_gen.generate_mx_my(
            -1900e3, n_angles=36, n_points_per_angle=80)
        M_low  = np.sqrt(
            self.nm_gen.generate_mx_my(-300e3, n_angles=36,
                                       n_points_per_angle=80)["Mx"] ** 2
            + self.nm_gen.generate_mx_my(-300e3, n_angles=36,
                                         n_points_per_angle=80)["My"] ** 2
        ).max()
        M_high = np.sqrt(
            data_high["Mx"] ** 2 + data_high["My"] ** 2).max()
        self.assertLess(M_high, M_low)


# ==================================================================
#  VerificationEngine — static helpers
# ==================================================================

class TestVerificationEngineHelpers(unittest.TestCase):
    """Static methods: resolve_ref, resolve_components."""

    def test_resolve_ref_from_demand_db(self):
        db = {"G": {"N": -500e3, "Mx": 20e6, "My": 5e6}}
        N, Mx, My = VerificationEngine.resolve_ref("G", db)
        self.assertAlmostEqual(N, -500e3)
        self.assertAlmostEqual(Mx, 20e6)
        self.assertAlmostEqual(My, 5e6)

    def test_resolve_ref_missing_raises_key_error(self):
        with self.assertRaises(KeyError):
            VerificationEngine.resolve_ref("X", {})

    def test_resolve_ref_from_combination_results(self):
        db = {}
        combo_results = {
            "SLU_1": {
                "resultant": {"N": -700e3, "Mx": 60e6, "My": 15e6}
            }
        }
        N, Mx, My = VerificationEngine.resolve_ref("SLU_1", db, combo_results)
        self.assertAlmostEqual(N, -700e3)

    def test_resolve_components_single_factor(self):
        db = {"G": {"N": -500e3, "Mx": 20e6, "My": 5e6}}
        result = VerificationEngine.resolve_components(
            [{"ref": "G", "factor": 2.0}], db)
        self.assertAlmostEqual(result["N"],  -1000e3)
        self.assertAlmostEqual(result["Mx"],   40e6)
        self.assertAlmostEqual(result["My"],   10e6)

    def test_resolve_components_sum_two_refs(self):
        db = {
            "G":  {"N": -500e3, "Mx": 20e6, "My":  0.0},
            "Q1": {"N": -100e3, "Mx": 40e6, "My":  0.0},
        }
        result = VerificationEngine.resolve_components(
            [{"ref": "G",  "factor": 1.0},
             {"ref": "Q1", "factor": 1.5}], db)
        self.assertAlmostEqual(result["N"],  -500e3 - 150e3)
        self.assertAlmostEqual(result["Mx"],  20e6  + 60e6)

    def test_resolve_components_default_factor_1(self):
        db = {"G": {"N": -500e3, "Mx": 20e6, "My": 5e6}}
        result = VerificationEngine.resolve_components(
            [{"ref": "G"}], db)
        self.assertAlmostEqual(result["N"], -500e3)


# ==================================================================
#  VerificationEngine — simple demands and combinations
# ==================================================================

class TestVerificationEngineSimple(unittest.TestCase):
    """check_demand, check_demands, check_combination (simple)."""

    @classmethod
    def setUpClass(cls):
        sec = _biaxial_section()
        sv = FiberSolver(sec)
        nm_gen = NMDiagram(sv)
        nm_3d = nm_gen.generate_biaxial(n_angles=24, n_points_per_angle=80)
        flags = {
            "eta_3D": True, "eta_2D": False,
            "eta_path": True, "eta_path_2D": False,
        }
        cls.engine = VerificationEngine(nm_3d, nm_gen, flags, n_points=80)

    def test_check_demand_result_keys(self):
        d = {"name": "G", **_DEMAND_DB["G"]}
        result = self.engine.check_demand(d)
        for key in ("name", "N_kN", "Mx_kNm", "My_kNm", "inside", "verified"):
            self.assertIn(key, result)

    def test_check_demand_eta_3D_present(self):
        d = {"name": "G", **_DEMAND_DB["G"]}
        result = self.engine.check_demand(d)
        self.assertIn("eta_3D", result)

    def test_check_demand_eta_2D_absent(self):
        # eta_2D is disabled
        d = {"name": "G", **_DEMAND_DB["G"]}
        result = self.engine.check_demand(d)
        self.assertNotIn("eta_2D", result)

    def test_check_demand_small_is_inside(self):
        d = {"name": "G", **_DEMAND_DB["G"]}
        result = self.engine.check_demand(d)
        self.assertTrue(result["inside"])
        self.assertTrue(result["verified"])

    def test_check_demand_eta_positive(self):
        d = {"name": "G", **_DEMAND_DB["G"]}
        result = self.engine.check_demand(d)
        self.assertGreater(result["eta_3D"], 0.0)

    def test_check_demand_units_conversion(self):
        d = {"name": "G", **_DEMAND_DB["G"]}
        result = self.engine.check_demand(d)
        self.assertAlmostEqual(result["N_kN"],   -500.0, delta=0.01)
        self.assertAlmostEqual(result["Mx_kNm"],   20.0, delta=0.001)
        self.assertAlmostEqual(result["My_kNm"],    5.0, delta=0.001)

    def test_check_demands_batch_count(self):
        demands = [{"name": k, **v} for k, v in _DEMAND_DB.items()]
        results = self.engine.check_demands(demands)
        self.assertEqual(len(results), len(_DEMAND_DB))

    def test_check_demands_all_have_verified(self):
        demands = [{"name": k, **v} for k, v in _DEMAND_DB.items()]
        results = self.engine.check_demands(demands)
        for r in results:
            self.assertIn("verified", r)

    # --- simple combination ---

    def test_simple_combination_type(self):
        combo = {
            "name": "SLU_1",
            "components": [
                {"ref": "G", "factor": 1.3},
                {"ref": "Q1", "factor": 1.5},
            ],
        }
        result = self.engine.check_combination(combo, _DEMAND_DB)
        self.assertEqual(result["type"], "simple")

    def test_simple_combination_resultant_keys(self):
        combo = {
            "name": "SLU_1",
            "components": [{"ref": "G", "factor": 1.0}],
        }
        result = self.engine.check_combination(combo, _DEMAND_DB)
        self.assertIn("resultant", result)
        self.assertIn("N_kN",  result["resultant"])
        self.assertIn("Mx_kNm", result["resultant"])

    def test_simple_combination_factored_sum_correct(self):
        # G only, factor=1.0 → resultant == G
        combo = {
            "name": "G_only",
            "components": [{"ref": "G", "factor": 1.0}],
        }
        result = self.engine.check_combination(combo, _DEMAND_DB)
        self.assertAlmostEqual(
            result["resultant"]["N_kN"],
            _DEMAND_DB["G"]["N"] / 1e3,
            delta=0.01,
        )

    def test_simple_combination_has_eta_3D(self):
        combo = {
            "name": "SLU_1",
            "components": [
                {"ref": "G",  "factor": 1.3},
                {"ref": "Q1", "factor": 1.5},
            ],
        }
        result = self.engine.check_combination(combo, _DEMAND_DB)
        self.assertIn("eta_3D", result)

    def test_simple_combination_has_verified(self):
        combo = {
            "name": "SLU_1",
            "components": [{"ref": "G", "factor": 1.0}],
        }
        result = self.engine.check_combination(combo, _DEMAND_DB)
        self.assertIn("verified", result)


# ==================================================================
#  VerificationEngine — staged combinations
# ==================================================================

class TestVerificationEngineStaged(unittest.TestCase):
    """check_combination with staged loading sequences."""

    @classmethod
    def setUpClass(cls):
        sec = _biaxial_section()
        sv = FiberSolver(sec)
        nm_gen = NMDiagram(sv)
        nm_3d = nm_gen.generate_biaxial(n_angles=24, n_points_per_angle=80)
        flags = {
            "eta_3D": True, "eta_2D": False,
            "eta_path": True, "eta_path_2D": False,
        }
        cls.engine = VerificationEngine(nm_3d, nm_gen, flags, n_points=80)

    @staticmethod
    def _staged_combo():
        return {
            "name": "SLU_sismico",
            "stages": [
                {
                    "name": "gravitazionale",
                    "components": [
                        {"ref": "G",  "factor": 1.0},
                        {"ref": "Q1", "factor": 0.3},
                    ],
                },
                {
                    "name": "sisma",
                    "components": [{"ref": "Ex", "factor": 1.0}],
                },
            ],
        }

    def test_staged_type(self):
        result = self.engine.check_combination(
            self._staged_combo(), _DEMAND_DB)
        self.assertEqual(result["type"], "staged")

    def test_staged_has_stages_list(self):
        result = self.engine.check_combination(
            self._staged_combo(), _DEMAND_DB)
        self.assertIn("stages", result)
        self.assertEqual(len(result["stages"]), 2)

    def test_staged_stage0_has_eta_3D(self):
        result = self.engine.check_combination(
            self._staged_combo(), _DEMAND_DB)
        self.assertIn("eta_3D", result["stages"][0])

    def test_staged_stage1_has_eta_path(self):
        result = self.engine.check_combination(
            self._staged_combo(), _DEMAND_DB)
        self.assertIn("eta_path", result["stages"][1])

    def test_staged_stage1_has_base(self):
        result = self.engine.check_combination(
            self._staged_combo(), _DEMAND_DB)
        self.assertIn("base", result["stages"][1])

    def test_staged_cumulative_N_correct(self):
        # N_cum = G + Q1*0.3 + Ex
        N_expected = (
            _DEMAND_DB["G"]["N"]
            + _DEMAND_DB["Q1"]["N"] * 0.3
            + _DEMAND_DB["Ex"]["N"]
        ) / 1e3
        result = self.engine.check_combination(
            self._staged_combo(), _DEMAND_DB)
        self.assertAlmostEqual(
            result["resultant"]["N_kN"], N_expected, delta=0.01)

    def test_staged_has_eta_governing(self):
        result = self.engine.check_combination(
            self._staged_combo(), _DEMAND_DB)
        self.assertIn("eta_governing", result)
        self.assertGreater(result["eta_governing"], 0.0)

    def test_staged_has_verified(self):
        result = self.engine.check_combination(
            self._staged_combo(), _DEMAND_DB)
        self.assertIn("verified", result)

    def test_staged_stage_names_preserved(self):
        result = self.engine.check_combination(
            self._staged_combo(), _DEMAND_DB)
        names = [s["name"] for s in result["stages"]]
        self.assertIn("gravitazionale", names)
        self.assertIn("sisma", names)

    def test_staged_stage1_increment_equals_Ex(self):
        result = self.engine.check_combination(
            self._staged_combo(), _DEMAND_DB)
        inc = result["stages"][1]["increment"]
        self.assertAlmostEqual(
            inc["N_kN"], _DEMAND_DB["Ex"]["N"] / 1e3, delta=0.01)


# ==================================================================
#  VerificationEngine — envelopes
# ==================================================================

class TestVerificationEngineEnvelope(unittest.TestCase):
    """check_envelope: ref members, inline members, governing η."""

    @classmethod
    def setUpClass(cls):
        sec = _biaxial_section()
        sv = FiberSolver(sec)
        nm_gen = NMDiagram(sv)
        nm_3d = nm_gen.generate_biaxial(n_angles=24, n_points_per_angle=80)
        flags = {"eta_3D": True, "eta_2D": False}
        cls.engine = VerificationEngine(nm_3d, nm_gen, flags, n_points=80)

    def test_envelope_has_eta_max(self):
        env = {
            "name": "Env1",
            "members": [{"ref": "G"}, {"ref": "Q1"}],
        }
        result = self.engine.check_envelope(env, _DEMAND_DB)
        self.assertIn("eta_max", result)

    def test_envelope_eta_max_is_maximum_member_eta(self):
        env = {
            "name": "Env1",
            "members": [{"ref": "G"}, {"ref": "Q1"}],
        }
        result = self.engine.check_envelope(env, _DEMAND_DB)
        member_etas = [m["eta_3D"] for m in result["members"]]
        self.assertAlmostEqual(result["eta_max"], max(member_etas), places=4)

    def test_envelope_governing_member_is_correct(self):
        env = {
            "name": "Env1",
            "members": [{"ref": "G"}, {"ref": "Ex"}],
        }
        result = self.engine.check_envelope(env, _DEMAND_DB)
        self.assertIn(result["governing_member"], ["G", "Ex"])
        # The governing member is the one with the highest eta
        etas = {m["name"]: m["eta_3D"] for m in result["members"]}
        gov = result["governing_member"]
        self.assertEqual(etas[gov], result["eta_max"])

    def test_envelope_members_count(self):
        env = {
            "name": "Env2",
            "members": [{"ref": "G"}, {"ref": "Q1"}, {"ref": "Ex"}],
        }
        result = self.engine.check_envelope(env, _DEMAND_DB)
        self.assertEqual(len(result["members"]), 3)

    def test_envelope_inline_demand(self):
        env = {
            "name": "Env_inline",
            "members": [
                {"N_kN": -300.0, "Mx_kNm": 15.0, "My_kNm": 5.0,
                 "name": "inline_1"},
            ],
        }
        result = self.engine.check_envelope(env, _DEMAND_DB)
        self.assertEqual(len(result["members"]), 1)
        self.assertIn("eta_max", result)

    def test_envelope_verified_true_small_demands(self):
        env = {
            "name": "Env_small",
            "members": [
                {"N_kN": -100.0, "Mx_kNm": 5.0, "My_kNm": 2.0,
                 "name": "s1"},
                {"N_kN": -150.0, "Mx_kNm": 8.0, "My_kNm": 3.0,
                 "name": "s2"},
            ],
        }
        result = self.engine.check_envelope(env, _DEMAND_DB)
        self.assertTrue(result["verified"])
        self.assertLessEqual(result["eta_max"], 1.0)

    def test_envelope_verified_false_large_demands(self):
        env = {
            "name": "Env_large",
            "members": [
                {"N_kN": -400.0, "Mx_kNm": 600.0, "My_kNm": 300.0,
                 "name": "big"},
            ],
        }
        result = self.engine.check_envelope(env, _DEMAND_DB)
        self.assertFalse(result["verified"])
        self.assertGreater(result["eta_max"], 1.0)

    def test_envelope_with_ref_and_inline_mixed(self):
        env = {
            "name": "Env_mixed",
            "members": [
                {"ref": "G"},
                {"N_kN": -200.0, "Mx_kNm": 10.0, "My_kNm": 3.0,
                 "name": "inline"},
            ],
        }
        result = self.engine.check_envelope(env, _DEMAND_DB)
        self.assertEqual(len(result["members"]), 2)


# ==================================================================
#  VerificationEngine — output flags and contour cache
# ==================================================================

class TestVerificationEngineFlags(unittest.TestCase):
    """Output flags control which η types are computed; contour caching."""

    @classmethod
    def setUpClass(cls):
        sec = _biaxial_section()
        sv = FiberSolver(sec)
        cls.nm_gen = NMDiagram(sv)
        cls.nm_3d = cls.nm_gen.generate_biaxial(n_angles=24, n_points_per_angle=80)

    def _engine(self, flags):
        return VerificationEngine(self.nm_3d, self.nm_gen, flags, n_points=60)

    def _demand(self):
        return {"name": "D", "N": -300e3, "Mx": 20e6, "My": 8e6}

    def test_eta_3D_only(self):
        eng = self._engine({"eta_3D": True, "eta_2D": False})
        result = eng.check_demand(self._demand())
        self.assertIn("eta_3D", result)
        self.assertNotIn("eta_2D", result)

    def test_eta_2D_enabled(self):
        eng = self._engine({"eta_3D": True, "eta_2D": True,
                             "n_angles_mx_my": 24})
        result = eng.check_demand(self._demand())
        self.assertIn("eta_2D", result)

    def test_both_disabled_falls_back_to_is_inside(self):
        eng = self._engine({"eta_3D": False, "eta_2D": False})
        result = eng.check_demand(self._demand())
        self.assertNotIn("eta_3D", result)
        self.assertNotIn("eta_2D", result)
        self.assertIn("inside", result)
        # verified == inside when no η enabled
        self.assertEqual(result["verified"], result["inside"])

    def test_default_flags_eta_3D_true(self):
        eng = VerificationEngine(self.nm_3d, self.nm_gen, {}, n_points=60)
        self.assertTrue(eng.do_3D)

    def test_default_flags_eta_2D_false(self):
        eng = VerificationEngine(self.nm_3d, self.nm_gen, {}, n_points=60)
        self.assertFalse(eng.do_2D)

    def test_delta_N_tol_stored(self):
        eng = self._engine({"eta_path_2D": True, "delta_N_tol": 0.05})
        self.assertAlmostEqual(eng.delta_N_tol, 0.05)

    def test_contour_cache_empty_initially(self):
        eng = self._engine({"eta_3D": True, "eta_2D": True,
                             "n_angles_mx_my": 24})
        self.assertEqual(len(eng._contour_cache), 0)

    def test_contour_cache_populated_after_check(self):
        eng = self._engine({"eta_3D": True, "eta_2D": True,
                             "n_angles_mx_my": 24})
        eng.check_demand(self._demand())
        self.assertEqual(len(eng._contour_cache), 1)

    def test_contour_cache_reused_for_same_N(self):
        eng = self._engine({"eta_3D": True, "eta_2D": True,
                             "n_angles_mx_my": 24})
        d1 = {"name": "D1", "N": -300e3, "Mx": 20e6, "My": 8e6}
        d2 = {"name": "D2", "N": -300e3, "Mx": 10e6, "My": 4e6}
        eng.check_demands([d1, d2])
        # Both demands at N≈-300 kN → only 1 contour generated
        self.assertEqual(len(eng._contour_cache), 1)

    def test_eta_path_2D_skipped_on_large_delta_N(self):
        """eta_path_2D should be None when ΔN/N_range > delta_N_tol."""
        eng = self._engine({"eta_3D": True, "eta_path": True,
                             "eta_path_2D": True, "delta_N_tol": 0.01,
                             "n_angles_mx_my": 24})
        combo = {
            "name": "big_jump",
            "stages": [
                {
                    "name": "s0",
                    "components": [{"ref": "G", "factor": 1.0}],
                },
                {
                    "name": "s1",
                    # Large ΔN: 2000 kN jump
                    "components": [
                        {"ref": "G", "factor": -3.0},
                    ],
                },
            ],
        }
        result = eng.check_combination(combo, _DEMAND_DB)
        stage1 = result["stages"][1]
        # eta_path_2D should be None (skipped) or have a warning
        if "eta_path_2D" in stage1:
            self.assertIsNone(stage1["eta_path_2D"])


if __name__ == "__main__":
    unittest.main()
