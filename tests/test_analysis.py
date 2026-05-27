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
Tests for :mod:`gensec.solver.analysis` — force decomposition and
on-demand :math:`\eta`.
"""

import unittest
import numpy as np


def _make_rc_section():
    r"""
    300×600 mm RC section with 3Φ20 top + 3Φ20 bottom.

    Returns ``(section, solver)`` ready for analysis.
    """
    from gensec.materials import Concrete, Steel
    from gensec.geometry import RebarLayer, RectSection
    from gensec.solver import FiberSolver

    conc = Concrete(fck=25.0)
    conc.name = "C25/30"
    steel = Steel(fyk=450.0)
    steel.name = "B450C"

    rebars = [
        RebarLayer(y=50.0,  As=942.5, material=steel),
        RebarLayer(y=550.0, As=942.5, material=steel),
    ]
    sec = RectSection(B=300, H=600, bulk_material=conc,
                      rebars=rebars, n_fibers_y=60)
    solver = FiberSolver(sec)
    return sec, solver


# ==================================================================
#  Force decomposition tests
# ==================================================================

class TestDecomposeForces(unittest.TestCase):
    r"""Verify that force decomposition sums match integration."""

    @classmethod
    def setUpClass(cls):
        cls.sec, cls.solver = _make_rc_section()
        from gensec.solver.analysis import AnalysisEngine
        cls.engine = AnalysisEngine(cls.solver)

    def test_decompose_sums_match_total(self):
        """Component forces must sum to the total."""
        result = self.engine.analyze(
            N=-1500e3, Mx=200e6, My=0.0)
        self.assertTrue(result["converged"])

        total = result["total"]
        comp_N = sum(c["N"] for c in result["components"])
        comp_Mx = sum(c["Mx"] for c in result["components"])
        comp_My = sum(c["My"] for c in result["components"])

        self.assertAlmostEqual(total["N"], comp_N, delta=10)
        self.assertAlmostEqual(total["Mx"], comp_Mx, delta=1000)
        self.assertAlmostEqual(total["My"], comp_My, delta=1000)

    def test_decompose_matches_integrate(self):
        """Total must match direct integration."""
        result = self.engine.analyze(
            N=-1500e3, Mx=200e6, My=0.0)
        ss = result["strain_state"]
        N_int, Mx_int, My_int = self.solver.integrate(
            ss["eps0"], ss["chi_x"], ss["chi_y"])

        self.assertAlmostEqual(result["total"]["N"], N_int,
                               delta=10)
        self.assertAlmostEqual(result["total"]["Mx"], Mx_int,
                               delta=1000)

    def test_component_types(self):
        """Must have at least one bulk and one rebar component."""
        result = self.engine.analyze(N=-1500e3, Mx=200e6)
        types = {c["type"] for c in result["components"]}
        self.assertIn("bulk", types)
        self.assertIn("rebar", types)

    def test_material_names_populated(self):
        """Material names must come from the .name attribute."""
        result = self.engine.analyze(N=-1500e3, Mx=200e6)
        for comp in result["components"]:
            self.assertIn("material_name", comp)
            self.assertNotEqual(comp["material_name"], "")

        bulk_names = {c["material_name"]
                      for c in result["components"]
                      if c["type"] == "bulk"}
        self.assertIn("C25/30", bulk_names)

        rebar_names = {c["material_name"]
                       for c in result["components"]
                       if c["type"] == "rebar"}
        self.assertIn("B450C", rebar_names)

    def test_rebar_layers_detail(self):
        """Rebar components must include per-bar detail."""
        result = self.engine.analyze(N=-1500e3, Mx=200e6)
        rebar_comps = [c for c in result["components"]
                       if c["type"] == "rebar"]
        self.assertGreater(len(rebar_comps), 0)

        for rc in rebar_comps:
            self.assertIn("layers", rc)
            for lay in rc["layers"]:
                for key in ("index", "x", "y", "A", "eps",
                            "sigma_net", "F_net_kN"):
                    self.assertIn(key, lay)

    def test_pure_axial_compression(self):
        """Pure compression: bulk must be in compression (negative N)."""
        result = self.engine.analyze(N=-3000e3, Mx=0.0)
        self.assertTrue(result["converged"])
        bulk = [c for c in result["components"]
                if c["type"] == "bulk"]
        self.assertLess(bulk[0]["N"], 0)

    def test_unconverged_returns_empty_components(self):
        """Extreme demand should fail; components list is empty."""
        result = self.engine.analyze(N=1e12, Mx=1e15)
        self.assertFalse(result["converged"])
        self.assertEqual(len(result["components"]), 0)

    def test_strains_ok_flag_inside(self):
        """A moderate demand should have strains_ok = True."""
        result = self.engine.analyze(N=-500e3, Mx=50e6)
        self.assertTrue(result["converged"])
        self.assertTrue(result["strains_ok"])


# ==================================================================
#  strains_within_limits tests
# ==================================================================

class TestStrainsWithinLimits(unittest.TestCase):
    """Verify FiberSolver.strains_within_limits."""

    @classmethod
    def setUpClass(cls):
        cls.sec, cls.solver = _make_rc_section()

    def test_zero_strain_within_limits(self):
        """Zero strain must be within all limits."""
        self.assertTrue(
            self.solver.strains_within_limits(0.0, 0.0, 0.0))

    def test_extreme_strain_outside_limits(self):
        """Extreme curvature pushes fibres beyond eps_cu2."""
        self.assertFalse(
            self.solver.strains_within_limits(-0.01, 0.0001))

    def test_moderate_strain_within_limits(self):
        """Moderate strain plane should be OK."""
        self.assertTrue(
            self.solver.strains_within_limits(-0.001, 1e-6))


# ==================================================================
#  On-demand η tests
# ==================================================================

class TestOnDemandEta(unittest.TestCase):
    r"""Verify AnalysisEngine.compute_eta against known properties."""

    @classmethod
    def setUpClass(cls):
        cls.sec, cls.solver = _make_rc_section()
        from gensec.solver.analysis import AnalysisEngine
        cls.engine = AnalysisEngine(cls.solver)

    def test_origin_eta_zero(self):
        """η at origin (zero demand) should be 0."""
        r = self.engine.compute_eta(0.0, 0.0, 0.0)
        self.assertAlmostEqual(r["eta"], 0.0, places=4)

    def test_inside_demand_eta_less_than_one(self):
        """A moderate demand should have η < 1."""
        r = self.engine.compute_eta(N=-500e3, Mx=50e6)
        self.assertLess(r["eta"], 1.0)
        self.assertTrue(r["demand_inside"])

    def test_outside_demand_eta_greater_than_one(self):
        """A very large demand should have η > 1."""
        r = self.engine.compute_eta(N=-500e3, Mx=2000e6)
        self.assertGreater(r["eta"], 1.0)
        self.assertFalse(r["demand_inside"])

    def test_eta_monotone_along_ray(self):
        r"""η must increase monotonically along a ray from origin."""
        etas = []
        for scale in [0.2, 0.5, 0.8, 1.0, 1.5]:
            r = self.engine.compute_eta(
                N=-500e3 * scale, Mx=100e6 * scale)
            etas.append(r["eta"])
        for i in range(1, len(etas)):
            self.assertGreaterEqual(etas[i], etas[i - 1] - 1e-6)

    def test_eta_from_staged_base(self):
        """η from a non-zero base point should be computable."""
        r = self.engine.compute_eta(
            N=-1200e3, Mx=180e6,
            base_N=-500e3, base_Mx=50e6)
        self.assertTrue(r["converged"])
        self.assertGreater(r["eta"], 0.0)


# ==================================================================
#  Batch analysis tests
# ==================================================================

class TestAnalyzeDemandsBatch(unittest.TestCase):
    """Verify analyze_demands batch processing."""

    @classmethod
    def setUpClass(cls):
        cls.sec, cls.solver = _make_rc_section()
        from gensec.solver.analysis import AnalysisEngine
        cls.engine = AnalysisEngine(cls.solver)

    def test_batch_returns_correct_count(self):
        demands = [
            {"name": "G", "N": -1500e3, "Mx": 200e6, "My": 0.0},
            {"name": "Q", "N": -800e3, "Mx": 100e6, "My": 0.0},
        ]
        results = self.engine.analyze_demands(demands)
        self.assertEqual(len(results), 2)

    def test_batch_names_preserved(self):
        demands = [
            {"name": "Gravity", "N": -1500e3, "Mx": 200e6,
             "My": 0.0},
        ]
        results = self.engine.analyze_demands(demands)
        self.assertEqual(results[0]["name"], "Gravity")

    def test_batch_has_kn_units(self):
        demands = [
            {"name": "G", "N": -1500e3, "Mx": 200e6, "My": 0.0},
        ]
        results = self.engine.analyze_demands(demands)
        r = results[0]
        self.assertAlmostEqual(r["N_kN"], -1500.0, places=0)
        self.assertAlmostEqual(r["Mx_kNm"], 200.0, places=0)


# ==================================================================
#  Combination analysis tests
# ==================================================================

class TestAnalyzeCombinations(unittest.TestCase):
    """Verify analyze_combinations for simple and staged."""

    @classmethod
    def setUpClass(cls):
        cls.sec, cls.solver = _make_rc_section()
        from gensec.solver.analysis import AnalysisEngine
        cls.engine = AnalysisEngine(cls.solver)
        cls.demand_db = {
            "G":  {"N": -1000e3, "Mx": 100e6, "My": 0.0},
            "Q1": {"N": -300e3,  "Mx": 80e6,  "My": 0.0},
            "Ex": {"N": -100e3,  "Mx": 50e6,  "My": 30e6},
        }

    def test_simple_combination(self):
        combo = {
            "name": "SLU_1",
            "components": [
                {"ref": "G", "factor": 1.3},
                {"ref": "Q1", "factor": 1.5},
            ],
        }
        results = self.engine.analyze_combinations(
            [combo], self.demand_db)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["name"], "SLU_1")
        self.assertTrue(r["converged"])
        # 1.3*(-1000) + 1.5*(-300) = -1750 kN
        self.assertAlmostEqual(r["N_kN"], -1750.0, places=0)

    def test_staged_combination(self):
        combo = {
            "name": "SLU_sisma",
            "stages": [
                {"name": "grav",
                 "components": [{"ref": "G", "factor": 1.0}]},
                {"name": "sisma",
                 "components": [{"ref": "Ex", "factor": 1.0}]},
            ],
        }
        results = self.engine.analyze_combinations(
            [combo], self.demand_db)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["type"], "staged")
        self.assertEqual(len(r["stages"]), 2)
        for stage in r["stages"]:
            self.assertIn("components", stage)

    def test_missing_ref_raises(self):
        combo = {
            "name": "bad",
            "components": [{"ref": "NONEXISTENT", "factor": 1.0}],
        }
        with self.assertRaises(KeyError):
            self.engine.analyze_combinations(
                [combo], self.demand_db)


if __name__ == "__main__":
    unittest.main()
