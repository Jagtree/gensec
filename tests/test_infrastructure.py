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
Unit tests for YAML loader, export functions, and domain checker.

Updated for v0.2.0: uses DomainChecker (replaces legacy DemandChecker),
v2.1 demand result format (Mx_kNm/My_kNm/eta_3D), and new YAML structure
(combinations with components/stages, envelopes with members).
"""

import unittest
import json
import csv
import os
import tempfile
import numpy as np
from pathlib import Path

from gensec.materials import Concrete, Steel
from gensec.geometry import RebarLayer, RectSection
from gensec.solver import FiberSolver, NMDiagram, DomainChecker
from gensec.io_yaml import load_yaml
from gensec.output import (
    export_nm_domain_csv, export_nm_domain_json,
    export_demand_results_csv, export_demand_results_json,
    export_fiber_results_csv,
)

_EXAMPLE_YAML = str(Path(__file__).parent.parent / "examples" / "example_input.yaml")
_V21_YAML = str(Path(__file__).parent.parent / "examples" / "example_v2_1.yaml")
_BIAXIAL_YAML = str(Path(__file__).parent.parent / "examples" / "biaxial_column.yaml")


# ==================================================================
#  YAML loader — basic section, demands, materials
# ==================================================================

class TestYAMLLoader(unittest.TestCase):
    """YAML loading: section geometry, demands, materials."""

    def test_loads_section(self):
        data = load_yaml(_EXAMPLE_YAML)
        self.assertEqual(data["section"].B, 300)
        self.assertEqual(data["section"].H, 600)

    def test_loads_demands(self):
        data = load_yaml(_EXAMPLE_YAML)
        self.assertEqual(len(data["demands"]), 2)
        self.assertEqual(data["demands"][0]["name"], "Gravity")

    def test_demand_has_Mx_My_keys(self):
        data = load_yaml(_EXAMPLE_YAML)
        for d in data["demands"]:
            self.assertIn("Mx", d)
            self.assertIn("My", d)
            self.assertIn("N", d)

    def test_demand_units_N_and_Nmm(self):
        # N in [N], Mx/My in [N·mm]
        data = load_yaml(_EXAMPLE_YAML)
        d = data["demands"][0]  # Gravity: N_kN=-1500, Mx_kNm=200
        self.assertAlmostEqual(d["N"],  -1500e3, delta=1)
        self.assertAlmostEqual(d["Mx"],  200e6,  delta=1)

    def test_loads_materials(self):
        data = load_yaml(_EXAMPLE_YAML)
        self.assertIn("concrete_1", data["materials"])
        self.assertIsInstance(data["materials"]["concrete_1"], Concrete)

    def test_loads_steel_material(self):
        data = load_yaml(_EXAMPLE_YAML)
        self.assertIn("steel_1", data["materials"])
        self.assertIsInstance(data["materials"]["steel_1"], Steel)

    def test_rebars_count(self):
        data = load_yaml(_EXAMPLE_YAML)
        self.assertEqual(len(data["section"].rebars), 3)

    def test_loads_envelopes(self):
        data = load_yaml(_EXAMPLE_YAML)
        self.assertIn("envelopes", data)
        self.assertGreater(len(data["envelopes"]), 0)

    def test_envelope_has_members(self):
        data = load_yaml(_EXAMPLE_YAML)
        for env in data["envelopes"]:
            self.assertIn("members", env)
            self.assertIn("name", env)

    def test_output_options_key(self):
        data = load_yaml(_EXAMPLE_YAML)
        self.assertIn("output_options", data)

    def test_output_options_defaults(self):
        data = load_yaml(_EXAMPLE_YAML)
        opts = data["output_options"]
        # example_input.yaml sets eta_3D: true
        self.assertTrue(opts["eta_3D"])


# ==================================================================
#  YAML loader — v2.1 combinations (components/stages)
# ==================================================================

class TestYAMLCombinationsV21(unittest.TestCase):
    """YAML loading: v2.1 combination and envelope structure."""

    @classmethod
    def setUpClass(cls):
        cls.data = load_yaml(_V21_YAML)

    def test_loads_combinations(self):
        combos = self.data["combinations"]
        self.assertEqual(len(combos), 3)

    def test_combination_names(self):
        names = {c["name"] for c in self.data["combinations"]}
        self.assertIn("SLU_1", names)
        self.assertIn("SLU_sismico_Ex", names)
        self.assertIn("SLU_sismico_Ey", names)

    def test_simple_combination_has_components(self):
        slu1 = next(c for c in self.data["combinations"]
                    if c["name"] == "SLU_1")
        self.assertIn("components", slu1)
        self.assertNotIn("stages", slu1)

    def test_simple_combination_components_have_ref_and_factor(self):
        slu1 = next(c for c in self.data["combinations"]
                    if c["name"] == "SLU_1")
        for comp in slu1["components"]:
            self.assertIn("ref", comp)
            self.assertIn("factor", comp)

    def test_staged_combination_has_stages(self):
        staged = next(c for c in self.data["combinations"]
                      if c["name"] == "SLU_sismico_Ex")
        self.assertIn("stages", staged)
        self.assertNotIn("components", staged)

    def test_staged_combination_stage_count(self):
        staged = next(c for c in self.data["combinations"]
                      if c["name"] == "SLU_sismico_Ex")
        self.assertEqual(len(staged["stages"]), 2)

    def test_staged_stage_has_name_and_components(self):
        staged = next(c for c in self.data["combinations"]
                      if c["name"] == "SLU_sismico_Ex")
        for stage in staged["stages"]:
            self.assertIn("name", stage)
            self.assertIn("components", stage)

    def test_loads_envelopes(self):
        envs = self.data["envelopes"]
        self.assertGreater(len(envs), 0)

    def test_envelope_members_ref_format(self):
        # Envelope_SLU references combinations
        env = next(e for e in self.data["envelopes"]
                   if e["name"] == "Envelope_SLU")
        for m in env["members"]:
            self.assertIn("ref", m)

    def test_envelope_members_inline_format(self):
        # Envelope_diretto has inline demands
        env = next(e for e in self.data["envelopes"]
                   if e["name"] == "Envelope_diretto")
        inline = [m for m in env["members"] if "ref" not in m]
        self.assertGreater(len(inline), 0)

    def test_output_flags_all_types(self):
        opts = self.data["output_options"]
        self.assertTrue(opts["eta_3D"])
        self.assertTrue(opts["eta_2D"])
        self.assertTrue(opts["eta_path"])
        self.assertTrue(opts["eta_path_2D"])

    def test_delta_N_tol_parsed(self):
        opts = self.data["output_options"]
        self.assertAlmostEqual(opts["delta_N_tol"], 0.03)


# ==================================================================
#  YAML loader — EC2 concrete material type
# ==================================================================

class TestYAMLConcreteEC2(unittest.TestCase):
    """Test concrete_ec2 material type in YAML."""

    def test_loads_ec2_class(self):
        data = load_yaml(_BIAXIAL_YAML)
        c = data["materials"]["C30"]
        self.assertAlmostEqual(c.fcd, 0.85 * 30 / 1.5, places=2)
        self.assertEqual(c.eps_cu2, -0.0035)
        self.assertEqual(c.n_parabola, 2.0)
        self.assertTrue(hasattr(c, "ec2"))
        self.assertAlmostEqual(c.ec2.fcm, 38.0)


# ==================================================================
#  Export functions
# ==================================================================

class TestExport(unittest.TestCase):
    """Numerical export: CSV and JSON."""

    def setUp(self):
        concrete = Concrete(fck=25.0)
        steel = Steel(fyk=450.0)
        rb = [RebarLayer(y=40, As=942.5, material=steel),
              RebarLayer(y=560, As=942.5, material=steel)]
        sec = RectSection(B=300, H=600, bulk_material=concrete,
                          rebars=rb, n_fibers_y=100, n_fibers_x=1)
        self.sv = FiberSolver(sec)
        self.nm = NMDiagram(self.sv).generate(200)
        self.td = tempfile.mkdtemp()

    def test_nm_csv_header(self):
        p = os.path.join(self.td, "nm.csv")
        export_nm_domain_csv(self.nm, p)
        with open(p) as f:
            reader = csv.reader(f)
            header = next(reader)
        self.assertEqual(header, ["N_kN", "Mx_kNm"])

    def test_nm_csv_row_count(self):
        p = os.path.join(self.td, "nm.csv")
        export_nm_domain_csv(self.nm, p)
        with open(p) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)
        self.assertEqual(len(rows), len(self.nm["N"]))

    def test_nm_json_n_points(self):
        p = os.path.join(self.td, "nm.json")
        export_nm_domain_json(self.nm, p)
        with open(p) as f:
            data = json.load(f)
        self.assertEqual(data["n_points"], len(self.nm["N"]))

    def test_nm_json_arrays_present(self):
        p = os.path.join(self.td, "nm.json")
        export_nm_domain_json(self.nm, p)
        with open(p) as f:
            data = json.load(f)
        self.assertIn("N_kN", data)
        self.assertIn("Mx_kNm", data)

    def test_fiber_csv_nonempty(self):
        fr = self.sv.get_fiber_results(-0.001, 5e-6)
        p = os.path.join(self.td, "fibers.csv")
        export_fiber_results_csv(fr, p)
        self.assertGreater(os.path.getsize(p), 100)

    def test_demand_csv_row_count(self):
        # v2.1 result format: Mx_kNm, My_kNm, eta_3D
        results = [
            {"name": "A", "N_kN": -1500.0, "Mx_kNm": 200.0, "My_kNm": 0.0,
             "eta_3D": 0.7, "inside": True, "verified": True},
        ]
        p = os.path.join(self.td, "dem.csv")
        export_demand_results_csv(results, p)
        with open(p) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)
        self.assertEqual(len(rows), 1)

    def test_demand_csv_header_contains_eta(self):
        results = [
            {"name": "A", "N_kN": -1500.0, "Mx_kNm": 200.0, "My_kNm": 0.0,
             "eta_3D": 0.7, "inside": True, "verified": True},
        ]
        p = os.path.join(self.td, "dem.csv")
        export_demand_results_csv(results, p)
        with open(p) as f:
            reader = csv.reader(f)
            header = next(reader)
        self.assertIn("eta_3D", header)
        self.assertIn("Mx_kNm", header)

    def test_demand_json_demands_array(self):
        results = [
            {"name": "A", "N_kN": -1500.0, "Mx_kNm": 200.0, "My_kNm": 0.0,
             "eta_3D": 0.7, "inside": True, "verified": True},
        ]
        p = os.path.join(self.td, "dem.json")
        export_demand_results_json(results, p)
        with open(p) as f:
            data = json.load(f)
        self.assertIn("demands", data)
        self.assertEqual(len(data["demands"]), 1)

    def test_demand_json_keys_v21(self):
        results = [
            {"name": "B", "N_kN": -800.0, "Mx_kNm": 100.0, "My_kNm": 30.0,
             "eta_3D": 0.55, "inside": True, "verified": True},
        ]
        p = os.path.join(self.td, "dem2.json")
        export_demand_results_json(results, p)
        with open(p) as f:
            data = json.load(f)
        d = data["demands"][0]
        self.assertIn("Mx_kNm", d)
        self.assertIn("My_kNm", d)
        self.assertIn("verified", d)


# ==================================================================
#  DomainChecker 2D (uniaxial N-M domain)
# ==================================================================

class TestDomainChecker2D(unittest.TestCase):
    """DomainChecker operating on a uniaxial N-M domain (ndim=2)."""

    @classmethod
    def setUpClass(cls):
        concrete = Concrete(fck=25.0)
        steel = Steel(fyk=450.0)
        rb = [RebarLayer(y=40, As=942.5, material=steel),
              RebarLayer(y=560, As=942.5, material=steel)]
        sec = RectSection(B=300, H=600, bulk_material=concrete,
                          rebars=rb, n_fibers_y=200, n_fibers_x=1)
        nm = NMDiagram(FiberSolver(sec)).generate(500)
        cls.checker = DomainChecker(nm)

    def test_ndim_is_2(self):
        self.assertEqual(self.checker.ndim, 2)

    def test_N_range_positive(self):
        self.assertGreater(self.checker.N_range, 0)

    def test_origin_inside(self):
        self.assertTrue(self.checker.is_inside(0.0, 0.0))

    def test_small_demand_inside(self):
        self.assertTrue(self.checker.is_inside(-500e3, 50e6))

    def test_outside_compression(self):
        self.assertFalse(self.checker.is_inside(-5000e3, 0))

    def test_outside_moment(self):
        self.assertFalse(self.checker.is_inside(-1500e3, 500e6))

    def test_eta_3D_inside_less_than_1(self):
        eta = self.checker.eta_3D(-1500e3, 200e6)
        self.assertGreater(eta, 0.0)
        self.assertLess(eta, 1.0)

    def test_eta_3D_outside_greater_than_1(self):
        eta = self.checker.eta_3D(-1500e3, 500e6)
        self.assertGreater(eta, 1.0)

    def test_eta_3D_origin_is_zero(self):
        eta = self.checker.eta_3D(0.0, 0.0)
        self.assertEqual(eta, 0.0)

    def test_is_inside_and_eta_consistent(self):
        # If eta < 1 → is_inside should be True
        N, Mx = -500e3, 50e6
        eta = self.checker.eta_3D(N, Mx)
        inside = self.checker.is_inside(N, Mx)
        self.assertLess(eta, 1.0)
        self.assertTrue(inside)


# ==================================================================
#  Section plotting
# ==================================================================

class TestSectionPlot(unittest.TestCase):
    """Section drawing: geometry only and with stress field."""

    def test_geometry_only(self):
        import matplotlib
        matplotlib.use("Agg")
        from gensec.output import plot_section
        concrete = Concrete(fck=25.0)
        steel = Steel(fyk=450.0)
        sec = RectSection(B=300, H=600, bulk_material=concrete,
                          rebars=[
                              RebarLayer(y=40, As=942, material=steel,
                                         diameter=20),
                              RebarLayer(y=560, As=942, material=steel,
                                         diameter=20),
                          ], n_fibers_y=50, n_fibers_x=1)
        fig = plot_section(sec)
        self.assertIsNotNone(fig)

    def test_with_stress_field(self):
        import matplotlib
        matplotlib.use("Agg")
        from gensec.output import plot_section
        concrete = Concrete(fck=25.0)
        steel = Steel(fyk=450.0)
        A20 = np.pi / 4 * 20 ** 2
        sec = RectSection(B=300, H=600, bulk_material=concrete,
                          rebars=[
                              RebarLayer(y=40, x=40, As=A20, material=steel,
                                         diameter=20),
                              RebarLayer(y=560, x=260, As=A20, material=steel,
                                         diameter=20),
                          ], n_fibers_y=30, n_fibers_x=15)
        sv = FiberSolver(sec)
        fr = sv.get_fiber_results(-0.001, 5e-6, 2e-6)
        fig = plot_section(sec, fr)
        self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main()
