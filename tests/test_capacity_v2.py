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
Tests for v0.2.0 capacity additions.

Covers:
- generate_moment_curvature: cracking detection, key-point ordering,
  array consistency, plain concrete fallback (no cracking)
- generate_biaxial: 3D surface properties
- generate_mx_my: Mx-My contour properties
- New export functions: moment_curvature, mx_my, 3d_surface,
  combination_results, envelope_results
"""

import unittest
import json
import csv
import os
import tempfile
import numpy as np
from pathlib import Path

from gensec.materials import Concrete, Steel, concrete_from_class
from gensec.geometry import RebarLayer, RectSection
from gensec.solver import FiberSolver, NMDiagram
from gensec.output import (
    export_moment_curvature_json, export_moment_curvature_csv,
    export_mx_my_json, export_mx_my_csv,
    export_3d_surface_csv, export_3d_surface_json,
    export_combination_results_json, export_envelope_results_json,
)


# ------------------------------------------------------------------
#  Fixture factories
# ------------------------------------------------------------------

def _ec2_section():
    """
    300×500 column with EC2 C25/30 concrete.
    EC2 concrete provides fctm/Ecm → cracking detection is active.
    """
    concrete = concrete_from_class("C25/30", ls="F", loadtype="slow",
                                   TypeConc="R")
    steel = Steel(fyk=450.0)
    As = np.pi / 4 * 16 ** 2
    rb = [
        RebarLayer(y=45,  As=3 * As, material=steel),
        RebarLayer(y=455, As=3 * As, material=steel),
    ]
    return RectSection(B=300, H=500, bulk_material=concrete,
                       rebars=rb, n_fibers_y=100, n_fibers_x=1)


def _plain_section():
    """
    300×500 with basic Concrete (no .ec2 attribute) → no cracking detection.
    """
    concrete = Concrete(fck=25.0)
    steel = Steel(fyk=450.0)
    As = np.pi / 4 * 16 ** 2
    rb = [
        RebarLayer(y=45,  As=As, material=steel),
        RebarLayer(y=455, As=As, material=steel),
    ]
    return RectSection(B=300, H=500, bulk_material=concrete,
                       rebars=rb, n_fibers_y=60, n_fibers_x=1)


def _biaxial_section():
    """300×500 with 4 corner bars — biaxial fixture."""
    concrete = Concrete(fck=30.0)
    steel = Steel(fyk=450.0)
    As = np.pi / 4 * 20 ** 2
    rb = [
        RebarLayer(y=45,  x=45,  As=As, material=steel),
        RebarLayer(y=45,  x=255, As=As, material=steel),
        RebarLayer(y=455, x=45,  As=As, material=steel),
        RebarLayer(y=455, x=255, As=As, material=steel),
    ]
    return RectSection(B=300, H=500, bulk_material=concrete,
                       rebars=rb, n_fibers_y=30, n_fibers_x=15)


# ==================================================================
#  Moment-curvature: basic output structure
# ==================================================================

class TestMomentCurvatureOutput(unittest.TestCase):
    """generate_moment_curvature: output keys and array consistency."""

    @classmethod
    def setUpClass(cls):
        sec = _plain_section()
        sv = FiberSolver(sec)
        cls.mc = NMDiagram(sv).generate_moment_curvature(
            N_fixed=-100e3, n_points=60)

    def test_has_chi_array(self):
        self.assertIn("chi", self.mc)
        self.assertGreater(len(self.mc["chi"]), 0)

    def test_has_M_array(self):
        self.assertIn("M", self.mc)
        self.assertGreater(len(self.mc["M"]), 0)

    def test_has_M_kNm(self):
        self.assertIn("M_kNm", self.mc)

    def test_has_chi_km(self):
        self.assertIn("chi_km", self.mc)

    def test_chi_and_M_same_length(self):
        self.assertEqual(len(self.mc["chi"]), len(self.mc["M"]))

    def test_M_kNm_consistent_with_M(self):
        np.testing.assert_allclose(
            self.mc["M_kNm"], self.mc["M"] / 1e6, rtol=1e-5)

    def test_chi_km_consistent_with_chi(self):
        np.testing.assert_allclose(
            self.mc["chi_km"], self.mc["chi"] * 1e6, rtol=1e-5)

    def test_N_fixed_kN_stored(self):
        self.assertIn("N_fixed_kN", self.mc)
        self.assertAlmostEqual(self.mc["N_fixed_kN"], -100.0, delta=0.01)

    def test_direction_stored(self):
        self.assertIn("direction", self.mc)
        self.assertEqual(self.mc["direction"], "x")

    def test_direction_y(self):
        sec = _plain_section()
        mc_y = NMDiagram(FiberSolver(sec)).generate_moment_curvature(
            N_fixed=0.0, n_points=40, direction="y")
        self.assertEqual(mc_y["direction"], "y")

    def test_cracking_keys_present(self):
        for key in ("cracking_chi_pos", "cracking_M_pos",
                    "cracking_chi_neg", "cracking_M_neg"):
            self.assertIn(key, self.mc)

    def test_yield_keys_present(self):
        for key in ("yield_chi_pos", "yield_M_pos",
                    "yield_chi_neg", "yield_M_neg"):
            self.assertIn(key, self.mc)

    def test_ultimate_keys_present(self):
        for key in ("ultimate_chi_pos", "ultimate_M_pos",
                    "ultimate_chi_neg", "ultimate_M_neg"):
            self.assertIn(key, self.mc)

    def test_eps_min_max_present(self):
        self.assertIn("eps_min", self.mc)
        self.assertIn("eps_max", self.mc)

    def test_M_antisymmetric_at_N0_symmetric_section(self):
        """At N=0 on a doubly symmetric section M(chi) = -M(-chi)."""
        sec = _plain_section()
        mc = NMDiagram(FiberSolver(sec)).generate_moment_curvature(
            N_fixed=0.0, n_points=60)
        chi = mc["chi"]
        M   = mc["M"]
        # The curve is assembled as [neg_reversed, pos[1:]]
        # At chi=0 (midpoint) M should be near zero
        mid = len(chi) // 2
        self.assertAlmostEqual(M[mid], 0.0, delta=abs(M).max() * 0.01)


# ==================================================================
#  Moment-curvature: cracking detection (EC2 concrete)
# ==================================================================

class TestMomentCurvatureCracking(unittest.TestCase):
    """
    Cracking detection requires EC2 concrete with fctm/Ecm.
    Verifies that cracking_chi and cracking_M are detected and
    that cracking precedes yield and ultimate.
    """

    @classmethod
    def setUpClass(cls):
        sec = _ec2_section()
        sv = FiberSolver(sec)
        nm_gen = NMDiagram(sv)
        cls.mc = nm_gen.generate_moment_curvature(
            N_fixed=-50e3, n_points=100)

    def test_cracking_chi_pos_detected(self):
        self.assertIsNotNone(self.mc["cracking_chi_pos"])

    def test_cracking_M_pos_positive(self):
        self.assertGreater(self.mc["cracking_M_pos"], 0.0)

    def test_cracking_chi_pos_positive(self):
        self.assertGreater(self.mc["cracking_chi_pos"], 0.0)

    def test_cracking_before_yield_pos(self):
        chi_cr = self.mc["cracking_chi_pos"]
        chi_y  = self.mc["yield_chi_pos"]
        if chi_cr is not None and chi_y is not None:
            self.assertLess(chi_cr, chi_y)

    def test_yield_before_ultimate_pos(self):
        chi_y = self.mc["yield_chi_pos"]
        chi_u = self.mc["ultimate_chi_pos"]
        if chi_y is not None and chi_u is not None:
            self.assertLess(chi_y, chi_u)

    def test_plain_concrete_no_cracking(self):
        """Basic Concrete (no .ec2 attribute) → cracking keys are None."""
        sec = _plain_section()
        mc = NMDiagram(FiberSolver(sec)).generate_moment_curvature(
            N_fixed=0.0, n_points=50)
        self.assertIsNone(mc["cracking_chi_pos"])
        self.assertIsNone(mc["cracking_M_pos"])

    def test_cracking_M_less_than_yield_M(self):
        chi_cr = self.mc["cracking_chi_pos"]
        M_cr   = self.mc["cracking_M_pos"]
        M_y    = self.mc["yield_M_pos"]
        if chi_cr is not None and M_y is not None:
            self.assertLess(M_cr, M_y)


# ==================================================================
#  Moment-curvature: yield and ultimate detection
# ==================================================================

class TestMomentCurvatureKeyPoints(unittest.TestCase):
    """Yield and ultimate key-point detection (plain concrete)."""

    @classmethod
    def setUpClass(cls):
        sec = _ec2_section()
        sv = FiberSolver(sec)
        cls.mc = NMDiagram(sv).generate_moment_curvature(
            N_fixed=0.0, n_points=100)

    def test_yield_chi_detected(self):
        self.assertIsNotNone(self.mc["yield_chi_pos"])

    def test_ultimate_chi_detected(self):
        self.assertIsNotNone(self.mc["ultimate_chi_pos"])

    def test_yield_M_positive(self):
        if self.mc["yield_M_pos"] is not None:
            self.assertGreater(self.mc["yield_M_pos"], 0.0)

    def test_ultimate_chi_greater_than_yield(self):
        y = self.mc["yield_chi_pos"]
        u = self.mc["ultimate_chi_pos"]
        if y is not None and u is not None:
            self.assertGreater(u, y)


# ==================================================================
#  3D resistance surface
# ==================================================================

class TestBiaxialSurface(unittest.TestCase):
    """generate_biaxial: point cloud properties."""

    @classmethod
    def setUpClass(cls):
        sec = _biaxial_section()
        sv = FiberSolver(sec)
        cls.nm_3d = NMDiagram(sv).generate_biaxial(
            n_angles=24, n_points_per_angle=80)

    def test_keys_present(self):
        for key in ("N", "Mx", "My", "N_kN", "Mx_kNm", "My_kNm"):
            self.assertIn(key, self.nm_3d)

    def test_min_point_count(self):
        # At least n_angles * n_points_per_angle points expected
        self.assertGreater(len(self.nm_3d["N"]), 1000)

    def test_arrays_same_length(self):
        n = len(self.nm_3d["N"])
        self.assertEqual(len(self.nm_3d["Mx"]), n)
        self.assertEqual(len(self.nm_3d["My"]), n)

    def test_N_kN_consistent(self):
        np.testing.assert_allclose(
            self.nm_3d["N_kN"], self.nm_3d["N"] / 1e3, rtol=1e-5)

    def test_Mx_kNm_consistent(self):
        np.testing.assert_allclose(
            self.nm_3d["Mx_kNm"], self.nm_3d["Mx"] / 1e6, rtol=1e-5)

    def test_symmetry_Mx(self):
        # Symmetric section → Mx and -Mx both present
        Mx = self.nm_3d["Mx"]
        self.assertGreater(Mx.max(), 0)
        self.assertLess(Mx.min(), 0)

    def test_symmetry_My(self):
        My = self.nm_3d["My"]
        self.assertGreater(My.max(), 0)
        self.assertLess(My.min(), 0)

    def test_compression_N_present(self):
        self.assertLess(self.nm_3d["N"].min(), 0)

    def test_tension_N_present(self):
        self.assertGreater(self.nm_3d["N"].max(), 0)


# ==================================================================
#  Mx-My contour
# ==================================================================

class TestMxMyContourGeneration(unittest.TestCase):
    """generate_mx_my: output structure and physical properties."""

    @classmethod
    def setUpClass(cls):
        sec = _biaxial_section()
        sv = FiberSolver(sec)
        cls.nm_gen = NMDiagram(sv)
        cls.mx_my = cls.nm_gen.generate_mx_my(
            -300e3, n_angles=36, n_points_per_angle=80)

    def test_keys_present(self):
        for key in ("Mx", "My", "Mx_kNm", "My_kNm", "N_fixed_kN"):
            self.assertIn(key, self.mx_my)

    def test_N_fixed_kN_correct(self):
        self.assertAlmostEqual(self.mx_my["N_fixed_kN"], -300.0, delta=0.01)

    def test_arrays_same_length(self):
        self.assertEqual(len(self.mx_my["Mx"]), len(self.mx_my["My"]))

    def test_Mx_kNm_consistent(self):
        np.testing.assert_allclose(
            self.mx_my["Mx_kNm"], self.mx_my["Mx"] / 1e6, rtol=1e-5)

    def test_My_kNm_consistent(self):
        np.testing.assert_allclose(
            self.mx_my["My_kNm"], self.mx_my["My"] / 1e6, rtol=1e-5)

    def test_contour_covers_both_signs_Mx(self):
        self.assertGreater(self.mx_my["Mx"].max(), 0)
        self.assertLess(self.mx_my["Mx"].min(), 0)

    def test_contour_covers_both_signs_My(self):
        self.assertGreater(self.mx_my["My"].max(), 0)
        self.assertLess(self.mx_my["My"].min(), 0)

    def test_contour_nonzero_area(self):
        # The Mx-My contour at any feasible N must span both signs
        # in both axes (i.e., non-degenerate convex hull).
        self.assertGreater(self.mx_my["Mx"].max(), 0)
        self.assertLess(self.mx_my["Mx"].min(), 0)
        self.assertGreater(self.mx_my["My"].max(), 0)
        self.assertLess(self.mx_my["My"].min(), 0)


# ==================================================================
#  Export: moment-curvature
# ==================================================================

class TestExportMomentCurvature(unittest.TestCase):
    """export_moment_curvature_json and _csv."""

    @classmethod
    def setUpClass(cls):
        sec = _ec2_section()
        sv = FiberSolver(sec)
        cls.mc = NMDiagram(sv).generate_moment_curvature(
            N_fixed=-50e3, n_points=60)
        cls.td = tempfile.mkdtemp()

    def test_json_keys(self):
        p = os.path.join(self.td, "mc.json")
        export_moment_curvature_json(self.mc, p)
        with open(p) as f:
            data = json.load(f)
        self.assertIn("chi_km", data)
        self.assertIn("M_kNm", data)
        self.assertIn("N_fixed_kN", data)
        self.assertIn("type", data)
        self.assertEqual(data["type"], "moment_curvature")

    def test_json_arrays_nonempty(self):
        p = os.path.join(self.td, "mc_arr.json")
        export_moment_curvature_json(self.mc, p)
        with open(p) as f:
            data = json.load(f)
        self.assertGreater(len(data["chi_km"]), 0)
        self.assertGreater(len(data["M_kNm"]), 0)

    def test_json_cracking_key_exported(self):
        p = os.path.join(self.td, "mc_cr.json")
        export_moment_curvature_json(self.mc, p)
        with open(p) as f:
            data = json.load(f)
        # If cracking was detected, key should be in JSON
        if self.mc["cracking_chi_pos"] is not None:
            self.assertIn("cracking_chi_pos_km", data)

    def test_json_N_fixed_value(self):
        p = os.path.join(self.td, "mc_n.json")
        export_moment_curvature_json(self.mc, p)
        with open(p) as f:
            data = json.load(f)
        self.assertAlmostEqual(data["N_fixed_kN"], -50.0, delta=0.01)

    def test_csv_header(self):
        p = os.path.join(self.td, "mc.csv")
        export_moment_curvature_csv(self.mc, p)
        with open(p) as f:
            reader = csv.reader(f)
            header = next(reader)
        self.assertIn("chi_km", header)
        self.assertIn("M_kNm", header)

    def test_csv_row_count_matches_array(self):
        p = os.path.join(self.td, "mc_rows.csv")
        export_moment_curvature_csv(self.mc, p)
        with open(p) as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
        self.assertEqual(len(rows), len(self.mc["chi_km"]))


# ==================================================================
#  Export: Mx-My contour
# ==================================================================

class TestExportMxMy(unittest.TestCase):
    """export_mx_my_json and _csv."""

    @classmethod
    def setUpClass(cls):
        sec = _biaxial_section()
        sv = FiberSolver(sec)
        cls.mx_my = NMDiagram(sv).generate_mx_my(
            -300e3, n_angles=24, n_points_per_angle=60)
        cls.td = tempfile.mkdtemp()

    def test_json_keys(self):
        p = os.path.join(self.td, "mxmy.json")
        export_mx_my_json(self.mx_my, p)
        with open(p) as f:
            data = json.load(f)
        self.assertIn("Mx_kNm", data)
        self.assertIn("My_kNm", data)
        self.assertIn("N_fixed_kN", data)
        self.assertEqual(data["type"], "mx_my_contour")

    def test_json_N_fixed(self):
        p = os.path.join(self.td, "mxmy_n.json")
        export_mx_my_json(self.mx_my, p)
        with open(p) as f:
            data = json.load(f)
        self.assertAlmostEqual(data["N_fixed_kN"], -300.0, delta=0.01)

    def test_json_arrays_nonempty(self):
        p = os.path.join(self.td, "mxmy_arr.json")
        export_mx_my_json(self.mx_my, p)
        with open(p) as f:
            data = json.load(f)
        self.assertGreater(len(data["Mx_kNm"]), 0)
        self.assertGreater(len(data["My_kNm"]), 0)

    def test_csv_header(self):
        p = os.path.join(self.td, "mxmy.csv")
        export_mx_my_csv(self.mx_my, p)
        with open(p) as f:
            reader = csv.reader(f)
            header = next(reader)
        self.assertEqual(header, ["Mx_kNm", "My_kNm"])

    def test_csv_row_count(self):
        p = os.path.join(self.td, "mxmy_rows.csv")
        export_mx_my_csv(self.mx_my, p)
        with open(p) as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
        self.assertEqual(len(rows), len(self.mx_my["Mx_kNm"]))


# ==================================================================
#  Export: 3D surface
# ==================================================================

class TestExport3DSurface(unittest.TestCase):
    """export_3d_surface_csv and _json."""

    @classmethod
    def setUpClass(cls):
        sec = _biaxial_section()
        sv = FiberSolver(sec)
        cls.nm_3d = NMDiagram(sv).generate_biaxial(
            n_angles=16, n_points_per_angle=60)
        cls.td = tempfile.mkdtemp()

    def test_csv_header(self):
        p = os.path.join(self.td, "surf.csv")
        export_3d_surface_csv(self.nm_3d, p)
        with open(p) as f:
            reader = csv.reader(f)
            header = next(reader)
        self.assertEqual(header, ["N_kN", "Mx_kNm", "My_kNm"])

    def test_csv_row_count_large(self):
        p = os.path.join(self.td, "surf_rows.csv")
        export_3d_surface_csv(self.nm_3d, p)
        with open(p) as f:
            reader = csv.reader(f)
            next(reader)
            rows = list(reader)
        self.assertGreater(len(rows), 100)

    def test_json_keys(self):
        p = os.path.join(self.td, "surf.json")
        export_3d_surface_json(self.nm_3d, p)
        with open(p) as f:
            data = json.load(f)
        for key in ("N_kN", "Mx_kNm", "My_kNm", "n_points"):
            self.assertIn(key, data)

    def test_json_n_points_consistent(self):
        p = os.path.join(self.td, "surf_n.json")
        export_3d_surface_json(self.nm_3d, p)
        with open(p) as f:
            data = json.load(f)
        self.assertEqual(data["n_points"], len(self.nm_3d["N"]))

    def test_json_arrays_length_consistent(self):
        p = os.path.join(self.td, "surf_len.json")
        export_3d_surface_json(self.nm_3d, p)
        with open(p) as f:
            data = json.load(f)
        self.assertEqual(len(data["N_kN"]), data["n_points"])
        self.assertEqual(len(data["Mx_kNm"]), data["n_points"])
        self.assertEqual(len(data["My_kNm"]), data["n_points"])


# ==================================================================
#  Export: combination and envelope results
# ==================================================================

class TestExportCombinationEnvelope(unittest.TestCase):
    """export_combination_results_json and export_envelope_results_json."""

    def setUp(self):
        self.td = tempfile.mkdtemp()

    def _simple_combo_result(self):
        return {
            "name": "SLU_1",
            "type": "simple",
            "resultant": {
                "N_kN": -700.0, "Mx_kNm": 76.0, "My_kNm": 15.0,
                "N": -700e3,    "Mx": 76e6,       "My": 15e6,
            },
            "eta_3D": 0.72,
            "inside": True,
            "verified": True,
        }

    def _staged_combo_result(self):
        return {
            "name": "SLU_staged",
            "type": "staged",
            "resultant": {
                "N_kN": -600.0, "Mx_kNm": 60.0, "My_kNm": 20.0,
                "N": -600e3,    "Mx": 60e6,       "My": 20e6,
            },
            "stages": [
                {
                    "name": "s0",
                    "increment":  {"N_kN": -500.0, "Mx_kNm": 20.0, "My_kNm": 5.0},
                    "cumulative": {"N_kN": -500.0, "Mx_kNm": 20.0, "My_kNm": 5.0},
                    "eta_3D": 0.35,
                },
                {
                    "name": "s1",
                    "increment":  {"N_kN": -100.0, "Mx_kNm": 40.0, "My_kNm": 15.0},
                    "cumulative": {"N_kN": -600.0, "Mx_kNm": 60.0, "My_kNm": 20.0},
                    "base": {"N_kN": -500.0, "Mx_kNm": 20.0, "My_kNm": 5.0},
                    "eta_path": 0.58,
                    "eta_3D": 0.62,
                },
            ],
            "eta_governing": 0.62,
            "inside": True,
            "verified": True,
        }

    def _envelope_result(self):
        return {
            "name": "Env1",
            "eta_max": 0.85,
            "governing_member": "D2",
            "verified": True,
            "members": [
                {"name": "D1", "N_kN": -300.0, "Mx_kNm": 20.0, "My_kNm": 5.0,
                 "eta_3D": 0.40, "inside": True, "verified": True},
                {"name": "D2", "N_kN": -400.0, "Mx_kNm": 30.0, "My_kNm": 10.0,
                 "eta_3D": 0.85, "inside": True, "verified": True},
            ],
        }

    # --- combination JSON ---

    def test_combination_json_top_key(self):
        p = os.path.join(self.td, "combos.json")
        export_combination_results_json([self._simple_combo_result()], p)
        with open(p) as f:
            data = json.load(f)
        self.assertIn("combinations", data)

    def test_combination_json_count(self):
        results = [self._simple_combo_result(), self._staged_combo_result()]
        p = os.path.join(self.td, "combos2.json")
        export_combination_results_json(results, p)
        with open(p) as f:
            data = json.load(f)
        self.assertEqual(len(data["combinations"]), 2)

    def test_combination_json_name_preserved(self):
        p = os.path.join(self.td, "combo_name.json")
        export_combination_results_json([self._simple_combo_result()], p)
        with open(p) as f:
            data = json.load(f)
        self.assertEqual(data["combinations"][0]["name"], "SLU_1")

    def test_combination_json_type_preserved(self):
        p = os.path.join(self.td, "combo_type.json")
        export_combination_results_json([self._simple_combo_result()], p)
        with open(p) as f:
            data = json.load(f)
        self.assertEqual(data["combinations"][0]["type"], "simple")

    def test_combination_json_resultant_present(self):
        p = os.path.join(self.td, "combo_res.json")
        export_combination_results_json([self._simple_combo_result()], p)
        with open(p) as f:
            data = json.load(f)
        self.assertIn("resultant", data["combinations"][0])

    def test_staged_combination_json_has_stages(self):
        p = os.path.join(self.td, "combo_staged.json")
        export_combination_results_json([self._staged_combo_result()], p)
        with open(p) as f:
            data = json.load(f)
        self.assertIn("stages", data["combinations"][0])
        self.assertEqual(len(data["combinations"][0]["stages"]), 2)

    def test_staged_combination_json_stage_names(self):
        p = os.path.join(self.td, "combo_staged_n.json")
        export_combination_results_json([self._staged_combo_result()], p)
        with open(p) as f:
            data = json.load(f)
        stages = data["combinations"][0]["stages"]
        names = [s["name"] for s in stages]
        self.assertIn("s0", names)
        self.assertIn("s1", names)

    # --- envelope JSON ---

    def test_envelope_json_top_key(self):
        p = os.path.join(self.td, "envs.json")
        export_envelope_results_json([self._envelope_result()], p)
        with open(p) as f:
            data = json.load(f)
        self.assertIn("envelopes", data)

    def test_envelope_json_eta_max(self):
        p = os.path.join(self.td, "env_eta.json")
        export_envelope_results_json([self._envelope_result()], p)
        with open(p) as f:
            data = json.load(f)
        self.assertAlmostEqual(
            data["envelopes"][0]["eta_max"], 0.85, places=4)

    def test_envelope_json_governing_member(self):
        p = os.path.join(self.td, "env_gov.json")
        export_envelope_results_json([self._envelope_result()], p)
        with open(p) as f:
            data = json.load(f)
        self.assertEqual(
            data["envelopes"][0]["governing_member"], "D2")

    def test_envelope_json_verified(self):
        p = os.path.join(self.td, "env_ver.json")
        export_envelope_results_json([self._envelope_result()], p)
        with open(p) as f:
            data = json.load(f)
        self.assertTrue(data["envelopes"][0]["verified"])

    def test_envelope_json_members_count(self):
        p = os.path.join(self.td, "env_mem.json")
        export_envelope_results_json([self._envelope_result()], p)
        with open(p) as f:
            data = json.load(f)
        self.assertEqual(len(data["envelopes"][0]["members"]), 2)

    def test_multiple_envelopes_json(self):
        r1 = self._envelope_result()
        r2 = {**self._envelope_result(), "name": "Env2"}
        p = os.path.join(self.td, "envs_multi.json")
        export_envelope_results_json([r1, r2], p)
        with open(p) as f:
            data = json.load(f)
        self.assertEqual(len(data["envelopes"]), 2)


if __name__ == "__main__":
    unittest.main()
