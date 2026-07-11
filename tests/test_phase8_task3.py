# ---------------------------------------------------------------------------
# GenSec — Copyright (c) 2026 Andrea Albero
# This file is part of GenSec.  AGPL-3.0-or-later.
# ---------------------------------------------------------------------------
r"""Phase-8 Task-3 (fork C4) regression tests: composite staged SLS."""
import unittest

import numpy as np
from shapely.geometry import box

from gensec.materials.elastic import LinearElastic
from gensec.geometry.geometry import GenericSection
from gensec.geometry.primitives import rect_poly
from gensec.solver.section_state import SectionState
from gensec.solver.sls import verify_sls_staged

E1, E2 = 35000.0, 31000.0
B, H1, H2 = 400.0, 800.0, 200.0
H = H1 + H2
YR = H / 2.0
MOD_C = {"precast": E1, "topping": E2}
MOD_1 = {"precast": E1}


def _composite():
    return GenericSection(
        polygon=rect_poly(B, H),
        bulk_material=LinearElastic(E=E1, name="precast"), rebars=[],
        bulk_materials=[(box(0, H1, B, H),
                         LinearElastic(E=E2, name="topping"), "topping")],
        mesh_size=50.0, n_grid_x=8, n_grid_y=20)


def _single():
    return GenericSection(
        polygon=rect_poly(B, H1),
        bulk_material=LinearElastic(E=E1, name="precast"), rebars=[],
        mesh_size=50.0, n_grid_x=8, n_grid_y=16)


def _state(sec, *, bulk_active=None, bulk_planes=None, idx=0, label=""):
    n = int(sec.x_rebars.size) + int(getattr(sec, "x_tendons",
                                             np.empty(0)).size)
    return SectionState(
        idx, np.ones(n, bool), np.ones(n, bool), np.zeros(n, float),
        bulk_active=(None if bulk_active is None
                     else np.asarray(bulk_active, bool)),
        bulk_planes=(None if bulk_planes is None
                     else np.asarray(bulk_planes, float)),
        label=label)


class TestCompositeSLS(unittest.TestCase):

    def test_masked_view_runs_and_reports_active_zones(self):
        sec = _composite()
        s = _state(sec, bulk_active=[True, False], label="transfer")
        res = verify_sls_staged(
            sec, [{"name": "transfer", "state": s,
                   "increment": (0.0, -1.2e8, 0.0)}],
            moduli=MOD_C, x_ref=B / 2.0, y_ref=YR)
        self.assertEqual(res["type"], "sls_staged")
        az = res["stages"][0].get("active_zones")
        self.assertIsNotNone(az)
        self.assertNotIn("topping", az)          # topping not yet cast

    def test_present_mask_excludes_uncast_topping(self):
        sec = _composite()
        s = _state(sec, bulk_active=[True, False], label="transfer")
        res = verify_sls_staged(
            sec, [{"name": "transfer", "state": s,
                   "increment": (0.0, -1.2e8, 0.0)}],
            moduli=MOD_C, x_ref=B / 2.0, y_ref=YR)
        c = res["stages"][0]["concrete"]
        self.assertLess(c["at_min"][1], H1 + 1e-6)
        self.assertLess(c["at_max"][1], H1 + 1e-6)

    def test_single_zone_path_unperturbed(self):
        sec = _single()
        s = _state(sec, label="one")
        res = verify_sls_staged(
            sec, [{"name": "one", "state": s,
                   "increment": (0.0, -1.2e8, 0.0)}],
            moduli=MOD_1, x_ref=B / 2.0, y_ref=YR)
        c = res["stages"][0]["concrete"]
        # pure moment on a doubly-symmetric zone -> antisymmetric stresses
        self.assertAlmostEqual(c["sigma_min_MPa"], -c["sigma_max_MPa"],
                               places=3)

    def test_invariant_guard_on_persisting_datum_change(self):
        sec = _composite()
        p0 = np.array([[0.10, 0.0, 0.0], [0.0, 0.0, 0.0]], float)
        p1 = np.array([[0.20, 0.0, 0.0], [0.0, 0.0, 0.0]], float)
        s0 = _state(sec, bulk_active=[True, True], bulk_planes=p0,
                    idx=0, label="a")
        s1 = _state(sec, bulk_active=[True, True], bulk_planes=p1,
                    idx=1, label="b")
        with self.assertRaises(NotImplementedError):
            verify_sls_staged(
                sec,
                [{"name": "a", "state": s0, "increment": (0.0, 0.0, 0.0)},
                 {"name": "b", "state": s1,
                  "increment": (0.0, -1e8, 0.0)}],
                moduli=MOD_C, x_ref=B / 2.0, y_ref=YR)


if __name__ == "__main__":
    unittest.main()
