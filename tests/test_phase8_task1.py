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
Phase 8, Task 1 — unit tests: engine-level bulk staging.

Coverage map (primer §5, unit-granularity mirror of the validation
axes):

- named zones and zone references (B8);
- ``Tendon.parent`` override and ``system`` retirement (B3, B7);
- ``SectionState`` zone arrays, ``with_bulk_activated`` atomicity,
  capacity-hash mask/plane terms with the per-manager curvature
  quantum :math:`q_\chi = \texttt{QUANT\_EPS}/\max(H, B)`
  (B2-engine, B4);
- ``resolve_stages`` activation-declarative pre-scan, the
  ``initially_inactive`` extension, containment invariant
  :math:`\mathrm{active}[i] \Rightarrow
  \mathrm{bulk\_active}[\mathrm{parent}(i)]`, empty-active-bulk
  guard, ``deactivate_bulk`` reservation;
- ``materialize_view`` re-slice, geometry overrides, pinned
  reference, fast-path byte-identity (B5);
- integrator per-fiber offset field vs the scalar fast path (B6),
  including a 2-DOF closed-form equilibrium with an eccentric
  locked-in plane;
- YAML ``activate_bulk`` schema, strict ``material_zones`` keys,
  ``system`` migration error;
- SLS fail-loud guard + composite (multi-zone, static) SLS smoke.

Green under both runners::

    python -m unittest tests.test_phase8_task1 -v
    python -m pytest tests/test_phase8_task1.py -q

Fixture code is self-contained (regression-test independence: no
production helpers in the fixtures beyond the public constructors
under test).
"""

import unittest

import numpy as np
from shapely.geometry import box

from gensec.materials.elastic import LinearElastic
from gensec.materials.concrete import Concrete
from gensec.materials.steel import Steel, PrestressingSteel
from gensec.geometry.fiber import RebarLayer, Tendon
from gensec.geometry.geometry import GenericSection
from gensec.geometry.primitives import rect_poly
from gensec.solver.section_state import (
    SectionState, StagedDomainManager, materialize_view, QUANT_EPS,
)
from gensec.solver.integrator import FiberSolver


# ==================================================================
#  Fixtures
# ==================================================================

def _lin(E, name):
    r"""Unbounded linear-elastic law (closed-form checks)."""
    return LinearElastic(E=E, name=name)


def _steel():
    return Steel(fyk=450.0, gamma_s=1.15)


def _concrete(name="C45"):
    r"""EC2 parabola-rectangle concrete with a post-hoc name.

    ``Concrete`` has no ``name`` field; the SLS moduli resolver
    (Phase-7 D3) matches overrides by ``material.name``, so the name
    is attached post-hoc exactly as the Phase-7 fixtures do.
    """
    c = Concrete(fck=45.0)
    c.name = name
    return c


def _composite(rebars=(), tendons=(), zone_name="topping",
               linear=True, mesh=100.0):
    r"""
    600 x 1400 composite: base = precast (zone 0), topping slab
    :math:`y \in [1200, 1400]` as zone 1.  Grid mesh with the
    interface on a grid line (``n_grid_y = 14``, ``dy = 100``), so a
    masked stage-0 view is fiber-for-fiber a standalone 600 x 1200
    section.
    """
    if linear:
        c1, c2 = _lin(35000.0, "precast"), _lin(31000.0, "topping")
    else:
        c1, c2 = _concrete("C_precast"), _concrete("C_topping")
    return GenericSection(
        polygon=rect_poly(600.0, 1400.0),
        bulk_material=c1,
        rebars=list(rebars),
        tendons=list(tendons),
        bulk_materials=[(box(0, 1200, 600, 1400), c2, zone_name)],
        mesh_size=mesh, n_grid_x=6, n_grid_y=14,
    )


def _ps():
    r"""Prestressing steel with the library defaults (f_p01d, Ep)."""
    return PrestressingSteel()


# ==================================================================
#  Named zones (B8)
# ==================================================================

class TestZoneNaming(unittest.TestCase):
    """3-tuple names, auto-names, reserved/numeric/duplicate rules."""

    def test_named_zone_and_base(self):
        sec = _composite()
        self.assertEqual(sec.zone_names, ["base", "topping"])
        self.assertEqual(sec.n_zones, 2)

    def test_two_tuple_auto_name(self):
        c1, c2 = _lin(35e3, "a"), _lin(31e3, "b")
        sec = GenericSection(
            polygon=rect_poly(600, 1400), bulk_material=c1, rebars=[],
            bulk_materials=[(box(0, 1200, 600, 1400), c2)],
            mesh_size=100, n_grid_x=6, n_grid_y=14)
        self.assertEqual(sec.zone_names, ["base", "zone_1"])

    def test_three_tuple_none_auto_names(self):
        c1, c2 = _lin(35e3, "a"), _lin(31e3, "b")
        sec = GenericSection(
            polygon=rect_poly(600, 1400), bulk_material=c1, rebars=[],
            bulk_materials=[(box(0, 1200, 600, 1400), c2, None)],
            mesh_size=100, n_grid_x=6, n_grid_y=14)
        self.assertEqual(sec.zone_names, ["base", "zone_1"])

    def test_reserved_base_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            _composite(zone_name="base")

    def test_numeric_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "numeric"):
            _composite(zone_name="2")

    def test_non_string_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "string"):
            _composite(zone_name=7)

    def test_duplicate_name_rejected(self):
        c1, c2 = _lin(35e3, "a"), _lin(31e3, "b")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            GenericSection(
                polygon=rect_poly(600, 1400), bulk_material=c1,
                rebars=[],
                bulk_materials=[
                    (box(0, 1200, 600, 1300), c2, "top"),
                    (box(0, 1300, 600, 1400), c2, "top")],
                mesh_size=100, n_grid_x=6, n_grid_y=14)

    def test_zone_index_resolution(self):
        sec = _composite()
        self.assertEqual(sec.zone_index("base"), 0)
        self.assertEqual(sec.zone_index("topping"), 1)
        self.assertEqual(sec.zone_index(0), 0)
        self.assertEqual(sec.zone_index(1), 1)
        with self.assertRaisesRegex(ValueError, "unknown"):
            sec.zone_index("slab")
        with self.assertRaisesRegex(ValueError, "out of range"):
            sec.zone_index(2)
        with self.assertRaisesRegex(ValueError, "boolean"):
            sec.zone_index(True)

    def test_normalized_storage_is_two_tuples(self):
        # Internal contract: downstream consumers unpack 2-tuples.
        sec = _composite()
        self.assertEqual(len(sec.bulk_materials[0]), 2)


# ==================================================================
#  Tendon.parent / system retirement (B3, B7)
# ==================================================================

class TestTendonParent(unittest.TestCase):

    def test_system_field_removed(self):
        t = Tendon(y=100.0, material=_ps(), Ap=1000.0, eps_pe=6e-3)
        self.assertFalse(hasattr(t, "system"))
        with self.assertRaises(TypeError):
            Tendon(y=100.0, material=_ps(), Ap=1000.0,
                   eps_pe=6e-3, system="pre")

    def test_parent_override_requires_non_embedded(self):
        t = Tendon(y=100.0, x=300.0, material=_ps(), Ap=1000.0,
                   eps_pe=6e-3, parent="topping")   # embedded default
        with self.assertRaisesRegex(ValueError, "embedded=False"):
            _composite(tendons=[t])

    def test_parent_override_resolves_for_non_embedded(self):
        t = Tendon(y=100.0, x=300.0, material=_ps(), Ap=1000.0,
                   eps_pe=6e-3, parent="topping", embedded=False)
        sec = _composite(tendons=[t])
        # Geometric zone is 0 (y=100 is in the precast); the staging
        # parent is overridden to the topping — the physics
        # (displaced bulk) keeps the geometric zone.
        self.assertEqual(int(sec.mat_indices_tendon[0]), 0)
        self.assertEqual(int(sec.staging_parent_tendon[0]), 1)

    def test_default_parent_is_geometric(self):
        t = Tendon(y=1300.0, x=300.0, material=_ps(), Ap=1000.0,
                   eps_pe=6e-3)
        sec = _composite(tendons=[t])
        self.assertEqual(int(sec.staging_parent_tendon[0]), 1)


# ==================================================================
#  SectionState: zone arrays, activation op, hash (B2, B4)
# ==================================================================

class TestStateAndHash(unittest.TestCase):

    def setUp(self):
        self.sec = _composite()
        self.mgr = StagedDomainManager(self.sec, biaxial=False)
        self.s0 = self.mgr.initial_state()

    def test_initial_state_arrays(self):
        self.assertTrue(np.all(self.s0.bulk_active))
        self.assertEqual(self.s0.bulk_active.shape, (2,))
        self.assertEqual(self.s0.bulk_planes.shape, (2, 3))
        self.assertTrue(np.all(self.s0.bulk_planes == 0.0))

    def test_copy_advanced_propagates_arrays(self):
        s1 = self.s0.copy_advanced(1)
        self.assertIsNot(s1.bulk_active, self.s0.bulk_active)
        self.assertTrue(np.array_equal(s1.bulk_active,
                                       self.s0.bulk_active))
        self.assertIsNot(s1.bulk_planes, self.s0.bulk_planes)

    def _inactive_topping(self):
        s = self.s0.copy_advanced(0)
        s.bulk_active[1] = False
        return s

    def test_with_bulk_activated_atomicity(self):
        s = self._inactive_topping()
        with self.assertRaisesRegex(KeyError, "datum plane"):
            s.with_bulk_activated([1], {})
        s2 = s.with_bulk_activated([1], {1: (2e-4, -3e-7, 0.0)})
        self.assertTrue(bool(s2.bulk_active[1]))
        np.testing.assert_allclose(s2.bulk_planes[1],
                                   [2e-4, -3e-7, 0.0])
        # Source state untouched (functional op).
        self.assertFalse(bool(s.bulk_active[1]))

    def test_with_bulk_activated_zone0_rejected(self):
        with self.assertRaisesRegex(ValueError, "always active"):
            self.s0.with_bulk_activated([0], {0: (0, 0, 0)})

    def test_with_bulk_activated_range_and_shape(self):
        s = self._inactive_topping()
        with self.assertRaisesRegex(ValueError, "out of range"):
            s.with_bulk_activated([5], {5: (0, 0, 0)})
        with self.assertRaisesRegex(ValueError, "finite"):
            s.with_bulk_activated([1], {1: (np.nan, 0, 0)})

    def test_with_bulk_activated_requires_arrays(self):
        n = 0
        bare = SectionState(0, np.ones(n, bool), np.ones(n, bool),
                            np.zeros(n))
        with self.assertRaisesRegex(ValueError, "no zone arrays"):
            bare.with_bulk_activated([1], {1: (0, 0, 0)})

    def test_hash_mask_flip_changes(self):
        h0 = self.mgr.hash_of(self.s0)
        self.assertNotEqual(self.mgr.hash_of(self._inactive_topping()),
                            h0)

    def test_hash_plane_quantum(self):
        s = self._inactive_topping()
        base = s.with_bulk_activated([1], {1: (2e-4, -3e-7, 0.0)})
        h = self.mgr.hash_of(base)
        # eps0 within the QUANT_EPS bucket -> same hash.
        same = s.with_bulk_activated(
            [1], {1: (2e-4 + 0.4 * QUANT_EPS, -3e-7, 0.0)})
        self.assertEqual(self.mgr.hash_of(same), h)
        # eps0 beyond -> different.
        diff = s.with_bulk_activated(
            [1], {1: (2e-4 + 1.1 * QUANT_EPS, -3e-7, 0.0)})
        self.assertNotEqual(self.mgr.hash_of(diff), h)
        # chi within its own quantum QUANT_EPS / max(H, B) -> same.
        qchi = self.mgr._chi_quantum
        self.assertAlmostEqual(qchi, QUANT_EPS / 1400.0)
        same_chi = s.with_bulk_activated(
            [1], {1: (2e-4, -3e-7 + 0.4 * qchi, 0.0)})
        self.assertEqual(self.mgr.hash_of(same_chi), h)
        diff_chi = s.with_bulk_activated(
            [1], {1: (2e-4, -3e-7 + 1.1 * qchi, 0.0)})
        self.assertNotEqual(self.mgr.hash_of(diff_chi), h)

    def test_hash_legacy_none_arrays_still_works(self):
        n = 0
        bare = SectionState(0, np.ones(n, bool), np.ones(n, bool),
                            np.zeros(n))
        self.assertIsInstance(self.mgr.hash_of(bare), int)


# ==================================================================
#  resolve_stages: pre-scan + invariants
# ==================================================================

class TestResolveStages(unittest.TestCase):

    def setUp(self):
        st = _steel()
        self.reb_bot = RebarLayer(x=60.0, y=60.0, As=800.0,
                                  material=st)
        self.reb_top = RebarLayer(x=60.0, y=1340.0, As=400.0,
                                  material=st)
        self.sec = _composite(rebars=[self.reb_bot, self.reb_top],
                              linear=False)
        self.mgr = StagedDomainManager(self.sec, biaxial=False,
                                       gen_kwargs={"n_points": 20})

    def test_prescan_starts_planned_zone_inactive(self):
        stages = [
            {"section_ops": {"deactivate": [1], "release": False}},
            {"section_ops": {"activate_bulk": {1: (0.0, 0.0, 0.0)},
                             "activate": [1]}},
        ]
        states, hashes, bundles, _ = self.mgr.resolve_stages(stages)
        self.assertFalse(bool(states[0].bulk_active[1]))
        self.assertTrue(bool(states[1].bulk_active[1]))
        self.assertNotEqual(hashes[0], hashes[1])

    def test_initially_inactive_without_activation(self):
        # Timeline-compiler case (reconciliation with the Task-2
        # primer): a prefix anchored before the zone's casting event
        # carries no activate_bulk — the zone must still start (and
        # stay) inactive.
        stages = [
            {"section_ops": {"deactivate": [1], "release": False}},
            {},
        ]
        states, _, _, _ = self.mgr.resolve_stages(
            stages, initially_inactive=[1])
        self.assertFalse(bool(states[0].bulk_active[1]))
        self.assertFalse(bool(states[1].bulk_active[1]))

    def test_initially_inactive_range_checked(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.mgr.resolve_stages([{}], initially_inactive=[9])

    def test_reactivation_rejected(self):
        stages = [
            {"section_ops": {"deactivate": [1], "release": False,
                             "activate_bulk": {1: (0.0, 0.0, 0.0)}}},
            {"section_ops": {"activate_bulk": {1: (0.0, 0.0, 0.0)},
                             "activate": [1]}},
        ]
        with self.assertRaisesRegex(ValueError, "re-activates"):
            self.mgr.resolve_stages(stages)

    def test_deactivate_bulk_reserved(self):
        with self.assertRaises(NotImplementedError):
            self.mgr.resolve_stages(
                [{"section_ops": {"deactivate_bulk": [1]}}])

    def test_zone_range_checked(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.mgr.resolve_stages(
                [{"section_ops":
                  {"activate_bulk": {3: (0.0, 0.0, 0.0)}}}])

    def test_containment_invariant(self):
        # Top rebar (index 1) sits in the topping: leaving it active
        # at stage 0 while the topping is not yet cast must raise.
        stages = [
            {"section_ops": {}},
            {"section_ops": {"activate_bulk": {1: (0.0, 0.0, 0.0)}}},
        ]
        with self.assertRaisesRegex(ValueError, "staging-parent"):
            self.mgr.resolve_stages(stages)

    def test_empty_active_bulk_rejected(self):
        # Zones covering the whole polygon, both cast later: stage 0
        # has no active fiber.
        c1 = _lin(35e3, "a")
        c2 = _lin(31e3, "b")
        sec = GenericSection(
            polygon=rect_poly(600, 1400), bulk_material=c1, rebars=[],
            bulk_materials=[(box(0, 0, 600, 1200), c2, "web"),
                            (box(0, 1200, 600, 1400), c2, "top")],
            mesh_size=100, n_grid_x=6, n_grid_y=14)
        mgr = StagedDomainManager(sec, biaxial=False,
                                  gen_kwargs={"n_points": 20})
        stages = [
            {"section_ops": {}},
            {"section_ops": {"activate_bulk":
                             {1: (0, 0, 0), 2: (0, 0, 0)}}},
        ]
        with self.assertRaisesRegex(ValueError, "active bulk is empty"):
            mgr.resolve_stages(stages)


# ==================================================================
#  materialize_view (B5)
# ==================================================================

class TestMaterializeView(unittest.TestCase):

    def setUp(self):
        self.sec = _composite()
        self.mgr = StagedDomainManager(self.sec, biaxial=False)
        self.s0 = self.mgr.initial_state()

    def test_fast_path_byte_identity(self):
        v = materialize_view(self.sec, self.s0)
        self.assertIs(v.x_fibers, self.sec.x_fibers)
        self.assertIs(v.mat_indices, self.sec.mat_indices)
        self.assertFalse(hasattr(v, "bulk_planes_active"))
        self.assertEqual(v.H, self.sec.H)

    def test_masked_reslice_and_geometry(self):
        s = self.s0.copy_advanced(0)
        s.bulk_active[1] = False
        v = materialize_view(self.sec, s)
        self.assertEqual(v.n_fibers,
                         int(np.sum(self.sec.mat_indices == 0)))
        self.assertEqual(v.H, 1200.0)
        self.assertEqual(v.bbox, (0.0, 0.0, 600.0, 1200.0))
        self.assertEqual(v.ideal_gross_area, 600.0 * 1200.0)
        # Reference pinned to the full-polygon centroid.
        self.assertEqual(v.y_centroid, self.sec.y_centroid)
        self.assertEqual(float(v.y_centroid), 700.0)
        # Base arrays untouched.
        self.assertEqual(self.sec.n_fibers, 84)

    def test_all_active_nonzero_planes_attaches_planes(self):
        # The fast-path condition is mask all-True AND planes
        # all-zero: an all-active state with non-zero planes (final
        # composite stage) shares the arrays but MUST attach the
        # planes, or the locked-in datum would silently vanish.
        s = self.s0.copy_advanced(0)
        s.bulk_active[1] = False
        s = s.with_bulk_activated([1], {1: (2e-4, 3e-7, 0.0)})
        self.assertTrue(np.all(s.bulk_active))
        v = materialize_view(self.sec, s)
        self.assertIs(v.x_fibers, self.sec.x_fibers)
        self.assertTrue(hasattr(v, "bulk_planes_active"))
        np.testing.assert_allclose(v.bulk_planes_active[1],
                                   [2e-4, 3e-7, 0.0])

    def test_zone0_inactive_rejected(self):
        s = self.s0.copy_advanced(0)
        s.bulk_active[0] = False
        with self.assertRaisesRegex(ValueError, "zone 0"):
            materialize_view(self.sec, s)


# ==================================================================
#  Integrator offset field (B6)
# ==================================================================

class TestOffsetField(unittest.TestCase):

    def test_fast_path_fields_none(self):
        sec = _composite()
        sv = FiberSolver(sec)
        self.assertIsNone(sv._bulk_eps_field)
        self.assertEqual(sv._bulk_offset(), 0.0)

    def test_field_matches_hand_formula(self):
        r"""Field equals the strain_field-convention evaluation.

        .. math::

           \varepsilon^{\mathrm{off}}_i = \varepsilon_{0,z(i)}
           + \chi_{x,z(i)} (y_i - y_{\mathrm{ref}})
           - \chi_{y,z(i)} (x_i - x_{\mathrm{ref}})
        """
        sec = _composite()
        mgr = StagedDomainManager(sec, biaxial=False)
        s = mgr.initial_state().copy_advanced(0)
        s.bulk_active[1] = False
        plane = (2e-4, 3e-7, -1e-7)
        s = s.with_bulk_activated([1], {1: plane})
        v = materialize_view(sec, s)
        sv = FiberSolver(v)
        mi = np.asarray(v.mat_indices)
        ly = v.y_fibers - sv.y_ref
        lx = v.x_fibers - sv.x_ref
        hand = np.where(
            mi == 1,
            plane[0] + plane[1] * ly - plane[2] * lx,
            0.0)
        np.testing.assert_allclose(sv._bulk_eps_field, hand,
                                   rtol=0, atol=1e-15)

    def test_scalar_bulk_eps_still_added(self):
        # Legacy uniform prestrain rides on top of the planes.
        sec = _composite()
        sec.bulk_eps_init = -3e-4
        mgr = StagedDomainManager(sec, biaxial=False)
        s = mgr.initial_state().copy_advanced(0)
        s.bulk_active[1] = False
        s = s.with_bulk_activated([1], {1: (2e-4, 0.0, 0.0)})
        v = materialize_view(sec, s)
        sv = FiberSolver(v)
        mi = np.asarray(v.mat_indices)
        expect = -3e-4 + np.where(mi == 1, 2e-4, 0.0)
        np.testing.assert_allclose(sv._bulk_eps_field, expect,
                                   rtol=0, atol=1e-15)

    def test_equilibrium_with_plane_linear(self):
        r"""2-DOF closed-form equilibrium with an eccentric datum.

        Uniform :math:`\varepsilon_0`-only datum on the (eccentric)
        topping, zero external demand:

        .. math::

           \mathbf{K}\,[\varepsilon_0, \chi_x]^{\mathsf T}
           = -\mathbf{F}_{\mathrm{lock}}

        with :math:`\mathbf{F}_{\mathrm{lock}}` the resultant of the
        datum plane over the topping fibers.  The eccentric lock-in
        force also *bends* the section — a 1-DOF axial-only closed
        form is wrong by construction.
        """
        sec = _composite()
        mgr = StagedDomainManager(sec, biaxial=False)
        s = mgr.initial_state().copy_advanced(0)
        s.bulk_active[1] = False
        eps_d = 2e-4
        s = s.with_bulk_activated([1], {1: (eps_d, 0.0, 0.0)})
        v = materialize_view(sec, s)
        sv = FiberSolver(v)
        mi = np.asarray(v.mat_indices)
        E = np.where(mi == 0, 35000.0, 31000.0)
        A = v.A_fibers
        ly = v.y_fibers - sv.y_ref
        K = np.array([[np.sum(E * A), np.sum(E * A * ly)],
                      [np.sum(E * A * ly), np.sum(E * A * ly * ly)]])
        top = mi == 1
        F_lock = np.array([np.sum(E[top] * A[top]) * eps_d,
                           np.sum(E[top] * A[top] * ly[top]) * eps_d])
        u = np.linalg.solve(K, -F_lock)
        sol = sv.solve_equilibrium(0.0, 0.0, 0.0)
        self.assertTrue(sol["converged"])
        self.assertAlmostEqual(sol["eps0"], u[0], places=12)
        self.assertAlmostEqual(sol["chi_x"], u[1], places=14)


# ==================================================================
#  YAML surface
# ==================================================================

class TestYamlSurface(unittest.TestCase):

    _BASE = """
materials:
  c1: {type: concrete_ec2_gen1, class: 'C45/55'}
  c2: {type: concrete_ec2_gen1, class: 'C30/37'}
  s:  {type: steel, fyk: 450, gamma_s: 1.15}
section:
  shape: rect
  params: {B: 600, H: 1400}
  bulk_material: c1
  mesh_size: 100
  material_zones:
    - shape: custom
      params: {exterior: [[0, 1200], [600, 1200], [600, 1400], [0, 1400]]}
      material: c2
      name: topping
  rebars:
    - {x: 60,  y: 60,   As: 800, material: s, name: B1}
    - {x: 540, y: 1340, As: 400, material: s, name: T1}
demands:
  - {name: G, N_kN: -100, Mx_kNm: -50}
"""

    @staticmethod
    def _load(text):
        import tempfile
        import os
        from gensec.io_yaml import load_yaml
        fd, path = tempfile.mkstemp(suffix=".yaml")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(text)
            return load_yaml(path)
        finally:
            os.unlink(path)

    def _combo(self, ops_stage1):
        return self._BASE + f"""
combinations:
  - name: c
    stages:
      - name: s0
        components: [{{ref: G, factor: 1.0}}]
        section_ops: {{deactivate: [T1], release: false}}
      - name: s1
        components: [{{ref: G, factor: 0.5}}]
        section_ops: {ops_stage1}
"""

    def test_zone_name_parsed(self):
        d = self._load(self._BASE)
        self.assertEqual(d["section"].zone_names, ["base", "topping"])

    def test_unknown_zone_key_rejected(self):
        bad = self._BASE.replace("name: topping", "nome: topping")
        with self.assertRaisesRegex(ValueError, "unknown key"):
            self._load(bad)

    def test_system_key_migration_error(self):
        text = self._BASE.replace(
            "  rebars:",
            "  tendons:\n"
            "    - {y: 100, x: 300, material: s, Ap: 1000,"
            " eps_pe: 0.006, system: pre}\n"
            "  rebars:")
        with self.assertRaisesRegex(ValueError, "retired"):
            self._load(text)

    def test_activate_bulk_resolved(self):
        d = self._load(self._combo(
            "{activate_bulk: {topping: {eps0: 2.0e-4, chi_x: 0.0, "
            "chi_y: 0.0}}, activate: [T1]}"))
        ops = d["combinations"][0]["stages"][1]["section_ops"]
        self.assertIn(1, ops["activate_bulk"])
        self.assertEqual(ops["activate_bulk"][1][0], 2.0e-4)

    def test_activate_bulk_by_index(self):
        d = self._load(self._combo(
            "{activate_bulk: {1: {eps0: 0.0, chi_x: 0.0, chi_y: 0.0}},"
            " activate: [T1]}"))
        ops = d["combinations"][0]["stages"][1]["section_ops"]
        self.assertIn(1, ops["activate_bulk"])

    def test_activate_bulk_base_rejected(self):
        with self.assertRaisesRegex(ValueError, "always active"):
            self._load(self._combo(
                "{activate_bulk: {base: {eps0: 0.0, chi_x: 0.0, "
                "chi_y: 0.0}}}"))

    def test_datum_missing_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing datum"):
            self._load(self._combo(
                "{activate_bulk: {topping: {eps0: 0.0, chi_x: 0.0}}}"))

    def test_datum_unknown_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown datum"):
            self._load(self._combo(
                "{activate_bulk: {topping: {eps0: 0.0, chi_x: 0.0, "
                "chi_y: 0.0, kappa: 1.0}}}"))

    def test_unknown_ops_key_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "[Uu]nknown"):
            self._load(self._combo("{aktivate_bulk: {topping: "
                                   "{eps0: 0.0, chi_x: 0.0, "
                                   "chi_y: 0.0}}}"))

    def test_deactivate_bulk_reserved_at_parse(self):
        with self.assertRaises(NotImplementedError):
            self._load(self._combo("{deactivate_bulk: [topping]}"))


# ==================================================================
#  SLS guard + composite static smoke
# ==================================================================

class TestSlsBoundary(unittest.TestCase):

    def _staged_input(self, state):
        from gensec.solver.sls import verify_sls_staged
        sec = _composite(rebars=[RebarLayer(x=60.0, y=60.0, As=800.0,
                                            material=_steel())],
                         linear=False)
        # D3 rule: concrete SLS moduli are explicit — by name here
        # (Material dataclasses are unhashable; by-instance overrides
        # go through resolve_sls_moduli's id() indirection, which the
        # public mapping does not expose).
        moduli = {"C_precast": 36000.0, "C_topping": 33000.0}
        return verify_sls_staged(
            sec,
            [{"name": "s0", "state": state,
              "increment": (-1e5, -5e7, 0.0)}],
            moduli=moduli)

    def test_masked_state_supported(self):
        # C4 lifted the former guard: a masked composite state (an
        # inactive zone) is now a supported SLS input, not a rejection.
        sec = _composite(linear=False)
        mgr = StagedDomainManager(sec, biaxial=False)
        s = mgr.initial_state().copy_advanced(0)
        s.bulk_active[1] = False
        n = 1
        s.active = np.ones(n, bool)
        s.bonded = np.ones(n, bool)
        s.eps_init = np.zeros(n)
        r = self._staged_input(s)
        self.assertIn("stages", r)
        self.assertIn("concrete", r["stages"][0])

    def test_nonzero_planes_supported(self):
        # C4 lifted the former guard: per-zone locked-in datum planes are
        # now honoured through the solver, not rejected.
        sec = _composite(linear=False)
        mgr = StagedDomainManager(sec, biaxial=False)
        s = mgr.initial_state().copy_advanced(0)
        s.bulk_active[1] = False
        s = s.with_bulk_activated([1], {1: (1e-4, 0.0, 0.0)})
        n = 1
        s.active = np.ones(n, bool)
        s.bonded = np.ones(n, bool)
        s.eps_init = np.zeros(n)
        r = self._staged_input(s)
        self.assertIn("stages", r)
        self.assertIn("concrete", r["stages"][0])

    def test_composite_static_sls_smoke(self):
        # Multi-zone, all-active, zero-planes: the supported path
        # (mask-transparent through the view; nothing new engaged).
        n = 1
        s = SectionState(0, np.ones(n, bool), np.ones(n, bool),
                         np.zeros(n),
                         bulk_active=np.ones(2, bool),
                         bulk_planes=np.zeros((2, 3)))
        r = self._staged_input(s)
        self.assertIn("stages", r)


if __name__ == "__main__":
    unittest.main()
