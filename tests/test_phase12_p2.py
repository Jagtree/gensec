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
Phase 12 / P2 — slot-keyed modulus resolution (12_0d S1).

Rule 3 first
------------
Two of these fixtures did not exist anywhere in the tree, which is why
12_0d's D-1 and D-2 were latent rather than active: *every* shipped
example and validator declares distinct material objects for its bulk
zones, and neither defect can be exhibited by a section that does.

- :func:`_shared_material_section` — two bulk zones holding the
  **identical** material object.  Reachable from the public YAML
  surface with no trickery (``io_yaml`` stores materials in a
  ``{name: object}`` mapping and hands ``materials[name]`` to every
  zone that declares that name).
- :func:`_distinct_material_section` — the same geometry with two
  material objects.  It is the *inertia* reference: nothing in P2 may
  move on it.

The two together are the acceptance signal of a re-keying: only the
shared-object fixture may move, and it must move from wrong to right.

What is asserted, and where it is read from
-------------------------------------------
Wherever a modulus is checked, it is read back out of the **substituted
view** (:class:`~gensec.materials.elastic.LinearElastic.E`), never out
of the mapping that was fed in — rule 2, applied inside the suite: an
assertion that reads the implementation's own input measures nothing.
"""

import unittest

import numpy as np
from shapely.geometry import Polygon

from gensec.geometry.fiber import RebarLayer, Tendon
from gensec.geometry.section import GenericSection
from gensec.materials.concrete import Concrete
from gensec.materials.elastic import LinearElastic
from gensec.materials.steel import PrestressingSteel, Steel
from gensec.solver.section_state import SectionState, materialize_view
from gensec.solver.sls import (
    MaterialMap, ModulusPolicy, SlsPolicy, covered_zone_slots,
    resolve_moduli, resolve_sls_moduli, sls_view, verify_sls_staged,
)


# ==================================================================
#  Model data
# ==================================================================

B, H = 300.0, 600.0
Y_TOP = 500.0                    # topping soffit
E_SUB, E_TOP = 36283.0, 22000.0  # E_cm(90 d) and E_cm(4 d) of a C45
ES, EP = 200000.0, 195000.0
PLACEHOLDER = 1.0                # losses._UNCAST_PLACEHOLDER_E


def _outline():
    return Polygon([(0.0, 0.0), (B, 0.0), (B, H), (0.0, H)])


def _topping():
    return Polygon([(0.0, Y_TOP), (B, Y_TOP), (B, H), (0.0, H)])


def _steel():
    st = Steel(fyk=450.0)
    st.name = "B450C"
    return st


def _shared_material_section(named_elements=True):
    r"""
    Two bulk zones, **one** material object.

    Parameters
    ----------
    named_elements : bool, optional
        Whether the rebars carry declared names (element slot keys by
        name are only legal for declared names).

    Returns
    -------
    tuple
        ``(section, concrete, steel)``.
    """
    conc = Concrete(fck=45.0, gamma_c=1.5, long_term_reduction_factor=0.85)      # real non-linear law
    conc.name = "C45"
    st = _steel()
    reb = [
        RebarLayer(y=50.0, As=628.0, material=st, x=150.0,
                   name="A1" if named_elements else None),
        RebarLayer(y=550.0, As=402.0, material=st, x=150.0,
                   name="A2" if named_elements else None),
    ]
    sec = GenericSection(
        polygon=_outline(), bulk_material=conc, rebars=reb,
        mesh_size=25.0,
        bulk_materials=[(_topping(), conc, "topping")],
    )
    return sec, conc, st


def _distinct_material_section():
    r"""
    Same geometry, two material objects — the inertia reference.

    Returns
    -------
    tuple
        ``(section, c_base, c_topping, steel)``.
    """
    c0 = Concrete(fck=45.0, gamma_c=1.5, long_term_reduction_factor=0.85)
    c0.name = "C45 precast"
    c1 = Concrete(fck=45.0, gamma_c=1.5, long_term_reduction_factor=0.85)
    c1.name = "C45 topping"
    st = _steel()
    reb = [RebarLayer(y=50.0, As=628.0, material=st, x=150.0, name="A1"),
           RebarLayer(y=550.0, As=402.0, material=st, x=150.0, name="A2")]
    sec = GenericSection(
        polygon=_outline(), bulk_material=c0, rebars=reb,
        mesh_size=25.0,
        bulk_materials=[(_topping(), c1, "topping")],
    )
    return sec, c0, c1, st


def _tendon_section():
    r"""
    Single-zone section with two rebars and one tendon, for the
    element-slot axes.

    Returns
    -------
    gensec.geometry.section.GenericSection
    """
    conc = Concrete(fck=35.0, gamma_c=1.5, long_term_reduction_factor=0.85)
    conc.name = "C35"
    st = _steel()
    ps = PrestressingSteel(f_p01d=1391.3, Ep=EP)
    ps.name = "Y1860"
    return GenericSection(
        polygon=_outline(), bulk_material=conc,
        rebars=[RebarLayer(y=50.0, As=628.0, material=st, x=150.0,
                           name="A1"),
                RebarLayer(y=550.0, As=402.0, material=st, x=150.0,
                           name="A2")],
        tendons=[Tendon(y=100.0, Ap=1500.0, material=ps, x=150.0,
                        eps_pe=0.006, name="P1")],
        mesh_size=25.0,
    )


def _state(section, *, bulk_active=None, active=None):
    n_union = len(section.rebars) + len(getattr(section, "tendons", []) or [])
    act = (np.ones(n_union, dtype=bool) if active is None
           else np.asarray(active, dtype=bool))
    return SectionState(
        0, act, np.ones(n_union, dtype=bool),
        np.zeros(n_union, dtype=float),
        bulk_active=(None if bulk_active is None
                     else np.asarray(bulk_active, dtype=bool)),
    )


def _zone_E(view):
    """Read the substituted moduli back out of a view (rule 2)."""
    return ([float(view.bulk_material.E)]
            + [float(m.E) for _p, m in
               getattr(view, "bulk_materials", None) or []])


# ==================================================================
#  D-1 — zone / material axis collapse
# ==================================================================

class TestSharedMaterialZones(unittest.TestCase):
    r"""The fixture the C5 arc never had."""

    def test_shared_object_zones_keep_own_moduli(self):
        # This is 12_0d M7: fed 36283 / 22000, the material-keyed map
        # read 22000 on both, -39.4 % on the substrate.
        sec, _c, _s = _shared_material_section()
        mm = resolve_sls_moduli(sec, {("zone", 0): E_SUB,
                                      ("zone", 1): E_TOP})
        got = _zone_E(sls_view(sec, mm))
        self.assertAlmostEqual(got[0], E_SUB, delta=1e-9)
        self.assertAlmostEqual(got[1], E_TOP, delta=1e-9)

    def test_zone_slot_by_name_and_by_index_agree(self):
        sec, _c, _s = _shared_material_section()
        by_index = _zone_E(sls_view(sec, resolve_sls_moduli(
            sec, {("zone", 0): E_SUB, ("zone", 1): E_TOP})))
        by_name = _zone_E(sls_view(sec, resolve_sls_moduli(
            sec, {("zone", "base"): E_SUB, ("zone", "topping"): E_TOP})))
        self.assertEqual(by_index, by_name)

    def test_one_law_per_name_and_modulus(self):
        # Dedup contract: (original name, E), not object identity.
        sec, _c, _s = _shared_material_section()
        split = sls_view(sec, resolve_sls_moduli(
            sec, {("zone", 0): E_SUB, ("zone", 1): E_TOP}))
        self.assertIsNot(split.bulk_material, split.bulk_materials[0][1])
        merged = sls_view(sec, resolve_sls_moduli(sec, {"C45": E_SUB}))
        self.assertIs(merged.bulk_material, merged.bulk_materials[0][1])
        # Two rebars of one steel still share one substituted law, so
        # the solver's identity grouping is preserved.
        self.assertIs(split.rebars[0].material, split.rebars[1].material)

    def test_placeholder_stays_in_its_own_zone(self):
        # 12_0d M8/D-2: with a shared object the ascending walk wrote
        # the 1 MPa placeholder over the substrate — a factor 36 283.
        sec, _c, _s = _shared_material_section()
        mm = resolve_sls_moduli(sec, {("zone", 0): E_SUB,
                                      ("zone", 1): PLACEHOLDER})
        got = _zone_E(sls_view(sec, mm))
        self.assertAlmostEqual(got[0], E_SUB, delta=1e-9)
        self.assertAlmostEqual(got[1], PLACEHOLDER, delta=1e-12)

    def test_zone_modulus_reaches_the_staged_engine(self):
        # Wiring, not algebra: the map must reach verify_sls_staged's
        # zone_E, so a different topping modulus must move an answer.
        sec, _c, _s = _shared_material_section()
        st = _state(sec)
        out = []
        for e_top in (E_TOP, 0.5 * E_TOP):
            res = verify_sls_staged(
                sec,
                [{"name": "s0", "state": st,
                  "increment": (-4.0e5, -6.0e7, 0.0)}],
                moduli={("zone", 0): E_SUB, ("zone", 1): e_top})
            out.append(
                float(res["stages"][0]["concrete"]["sigma_min_MPa"]))
        self.assertNotAlmostEqual(out[0], out[1], places=5)


# ==================================================================
#  Inertia — the fixture that must not move
# ==================================================================

class TestDistinctMaterialInertia(unittest.TestCase):

    def test_distinct_objects_unchanged_by_key_form(self):
        sec, c0, c1, _s = _distinct_material_section()
        by_name = _zone_E(sls_view(sec, resolve_sls_moduli(
            sec, {c0.name: E_SUB, c1.name: E_TOP})))
        by_slot = _zone_E(sls_view(sec, resolve_sls_moduli(
            sec, {("zone", 0): E_SUB, ("zone", 1): E_TOP})))
        self.assertEqual(by_name, by_slot)
        self.assertAlmostEqual(by_name[0], E_SUB, delta=1e-9)
        self.assertAlmostEqual(by_name[1], E_TOP, delta=1e-9)

    def test_intrinsic_steel_modulus_is_not_an_override(self):
        sec = _tendon_section()
        mm = resolve_sls_moduli(sec, {"C35": 32000.0})
        self.assertEqual(float(mm.E_elem[0]), ES)
        self.assertEqual(float(mm.E_elem[1]), ES)
        self.assertEqual(float(mm.E_elem[2]), EP)
        self.assertEqual(mm.labels_elem, ("A1", "A2", "P1"))


# ==================================================================
#  Positional consumption through _union_index
# ==================================================================

class TestPositionalConsumption(unittest.TestCase):

    def test_masked_state_reads_the_element_own_slot(self):
        # The sharp case: with element 0 out of the view, view element
        # 0 is union element 1 and must read *its* modulus.
        sec = _tendon_section()
        mm = resolve_moduli(
            sec, SlsPolicy(),
            explicit={"C35": 32000.0, ("elem", 0): 100000.0,
                      ("elem", 1): 210000.0, ("tendon", "P1"): 190000.0})
        view = materialize_view(sec, _state(sec, active=[False, True, True]))
        vw = sls_view(view, mm)
        self.assertEqual(len(vw.rebars), 1)
        self.assertAlmostEqual(float(vw.rebars[0].material.E), 210000.0)
        self.assertAlmostEqual(float(vw.tendons[0].material.E), 190000.0)

    def test_element_keys_by_name_index_and_union_index_agree(self):
        sec = _tendon_section()
        base = {"C35": 32000.0}
        a = resolve_sls_moduli(sec, {**base, ("rebar", "A2"): 210000.0})
        b = resolve_sls_moduli(sec, {**base, ("rebar", 1): 210000.0})
        c = resolve_sls_moduli(sec, {**base, ("elem", 1): 210000.0})
        for mm in (a, b, c):
            self.assertEqual(float(mm.E_elem[1]), 210000.0)
            self.assertEqual(float(mm.E_elem[0]), ES)

    def test_tendon_local_index_is_not_the_union_index(self):
        sec = _tendon_section()
        mm = resolve_sls_moduli(sec, {"C35": 32000.0,
                                      ("tendon", 0): 190000.0})
        self.assertEqual(float(mm.E_elem[2]), 190000.0)


# ==================================================================
#  The retired key forms, and the contract that replaced them
# ==================================================================

class TestKeyContract(unittest.TestCase):

    def test_bare_int_is_rejected_and_points_at_the_slot_key(self):
        sec, conc, _s = _shared_material_section()
        with self.assertRaises(ValueError) as ctx:
            resolve_sls_moduli(sec, {id(conc): E_SUB})
        msg = str(ctx.exception)
        self.assertIn("('zone'", msg)
        self.assertIn("bare integer", msg)

    def test_material_object_key_is_unrepresentable_or_rejected(self):
        # A dataclass material is unhashable by contract: the mapping
        # cannot even be built.  A hashable outlier (a plain-class
        # material, as TabulatedMaterial is) reaches the resolver and
        # is refused there — unhashability alone does not close the
        # pattern.
        sec, conc, _s = _shared_material_section()
        with self.assertRaises(TypeError):
            {conc: E_SUB}

        class _Hashable:
            name = "exotic"

        with self.assertRaises(ValueError) as ctx:
            resolve_sls_moduli(sec, {_Hashable(): E_SUB})
        self.assertIn("not a key", str(ctx.exception))

    def test_dead_override_raises(self):
        sec, _c, _s = _shared_material_section()
        with self.assertRaises(ValueError):
            resolve_sls_moduli(sec, {"C45": E_SUB, "Ghost": 1.0e4})

    def test_fully_shadowed_name_key_is_dead(self):
        # The sugar reads as being in force on both zones and is in
        # force on neither: a silent-error trap, so it raises.
        sec, _c, _s = _shared_material_section()
        with self.assertRaises(ValueError):
            resolve_sls_moduli(sec, {"C45": E_SUB,
                                     ("zone", 0): E_SUB,
                                     ("zone", 1): E_TOP})

    def test_slot_key_wins_over_the_name_sugar(self):
        sec, _c, _s = _shared_material_section()
        got = _zone_E(sls_view(sec, resolve_sls_moduli(
            sec, {"C45": E_SUB, ("zone", 1): E_TOP})))
        self.assertAlmostEqual(got[0], E_SUB, delta=1e-9)
        self.assertAlmostEqual(got[1], E_TOP, delta=1e-9)

    def test_two_slot_keys_on_one_slot_raise(self):
        sec = _tendon_section()
        with self.assertRaises(ValueError):
            resolve_sls_moduli(sec, {"C35": 32000.0,
                                     ("rebar", "A1"): 210000.0,
                                     ("elem", 0): 205000.0})

    def test_unknown_zone_name_and_out_of_range_index_raise(self):
        sec, _c, _s = _shared_material_section()
        with self.assertRaises(ValueError):
            resolve_sls_moduli(sec, {("zone", "deck"): E_SUB})
        with self.assertRaises(ValueError):
            resolve_sls_moduli(sec, {("zone", 7): E_SUB})
        with self.assertRaises(ValueError):
            resolve_sls_moduli(sec, {("chunk", 0): E_SUB})

    def test_boolean_reference_rejected(self):
        sec, _c, _s = _shared_material_section()
        with self.assertRaises(ValueError):
            resolve_sls_moduli(sec, {("zone", True): E_SUB})

    def test_synthetic_label_is_not_a_reference(self):
        sec, _c, _s = _shared_material_section(named_elements=False)
        mm = resolve_sls_moduli(sec, {"C45": E_SUB})
        self.assertEqual(mm.labels_elem, ("rebar_0", "rebar_1"))
        with self.assertRaises(ValueError):
            resolve_sls_moduli(sec, {"C45": E_SUB,
                                     ("rebar", "rebar_0"): ES})

    def test_missing_concrete_modulus_lists_the_slot(self):
        sec, _c, _s = _shared_material_section()
        with self.assertRaises(ValueError) as ctx:
            resolve_sls_moduli(sec, {("zone", 0): E_SUB})
        self.assertIn("topping", str(ctx.exception))

    def test_covered_zone_slots_sees_every_key_form(self):
        sec, _c, _s = _shared_material_section()
        self.assertEqual(covered_zone_slots(sec, {"C45": E_SUB}), {0, 1})
        self.assertEqual(covered_zone_slots(sec, {("zone", 1): E_TOP}), {1})
        self.assertEqual(covered_zone_slots(sec, None), set())


# ==================================================================
#  The container itself
# ==================================================================

class TestMaterialMapContract(unittest.TestCase):

    def test_shape_mismatch_raises(self):
        # This is the assertion that makes D-1 unrepresentable: a map
        # with one entry for two zones cannot be built at all.
        with self.assertRaises(ValueError):
            MaterialMap(bulk=np.array([E_SUB]), elem=np.zeros(0),
                       labels_bulk=("base", "topping"), labels_elem=())
        with self.assertRaises(ValueError):
            MaterialMap(bulk=np.array([[E_SUB, E_TOP]]), elem=np.zeros(0),
                       labels_bulk=("base",), labels_elem=())

    def test_non_positive_and_non_finite_raise(self):
        for bad in (0.0, -1.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                MaterialMap(bulk=np.array([bad]), elem=np.zeros(0),
                           labels_bulk=("base",), labels_elem=())

    def test_arrays_are_read_only(self):
        r"""
        Both axes refuse mutation, by different mechanisms and for the
        same reason.

        The materials are held in a tuple, which cannot be assigned
        into at all; the derived moduli are a NumPy array with its
        write flag cleared.  "Frozen" must not be a promise the object
        breaks through whichever field it happens to hold by reference.
        """
        sec, _c, _s = _shared_material_section()
        mm = resolve_sls_moduli(sec, {("zone", 0): E_SUB,
                                      ("zone", 1): E_TOP})
        with self.assertRaises(TypeError):
            mm.bulk[0] = 1.0
        with self.assertRaises(ValueError):
            mm.E_bulk[0] = 1.0
        self.assertEqual(mm.n_zones, 2)
        self.assertEqual(mm.n_union, 2)

    def test_require_cover_fails_loud(self):
        sec, _c, _s = _shared_material_section()
        mm = resolve_sls_moduli(sec, {("zone", 0): E_SUB,
                                      ("zone", 1): E_TOP})
        with self.assertRaises(ValueError):
            mm.require_cover(3, 2, where="test")
        with self.assertRaises(ValueError):
            mm.require_cover(2, 5, where="test")
        mm.require_cover(2, 1, where="test")     # a view: fewer elements

    def test_a_policy_supplies_what_the_overrides_do_not(self):
        class _Fixed(ModulusPolicy):
            def bulk_material(self, slot):
                return E_SUB if slot.index == 0 else E_TOP

            def elem_material(self, slot):
                return ES

        sec, _c, _s = _shared_material_section()
        mm = resolve_moduli(sec, _Fixed())
        self.assertAlmostEqual(float(mm.E_bulk[0]), E_SUB)
        self.assertAlmostEqual(float(mm.E_bulk[1]), E_TOP)
        self.assertEqual(mm.labels_bulk, ("base", "topping"))

    def test_view_substitution_does_not_mutate_the_base(self):
        sec, conc, _s = _shared_material_section()
        vw = sls_view(sec, resolve_sls_moduli(sec, {("zone", 0): E_SUB,
                                                    ("zone", 1): E_TOP}))
        self.assertIsInstance(sec.bulk_material, Concrete)
        self.assertIsInstance(vw.bulk_material, LinearElastic)
        self.assertIs(sec.bulk_materials[0][1], conc)
        # Since Phase 27 elements are handed through unchanged rather
        # than replaced: the view holds the object the section declared.
        self.assertIs(sec.rebars[0].material, vw.rebars[0].material)


if __name__ == "__main__":
    unittest.main()
