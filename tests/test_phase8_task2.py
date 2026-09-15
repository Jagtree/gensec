# SPDX-License-Identifier: AGPL-3.0-or-later
# GenSec — Phase-8 Task-2 (construction timeline) test suite.
# Copyright (C) 2026  Andrea (GenSec project).
r"""
Test suite for the Phase-8 Task-2 construction-timeline compiler.

Runs under both runners::

    python -m unittest tests.test_phase8_task2 -v
    pytest tests/test_phase8_task2.py -v

Covers the timeline object (parse + validation + fail-loud), the
auto-datum resolution walk (C3, checked against an independently
assembled substrate plane), the compiler (C1 factors + :math:`\gamma_P`,
R1 ``initially_inactive``), multi-point anchoring (C2), and the
governing-point reduction.  Prestress driver-reconciliation
(exact equivalence with ``solve_posttension_sequence``) and the
io_yaml / cli / api wiring are covered by the repository integration and
the full suite, not here (recap ``10_4`` §status).
"""

import io
import os
import unittest
from contextlib import redirect_stderr

import numpy as np

from gensec.io_yaml import load_yaml
from gensec.solver import FiberSolver, NMDiagram
from gensec.solver.check import VerificationEngine
from gensec.solver.section_state import (
    StagedDomainManager, _staging_parents,
)
from gensec.solver.timeline import (
    ConstructionTimeline, TimelineEvent, EVENT_KINDS,
    governing_point, _resolve_gamma_P,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _example(name):
    r"""
    The absolute path of a distributed example, or a loud failure.

    F-P00-1, closed in Phase 18.  This module used to hold two copies
    of::

        _X = os.path.join(_HERE, "..", "examples", "x.yaml")
        if not os.path.exists(_X):
            _X = "x.yaml"                       # silent fallback

    The fallback substituted a **bare relative name** when the example
    was not where it should be, so the file was then looked for in
    whatever the current working directory happened to be.  Run from
    the repository root it worked by coincidence; run from anywhere
    else it produced a ``FileNotFoundError`` naming ``x.yaml`` with no
    path, several frames away from the assumption that failed.  Both
    fallbacks were measured **alive** in Phase 18 (both examples
    present, so neither had ever fired), which is why they had survived
    five phases unnoticed: a substituting default that never
    substitutes is invisible until the day it does.

    The single home replaces both, and it fails where the assumption
    is, naming the path it looked at — the same rule the ``getattr``
    inventory enforces on the package.

    Parameters
    ----------
    name : str
        File name inside ``examples/``.

    Returns
    -------
    str
        Absolute path.

    Raises
    ------
    FileNotFoundError
        If the example is absent, naming the resolved path.
    """
    path = os.path.abspath(os.path.join(_HERE, "..", "examples", name))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"distributed example {name!r} not found at {path}. "
            f"The examples/ directory is part of the package, not an "
            f"optional extra: this test measures GenSec against the "
            f"documents it ships.")
    return path


_EXAMPLE = _example("example_composite_topping.yaml")
_PRESTRESS = _example("example_prestress.yaml")


def _zone_materials(section):
    r"""
    ``[material]`` per bulk zone, index-aligned with the state.

    ``section.bulk_materials`` holds ``(polygon, material)`` pairs; the
    material is the second element.

    Parameters
    ----------
    section : GenericSection

    Returns
    -------
    list of Material
    """
    out = [section.bulk_material]
    for entry in (getattr(section, "bulk_materials", []) or []):
        out.append(entry[1] if isinstance(entry, tuple) else entry)
    return out


def _reference_E(material):
    r"""
    The conventional 28-day modulus of *material*, assembled
    **independently** of the resolver.

    Rule 2: a reference must not read the implementation's own inputs.
    The shipped rule is
    :class:`~gensec.materials.rheology.EC2ReferenceModulus`, which
    reads the material's EC2 bridge; this reads
    :class:`~gensec.materials.rheology.EC2RheologicalModel` at
    :math:`t = 28` d instead -- a different class, a different code
    path, the same clause (EN 1992-1-1 Table 3.1).  The two agreeing is
    itself a result, asserted in
    :meth:`TestAutoDatum.test_the_two_assemblies_of_Ecm_agree`.

    Parameters
    ----------
    material : Material

    Returns
    -------
    float
        :math:`E_{cm}(28\,\mathrm{d})` [MPa].
    """
    from gensec.materials.rheology import EC2RheologicalModel
    return EC2RheologicalModel(fck=float(material.fck),
                               cement_class="N").E_c(28.0)


def _service_bundle(section, state):
    r"""
    A linear service view of *section* at *state*, built by hand.

    Assembled from :func:`~gensec.solver.sls.sls_view` over an
    explicit modulus per zone, so it shares no code with
    :meth:`ConstructionTimeline._datum_moduli` beyond the resolver
    every SLS path already uses.

    Parameters
    ----------
    section : GenericSection
    state : SectionState

    Returns
    -------
    FiberSolver
    """
    from gensec.solver.sls import SlsPolicy, resolve_moduli, sls_view
    from gensec.solver.section_state import materialize_view
    mats = _zone_materials(section)
    explicit = {("zone", z): _reference_E(m) for z, m in enumerate(mats)}
    modmap = resolve_moduli(section, SlsPolicy(), explicit=explicit)
    return FiberSolver(sls_view(materialize_view(section, state), modmap))


def _load():
    data = load_yaml(_EXAMPLE)
    ddb = {x["name"]: {"N": x.get("N", 0.0), "Mx": x.get("Mx", 0.0),
                       "My": x.get("My", 0.0)} for x in data["demands"]}
    return data["section"], ddb


class TestTimelineParse(unittest.TestCase):
    r"""Parsing and structural validation of the timeline block."""

    def test_event_kinds_frozen(self):
        r"""
        The event vocabulary is pinned, and grows only deliberately.

        ``retrofit`` joins in Phase 34 (D1b-3).  It is not an ordinary
        event: it declares the state the member is **already** in when
        the history opens, which is why
        :func:`~gensec.solver.timeline._parse_retrofit_event` refuses
        it anywhere but first.  It is listed here all the same,
        because this test exists to make a widening of the vocabulary
        visible rather than to say which kinds are events in the
        physical sense.
        """
        self.assertEqual(
            EVENT_KINDS,
            frozenset({"cast", "stress", "grout", "interval",
                       "load", "point", "retrofit"}))

    def test_point_scalar_and_mapping(self):
        tl = ConstructionTimeline.from_block(
            [{"point": "a"}, {"cast": {"zone": "topping"}},
             {"point": {"name": "b"}}])
        self.assertEqual(tl.points, {"a": 0, "b": 1})

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            ConstructionTimeline.from_block([{"pour": {"zone": "z"}}])

    def test_duplicate_point_raises(self):
        with self.assertRaises(ValueError):
            ConstructionTimeline.from_block([{"point": "p"}, {"point": "p"}])

    def test_interval_losses_is_implemented_in_c5(self):
        """C5 landed: 'interval' + 'losses' is no longer a placeholder.
        A bare model name is still refused -- a zone's rheology is
        meaningless without its age and its curing age -- but with
        ValueError now, not NotImplementedError."""
        with self.assertRaises(ValueError):
            ConstructionTimeline.from_block(
                [{"interval": {"days": 28, "losses": "creep"}}])
        tl = ConstructionTimeline.from_block(
            [{"interval": {"days": 28,
                           "losses": {"base": {"model": "m", "age": 28,
                                               "curing": 3}}}}])
        self.assertEqual(len(tl.events), 1)

    def test_bad_event_object_construction(self):
        with self.assertRaises(ValueError):
            TimelineEvent("frobnicate", {}, 0)


class TestTimelineValidate(unittest.TestCase):
    r"""Semantic validation against a built section."""

    def setUp(self):
        self.section, self.ddb = _load()

    def test_cast_base_zone_raises(self):
        tl = ConstructionTimeline.from_block([{"cast": {"zone": "base"}}])
        with self.assertRaises(ValueError):
            tl.validate(self.section)

    def test_cast_unknown_zone_raises(self):
        tl = ConstructionTimeline.from_block([{"cast": {"zone": "nope"}}])
        with self.assertRaises(ValueError):
            tl.validate(self.section)

    def test_unknown_demand_in_load_raises(self):
        tl = ConstructionTimeline.from_block([{"load": {"demand": "ZZ"}}])
        with self.assertRaises(ValueError):
            tl.resolve(self.section, self.ddb)

    def test_bad_datum_spec_raises(self):
        tl = ConstructionTimeline.from_block(
            [{"cast": {"zone": "topping", "datum": "guess"}}])
        with self.assertRaises(ValueError):
            tl.resolve(self.section, self.ddb)


class TestAutoDatum(unittest.TestCase):
    r"""C3 — auto datum equals the negated substrate plane (two ways)."""

    def setUp(self):
        self.section, self.ddb = _load()

    def _independent_datum(self):
        mgr = StagedDomainManager(self.section, biaxial=False,
                                  gen_kwargs={"n_points": 40})
        parents = _staging_parents(self.section)
        nonbase = [i for i in range(parents.size) if parents[i] == 1]
        states, _h, _b, _d = mgr.resolve_stages(
            [{"name": "g1",
              "components": [{"ref": "G1", "factor": 1.0}],
              "section_ops": {"deactivate": nonbase, "release": False}}],
            initially_inactive=[1])
        # P3: the datum is drawn on the SERVICE response, so the
        # independent assembly must be too.  Reading the design bundle
        # here would once again be a reference that shares the
        # implementation's error -- which is precisely how F5 survived
        # every phase before this one.
        g = self.ddb["G1"]
        s = _service_bundle(self.section, states[-1]).solve_equilibrium(
            g["N"], g["Mx"], g["My"], tol=1e-10, max_iter=100)
        return (-s["eps0"], -s["chi_x"], -s["chi_y"])

    def test_the_two_assemblies_of_Ecm_agree(self):
        r"""The shipped reference rule and the rheological provider must
        give the same 28-day modulus; if they diverge, the datum and
        the reference below are measuring two different concretes."""
        from gensec.materials.rheology import default_reference_modulus
        rule = default_reference_modulus()
        for mat in _zone_materials(self.section):
            self.assertAlmostEqual(rule.E_c(mat), _reference_E(mat),
                                   places=9)

    def test_auto_datum_matches_substrate_plane(self):
        tl = ConstructionTimeline.from_block([
            {"load": {"demand": "G1"}},
            {"cast": {"zone": "topping", "datum": "auto"}},
            {"point": "svc"}])
        res = tl.resolve(self.section, self.ddb)
        way1 = tuple(float(v) for v in res.datums[1])
        way2 = self._independent_datum()
        for a, b in zip(way1, way2):
            self.assertAlmostEqual(a, b, places=12)

    def test_explicit_datum_overrides_auto(self):
        triple = {"eps0": 1e-4, "chi_x": 2e-7, "chi_y": 0.0}
        tl = ConstructionTimeline.from_block([
            {"cast": {"zone": "topping", "datum": triple}}, {"point": "p"}])
        res = tl.resolve(self.section, self.ddb)
        self.assertEqual(res.datums[1], (1e-4, 2e-7, 0.0))
        self.assertIn(1, res.explicit_datums)


class TestCompiler(unittest.TestCase):
    r"""C1/C2/R1 — compiler output is valid engine input."""

    def setUp(self):
        self.section, self.ddb = _load()
        self.tl = ConstructionTimeline.from_block([
            {"load": {"demand": "G1"}},
            {"cast": {"zone": "topping", "datum": "auto"}},
            {"load": {"demand": "G2"}},
            {"point": "transfer"},
            {"point": "service"}])
        self.res = self.tl.resolve(self.section, self.ddb)

    def _engine(self):
        solver = FiberSolver(self.section)
        nm = NMDiagram(solver)
        nm_data = nm.generate(n_points=40)
        mgr = StagedDomainManager(self.section, biaxial=False,
                                  gen_kwargs={"n_points": 40})
        return VerificationEngine(
            nm_data, nm, {"eta_norm": True, "eta_norm_ray": True},
            n_points=40, staged_manager=mgr)

    def test_compiled_stages_run_on_engine(self):
        combo = {"name": "ULS", "at": "service",
                 "history_factors": {"G1": 1.35, "G2": 1.35},
                 "components": [{"ref": "Q", "factor": 1.5}]}
        compiled = self.tl.compile_combination(
            combo, self.res, self.section, self.ddb)
        self.assertEqual(len(compiled), 1)
        pt, stages, inact = compiled[0]
        engine = self._engine()
        r = engine.check_combination({"name": "ULS", "stages": stages},
                                     self.ddb)
        self.assertTrue(np.isfinite(r["eta_governing"]))

    def test_history_factor_applied(self):
        combo = {"name": "ULS", "at": "transfer",
                 "history_factors": {"G1": 1.35, "G2": 1.35},
                 "components": []}
        _pt, stages, _i = self.tl.compile_combination(
            combo, self.res, self.section, self.ddb)[0]
        engine = self._engine()
        r = engine.check_combination({"name": "ULS", "stages": stages},
                                     self.ddb)
        # first stage carries 1.35 * G1 (N = -300 kN)
        self.assertAlmostEqual(
            r["stages"][0]["cumulative"]["N_kN"], 1.35 * -300.0, places=6)

    def test_multi_point_anchoring(self):
        combo = {"name": "ULS", "at": ["transfer", "service"],
                 "history_factors": {"G1": 1.0, "G2": 1.0},
                 "components": [{"ref": "Q", "factor": 1.0}]}
        compiled = self.tl.compile_combination(
            combo, self.res, self.section, self.ddb)
        self.assertEqual([c[0] for c in compiled], ["transfer", "service"])

    def test_initially_inactive_for_post_anchor_cast(self):
        # anchor BEFORE the topping cast -> topping (zone 1) inactive
        tl = ConstructionTimeline.from_block([
            {"load": {"demand": "G1"}},
            {"point": "before_topping"},
            {"cast": {"zone": "topping", "datum": "auto"}},
            {"point": "after_topping"}])
        res = tl.resolve(self.section, self.ddb)
        compiled = tl.compile_combination(
            {"name": "c", "at": "before_topping",
             "history_factors": {"G1": 1.0}, "components": []},
            res, self.section, self.ddb)
        self.assertEqual(compiled[0][2], [1])  # zone 1 initially inactive

    def test_unknown_anchor_raises(self):
        with self.assertRaises(ValueError):
            self.tl.compile_combination(
                {"name": "c", "at": "nowhere", "components": []},
                self.res, self.section, self.ddb)

    def test_interleave_stress_after_grout_raises(self):
        tl = ConstructionTimeline.from_block([
            {"grout": {"tendons": ["Tx"]}},
            {"stress": {"tendons": ["Tx"]}},
            {"point": "p"}])
        with self.assertRaises(NotImplementedError):
            tl._check_interleave(tl.events, "c")


class TestFactorsAndReduction(unittest.TestCase):
    r"""C1 gamma_P, warn-on-omission, and C2 governing-point reduction."""

    def setUp(self):
        self.section, self.ddb = _load()

    def test_gamma_P_keywords_and_provider(self):
        self.assertEqual(_resolve_gamma_P("favourable", None), 1.0)
        self.assertEqual(_resolve_gamma_P("unfavourable", None), 1.3)
        self.assertEqual(_resolve_gamma_P(1.25, None), 1.25)
        self.assertEqual(
            _resolve_gamma_P("unfavourable", lambda k: 1.1), 1.1)

    def test_gamma_P_bad_keyword_raises(self):
        with self.assertRaises(ValueError):
            _resolve_gamma_P("sometimes", None)

    def test_warn_on_omitted_history_load(self):
        tl = ConstructionTimeline.from_block([
            {"load": {"demand": "G1"}},
            {"cast": {"zone": "topping", "datum": "auto"}},
            {"load": {"demand": "G2"}},
            {"point": "service"}])
        res = tl.resolve(self.section, self.ddb)
        buf = io.StringIO()
        with redirect_stderr(buf):
            tl.compile_combination(
                {"name": "c", "at": "service",
                 "history_factors": {"G1": 1.0}, "components": []},
                res, self.section, self.ddb)
        self.assertIn("history load", buf.getvalue())

    def test_no_warn_when_explicitly_listed(self):
        tl = ConstructionTimeline.from_block([
            {"load": {"demand": "G1"}},
            {"cast": {"zone": "topping", "datum": "auto"}},
            {"load": {"demand": "G2"}},
            {"point": "service"}])
        res = tl.resolve(self.section, self.ddb)
        buf = io.StringIO()
        with redirect_stderr(buf):
            tl.compile_combination(
                {"name": "c", "at": "service",
                 "history_factors": {"G1": 1.0, "G2": 1.0},
                 "components": []}, res, self.section, self.ddb)
        self.assertNotIn("history load", buf.getvalue())

    def test_governing_point_is_transparent_max(self):
        gov = governing_point({
            "t1": {"eta_governing": 0.7, "verified": True},
            "t2": {"eta_governing": 1.1, "verified": False},
            "t3": {"eta_governing": 0.9, "verified": True}})
        self.assertEqual(gov["governing_point"], "t2")
        self.assertEqual(gov["eta_governing"], 1.1)
        self.assertFalse(gov["verified"])


class TestPrestressReconciliation(unittest.TestCase):
    r"""Task-2 prestress closure: stress → grout reconciliation reuse."""

    def setUp(self):
        data = load_yaml(_PRESTRESS)
        self.section = data["section"]
        self.T = "tendon_0"
        self.sigma = 1400.0
        self.tl = ConstructionTimeline.from_block([
            {"stress": {"tendons": [self.T], "sigma_p0": self.sigma}},
            {"grout": {"tendons": [self.T]}},
            {"point": "after"}])
        self.res = self.tl.resolve(self.section, {})

    def _driver_eps_init(self):
        from gensec.solver.posttension import (
    grout_selfconsistent,
            solve_posttension_sequence, grout)
        from gensec.solver.timeline import _tendon_local_index
        li = _tendon_local_index(self.section, self.T)
        sig = [0.0] * len(self.section.tendons)
        sig[li] = self.sigma
        # P3: the timeline stresses against the SERVICE view, so the
        # bare driver call must be put on the same footing or the two
        # paths are no longer comparable.  The modulus is assembled
        # independently (``_reference_E``), not read back from the
        # timeline.
        from gensec.solver.sls import SlsPolicy, resolve_moduli
        modmap = resolve_moduli(
            self.section, SlsPolicy(),
            explicit={("zone", 0): _reference_E(
                self.section.bulk_material)})
        result = solve_posttension_sequence(
            self.section, sigma_p0=sig, order=[li],
            base_N=0.0, base_Mx=0.0, base_My=0.0, modmap=modmap)
        # P3/F5: the reference is the *self-consistent* grout, because
        # that is what the timeline now compiles.  Single-pass ``grout``
        # reads the concrete strain on a section in which the tendon is
        # still an external load, so its datum makes the bonded tendon
        # read back a stress that is not the jack -- an O(n*rho) residue
        # (measured -0.772 MPa at Ap=1500 on a 400x800).  Comparing the
        # compiled path against the single pass would assert equivalence
        # to the defect.
        _gres, by_local, _it, _res = grout_selfconsistent(
            self.section, result, [li],
            demand_after=(0.0, 0.0, 0.0), modmap=modmap)
        return by_local[li]

    def test_reconciled_eps_init_byte_equivalent_to_driver(self):
        self.assertAlmostEqual(
            self.res.grout_eps_init[self.T], self._driver_eps_init(),
            places=15)

    def test_compiled_grout_state_active_bonded_at_strain(self):
        from gensec.solver.timeline import _tendon_union_index
        ui = _tendon_union_index(self.section, self.T)
        _pt, stages, _i = self.tl.compile_combination(
            {"name": "c", "at": "after", "components": []},
            self.res, self.section, {})[0]
        mgr = StagedDomainManager(self.section, biaxial=False,
                                  gen_kwargs={"n_points": 40})
        states, _h, _b, _d = mgr.resolve_stages(stages)
        f = states[-1]
        self.assertEqual(int(f.active[ui]), 1)
        self.assertEqual(int(f.bonded[ui]), 1)
        self.assertAlmostEqual(f.eps_init[ui], self._driver_eps_init(),
                               places=15)

    def test_single_side_invariant_demand_nets_to_zero(self):
        _pt, stages, _i = self.tl.compile_combination(
            {"name": "c", "at": "after", "components": []},
            self.res, self.section, {})[0]
        mgr = StagedDomainManager(self.section, biaxial=False,
                                  gen_kwargs={"n_points": 40})
        nm = NMDiagram(FiberSolver(self.section))
        eng = VerificationEngine(nm.generate(n_points=40), nm,
                                 {"eta_norm": True}, n_points=40,
                                 staged_manager=mgr)
        r = eng.check_combination({"name": "c", "stages": stages}, {})
        cum = r["stages"][-1]["cumulative"]
        self.assertAlmostEqual(cum["N_kN"], 0.0, places=6)
        self.assertAlmostEqual(cum["Mx_kNm"], 0.0, places=6)
        self.assertAlmostEqual(cum["My_kNm"], 0.0, places=6)

    def test_grout_without_stress_raises(self):
        tl = ConstructionTimeline.from_block([
            {"grout": {"tendons": [self.T]}}, {"point": "p"}])
        with self.assertRaises(ValueError):
            tl.resolve(self.section, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
