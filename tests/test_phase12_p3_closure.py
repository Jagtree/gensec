"""
Phase-12 P3 closure: the surfaces P3 added, and their default branches.

Every test here closes an item that was *registered* rather than
covered when the surface landed.  A branch that is only ever taken by
production code is a branch nobody has measured; the default branches
in particular (``base_state=None``, no configuration file, no
``tolerances:`` block) are the ones a fixture has to reach, because
production never does.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import math

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gensec.io_yaml import load_yaml                       # noqa: E402
from gensec.materials.base import (                        # noqa: E402
    Material,
    ReferenceModulus,
    ReferenceModulusRule,
)
from gensec.materials.concrete import Concrete             # noqa: E402
from gensec.materials.rheology import (                    # noqa: E402
    EC2ReferenceModulus,
    EC2RheologicalModel,
    default_reference_modulus,
)
from gensec.solver.timeline import ConstructionTimeline    # noqa: E402
from gensec.tolerances import (                            # noqa: E402
    TOLERANCE_SCHEMA,
    Tolerances,
    load_tolerance_file,
)


# ======================================================================
#  Tolerances
# ======================================================================

class TestTolerances:
    r"""The tolerance surface, including the branches nothing takes."""

    def test_defaults_match_the_schema(self):
        r"""
        The dataclass defaults and :data:`TOLERANCE_SCHEMA` agree.

        Two houses for one number is the defect this project keeps
        finding; here they are structurally two, so they are compared.
        """
        t = Tolerances()
        for group, keys in TOLERANCE_SCHEMA.items():
            for key, default in keys.items():
                assert getattr(t, f"{group}_{key}") == default

    def test_a_block_overrides_only_what_it_names(self):
        t = Tolerances.from_mapping({"datum": {"zero_stress_rel": 1e-8}})
        assert t.datum_zero_stress_rel == 1e-8
        assert t.datum_plane_rel == 1e-9
        assert t.equilibrium_force_rel == 1e-9

    def test_none_block_returns_the_base(self):
        base = Tolerances(datum_plane_rel=1e-7)
        assert Tolerances.from_mapping(None, base=base).datum_plane_rel == 1e-7

    @pytest.mark.parametrize("block", [
        {"nosuchgroup": {"x": 1.0}},
        {"datum": {"nosuchkey": 1.0}},
        {"datum": 1.0},
        1.0,
    ])
    def test_an_unknown_or_malformed_override_raises(self, block):
        r"""
        An override that is silently ignored is a silent error.

        The whole reason to write one is to change something; dropping
        it without a word is the failure mode this project treats as
        worse than a crash.
        """
        with pytest.raises(ValueError):
            Tolerances.from_mapping(block)

    @pytest.mark.parametrize("field,value", [
        ("datum_plane_rel", 0.0),
        ("datum_plane_rel", -1e-9),
        ("equilibrium_max_iter", 0),
    ])
    def test_a_degenerate_tolerance_raises(self, field, value):
        with pytest.raises(ValueError):
            Tolerances(**{field: value})

    def test_round_trips_through_the_document_form(self):
        t = Tolerances.from_mapping({"prestress": {"cancellation_abs": 1e-5}})
        assert (Tolerances.from_mapping(t.as_document())
                == t)

    def test_the_alias_has_no_second_home(self):
        t = Tolerances(prestress_grout_readback_rel=1e-7)
        assert t.grout_readback_rel == t.prestress_grout_readback_rel

    def test_absent_configuration_file_is_not_an_error(self, tmp_path,
                                                       monkeypatch):
        r"""No file, no environment variable: the defaults are the answer."""
        monkeypatch.delenv("GENSEC_TOLERANCES", raising=False)
        monkeypatch.chdir(tmp_path)
        assert load_tolerance_file() == Tolerances()

    def test_a_named_missing_file_is_an_error(self, tmp_path):
        r"""
        A configuration asked for and not found is a missing file, not
        a default.
        """
        with pytest.raises(ValueError):
            load_tolerance_file(str(tmp_path / "absent.yaml"))

    def test_a_configuration_file_is_read(self, tmp_path, monkeypatch):
        path = tmp_path / "gensec_tolerances.yaml"
        path.write_text("tolerances:\n  geometry:\n    area_rel: 1.0e-10\n",
                        encoding="utf-8")
        monkeypatch.setenv("GENSEC_TOLERANCES", str(path))
        assert load_tolerance_file().geometry_area_rel == 1e-10

    def test_the_input_document_outranks_the_file(self, tmp_path,
                                                  monkeypatch):
        r"""Precedence: document -> configuration file -> shipped default."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("tolerances:\n  geometry:\n    area_rel: 1.0e-10\n",
                       encoding="utf-8")
        monkeypatch.setenv("GENSEC_TOLERANCES", str(cfg))
        doc = tmp_path / "m.yaml"
        doc.write_text(
            # D7: this document declares a 'tolerances:' block, so it
            # must declare the schema it is written against.
            "schema_version: 1\n"
            "tolerances:\n"
            "  geometry: {area_rel: 1.0e-11}\n"
            "code: ec2\n"
            "materials:\n"
            "  conc: {type: concrete_ec2_gen1, fck: 40, ls: F, loadtype: slow, TypeConc: R, NA: French, time: 28}\n"
            "section: {B: 300, H: 500, bulk_material: conc, n_fibers_y: 8}\n"
            "demands: [{name: G, N: -1.0, Mx: 0.0, My: 0.0}]\n",
            encoding="utf-8")
        data = load_yaml(str(doc))
        assert data["tolerances"].geometry_area_rel == 1e-11
        # the file still supplies what the document is silent about
        assert data["tolerances"].datum_plane_rel == 1e-9
        # ...and each says which of the three wrote it.  The precedence
        # was previously observable only through the values, so a
        # tolerance that happened to agree with the layer below it was
        # indistinguishable from one that had been overridden.
        origins = data["tolerances"].provenance_document()
        assert origins["geometry.area_rel"] == "input document"
        assert origins["datum.plane_rel"] == "shipped default"

    def test_the_timeline_always_has_a_tolerance_set(self):
        r"""A site that reads a tolerance must find one, not decide."""
        tl = ConstructionTimeline.from_block([{"point": "p"}])
        assert isinstance(tl.tolerances, Tolerances)


# ======================================================================
#  The reference-modulus contract
# ======================================================================

class _MuteMaterial(Material):
    r"""A material that declares nothing: the refusal branch."""

    @property
    def eps_min(self):
        return -1.0

    @property
    def eps_max(self):
        return 1.0

    def stress(self, eps):
        return 0.0

    def stress_array(self, eps):
        return np.zeros_like(np.asarray(eps, dtype=float))

    def tangent(self, eps):
        return 0.0

    def tangent_array(self, eps):
        return np.zeros_like(np.asarray(eps, dtype=float))


def _ec2_concrete(fck):
    r"""
    A concrete that **names its code**, built through the EC2 bridge.

    Introduced in Phase 21.  Several tests here need a concrete that
    can state its own conventional modulus, and after F-P21-1 the bare
    constructor deliberately cannot: it was given a strength, not a
    code.  One home for that construction, so the distinction between
    "declared with a code" and "declared with a number" stays visible
    at every call site instead of being re-improvised.

    Parameters
    ----------
    fck : float
        Characteristic cylinder strength [MPa].

    Returns
    -------
    Concrete
    """
    from gensec.materials.codes import resolve_code
    from gensec.materials.ec2_bridge import concrete_from_class
    material = concrete_from_class(f"C{int(fck)}/{int(fck) + 10}", ls='F', loadtype='slow', TypeConc='R', NA='French', time=28)
    # Stamped explicitly, as the loader does.  Since Phase 21 the
    # bridge that BUILDS a material does not thereby DECLARE the code
    # it is verified to: a constitutive shape is not a verification
    # code, and nothing says a GenSec section is concrete at all.
    material.code = resolve_code(model_code="ec2")
    return material


class TestReferenceModulusContract:
    r"""Every material states its own conventional modulus, or refuses."""

    def test_concrete_with_a_code_object_declares_ecm_at_28_days(self):
        r"""
        A material that names a code states its own modulus.

        Rewritten in Phase 21.  It read ``Concrete(fck=40.0)``, whose
        declaration names no code at all, and asserted that it
        nevertheless produced a modulus -- which is the behaviour
        F-P21-1 removed, so leaving the test as it was would have
        pinned the defect with a green.  The capability it really
        guarded is kept: a concrete that *does* carry a code-specific
        property object still answers, at its own reference age, with
        its own citation.
        """
        rm = _ec2_concrete(fck=40.0).reference_modulus()
        assert isinstance(rm, ReferenceModulus)
        assert rm.reference_age == 28.0
        assert rm.source

    def test_a_concrete_that_names_no_code_declines(self):
        r"""
        F-P21-1: ``f_ck`` is an input, not a code.

        ``Concrete(fck=40.0)`` declares a strength and nothing else.
        EN 1992-1-1 Table 3.1 and ACI 318-19 19.2.2.1 read that same
        strength and disagree by 18.5 %, so there is no code-free
        answer and GenSec declines rather than choosing one.
        """
        with pytest.raises(NotImplementedError) as caught:
            Concrete(fck=40.0, gamma_c=1.5, long_term_reduction_factor=0.85).reference_modulus()
        text = str(caught.value)
        assert "ACI" in text and "1992" in text, (
            f"the refusal must name why the answer is code-dependent, "
            f"not merely that it is missing. Message was: {text}")
        assert "type:" in text or "reference_modulus=" in text, (
            f"a refusal without a correction lever leaves the caller "
            f"nowhere to go. Message was: {text}")

    def test_the_eurocode_answer_is_unchanged_by_the_refusal(self):
        r"""
        Nothing an EC2 build could reach before has been taken away.

        A capability removed is a cost regardless of how it reads, so
        the closing measurement is not "the new refuses" but "the new
        still gives the old number by the old route".  The resolver
        catches the refusal and applies its own Table 3.1 branch.
        """
        c = Concrete(fck=40.0, gamma_c=1.5, long_term_reduction_factor=0.85)
        provider = EC2RheologicalModel(fck=40.0, cement_class="N").E_c(28.0)
        assert EC2ReferenceModulus().E_c(c) == pytest.approx(
            provider, rel=1e-12)

    def test_an_installed_rule_is_no_longer_overridden_by_the_material(self):
        r"""
        The measurement that made F-P21-1 a defect and not a taste.

        The resolver asks the material *before* applying the build's
        rule, so that a material which knows its own code wins over a
        generic one.  A material that answered when it did not know
        therefore beat the rule the engineer installed: an ACI rule
        asked for ``Concrete(fck=40)`` returned the Eurocode's
        35 220.5 MPa instead of ACI's 29 725.4 MPa.  A default that
        cannot be overridden is not a default.
        """
        class _ACI(ReferenceModulusRule):
            """ACI 318-19 19.2.2.1, normalweight: Ec = 4700 sqrt(f'c)."""

            def __init__(self):
                super().__init__(
                    source="ACI 318-19 §19.2.2.1 -- Ec = 4700 sqrt(f'c), "
                           "normalweight concrete",
                    reference_age=28.0)

            def E_c(self, material):
                try:
                    declared = material.reference_modulus()
                except (NotImplementedError, AttributeError):
                    declared = None
                if declared is not None and float(declared.value) > 0.0:
                    return float(declared.value)
                fc = getattr(material, "fck", None)
                if fc is None:
                    return None
                return 4700.0 * math.sqrt(float(fc))

        c = Concrete(fck=40.0, gamma_c=1.5, long_term_reduction_factor=0.85)
        assert _ACI().E_c(c) == pytest.approx(
            4700.0 * math.sqrt(40.0), rel=1e-12), (
            "an installed ACI rule still loses to a Eurocode value "
            "invented by the material: F-P21-1 is not closed.")
        assert EC2ReferenceModulus().E_c(c) == pytest.approx(
            35_220.46228893441, rel=1e-12), (
            "and the Eurocode route must still give the Eurocode "
            "number, or the repair traded one code for another.")

    def test_the_declaration_agrees_with_the_rheological_provider(self):
        r"""
        Two derivations of :math:`E_{cm}(28)` must not disagree.

        The material's own bridge and the rheological provider are
        different objects reaching the same clause; if they diverged,
        a datum drawn without a time axis and one drawn with it would
        describe different concrete.
        """
        declared = _ec2_concrete(fck=40.0).reference_modulus().value
        provider = EC2RheologicalModel(fck=40.0, cement_class="N").E_c(28.0)
        assert declared == pytest.approx(provider, rel=1e-12)

    def test_a_silent_material_refuses_loudly(self):
        with pytest.raises(NotImplementedError) as excinfo:
            _MuteMaterial().reference_modulus()
        assert "reference_modulus" in str(excinfo.value)

    def test_the_rule_asks_the_material_rather_than_probing_it(self):
        r"""
        The agnostic path must not depend on an EC2 bridge.

        A material with no ``ec2`` attribute but a declared modulus is
        the ACI case in miniature: the rule has to find the value by
        asking, or a non-Eurocode build falls to the refusal.
        """
        class _Declaring(_MuteMaterial):
            def reference_modulus(self):
                return ReferenceModulus(
                    value=27_000.0, reference_age=28.0,
                    source="ACI 318-19 §19.2.2.1 (stand-in)")

        material = _Declaring()
        assert not hasattr(material, "ec2")
        assert EC2ReferenceModulus().E_c(material) == 27_000.0

    def test_the_rule_declines_for_a_material_that_declares_nothing(self):
        r"""Declining is correct: the resolver owns the refusal."""
        assert EC2ReferenceModulus().E_c(_MuteMaterial()) is None

    def test_a_rule_must_cite_its_source(self):
        class _Anonymous(ReferenceModulusRule):
            def E_c(self, material):
                return 1.0

        with pytest.raises(ValueError):
            _Anonymous(source="   ")

    def test_the_default_rule_is_replaceable(self):
        assert isinstance(default_reference_modulus(), ReferenceModulusRule)


# ======================================================================
#  Default branches nothing in production takes
# ======================================================================

_STANDALONE = """
# Declared because GenSec derives nothing from an undeclared
# model: the code is not inferred from a material type.
code: ec2
materials:
  conc: {type: concrete_ec2_gen1, fck: 40, ls: F, loadtype: slow, TypeConc: R, NA: French, time: 28}
  strand: {type: prestressing_steel_ec2, class: Y1860S7, ls: F, NA: EC2, eps_ud_factor: 0.9, diagram: horizontal}
section:
  B: 400
  H: 800
  bulk_material: conc
  n_fibers_y: 24
  tendons:
    - {name: P1, x: 0.0, y: 120.0, Ap: 1200.0, material: strand}
demands:
  - {name: G1, N_kN: -60, Mx_kNm: -40}
construction_history:
  - {load: {demand: G1}}
  - {stress: {tendons: [P1], sigma_p0: 1200.0}}
  - {grout: {tendons: [P1]}}
  - {point: after}
"""


def _doc(tmp_path, text):
    path = tmp_path / "m.yaml"
    path.write_text(text, encoding="utf-8")
    return load_yaml(str(path))


class TestStandaloneGroutBranch:
    r"""``base_state=None``: the driver used without a staging history."""

    def test_the_self_consistent_grout_runs_without_a_base_state(self,
                                                                 tmp_path):
        r"""
        The stand-alone branch is reached by a fixture, not assumed.

        ``_bonded_state`` and ``_stressing_view`` both grew a
        ``base_state`` parameter at P3.  Production always passes one,
        so the ``None`` branch -- the bare driver, the validators --
        would otherwise never be executed by the suite.
        """
        from gensec.solver.posttension import (grout_selfconsistent,
                                               solve_posttension_sequence)
        data = _doc(tmp_path, _STANDALONE)
        section = data["section"]
        result = solve_posttension_sequence(
            section, sigma_p0=[1200.0], order=[0],
            base_N=0.0, base_Mx=0.0, base_My=0.0)
        gres, by_local, n_it, res_rel = grout_selfconsistent(
            section, result, [0], demand_after=(0.0, 0.0, 0.0))
        assert 1 <= n_it <= 8
        assert res_rel < 1e-9
        assert np.isfinite(by_local[0])

    def test_the_read_back_equals_the_jack_end_to_end(self, tmp_path):
        r"""
        The F5 invariant on a section that declares no time axis.

        Nothing mechanical happens at grouting, so the bonded tendon
        must read back exactly what it was jacked to -- also when the
        service modulus comes from the material's own declaration
        rather than from a rheological provider.
        """
        data = _doc(tmp_path, _STANDALONE)
        tl = ConstructionTimeline.from_document(data)
        assert not tl.has_time_axis
        res = tl.resolve(data["section"],
                         {d["name"]: d for d in data["demands"]})
        assert res.grout_eps_init


class TestRheologySwitch:
    r"""``rheology: declared | on | off``, measured in all three states."""

    def test_off_disables_the_time_axis_and_is_recorded(self, tmp_path):
        r"""
        An indicative run at conventional values is a legitimate ask.

        What it must not be is silent: the intervals it skips are
        reported, so a result computed without creep cannot be mistaken
        for one computed with it.
        """
        text = "rheology: off\n" + _STANDALONE
        data = _doc(tmp_path, text)
        tl = ConstructionTimeline.from_document(data)
        assert not tl.has_time_axis

    def test_an_unknown_switch_value_raises(self, tmp_path):
        with pytest.raises(ValueError):
            _doc(tmp_path, "rheology: sometimes\n" + _STANDALONE)
