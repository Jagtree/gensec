# SPDX-License-Identifier: AGPL-3.0-or-later
# GenSec — reinforced/prestressed concrete sectional analysis.
# Copyright (C) 2026  Andrea (GenSec project).
r"""
Phase 21 — F-P21-1: the input declares the code, or declares custom.

The rule
--------
**The input declares the normative code, and everything is derived from
it.  Or the input declares that the material is custom, and then GenSec
derives nothing.**  There is no third option, and in particular no
*implicit* one.

What was wrong
--------------
The container was agnostic and the providers were replaceable, but the
**declaration** was missing, so each derived quantity resolved its own
convention locally.  Where a material could derive a value from what it
happened to carry, it did:
``Concrete(fck=40).reference_modulus()`` returned EN 1992-1-1's
35 220.5 MPa for a material whose declaration named no code at all.

Two measurements made that a defect rather than a taste:

* the same :math:`f_{ck}` under ACI 318-19 §19.2.2.1 is **33.9 %** away
  at C25, **18.5 %** at C40 and **7.4 %** at C60 --- so there is no
  reading of "the modulus of a C40" that is true under both;
* the resolver asks the material **before** applying the build's rule,
  by design, so that a material which knows its own code beats a
  generic one.  A material that answered when it did *not* know
  therefore beat the rule the engineer installed: an ACI rule asked for
  ``Concrete(fck=40)`` returned 35 220.5 and not 29 725.4.  **A default
  that cannot be overridden is not a default.**

The shape of the fixtures
-------------------------
``TestTheCodesDisagree``
    **Guard**, green on both trees.  It asserts a property of the two
    *clauses*, not of GenSec, and it is what makes the rest of this
    file worth having: if EN 1992-1-1 and ACI 318-19 agreed, declaring
    which one applies would be bookkeeping instead of engineering.

``TestTheDeclarationGoverns``
    **Discriminating.**  Red on the pristine Phase 20 tree.  Every red
    is an :class:`AssertionError` or a missing-key
    :class:`KeyError`-free failure of behaviour; nothing here imports a
    Phase 21 symbol at module scope.

``TestThereIsNoDefault``
    **Discriminating**, against the looser tree that still carried two
    fall-backs: a material *type* read as a declaration, and a shipped
    Eurocode rule beneath an undeclared timeline.  Both are gone.  The
    class also carries the cost measurement --- a model that says
    ``code: ec2`` gets exactly the number the tree gave before any of
    this existed --- and a positive control, without which "the rule
    declines" would pass on a rule that never answers at all.
"""

from __future__ import annotations

import math
import os

import pytest

from gensec.io_yaml import load_yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

#: EN 1992-1-1:2004 Table 3.1 and ACI 318-19 §19.2.2.1, written out
#: here rather than imported, so this file measures GenSec against the
#: clauses and not against its own implementation of them.
def _ec2_ecm(fck):
    r"""
    :math:`E_{cm} = 22\,000\,(f_{cm}/10)^{0.3}`, :math:`f_{cm}=f_{ck}+8`.

    Parameters
    ----------
    fck : float
        Characteristic cylinder strength [MPa].

    Returns
    -------
    float
        Secant modulus at 28 days [MPa].
    """
    return 22_000.0 * ((float(fck) + 8.0) / 10.0) ** 0.3


def _aci_ec(fc):
    r"""
    :math:`E_c = 4700\sqrt{f'_c}` [MPa], normalweight concrete.

    Parameters
    ----------
    fc : float
        Specified compressive strength [MPa].

    Returns
    -------
    float
    """
    return 4700.0 * math.sqrt(float(fc))


def _code(name):
    r"""
    Resolve a code by name, importing inside the call.

    Module-scope, the import would be a Phase 21 symbol named at
    collection time, and the pristine tree would report **one
    collection error** instead of a census of reds --- proving that a
    module is absent, which is not the same as proving a defect
    (``19_1 §4``).  Imported here, an absent module fails the one
    fixture that asked for it.

    Parameters
    ----------
    name : str

    Returns
    -------
    NormativeCode
    """
    from gensec.materials.codes import resolve_code
    return resolve_code(model_code=name)


def _outcome_of(call):
    r"""
    Run *call* and return its result, or the exception it raised.

    The ``_compile_outcome`` / ``_refusal`` pattern of ``18_1 §4``,
    in its smallest form.  A fixture that observes a refusal must
    observe **behaviour**, not a name: naming the exception class
    introduced by the phase makes the pristine tree fail with
    :class:`ImportError`, which proves a symbol is absent and not that
    the defect is present.  Catching everything and asserting on what
    came back keeps the red an :class:`AssertionError` on both trees.

    Parameters
    ----------
    call : callable
        Zero-argument.

    Returns
    -------
    object or BaseException
        The return value, or the exception instance.
    """
    try:
        return call()
    except Exception as exc:                # noqa: BLE001 — outcome is data
        return exc


def _example(name):
    r"""
    Absolute path of a distributed example, or a loud failure.

    Same single home and same reason as the ``_example`` helpers in
    ``test_phase19_hull.py`` and ``test_phase20_axial.py`` (F-P00-1).

    Parameters
    ----------
    name : str

    Returns
    -------
    str

    Raises
    ------
    FileNotFoundError
    """
    path = os.path.join(_ROOT, "examples", name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"distributed example {name!r} not found at {path}. The "
            f"examples/ directory is part of the package, not an "
            f"optional extra.")
    return path


def _write(tmp_path, text):
    r"""
    Write *text* as a YAML input and return its path.

    Parameters
    ----------
    tmp_path : pathlib.Path
    text : str

    Returns
    -------
    str
    """
    path = tmp_path / "model.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


_MODEL = """\
{code_line}
materials:
  conc: {{type: concrete_ec2_gen1_custom, fck: {fck}, gamma_c: 1.5,
          long_term_reduction_factor: 0.85{mat_code}}}
  bar:  {{type: steel, fyk: 420.0}}
section:
  B: 300
  H: 600
  bulk_material: conc
  n_fibers_y: 20
  rebars:
    - {{y: 50, As: 1000.0, material: bar}}
"""


def _model(tmp_path, fck=40.0, code=None, material_code=None):
    r"""
    Build and load a minimal model with the given declarations.

    Parameters
    ----------
    tmp_path : pathlib.Path
    fck : float
    code : str, optional
        Top-level ``code:``.
    material_code : str, optional
        Per-material ``code:``.

    Returns
    -------
    dict
        The loaded document.
    """
    text = _MODEL.format(
        code_line=f"code: {code}" if code else "",
        fck=fck,
        mat_code=f", code: {material_code}" if material_code else "")
    return load_yaml(_write(tmp_path, text))


def _modulus_of(document, name="conc"):
    r"""
    The conventional modulus a material states, or ``None``.

    Read through ``getattr`` and a broad ``except`` so that a tree
    where the material has a different contract fails this file's
    assertions rather than its imports.

    Parameters
    ----------
    document : dict
    name : str

    Returns
    -------
    tuple of (float or None, str)
        ``(value, outcome text)``.  The text is the citation on
        success and ``"<Type>: <message>"`` on a refusal.
    """
    material = document["materials"][name]
    try:
        declared = material.reference_modulus()
    except Exception as exc:                # noqa: BLE001 — outcome is data
        return None, f"{type(exc).__name__}: {exc}"
    return float(declared.value), str(declared.source)


class TestTheCodesDisagree:
    r"""
    Why the declaration is engineering and not bookkeeping.

    **Guard**, green on both trees: it asserts a property of two
    published clauses, which no change to GenSec affects.
    """

    @pytest.mark.parametrize("fck, gap", [(25.0, 33.9), (40.0, 18.5),
                                          (60.0, 7.4)])
    def test_the_two_clauses_are_far_apart_on_the_same_input(
            self, fck, gap):
        r"""
        One :math:`f_{ck}`, two codes, up to a third of the answer.

        The gaps are the figures quoted in the refusal messages and in
        ``handoff/21_0``; if they move, those texts are wrong.
        """
        measured = 100.0 * (_ec2_ecm(fck) - _aci_ec(fck)) / _aci_ec(fck)
        assert measured == pytest.approx(gap, abs=0.05), (
            f"at fck = {fck:g} the Eurocode exceeds ACI by "
            f"{measured:.2f} %, not the {gap:.1f} % recorded in the "
            f"refusal messages.")


class TestTheDeclarationGoverns:
    r"""
    F-P21-1 closed: the code is declared and everything follows.

    **Discriminating.**  Red on the pristine Phase 20 tree, on
    behaviour: the outcome of asking a material for its modulus is
    captured as either a value or an exception text, so no Phase 21
    symbol is named and no red is an :class:`ImportError`.
    """

    def test_a_model_that_declares_no_code_gets_no_derived_modulus(
            self, tmp_path):
        r"""
        ``f_ck`` is an input, not a code.

        The material type here is ``concrete_ec2_gen1_custom``, which
        *does* name EC2, so the top-level absence is not what is under
        test --- ``code: custom`` on the material is.  Declared custom
        means the input supplies the value and GenSec derives nothing.
        """
        value, text = _modulus_of(
            _model(tmp_path, fck=40.0, code="ec2",
                   material_code="custom"))
        assert value is None, (
            f"a material declared 'code: custom' still had a modulus "
            f"derived for it ({value:.3f} MPa, {text}). Custom means "
            f"GenSec derives nothing: that is what the declaration "
            f"asks for.")
        assert "custom" in text, (
            f"the refusal does not say that the material was declared "
            f"custom, so the caller cannot tell an omission from a "
            f"choice. Outcome was {text}")

    def test_declaring_aci_gives_the_aci_number_and_the_aci_citation(
            self, tmp_path):
        r"""The whole claim, in one assertion pair."""
        value, text = _modulus_of(_model(tmp_path, fck=28.0,
                                         code="aci318-19"))
        assert value == pytest.approx(_aci_ec(28.0), rel=1e-12), (
            f"a model declared under ACI 318-19 got {value} MPa where "
            f"§19.2.2.1 gives {_aci_ec(28.0):.3f}. Outcome was {text}")
        assert "ACI" in text, (
            f"the value is ACI's and the citation is not: a report "
            f"that derives under one code and cites another is not "
            f"verifiable by whoever signs it. Citation was {text}")

    def test_declaring_ntc_gives_the_ec2_number_and_the_ntc_citation(
            self, tmp_path):
        r"""
        NTC 2018 adopts the Eurocode formulation, and says so itself.

        The value must match EC2 to the last digit --- one formula, one
        home --- while the citation must be NTC's, for the same reason
        the ACI citation must be ACI's.
        """
        value, text = _modulus_of(_model(tmp_path, fck=40.0,
                                         code="ntc18"))
        assert value == pytest.approx(_ec2_ecm(40.0), rel=1e-9)
        assert "NTC" in text and "1992" not in text.split("EN")[0][:0] + text[:4], (
            f"a model declared under NTC 2018 cites {text!r}")
        assert "NTC" in text

    def test_a_material_may_override_the_model(self, tmp_path):
        r"""
        A precast beam under EC2 with an ACI topping is a real model.

        The per-material declaration has to win over the model's, or
        mixed-code construction is unrepresentable and the declaration
        is a global switch rather than a property of the material.
        """
        document = _model(tmp_path, fck=28.0, code="ec2",
                          material_code="aci318-19")
        value, text = _modulus_of(document)
        assert value == pytest.approx(_aci_ec(28.0), rel=1e-12), (
            f"the material's own 'code:' lost to the model's. "
            f"Outcome was {text}")

    def test_a_mistyped_code_is_refused_and_not_defaulted(self, tmp_path):
        r"""
        A typo must not silently become the Eurocode.

        This is the failure the declaration exists to remove, so a
        fallback here would reintroduce it at the one place a reader
        would never look.
        """
        with pytest.raises(ValueError) as caught:
            _model(tmp_path, fck=40.0, code="euroocde2")
        text = str(caught.value)
        assert "euroocde2" in text and "aci318-19" in text, (
            f"the refusal names neither what was written nor what is "
            f"available. Message was: {text}")

    def test_the_distributed_aci_example_is_loadable_and_is_aci(self):
        r"""
        The battery must contain a model that is not a Eurocode model.

        Until Phase 21 every distributed example was EC2, so a
        container that had quietly become Eurocode-shaped would have
        looked identical from outside --- the Phase 19 lesson, applied
        to the normative axis.
        """
        document = load_yaml(_example("example_aci318.yaml"))
        value, text = _modulus_of(document, "concrete_4ksi")
        assert value == pytest.approx(_aci_ec(28.0), rel=1e-12), (
            f"the distributed ACI example does not resolve under ACI. "
            f"Outcome was {text}")
        assert value == pytest.approx(24_870.062, abs=1e-3)


class TestThereIsNoDefault:
    r"""
    The last implicit surface, removed.

    Until this was tightened, two fall-backs survived on the ground of
    backward compatibility: a material *type* that named a code was
    read as a declaration of it, and a timeline with no declaration got
    the build's shipped rule, which was the Eurocode's.  Both are gone.
    Nobody is using GenSec yet, and the cost of keeping them was that a
    normative choice could still happen without anybody writing it
    down.

    **Discriminating**, on the tree that still had the fall-backs.
    """

    def test_a_material_type_is_not_a_declaration(self, tmp_path):
        r"""
        ``type: concrete_ec2_gen1`` names a curve, not a code.

        The type selects a constitutive shape.  Which partial factors,
        which conventional modulus and which creep law accompany that
        shape are separate facts, and a model may legitimately pair the
        curve with a code that did not invent it.  Reading the code out
        of the type is also a rule that only works for the sections
        somebody happened to think of first: nothing says a GenSec
        section is concrete at all.
        """
        with pytest.raises(ValueError) as caught:
            _model(tmp_path, fck=40.0)
        text = str(caught.value)
        assert "no normative code is declared" in text, (
            f"an EC2-named material type was still read as a "
            f"declaration. Message was: {text}")
        assert "code: custom" in text and "ec2" in text, (
            f"the refusal states neither the remedy nor what is "
            f"declarable. Message was: {text}")

    def test_the_refusal_happens_at_load_and_not_at_first_use(
            self, tmp_path):
        r"""
        An undeclared model never becomes an object.

        A refusal deferred to the first quantity that needs a modulus
        would surface halfway through an analysis, attached to whatever
        happened to ask first.  Refusing at the door names the material
        and costs nothing.
        """
        with pytest.raises(ValueError) as caught:
            _model(tmp_path, fck=40.0)
        assert "'conc'" in str(caught.value), (
            f"the refusal does not name which material is undeclared: "
            f"{caught.value}")

    def test_a_timeline_has_no_shipped_rule_beneath_the_declaration(
            self):
        r"""
        No ``default_reference_modulus`` under an undeclared timeline.

        The rule a timeline resolves to must be the one that asks each
        material for its own code, with no code of its own underneath.
        Asserted on behaviour --- the rule declines for an unstamped
        material --- rather than on the rule's class name, so it stays
        valid if the class is renamed.
        """
        from gensec.materials.concrete import Concrete
        from gensec.solver.timeline import ConstructionTimeline

        rule = ConstructionTimeline([], {})._reference_rule()
        assert rule is not None
        assert rule.E_c(Concrete(fck=40.0, gamma_c=1.5, long_term_reduction_factor=0.85)) is None, (
            "an undeclared timeline still derives a modulus for an "
            "undeclared material, so a normative choice is still "
            "being made without anybody writing it down.")

    def test_a_declared_timeline_answers_under_its_own_code(self):
        r"""
        Positive control: the declaration is what makes it work.

        Without this the test above would pass on a rule that never
        answers at all, which would be a different defect wearing the
        same green.
        """
        from gensec.materials.codes import resolve_code
        from gensec.materials.concrete import Concrete
        from gensec.solver.timeline import ConstructionTimeline

        material = Concrete(fck=28.0, gamma_c=1.5, long_term_reduction_factor=0.85)
        material.code = resolve_code(model_code="aci318-19")
        rule = ConstructionTimeline([], {}, code="aci318-19")._reference_rule()
        assert rule.E_c(material) == pytest.approx(_aci_ec(28.0),
                                                   rel=1e-12)

    def test_declaring_ec2_gives_the_eurocode_number(self, tmp_path):
        r"""
        Nothing about the Eurocode's own answer changed.

        A model that says ``code: ec2`` gets exactly the number the
        tree gave before any of this existed.  The declaration removed
        the guessing, not the Eurocode.
        """
        value, text = _modulus_of(_model(tmp_path, fck=40.0, code="ec2"))
        assert value == pytest.approx(_ec2_ecm(40.0), rel=1e-9), (
            f"outcome was {text}")
        assert value == pytest.approx(35_220.46228893441, rel=1e-12)


class TestTheSafetyFormatIsOwnedByTheCode:
    r"""
    Phase 21b — where a code turns characteristic into design.

    The two shipped families do it at **opposite ends** of the
    computation, and until this existed GenSec assumed the Eurocode's
    end without saying so:

    * EN 1992-1-1 and NTC 2018 reduce the **materials** before
      integration, and the resistance that comes out is already a
      design resistance;
    * ACI 318-19 integrates with the specified strengths and applies
      :math:`\phi` to the **nominal** resistance afterwards.

    So the interface is two hooks, each the identity in the other
    family: :meth:`NormativeCode.design_reduction` on a material, and
    :meth:`NormativeCode.capacity_factor` on a solved state.

    **Discriminating** against a tree without the hooks; every access
    is guarded so a missing hook fails as an
    :class:`AssertionError` and not an :class:`AttributeError`.
    """

    def _code(self, key):
        from gensec.materials.codes import resolve_code
        return resolve_code(model_code=key)

    def _concrete(self):
        from gensec.materials.concrete import Concrete
        return Concrete(fck=40.0, gamma_c=1.5, long_term_reduction_factor=0.85)

    def test_the_eurocode_reduces_at_the_material_end(self):
        r"""``where='material'``, and the partial factor is reported."""
        reduction = getattr(self._code("ec2"), "design_reduction",
                            lambda _m: None)(self._concrete())
        assert reduction is not None, (
            "the code states no design reduction at all, so where its "
            "safety format lives is still an assumption")
        assert reduction.where == "material"
        assert reduction.gamma_c == pytest.approx(1.5)
        assert "1992" in reduction.source

    def test_aci_reduces_at_the_capacity_end_and_says_so(self):
        r"""
        ``where='capacity'``, and the partial factors are ``None``.

        ``None`` and ``1.0`` are different statements.  A caller that
        divided by a returned ``1.0`` would turn "this code does not
        reduce materials" into "this code reduces them by unity", and
        only one of those is true.
        """
        reduction = getattr(self._code("aci318-19"), "design_reduction",
                            lambda _m: None)(self._concrete())
        assert reduction is not None
        assert reduction.where == "capacity"
        assert reduction.gamma_c is None and reduction.gamma_s is None, (
            f"ACI reports partial factors on materials: {reduction}")
        assert "ACI" in reduction.source

    def test_the_eurocode_capacity_factor_is_unity_as_a_statement(self):
        r"""
        Unity here means *the reduction already happened*.

        Distinguished from "no factor known", which declines.  The
        difference is carried in ``governed_by``, so a reader can
        check the number instead of trusting it.
        """
        from gensec.materials.codes import SectionalState

        factor = getattr(self._code("ec2"), "capacity_factor",
                         lambda _s: None)(SectionalState(eps_t=0.010))
        assert factor is not None
        assert factor.value == pytest.approx(1.0)
        assert "material" in factor.governed_by, (
            f"unity is published without saying why: {factor}")

    def test_aci_declines_its_factor_and_never_substitutes_unity(self):
        r"""
        The one refusal here that must never soften into a default.

        :math:`\phi` is a declared empty slot.  Returning ``1.0``
        would publish a **nominal** strength as a design strength ---
        unconservative by 10 % to 35 %, and silent.  Every other
        refusal in GenSec costs the caller an answer; this one would
        cost them a section.
        """
        from gensec.materials.codes import SectionalState

        code = self._code("aci318-19")
        assert hasattr(code, "capacity_factor")
        with pytest.raises(NotImplementedError) as caught:
            code.capacity_factor(SectionalState(eps_t=0.010))
        text = str(caught.value)
        assert "1.0" in text or "unity" in text.lower(), (
            f"the refusal does not say that unity is not a fallback: "
            f"{text}")
        assert "nominal" in text.lower()

    @pytest.mark.parametrize("key", ["custom", "undeclared"])
    def test_custom_and_undeclared_decline_both_hooks(self, key):
        r"""Neither invents a safety format, and neither returns 1.0."""
        from gensec.materials.codes import SectionalState

        code = self._code("ec2") if key == "ec2" else None
        from gensec.materials.codes import CUSTOM, UNDECLARED
        code = CUSTOM if key == "custom" else UNDECLARED
        with pytest.raises(NotImplementedError):
            code.design_reduction(self._concrete())
        with pytest.raises(NotImplementedError):
            code.capacity_factor(SectionalState(eps_t=0.010))

    def test_the_solver_can_build_the_state_a_factor_needs(self):
        r"""
        The information a strain-based factor needs is already there.

        ``eps_t`` is the **net** tensile strain --- a tendon's
        locked-in prestrain is excluded, because a prestrain says
        nothing about how far the section is from a brittle failure,
        and including it would put every prestressed section in the
        ductile branch.
        """
        from gensec.solver.integrator import FiberSolver

        document = load_yaml(_example("example_aci318.yaml"))
        solver = FiberSolver(document["section"])
        assert hasattr(solver, "sectional_state"), (
            "the solver cannot express its own solved state in the "
            "form a capacity factor reads, so a code that reduces at "
            "the capacity end cannot be implemented against it")
        state = solver.sectional_state(-5.0e-4, 8.0e-6,
                                       action="flexure",
                                       confinement="tied")
        assert state.eps_t is not None and state.eps_t > 0.0
        assert state.eps_c is not None and state.eps_c < 0.0
        assert state.eps_yield == pytest.approx(420.0 / 1.15 / 200000.0,
                                                rel=1e-9)
        assert state.action == "flexure"
        assert state.confinement == "tied"

    def test_the_interface_is_sufficient_for_a_strain_based_factor(self):
        r"""
        **ACI-readiness, proved rather than asserted.**

        A claim that an interface is ready for a code is worth nothing
        until somebody writes that code against it.  So this test
        implements :math:`\phi` per ACI 318-19 §21.2.2 in a dozen
        lines, using only what the interface hands it, and checks the
        three branches.  It is a *fixture*, not a shipped code: GenSec
        still declines for ACI, and this exists to measure that the
        shape of the hook is right.

        The three branches, with the 2019 edition's limits written
        against :math:`\varepsilon_{ty}` rather than the fixed
        0.002 / 0.005 of earlier editions.
        """
        from gensec.materials.codes import (ACI318Code, CapacityFactor,
                                            SectionalState)

        class _ACIWithPhi(ACI318Code):
            """ACI 318-19 §21.2.2, implemented against the interface."""

            def capacity_factor(self, state):
                if state.eps_t is None or state.eps_yield is None:
                    raise NotImplementedError(
                        "phi is strain-controlled and this state "
                        "carries no tension element to read it at.")
                if state.action in ("shear", "torsion"):
                    return CapacityFactor(0.75, "ACI 318-19 §21.2.1",
                                          state.action)
                low = 0.75 if state.confinement == "spiral" else 0.65
                ety, ecap = state.eps_yield, state.eps_yield + 0.003
                if state.eps_t >= ecap:
                    return CapacityFactor(0.90, "ACI 318-19 §21.2.2",
                                          "tension-controlled")
                if state.eps_t <= ety:
                    return CapacityFactor(low, "ACI 318-19 §21.2.2",
                                          "compression-controlled")
                phi = low + (0.90 - low) * (state.eps_t - ety) / 0.003
                return CapacityFactor(phi, "ACI 318-19 §21.2.2",
                                      "transition")

        code, ety = _ACIWithPhi(), 420.0 / 200000.0
        tied = dict(eps_yield=ety, confinement="tied")

        assert code.capacity_factor(
            SectionalState(eps_t=0.020, **tied)).value == \
            pytest.approx(0.90)
        assert code.capacity_factor(
            SectionalState(eps_t=0.001, **tied)).value == \
            pytest.approx(0.65)
        mid = code.capacity_factor(SectionalState(eps_t=ety + 0.0015,
                                                  **tied))
        assert mid.value == pytest.approx(0.775, abs=1e-9)
        assert mid.governed_by == "transition"
        assert code.capacity_factor(
            SectionalState(eps_t=0.001, eps_yield=ety,
                           confinement="spiral")).value == \
            pytest.approx(0.75)
        assert code.capacity_factor(
            SectionalState(eps_t=0.020, action="shear", **tied)).value \
            == pytest.approx(0.75)
        # And it still resolves the modulus it inherited, so the
        # subclass is a code and not a bare factor holder.
        from gensec.materials.concrete import Concrete
        assert code.reference_modulus(
            Concrete(fck=28.0, gamma_c=1.5, long_term_reduction_factor=0.85)).value == \
            pytest.approx(_aci_ec(28.0), rel=1e-12)


class TestTheSafetyFormatIsDeclaredNotAssumed:
    r"""
    Characteristic to design, in two hooks instead of one.

    Codes do not agree on **where** the reduction happens, and GenSec
    assumed they did.  EN 1992-1-1 divides the strengths and integrates
    reduced laws, so the design resistance falls out of the
    integration.  ACI 318-19 integrates the specified strengths and
    multiplies the resulting *nominal* resistance by :math:`\phi`.
    Those are opposite ends of the computation, and collapsing them
    into one is the Eurocode's shape wearing the costume of
    generality.

    **Architecture only.**  ACI's factors are a declared empty slot in
    this build, exactly like ACI 209R-92; these fixtures assert that
    the *shape* holds both formats and that the unimplemented half
    refuses instead of guessing.

    Why not one hook with a partial factor on the materials
    -------------------------------------------------------
    Because pushing :math:`\phi` inside as a scale :math:`\lambda` on
    the strengths is exact only where the nominal resistance is
    homogeneous of degree 1 in them, and it is not.  Measured by hand
    calculation on the Whitney block, scaling both :math:`f'_c` and
    :math:`f_y` by :math:`\lambda`:

    ==============================  ======================
    case                            error at lambda = 0.65
    ==============================  ======================
    singly reinforced RC            exact
    prestressed, ``f_ps`` capped    exact
    prestressed, strain-governed    **+16.68 %**
    doubly reinforced               +0.15 %
    ==============================  ======================

    It breaks wherever a **strain** governs instead of a strength: a
    locked-in ``eps_pe`` does not scale, and neither does ``eps_cu``.
    That it is exact on the singly reinforced case is what makes it a
    trap --- green on the first thing anyone would test.
    """

    def test_the_two_families_reduce_at_opposite_ends(self):
        r"""One hook is loaded in each code, never both, never neither."""
        from gensec.materials.codes import resolve_code
        from gensec.materials.concrete import Concrete

        material = Concrete(fck=40.0, gamma_c=1.5, long_term_reduction_factor=0.85)
        ec2 = resolve_code(model_code="ec2").design_reduction(material)
        aci = resolve_code(model_code="aci318-19").design_reduction(material)
        assert ec2.where == "material" and ec2.gamma_c == pytest.approx(1.5)
        assert aci.where == "capacity"
        assert aci.gamma_c is None, (
            "ACI reports a partial factor of "
            f"{aci.gamma_c}. 'this code does not reduce materials' and "
            "'this code reduces materials by unity' are different "
            "statements and only one of them is true.")

    def test_a_material_reducing_code_returns_unity_as_a_statement(self):
        r"""
        The Eurocode's ``1.0`` says *the reduction already happened*.

        Not an absence, and the fixture checks the provenance rather
        than the number, because a bare 1.0 cannot be told apart from
        a factor nobody implemented.
        """
        from gensec.materials.codes import resolve_code, SectionalState

        factor = resolve_code(model_code="ec2").capacity_factor(
            SectionalState(eps_t=0.010, eps_yield=0.0021,
                           action="flexure"))
        assert factor.value == pytest.approx(1.0)
        assert factor.governed_by, (
            "a factor without the branch that produced it cannot be "
            "checked by a reader.")

    def test_the_unimplemented_half_refuses_instead_of_guessing(self):
        r"""
        ACI's phi is a declared empty slot, and says so.

        Returning 1.0 here would publish a nominal resistance as
        though it were a design resistance --- unconservative by
        10 % to 35 % depending on the branch, and silent.
        """
        from gensec.materials.codes import resolve_code, SectionalState

        with pytest.raises(NotImplementedError) as caught:
            resolve_code(model_code="aci318-19").capacity_factor(
                SectionalState(eps_t=0.010, eps_yield=0.0021,
                               action="flexure"))
        assert "21.2" in str(caught.value), (
            f"the refusal does not name the clause it is standing in "
            f"for: {caught.value}")

    def test_a_domain_is_factored_point_by_point(self):
        r"""
        The factor varies along the curve because the strain does.

        A caller holding one scalar would be right in the
        tension-controlled region and wrong elsewhere.
        """
        from gensec.materials.codes import resolve_code, SectionalState

        code = resolve_code(model_code="ec2")
        states = [SectionalState(eps_t=e, eps_yield=0.0021,
                                 action="flexure")
                  for e in (0.030, 0.010, 0.003)]
        N, Mx, My, phi = code.capacity_envelope(
            [1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [0.0, 0.0, 0.0], states)
        assert len(phi) == 3
        assert N[2] == pytest.approx(3.0) and Mx[1] == pytest.approx(20.0)

    def test_one_state_cannot_be_reused_across_points(self):
        r"""
        The one mistake the signature exists to prevent.

        A factor defined on the solved strain is meaningless anywhere
        but at the point that produced it, so a length mismatch is an
        error and never a broadcast.
        """
        from gensec.materials.codes import resolve_code, SectionalState

        code = resolve_code(model_code="ec2")
        with pytest.raises(ValueError) as caught:
            code.capacity_envelope([1.0, 2.0, 3.0], [1.0, 1.0, 1.0],
                                   [0.0, 0.0, 0.0],
                                   [SectionalState(eps_t=0.01)])
        assert "3 point(s)" in str(caught.value)

    def test_a_cap_is_a_separate_hook_because_it_truncates(self):
        r"""
        A truncation is not a scaling and cannot be one.

        ACI 318-19 §22.4.2.1 cuts the compression branch flat at
        :math:`0.80 P_o`.  A scalar multiplier applied along the curve
        cannot express a flat cut, so it lives in its own hook, and a
        code that states no cap answers ``None`` --- which is not the
        same as a cap equal to :math:`P_o`.
        """
        from gensec.materials.codes import resolve_code

        assert resolve_code(model_code="ec2").axial_cap(
            -1.0e6, "tied") is None

    def test_a_code_that_has_a_cap_declines_rather_than_reporting_none(
            self):
        r"""
        ``None`` and "not implemented" are different answers.

        The base hook returns ``None``, which *states* that the code
        puts no cap on the axial branch.  That statement is false of
        ACI 318-19, which caps at :math:`0.80 P_o` (tied).  Inheriting
        the base answer would therefore publish an interaction domain
        running about 20 % too tall on the compression branch ---
        unconservative exactly where a column is governed, and silent.

        This is the whole failure mode the empty-slot discipline
        exists to prevent, arriving through inheritance rather than
        through a fallback: a default that is right for most members
        of a base class and wrong for one.
        """
        outcome = _outcome_of(
            lambda: _code("aci318-19").axial_cap(-1.0e6, "tied"))
        assert isinstance(outcome, NotImplementedError), (
            f"asking ACI 318-19 for its axial cap did not put GenSec "
            f"in the scope-boundary half of the exception taxonomy. "
            f"Outcome was {outcome!r}")
        text = str(outcome)
        assert "22.4.2.1" in text and "0.80" in text, (
            f"the refusal does not name the clause and the value it "
            f"would apply, so it reads as an omission rather than a "
            f"specification. Message was: {text}")
