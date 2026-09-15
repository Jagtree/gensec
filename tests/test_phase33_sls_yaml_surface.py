# SPDX-License-Identifier: AGPL-3.0-or-later
# GenSec — reinforced/prestressed concrete sectional analysis.
# Copyright (C) 2026  Andrea (GenSec project).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.  See <https://www.gnu.org/licenses/>.
r"""
Phase 33 --- the YAML surface of the SLS engine.

What these fixtures pin
-----------------------
Until Phase 33 the staged SLS engine was reachable **only from the
Python API**.  ``grep -c 'sls'`` on ``src/gensec/cli.py`` and on
``src/gensec/io_yaml.py`` returned ``0`` and ``0``;
:func:`gensec.solver.sls.verify_sls_staged` had zero callers in
``src/``; :func:`gensec.output.report.print_sls_results` had zero
callers anywhere.  A user writing YAML received a ULS domain check and
**no stress check at all**, and nothing said so.

Every fixture here fails on the tree before the repair for a reason
that is about *behaviour*, not about infrastructure: a block that
loads and is dropped, a malformed block that is accepted, a
combination that produces neither a verdict nor a record of its
absence.  A fixture that failed with ``ImportError`` would discriminate
nothing (``32_0`` §9), so the ones that need the driver reach it
through :func:`getattr` and assert on its absence with a message that
names what is missing.

The bench section
-----------------
A 600x1400 precast web (zone ``base``, C45) with a 200 mm topping
(zone ``topping``, C30) cast on top after the precast has taken its own
weight.  The two strengths are deliberately different: at
:math:`k_1 = 0.6` the limits are 27.0 and 18.0 MPa, so a resolution
that collapsed to one section-wide limit would show as **one** entry in
the applied-limit set instead of two --- a set difference, not a
rounding.
"""
from __future__ import annotations

import copy
import os
import sys
import tempfile

import pytest
import yaml

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from gensec.io_yaml import load_yaml                      # noqa: E402
import gensec.solver.timeline_run as timeline_run         # noqa: E402


# ----------------------------------------------------------------------
#  The document under test
# ----------------------------------------------------------------------

#: The SLS moduli the fixtures declare.  Written, never derived: an SLS
#: modulus is a normative, time-dependent choice and GenSec derives none
#: (Phase-7 D3).  The two differ because the topping is the younger
#: concrete, and a bench in which they were equal could not tell a
#: per-zone resolution from a section-wide one.
MODULI = {"base": 36000.0, "topping": 31000.0}

#: The ordinary limits block: characteristic, EC2 by inheritance from
#: the document's top-level ``code``.
LIMITS = {"kind": "characteristic", "fyk": 450.0, "fpk": 1860.0}


def _doc(sls_block=None, *, second=True):
    r"""
    Build the two-zone composite document.

    Parameters
    ----------
    sls_block : dict or None
        Placed on the first combination (``SLS_char``).  ``None``
        omits it, which is the case that must produce an explicit
        record of absence.
    second : bool, optional
        Append a second combination that declares no ``sls`` block.

    Returns
    -------
    dict
    """
    combo = {
        "name": "SLS_char",
        "at": "service",
        "history_factors": {"G1": 1.0, "G2": 1.0},
        "components": [{"ref": "Q", "factor": 1.0}],
    }
    if sls_block is not None:
        combo["sls"] = copy.deepcopy(sls_block)
    combos = [combo]
    if second:
        combos.append({
            "name": "ULS_composite",
            "at": "service",
            "history_factors": {"G1": 1.35, "G2": 1.35},
            "components": [{"ref": "Q", "factor": 1.5}],
        })
    return {
        "code": "ec2",
        "schema_version": 1,
        "materials": {
            "c_precast": {"type": "concrete", "fck": 45.0,
                          "gamma_c": 1.5,
                          "long_term_reduction_factor": 0.85},
            "c_topping": {"type": "concrete", "fck": 30.0,
                          "gamma_c": 1.5,
                          "long_term_reduction_factor": 0.85},
            "steel": {"type": "steel", "fyk": 450.0, "gamma_s": 1.15},
        },
        "section": {
            "shape": "rect",
            "params": {"B": 600, "H": 1400},
            "bulk_material": "c_precast",
            "mesh_size": 50,
            "material_zones": [{
                "shape": "custom", "name": "topping",
                "material": "c_topping",
                "params": {"exterior": [[0, 1200], [600, 1200],
                                        [600, 1400], [0, 1400]]},
            }],
            "rebars": [
                {"name": "A1", "x": 60, "y": 60, "diameter": 26,
                 "material": "steel"},
                {"name": "A2", "x": 540, "y": 60, "diameter": 26,
                 "material": "steel"},
                {"name": "T1", "x": 60, "y": 1340, "diameter": 16,
                 "material": "steel"},
                {"name": "T2", "x": 540, "y": 1340, "diameter": 16,
                 "material": "steel"},
            ],
        },
        "losses_models": {
            "rheo_precast": {"provider": "ec2", "fck": 45.0,
                             "cement_class": "N", "RH": 70.0},
            "rheo_topping": {"provider": "ec2", "fck": 30.0,
                             "cement_class": "N", "RH": 70.0},
        },
        "zone_rheology": {
            "base": {"model": "rheo_precast", "age": 28, "curing": 3},
            "topping": {"model": "rheo_topping", "age": 0, "curing": 3},
        },
        "demands": [
            {"name": "G1", "N_kN": -300, "Mx_kNm": -220},
            {"name": "G2", "Mx_kNm": -180},
            {"name": "Q", "Mx_kNm": -260},
        ],
        "construction_history": [
            {"load": {"demand": "G1"}},
            {"cast": {"zone": "topping", "datum": "auto"}},
            {"point": "transfer"},
            {"load": {"demand": "G2"}},
            {"point": "service"},
        ],
        "combinations": combos,
        "output": {"verification_top_k": 10},
    }


def _load(doc):
    r"""
    Write *doc* to a temporary file and load it.

    Through the file, not through the parser directly: the surface
    under test is *a YAML document*, and a test that called the parser
    with a dict would not exercise the load path an author actually
    uses.

    Parameters
    ----------
    doc : dict

    Returns
    -------
    dict
        The parsed model.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, sort_keys=False)
        return load_yaml(path)


def _run_sls(data):
    r"""
    Call the driver, or fail with a message that names its absence.

    Reached through :func:`getattr` deliberately.  An ``ImportError``
    at module scope would take every fixture in this file down with one
    traceback that says nothing about which behaviour is missing;
    ``32_0`` §9 calls that the wrong RED.

    Parameters
    ----------
    data : dict

    Returns
    -------
    dict
    """
    driver = getattr(timeline_run, "run_sls", None)
    assert driver is not None, (
        "gensec.solver.timeline_run.run_sls does not exist: the SLS "
        "engine has no driver reachable from a loaded YAML document, "
        "so a document can declare stress limits and receive no "
        "stress verification."
    )
    return driver(data, n_points=12)


def _concrete_checks(stage):
    r"""
    The concrete-compression records of a stage, matched on the stem.

    Stem, never equality: the record is
    ``concrete_compression[<zone>]`` since Phase 32, and a probe asking
    for the bare name reports ``absent`` for a check that ran (E32-2).

    Parameters
    ----------
    stage : dict

    Returns
    -------
    list of dict
    """
    return [c for c in stage.get("checks", []) or []
            if c.get("name", "").split("[")[0] == "concrete_compression"]


# ----------------------------------------------------------------------
#  1. The block reaches the engine
# ----------------------------------------------------------------------

class TestBlockReachesTheEngine:
    """A declared ``sls`` block survives the load and produces records."""

    def test_the_block_survives_the_load(self):
        """
        Before Phase 33 the block loaded and was dropped.

        This is the silence itself: the document is accepted, the run
        completes, a verdict is reported, and the stress verification
        the author asked for never happened.
        """
        data = _load(_doc({"moduli": MODULI, "limits": LIMITS}))
        combo = data["combinations"][0]
        assert "sls" in combo, (
            "the 'sls' block was dropped by the loader: a declaration "
            "the engine never acts on, with nothing to tell the author."
        )

    def test_a_combination_with_limits_produces_per_zone_records(self):
        """Two zones, two records, at every stage of the walk."""
        out = _run_sls(_load(_doc({"moduli": MODULI,
                                   "limits": LIMITS})))
        rec = [c for c in out["combinations"]
               if c["name"] == "SLS_char"][0]
        assert not rec.get("skipped")
        res = rec["points"]["service"]
        last = res["stages"][-1]
        names = sorted(c["name"] for c in _concrete_checks(last))
        assert names == ["concrete_compression[base]",
                         "concrete_compression[topping]"], names

    def test_the_two_zones_get_their_own_limits(self):
        r"""
        :math:`0.6\times45 = 27` and :math:`0.6\times30 = 18` MPa.

        A section-wide limit would put **one** value in this set.  The
        difference is a set difference and cannot be a rounding.
        """
        out = _run_sls(_load(_doc({"moduli": MODULI,
                                   "limits": LIMITS})))
        rec = [c for c in out["combinations"]
               if c["name"] == "SLS_char"][0]
        last = rec["points"]["service"]["stages"][-1]
        lims = sorted(round(float(c["limit_MPa"]), 6)
                      for c in _concrete_checks(last))
        assert lims == [18.0, 27.0], lims


# ----------------------------------------------------------------------
#  2. Absence is a record, never a silence
# ----------------------------------------------------------------------

class TestAbsenceIsRecorded:
    """A combination without ``sls`` says so, in the output."""

    def test_a_combination_without_a_block_leaves_a_record(self):
        """
        Not silence, and not an omission from the list either.

        "No stress verification was requested here" and "no stress
        verification exists" were the same observable until this phase.
        They must be different lines in the report, or a document that
        forgot the block is indistinguishable from a GenSec that cannot
        do the check.
        """
        out = _run_sls(_load(_doc(None)))
        names = sorted(c["name"] for c in out["combinations"])
        assert names == ["SLS_char", "ULS_composite"], names
        for c in out["combinations"]:
            assert c.get("skipped") is True, c["name"]
            assert "sls" in c["reason"], (
                f"the record for '{c['name']}' does not name the lever: "
                f"{c['reason']!r}")

    def test_requested_counts_only_what_was_asked_for(self):
        """A document that asks for nothing reports zero, not a
        vacuous ``True`` over an empty set."""
        out = _run_sls(_load(_doc(None)))
        assert out["requested"] == 0

    def test_one_asked_one_not_are_distinguishable(self):
        """The ordinary document holds both kinds of combination."""
        out = _run_sls(_load(_doc({"moduli": MODULI,
                                   "limits": LIMITS})))
        by_name = {c["name"]: c for c in out["combinations"]}
        assert by_name["ULS_composite"].get("skipped") is True
        assert not by_name["SLS_char"].get("skipped")
        assert out["requested"] == 1


# ----------------------------------------------------------------------
#  3. Refusals, each naming its lever
# ----------------------------------------------------------------------

class TestRefusals:
    """Malformed blocks are refused by name, never absorbed."""

    def test_an_unknown_field_inside_limits_is_refused_by_name(self):
        """
        ``f_ckk`` is the misspelling this surface makes easy.

        Accepting it would leave the verification running on the
        provider's values while the document says otherwise --- the
        defect of Phase 32, one level down.
        """
        with pytest.raises(ValueError) as exc:
            _load(_doc({"moduli": MODULI,
                        "limits": {**LIMITS, "f_ckk": 30.0}}))
        msg = str(exc.value)
        assert "f_ckk" in msg, msg
        assert "kind" in msg, ("the refusal must also name the valid "
                               "set, or the author is sent to the "
                               "source: " + msg)

    def test_an_unknown_key_in_the_sls_block_is_refused_by_name(self):
        with pytest.raises(ValueError) as exc:
            _load(_doc({"moduli": MODULI, "limits": LIMITS,
                        "modulli": {"base": 1.0}}))
        assert "modulli" in str(exc.value)

    def test_a_missing_kind_is_refused_and_names_the_lever(self):
        """No default kind: the families are different coefficients on
        different strengths, and guessing reports a verdict against
        limits the document never asked for."""
        with pytest.raises(ValueError) as exc:
            _load(_doc({"moduli": MODULI, "limits": {"fyk": 450.0}}))
        assert "kind" in str(exc.value)

    def test_a_missing_moduli_sub_block_is_refused(self):
        with pytest.raises(ValueError) as exc:
            _load(_doc({"limits": LIMITS}))
        assert "moduli" in str(exc.value)

    def test_a_missing_limits_sub_block_is_refused(self):
        with pytest.raises(ValueError) as exc:
            _load(_doc({"moduli": MODULI}))
        assert "limits" in str(exc.value)

    def test_a_non_positive_modulus_is_refused(self):
        with pytest.raises(ValueError) as exc:
            _load(_doc({"moduli": {"base": 0.0, "topping": 31000.0},
                        "limits": LIMITS}))
        assert "base" in str(exc.value)

    def test_transfer_without_declared_strengths_is_refused(self):
        r"""
        The YAML surface is a way **to** the provider, never around it.

        At transfer the governing strength is :math:`f_{ck}(t)` at the
        age the tendons are released, which no zone material knows.
        Deriving it from the 28-day value would raise the limit above
        what the concrete carries on the day it is loaded --- a silent
        fallback in the unsafe direction.
        """
        with pytest.raises(ValueError) as exc:
            _load(_doc({"moduli": MODULI, "limits": {"kind": "transfer"}}))
        assert "custom_zone_strengths" in str(exc.value)

    def test_transfer_with_declared_strengths_is_accepted(self):
        """The positive control of the refusal above: without it, a
        blanket refusal of ``transfer`` would pass the test too."""
        data = _load(_doc({
            "moduli": MODULI,
            "limits": {"kind": "transfer",
                       "custom_zone_strengths": {
                           "base": {"f_ck": 28.0},
                           "topping": {"f_ck": 12.0}}}}))
        assert data["combinations"][0]["sls"]["limits"].k_c == 0.6

    def test_a_code_with_no_provider_is_refused_by_name(self):
        """
        Hermetic isolation, enforced at the surface.

        Serving EN 1992-1-1 coefficients to a document that declared
        ACI 318-19 would be a normative substitution nothing in the
        output could reveal.
        """
        with pytest.raises(ValueError) as exc:
            _load(_doc({"moduli": MODULI,
                        "limits": {**LIMITS, "code": "aci318-19"}}))
        msg = str(exc.value)
        assert "aci318-19" in msg and "provider" in msg, msg


# ----------------------------------------------------------------------
#  4. Provenance travels to the output
# ----------------------------------------------------------------------

class TestProvenanceReachesTheResult:
    r"""
    ``material`` and ``custom`` are different assertions.

    A check passed against a strength GenSec **derived** from the
    zone's material and one passed against a strength the caller
    **declared** are not the same statement, and the report has to keep
    them apart --- ``provenance`` says who produced the number, never
    whether it complies.
    """

    def test_a_derived_strength_reads_material(self):
        out = _run_sls(_load(_doc({"moduli": MODULI,
                                   "limits": LIMITS})))
        rec = [c for c in out["combinations"]
               if c["name"] == "SLS_char"][0]
        last = rec["points"]["service"]["stages"][-1]
        prov = {c["name"].split("[", 1)[1].rstrip("]"): c["provenance"]
                for c in _concrete_checks(last)}
        assert prov == {"base": "material", "topping": "material"}, prov

    def test_a_declared_strength_reads_custom_and_only_there(self):
        """One zone declared, one derived, in the same result."""
        out = _run_sls(_load(_doc({
            "moduli": MODULI,
            "limits": {**LIMITS,
                       "custom_zone_strengths": {
                           "topping": {"f_ck": 22.0}}}})))
        rec = [c for c in out["combinations"]
               if c["name"] == "SLS_char"][0]
        last = rec["points"]["service"]["stages"][-1]
        by_zone = {c["name"].split("[", 1)[1].rstrip("]"): c
                   for c in _concrete_checks(last)}
        assert by_zone["topping"]["provenance"] == "custom"
        assert by_zone["base"]["provenance"] == "material"
        assert round(by_zone["topping"]["limit_MPa"], 6) == 13.2
        assert round(by_zone["base"]["limit_MPa"], 6) == 27.0


# ----------------------------------------------------------------------
#  5. The formatter has a caller (Z-12)
# ----------------------------------------------------------------------

def test_print_sls_results_is_reachable_from_the_cli(capsys):
    r"""
    ``print_sls_results`` had **zero callers anywhere** in the tree.

    A formatter nobody calls is dead code; this phase decided which of
    the two it was by connecting it.  The fixture drives the real
    result through the real formatter rather than asserting on an
    import, because the question is whether the output exists.
    """
    from gensec.output.report import print_sls_results

    out = _run_sls(_load(_doc({"moduli": MODULI, "limits": LIMITS})))
    rec = [c for c in out["combinations"]
           if c["name"] == "SLS_char"][0]
    print_sls_results(rec["points"]["service"])
    text = capsys.readouterr().out
    assert "SLS STAGED STRESS VERIFICATION" in text
    assert "concrete_compression[base]" in text
    assert "concrete_compression[topping]" in text


def test_the_cli_run_verb_names_sls(capsys):
    """
    ``grep -c 'sls' src/gensec/cli.py`` was **0** before this phase.

    Measured, not remembered: the file is read here, so a future
    refactor that removed the wiring would turn this red instead of
    leaving the surface unreachable in silence.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "src", "gensec", "cli.py"),
              encoding="utf-8") as fh:
        text = fh.read()
    assert "run_sls" in text
    assert "print_sls_results" in text


# ----------------------------------------------------------------------
#  6. Nothing distributed moved
# ----------------------------------------------------------------------

def test_every_distributed_example_still_loads():
    """
    The block is optional, so the denominator of this change is zero
    examples.  Pinned rather than argued.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ex = sorted(f for f in os.listdir(os.path.join(root, "examples"))
                if f.endswith(".yaml"))
    assert len(ex) == 24, ex
    for name in ex:
        load_yaml(os.path.join(root, "examples", name))


def test_no_distributed_example_declares_an_sls_block():
    """
    Zero examples changed, and the fixture says so mechanically.

    If a later phase adds one, this turns red and the count above has
    to be revisited deliberately rather than drifting.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    declaring = []
    for name in sorted(os.listdir(os.path.join(root, "examples"))):
        if not name.endswith(".yaml"):
            continue
        data = load_yaml(os.path.join(root, "examples", name))
        for combo in data.get("combinations", []) or []:
            if "sls" in combo:
                declaring.append(name)
    assert declaring == [], declaring
# GENSEC-COMPLETE phase=33 utc=2026-09-05T19:52:30Z sha256=e61e9e076afffa8035ad7fe5fd3d08cdf701c813596852e5126c2d6e78fbcd04 body_lines=545
