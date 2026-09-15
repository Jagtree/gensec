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
Phase 30, Task B: four silences in the SLS check block.

Every one of them has the same shape.  Something the caller declared, or
something the caller is entitled to assume happened, does not happen,
and nothing in the output says so.  A check that is not run and a check
that passed are indistinguishable in the record.

B1 -- a limit that is accepted and never read
---------------------------------------------
``StressLimits.fct_eff`` exists, three live sites inside
:func:`gensec.materials.verification_limits.ec2_stress_limits` fill it,
and :mod:`gensec.solver.sls` reads it **zero** times.  It is the socket
left behind by the uncracked-basis flag that Phase 27 removed when it
put the real service law into the view: the state the flag warned about
became unreachable, the flag went, the field stayed.  A caller who
declares it believes a tensile check is running.  None is.

The remedy ratified for this phase is deletion rather than refusal.
``StressLimits`` is the container of *thresholds the engine verifies*; a
field the engine never verifies has no title to be in it.  And
:math:`f_{ct,eff}` is a material property that already lives where it
belongs, on
:class:`gensec.materials.service_concrete.ServiceConcrete`, so a second
copy in the limits provider would be the same normative symbol entering
through two doors.

B2 -- a division that is not guarded
------------------------------------
``_check_entry`` computes ``eta = value / limit``.  A zero limit raises
``ZeroDivisionError`` from the arithmetic, which names nothing a user
can act on; a negative limit does not raise at all and yields a
meaningless negative utilisation.  Both are malformed input and must be
refused where the division is, by an error that names the limit.

B3 -- a record that contradicts its own verdict
-----------------------------------------------
The published record rounds to six decimals; the verdict is taken on the
unrounded numbers.  There are two rounding artefacts with two different
widths.  ``value_MPa`` collapses onto ``limit_MPa`` for a constant band
of :math:`5\times10^{-7}` MPa.  ``eta`` collapses onto 1.0 for a band of
:math:`5\times10^{-7}\,\cdot` ``limit``, which **scales with the
limit**: measured 1.062e-05 MPa at a limit of 21 MPa and 9.449e-04 MPa
at 1860 MPa, the latter 1833 times the band on the value.  On a tendon
limit the record can read

.. code-block:: text

    value_MPa = 1860.0009   limit_MPa = 1860.0   eta = 1.0   passed = False

which asserts simultaneously that the demand exceeds the limit, that it
sits exactly at capacity, and that the check failed.  Two of those three
disagree.  The ratified remedy is one arithmetic: the record publishes
unrounded numbers and the verdict stays on the unrounded numbers.
Rounding belongs to the formatter.

B4 -- a check that vanishes without a trace
-------------------------------------------
``tendon_stress`` is emitted inside ``if ten_present.any():`` with no
``else``.  A section whose tendons are all unbonded loses the check and
gains no ``skipped`` entry, while the decompression check in the same
block does emit one when it has nothing to work on.  The same shape
occurs a second time, not named in the phase brief but measured here:
``steel_tension`` sits inside ``if reb_present.any():``, equally
without an ``else``, so a section with no rebars loses that check just
as silently.

Expected state
--------------
RED on the pristine Phase-29 tree, GREEN after the Task-B repair.
"""

import dataclasses

import pytest

from gensec.materials.verification_limits import StressLimits, ec2_stress_limits


# ======================================================================
#  B1 -- fct_eff
# ======================================================================

def test_stress_limits_has_no_unverified_field():
    r"""
    Every field of ``StressLimits`` must be a threshold the engine reads.

    ``name`` identifies the combination and ``c_dec`` parameterises the
    decompression check rather than bounding a stress, so both are
    declared here as non-thresholds.  Anything else must correspond to
    a check the engine emits.
    """
    # ``custom_zone_strengths`` joined the non-thresholds in Phase 32:
    # it carries strengths, not limits, and the check it feeds is the
    # per-zone ``concrete_compression`` that ``k_c`` answers for.
    non_thresholds = {"name", "c_dec", "decompression",
                      "custom_zone_strengths"}
    fields = {f.name for f in dataclasses.fields(StressLimits)}
    thresholds = fields - non_thresholds
    expected = {"k_c", "sigma_s_tens", "sigma_p_max"}
    assert thresholds == expected, (
        "StressLimits carries %r as a threshold; the engine reads only "
        "k_c, sigma_s_tens and sigma_p_max"
        % sorted(thresholds - expected))


def test_declaring_fct_eff_as_a_limit_is_refused():
    r"""
    A caller must not be able to declare a tensile limit here.

    ``f_ct,eff`` is a property of the concrete and is declared on
    ``ServiceConcrete``.  Accepting it here and ignoring it is the
    defect; accepting it here and honouring it would put the same
    normative quantity behind two declarations.
    """
    with pytest.raises(TypeError):
        StressLimits(name="characteristic", k_c=0.6, fct_eff=3.2)
    # The provider absorbs unknown keywords into ``**overrides``, so
    # its refusal arrives as ValueError rather than TypeError.  What
    # matters is not which type: it is that the message names where the
    # quantity now lives.  A refusal that only lists valid coefficient
    # keys would leave a migrating caller with no idea that f_ct,eff is
    # still declarable, just elsewhere.
    with pytest.raises((TypeError, ValueError)) as excinfo:
        ec2_stress_limits("characteristic", fyk=450.0,
                          fpk=1860.0, fct_eff=3.2)
    assert "ServiceConcrete" in str(excinfo.value), (
        "the refusal does not name the lever that replaced it: %r"
        % str(excinfo.value))


def test_no_residual_prose_describes_the_removed_flag():
    r"""
    The uncracked-basis flag is gone; its description must go with it.

    Four sites still describe ``basis_checked`` and
    ``uncracked_basis_violated`` as if they existed.  Prose that
    documents a removed mechanism is a declared capability that is
    absent -- the same defect as B1, in a comment instead of a field.
    """
    import inspect

    from gensec.solver import sls
    from gensec.materials import elastic, verification_limits

    offenders = []
    for module in (sls, elastic, verification_limits):
        text = inspect.getsource(module)
        for token in ("basis_checked", "uncracked_basis_violated"):
            if token in text:
                offenders.append("%s: %s" % (module.__name__, token))
    assert not offenders, (
        "removed mechanism still described in %r" % offenders)


# ======================================================================
#  B2 -- the unguarded division
# ======================================================================

@pytest.mark.parametrize("limit", [0.0, -1.0, -21.0])
def test_a_non_positive_limit_is_refused_by_name(limit):
    r"""
    A limit that is not a positive stress is malformed input.

    Zero raises ``ZeroDivisionError`` today, which names nothing; a
    negative limit is accepted and produces a negative utilisation,
    which is worse because it is silent.  Both must be refused where
    the division happens, by an error whose message names the limit.
    """
    from gensec.solver.sls import _check_entry

    with pytest.raises(ValueError) as excinfo:
        _check_entry("concrete_compression", 5.0, limit)
    message = str(excinfo.value)
    assert "concrete_compression" in message, (
        "the refusal does not name the check: %r" % message)
    assert repr(limit) in message or ("%g" % limit) in message, (
        "the refusal does not name the offending limit: %r" % message)


def test_zero_division_is_not_how_the_refusal_arrives():
    r"""
    ``ZeroDivisionError`` is the arithmetic complaining, not the model.

    An exception type must say what kind of thing went wrong.  A
    declared limit of zero is an undeclared-input problem, not an
    arithmetic accident, and the two carry different remedies.
    """
    from gensec.solver.sls import _check_entry

    with pytest.raises(Exception) as excinfo:
        _check_entry("concrete_compression", 5.0, 0.0)
    assert not isinstance(excinfo.value, ZeroDivisionError), (
        "the zero limit still surfaces as ZeroDivisionError")


# ======================================================================
#  B3 -- the record that contradicts the verdict
# ======================================================================

@pytest.mark.parametrize(
    "limit,delta",
    [(21.0, 4.0e-7), (21.0, 1.0e-5), (100.0, 4.0e-5), (1860.0, 9.0e-4)])
def test_the_record_reproduces_its_own_verdict(limit, delta):
    r"""
    ``eta > 1`` if and only if the check failed.

    The four cases straddle both rounding bands: the first is inside the
    band on ``value_MPa``, the rest are inside the band on ``eta`` only,
    which is the wider one and the one that scales with the limit.
    """
    from gensec.solver.sls import _check_entry

    record = _check_entry("tendon_stress", limit + delta, limit)
    assert record["passed"] is False
    assert record["eta"] > 1.0, (
        "eta=%r asserts the demand sits at or below capacity while "
        "passed=False asserts it does not" % record["eta"])


@pytest.mark.parametrize("limit,delta", [(21.0, 4.0e-7), (1860.0, 9.0e-4)])
def test_the_record_does_not_hide_the_exceedance(limit, delta):
    r"""
    A value above the limit must not be published as equal to it.
    """
    from gensec.solver.sls import _check_entry

    record = _check_entry("tendon_stress", limit + delta, limit)
    assert record["value_MPa"] > record["limit_MPa"], (
        "value_MPa=%r is published as not exceeding limit_MPa=%r, yet "
        "passed=%r" % (record["value_MPa"], record["limit_MPa"],
                       record["passed"]))


def test_the_verdict_is_unchanged_by_the_repair():
    r"""
    Control: unrounding the record must not move any verdict.

    The repair removes an inconsistency between two published numbers
    and a boolean.  The boolean was always taken on the unrounded
    comparison and must keep exactly the values it had.
    """
    from gensec.solver.sls import _check_entry

    for limit, value, expected in ((21.0, 20.0, True),
                                   (21.0, 21.0, True),
                                   (21.0, 22.0, False),
                                   (1860.0, 1859.9, True),
                                   (1860.0, 1860.0009, False)):
        assert _check_entry("x", value, limit)["passed"] is expected


# ======================================================================
#  B4 -- the checks that vanish
# ======================================================================

def _checks_for(bonded_tendon=True, with_rebar=True):
    """
    Run the staged SLS verification on the Phase-7 bench.

    The bench is the one the project already trusts for this path,
    reused rather than reinvented so that a failure here is a failure of
    the check block and not of a fixture's own section.

    Parameters
    ----------
    bonded_tendon : bool, optional
        Whether the tendon is bonded at the verified stage.
    with_rebar : bool, optional
        Whether the rebars are present at the verified stage.

    Returns
    -------
    list of dict
        The check records of the single stage.
    """
    import os
    import sys

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import run_phase7_sls_validation as bench
    from gensec.solver.sls import verify_sls_staged

    section = bench._build_section()
    active = [1 if with_rebar else 0, 1 if with_rebar else 0, 1]
    bonded = [1 if with_rebar else 0, 1 if with_rebar else 0,
              1 if bonded_tendon else 0]
    state = bench._state(active, bonded, bench.EPS_PE, 0)
    # Every threshold declared, and the decompression switch on: the
    # closure gate below asks whether a fully populated limit set
    # reaches a named check for each of its thresholds, and a limit set
    # that declares nothing cannot answer that question.
    limits = ec2_stress_limits("characteristic", fyk=450.0,
                               fpk=1860.0, decompression=True)
    result = verify_sls_staged(
        section, [dict(name="s", state=state,
                       increment=(0.0, 600.0e6, 0.0), limits=limits)],
        moduli={"C35": bench.EC}, x_ref=bench.XR, y_ref=bench.YR)
    return result["stages"][0]["checks"]


def test_an_unbonded_tendon_leaves_a_skipped_entry():
    r"""
    ``tendon_stress`` may be absent from the verdicts, never from the
    record.

    The decompression check in the same block already emits a skipped
    entry when it has no tendon to work on.  Two checks in one block
    must not treat the same absence in two ways.
    """
    names = {c["name"]: c for c in _checks_for(bonded_tendon=False)}
    assert "tendon_stress" in names, (
        "tendon_stress disappeared entirely; emitted: %r"
        % sorted(names))
    assert names["tendon_stress"].get("skipped"), (
        "tendon_stress is present but not marked skipped: %r"
        % names["tendon_stress"])


def test_a_section_without_rebars_leaves_a_skipped_entry():
    r"""
    The same obligation for ``steel_tension``.

    Not named in the phase brief; measured during Task B and repaired
    with its twin, because a repair that fixed one guard and left the
    identical one beside it would be a repair that treated the symptom.
    """
    names = {c["name"]: c for c in _checks_for(with_rebar=False)}
    assert "steel_tension" in names, (
        "steel_tension disappeared entirely; emitted: %r" % sorted(names))
    assert names["steel_tension"].get("skipped"), (
        "steel_tension is present but not marked skipped: %r"
        % names["steel_tension"])


def test_every_declared_limit_reaches_a_named_check():
    r"""
    The closure gate of Task B, in executable form.

    For a fully populated ``StressLimits`` the set of thresholds
    declared and the set of check names emitted must correspond, with no
    threshold left without a check.  On the pristine tree the difference
    is exactly ``fct_eff``.
    """
    checks = _checks_for()
    # Per-element checks carry the element in brackets --
    # ``decompression[tendon 0]``, ``tendon_stress[T_ext]``.  The gate
    # is about which limits reach a check, not how many elements each
    # check visited, so the suffix is stripped.
    emitted = {c["name"].split("[")[0] for c in checks}
    expected = {"concrete_compression", "steel_tension", "tendon_stress",
                "decompression"}
    assert expected <= emitted, (
        "declared limits with no check emitted: %r" % sorted(expected - emitted))
    assert not (emitted - expected), (
        "checks emitted that no limit declares: %r" % sorted(emitted - expected))
# GENSEC-COMPLETE phase=32 utc=2026-09-05T13:56:13Z sha256=5ee29ea31a1d52020eb9769077beed3e0fd311d4e8d5d52deca14788dd05e701 body_lines=376
