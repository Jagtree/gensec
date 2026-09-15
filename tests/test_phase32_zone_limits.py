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
Phase 32, Task B: the SLS concrete compression limit is derived per zone.

The defect this closes
----------------------
``StressLimits.sigma_c_comp`` was **one number for the whole stage**,
compared against the largest compression anywhere in the section.  On a
composite that is not an imprecise limit, it is a **false pass**.
Measured on the two-zone bench (base C45, topping C30,
:math:`M_x = -350` kNm):

.. math::

    \sigma_{\min}(\text{topping}) = -22.5753\ \text{MPa},
    \qquad
    k_1 f_{ck}^{\,\mathrm{base}} = 27.0,
    \qquad
    k_1 f_{ck}^{\,\mathrm{top}} = 18.0\ \text{MPa}

The topping passed against 27.0 and the stage was reported verified
while that zone stood at 125 % of what its own material prescribes.

The formula, ratified
---------------------
.. math::

    \sigma_{\lim}(\text{zone}) = k_c \times
    \begin{cases}
        f_{ck}^{\,\mathrm{custom}}(\text{zone}) & \text{if declared}\\
        f_{ck}(\text{material of the zone})     & \text{otherwise}
    \end{cases}

The ``otherwise`` branch derives **per zone, always**.  A section-wide
``sigma_c_comp`` is a collapsed product with no place in that formula,
so it is deleted rather than kept beside the new path with a precedence
rule -- the one move ``30_2`` §4 forbids by name, and the same move
Phase 30 already made on ``fct_eff``.

What these fixtures pin, and why each one
-----------------------------------------
**The deletion, not just the addition.**  A tree that grew ``k_c`` and
kept ``sigma_c_comp`` would satisfy every behavioural assertion below
while carrying two doors onto one quantity.  So the first fixture
asserts the field is *gone* and that naming it raises a message which
says what to write instead -- the pattern ``fct_eff`` established.

**A coefficient, not a product.**  ``k_c`` above one is refused.  No
stress-limit coefficient in any normative document exceeds unity, and
the failure this catches is the obvious one: someone writing
``k_c=0.6*35`` out of habit.  Silently accepting 21.0 as a coefficient
would multiply it by :math:`f_{ck}` again.

**Provenance.**  A check passed against a custom strength and one
passed against the code's own prescription are not the same statement,
and a reader who cannot tell them apart cannot audit the run.

**Every zone leaves a record.**  Obligation B4 of Phase 30, applied to
the new axis: a zone with no derivable limit produces a ``skipped``
entry naming the lever, never silence.

**Per field, not all-or-nothing.**  A custom ``f_ck`` on one zone must
not disturb the other zone, nor any other threshold.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from shapely.geometry import Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

#: The bench geometry and materials, deliberately the same numbers the
#: numeric fingerprint's ``checks.multizone`` bench uses.  Two probes
#: that disagree about the section they measure cannot corroborate each
#: other.
B, H, HT = 300.0, 600.0, 150.0
FCK_BASE, FCK_TOP = 45.0, 30.0
AS_, AP_ = 800.0, 1500.0
EPS_PE = 0.0006
MX = -350.0e6

#: EC2 §7.2(2) characteristic coefficient.  Written here as a literal
#: because a fixture that reads its expectation from the code under
#: test asserts only that the code agrees with itself.
K1 = 0.6


def _service_law(fck):
    """
    The EN 1992-1-1 (3.14) service law for a zone of strength *fck*.

    Returns
    -------
    ServiceConcrete
    """
    from gensec.materials.ec2_bridge import fben2
    from gensec.materials.service_concrete import ServiceConcrete

    p = fben2(fck=fck, ls="F", loadtype="slow", TypeConc="N")
    return ServiceConcrete(fcm=p.fcm, Ecm=p.ecm,
                           eps_c1=abs(float(p.eps_c1)),
                           eps_cu1=0.0035, fct_eff=float(p.fctm),
                           reference_age=28.0, name="phase32 svc")


def _bench():
    """
    The two-zone bench: C45 base, C30 topping, one bonded tendon.

    Returns
    -------
    tuple
        ``(section, x_ref, y_ref, moduli)``.
    """
    from gensec.geometry.fiber import RebarLayer, Tendon
    from gensec.geometry.geometry import GenericSection
    from gensec.materials.concrete import Concrete
    from gensec.materials.steel import PrestressingSteel, Steel

    base = Concrete(fck=FCK_BASE, gamma_c=1.5,
                    long_term_reduction_factor=0.85)
    base.name = "C45"
    topping = Concrete(fck=FCK_TOP, gamma_c=1.5,
                       long_term_reduction_factor=0.85)
    topping.name = "C30"
    st = Steel(fyk=450.0)
    st.name = "B450C"
    ps = PrestressingSteel(f_p01d=1391.3, Ep=195000.0)
    ps.name = "Y1860"
    poly = Polygon([(0.0, 0.0), (B, 0.0), (B, H), (0.0, H)])
    top = Polygon([(0.0, H - HT), (B, H - HT), (B, H), (0.0, H)])
    xr, yr = B / 2.0, H / 2.0
    reb = [RebarLayer(y=50.0, As=AS_, material=st, x=xr),
           RebarLayer(y=H - 50.0, As=AS_, material=st, x=xr)]
    ten = [Tendon(y=100.0, Ap=AP_, material=ps, x=xr,
                  eps_pe=EPS_PE, name="T1")]
    section = GenericSection(poly, base, reb, tendons=ten,
                             bulk_materials=[(top, topping, "topping")],
                             n_grid_y=60)
    moduli = {"C45": _service_law(FCK_BASE), "C30": _service_law(FCK_TOP)}
    return section, xr, yr, moduli


def _run(limits):
    """
    Run the bench through ``verify_sls_staged`` with *limits*.

    Returns
    -------
    dict
        The single stage record.
    """
    from gensec.solver.section_state import SectionState
    from gensec.solver.sls import verify_sls_staged

    section, xr, yr, moduli = _bench()
    state = SectionState(0, np.ones(3, dtype=bool), np.ones(3, dtype=bool),
                         np.array([0.0, 0.0, EPS_PE], dtype=float),
                         bulk_eps_init=0.0, label="mz")
    stages = [{"name": "mz", "state": state,
               "increment": (0.0, MX, 0.0), "limits": limits}]
    res = verify_sls_staged(section, stages, moduli=moduli,
                            x_ref=xr, y_ref=yr)
    return res["stages"][0]


def _concrete_checks(stage):
    """Every emitted record whose bare name is ``concrete_compression``."""
    return [c for c in stage.get("checks", []) or []
            if c.get("name", "").split("[")[0] == "concrete_compression"]


def _by_name(stage, zone):
    """The concrete record belonging to *zone*, or ``None``."""
    for c in _concrete_checks(stage):
        if c.get("name", "").endswith(f"[{zone}]"):
            return c
    return None


# ==================================================================
#  The deletion
# ==================================================================


def test_sigma_c_comp_is_gone_from_the_container():
    """
    ``StressLimits`` no longer carries a section-wide product.

    Asserted on the dataclass fields rather than on a constructor call,
    so that a field kept but undocumented is still caught.
    """
    import dataclasses

    from gensec.materials.verification_limits import StressLimits

    names = {f.name for f in dataclasses.fields(StressLimits)}
    assert "sigma_c_comp" not in names, (
        "a section-wide concrete limit has no place in a formula that "
        "derives per zone; keeping it beside the new path and adding a "
        "precedence rule is the move 30_2 §4 forbids by name"
    )
    assert "k_c" in names
    assert "custom_zone_strengths" in names


def test_naming_the_deleted_field_says_what_to_write_instead():
    """
    ``sigma_c_comp=`` raises, and the message carries the replacement.

    The pattern Phase 30 established for ``fct_eff``: a removed field
    whose name still appears in somebody's script must not fail with a
    bare ``TypeError`` naming only the keyword.
    """
    from gensec.materials.verification_limits import StressLimits

    with pytest.raises((TypeError, ValueError)) as excinfo:
        StressLimits(name="x", sigma_c_comp=21.0)
    msg = str(excinfo.value)
    assert "k_c" in msg, (
        "the refusal must name the lever that replaces the deleted "
        "field, or the reader is left to search the source"
    )


def test_the_ec2_provider_returns_a_coefficient_not_a_product():
    """
    ``ec2_stress_limits`` hands over :math:`k_1`, not :math:`k_1 f_{ck}`.

    The provider's job is the normative coefficient.  Which strength it
    multiplies is a property of a zone, and a zone is not visible from
    a provider that builds one object for the whole stage.
    """
    from gensec.materials.verification_limits import ec2_stress_limits

    lim = ec2_stress_limits("characteristic", fyk=450.0, fpk=1860.0)
    assert lim.k_c == pytest.approx(K1)


# ==================================================================
#  A coefficient, not a product
# ==================================================================


@pytest.mark.parametrize("bad", [0.0, -0.6, float("nan"), float("inf")])
def test_a_non_positive_or_non_finite_coefficient_is_refused(bad):
    """A coefficient that is not a finite positive number is refused."""
    from gensec.materials.verification_limits import StressLimits

    with pytest.raises(ValueError):
        StressLimits(name="x", k_c=bad)


def test_a_coefficient_above_one_is_refused_as_a_probable_product():
    """
    ``k_c=21.0`` is refused, and the message says why.

    This is the failure mode the deletion creates: a caller who used to
    write ``sigma_c_comp=0.6*35`` and now writes ``k_c=0.6*35``.  Taken
    at face value it would be multiplied by :math:`f_{ck}` a second
    time and produce a limit of 735 MPa, which no check would ever
    fail.  A silent acceptance here is worse than the defect being
    repaired.
    """
    from gensec.materials.verification_limits import StressLimits

    with pytest.raises(ValueError) as excinfo:
        StressLimits(name="x", k_c=K1 * 35.0)
    assert "coefficient" in str(excinfo.value).lower()


# ==================================================================
#  One check per zone, each against its own material
# ==================================================================


def test_one_concrete_check_per_present_zone():
    """Two zones present, two concrete records, named for their zones."""
    from gensec.materials.verification_limits import StressLimits

    stage = _run(StressLimits(name="mz", k_c=K1, sigma_s_tens=0.8 * 450.0,
                              sigma_p_max=0.75 * 1860.0))
    names = sorted(c["name"] for c in _concrete_checks(stage))
    assert names == ["concrete_compression[base]",
                     "concrete_compression[topping]"]


def test_each_zone_is_checked_against_its_own_characteristic_strength():
    """
    The two limits are :math:`k_1 f_{ck}` of the two materials.

    Not "two different numbers" -- the two *right* numbers, computed
    here from the declared strengths rather than read back from the
    record.
    """
    from gensec.materials.verification_limits import StressLimits

    stage = _run(StressLimits(name="mz", k_c=K1, sigma_s_tens=0.8 * 450.0,
                              sigma_p_max=0.75 * 1860.0))
    assert _by_name(stage, "base")["limit_MPa"] == pytest.approx(
        K1 * FCK_BASE)
    assert _by_name(stage, "topping")["limit_MPa"] == pytest.approx(
        K1 * FCK_TOP)


def test_the_false_pass_becomes_a_fail():
    """
    The topping fails, and the stage is no longer reported verified.

    This is the whole defect in two booleans.  The old behaviour was
    not a slightly wrong number: it was a stage declared verified with
    a zone at 125 % of its own material's prescription.
    """
    from gensec.materials.verification_limits import StressLimits

    stage = _run(StressLimits(name="mz", k_c=K1, sigma_s_tens=0.8 * 450.0,
                              sigma_p_max=0.75 * 1860.0))
    top = _by_name(stage, "topping")
    assert top["value_MPa"] > K1 * FCK_TOP
    assert top["passed"] is False
    assert stage["verified"] is False


def test_the_base_zone_still_passes():
    """
    The stronger zone is unaffected.

    A repair that turned every composite run red would satisfy the
    fixture above and be useless.  The base carries the same stress it
    always did and passes against its own 27.0.
    """
    from gensec.materials.verification_limits import StressLimits

    stage = _run(StressLimits(name="mz", k_c=K1, sigma_s_tens=0.8 * 450.0,
                              sigma_p_max=0.75 * 1860.0))
    assert _by_name(stage, "base")["passed"] is True


def test_the_checked_value_is_the_zones_own_extreme_not_the_sections():
    """
    Each record's value is that zone's extreme, from ``by_zone``.

    Constraint 8: the per-zone surface already exists and is already
    emitted.  A repair that recomputed it would have created a second
    home for one fact; this asserts the record and the check agree,
    which is only possible if there is one.
    """
    from gensec.materials.verification_limits import StressLimits

    stage = _run(StressLimits(name="mz", k_c=K1, sigma_s_tens=0.8 * 450.0,
                              sigma_p_max=0.75 * 1860.0))
    for zone in ("base", "topping"):
        published = stage["concrete"]["by_zone"][zone]["sigma_min_MPa"]
        assert _by_name(stage, zone)["value_MPa"] == pytest.approx(
            max(0.0, -published), abs=0.0, rel=1e-15)


# ==================================================================
#  Provenance
# ==================================================================


def test_a_derived_limit_declares_material_provenance():
    """A limit from the zone's own material says so."""
    from gensec.materials.verification_limits import StressLimits

    stage = _run(StressLimits(name="mz", k_c=K1, sigma_s_tens=0.8 * 450.0,
                              sigma_p_max=0.75 * 1860.0))
    for zone in ("base", "topping"):
        assert _by_name(stage, zone)["provenance"] == "material"


def test_a_custom_strength_declares_custom_provenance_and_moves_the_limit():
    """
    A declared strength changes the limit **and** the provenance.

    Both halves.  A record whose number moved without its provenance
    moving would be the more dangerous of the two failures: the reader
    would take a departure from the code's prescription for the code's
    prescription.
    """
    from gensec.materials.verification_limits import StressLimits

    lim = StressLimits(name="mz", k_c=K1, sigma_s_tens=0.8 * 450.0,
                       sigma_p_max=0.75 * 1860.0,
                       custom_zone_strengths={"topping": {"f_ck": 25.0}})
    stage = _run(lim)
    top = _by_name(stage, "topping")
    assert top["limit_MPa"] == pytest.approx(K1 * 25.0)
    assert top["provenance"] == "custom"


def test_a_custom_strength_on_one_zone_leaves_the_other_alone():
    """
    Per field and per zone, not all-or-nothing.

    Constraint 3.  Declaring the topping's strength must not silently
    discard what the base derives from its own material.
    """
    from gensec.materials.verification_limits import StressLimits

    lim = StressLimits(name="mz", k_c=K1, sigma_s_tens=0.8 * 450.0,
                       sigma_p_max=0.75 * 1860.0,
                       custom_zone_strengths={"topping": {"f_ck": 25.0}})
    stage = _run(lim)
    base = _by_name(stage, "base")
    assert base["limit_MPa"] == pytest.approx(K1 * FCK_BASE)
    assert base["provenance"] == "material"


def test_no_path_applies_the_coefficient_twice():
    """
    Constraint 4: the block is a strength, never a product.

    Asserted as an identity rather than as a number, so it holds
    whatever the coefficient is.
    """
    from gensec.materials.verification_limits import StressLimits

    for k in (0.45, K1, 0.8):
        lim = StressLimits(name="mz", k_c=k,
                           custom_zone_strengths={"topping": {"f_ck": 25.0}})
        stage = _run(lim)
        assert _by_name(stage, "topping")["limit_MPa"] == pytest.approx(
            k * 25.0)
        assert _by_name(stage, "base")["limit_MPa"] == pytest.approx(
            k * FCK_BASE)


def test_an_unknown_field_in_a_custom_zone_block_is_refused():
    """
    An unknown strength field raises, naming itself.

    Same policy as the top level and as ``material_zones``: a field
    that is read by nothing must not be accepted by anything.
    """
    from gensec.materials.verification_limits import StressLimits

    with pytest.raises(ValueError) as excinfo:
        StressLimits(name="mz", k_c=K1,
                     custom_zone_strengths={"topping": {"f_ckk": 25.0}})
    assert "f_ckk" in str(excinfo.value)


# ==================================================================
#  Every branch leaves a record
# ==================================================================


def test_no_coefficient_gives_a_skipped_entry_per_zone_naming_the_lever():
    """
    Obligation B4 on the new axis.

    Without ``k_c`` no limit is derivable for any zone.  That is an
    ordinary state -- a caller checking only steel -- and it is exactly
    the state a reader most needs told, per zone, with the lever named.
    """
    from gensec.materials.verification_limits import StressLimits

    stage = _run(StressLimits(name="mz", sigma_s_tens=0.8 * 450.0))
    recs = _concrete_checks(stage)
    assert sorted(c["name"] for c in recs) == [
        "concrete_compression[base]", "concrete_compression[topping]"]
    for c in recs:
        assert c.get("skipped") is True
        assert "k_c" in c.get("reason", "")


def test_every_present_zone_owns_a_record_in_every_configuration():
    """
    No zone is ever silent, whatever is or is not declared.

    The measurable form of the obligation, run across the branches
    rather than asserted on one of them.
    """
    from gensec.materials.verification_limits import StressLimits

    configs = [
        StressLimits(name="a", k_c=K1),
        StressLimits(name="b", sigma_s_tens=0.8 * 450.0),
        StressLimits(name="c", k_c=K1,
                     custom_zone_strengths={"topping": {"f_ck": 25.0}}),
    ]
    for lim in configs:
        stage = _run(lim)
        named = {c["name"].split("[", 1)[1].rstrip("]")
                 for c in _concrete_checks(stage) if "[" in c["name"]}
        assert named == {"base", "topping"}, (
            f"limit set {lim.name!r} left a zone with no record"
        )


# ==================================================================
#  What deliberately did not move
# ==================================================================


def test_decompression_and_c_dec_stay_on_the_limit_set():
    """
    Constraint 7: neither is a strength, neither is derivable.

    They belong to the combination and to the normative document, not
    to a material, so the per-zone move must leave them where they are.
    """
    import dataclasses

    from gensec.materials.verification_limits import StressLimits

    names = {f.name for f in dataclasses.fields(StressLimits)}
    assert {"decompression", "c_dec"} <= names


def test_steel_and_tendon_checks_are_untouched():
    """
    Rebars and tendons are elements, not zones.

    Their thresholds stay section-wide because the quantity they limit
    is a property of an element's own material, which the element
    already carries.  A repair that swept them into the zone axis would
    have been a change of subject.
    """
    from gensec.materials.verification_limits import StressLimits

    stage = _run(StressLimits(name="mz", k_c=K1, sigma_s_tens=0.8 * 450.0,
                              sigma_p_max=0.75 * 1860.0))
    bare = {c["name"].split("[")[0] for c in stage["checks"]}
    assert "steel_tension" in bare
    assert "tendon_stress" in bare


def test_a_single_zone_section_keeps_the_number_it_always_had():
    """
    The repair must not move a section it was never about.

    On one zone of C35 the derived limit is :math:`0.6 \\times 35 =
    21.0`, which is the number the old section-wide field carried.
    Identical by construction -- and asserted, because "identical by
    construction" is a claim like any other.
    """
    import numpy as _np

    from gensec.geometry.fiber import RebarLayer
    from gensec.geometry.geometry import GenericSection
    from gensec.materials.concrete import Concrete
    from gensec.materials.steel import Steel
    from gensec.materials.verification_limits import StressLimits
    from gensec.solver.section_state import SectionState
    from gensec.solver.sls import verify_sls_staged

    conc = Concrete(fck=35.0, gamma_c=1.5, long_term_reduction_factor=0.85)
    conc.name = "C35"
    st = Steel(fyk=450.0)
    st.name = "B450C"
    poly = Polygon([(0.0, 0.0), (300.0, 0.0), (300.0, 600.0), (0.0, 600.0)])
    reb = [RebarLayer(y=50.0, As=800.0, material=st, x=150.0)]
    section = GenericSection(poly, conc, reb, n_grid_y=40)
    state = SectionState(0, _np.ones(1, dtype=bool), _np.ones(1, dtype=bool),
                         _np.zeros(1, dtype=float), bulk_eps_init=0.0,
                         label="s")
    stages = [{"name": "s", "state": state,
               "increment": (0.0, -90.0e6, 0.0),
               "limits": StressLimits(name="s", k_c=K1)}]
    res = verify_sls_staged(section, stages,
                            moduli={"C35": _service_law(35.0)},
                            x_ref=150.0, y_ref=300.0)
    recs = _concrete_checks(res["stages"][0])
    assert len(recs) == 1
    assert recs[0]["name"] == "concrete_compression[base]"
    assert recs[0]["limit_MPa"] == pytest.approx(21.0)
# GENSEC-COMPLETE phase=32 utc=2026-09-05T13:56:13Z sha256=b012b6a1cf13d0ed87cc16795389f3030dbe42eba41e1b516f20deaed3743aa8 body_lines=593
