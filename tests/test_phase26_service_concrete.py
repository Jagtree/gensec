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
Phase 26 — :class:`~gensec.materials.service_concrete.ServiceConcrete`.

The physics first, the clause afterwards
----------------------------------------
What is asserted here is what a concrete fibre does, not what a clause
prints.  Four facts define the material at one age — stiffness, mean
strength, strain at peak, tensile strength — and every test below fixes
one consequence of those four:

* the curve passes through its own peak;
* its slope at the origin is the concrete's tangent modulus, approached
  from **both** sides, so the zero strain that every bent section
  contains is not a corner;
* it carries tension until it cracks and nothing after;
* it refuses constants that describe no concrete rather than returning
  a number for them.

The binding test is :func:`test_compression_branch_is_en2sargin`: the
class holds a vectorised transcription of EN 1992-1-1 (3.14) because a
Python call per fibre is not affordable in the integration loop, and
that transcription is tied here to the expression's one home in
:mod:`gensec.materials.ec2_properties`.  Without this test the two
could drift, and a drift in a constitutive law is silent.
"""

import numpy as np
import pytest

from gensec.materials.ec2_properties import fben2, en2sargin
from gensec.materials.service_concrete import (ServiceConcrete,
                                               TANGENT_OVER_SECANT)

#: Relative tolerance binding the transcription to its source.
RTOL_BIND = 1.0e-12

#: Ages at which the law is built.  28 d is where the normative
#: constants are tabulated; 7 d is a transfer age, and it is the age at
#: which F-P26-0 used to put the initial tangent 22 % low.
AGES = (7.0, 28.0, 90.0)


def _props(fck=35.0, t=28.0):
    r"""The normative constants at one age, from their only home."""
    return fben2(fck=fck, ls="F", loadtype="slow", TypeConc="N", time=t)


def _law(fck=35.0, t=28.0):
    r"""A service law built on those constants, deriving nothing."""
    p = _props(fck, t)
    return ServiceConcrete(fcm=float(p.fcm), Ecm=float(p.ecm),
                           eps_c1=float(p.eps_c1), eps_cu1=float(p.eps_cu1),
                           fct_eff=float(p.fctm), reference_age=float(t),
                           name=f"C{int(fck)} service @{t:g}d")


# ---------------------------------------------------------------------------
# the compression branch is the normative expression, not a lookalike
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("t", AGES)
def test_compression_branch_is_en2sargin(t):
    r"""
    The vectorised transcription equals its source at every strain.

    Sampled over the whole compressive range, in GenSec's sign
    convention against ``en2sargin``'s own, which is the conversion
    :math:`\sigma(\varepsilon) = -S(-\varepsilon)` the class performs
    once.
    """
    p, law = _props(t=t), _law(t=t)
    eps = -np.linspace(1.0e-7, float(p.eps_cu1), 2001)
    mine = law.stress_array(eps)
    ref = np.array([-en2sargin(-float(v), p) for v in eps])
    assert mine == pytest.approx(ref, rel=RTOL_BIND, abs=1.0e-12), (
        f"the transcription has drifted from EN 1992-1-1 (3.14) at "
        f"t={t:g} d; largest gap "
        f"{float(np.max(np.abs(mine - ref))):.3e} MPa")


@pytest.mark.parametrize("t", AGES)
def test_curve_passes_through_its_own_peak(t):
    r"""At the strain at peak the stress is the mean strength."""
    p, law = _props(t=t), _law(t=t)
    assert law.stress(-float(p.eps_c1)) == pytest.approx(
        -float(p.fcm), rel=1.0e-12)


# ---------------------------------------------------------------------------
# the slope at the origin, from both sides
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("t", AGES)
def test_initial_tangent_is_1_05_ecm_from_both_sides(t):
    r"""
    The law is :math:`C^1` at zero strain.

    :math:`E_{cm}` is the secant at :math:`0.4 f_{cm}`, so the tangent
    at the origin is 5 % steeper.  Joining a tension branch at
    :math:`E_{cm}` instead would put a slope discontinuity exactly
    where the strain plane of a bent section crosses zero.
    """
    p, law = _props(t=t), _law(t=t)
    expected = TANGENT_OVER_SECANT * float(p.ecm)
    assert law.tangent(0.0) == pytest.approx(expected, rel=1.0e-12), (
        f"at zero strain the tangent is {law.tangent(0.0):.6g} MPa = "
        f"{law.tangent(0.0) / p.ecm:.9f} Ecm, not "
        f"{TANGENT_OVER_SECANT} Ecm")
    # Continuity of the slope across zero.  The two one-sided values
    # are read a strain apart, so they differ by the curvature of the
    # compression branch over that interval and not by a corner: the
    # bound is loose on purpose, and a branch joined at Ecm instead of
    # 1.05 Ecm would miss it by five orders of magnitude.
    left, right = law.tangent(-1.0e-9), law.tangent(1.0e-9)
    assert left == pytest.approx(right, rel=1.0e-5), (
        f"the slope jumps across zero strain: {left:.6g} MPa on the "
        f"compression side, {right:.6g} MPa on the tension side "
        f"({abs(left / right - 1.0) * 100:.4f} %)")


@pytest.mark.parametrize("t", AGES)
def test_analytic_tangent_matches_central_differences(t):
    r"""
    The closed-form tangent is the derivative of the stress it claims.

    Sampled away from the two boundaries of the admissible range and
    away from the peak, where the tangent passes through zero and a
    relative comparison stops meaning anything.
    """
    law = _law(t=t)
    h = 1.0e-9
    eps = np.linspace(-0.9 * abs(law.eps_min), 0.9 * law.eps_max, 1501)
    eps = eps[np.abs(np.abs(eps) - law.eps_c1) > 1.0e-4]
    fd = (law.stress_array(eps + h) - law.stress_array(eps - h)) / (2.0 * h)
    an = law.tangent_array(eps)
    assert an == pytest.approx(fd, rel=1.0e-5), (
        "the analytic tangent is not the derivative of the stress")


# ---------------------------------------------------------------------------
# the tension branch cracks
# ---------------------------------------------------------------------------
def test_tension_is_linear_up_to_the_cracking_strain():
    r"""Below :math:`\varepsilon_{ct}` the branch is the straight line."""
    p, law = _props(), _law()
    assert law.eps_ct == pytest.approx(
        float(p.fctm) / (TANGENT_OVER_SECANT * float(p.ecm)), rel=1.0e-14)
    eps = np.linspace(0.0, law.eps_ct, 101)
    assert law.stress_array(eps) == pytest.approx(
        TANGENT_OVER_SECANT * float(p.ecm) * eps, rel=1.0e-12, abs=1.0e-14)
    assert law.stress(law.eps_ct) == pytest.approx(
        float(p.fctm), rel=1.0e-12)


def test_beyond_the_cracking_strain_the_fibre_carries_nothing():
    r"""
    Above :math:`\varepsilon_{ct}` there is no stress and no stiffness.

    This is what makes a section cracked by arithmetic rather than by a
    downstream verdict, and what stops the law reporting a tensile
    stress no concrete of that strength can carry.
    """
    law = _law()
    for f in (1.0 + 1.0e-9, 1.5, 10.0):
        assert law.stress(f * law.eps_ct) == 0.0
        assert law.tangent(f * law.eps_ct) == 0.0


def test_admissible_range_is_the_real_one():
    r"""
    ``eps_max`` is the cracking strain, not a scan hint.

    Beyond it the material has no state to report, so the limit the
    solver reads is the limit the physics has.
    """
    p, law = _props(), _law()
    assert law.eps_max == pytest.approx(law.eps_ct, rel=0.0)
    assert law.eps_min == pytest.approx(-float(p.eps_cu1), rel=0.0)
    assert law.eps_min < 0.0 < law.eps_max


def test_beyond_the_ultimate_strain_the_fibre_carries_nothing():
    r"""Past :math:`\varepsilon_{cu1}` the compression branch stops."""
    law = _law()
    for f in (1.0 + 1.0e-9, 2.0):
        assert law.stress(f * law.eps_min) == 0.0
        assert law.tangent(f * law.eps_min) == 0.0


# ---------------------------------------------------------------------------
# it refuses what describes no concrete
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field_name", ["fcm", "Ecm", "eps_c1", "eps_cu1",
                                        "fct_eff", "reference_age"])
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_every_constant_must_be_finite_and_positive(field_name, bad):
    r"""
    No constant of this law has a default.

    ``fct_eff = 0`` is the single admissible zero and is skipped here:
    it is a declaration --- a law with no tension branch --- and not a
    constant that was never evaluated.  It has its own fixture in
    ``test_phase26_service_law.py``.
    """
    if field_name == "fct_eff" and bad == 0.0:
        pytest.skip("a declared zero tensile strength is a law, not an "
                    "omission")
    p = _props()
    kwargs = dict(fcm=float(p.fcm), Ecm=float(p.ecm),
                  eps_c1=float(p.eps_c1), eps_cu1=float(p.eps_cu1),
                  fct_eff=float(p.fctm), reference_age=28.0)
    kwargs[field_name] = bad
    with pytest.raises(ValueError, match=field_name):
        ServiceConcrete(**kwargs)


def test_peak_outside_the_admissible_range_is_refused():
    r"""A law that can never reach its own strength is not a law."""
    p = _props()
    with pytest.raises(ValueError, match="eps_cu1"):
        ServiceConcrete(fcm=float(p.fcm), Ecm=float(p.ecm),
                        eps_c1=float(p.eps_c1),
                        eps_cu1=0.5 * float(p.eps_c1),
                        fct_eff=float(p.fctm), reference_age=28.0)


def test_constants_below_three_days_are_refused_by_name():
    r"""
    ``fben2`` reports :math:`f_{cm} = 0` below three days as its own
    out-of-domain marker, and :math:`\varepsilon_{c1}` follows it to
    zero.  Handed those, the law refuses with a ``ValueError`` — an
    undeclared input, not unmodelled physics — and never lets an
    arithmetic accident stand in for the refusal.
    """
    p = _props(t=1.0)
    assert float(p.fcm) == 0.0 and float(p.eps_c1) == 0.0
    with pytest.raises(ValueError, match="fcm"):
        ServiceConcrete(fcm=float(p.fcm), Ecm=float(p.ecm),
                        eps_c1=float(p.eps_c1), eps_cu1=float(p.eps_cu1),
                        fct_eff=1.0, reference_age=1.0)


# ---------------------------------------------------------------------------
# identity and reporting
# ---------------------------------------------------------------------------
def test_declared_content_token_is_stable_and_sensitive():
    r"""
    Two laws built from the same constants are the same material; two
    built at different ages are not.

    The second half matters more than the first: a material's identity
    covers time as well as stress, and two zones of nominally identical
    concrete cast on different days must not collide.
    """
    assert _law(t=28.0).content_token() == _law(t=28.0).content_token()
    assert _law(t=28.0).content_token() != _law(t=7.0).content_token()
    assert _law(fck=35.0).content_token() != _law(fck=40.0).content_token()


def test_reference_modulus_reports_the_secant_and_its_age():
    r"""
    The law reports :math:`E_{cm}` and not its own initial tangent.

    Consumers that ask for "the" modulus are building homogenisation
    ratios and creep bases, and EN 1992-1-1 writes those on the secant.
    """
    p, law = _props(t=7.0), _law(t=7.0)
    rm = law.reference_modulus()
    assert rm.value == pytest.approx(float(p.ecm), rel=0.0)
    assert rm.reference_age == pytest.approx(7.0, rel=0.0)
    assert law.tangent(0.0) == pytest.approx(
        TANGENT_OVER_SECANT * rm.value, rel=1.0e-9)


def test_the_design_law_is_not_an_alternative_to_this_one():
    r"""
    The parabola-rectangle's slope at the origin carries
    :math:`\gamma_c`.

    :math:`E_0 = 2 f_{cd} / \varepsilon_{c2} = 2 \alpha_{cc} f_{ck} /
    (\gamma_c \varepsilon_{c2})` is a function of the partial factor,
    and a stiffness that moves when a safety format moves is not a
    stiffness.  Pinned here as a measurement, not an opinion.
    """
    from gensec.materials.concrete import Concrete
    p = _props()
    design = Concrete(fck=35.0, gamma_c=1.5, long_term_reduction_factor=0.85,
                      n_parabola=2.0, eps_c2=-0.002, eps_cu2=-0.0035)
    ratio = float(design.tangent_array(np.array([0.0]))[0]) / float(p.ecm)
    assert ratio == pytest.approx(0.5820, abs=5.0e-4), (
        f"the design law's initial tangent is {ratio:.4f} Ecm")
    assert _law().tangent(0.0) / float(p.ecm) == pytest.approx(
        TANGENT_OVER_SECANT, rel=1.0e-9)
