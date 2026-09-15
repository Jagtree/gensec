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
Phase 26, F-P26-0 — age consistency of EN 1992-1-1 expression (3.14).

The physics first, the clause afterwards
----------------------------------------
A concrete of age :math:`t` has one stiffness, one mean strength and one
strain at peak.  Expression (3.14) is a single curve through those three
facts:

.. math::

    \sigma_c = f_{cm}\,
        \frac{k\eta - \eta^{2}}{1 + (k-2)\eta},
    \qquad
    \eta = \frac{\varepsilon_c}{\varepsilon_{c1}},
    \qquad
    k = \frac{1.05\,E_{cm}\,\lvert\varepsilon_{c1}\rvert}{f_{cm}} .

The shape parameter :math:`k` is what ties the curve's slope at the
origin to the concrete's modulus: substituting :math:`\eta\to0` gives

.. math::

    \left.\frac{d\sigma_c}{d\varepsilon_c}\right|_{0}
        = \frac{f_{cm}\,k}{\varepsilon_{c1}}
        = 1.05\,E_{cm},

the initial tangent, which exceeds :math:`E_{cm}` by exactly 5 % because
:math:`E_{cm}` is the *secant* at :math:`0.4 f_{cm}` and not the tangent
at the origin.  That identity is a property of the curve, not of the
code that prints it: it must hold at **every** age at which the three
constants are evaluated, because a concrete does not know what day the
clause was written for.

What this file measures
-----------------------
Until Phase 26, :func:`~gensec.materials.ec2_properties.en2sargin` built
:math:`k` on ``fcm_28`` while scaling the stress by ``fcm``, which is
:math:`f_{cm}(t)`.  At :math:`t = 28\,\mathrm{d}` the two coincide and
the identity holds; away from it the numerator ages and the denominator
does not, and the curve's initial tangent drifts:

===========  ==========================
:math:`t`    :math:`E_0 / E_{cm}(t)`
===========  ==========================
3 d          0.6282
7 d          0.8177
14 d         0.9467
28 d         1.0500
===========  ==========================

which is 40 %, 22 % and 10 % too soft at exactly the ages at which a
tendon is stressed.  Note what the defect does **not** disturb: at
:math:`\eta = 1` the fraction collapses to :math:`(k-1)/(k-1) = 1` for
*any* :math:`k`, so the peak is at :math:`f_{cm}(t)` whatever the shape
parameter.  A probe placed on the peak is structurally blind to this
family of error, which is one reason it survived.

The probes below are written on the initial tangent and on :math:`k`
read back by algebra, never on the peak.
"""

import numpy as np
import pytest

from gensec.materials.ec2_properties import fben2, en2sargin


#: Ages at which the identity is checked [days].  ``3`` is the lower
#: edge of :class:`fben2`'s domain; ``28`` is the only age at which the
#: pre-Phase-26 form was correct, and it is kept precisely so that a
#: regression that re-pins the law to 28 days cannot hide here.
AGES = (3.0, 7.0, 14.0, 28.0, 90.0, 365.0)

#: Relative tolerance on the initial tangent.  The tangent is taken by
#: central difference on ``en2sargin`` itself, so the tolerance covers
#: the truncation error of the difference and nothing else.
RTOL_TANGENT = 1.0e-6

#: Step of the central difference [-].  Chosen so that the truncation
#: error :math:`O(h^2 \sigma''')` and the cancellation error
#: :math:`O(\epsilon_{mach}\sigma/h)` are both far below
#: :data:`RTOL_TANGENT`.
H_DIFF = 1.0e-9


def _props(age, fck=35.0):
    r"""EN 1992-1-1 constants at ``age``, through their only home."""
    return fben2(fck=float(fck), ls="F", loadtype="slow", TypeConc="N",
                 time=float(age))


def _initial_tangent(props):
    r"""
    :math:`d\sigma/d\varepsilon` at the origin of ``en2sargin``.

    Taken as a central difference **on the shipped function**, so that
    the probe cannot drift away from the law it watches: an analytic
    derivative written here would keep agreeing with itself after the
    law changed.
    """
    return ((en2sargin(H_DIFF, props) - en2sargin(-H_DIFF, props))
            / (2.0 * H_DIFF))


@pytest.mark.parametrize("age", AGES)
def test_initial_tangent_is_1_05_ecm_at_every_age(age):
    r"""
    The curve's slope at the origin is :math:`1.05\,E_{cm}(t)`.

    This is the identity the shape parameter exists to produce.  It is
    checked against :math:`E_{cm}(t)` read from the same ``fben2``
    instance that built the curve, so the two cannot be reconciled by
    moving the modulus.
    """
    props = _props(age)
    measured = _initial_tangent(props)
    expected = 1.05 * props.ecm
    assert measured == pytest.approx(expected, rel=RTOL_TANGENT), (
        f"at t={age:g} d the initial tangent of (3.14) is "
        f"{measured:.6g} MPa = {measured / props.ecm:.4f} Ecm(t), "
        f"not 1.05 Ecm(t) = {expected:.6g} MPa"
    )


@pytest.mark.parametrize("age", AGES)
def test_shape_parameter_is_built_on_the_age_matched_mean_strength(age):
    r"""
    :math:`k` is read back from the curve and compared to
    :math:`1.05 E_{cm}(t)|\varepsilon_{c1}(t)| / f_{cm}(t)`.

    The read-back is algebraic rather than an attribute lookup, because
    :math:`k` is a local of ``en2sargin`` and never leaves it.  Solving
    (3.14) at a single :math:`\eta` for :math:`k` gives

    .. math::

        k = \frac{r\,(1 - 2\eta) + \eta^{2}}{\eta\,(1 - r)},
        \qquad r = \frac{\sigma_c(\eta)}{f_{cm}},

    which is exact wherever :math:`r \neq 1`, i.e. away from the peak —
    and the peak is exactly where the defect is invisible.
    """
    props = _props(age)
    eta = 0.35
    strain = eta * props.eps_c1
    r = en2sargin(strain, props) / props.fcm
    k_back = (r * (1.0 - 2.0 * eta) + eta * eta) / (eta * (1.0 - r))
    k_expected = 1.05 * props.ecm * abs(props.eps_c1) / props.fcm
    assert k_back == pytest.approx(k_expected, rel=1.0e-9), (
        f"at t={age:g} d expression (3.14) carries k={k_back:.6g}; "
        f"the age-matched constants give k={k_expected:.6g} "
        f"(ratio {k_back / k_expected:.4f})"
    )


def test_peak_is_blind_to_the_shape_parameter():
    r"""
    Declared blindness, kept as a probe.

    :math:`\sigma_c(\varepsilon_{c1}) = f_{cm}(t)` for any :math:`k`.
    This test passes both before and after the F-P26-0 repair, and it
    exists to *record* that fact: it is the reason a peak-valued check
    could never have caught the defect, and a future reader who adds
    one must see here why it would prove nothing.
    """
    for age in AGES:
        props = _props(age)
        peak = en2sargin(props.eps_c1, props)
        assert peak == pytest.approx(props.fcm, rel=1.0e-12)


def test_law_refuses_below_the_domain_of_its_constants():
    r"""
    Under 3 days ``fben2`` reports :math:`f_{cm} = 0` as a marker, hence
    :math:`\varepsilon_{c1} = 0`, and :math:`\eta = \varepsilon / 0`.

    A curve through a zero strength is not a soft concrete, it is no
    concrete: the constants are outside the domain their source claims.
    The refusal must name that, as a :class:`ValueError`, and not
    surface as the :class:`ZeroDivisionError` of an arithmetic accident
    — an undeclared input is not unmodelled physics.
    """
    props = _props(1.0)
    assert props.eps_c1 == 0.0, (
        "precondition of this probe: fben2 reports a zero strain at "
        "peak below 3 days"
    )
    with pytest.raises(ValueError) as excinfo:
        en2sargin(-1.0e-4, props)
    message = str(excinfo.value)
    assert "eps_c1" in message or "strain at peak" in message, (
        f"the refusal must name the constant that is missing; got: "
        f"{message!r}"
    )


@pytest.mark.parametrize("age", AGES)
def test_tangent_is_monotone_decreasing_up_to_the_peak(age):
    r"""
    Physical shape check, independent of the defect.

    Concrete softens monotonically from the origin to the peak; a curve
    whose tangent rises anywhere in :math:`[0, \varepsilon_{c1}]` is not
    a concrete, whatever clause produced it.  Kept because the repair
    moves :math:`k`, and a shape parameter far enough from its intended
    value turns (3.14) non-monotone.
    """
    props = _props(age)
    eps = -np.linspace(1.0e-6, abs(props.eps_c1) * 0.98, 40)
    # sigma(e) = -S(-e) in GenSec's convention, so a central difference
    # of sigma is  [S(-e+h) - S(-e-h)] / 2h  and is positive.
    tang = np.array([
        (en2sargin(-(e - H_DIFF), props) - en2sargin(-(e + H_DIFF), props))
        / (2.0 * H_DIFF) for e in eps
    ])
    assert np.all(tang > 0.0), "the tangent of (3.14) must be positive"
    assert np.all(np.diff(tang) < 0.0), (
        f"at t={age:g} d the tangent of (3.14) is not monotone "
        f"decreasing up to the peak"
    )


# ---------------------------------------------------------------------------
# ServiceConcrete — the class the linearisation is replaced by
# ---------------------------------------------------------------------------


def _svc_props(fck=35.0, time=None):
    r"""EN 1992-1-1 Table 3.1 constants, from the module that owns them."""
    from gensec.materials.ec2_properties import fben2
    if time is None:
        return fben2(fck=fck, ls="F", loadtype="slow", TypeConc="N")
    return fben2(fck=fck, ls="F", loadtype="slow", TypeConc="N", time=time)


def _svc_law(fck=35.0, time=None, age=28.0):
    from gensec.materials.service_concrete import ServiceConcrete
    p = _svc_props(fck, time)
    return ServiceConcrete(fcm=p.fcm, Ecm=p.ecm, eps_c1=p.eps_c1,
                           eps_cu1=p.eps_cu1, fct_eff=p.fctm,
                           reference_age=age if time is None else float(time),
                           name=f"C{int(fck)} service"), p


@pytest.mark.parametrize("fck", [25.0, 35.0, 50.0])
@pytest.mark.parametrize("t", [3.0, 7.0, 28.0, 90.0])
def test_initial_tangent_of_the_class_is_1_05_ecm_at_every_age(fck, t):
    r"""
    The identity the shape parameter exists to produce, read off the
    class rather than off the free function.

    The class never writes 1.05 down: it measures the slope of its own
    compression branch at the origin.  If that measurement drifted from
    the curve, this is where it would show.
    """
    law, p = _svc_law(fck, t)
    assert law.initial_tangent == pytest.approx(1.05 * p.ecm, rel=1e-9)


def test_tension_branch_is_the_compression_branch_own_tangent():
    r"""
    No kink at the origin: one material, one slope at zero strain.

    Fitting the tension branch to :math:`E_{cm}` instead would put a
    5 % step in the tangent exactly where the strain plane of a bent
    section crosses zero.
    """
    law, p = _svc_law()
    left = law.tangent(-1.0e-12)
    right = law.tangent(+1.0e-12)
    assert left == pytest.approx(right, rel=1e-9)
    assert right == pytest.approx(1.05 * p.ecm, rel=1e-9)
    assert right != pytest.approx(p.ecm, rel=1e-3)


def test_the_cap_is_at_the_declared_tensile_strength():
    r"""``eps_max`` is a real limit, not a scan hint."""
    law, p = _svc_law()
    assert law.eps_ct == pytest.approx(p.fctm / (1.05 * p.ecm), rel=1e-12)
    assert law.eps_max == pytest.approx(law.eps_ct, rel=0.0)
    assert law.stress(law.eps_ct) == pytest.approx(p.fctm, rel=1e-9)
    assert law.stress(law.eps_ct * (1.0 + 1.0e-9)) == 0.0
    assert law.tangent(law.eps_ct * (1.0 + 1.0e-9)) == 0.0


def test_below_cracking_the_capped_law_is_the_uncapped_svc_law():
    r"""
    The cap costs nothing until the concrete reaches its strength.

    Every strain below :math:`\varepsilon_{ct}` is carried by the same
    straight line either way; this is why capping is not a change of
    model in the uncracked range.
    """
    law, _ = _svc_law()
    e = np.linspace(0.0, law.eps_ct, 41)
    assert law.stress_array(e) == pytest.approx(law.initial_tangent * e,
                                                rel=1e-12, abs=1e-15)


def test_compression_branch_is_expression_3_14_and_not_a_copy_of_it():
    r"""
    The class forms no curve: it calls the one home of (3.14).

    Checked by evaluating the free function directly and comparing,
    element by element, in GenSec's sign convention.
    """
    from gensec.materials.ec2_properties import en2sargin
    law, p = _svc_law()
    e = np.linspace(-abs(p.eps_cu1), 0.0, 61)
    direct = np.array([-en2sargin(float(-v), p) for v in e])
    assert law.stress_array(e) == pytest.approx(direct, rel=1e-12,
                                                abs=1e-15)


def test_class_tangent_matches_a_central_difference_of_its_own_stress():
    r"""The reported slope cannot drift from the curve it differentiates."""
    law, p = _svc_law()
    e = np.linspace(-0.9 * abs(p.eps_cu1), -1.0e-6, 80)
    h = 1.0e-9
    fd = (law.stress_array(e + h) - law.stress_array(e - h)) / (2.0 * h)
    assert law.tangent_array(e) == pytest.approx(fd, rel=1.0e-6)


@pytest.mark.parametrize(
    "field", ["fcm", "Ecm", "eps_c1", "eps_cu1", "reference_age"])
def test_every_constant_must_be_declared_and_positive(field):
    r"""
    No silent default anywhere on the surface.

    A zero is not a soft concrete: below three days ``fben2`` reports
    ``fcm = 0`` as its own out-of-domain marker, and an undeclared
    input is refused with ``ValueError`` rather than carried.

    ``fct_eff`` is deliberately absent from this list and is covered by
    :func:`test_a_zero_tensile_strength_is_a_declaration_not_an_omission`
    below.
    """
    from gensec.materials.service_concrete import ServiceConcrete
    p = _svc_props()
    kw = dict(fcm=p.fcm, Ecm=p.ecm, eps_c1=p.eps_c1, eps_cu1=p.eps_cu1,
              fct_eff=p.fctm, reference_age=28.0)
    kw[field] = 0.0
    with pytest.raises(ValueError) as exc:
        ServiceConcrete(**kw)
    assert field in str(exc.value)


def test_the_law_refuses_the_constants_of_a_one_day_concrete():
    r"""The out-of-domain marker travels as a refusal, not as a curve."""
    from gensec.materials.service_concrete import ServiceConcrete
    p = _svc_props(time=1.0)
    with pytest.raises(ValueError):
        ServiceConcrete(fcm=p.fcm, Ecm=p.ecm, eps_c1=p.eps_c1,
                        eps_cu1=p.eps_cu1, fct_eff=p.fctm, reference_age=1.0)


def test_age_is_part_of_the_material_identity():
    r"""
    Two zones of nominally the same concrete cast on different days are
    different materials, and the declared-content token says so.
    """
    a, _ = _svc_law(time=7.0)
    b, _ = _svc_law(time=14.0)
    c, _ = _svc_law(time=7.0)
    assert a.content_token() == c.content_token()
    assert a.content_token() != b.content_token()


def test_a_zero_tensile_strength_is_a_declaration_not_an_omission():
    r"""
    ``fct_eff = 0`` is the law of a section that carries no tension.

    Every other constant of expression (3.14) is a property of the
    concrete and a zero there means the constants were never
    evaluated.  ``fct_eff`` is not: it is the strength at which the
    tension branch stops, and a model is entitled to declare that it
    stops immediately --- a crack-width check reads the section *at*
    the crack, where the concrete carries nothing.  The cracking
    strain then collapses to zero and the tension branch has no width,
    which is a law, not a missing input.
    """
    from gensec.materials.service_concrete import ServiceConcrete
    p = _svc_props()
    law = ServiceConcrete(fcm=p.fcm, Ecm=p.ecm, eps_c1=p.eps_c1,
                          eps_cu1=p.eps_cu1, fct_eff=0.0, reference_age=28.0)
    assert law.eps_ct == 0.0
    assert law.eps_max == 0.0
    assert law.stress(1.0e-6) == 0.0
    assert law.stress(-1.0e-4) < 0.0
    with pytest.raises(ValueError) as exc:
        ServiceConcrete(fcm=p.fcm, Ecm=p.ecm, eps_c1=p.eps_c1,
                        eps_cu1=p.eps_cu1, fct_eff=-1.0, reference_age=28.0)
    assert "fct_eff" in str(exc.value)
