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
Declared validity domains (Phase 12, step 0a).

Why this exists
---------------
``linearity_limit`` used to return ``0.45 * max(f_ck(t), 0)``.  Below
three days ``fben2`` returns ``f_ck(t) = 0`` — EN 1992-1-1 §3.1.2(5)
does not define it there — so the limit silently became **zero**, and a
limit of zero is not a limit: it is a tripwire that any compression
whatsoever crosses.  Measured on the shipped flagship, the one-day
topping cleared it with a margin of exactly ``0.0000`` MPa, and cleared
it only because its most-compressed fibre happened to be in *tension*.

The fix is deliberately **not** an EC2-shaped one.  A floor of three
days is a fact about one document, not about GenSec: ACI 209R-92 states
:math:`f'_c(t) = t/(a+bt)\, f'_c(28)` from :math:`t \to 0^{+}` and has
no such edge, and a tabulated provider is defined exactly over its own
grid.  So the provider declares *its own* interval, per quantity, with
a citation, and refuses outside it; and the escape hatch —
:class:`~gensec.materials.base.DeclaredRule` — lives on the abstract
base so that every provider inherits it.

What is asserted here
---------------------
1. EC2 declares an interval and cites it; ACI-shaped providers declare
   none and are untouched.  (The agnosticism falsification.)
2. A declaration is mandatory-provenance, immutable, and enters the
   content token — two analyses differing only in a declared rule are
   two different analyses.
3. Precedence is ``declared -> provider -> raise``, the same rule
   :func:`~gensec.solver.sls.resolve_moduli` applies to moduli.
4. The container asks the provider **only when compression exists**, so
   a stress-free zone never walks to the edge of a declared domain.
5. The guard names where its number came from.
"""

import math

import numpy as np
import pytest

from gensec.materials.base import (
    DeclaredRule, OutOfDeclaredDomain, RheologicalModel,
)
from gensec.materials.rheology import (
    EC2RheologicalModel, TabulatedRheologicalModel,
)
from gensec.solver.losses import _check_linearity


AC, U = 300.0 * 600.0, 2.0 * (300.0 + 600.0)


@pytest.fixture
def ec2():
    return EC2RheologicalModel(fck=30, cement_class="N").with_geometry(
        A_c=AC, u=U)


# ======================================================================
#  1.  The domain is the provider's, and it is cited
# ======================================================================

class TestDeclaredDomain:

    def test_ec2_declares_the_three_day_floor(self, ec2):
        assert ec2.domain("linearity_limit") == (3.0, math.inf)
        assert ec2.domain("f_ck") == (3.0, math.inf)

    def test_the_floor_carries_a_citation(self, ec2):
        assert "3.1.2(5)" in ec2.domain_source("f_ck")
        assert "EN 1992-1-1" in ec2.domain_source("linearity_limit")

    def test_the_modulus_is_not_bounded(self, ec2):
        r"""§3.1.3(3) builds E_cm(t) from beta_cc, not through f_cm, and
        is stated at every age.  Bounding it would break every young
        zone in every timeline."""
        assert ec2.domain("E_c") is None
        assert ec2.E_c(1.0) > 0.0
        assert ec2.E_c(0.5) > 0.0

    def test_inside_the_domain_nothing_changed(self, ec2):
        for t in (3.0, 7.0, 28.0, 90.0, 365.0):
            assert ec2.linearity_limit(t) == pytest.approx(
                0.45 * max(ec2._p(t).fck, 0.0), rel=1e-12)

    def test_outside_it_refuses(self, ec2):
        with pytest.raises(OutOfDeclaredDomain) as exc:
            ec2.linearity_limit(1.0)
        assert exc.value.quantity == "linearity_limit"
        assert exc.value.t == 1.0
        assert exc.value.domain == (3.0, math.inf)

    def test_the_refusal_is_a_scope_boundary_not_a_bad_input(self, ec2):
        r"""An age of one day is legitimate; GenSec simply has no
        authority to answer.  That is NotImplementedError territory,
        not ValueError."""
        assert issubclass(OutOfDeclaredDomain, NotImplementedError)
        assert not issubclass(OutOfDeclaredDomain, ValueError)

    def test_the_refusal_says_what_to_do(self, ec2):
        with pytest.raises(OutOfDeclaredDomain) as exc:
            ec2.linearity_limit(1.0)
        msg = str(exc.value)
        assert "DeclaredRule" in msg
        assert "3.1.2(5)" in msg

    def test_zero_is_never_returned_below_the_floor(self, ec2):
        r"""The whole point.  A zero limit is a tripwire, and a guard
        that fires on a quantity the code never defined reports the
        wrong cause."""
        for t in (0.5, 1.0, 2.0, 2.999):
            with pytest.raises(OutOfDeclaredDomain):
                ec2.linearity_limit(t)


# ======================================================================
#  2.  Agnosticism: a provider with no such edge keeps none
# ======================================================================

class _ACIShaped(RheologicalModel):
    r"""
    Minimal ACI 209R-92-shaped provider.

    ACI's strength development :math:`f'_c(t) = t/(a+bt)\, f'_c(28)` is
    stated from :math:`t \to 0^{+}`, so this provider declares **no**
    domain at all.  It is the falsification of the EC2 floor: if the
    three-day edge had been written into GenSec rather than into the
    EC2 provider, this class would inherit a limit its own code does
    not impose.
    """

    CONTENT_FIELDS = ("fc_28",)

    def __init__(self, fc_28=30.0, name="aci209"):
        self.fc_28 = float(fc_28)
        self.name = name

    def fc(self, t):
        return float(t) / (4.0 + 0.85 * float(t)) * self.fc_28

    def J(self, t, t_prime):
        return 1.0 / self.E_c(t_prime)

    def eps_imposed(self, t, t_s):
        return 0.0

    def relaxation(self, t, mu):
        return 0.0

    def E_c(self, t):
        return 4700.0 * math.sqrt(self.fc(t))

    def service_law(self, t):
        r"""
        Refuse: this provider declares ACI's strength development and
        no ACI constitutive law.

        The refusal is part of the falsification, not an omission in
        it.  If ``service_law`` returned EN 1992-1-1 (3.14) by
        inheritance or by convenience, an ACI-shaped rheology would
        answer strains with a Eurocode law, which is the very
        cross-contamination this class exists to detect.
        """
        raise NotImplementedError(
            "_ACIShaped declares ACI 209R-92 strength development and "
            "no constitutive law; borrowing EN 1992-1-1 (3.14) would "
            "attribute a code this provider has not declared."
        )

    def linearity_limit(self, t):
        return 0.40 * self.fc(t)


class TestNormativeAgnosticism:

    def test_an_aci_shaped_provider_declares_no_floor(self):
        aci = _ACIShaped()
        assert aci.domain("linearity_limit") is None
        assert aci.states("linearity_limit", 0.5)

    def test_and_therefore_answers_at_one_day(self):
        aci = _ACIShaped()
        assert aci.linearity_limit(1.0) == pytest.approx(
            0.40 * 30.0 / 4.85, rel=1e-12)

    def test_the_number_three_lives_only_in_the_ec2_provider(self):
        r"""Rule 5, applied to a normative datum instead of a formula:
        the floor has one home."""
        assert 3.0 in {lo for lo, _ in EC2RheologicalModel.DOMAIN.values()}
        assert RheologicalModel.DOMAIN == {}

    def test_a_tabulated_provider_publishes_its_own_grid(self):
        tp = np.array([7.0, 28.0, 90.0])
        tt = np.array([7.0, 28.0, 90.0, 365.0])
        J = np.array([[1.0 / 30000.0 * (1.0 + 0.1 * i + 0.05 * k)
                       for k in range(3)] for i in range(4)])
        tab = TabulatedRheologicalModel(t_prime_grid=tp, t_grid=tt,
                                        J_table=J, name="lab")
        assert tab.domain("J") == (7.0, 365.0)
        assert "tabulated data" in tab.domain_source("J")

    def test_extrapolate_widens_the_published_domain(self):
        tp = np.array([7.0, 28.0])
        tt = np.array([7.0, 28.0])
        J = np.full((2, 2), 1.0 / 30000.0)
        tab = TabulatedRheologicalModel(t_prime_grid=tp, t_grid=tt,
                                        J_table=J, extrapolate=True)
        assert tab.domain("J") is None


# ======================================================================
#  3.  A declaration: provenance, immutability, identity
# ======================================================================

class TestDeclaredRule:

    def test_source_is_mandatory(self):
        with pytest.raises(ValueError, match="source is mandatory"):
            DeclaredRule(value=1.0, source="")
        with pytest.raises(ValueError, match="source is mandatory"):
            DeclaredRule(value=1.0, source=None)

    def test_it_is_immutable(self):
        r = DeclaredRule(value=1.0, source="judgement")
        with pytest.raises(AttributeError):
            r.value = 2.0

    def test_a_constant_rule(self):
        r = DeclaredRule(value=1.0, source="judgement", domain=(0.0, 3.0))
        assert r.at(1.0) == 1.0
        assert r.covers(0.0) and r.covers(3.0)
        assert not r.covers(3.001)

    def test_a_callable_rule(self):
        r = DeclaredRule(value=lambda t: 0.5 * t, source="paper X",
                         domain=(0.0, 3.0))
        assert r.at(2.0) == pytest.approx(1.0)

    def test_a_rule_refuses_outside_its_own_domain(self):
        r = DeclaredRule(value=1.0, source="judgement", domain=(0.0, 3.0))
        with pytest.raises(ValueError, match="does not cover"):
            r.at(10.0)

    def test_a_non_finite_callable_is_caught(self):
        r = DeclaredRule(value=lambda t: float("nan"), source="broken")
        with pytest.raises(ValueError, match="non-finite"):
            r.at(1.0)


class TestDeclarationIsContent:

    def test_a_declaration_changes_the_content_token(self, ec2):
        other = ec2.with_declared(
            linearity_limit=DeclaredRule(1.0, "judgement", (0.0, 3.0)))
        assert ec2.content_token() != other.content_token()

    def test_the_source_alone_changes_it(self, ec2):
        a = ec2.with_declared(
            linearity_limit=DeclaredRule(1.0, "annex A", (0.0, 3.0)))
        b = ec2.with_declared(
            linearity_limit=DeclaredRule(1.0, "annex B", (0.0, 3.0)))
        assert a.content_token() != b.content_token()

    def test_the_token_does_not_depend_on_declaration_order(self, ec2):
        r1 = DeclaredRule(1.0, "s1", (0.0, 3.0))
        r2 = DeclaredRule(2.0, "s2", (0.0, 3.0))
        a = ec2.with_declared(linearity_limit=r1, f_ck=r2)
        b = ec2.with_declared(f_ck=r2, linearity_limit=r1)
        assert a.content_token() == b.content_token()

    def test_binding_is_late_and_does_not_mutate(self, ec2):
        ec2.with_declared(
            linearity_limit=DeclaredRule(1.0, "judgement", (0.0, 3.0)))
        assert ec2.declared == {}
        with pytest.raises(OutOfDeclaredDomain):
            ec2.linearity_limit(1.0)

    def test_only_declared_rules_are_accepted(self, ec2):
        with pytest.raises(TypeError, match="must be a DeclaredRule"):
            ec2.with_declared(linearity_limit=1.0)


# ======================================================================
#  4.  Precedence: declared -> provider -> raise
# ======================================================================

class TestPrecedence:

    def test_a_declaration_fills_the_gap(self, ec2):
        m = ec2.with_declared(
            linearity_limit=DeclaredRule(1.0, "judgement", (0.0, 3.0)))
        assert m.linearity_limit(1.0) == 1.0

    def test_and_leaves_the_provider_alone_where_it_does_not_reach(self, ec2):
        m = ec2.with_declared(
            linearity_limit=DeclaredRule(1.0, "judgement", (0.0, 3.0)))
        assert m.linearity_limit(28.0) == pytest.approx(13.5, rel=1e-9)

    def test_an_overlapping_declaration_wins(self, ec2):
        r"""The same rule resolve_moduli applies to moduli: an explicit
        value that can be silently disregarded is a silent error."""
        m = ec2.with_declared(
            linearity_limit=DeclaredRule(99.0, "annex", (0.0, math.inf)))
        assert m.linearity_limit(28.0) == 99.0

    def test_a_declaration_narrower_than_the_gap_still_raises_outside(
            self, ec2):
        m = ec2.with_declared(
            linearity_limit=DeclaredRule(1.0, "judgement", (2.0, 3.0)))
        assert m.linearity_limit(2.5) == 1.0
        with pytest.raises(OutOfDeclaredDomain):
            m.linearity_limit(1.0)


# ======================================================================
#  5.  The container asks only when there is something to bound
# ======================================================================

class _Counting(EC2RheologicalModel):
    """EC2 provider that records every linearity_limit query."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.queries = []

    def linearity_limit(self, t):
        self.queries.append(float(t))
        return super().linearity_limit(t)


class TestGuardOrdering:

    def test_a_stress_free_zone_is_never_asked(self):
        p = _Counting(fck=30, cement_class="N")
        _check_linearity(p, "topping", 1.0, 0.0)
        assert p.queries == []

    def test_a_wholly_tensile_zone_is_never_asked(self):
        p = _Counting(fck=30, cement_class="N")
        _check_linearity(p, "topping", 1.0, +0.9110)
        assert p.queries == []

    def test_a_compressed_zone_is_asked(self):
        p = _Counting(fck=30, cement_class="N")
        with pytest.raises(OutOfDeclaredDomain):
            _check_linearity(p, "topping", 1.0, -0.5)
        assert p.queries == [1.0]

    def test_a_compressed_zone_inside_the_domain_passes(self):
        p = _Counting(fck=30, cement_class="N")
        _check_linearity(p, "base", 28.0, -5.0)
        assert p.queries == [28.0]

    def test_the_violation_still_raises(self):
        p = EC2RheologicalModel(fck=30, cement_class="N")
        with pytest.raises(ValueError, match="linear-viscoelasticity"):
            _check_linearity(p, "base", 28.0, -20.0)

    def test_the_message_names_a_declared_provenance(self):
        p = EC2RheologicalModel(fck=30, cement_class="N").with_declared(
            linearity_limit=DeclaredRule(1.0, "project judgement XYZ",
                                         (0.0, 3.0)))
        with pytest.raises(ValueError) as exc:
            _check_linearity(p, "topping", 1.0, -2.26)
        assert "declared by the analyst" in str(exc.value)
        assert "project judgement XYZ" in str(exc.value)

    def test_the_message_names_the_norm_when_the_norm_supplied_it(self):
        p = EC2RheologicalModel(fck=30, cement_class="N")
        with pytest.raises(ValueError) as exc:
            _check_linearity(p, "base", 28.0, -20.0)
        assert "EN 1992-1-1" in str(exc.value)
        assert "declared by the analyst" not in str(exc.value)


# ======================================================================
#  6.  The YAML surface is provider-agnostic
# ======================================================================

YAML_DECLARED = """
losses_models:
  rheo:
    provider: ec2
    fck: 30.0
    cement_class: N
    declared:
      linearity_limit:
        value:  1.0
        domain: [0.0, 3.0]
        source: project judgement, outside EN 1992-1-1 3.1.2(5)
"""

YAML_TABLE = """
losses_models:
  rheo:
    provider: ec2
    fck: 30.0
    declared:
      linearity_limit:
        value:  [[0.0, 0.2], [1.0, 0.6], [3.0, 1.4]]
        domain: [0.0, 3.0]
        source: lab campaign 2026-03
"""

YAML_NO_SOURCE = """
losses_models:
  rheo:
    provider: ec2
    fck: 30.0
    declared:
      linearity_limit:
        value:  1.0
        domain: [0.0, 3.0]
"""


def _models(text):
    import yaml
    from gensec.io_yaml import _parse_losses_models
    return _parse_losses_models(yaml.safe_load(text)["losses_models"])


class TestYamlSurface:

    def test_a_declared_block_round_trips(self):
        p = _models(YAML_DECLARED)["rheo"].provider
        assert p.linearity_limit(1.0) == 1.0
        assert "3.1.2(5)" in p.declared["linearity_limit"].source

    def test_a_tabulated_value_interpolates(self):
        p = _models(YAML_TABLE)["rheo"].provider
        assert p.linearity_limit(1.0) == pytest.approx(0.6)
        assert p.linearity_limit(2.0) == pytest.approx(1.0)

    def test_a_rule_without_a_source_is_refused(self):
        with pytest.raises(ValueError, match="source is mandatory"):
            _models(YAML_NO_SOURCE)

    def test_an_unknown_key_is_refused(self):
        bad = YAML_DECLARED.replace("domain: [0.0, 3.0]", "rangee: [0, 3]")
        # Phase 36: the six hand-written unknown-key refusals in
        # io_yaml were migrated onto one helper, so every one of
        # them now says "unknown key(s)".  The change of wording
        # is declared, not incidental.
        with pytest.raises(ValueError, match=r"unknown key\(s\)"):
            _models(bad)


# ======================================================================
#  7.  The flagship declares its own floor
# ======================================================================

class TestFlagship:

    def test_the_topping_declares_a_floor_with_a_source(self):
        import os
        from gensec.io_yaml import load_yaml
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "examples",
                            "example_composite_losses.yaml")
        data = load_yaml(path)
        p = data["losses_models"]["rheo_topping"].provider
        rule = p.declared.get("linearity_limit")
        assert rule is not None
        assert rule.covers(1.0) and not rule.covers(4.0)
        assert "EN 1992-1-1" in rule.source

    def test_the_floor_still_catches_the_f4_symptom(self):
        r"""2.26 MPa is what the topping reaches when the service datum
        is suppressed.  A floor that let that through would have bought
        nothing."""
        import os
        from gensec.io_yaml import load_yaml
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "examples",
                            "example_composite_losses.yaml")
        p = load_yaml(path)["losses_models"]["rheo_topping"].provider
        assert p.linearity_limit(1.0) < 2.26
