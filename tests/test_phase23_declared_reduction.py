# ---------------------------------------------------------------------------
#  GenSec — fibre-based sectional analysis of concrete sections
#  Copyright (C) 2025  Andrea
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as
#  published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public
#  License along with this program.  If not, see
#  <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------
r"""
Phase 23 — the design-strength reduction must be **declared**.

Defect F-P23-1.  :class:`~gensec.materials.concrete.Concrete` is the
*aggregator* of a concrete's constitutive data.  It is not the owner of
any normative constant, and it must not invent one.  Before this phase
it carried three container-side defaults

.. code-block:: text

    fck      = 25.0     an entire concrete class, out of nothing
    gamma_c  = 1.5      EN 1992-1-1 §2.4.2.4, recommended value
    alpha_cc = 0.85     EN 1992-1-1 §3.1.6(1)P, recommended value

so that an input which declared none of them still received a design
strength — :math:`f_{cd} = 14.167` MPa, a C25/30 nobody asked for —
and an input that declared a code other than the Eurocode still
received the Eurocode's two coefficients.

The architecture this phase installs is the one Andrea specified: a
normative parameter enters GenSec **only** through the module dedicated
to the normative document that owns it, and the container merely
aggregates what it is handed.  The Eurocode symbol
:math:`\alpha_{cc}` therefore stays in
:mod:`gensec.materials.ec2_properties`, which is its home, and the
container carries a neutrally named field,
``long_term_reduction_factor``, whose value that module supplies.  The
neutral name is not cosmetic: it leaves room for
:math:`\alpha_{ct}` on the tension branch, and it does not oblige a
non-Eurocode provider to fill a field named after a clause it does not
contain.

What this file asserts, and what it deliberately does not
--------------------------------------------------------

It asserts that the three values must be **declared**, that each
refusal **names** what is missing, and that the factor reaches a
Eurocode material from the Eurocode module.  It does **not** assert any
new number: the algebra

.. math::

   f_{cd} \;=\; k_{lt}\,\frac{f_{ck}}{\gamma_{c}}

is unchanged, and :func:`test_declared_algebra_is_unchanged` is written
to be green on the tree that precedes this phase as well as on the one
that follows it.  A capacity was taken away; nothing was recomputed.
"""

import dataclasses
import os
import tempfile

import pytest

from gensec.materials.concrete import Concrete


# ======================================================================
#  The container refuses what it is not told
# ======================================================================

def _factor_field():
    """
    The container's own name for the long-term factor.

    Read off the dataclass so that the two behavioural fixtures below
    are green on the tree that precedes this phase as well as on the
    one that follows it.  A fixture that hard-coded the new name would
    fail on the earlier tree with ``TypeError: unexpected keyword
    argument``, which demonstrates that a name is missing and not that
    a number is right.
    """
    names = {f.name for f in dataclasses.fields(Concrete)}
    return ("long_term_reduction_factor"
            if "long_term_reduction_factor" in names else "alpha_cc")


def _field_names():
    """Field names of :class:`Concrete`, read off the dataclass."""
    return {f.name for f in dataclasses.fields(Concrete)}


def test_container_field_is_neutrally_named():
    """
    The aggregator carries ``long_term_reduction_factor``.

    The Eurocode spelling ``alpha_cc`` must not survive on the
    container: a field named after EN 1992-1-1 §3.1.6(1)P on an object
    that is supposed to be normatively agnostic is a declaration by
    the back door.
    """
    names = _field_names()
    assert "long_term_reduction_factor" in names, (
        "Concrete must carry the neutrally named long-term reduction "
        f"factor; fields are {sorted(names)}")
    assert "alpha_cc" not in names, (
        "the EN 1992-1-1 symbol alpha_cc belongs to ec2_properties, "
        "not to the aggregator")


@pytest.mark.parametrize("missing", ["fck", "gamma_c",
                                     "long_term_reduction_factor"])
def test_no_container_side_default(missing):
    """
    None of the three carries a default.

    Read off the dataclass rather than inferred from behaviour, so the
    assertion fails on a tree that reintroduces a default even if some
    other guard happens to mask it at run time.
    """
    fields = {f.name: f for f in dataclasses.fields(Concrete)}
    assert missing in fields, f"Concrete has no field {missing!r}"
    f = fields[missing]
    assert f.default is dataclasses.MISSING, (
        f"Concrete.{missing} carries the container-side default "
        f"{f.default!r}; a normative value must be supplied by the "
        f"module that owns the clause, never invented here")
    assert f.default_factory is dataclasses.MISSING, (
        f"Concrete.{missing} carries a default factory")


def test_bare_container_is_refused():
    """
    ``Concrete()`` raises, and the refusal names every missing field.

    The previous behaviour was to build a C25/30 in silence.
    """
    with pytest.raises(TypeError) as excinfo:
        Concrete()
    msg = str(excinfo.value)
    for name in ("fck", "gamma_c", "long_term_reduction_factor"):
        assert name in msg, (
            f"the refusal does not name {name!r}: {msg!r}")


def test_partial_declaration_is_refused():
    """
    Declaring two of the three is still not declaring the third.

    This is the case the container used to answer with the Eurocode's
    recommended 0.85 regardless of which code the model declared.
    """
    with pytest.raises(TypeError) as excinfo:
        Concrete(fck=40.0, gamma_c=1.5)
    assert "long_term_reduction_factor" in str(excinfo.value)


@pytest.mark.parametrize("bad", [0.0, -0.5])
def test_non_positive_factor_is_refused_by_name(bad):
    """
    A declared but nonsensical factor is refused with its reason.

    A zero long-term factor would publish :math:`f_{cd} = 0` and every
    downstream capacity would come out zero without anything being
    raised, which is the silent-failure shape this project refuses.
    """
    with pytest.raises(ValueError) as excinfo:
        Concrete(fck=40.0, gamma_c=1.5, long_term_reduction_factor=bad)
    msg = str(excinfo.value)
    assert "long_term_reduction_factor" in msg
    assert "positive" in msg.lower()


# ======================================================================
#  The algebra did not move
# ======================================================================

def test_declared_algebra_is_unchanged():
    r"""
    A fully declared material publishes the same :math:`f_{cd}` as before.

    Green on the tree that precedes this phase *and* on the one that
    follows it — on the earlier tree the three values were reachable as
    keyword arguments under their older names, and the product is the
    same product.  This fixture is the record that the repair removed a
    capacity and recomputed nothing.
    """
    kw = {"fck": 40.0, "gamma_c": 1.5,
          _factor_field(): 0.85}
    mat = Concrete(**kw)
    assert mat.fcd == pytest.approx(0.85 * 40.0 / 1.5, rel=1e-15)
    assert mat.fcd == pytest.approx(22.666666666666668, rel=1e-15)


def test_factor_is_carried_not_recomputed():
    """
    The container stores what it was handed, unmodified.

    A container that adjusted the value it was given would be owning
    the clause after all.
    """
    kw = {"fck": 35.0, "gamma_c": 1.4, _factor_field(): 1.0}
    mat = Concrete(**kw)
    assert getattr(mat, _factor_field()) == 1.0
    assert mat.fcd == pytest.approx(35.0 / 1.4, rel=1e-15)


# ======================================================================
#  The Eurocode module supplies the Eurocode's value
# ======================================================================

def test_ec2_bridge_supplies_the_factor():
    r"""
    A material built through the EC2 bridge receives
    :math:`\alpha_{cc}` from :mod:`gensec.materials.ec2_properties`.

    This is the positive half of the architecture: taking the default
    away from the container is only correct if the module that owns
    the clause hands the value over.
    """
    from gensec.materials.ec2_bridge import concrete_from_class

    mat = concrete_from_class("C30/37", ls='F', loadtype='slow', TypeConc='R', NA='French', time=28)
    # Read through the container's own field name, so that the red on
    # the preceding tree is an *assertion* about where the value comes
    # from rather than an AttributeError about a name that is missing.
    assert _factor_field() == "long_term_reduction_factor", (
        "the bridge cannot pass a value into a neutrally named field "
        "while the container still names the field after the clause")
    assert getattr(mat, _factor_field()) == pytest.approx(
        mat.ec2.alpha_cc, rel=1e-15), (
        "the bridge must pass the EC2 module's alpha_cc through to the "
        "aggregator's neutral field")
    assert mat.fcd == pytest.approx(
        mat.ec2.alpha_cc * mat.fck / mat.gamma_c, rel=1e-12)


# ======================================================================
#  The loader refuses, by name, and with a remedy
# ======================================================================

_YAML = """schema_version: "1.0"
code: ec2
materials:
  c:
    type: concrete_ec2_gen1_custom
{matblock}  s:
    type: steel
    fyk: 450.0
    gamma_s: 1.15
    Es: 200000.0
section:
  B: 300
  H: 600
  bulk_material: c
  n_fibers_y: 20
  rebars:
    - y: 50
      As: 1000.0
      material: s
      n_bars: 3
      diameter: 20.0
demands:
  - name: D
    N: 0.0
    Mx: 100.0e6
"""


def _load(matblock):
    """Load a one-material model from an inline material block."""
    from gensec.io_yaml import load_yaml

    with tempfile.NamedTemporaryFile("w", suffix=".yaml",
                                     delete=False,
                                     encoding="utf-8") as fh:
        fh.write(_YAML.format(matblock=matblock))
        path = fh.name
    try:
        return load_yaml(path)
    finally:
        os.unlink(path)


def test_loader_accepts_a_fully_declared_material():
    """The complete declaration still loads, and reaches the same value."""
    model = _load("    fck: 30.0\n"
                  "    gamma_c: 1.5\n"
                  "    long_term_reduction_factor: 0.85\n")
    mat = model["materials"]["c"]
    assert mat.fcd == pytest.approx(0.85 * 30.0 / 1.5, rel=1e-15)


@pytest.mark.parametrize("matblock,missing", [
    ("    gamma_c: 1.5\n    long_term_reduction_factor: 0.85\n", "fck"),
    ("    fck: 30.0\n    long_term_reduction_factor: 0.85\n", "gamma_c"),
    ("    fck: 30.0\n    gamma_c: 1.5\n", "long_term_reduction_factor"),
])
def test_loader_refuses_undeclared_field_by_name(matblock, missing):
    """
    An input that omits one of the three is refused, and told which.

    Before this phase every one of these three loaded, and the omitted
    value arrived from the container's own signature.
    """
    with pytest.raises(ValueError) as excinfo:
        _load(matblock)
    msg = str(excinfo.value)
    assert missing in msg, f"the refusal does not name {missing!r}: {msg}"
    assert "'c'" in msg or '"c"' in msg, (
        "the refusal must name the material it is about")


def test_loader_refuses_nothing_declared():
    """
    A material block with no numbers at all is refused.

    This is measurement M3 case A of the phase-23 opening: on the
    preceding tree it loaded, and produced a C25/30.
    """
    with pytest.raises(ValueError) as excinfo:
        _load("")
    assert "fck" in str(excinfo.value)


def test_loader_refuses_the_eurocode_symbol():
    r"""
    ``alpha_cc:`` in the input is refused, naming the neutral field.

    Accepting it silently would put a Eurocode clause name in an
    input that may declare any code, and ignoring it silently would
    be worse: the user would believe they had declared the factor.
    """
    with pytest.raises(ValueError) as excinfo:
        _load("    fck: 30.0\n    gamma_c: 1.5\n    alpha_cc: 0.85\n")
    msg = str(excinfo.value)
    assert "alpha_cc" in msg
    assert "long_term_reduction_factor" in msg
