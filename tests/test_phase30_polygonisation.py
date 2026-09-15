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
Phase 30: a discretisation parameter whose name states what it measures.

Where this came from
--------------------
Task A repairs the coordinate a record publishes so that it lands on
material.  That raised the question of what residual error the published
coordinate still carries on a curved boundary, and the answer turned out
to expose a second defect.

A circle declared in YAML is built as a polygon.  The published
coordinate is therefore **exact on the section as declared**; the
distance to the circle the author had in mind is the sagitta of their
own discretisation, and GenSec neither sees that circle nor may guess at
it.  Reporting the sagitta in the record would require inferring that
the polygon *was meant to be* a circle and fitting a radius to it --
inventing an input nobody declared.  So the obligation is documentary:
state the magnitude and let the user compute it.

The defect that surfaced
------------------------
The magnitude cannot be computed from what the code says.
``circle_poly(D, resolution=64)`` forwards ``resolution`` to Shapely as
``quad_segs``, which is segments **per quadrant**, so it produces 256
vertices.  The docstring says "Number of vertices on the circumference.
Default 64."  That is wrong by a factor of four, and on the sagitta

.. math::

    s = R\left(1 - \cos\frac{\pi}{n}\right)

wrong by a factor of sixteen, since :math:`s` is quadratic in
:math:`1/n` for large :math:`n`.  Measured at :math:`R = 600` mm: the
true sagitta is 4.517890e-02 mm, the one a reader computes from the
docstring is 7.227263e-01 mm.  Verified at four values of the
parameter; the ratio is 15.2, 15.8, 16.0, 16.0.

The example set already carries the consequence.  ``example_annulus``
declares ``resolution: 64`` and its author believed the boundary had 64
vertices; it has 256.

The repair
----------
Rename the parameter to ``segments_per_quadrant``, so the factor of four
is in the name and cannot mislead, and make it **mandatory**.  A default
that silently selects a geometric error is the shape this project
refuses everywhere else: the parameter does not tune the accuracy of an
integration over a fixed shape, it selects *which shape the section is*.
A 16-gon is not a circle integrated coarsely; it is a different section.

Enforcement is measured, not assumed: of the 24 distributed examples,
two mention curved boundaries and only one builds any, and that one
already declares the parameter.  Making it mandatory breaks nothing.

Expected state
--------------
RED on the pristine Phase-29 tree, GREEN after the repair.
"""

import math
import os

import pytest

from gensec.geometry import primitives as prim


def _examples_dir():
    """
    Absolute path of the distributed example directory.

    Returns
    -------
    str
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "examples")


def _ring(polygon):
    """
    Exterior vertices of a polygon, without the repeated closing point.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon

    Returns
    -------
    list of (float, float)
    """
    return list(polygon.exterior.coords)[:-1]


# ======================================================================
#  The name must be true
# ======================================================================

@pytest.mark.parametrize("n", [4, 8, 16, 64])
def test_segments_per_quadrant_produces_four_times_as_many_vertices(n):
    r"""
    The parameter counts segments per quadrant, and says so.

    This is the whole point of the rename: a reader who wants the
    sagitta needs :math:`n` in :math:`R(1-\cos(\pi/n))`, and must be
    able to get it from the parameter without discovering a factor of
    four by experiment.
    """
    poly = prim.circle_poly(1200.0, segments_per_quadrant=n)
    assert len(_ring(poly)) == 4 * n


@pytest.mark.parametrize("n", [8, 16, 64])
def test_the_documented_sagitta_formula_matches_the_geometry(n):
    r"""
    The stated magnitude must be the magnitude.

    Checks the polygon against the circle it approximates: the greatest
    distance from the ideal circle to the polygon boundary is the
    sagitta of one side.  Documenting a formula that the geometry does
    not obey would replace a wrong number with a wrong law.
    """
    R = 600.0
    poly = prim.circle_poly(2.0 * R, segments_per_quadrant=n)
    sides = 4 * n
    ring = _ring(poly)
    cx = sum(x for x, _ in ring) / sides
    cy = sum(y for _, y in ring) / sides
    # Deepest point of a side is its midpoint.
    worst = 0.0
    for i in range(sides):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % sides]
        mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        worst = max(worst, R - math.hypot(mx - cx, my - cy))
    expected = R * (1.0 - math.cos(math.pi / sides))
    assert worst == pytest.approx(expected, rel=1.0e-9)


# ======================================================================
#  The declaration is mandatory
# ======================================================================

def test_a_circle_without_a_declared_discretisation_is_refused():
    r"""
    No default may silently choose the section's shape.
    """
    with pytest.raises(TypeError):
        prim.circle_poly(1200.0)


def test_an_annulus_without_a_declared_discretisation_is_refused():
    """
    The same obligation for the annulus.
    """
    with pytest.raises(TypeError):
        prim.annulus_poly(1200.0, 800.0)


def test_the_old_parameter_name_is_gone():
    r"""
    ``resolution`` must not survive as an alias.

    Keeping it would preserve exactly the misreading the rename exists
    to prevent, and there are no users to keep compatible with.
    """
    with pytest.raises(TypeError):
        prim.circle_poly(1200.0, resolution=64)


def test_yaml_refuses_a_curved_shape_without_the_declaration():
    r"""
    The refusal must reach the YAML surface, not only the Python API.

    A parameter that is mandatory in the library and defaulted in the
    loader is not mandatory.
    """
    from gensec.io_yaml import _SHAPE_FACTORIES

    with pytest.raises((KeyError, ValueError, TypeError)):
        _SHAPE_FACTORIES["circle"]({"D": 1200.0})
    with pytest.raises((KeyError, ValueError, TypeError)):
        _SHAPE_FACTORIES["annulus"]({"D_ext": 1200.0, "D_int": 800.0})


def test_the_distributed_annulus_example_declares_it():
    r"""
    The example set must state what it relies on.

    ``example_annulus`` is the only distributed file that builds a
    curved boundary.  It must declare ``segments_per_quadrant`` and
    still produce the geometry it produced before, so that the rename
    is a rename and not a change of section.
    """
    from gensec.io_yaml import load_yaml
    from gensec.solver.sls import _zone_regions

    path = os.path.join(_examples_dir(), "example_annulus.yaml")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    assert "segments_per_quadrant" in text, (
        "example_annulus.yaml does not declare segments_per_quadrant")
    assert "resolution" not in text, (
        "example_annulus.yaml still mentions the old parameter name")

    region = _zone_regions(load_yaml(path)["section"])[0]
    parts = region.geoms if hasattr(region, "geoms") else [region]
    outer = max(parts, key=lambda g: g.area)
    assert len(_ring(outer)) == 256, (
        "the annulus geometry changed: %d exterior vertices, expected 256"
        % len(_ring(outer)))


# ======================================================================
#  Control
# ======================================================================

def test_straight_sided_primitives_are_untouched():
    r"""
    Control: shapes with no curved boundary take no new obligation.

    Without this, a repair that made every primitive demand a
    discretisation argument would pass the tests above while breaking
    every rectangular section in the example set.
    """
    assert len(_ring(prim.rect_poly(300.0, 500.0))) == 4
    assert len(_ring(prim.tee_poly(1000.0, 200.0, 300.0, 600.0))) == 8
# GENSEC-COMPLETE phase=30 utc=2026-09-04T07:30:41Z sha256=224110212e0f428cadcbcb0de358e831ef43baa33969ac1ff82c752c769dc181 body_lines=245
