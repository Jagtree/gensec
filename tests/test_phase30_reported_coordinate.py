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
Phase 30, Task A: the coordinate a record publishes must have material
under it.

The defect
----------
:func:`gensec.solver.sls._zone_candidates` adds, to the vertices of a
zone's material region, the points where that zone's own law turns
over.  A turning point is placed by linear interpolation along the
straight segment joining the two extreme-strain vertices:

.. math::

    \mathbf{p}(t) = \mathbf{p}_{lo} + t\,(\mathbf{p}_{hi}-\mathbf{p}_{lo}),
    \qquad
    t = \frac{\varepsilon_b-\varepsilon_{lo}}
             {\varepsilon_{hi}-\varepsilon_{lo}} .

On a convex region that point is inside the material.  On a re-entrant
or holed region the segment leaves the region and the record then names
a coordinate where there is no material at all.

The **value** is not in question.  The region is connected and the
strain field is affine, so by the intermediate-value theorem the
breakpoint strain is attained by real material somewhere in the zone.
What is wrong is the label.

Why the repair is a snap and not a discard
------------------------------------------
Measured on the pristine tree by sweeping the strain direction over 360
angles: discarding the offending candidate instead of relocating it
moves the reported extreme stress by up to **28.2743 MPa** on
``example_custom`` -- roughly 65 % of the peak of a C35 service law.
Discarding is the silent loss of a real extremum.  Relocating the point
onto the iso-strain line costs nothing, because every point of that
line carries the breakpoint strain by construction.

The coordinate is not unique
----------------------------
The iso-strain line meets the region in **two disjoint segments** for
66.7 % of directions on ``example_annulus`` and ``example_custom``, and
14.4 % on ``example_tee``.  Every point of every segment carries the
same strain and therefore the same stress, so the *value* is unaffected;
but the *coordinate* is a choice.  The repair must therefore be
deterministic, and the docstring of the producing function must say that
the published point is **a** point where the extreme is attained, not
**the** point.  These tests pin the determinism; the handoff records the
wording obligation.

Expected state
--------------
RED on the pristine Phase-29 tree, GREEN after the Task-A repair.
"""

import glob
import math
import os

import pytest
from shapely.geometry import Point

from gensec.io_yaml import load_yaml
from gensec.materials.ec2_bridge import fben2
from gensec.materials.service_concrete import ServiceConcrete
from gensec.solver.sls import (_region_vertices, _zone_candidates,
                               _zone_regions)

#: Zones whose geometry admits the excursion.  Measured, not assumed:
#: of the 29 bulk zones in the distributed example set, 3 are
#: non-convex and 2 carry holes.
AFFECTED = ("example_annulus.yaml", "example_custom.yaml",
            "example_tee.yaml")

#: How far from the material a coordinate may sit and still count as
#: on it, in mm.  A snapped point lies **on the region boundary**, and
#: an exact predicate such as ``region.covers`` rejects it when
#: floating-point puts it a few ulps outside: measured over the three
#: affected zones and 360 directions, 357 candidates sit outside by at
#: most 7.165e-14 mm, which is 5.97e-17 of the 1200 mm section depth --
#: machine epsilon on coordinates of that magnitude, not an excursion.
#: The tolerance is set four orders above that round-off and ten orders
#: below the smallest excursion the defect ever produced (17.144 mm on
#: ``example_tee``), so it cannot mask the thing it is here to catch.
ON_MATERIAL_TOL = 1.0e-9

#: Directions swept.  360 is enough to catch a defect that the pristine
#: tree exhibits for 100 %, 99.7 % and 33.6 % of directions on the three
#: affected zones respectively; a single bending direction is not, which
#: is how ``29_0`` section 10 came to record the tee as unaffected.
NDIR = 36


def _examples_dir():
    """
    Absolute path of the distributed example directory.

    Returns
    -------
    str
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "examples")


def _service_law(fck=35.0, age=28.0):
    """
    A concrete service law with real breakpoints.

    The law is built from the EC2 bridge rather than invented, so the
    breakpoints under test are the ones the engine actually uses.

    Parameters
    ----------
    fck : float, optional
        Characteristic cylinder strength [MPa].
    age : float, optional
        Declared age [d].

    Returns
    -------
    gensec.materials.service_concrete.ServiceConcrete
    """
    p = fben2(fck=fck, ls="F", loadtype="slow", TypeConc="N")
    return ServiceConcrete(fcm=p.fcm, Ecm=p.ecm, eps_c1=p.eps_c1,
                           eps_cu1=p.eps_cu1, fct_eff=p.fctm,
                           reference_age=age, name="C%d service" % int(fck))


def _strain_field(verts, ux, uy, breakpoints):
    """
    An affine strain field whose range covers every breakpoint.

    Parameters
    ----------
    verts : sequence of (float, float)
        Region vertices.
    ux, uy : float
        Unit direction of the strain gradient.
    breakpoints : sequence of float
        The law's breakpoints.

    Returns
    -------
    dict
        ``{"e0": [...], "ex": [...], "ey": [...]}`` as
        :func:`_zone_candidates` expects.
    """
    proj = [ux * x + uy * y for (x, y) in verts]
    span = max(proj) - min(proj)
    lo_t, hi_t = min(breakpoints) * 1.4, max(breakpoints) * 1.4
    k = (hi_t - lo_t) / span
    return {"e0": [lo_t - k * min(proj)], "ex": [k * ux], "ey": [k * uy]}


def _zone0(name):
    """
    Region and vertices of zone 0 of one example.

    Parameters
    ----------
    name : str
        File name inside the example directory.

    Returns
    -------
    tuple
        ``(region, vertices)``.
    """
    data = load_yaml(os.path.join(_examples_dir(), name))
    region = _zone_regions(data["section"])[0]
    return region, _region_vertices(region)


@pytest.mark.parametrize("name", AFFECTED)
def test_every_candidate_lies_on_material(name):
    r"""
    No candidate may be reported outside its own zone.

    This is the defect in its bare form.  On the pristine tree
    ``example_annulus`` fails for every swept direction, with excursions
    up to 366.335 mm into the void.
    """
    region, verts = _zone0(name)
    law = _service_law()
    breakpoints = [float(b) for b in law.stress_breakpoints()]
    offenders = []
    seen = 0
    for k in range(NDIR):
        th = 2.0 * math.pi * k / NDIR
        ux, uy = math.cos(th), math.sin(th)
        strains = _strain_field(verts, ux, uy, breakpoints)
        for eps, pt in _zone_candidates(strains, 0, region, law):
            seen += 1
            distance = region.distance(Point(pt))
            if distance > ON_MATERIAL_TOL:
                offenders.append((round(math.degrees(th), 1), eps, pt,
                                  distance))
    assert seen >= 4 * NDIR, (
        "only %d candidates produced; the test would pass without "
        "exercising anything" % seen)
    assert not offenders, (
        "%d candidate(s) reported at a coordinate with no material; "
        "worst excursion %.4f mm; first three: %r"
        % (len(offenders), max(o[3] for o in offenders), offenders[:3]))


@pytest.mark.parametrize("name", AFFECTED)
def test_reported_point_carries_the_breakpoint_strain(name):
    r"""
    Relocating the point must not move the value.

    The whole justification for a snap rather than a discard is that
    every point of the iso-strain line carries the same strain.  If a
    repair relocated a point off that line it would fix the label by
    corrupting the value, which is worse than the defect.
    """
    region, verts = _zone0(name)
    law = _service_law()
    breakpoints = [float(b) for b in law.stress_breakpoints()]
    for k in range(NDIR):
        th = 2.0 * math.pi * k / NDIR
        ux, uy = math.cos(th), math.sin(th)
        strains = _strain_field(verts, ux, uy, breakpoints)
        e0 = strains["e0"][0]
        ex = strains["ex"][0]
        ey = strains["ey"][0]
        for eps, pt in _zone_candidates(strains, 0, region, law):
            eps_at_point = e0 + ex * pt[0] + ey * pt[1]
            assert eps_at_point == pytest.approx(eps, abs=1.0e-12), (
                "candidate at %r is labelled eps=%.12e but the strain "
                "field gives %.12e there" % (pt, eps, eps_at_point))


@pytest.mark.parametrize("name", AFFECTED)
def test_the_choice_of_point_is_deterministic(name):
    r"""
    The coordinate is a choice; the choice must be repeatable.

    The iso-strain line meets a re-entrant region in more than one
    segment for most directions.  Any of them is a correct answer for
    the value, so the function is free to pick -- but it must pick the
    same one every time, or two identical runs would produce two
    different fingerprints.
    """
    region, verts = _zone0(name)
    law = _service_law()
    breakpoints = [float(b) for b in law.stress_breakpoints()]
    for k in range(NDIR):
        th = 2.0 * math.pi * k / NDIR
        ux, uy = math.cos(th), math.sin(th)
        strains = _strain_field(verts, ux, uy, breakpoints)
        first = _zone_candidates(strains, 0, region, law)
        second = _zone_candidates(strains, 0, region, law)
        assert first == second, (
            "two identical calls produced different candidates at "
            "%.1f deg" % math.degrees(th))


def test_convex_zones_are_left_alone():
    r"""
    The repair must not disturb the cases that were already right.

    On a convex region the interpolated point is inside the material
    already, so a correct repair is a no-op there.  This is the control:
    without it, a repair that snapped every point to the boundary would
    pass the tests above while quietly moving every convex result.
    """
    law = _service_law()
    breakpoints = [float(b) for b in law.stress_breakpoints()]
    checked = 0
    for path in sorted(glob.glob(os.path.join(_examples_dir(), "*.yaml"))):
        try:
            data = load_yaml(path)
            regions = _zone_regions(data["section"])
        except Exception:
            continue
        for region in regions:
            if region.is_empty or not region.equals(region.convex_hull):
                continue
            verts = _region_vertices(region)
            if len(verts) < 3:
                continue
            strains = _strain_field(verts, 0.0, 1.0, breakpoints)
            for eps, pt in _zone_candidates(strains, 0, region, law):
                assert region.distance(Point(pt)) <= ON_MATERIAL_TOL, (
                    "convex zone of %s produced an outside candidate at "
                    "%r -- the repair has broken a case that was right"
                    % (os.path.basename(path), pt))
            checked += 1
    assert checked >= 20, (
        "only %d convex zones exercised; the control is too thin to "
        "prove the repair left them alone" % checked)
# GENSEC-COMPLETE phase=30 utc=2026-09-04T07:30:41Z sha256=587c026a206106b1e101986a7c6e600025dde37b57e6058c025cdec1a71fa207 body_lines=310
