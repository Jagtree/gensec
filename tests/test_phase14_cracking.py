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
# along with GenSec. If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------

r"""
Phase 14 — the first-cracking marker on a multi-zone section.

The defect these tests exist to exhibit (**F-M1**): the cracking strain
:math:`\varepsilon_{cr}` was resolved once, from ``sec.bulk_material``
— zone 0 — and then applied to the *pooled* bulk fibres of every zone.
On a composite section the zone cast later is a different concrete of a
different age, hence a different :math:`f_{ctm}` and a different
:math:`E_{cm}`, so the pooled comparison asks the wrong question of
every fibre outside zone 0.

Two symptoms follow from that single cause, and they are separable:

* **the marker is present and wrong** — every zone has EC2 properties,
  so the chain resolves, but the topping is checked against the
  precast's threshold.  On ``example_composite_topping`` the precast is
  C45 and the topping C30: the threshold applied to the topping is
  18.6 % too high, and the reported cracking moment is therefore
  **non-conservative**;
* **the marker is absent** — zone 0 carries no cracking information at
  all, the chain falls to its third branch, and detection is skipped
  for the whole section even when another zone *does* carry the data.

The second symptom is the one that was reported as «the marker
disappears».  It is not a separate defect: it is the first one seen in
the case where the wrong material happens to be the one with no data.

Reference discipline
--------------------
The expected values are not read back from the implementation.  Each
threshold is recomputed here from the zone's own EC2 properties, and
the assertions are stated on the *strain field the public result
already reports* (``eps_max`` per curvature step), so that a test can
fail without sharing the code path it is testing.
"""

import os
import unittest

import numpy as np

from gensec.geometry import GenericSection, RebarLayer, RectSection
from gensec.io_yaml import load_yaml
from gensec.materials import Concrete, Steel, concrete_from_class
from gensec.solver import FiberSolver, NMDiagram


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_EXAMPLES = os.path.join(_ROOT, "examples")

#: Axial level at which the cracking threshold is actually resolved by
#: the curvature grid.  At :math:`N = 0` this section cracks below the
#: first curvature step, so the reported marker is grid-saturated and
#: *cannot* discriminate between two thresholds — a fixture that used
#: pure bending would pass before and after the correction and would
#: therefore test nothing.
_N_LEVEL = -1000.0e3

#: Curvature steps.  Fine enough that the cracking step is resolved,
#: coarse enough to keep the fixture inside a few seconds.
_N_CHI = 400


def _eps_cr(material):
    r"""
    Mean cracking strain of a material, from its own EC2 properties.

    Parameters
    ----------
    material : Material
        A material carrying an ``ec2`` properties object.

    Returns
    -------
    float
        :math:`\varepsilon_{cr} = f_{ctm} / E_{cm}`.
    """
    ec2 = material.ec2
    return float(ec2.fctm) / float(ec2.ecm)


def _mixed_zone_section():
    r"""
    Composite section whose **base** zone carries no cracking data.

    Zone 0 is a plain :class:`~gensec.materials.concrete.Concrete`
    (``fct = 0``, no ``ec2``); the topping is an EC2 C30/37.  The
    section therefore has cracking information — just not in zone 0.

    This is the fixture that separates «the marker is missing» from
    «the marker is computed on the wrong material»: a chain that reads
    zone 0 finds nothing and reports ``no_ec2_properties`` for the
    whole section, while the topping is sitting on a fully declared
    :math:`f_{ctm}` the analysis never asks for.

    Returns
    -------
    RectSection
    """
    from shapely.geometry import Polygon

    base = Concrete(fck=45.0, gamma_c=1.5, long_term_reduction_factor=0.85)                     # no ec2, fct = 0
    topping = concrete_from_class("C30/37", ls="F", loadtype="slow",
                                  TypeConc="R", NA='French', time=28)
    steel = Steel(fyk=450.0)
    As = np.pi / 4 * 20 ** 2
    rebars = [
        RebarLayer(y=60, As=4 * As, material=steel),
        RebarLayer(y=1340, As=2 * As, material=steel),
    ]
    return GenericSection(
        polygon=Polygon([(0, 0), (600, 0), (600, 1400), (0, 1400)]),
        bulk_material=base,
        rebars=rebars,
        mesh_size=50.0,
        bulk_materials=[
            (Polygon([(0, 1200), (600, 1200), (600, 1400), (0, 1400)]),
             topping, "topping"),
        ],
    )


class TestZoneAwareCrackingThreshold(unittest.TestCase):
    r"""
    F-M1 — the threshold must belong to the zone that is cracking.
    """

    @classmethod
    def setUpClass(cls):
        document = load_yaml(
            os.path.join(_EXAMPLES, "example_composite_topping.yaml"))
        cls.sec = document["section"]
        cls.mc = NMDiagram(FiberSolver(cls.sec)).generate_moment_curvature(
            N_fixed=_N_LEVEL, n_chi=_N_CHI, direction="x")
        cls.eps_cr_base = _eps_cr(cls.sec.bulk_material)
        cls.eps_cr_topping = _eps_cr(cls.sec.bulk_materials[0][1])

    def test_the_two_zones_have_different_thresholds(self):
        r"""
        Guard on the fixture itself.

        If the two zones ever came to share a threshold this file would
        keep passing while testing nothing, so the premise is asserted
        rather than assumed.
        """
        self.assertGreater(self.eps_cr_base, self.eps_cr_topping,
                           "fixture premise lost: the precast no longer "
                           "has the higher cracking strain")
        self.assertGreater(self.eps_cr_base / self.eps_cr_topping, 1.10,
                           "fixture premise weakened: the two thresholds "
                           "are too close to discriminate")

    def test_the_topping_governs_the_positive_direction(self):
        r"""
        Guard on the geometry: for :math:`\chi_x > 0` the extreme
        tensile bulk fibre is in the topping, so the positive branch is
        the one whose threshold the correction must move.
        """
        mat_indices = np.asarray(self.sec.mat_indices)
        ly = self.sec.y_fibers - FiberSolver(self.sec).y_ref
        self.assertEqual(int(mat_indices[int(np.argmax(ly))]), 1,
                         "the topmost bulk fibre is no longer in the "
                         "topping zone")

    def test_positive_cracking_is_detected_at_the_topping_threshold(self):
        r"""
        The cracking step in the positive direction must be the **first**
        step at which the extreme tensile fibre reaches the *topping's*
        threshold — not the precast's.

        Stated on the reported strain field: the step before the marker
        must still be below the topping's threshold.  Before the
        correction the marker fires at the precast's (higher) threshold,
        so the preceding step has already passed the topping's and this
        assertion fails.
        """
        chi = np.asarray(self.mc["chi"])
        eps_max = np.asarray(self.mc["eps_max"])
        cracking_chi = self.mc["cracking_chi_pos"]
        self.assertIsNotNone(cracking_chi,
                             "no marker at all on a section whose every "
                             "zone declares EC2 properties")

        k = int(np.argmin(np.abs(chi - cracking_chi)))
        self.assertGreaterEqual(
            eps_max[k], self.eps_cr_topping,
            "the reported cracking step is below the topping's own "
            "cracking strain")
        self.assertLess(
            eps_max[k - 1], self.eps_cr_topping,
            f"the step before the marker (eps={eps_max[k - 1]:.6e}) has "
            f"already exceeded the topping's cracking strain "
            f"({self.eps_cr_topping:.6e}): the marker was placed with a "
            f"threshold that is not the cracking zone's")

    def test_negative_cracking_stays_on_the_base_threshold(self):
        r"""
        The discriminating half of the pair.

        For :math:`\chi_x < 0` the extreme tensile fibre is in zone 0,
        whose threshold the chain already used, so a *correct*
        zone-aware rule must leave this branch exactly where it was.  A
        correction that merely lowered the threshold everywhere — for
        instance to the minimum over zones — would move this marker and
        fail here.
        """
        chi = np.asarray(self.mc["chi"])
        eps_max = np.asarray(self.mc["eps_max"])
        cracking_chi = self.mc["cracking_chi_neg"]
        self.assertIsNotNone(cracking_chi)

        k = int(np.argmin(np.abs(chi - cracking_chi)))
        self.assertGreaterEqual(eps_max[k], self.eps_cr_base)
        self.assertLess(
            eps_max[k + 1], self.eps_cr_base,
            "the negative-direction marker moved off the base zone's "
            "threshold: the rule is no longer zone-aware, it is merely "
            "more permissive")


class TestMarkerAbsence(unittest.TestCase):
    r"""
    When the marker is absent, the absence must be the section's, not
    zone 0's.
    """

    def test_no_zone_has_data_is_still_honest(self):
        r"""
        Non-regression pin on the honest branch.

        On the flagship both zones are plain ``Concrete`` with
        ``fct = 0``: there is genuinely no cracking information
        anywhere, and reporting no marker with a named reason is the
        correct behaviour.  The correction must not invent one.
        """
        document = load_yaml(
            os.path.join(_EXAMPLES, "example_composite_losses.yaml"))
        sec = document["section"]
        mc = NMDiagram(FiberSolver(sec)).generate_moment_curvature(
            N_fixed=0.0, n_chi=60, direction="x")
        for suffix in ("pos", "neg"):
            self.assertIsNone(mc[f"cracking_chi_{suffix}"])
            self.assertEqual(
                (mc[f"diagnostics_{suffix}"] or {}).get("cracking_reason"),
                "no_ec2_properties")

    def test_a_zone_with_data_is_not_silenced_by_zone_zero(self):
        r"""
        The strict improvement, and the sharpest statement of F-M1.

        Zone 0 carries no cracking data; the topping carries a full EC2
        declaration.  Reading only zone 0 reports
        ``no_ec2_properties`` for the entire section and discards a
        declared :math:`f_{ctm}` — an absence asserted about a section
        that is not absent.
        """
        sec = _mixed_zone_section()
        mc = NMDiagram(FiberSolver(sec)).generate_moment_curvature(
            N_fixed=0.0, n_chi=200, direction="x")
        reason = (mc["diagnostics_pos"] or {}).get("cracking_reason")
        self.assertIsNotNone(
            mc["cracking_chi_pos"],
            f"no marker although the topping declares its own cracking "
            f"strain; reported reason was {reason!r}")


if __name__ == "__main__":
    unittest.main()
