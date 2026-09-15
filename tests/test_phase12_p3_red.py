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
The two fixtures P3 has to turn green — written **RED first** (rule 3).

Rule 3: *a test whose fixture cannot exhibit the bug does not test for
it.*  Both defects below have survived every phase so far precisely
because nothing in the suite could show them.

F5 — the grout datum is drawn on the design law
-----------------------------------------------
Every prestress-transfer artifact in the tree builds its section from
``LinearElastic``.  In a linear law the design tangent **is** the
service modulus, so the mismatch F5 describes cannot exist in the
fixture.  Census at the time of writing:

===========================================  ====  ====
file                                         real  mock
===========================================  ====  ====
``test_posttension.py``                         0     5
``test_prestress_transfer.py``                  0     4
``run_posttension_sequence_validation.py``      0     3
``test_validate_prestress_phase1.py``           0     4
===========================================  ====  ====

:class:`TestF5GroutDatum` supplies what was missing: a real,
non-linear ``Concrete(gamma_c=1.5)``, where the initial tangent
:math:`f_{cd}` differs from :math:`E_{cm}` by about 30 %.

The assertion is derived, not tabulated.  A tendon jacked to
:math:`\sigma` and read back at its own grouting datum must read
:math:`\sigma`: the datum is *defined* as the state at which the
tendon carries exactly what the jack put in.  Any residue is

.. math::

    \Delta\sigma \;=\; E_p\,
        \bigl(\varepsilon_{\text{sec,service}}
              - \varepsilon_{\text{sec,design}}\bigr)

at the tendon — the two views disagreeing about how much the section
shortened.  It vanishes if and only if the datum and its read-back use
the same modulus, which is what P3 makes true.

F-A — the branch nothing walks
------------------------------
``timeline.py::_service_datum`` softens a zone with no declared
rheological model to the 1 MPa uncast placeholder, while ``losses.py``
raises for the same zone.  Two files, one question, two answers.
Instrumented over the timeline suite: ``_service_datum`` is entered 16
times, 11 of which return early, and the divergent branch is walked
**zero** times.  :class:`TestFADivergence` is the section that walks it.

Both are marked ``xfail(strict=True)``: the suite stays green while the
defect stands, and the moment P3 fixes it the marker itself fails,
which is what forces its removal instead of leaving a stale annotation
behind.
"""

import os

import pytest

from gensec.io_yaml import load_yaml
from gensec.solver.timeline import ConstructionTimeline


# ======================================================================
#  Fixtures
# ======================================================================

#: A single-zone post-tensioned beam with a **real** non-linear law.
#: gamma_c = 1.5 is what makes the design tangent differ from the
#: service modulus; with gamma_c = 1 the fixture would be back to
#: measuring nothing.
F5_YAML = """
schema_version: 1        # D7: this document declares zone_rheology
# Declared because GenSec derives nothing from an undeclared
# model: the code is not inferred from a material type.
code: ec2
materials:
  C45:
    type: concrete
    fck: 45.0
    gamma_c: 1.5
    long_term_reduction_factor: 0.85
  rebar:
    type: steel
    fyk: 450.0
    gamma_s: 1.15
  strand:
    type: prestressing_steel_ec2
    class: Y1860S7
    ls: F
    diagram: inclined
    NA: EC2
    eps_ud_factor: 0.9

section:
  shape: rect
  params: {B: 400, H: 800}
  bulk_material: C45
  mesh_size: 40
  rebars:
    - {name: A1, x: 60,  y: 60,  diameter: 20, material: rebar}
    - {name: A2, x: 340, y: 60,  diameter: 20, material: rebar}
    - {name: T1, x: 60,  y: 740, diameter: 16, material: rebar}
    - {name: T2, x: 340, y: 740, diameter: 16, material: rebar}
  tendons:
    - name: P1
      x: 200
      y: 120
      Ap: 1500.0
      material: strand
      eps_pe: 0.0
      bonded: true

demands:
  - {name: G1, N_kN: -80, Mx_kNm: -60}

losses_models:
  rheo:
    provider: ec2
    fck: 45.0
    cement_class: R
    RH: 70.0
    chi: lump

# The zone's rheological identity, declared once and outside any
# interval.  Without it the grout -- which happens *before* the first
# interval -- has a provider for the base and no age to evaluate
# E_c(t) at.  Before P3 the stopgap filled that hole with a silent
# 28-day default; here the 28 days are stated, which is the whole
# difference between the two.
zone_rheology:
  base: {model: rheo, age: 28, curing: 3}

construction_history:
  - load:   {demand: G1}
  - stress: {tendons: [P1], sigma_p0: 1250.0}
  - grout:  {tendons: [P1]}
  - point:  transfer
  - interval:
      days: 60
      losses:
        base: {model: rheo, age: 28, curing: 3}
  - point:  service
"""

#: A zone that is **active** and carries no rheological model.  Today
#: ``_service_datum`` quietly gives it the 1 MPa placeholder; after P3
#: the resolver must refuse, with the message that explains why a
#: non-creeping zone still matters.
FA_YAML = """
schema_version: 1        # D7: this document declares zone_rheology
# Declared because GenSec derives nothing from an undeclared
# model: the code is not inferred from a material type.
code: ec2
materials:
  C45:
    type: concrete
    fck: 45.0
    gamma_c: 1.5
    long_term_reduction_factor: 0.85
  C30:
    type: concrete
    fck: 30.0
    gamma_c: 1.5
    long_term_reduction_factor: 0.85
  rebar:
    type: steel
    fyk: 450.0
    gamma_s: 1.15

section:
  shape: rect
  params: {B: 400, H: 600}
  bulk_material: C45
  mesh_size: 50
  material_zones:
    - shape: custom
      name: topping
      material: C30
      params:
        exterior: [[0, 500], [400, 500], [400, 600], [0, 600]]
  rebars:
    - {name: A1, x: 60,  y: 60,  diameter: 20, material: rebar}
    - {name: A2, x: 340, y: 60,  diameter: 20, material: rebar}

demands:
  - {name: G1, N_kN: -60, Mx_kNm: -40}

# Only the topping declares a rheological model.  The base is active,
# load-bearing, and modelless: it does not creep, but it keeps its
# stiffness and therefore restrains the zone that does.
losses_models:
  rheo_top:
    provider: ec2
    fck: 30.0
    cement_class: N
    chi: lump

construction_history:
  - load:   {demand: G1}
  - point:  bare
  - cast:   {zone: topping, datum: auto}
  - interval:
      days: 60
      losses:
        topping: {model: rheo_top, age: 7, curing: 3}
  - point:  service
"""


def _resolve(tmp_path, text, stem):
    r"""
    Load *text* as a YAML model and resolve its construction timeline.

    Parameters
    ----------
    tmp_path : pathlib.Path
        pytest's per-test directory.
    text : str
        The YAML document.
    stem : str
        File stem, for readable failure output.

    Returns
    -------
    tuple
        ``(section, TimelineResolution)``.
    """
    path = tmp_path / f"{stem}.yaml"
    path.write_text(text, encoding="utf-8")
    data = load_yaml(str(path))
    tl = ConstructionTimeline.from_document(data)
    ddb = {d["name"]: d for d in data["demands"]}
    return data["section"], tl.resolve(data["section"], ddb)


# ======================================================================
#  F5 — the read-back must equal the jack
# ======================================================================

class TestF5GroutDatum:

    def test_the_fixture_really_is_non_linear(self, tmp_path):
        r"""Rule 3, checked rather than asserted in a comment: if the
        design tangent equalled the service modulus this whole class
        would be measuring nothing."""
        sec, _ = _resolve(tmp_path, F5_YAML, "f5")
        design_tangent = float(sec.bulk_material.tangent(0.0))
        from gensec.materials.rheology import EC2RheologicalModel
        service = EC2RheologicalModel(fck=45.0,
                                      cement_class="R").E_c(28.0)
        assert abs(design_tangent / service - 1.0) > 0.10

    def test_sigma_p0_read_back_equals_the_jack(self, tmp_path):
        _, res = _resolve(tmp_path, F5_YAML, "f5")
        jacked = res.stress_sigma["P1"]
        read_back = None
        for _, trace in sorted(res.losses_trace.items()):
            for rec in trace:
                sp0 = getattr(rec, "sigma_p0", None)
                if sp0 and "P1" in sp0:
                    read_back = float(sp0["P1"])
                    break
            if read_back is not None:
                break
        assert read_back is not None, "no sigma_p0 recorded"
        assert read_back == pytest.approx(jacked, rel=1e-9)


# ======================================================================
#  F-A — the divergent branch
# ======================================================================

class TestFADivergence:

    def test_the_fixture_really_walks_the_branch(self, tmp_path):
        r"""The zone must be cast, active, and modelless — otherwise the
        divergent branch is not reached and, again, nothing is tested."""
        import gensec.solver.timeline as T
        seen = []
        orig = T.ConstructionTimeline._zone_loss_model

        def spy(self, z, zone_names):
            out = orig(self, z, zone_names)
            seen.append((z, out is None))
            return out

        T.ConstructionTimeline._zone_loss_model = spy
        try:
            _resolve(tmp_path, FA_YAML, "fa")
        except Exception:                                    # noqa: BLE001
            pass
        finally:
            T.ConstructionTimeline._zone_loss_model = orig
        assert any(is_none for _, is_none in seen), (
            "the modelless zone was never looked up — the fixture does "
            "not reach the divergent branch")

    def test_a_modelless_active_zone_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="loss model|rheological"):
            _resolve(tmp_path, FA_YAML, "fa")
