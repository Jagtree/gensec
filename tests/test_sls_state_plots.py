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
``section_sls_*`` / ``state_sls_*`` maps for the SLS staged path.

Until this phase ``gensec run`` drew ``section_<demand>.png`` /
``state_<demand>.png`` for every fibre-detailed ULS demand
(:mod:`gensec.output.state_plots`) but nothing at all for the SLS
stress path (:func:`gensec.solver.sls.verify_sls_staged`, wired into
the CLI as ``run_sls``): a user checking service stresses got numbers
with no picture, while the same demand checked at the ultimate limit
state got two.  ``verify_sls_staged`` now carries a ``"fiber_field"``
on every stage record -- the already-accumulated bulk/rebar
strain-stress arrays in the shape
:meth:`~gensec.solver.integrator.FiberSolver.get_fiber_results`
returns, plus the stage's materialised geometry -- and
:mod:`gensec.output.sls_state_plots` repackages it into the same
plotting functions the ULS path uses.  These tests check the wiring:
the field is shaped correctly, and the exporter writes the three files
for every checked stage under the documented naming convention.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

import run_phase7_sls_validation as bench  # noqa: E402

from gensec.materials.verification_limits import ec2_stress_limits  # noqa: E402
from gensec.solver.sls import verify_sls_staged  # noqa: E402
from gensec.output.sls_state_plots import export_sls_state_plots  # noqa: E402


def _one_stage_result(name="s1", decompression=False):
    section = bench._build_section()
    state = bench._state([1, 1, 1], [1, 1, 1], bench.EPS_PE, 0)
    limits = ec2_stress_limits("characteristic", fyk=450.0, fpk=1860.0,
                               decompression=decompression)
    return verify_sls_staged(
        section,
        [dict(name=name, state=state,
             increment=(0.0, 600.0e6, 0.0), limits=limits)],
        moduli={"C35": bench.EC}, x_ref=bench.XR, y_ref=bench.YR)


def test_fiber_field_is_populated_and_shaped_for_plotting():
    result = _one_stage_result()
    stage = result["stages"][0]
    assert "fiber_field" in stage

    ff = stage["fiber_field"]
    bulk = ff["bulk"]
    for key in ("x", "y", "eps", "sigma", "dA"):
        assert key in bulk
    n = len(bulk["x"])
    assert n > 0
    assert all(len(bulk[k]) == n for k in ("y", "eps", "sigma", "dA"))

    rebars = ff["rebars"]
    # Both rebar layers of the bench section are active and bonded.
    assert len(rebars["x"]) == 2
    for key in ("y", "eps", "sigma", "A"):
        assert len(rebars[key]) == 2

    sec = ff["section_view"]
    assert len(sec.rebars) == 2


def test_export_writes_the_three_files_per_checked_stage(tmp_path):
    result = _one_stage_result(name="s1")
    sls_out = {"combinations": [
        {"name": "combo1", "points": {"-": result}},
    ]}

    records = export_sls_state_plots(sls_out, str(tmp_path))

    assert len(records) == 1
    assert records[0]["name"] == "sls_combo1_s1"
    expected = {
        tmp_path / "fibers_sls_combo1_s1.csv",
        tmp_path / "section_sls_combo1_s1.png",
        tmp_path / "state_sls_combo1_s1.png",
    }
    for p in expected:
        assert p.exists(), p
        assert p.stat().st_size > 0


def test_anchored_point_name_enters_the_label(tmp_path):
    result = _one_stage_result(name="s1")
    sls_out = {"combinations": [
        {"name": "combo1", "points": {"service": result}},
    ]}

    records = export_sls_state_plots(sls_out, str(tmp_path))

    assert records[0]["name"] == "sls_combo1_service_s1"
    assert (tmp_path / "section_sls_combo1_service_s1.png").exists()


def test_skipped_combination_produces_no_files(tmp_path):
    sls_out = {"combinations": [
        {"name": "combo1", "skipped": True, "reason": "no sls block"},
    ]}

    records = export_sls_state_plots(sls_out, str(tmp_path))

    assert records == []
    assert list(tmp_path.iterdir()) == []


def test_stage_without_limits_is_not_plotted(tmp_path):
    section = bench._build_section()
    state = bench._state([1, 1, 1], [1, 1, 1], bench.EPS_PE, 0)
    # No 'limits' key: the stage accumulates but produces no verdict,
    # so it must not appear in the report or on disk.
    result = verify_sls_staged(
        section,
        [dict(name="s1", state=state, increment=(0.0, 600.0e6, 0.0))],
        moduli={"C35": bench.EC}, x_ref=bench.XR, y_ref=bench.YR)
    assert "checks" not in result["stages"][0]

    sls_out = {"combinations": [
        {"name": "combo1", "points": {"-": result}},
    ]}
    records = export_sls_state_plots(sls_out, str(tmp_path))

    assert records == []
    assert list(tmp_path.iterdir()) == []
