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
Section-state plots for the SLS staged verification path.

Why this module exists
-----------------------
``gensec run`` has always drawn ``section_<demand>.png`` /
``state_<demand>.png`` for the ULS demands selected for fibre-level
detail (:mod:`gensec.output.state_plots`).  The SLS stress path
(:func:`gensec.solver.sls.verify_sls_staged`, wired into the CLI as
``run_sls`` in Phase 33) never drew anything: it printed a text report
and stopped.  A user checking crack widths and concrete/steel stresses
at service therefore got numbers with no picture, while the exact same
demand checked at the ultimate limit state got two.  That was not a
deliberate scope choice, only a feature that was never connected once
the SLS engine grew its own verification path.

What is drawn, and from what
-----------------------------
Nothing is re-solved here.  ``verify_sls_staged`` already accumulates
the per-fibre strain field and reads every material's own law at it
(Phase 28); the result of that accumulation is carried on each stage
record under ``"fiber_field"`` — the bulk and rebar arrays in the same
shape :meth:`~gensec.solver.integrator.FiberSolver.get_fiber_results`
returns, plus the stage's materialised (active-only) geometry under
``"section_view"``.  This module only repackages those arrays into the
``fiber_results`` dict :func:`~gensec.output.plots.plot_section` and
:func:`~gensec.output.plots.plot_stress_profile` already know how to
draw.  Re-solving via a fresh :class:`~gensec.solver.integrator.FiberSolver`
call on the stage's plane would not reproduce the same field: the SLS
walk accumulates *strain* stage by stage precisely because a single
solve at the cumulative demand does not see the locked-in datum a zone
carried at the moment it entered (module docstring of
:func:`gensec.solver.sls.verify_sls_staged`).

What gets a picture
--------------------
Every stage that declares an ``sls.limits`` block, i.e. every stage
that produced a verdict (``"checks" in stage``).  A stage that only
accumulates state (an intermediate construction step with no check
requested) is not plotted, mirroring the SLS report itself, which
gives such a stage no verdict either.

Naming
------
``fibers_sls_<combo>[_<point>]_<stage>.csv``,
``section_sls_<combo>[_<point>]_<stage>.png``,
``state_sls_<combo>[_<point>]_<stage>.png``.  The ``sls_`` token keeps
these grouped and visibly distinct from the ULS demand files
(``section_<demand>.png``, possibly ``_boundary``) in the same output
directory; the point segment is omitted for an unanchored combination
(``point == "-"``), matching the ``@ <point>`` convention the console
report already uses.
"""

import os

import matplotlib.pyplot as plt

from .export import export_fiber_results_csv
from .plots import plot_section, plot_stress_profile


def _label(combo_name, point_name, stage_name):
    parts = [combo_name]
    if point_name != "-":
        parts.append(point_name)
    parts.append(stage_name)
    return "sls_" + "_".join(parts)


def export_sls_state_plots(sls_out, outdir, show_rebar_labels=True):
    r"""
    Write fibre tables and section/state maps for every checked SLS
    stage.

    Parameters
    ----------
    sls_out : dict
        Output of :func:`gensec.solver.timeline_run.run_sls`.
    outdir : str
        Output directory; must exist.
    show_rebar_labels : bool, optional
        Annotate each rebar with its index and field value. Default
        ``True``.

    Returns
    -------
    list of dict
        One record per plotted stage, in encounter order:

        ``name`` : str
            The label the files were written under.
        ``files`` : list of str
            Paths written (fibre CSV, section PNG, state PNG).
    """
    records = []
    for combo in sls_out.get("combinations", []):
        if combo.get("skipped"):
            continue
        for pname, point in combo["points"].items():
            for stage in point["stages"]:
                if "checks" not in stage:
                    continue
                ff = stage["fiber_field"]
                fr = {"bulk": ff["bulk"], "rebars": ff["rebars"]}
                sec = ff["section_view"]
                label = _label(combo["name"], pname, stage["name"])

                files = []
                p = os.path.join(outdir, f"fibers_{label}.csv")
                export_fiber_results_csv(fr, p)
                print(f"    Exported: {p}")
                files.append(p)

                cum = stage["cumulative"]
                at = "" if pname == "-" else f" @ {pname}"
                heading = f"SLS {combo['name']}{at} / {stage['name']}"
                title = (f"{heading}: N={cum['N_kN']:.0f}, "
                        f"Mx={cum['Mx_kNm']:.0f}, "
                        f"My={cum['My_kNm']:.0f} kNm")

                fig = plot_section(sec, fr, title=title,
                                   show_rebar_labels=show_rebar_labels)
                p = os.path.join(outdir, f"section_{label}.png")
                fig.savefig(p, dpi=150)
                plt.close(fig)
                print(f"  Exported: {p}")
                files.append(p)

                fig = plot_stress_profile(
                    fr, sec, title=title,
                    show_rebar_labels=show_rebar_labels)
                p = os.path.join(outdir, f"state_{label}.png")
                fig.savefig(p, dpi=150)
                plt.close(fig)
                print(f"    Exported: {p}")
                files.append(p)

                records.append({"name": label, "files": files})

    if records:
        print(f"\n  SLS state plots: {len(records)} stage(s) plotted.")
    return records
