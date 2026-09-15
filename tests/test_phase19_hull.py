r"""
Phase 19 — the scan hull, the axial bracket, and the gate on vacuous
``--selfcheck`` passes.

Every discriminating claim here was seen **false** on the pristine
Phase 18 tree (``0faaee4``, rebuilt from ``gensec_phase18.bundle``)
before the repair that makes it true, and false *on the measured value*
— not on a missing name.  No test in this module imports or names a
symbol introduced by Phase 19, for the reason set out in ``18_1`` §4:
a fixture that dies at import proves that a name is absent, not that a
defect is present.  What is asserted throughout is **behaviour** of
functions that exist on both trees.

Sections
--------
1.  N-16 — ``NMDiagram`` builds its lever-arm hull, and therefore the
    projected depth and the curvature ceiling of every sweep, from
    bulk fibres and rebars only.
2.  N-15 — ``FiberIntegrator._solve_pure_axial`` builds its strain
    scan range from bulk and rebar only, so part of the section's real
    axial tensile capacity has no bracket.
3.  N-3 — no distributed example reached a cracking event on a staged
    section.
4.  The health-list gate on ``--selfcheck`` passes that compare zero
    bytes.

Measured figures quoted in the docstrings below were taken on this
container, Python 3.12, with ``PYTHONPATH=src MPLBACKEND=Agg``.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from gensec.io_yaml import load_yaml
from gensec.solver.capacity import NMDiagram
from gensec.solver.integrator import FiberSolver

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))


def _example(name):
    r"""
    The absolute path of a distributed example, or a loud failure.

    Same single home, and same reason, as
    ``tests/test_phase8_task2.py::_example`` (F-P00-1, closed in Phase
    18): a substituting default that resolves to a bare relative name
    is invisible until the day it substitutes.

    Parameters
    ----------
    name : str
        File name inside ``examples/``.

    Returns
    -------
    str
        Absolute path.

    Raises
    ------
    FileNotFoundError
        If the example is absent, naming the resolved path.
    """
    path = os.path.join(_ROOT, "examples", name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"distributed example {name!r} not found at {path}. The "
            f"examples/ directory is part of the package, not an "
            f"optional extra: this test measures GenSec against the "
            f"documents it ships.")
    return path


_EXTERNAL = _example("example_external_tendon.yaml")
_STAGED_CRACKING = _example("example_staged_cracking.yaml")


# ======================================================================
#  1.  N-16 — the scan hull without the tendon lever arms
# ======================================================================

def _true_projection(sec, solver, cos_t, sin_t):
    r"""
    The projected coordinate of **every** section element.

    The quantity the sweep needs is the extent of the section along a
    curvature direction,

    .. math::

        p_i(\theta) = l_{y,i}\cos\theta - l_{x,i}\sin\theta ,

    taken over the union of bulk fibres, rebars **and** tendons.  It is
    assembled here from the geometry's own coordinate arrays, which
    exist on every tree, so this helper is a statement about the
    section and not about :class:`NMDiagram`.

    Parameters
    ----------
    sec : GenericSection
    solver : FiberSolver
    cos_t, sin_t : float
        Direction cosines.

    Returns
    -------
    numpy.ndarray
    """
    lx = np.concatenate([sec.x_fibers - solver.x_ref,
                         sec.x_rebars - solver.x_ref,
                         sec.x_tendons - solver.x_ref])
    ly = np.concatenate([sec.y_fibers - solver.y_ref,
                         sec.y_rebars - solver.y_ref,
                         sec.y_tendons - solver.y_ref])
    return ly * cos_t - lx * sin_t


class TestScanHullCoversEveryElement:
    r"""
    N-16: the sweep must not explore curvature the section cannot take.

    ``NMDiagram`` caches ``_all_lx`` / ``_all_ly`` once at construction
    and every sweep consumer projects them onto its direction:
    :meth:`~gensec.solver.capacity.NMDiagram._chi_max_for_direction`,
    :meth:`~gensec.solver.capacity.NMDiagram._compute_angle_params`
    (and through it ``generate_biaxial``), ``generate_mx_my``,
    ``generate_polar_curvature`` and ``_ductility_at_direction``.  The
    depth they divide by,

    .. math::

        \chi_{\max}(\theta) =
            \frac{\varepsilon_{\max,\mathrm{global}}
                  - \varepsilon_{\min,\mathrm{bulk}}}
                 {\mathrm{span}(\theta)},

    was assembled from bulk and rebar only.  An embedded tendon lies
    inside the fibre hull and is invisible to this; a non-embedded one
    need not.
    """

    def test_the_hull_spans_the_whole_section_on_the_bending_axis(self):
        r"""
        **Discriminating.**  Red on the pristine tree on the measured
        value.

        On ``examples/example_external_tendon.yaml`` the pristine tree
        produced::

            span used  : 594 mm     span true : 747 mm
            chi_max    : 1.195286e-04  supported : 9.504685e-05
            excess     : 1.257576   (= 747/594, exact)

        Both ``_chi_max_for_direction`` and ``_all_ly`` exist on the
        pristine tree, so the failure is an ``AssertionError`` on the
        number, never an ``ImportError`` or an ``AttributeError``.
        """
        sec = load_yaml(_EXTERNAL)["section"]
        solver = FiberSolver(sec)
        nm = NMDiagram(solver)

        cos_t, sin_t = 1.0, 0.0
        chi_max, span, _p_min = nm._chi_max_for_direction(
            cos_t, sin_t, nm._all_lx, nm._all_ly)

        proj = _true_projection(sec, solver, cos_t, sin_t)
        span_true = float(proj.max() - proj.min())

        assert span == pytest.approx(span_true, rel=1e-12), (
            f"the sweep projects a depth of {span:.6g} mm on a section "
            f"whose elements span {span_true:.6g} mm along the same "
            f"direction: an excess curvature ceiling of "
            f"{span_true / span:.6f}. The element left out is at "
            f"projected coordinate {float(proj.min()):.6g} mm, outside "
            f"the hull the sweep used "
            f"[{float(nm._all_ly.min()):.6g}, "
            f"{float(nm._all_ly.max()):.6g}].")

        chi_true = (nm._exg - nm._emb) / span_true
        assert chi_max == pytest.approx(chi_true, rel=1e-12)

    def test_the_hull_contains_every_tendon_in_both_coordinates(self):
        r"""
        **Discriminating.**  The same claim stated on the arrays, so a
        repair that fixed one direction only would still be red.
        """
        sec = load_yaml(_EXTERNAL)["section"]
        solver = FiberSolver(sec)
        nm = NMDiagram(solver)

        lx_t = sec.x_tendons - solver.x_ref
        ly_t = sec.y_tendons - solver.y_ref
        assert float(nm._all_lx.min()) <= float(lx_t.min())
        assert float(nm._all_lx.max()) >= float(lx_t.max())
        assert float(nm._all_ly.min()) <= float(ly_t.min()), (
            f"tendon lever arm {float(ly_t.min()):.6g} mm is below the "
            f"bottom of the sweep hull {float(nm._all_ly.min()):.6g} mm")
        assert float(nm._all_ly.max()) >= float(ly_t.max())

    def test_an_embedded_tendon_does_not_move_the_hull(self):
        r"""
        **Regression guard, declared as such.**  Passes on both trees
        by construction, which is what makes it useless as evidence and
        useful as a fence: it is the assertion that the repair is inert
        on every distributed model that had its tendons inside the
        hull, i.e. that the fingerprint movement of this phase is
        confined to the one example built to expose the defect.
        """
        sec = load_yaml(_example("example_prestress.yaml"))["section"]
        solver = FiberSolver(sec)
        nm = NMDiagram(solver)

        ly_t = sec.y_tendons - solver.y_ref
        fibre_hull_min = float(
            np.concatenate([sec.y_fibers - solver.y_ref,
                            sec.y_rebars - solver.y_ref]).min())
        assert float(ly_t.min()) > fibre_hull_min
        assert float(nm._all_ly.min()) == pytest.approx(fibre_hull_min)


# ======================================================================
#  2.  N-15 — the axial bracket without the tendon limits
# ======================================================================

_SLAB_YAML = """
# Declared because GenSec derives nothing from an undeclared
# model: the code is not inferred from a material type.
code: ec2
materials:
  conc: {type: concrete_ec2_gen1, class: C45/55, ls: F, loadtype: slow, TypeConc: R, NA: French, time: 28}
  y1860:
    type: prestressing_steel_ec2
    class: Y1860S7
    ls: F
    diagram: inclined
    NA: EC2
    eps_ud_factor: 0.9
section:
  B: 1200
  H: 200
  bulk_material: conc
  n_fibers_y: 40
  tendons:
    - name: S1
      x: 600
      y: 40
      material: y1860
      Ap: 700
      eps_pe: 0.0060
      bonded: true
"""


def _strand_only_slab(tmp_path):
    r"""
    A strand-only slab: the geometry on which N-15 is not screened.

    This is the hollow-core / double-tee case that
    ``NMDiagram.__init__`` already names for N-14: a section with
    prestressing strands and **no ordinary reinforcement**.  On it the
    old ``eps_hi`` stayed at exactly ``0.0``, because only rebars ever
    raised it, so the whole tensile branch of the axial scan did not
    exist.

    It is built here rather than distributed as an example on purpose:
    it exists to exhibit one defect, not to be read as a model, and
    adding it to ``examples/`` would enlarge the numeric fingerprint
    for no discrimination the two new distributed examples do not
    already provide.

    Parameters
    ----------
    tmp_path : pathlib.Path

    Returns
    -------
    GenericSection
    """
    path = tmp_path / "slab.yaml"
    path.write_text(_SLAB_YAML, encoding="utf-8")
    return load_yaml(str(path))["section"]


def _true_axial_tensile_capacity(solver, eps_hi, n=4000):
    r"""
    Peak of :math:`N(\varepsilon_0)` at zero curvature.

    Parameters
    ----------
    solver : FiberSolver
    eps_hi : float
        Upper end of the scan.
    n : int, optional

    Returns
    -------
    float
        Maximum axial force [N], tension positive.
    """
    eps = np.linspace(solver.sec.bulk_material.eps_min * 1.01, eps_hi, n)
    N, _, _ = solver.integrate_batch(eps, np.zeros(n), np.zeros(n))
    return float(np.max(N))


class TestAxialBracketCoversEveryElement:
    r"""
    N-15: a target inside the section's real capacity must have a
    bracket.
    """

    def test_a_target_inside_the_real_capacity_converges(self, tmp_path):
        r"""
        **Discriminating.**  Red on the pristine tree on the measured
        value.

        Measured on the pristine tree, this slab::

            peak N reachable by the old scan range :   819.0 kN
            true axial tensile capacity            :  1112.3 kN
            band with no bracket                   :   26.37 %

        and targets at 90 %, 95 % and 99 % of the true capacity all
        came back ``converged=False``.  The pristine failure is an
        ``AssertionError`` on ``converged``, not an import error: both
        ``_solve_pure_axial`` and ``integrate_batch`` exist on both
        trees.

        The registered figure for this defect was **13.6 %**
        (``18_0`` §4).  It is not reproduced here and is not disputed:
        the band is a function of the geometry — of the tendon's reach
        beyond the rebar's, and of how much of the section's force that
        reach carries — and ``18_0`` does not record the geometry it
        was measured on.  26.37 % is this slab's figure, stated with
        its section.
        """
        sec = _strand_only_slab(tmp_path)
        solver = FiberSolver(sec)

        # The true reach, assembled from the section itself: a tendon's
        # own limit enters the section strain shifted by its locked-in
        # strain.
        eps_hi_true = 0.0
        for mat, _bulk, _idx, eps_init in solver._tendon_groups:
            eps_hi_true = max(
                eps_hi_true,
                float(mat.eps_max) - float(np.min(eps_init)))
        assert eps_hi_true > 0.0

        N_true = _true_axial_tensile_capacity(solver, eps_hi_true * 1.01)
        target = 0.95 * N_true
        sol = solver._solve_pure_axial(target, tol=1.0, max_iter=200)

        assert sol["converged"], (
            f"_solve_pure_axial reports no solution for N = "
            f"{target:.1f} N, which is 95 % of this section's axial "
            f"tensile capacity of {N_true:.1f} N. The scan range it "
            f"built does not reach the strain at which the section "
            f"carries that force: the tendon's reach of "
            f"{eps_hi_true:.6g} is outside it.")
        assert sol["N"] == pytest.approx(target, abs=10.0)

    def test_the_routine_answers_without_help_from_its_caller(
            self, tmp_path):
        r"""
        **Discriminating**, and the measurement that corrects the
        record.

        ``18_0`` §4 registers N-15 as *"screened today by the uniaxial
        routing of ``solve_equilibrium`` — by accident and not by
        contract"*.  Measured on this slab, on both trees, that is
        **false**: the caller screens nothing.

        ==================  ==============  ==============
        target                 pristine        repaired
        ==================  ==============  ==============
        inner, 90 %            False           True
        inner, 95 %            False           True
        inner, 99 %            False           True
        outer, 90/95/99 %      False           False
        ==================  ==============  ==============

        The bracket defect was not hidden by the fall-through; it was
        simply never asked, because no distributed model requests a
        pure-tension target near capacity.  The repair is therefore
        justified by the inner verdict alone, which is what this test
        asserts.

        The outer route did **not** converge on the Phase 19 tree
        either, and this test used to observe that by reading
        ``outer["converged"]`` inside its own failure message.  Phase
        20 measured *why*, and the answer changed the shape of the
        outcome rather than the verdict: no strain plane of this slab
        carries :math:`N \approx 1.06` MN with :math:`M_x = 0`, so
        the outer route now refuses with
        :class:`AxialStateNotAttainable` instead of returning a mute
        ``converged = False``.  The assertion below is therefore taken
        on the inner verdict alone, with the outer outcome captured as
        *either* a refusal *or* a dict — the claim being made is about
        ``_solve_pure_axial``, and it must not become sensitive to how
        its caller reports an unrelated fact.
        """
        sec = _strand_only_slab(tmp_path)
        solver = FiberSolver(sec)
        eps_hi_true = max(
            float(mat.eps_max) - float(np.min(eps_init))
            for mat, _b, _i, eps_init in solver._tendon_groups)
        target = 0.95 * _true_axial_tensile_capacity(
            solver, eps_hi_true * 1.01)

        inner = solver._solve_pure_axial(target, tol=1.0, max_iter=200)
        try:
            outer = repr(solver.solve_equilibrium(target, 0.0, 0.0))
        except Exception as exc:            # noqa: BLE001 — outcome is data
            outer = f"{type(exc).__name__}: {exc}"

        assert inner["converged"], (
            f"the pure-axial routine reports no solution for N = "
            f"{target:.1f} N while the outer solve reports "
            f"{outer}: the routine must be "
            f"answerable on its own, not through its caller.")

    def test_a_section_without_tendons_is_untouched(self, tmp_path):
        r"""
        **Regression guard, declared as such.**  The repair must add a
        term that is empty when there are no tendons, so every
        non-prestressed model keeps its bracket bit-for-bit.
        """
        sec = load_yaml(_example("example_input.yaml"))["section"]
        solver = FiberSolver(sec)
        assert len(solver._tendon_groups) == 0
        sol = solver._solve_pure_axial(-500.0e3, tol=1.0, max_iter=200)
        assert sol["converged"]


# ======================================================================
#  3.  N-3 — cracking on a staged section
# ======================================================================

class TestStagedCracking:
    r"""
    N-3: at least one distributed model must reach a cracking event on
    a materialised stage view.
    """

    def test_a_distributed_example_cracks_on_a_staged_section(self):
        r"""
        **Discriminating.**  Red on the pristine tree because the
        example did not exist; red for the right reason nonetheless,
        because what it asserts is a property of the distributed set
        and the failure names the property.

        Measured on the repaired tree, ``service`` view, ``N = 0``::

            cracking_chi_pos  5.025126e-08   zone 1 (topping, C30/37)
            cracking_chi_neg -1.507538e-07   zone 0 (precast, C45/55)

        against the **base** section, same axial level::

            cracking_chi_pos  1.005025e-07

        i.e. the locked-in casting datum halves the curvature at which
        the topping reaches its own threshold.  Before Phase 19 every
        ``cracking_reason`` on every staged view of every distributed
        model read ``no_ec2_properties``: the detector was never
        reached on a staged section, so nothing about it was under
        test.
        """
        from gensec.solver.section_state import (StagedDomainManager,
                                                 materialize_view)
        from gensec.solver.timeline import ConstructionTimeline

        data = load_yaml(_STAGED_CRACKING)
        section = data["section"]
        ddb = {d["name"]: {"N": d.get("N", 0.0), "Mx": d.get("Mx", 0.0),
                           "My": d.get("My", 0.0)}
               for d in (data.get("demands") or [])}
        tl = ConstructionTimeline.from_block(
            data["construction_history"],
            losses_models=data.get("losses_models"),
            zone_rheology=data.get("zone_rheology"))
        resolution = tl.resolve(section, ddb)

        combos = [c for c in data["combinations"]
                  if c.get("at") is not None]
        views = {}
        for combo in combos:
            for point, stages, inactive in tl.compile_combination(
                    combo, resolution, section, ddb):
                if point in views:
                    continue
                mgr = StagedDomainManager(section, biaxial=False,
                                          gen_kwargs={"n_points": 12})
                states, _h, _b, _d = mgr.resolve_stages(
                    stages, initially_inactive=inactive)
                views[point] = materialize_view(section, states[-1])

        assert "service" in views
        mc = NMDiagram(FiberSolver(views["service"])
                       ).generate_moment_curvature(
            N_fixed=0.0, n_chi=200, direction="x")

        diag_pos = mc.get("diagnostics_pos") or {}
        assert mc.get("cracking_chi_pos") is not None, (
            f"no cracking event on the staged view; the detector "
            f"reports {diag_pos.get('cracking_reason')!r}")
        assert mc.get("cracking_chi_neg") is not None

        # The per-zone threshold (F-M1) is exercised, not bypassed: the
        # two signs must find their event in different zones, or the
        # zone map is not carrying the claim.
        diag_neg = mc.get("diagnostics_neg") or {}
        assert diag_pos.get("cracking_zone") != diag_neg.get(
            "cracking_zone")

    def test_the_staging_changes_the_answer(self):
        r"""
        **Discriminating.**  An example that cracks on a staged section
        but gives the same number as the base section would close N-3
        on paper only.
        """
        from gensec.solver.section_state import (StagedDomainManager,
                                                 materialize_view)
        from gensec.solver.timeline import ConstructionTimeline

        data = load_yaml(_STAGED_CRACKING)
        section = data["section"]
        ddb = {d["name"]: {"N": d.get("N", 0.0), "Mx": d.get("Mx", 0.0),
                           "My": d.get("My", 0.0)}
               for d in (data.get("demands") or [])}
        tl = ConstructionTimeline.from_block(
            data["construction_history"],
            losses_models=data.get("losses_models"),
            zone_rheology=data.get("zone_rheology"))
        resolution = tl.resolve(section, ddb)
        combo = [c for c in data["combinations"]
                 if c.get("at") is not None][0]
        compiled = tl.compile_combination(combo, resolution, section, ddb)
        point, stages, inactive = compiled[-1]
        mgr = StagedDomainManager(section, biaxial=False,
                                  gen_kwargs={"n_points": 12})
        states, _h, _b, _d = mgr.resolve_stages(
            stages, initially_inactive=inactive)
        view = materialize_view(section, states[-1])

        staged = NMDiagram(FiberSolver(view)).generate_moment_curvature(
            N_fixed=0.0, n_chi=200, direction="x")
        base = NMDiagram(FiberSolver(section)
                         ).generate_moment_curvature(
            N_fixed=0.0, n_chi=200, direction="x")

        assert staged["cracking_chi_pos"] is not None
        assert base["cracking_chi_pos"] is not None
        assert staged["cracking_chi_pos"] != base["cracking_chi_pos"], (
            f"the staged view and the base section reach the cracking "
            f"threshold at the same curvature "
            f"{staged['cracking_chi_pos']!r}: the casting datum is not "
            f"reaching the criterion, so this example does not test "
            f"what it is distributed to test")


# ======================================================================
#  4.  The gate on vacuous --selfcheck passes
# ======================================================================

_TOOL = os.path.join(_ROOT, "tools", "selfcheck_inventory.py")


def _run_gate(root):
    r"""
    Run the inventory gate against a tree and return its exit code.

    The exit code is read from the process, never through a pipe: a
    pipeline reports the exit status of its **last** element, which is
    how three measurements were read as green in Phase 17 while the
    command was exiting 1.

    Parameters
    ----------
    root : str
        Repository root of the tree to interrogate.

    Returns
    -------
    tuple of (int, str)
        Exit code and combined output.
    """
    proc = subprocess.run(
        [sys.executable, os.path.join(root, "tools",
                                      "selfcheck_inventory.py"), "--gate"],
        capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=os.path.join(root, "src")))
    return proc.returncode, proc.stdout + proc.stderr


class TestVacuousPassGate:
    r"""
    A ``SELFCHECK PASS`` that does not say how many files it compared
    is not a verification (``00_BOOTSTRAP.md`` §4.2).

    The gate does not repair the historical patchers and is not
    supposed to: rule F-B14 keeps only the most recent patcher on the
    health list, so an old vacuous PASS is never read as a health
    signal.  What it prevents is a **future** patcher entering the
    health list while verifying nothing.
    """

    def test_the_gate_passes_on_the_delivered_tree(self):
        """The delivered health list declares real byte comparisons."""
        code, out = _run_gate(_ROOT)
        assert code == 0, out
        assert "byte comparison" in out

    def test_the_gate_refuses_a_patcher_that_compares_nothing(
            self, tmp_path):
        r"""
        **Discriminating on the gate itself.**  A gate never seen red
        is vacuous in exactly the way it exists to forbid.

        The tree is copied and its health list rewritten to name
        ``p03a_part1_base.py``, measured in this phase as printing one
        line — ``already applied`` — and exiting 0 with no
        ``--selfcheck`` option at all.
        """
        root = tmp_path / "tree"
        (root / "tools").mkdir(parents=True)
        (root / "patchers").mkdir()
        src = open(_TOOL, encoding="utf-8").read()
        assert 'HEALTH_LIST = (' in src
        head, _sep, tail = src.partition('HEALTH_LIST = (')
        _old, _sep2, rest = tail.partition(')')
        (root / "tools" / "selfcheck_inventory.py").write_text(
            head + 'HEALTH_LIST = ("p03a_part1_base.py",)' + rest,
            encoding="utf-8")
        for name in ("_patchkit.py", "p03a_part1_base.py"):
            (root / "patchers" / name).write_bytes(
                open(os.path.join(_ROOT, "patchers", name), "rb").read())

        code, out = _run_gate(str(root))
        assert code == 1, out
        assert "VACUOUS" in out
        assert "p03a_part1_base.py" in out

    def test_the_gate_refuses_a_health_list_naming_an_absent_patcher(
            self, tmp_path):
        """**Discriminating.** A name that resolves to nothing is not a
        pass."""
        root = tmp_path / "tree"
        (root / "tools").mkdir(parents=True)
        (root / "patchers").mkdir()
        src = open(_TOOL, encoding="utf-8").read()
        head, _sep, tail = src.partition('HEALTH_LIST = (')
        _old, _sep2, rest = tail.partition(')')
        (root / "tools" / "selfcheck_inventory.py").write_text(
            head + 'HEALTH_LIST = ("run_phase99_nowhere.py",)' + rest,
            encoding="utf-8")
        code, out = _run_gate(str(root))
        assert code == 1, out
        assert "ABSENT" in out

    def test_the_inventory_counts_bytes_and_sentinels_apart(self):
        r"""
        **Discriminating on the instrument.**

        The tool measured at the opening of this phase reported
        ``vacuous: 0`` while ``00_BOOTSTRAP.md`` §4.1 recorded six
        patchers in a vacuous PASS.  The two were not in conflict about
        the tree: they used two definitions of "vacuous". The tool
        counted ``require`` / ``retire`` **sentinels**, so
        ``p00_measurement_blockers.py`` with five of them read as
        non-vacuous; the bootstrap counted **byte comparisons**, of
        which it has none.  Worse, the six patchers with no
        ``--selfcheck`` at all reported ``-`` and were excluded from
        the count entirely — the instrument was blind on precisely the
        class it was pointed at.

        The two counts are separate columns now.  This test asserts
        they are, so a future collapse back into one is red.
        """
        sys.path.insert(0, os.path.join(_ROOT, "tools"))
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_si_under_test", _TOOL)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            sys.path.pop(0)

        rows = {r["name"]: r for r in mod.inventory()}
        sentinel = rows["p00_measurement_blockers.py"]
        assert sentinel["n_checked"] > 0
        assert sentinel["n_bytes"] == 0
        assert sentinel["vacuous"] is True

        silent = rows["p03a_part1_base.py"]
        assert silent["mechanism"] == "none"
        assert silent["n_bytes"] == 0
        assert silent["vacuous"] is True, (
            "a patcher that offers no --selfcheck at all must count as "
            "verifying nothing, not as unclassifiable")
