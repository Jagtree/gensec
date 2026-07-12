# SPDX-License-Identifier: AGPL-3.0-or-later
# GenSec — Phase-8 Task-2 (construction timeline) independent validator.
# Copyright (C) 2026  Andrea (GenSec project).
r"""
Independent validator for the Phase-8 Task-2 construction-timeline compiler.

Run:  ``python run_phase8_task2_validation.py``  (repo root, package importable).

The validator exercises the §5 axes of ``10_3`` that are closed by the
Task-2 **core** (timeline object + auto-datum walk + compiler), each
assembled by a route independent of the code under test:

A1  auto-datum equivalence
    The datum produced by :meth:`ConstructionTimeline.resolve` is
    compared, component by component, against an **independently
    assembled** datum: negate the plane the substrate reaches under the
    cumulative characteristic demand, solved on a manually staged
    :class:`~gensec.solver.section_state.StagedDomainManager`.  This is
    the master-plan §3 linear-equivalence identity and the C3 crux.
    Pass criterion: machine-precision agreement on all three components.

A2  compiler ⇒ engine
    :meth:`ConstructionTimeline.compile_combination` output is fed to the
    **unmodified** :meth:`~gensec.solver.check.VerificationEngine.check_combination`;
    the run must converge and return a finite governing :math:`\eta`,
    and the stage bookkeeping (``domain_reset``, cumulative demand) must
    reflect ``history_factors``.  This proves the emitted stage list is
    valid input to the existing walk (the hard constraint).

A3  factor semantics
    ``history_factors`` scale the prefix ``load`` demands; the omission
    of a prefix ``load`` warns (stderr) while an explicit ``1.0`` does
    not; :math:`\gamma_P` resolves favourable/unfavourable/explicit via
    the pluggable provider.

A4  fail-loud inventory
    Every documented illegal input raises the documented exception.

A5  multi-point anchoring (C2)
    ``at:`` as a list yields one result per point; the governing point
    is the transparent maximum, not a v2.1 envelope collapse.

Axes that require the full prestress driver-reconciliation integration
(exact byte-equivalence vs :func:`gensec.solver.posttension.solve_posttension_sequence`)
and the io_yaml / cli / api wiring are **out of this validator's scope**;
they are the repository integration + full-suite gate (recap ``10_4``
§status).  This validator asserts only what it independently reproduces.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stderr

import numpy as np

from gensec.io_yaml import load_yaml
from gensec.solver import FiberSolver, NMDiagram
from gensec.solver.check import VerificationEngine
from gensec.solver.section_state import StagedDomainManager
from gensec.solver.timeline import (
    ConstructionTimeline, governing_point, _resolve_gamma_P,
)

EXAMPLE = "examples/example_composite_topping.yaml"
TOL = 1e-12


def _demand_db(data):
    r"""SI demand database ``{name: {N, Mx, My}}`` from parsed demands."""
    return {x["name"]: {"N": x.get("N", 0.0), "Mx": x.get("Mx", 0.0),
                        "My": x.get("My", 0.0)} for x in data["demands"]}


# ---------------------------------------------------------------------------

def axis_A1_auto_datum(section, ddb) -> bool:
    r"""A1 — auto datum equals independently assembled substrate plane."""
    tl = ConstructionTimeline.from_block([
        {"load": {"demand": "G1"}},
        {"cast": {"zone": "topping", "datum": "auto"}},
        {"load": {"demand": "G2"}},
        {"point": "service"},
    ])
    res = tl.resolve(section, ddb)
    way1 = tuple(float(v) for v in res.datums[1])

    # Independent route: precast-only substrate (topping zone inactive,
    # its rebars deactivated) under characteristic G1; negate the plane.
    mgr = StagedDomainManager(section, biaxial=False,
                              gen_kwargs={"n_points": 40})
    nonbase = [i for i in range(_n_union(section))
               if _parents(section)[i] == 1]
    states, _h, _b, _d = mgr.resolve_stages(
        [{"name": "g1", "components": [{"ref": "G1", "factor": 1.0}],
          "section_ops": {"deactivate": nonbase, "release": False}}],
        initially_inactive=[1])
    _hh, bundle, _bb = mgr.get_bundle(states[-1])
    g = ddb["G1"]
    s = bundle.solver.solve_equilibrium(g["N"], g["Mx"], g["My"],
                                        tol=1e-10, max_iter=100)
    way2 = (-s["eps0"], -s["chi_x"], -s["chi_y"])

    ok = all(abs(a - b) <= TOL for a, b in zip(way1, way2))
    print(f"  A1 auto-datum: way1={_fmt(way1)}  way2={_fmt(way2)}  "
          f"max|Δ|={max(abs(a-b) for a,b in zip(way1,way2)):.2e}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def axis_A2_A3_A5_compiler(section, ddb) -> bool:
    r"""A2/A3/A5 — compiler output runs on the engine; factors; C2."""
    tl = ConstructionTimeline.from_block([
        {"load": {"demand": "G1"}},
        {"cast": {"zone": "topping", "datum": "auto"}},
        {"load": {"demand": "G2"}},
        {"point": "transfer"},
        {"point": "service"},
    ])
    res = tl.resolve(section, ddb)
    combo = {"name": "ULS", "at": ["transfer", "service"],
             "history_factors": {"G1": 1.35, "G2": 1.35}, "gamma_P": 1.0,
             "components": [{"ref": "Q", "factor": 1.5}]}

    buf = io.StringIO()
    with redirect_stderr(buf):
        compiled = tl.compile_combination(combo, res, section, ddb)
    # A3: explicit factors listed -> no omission warning for G1/G2
    warned = "history load" in buf.getvalue()

    solver = FiberSolver(section)
    nm = NMDiagram(solver)
    nm_data = nm.generate(n_points=40)
    flags = {"eta_norm": True, "eta_norm_beta": True, "eta_norm_ray": True}
    mgr = StagedDomainManager(section, biaxial=False,
                              gen_kwargs={"n_points": 40})
    engine = VerificationEngine(nm_data, nm, flags, n_points=40,
                                staged_manager=mgr)

    per_point = {}
    for pt, stages, inact in compiled:
        r = engine.check_combination({"name": f"ULS@{pt}", "stages": stages},
                                     ddb)
        per_point[pt] = r
    gov = governing_point(per_point)

    # A2: both anchors return finite governing eta
    finite = all(np.isfinite(per_point[p].get("eta_governing", np.nan))
                 for p in per_point)
    # A3: G1 factored 1.35 -> first-stage cumulative N == 1.35 * (-300 kN)
    first = per_point["transfer"]["stages"][0]["cumulative"]
    factored = abs(first["N_kN"] - 1.35 * (-300.0)) < 1e-6
    # A5: governing point is the max-eta point
    gov_ok = (gov["governing_point"] in per_point
              and gov["eta_governing"] == max(
                  round(per_point[p]["eta_governing"], 4) for p in per_point))

    ok = finite and factored and gov_ok and (not warned)
    print(f"  A2 compiler⇒engine: anchors="
          f"{[ (p, per_point[p].get('eta_governing')) for p in per_point]}  "
          f"finite={finite}")
    print(f"  A3 factors: 1.35·G1 applied={factored}  "
          f"no-warn-when-listed={not warned}")
    print(f"  A5 governing point (C2 max): {gov['governing_point']} "
          f"@ eta={gov['eta_governing']}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def axis_A3_warn_on_omission(section, ddb) -> bool:
    r"""A3 — an unlisted prefix ``load`` warns; a listed 1.0 does not."""
    tl = ConstructionTimeline.from_block([
        {"load": {"demand": "G1"}},
        {"cast": {"zone": "topping", "datum": "auto"}},
        {"load": {"demand": "G2"}},
        {"point": "service"},
    ])
    res = tl.resolve(section, ddb)

    def warns(hist):
        buf = io.StringIO()
        with redirect_stderr(buf):
            tl.compile_combination(
                {"name": "c", "at": "service", "history_factors": hist,
                 "components": []}, res, section, ddb)
        return "history load" in buf.getvalue()

    omitted_warns = warns({"G1": 1.0})          # G2 omitted -> warn
    listed_silent = not warns({"G1": 1.0, "G2": 1.0})
    ok = omitted_warns and listed_silent
    print(f"  A3 omission warns={omitted_warns}  explicit-1.0 silent="
          f"{listed_silent}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def axis_A4_fail_loud(section, ddb) -> bool:
    r"""A4 — every documented illegal input raises the documented error."""
    cases = []

    def check(label, fn, exc):
        try:
            fn()
            cases.append((label, False, "no raise"))
        except exc:
            cases.append((label, True, exc.__name__))
        except Exception as e:  # noqa: BLE001
            cases.append((label, False, f"{type(e).__name__}!={exc.__name__}"))

    C = ConstructionTimeline
    check("unknown kind", lambda: C.from_block([{"pour": {}}]), ValueError)
    check("dup point", lambda: C.from_block(
        [{"point": "p"}, {"point": "p"}]), ValueError)
    # C5 landed: 'interval' + 'losses' is no longer a placeholder.  A bare
    # model name is still refused -- a zone's rheology is meaningless
    # without its age and its curing age, and guessing them would be a
    # silent normative choice -- but with ValueError now.
    check("interval+losses (bare name)", lambda: C.from_block(
        [{"interval": {"days": 1, "losses": "creep"}}]), ValueError)
    check("cast base", lambda: C.from_block(
        [{"cast": {"zone": "base"}}]).validate(section), ValueError)
    check("unknown zone", lambda: C.from_block(
        [{"cast": {"zone": "nope"}}]).validate(section), ValueError)
    check("bad datum", lambda: C.from_block(
        [{"cast": {"zone": "topping", "datum": "x"}}]).resolve(section, ddb),
        ValueError)
    check("unknown demand", lambda: C.from_block(
        [{"load": {"demand": "ZZ"}}]).resolve(section, ddb), ValueError)
    check("bad gamma_P", lambda: _resolve_gamma_P("nope", None), ValueError)

    ok = all(p for _, p, _ in cases)
    for label, passed, got in cases:
        print(f"  A4 {label:<18} {'ok' if passed else 'FAIL'} ({got})")
    print(f"  A4 fail-loud inventory -> {'PASS' if ok else 'FAIL'}")
    return ok


def axis_A3_gamma_P() -> bool:
    r"""A3 — gamma_P favourable/unfavourable/explicit/provider."""
    ok = (_resolve_gamma_P("favourable", None) == 1.0
          and _resolve_gamma_P("unfavourable", None) == 1.3
          and _resolve_gamma_P(1.2, None) == 1.2
          and _resolve_gamma_P("unfavourable", lambda k: 1.10) == 1.10)
    print(f"  A3 gamma_P (EC2 default + provider) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def axis_A6_prestress_equivalence() -> bool:
    r"""
    A6 — a compiled ``stress`` → ``grout`` sequence is byte-equivalent to
    the post-tensioning driver (``10_3`` §5 axis-1).

    The reconciled locked-in strain the timeline bakes into the grout
    stage is compared against
    :func:`gensec.solver.posttension.grout` applied to
    :func:`gensec.solver.posttension.solve_posttension_sequence` — the
    timeline **reuses** exactly those, so agreement is to machine zero.
    Also asserts the resolved grout state is ``active & bonded`` at that
    strain, and that the tendon's demand-side action is cancelled at
    grout (the single-side invariant: cumulative demand nets to zero).
    """
    from gensec.solver.posttension import (
        solve_posttension_sequence, grout)
    from gensec.solver.timeline import (
        _tendon_local_index, _tendon_union_index)
    from gensec.solver.section_state import StagedDomainManager
    from gensec.solver.check import VerificationEngine

    data = load_yaml("examples/example_prestress.yaml")
    section = data["section"]
    T, SIGMA = "tendon_0", 1400.0

    tl = ConstructionTimeline.from_block([
        {"stress": {"tendons": [T], "sigma_p0": SIGMA}},
        {"grout": {"tendons": [T]}},
        {"point": "after"}])
    res = tl.resolve(section, {})
    w1 = res.grout_eps_init[T]

    li = _tendon_local_index(section, T)
    ui = _tendon_union_index(section, T)
    sig = [0.0] * len(section.tendons)
    sig[li] = SIGMA
    result = solve_posttension_sequence(
        section, sigma_p0=sig, order=[li], base_N=0.0, base_Mx=0.0,
        base_My=0.0)
    w2 = grout(section, result, indices=[li]).report[
        "reconciliation"][0]["eps_init"]
    byte_eq = abs(w1 - w2) <= 1e-15

    _pt, stages, _i = tl.compile_combination(
        {"name": "c", "at": "after", "components": []}, res, section, {})[0]
    mgr = StagedDomainManager(section, biaxial=False,
                              gen_kwargs={"n_points": 40})
    states, _h, _b, _d = mgr.resolve_stages(stages)
    f = states[-1]
    state_ok = (int(f.active[ui]) == 1 and int(f.bonded[ui]) == 1
                and abs(f.eps_init[ui] - w2) <= 1e-15)

    nm = NMDiagram(FiberSolver(section))
    eng = VerificationEngine(nm.generate(n_points=40), nm, {"eta_norm": True},
                             n_points=40, staged_manager=mgr)
    r = eng.check_combination({"name": "c", "stages": stages}, {})
    cum = r["stages"][-1]["cumulative"]
    single_side = (abs(cum["N_kN"]) < 1e-6 and abs(cum["Mx_kNm"]) < 1e-6
                   and abs(cum["My_kNm"]) < 1e-6)

    ok = byte_eq and state_ok and single_side
    print(f"  A6 byte-equiv eps_init: timeline={w1:.6e} driver={w2:.6e} "
          f"|Δ|={abs(w1-w2):.1e}")
    print(f"  A6 grout state active&bonded @ reconciled strain: {state_ok}")
    print(f"  A6 single-side (demand nets to 0 after grout): {single_side} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


# -- small geometry helpers (independent of timeline internals) -------------

def _parents(section):
    from gensec.solver.section_state import _staging_parents
    return _staging_parents(section)


def _n_union(section):
    return _parents(section).size


def _fmt(t):
    return "(" + ", ".join(f"{v:+.4e}" for v in t) + ")"


# ---------------------------------------------------------------------------

def main() -> int:
    data = load_yaml(EXAMPLE)
    section = data["section"]
    ddb = _demand_db(data)

    print("GenSec Phase-8 Task-2 — construction-timeline validation")
    print("=" * 64)
    results = {
        "A1 auto-datum (two independent ways)": axis_A1_auto_datum(section, ddb),
        "A2/A3/A5 compiler ⇒ engine + factors + C2":
            axis_A2_A3_A5_compiler(section, ddb),
        "A3 warn-on-omission": axis_A3_warn_on_omission(section, ddb),
        "A3 gamma_P resolution": axis_A3_gamma_P(),
        "A4 fail-loud inventory": axis_A4_fail_loud(section, ddb),
        "A6 prestress byte-equivalence (stress→grout)":
            axis_A6_prestress_equivalence(),
    }
    print("=" * 64)
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    allok = all(results.values())
    print("=" * 64)
    print("RESULT:", "ALL PASS" if allok else "FAILURES PRESENT")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
