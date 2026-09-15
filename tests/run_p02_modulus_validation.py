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
Validator for P2 — slot-keyed modulus resolution (12_0d S1).

Two independent assemblies, as the phase deliverable requires.

**Assembly I — the reference is a section built by hand.**
    Axes B1–B4 and B6 never call :func:`~gensec.solver.sls.sls_view`
    or :func:`~gensec.solver.sls.resolve_moduli` to produce their
    reference.  They *construct* a second
    :class:`~gensec.geometry.section.GenericSection` whose bulk and
    element laws are :class:`~gensec.materials.elastic.LinearElastic`
    objects wired in place, solve it with the same
    :class:`~gensec.solver.integrator.FiberSolver`, and compare the
    resulting strain plane and stresses.  Rule 2 holds: the reference
    shares the *geometry* with the implementation, which is the object
    under test's input, but not the *resolution*, which is the object
    under test's job.

**Assembly II — the reference is closed-form algebra.**
    Axis B5 assembles the homogenised area and second moment of a
    single-zone section by hand,

    .. math::

        A_{\mathrm{id}} = A_c + \sum_i (n_i - 1)\,A_i,
        \qquad n_i = \frac{E_i}{E_c},

    and compares them with
    :func:`~gensec.solver.sls.sls_transformed_properties`.  No fibre
    integration enters the reference.

Rule 3, restated for this validator
-----------------------------------
Axis B2 exists only to prove that the fixtures used by B1/B3 *can*
exhibit the defect the retrofit removes: it re-runs the reference with
the two zones collapsed onto one modulus — which is exactly what the
material-keyed map produced — and requires the answer to **differ**.
A validator whose fixture cannot exhibit the bug does not validate the
fix.

Axis map
--------

======  ====================================================================
B1      Do two zones sharing one material object solve as the hand-built
        two-modulus section?
B2      Can the fixture exhibit the collapse at all (rule-3 self-check)?
B3      Does the uncast-zone placeholder stay confined to its own zone?
B4      Is the answer on a distinct-material section unchanged
        (numerical inertia — the acceptance signal of a re-keying)?
B5      Do the transformed properties match the closed-form
        homogenised section?
B6      Does a masked stage view read each element's own slot?
B7      Does every retired or ambiguous key form fail loud?
B8      Is there exactly one substituted law per distinct (name, E)?
======  ====================================================================

Exit status is ``0`` when every axis passes, ``1`` otherwise.

Usage
-----
::

    python tests/run_p02_modulus_validation.py
    PYTHONPATH=src python tests/run_p02_modulus_validation.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np                                          # noqa: E402
from shapely.geometry import Polygon                         # noqa: E402

from gensec.geometry.fiber import RebarLayer, Tendon         # noqa: E402
from gensec.geometry.section import GenericSection           # noqa: E402
from gensec.materials.concrete import Concrete               # noqa: E402
from gensec.materials.elastic import LinearElastic           # noqa: E402
from gensec.materials.steel import (                         # noqa: E402
    PrestressingSteel, Steel,
)
from gensec.solver.integrator import FiberSolver             # noqa: E402
from gensec.solver.section_state import (                    # noqa: E402
    SectionState, materialize_view,
)
from gensec.solver.sls import (                              # noqa: E402
    MaterialMap, resolve_moduli, resolve_sls_moduli, SlsPolicy,
    sls_transformed_properties, sls_view,
)


# ==================================================================
#  Model data
# ==================================================================

B, H = 300.0, 600.0
Y_TOP = 500.0
E_SUB, E_TOP = 36283.0, 22000.0
E_SINGLE = 32000.0
ES, EP = 200000.0, 195000.0
PLACEHOLDER = 1.0
AS1, AS2, AP = 628.0, 402.0, 1500.0
Y_S1, Y_S2, Y_T = 50.0, 550.0, 100.0
MESH = 25.0
DEMAND = (-4.0e5, -6.0e7, 1.2e7)        # N [N], Mx, My [N·mm]
TOL = 1.0e-10


def _outline():
    return Polygon([(0.0, 0.0), (B, 0.0), (B, H), (0.0, H)])


def _topping():
    return Polygon([(0.0, Y_TOP), (B, Y_TOP), (B, H), (0.0, H)])


def _steel():
    st = Steel(fyk=450.0)
    st.name = "B450C"
    return st


def _rebars(material):
    return [RebarLayer(y=Y_S1, As=AS1, material=material, x=150.0,
                       name="A1"),
            RebarLayer(y=Y_S2, As=AS2, material=material, x=150.0,
                       name="A2")]


def _two_zone(shared: bool):
    r"""
    Two-zone section; *shared* selects one material object or two.

    Parameters
    ----------
    shared : bool

    Returns
    -------
    gensec.geometry.section.GenericSection
    """
    c0 = Concrete(fck=45.0, gamma_c=1.5, long_term_reduction_factor=0.85)
    c0.name = "C45"
    c1 = c0 if shared else Concrete(fck=45.0, gamma_c=1.5, long_term_reduction_factor=0.85)
    if not shared:
        c1.name = "C45 topping"
    return GenericSection(
        polygon=_outline(), bulk_material=c0, rebars=_rebars(_steel()),
        mesh_size=MESH, bulk_materials=[(_topping(), c1, "topping")])


def _hand_substituted(e_base, e_topping, e_steel=ES):
    r"""
    Assembly I: the same geometry with linear laws wired **in place**.

    No resolver and no view substitution is involved; the moduli are
    written into the constructor.

    Parameters
    ----------
    e_base, e_topping, e_steel : float
        Moduli [MPa].

    Returns
    -------
    gensec.geometry.section.GenericSection
    """
    return GenericSection(
        polygon=_outline(),
        bulk_material=LinearElastic(E=e_base, name="hand base"),
        rebars=_rebars(LinearElastic(E=e_steel, name="hand steel")),
        mesh_size=MESH,
        bulk_materials=[(_topping(),
                         LinearElastic(E=e_topping, name="hand topping"),
                         "topping")])


def _solve(section):
    r"""
    Strain plane and element stresses of a linear section.

    Parameters
    ----------
    section : GenericSection

    Returns
    -------
    numpy.ndarray
        ``[eps0, chi_x, chi_y]``.
    """
    sol = FiberSolver(section).solve_equilibrium(*DEMAND, tol=1e-10,
                                                 max_iter=60)
    if not sol["converged"]:
        raise RuntimeError("reference solve did not converge")
    return np.array([sol["eps0"], sol["chi_x"], sol["chi_y"]], dtype=float)


def _rel(a, b):
    r"""Largest relative difference between two planes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = np.maximum(np.abs(a), np.abs(b))
    scale[scale == 0.0] = 1.0
    return float(np.max(np.abs(a - b) / scale))


def _state(n_union, active=None):
    act = (np.ones(n_union, dtype=bool) if active is None
           else np.asarray(active, dtype=bool))
    return SectionState(0, act, np.ones(n_union, dtype=bool),
                        np.zeros(n_union, dtype=float))


# ==================================================================
#  Axis plumbing
# ==================================================================

class Axis:
    """One validation axis: tag, question, observation, verdict."""

    __slots__ = ("tag", "question", "observed", "ok")

    def __init__(self, tag, question, observed, ok):
        self.tag = tag
        self.question = question
        self.observed = observed
        self.ok = bool(ok)


def _report(axes):
    """Print the axis table; return the exit status."""
    width = 74
    print("=" * width)
    print("P2 — SLOT-KEYED MODULUS RESOLUTION  (12_0d S1)")
    print("=" * width)
    for ax in axes:
        mark = "PASS" if ax.ok else "FAIL"
        print(f"\n[{mark}] {ax.tag}  {ax.question}")
        body = textwrap.wrap(ax.observed, width - 14)
        for k, line in enumerate(body):
            print(("    observed : " if k == 0 else "               ") + line)
    good = sum(1 for a in axes if a.ok)
    print("\n" + "-" * width)
    print(f"{good} / {len(axes)} axes green")
    print("-" * width)
    return 0 if good == len(axes) else 1


# ==================================================================
#  B1 / B2 — zone separation, and the fixture's ability to show it
# ==================================================================

def axis_b1():
    r"""Shared material object, two moduli: pipeline vs hand-built."""
    sec = _two_zone(shared=True)
    mm = resolve_sls_moduli(sec, {("zone", 0): E_SUB, ("zone", 1): E_TOP})
    p_pipe = _solve(sls_view(sec, mm))
    p_hand = _solve(_hand_substituted(E_SUB, E_TOP))
    d = _rel(p_pipe, p_hand)
    return Axis(
        "B1", "Do two zones sharing one material object solve as the "
              "hand-built two-modulus section?",
        f"strain planes agree to {d:.2e} relative "
        f"(eps0 {p_pipe[0]:.6e} vs {p_hand[0]:.6e}).",
        d < TOL)


def axis_b2():
    r"""Rule-3 self-check: the collapse is visible on this fixture."""
    p_right = _solve(_hand_substituted(E_SUB, E_TOP))
    p_collapsed = _solve(_hand_substituted(E_TOP, E_TOP))
    d = _rel(p_right, p_collapsed)
    return Axis(
        "B2", "Can the fixture exhibit the collapse at all "
              "(rule-3 self-check)?",
        f"collapsing both zones onto {E_TOP:.0f} MPa — what the "
        f"material-keyed map produced — moves the strain plane by "
        f"{100.0 * d:.1f} %.  A fixture on which this were zero could "
        f"not validate the fix.",
        d > 1.0e-3)


# ==================================================================
#  B3 — placeholder containment
# ==================================================================

def axis_b3():
    r"""The 1 MPa placeholder must not reach the substrate."""
    sec = _two_zone(shared=True)
    mm = resolve_sls_moduli(sec, {("zone", 0): E_SUB,
                                  ("zone", 1): PLACEHOLDER})
    vw = sls_view(sec, mm)
    e_base = float(vw.bulk_material.E)
    e_top = float(vw.bulk_materials[0][1].E)
    p_pipe = _solve(vw)
    p_hand = _solve(_hand_substituted(E_SUB, PLACEHOLDER))
    d = _rel(p_pipe, p_hand)
    ok = (abs(e_base - E_SUB) < 1e-9 and abs(e_top - PLACEHOLDER) < 1e-12
          and d < TOL)
    return Axis(
        "B3", "Does the uncast-zone placeholder stay confined to its "
              "own zone?",
        f"substituted substrate reads {e_base:.1f} MPa (fed "
        f"{E_SUB:.0f}), cast zone {e_top:.1f} MPa; the solve agrees "
        f"with the hand-built section to {d:.2e}.",
        ok)


# ==================================================================
#  B4 — inertia on a distinct-material section
# ==================================================================

def axis_b4():
    r"""The acceptance signal: nothing moves where nothing may move."""
    sec = _two_zone(shared=False)
    by_name = _solve(sls_view(sec, resolve_sls_moduli(
        sec, {"C45": E_SUB, "C45 topping": E_TOP})))
    by_slot = _solve(sls_view(sec, resolve_sls_moduli(
        sec, {("zone", 0): E_SUB, ("zone", 1): E_TOP})))
    hand = _solve(_hand_substituted(E_SUB, E_TOP))
    d1, d2 = _rel(by_name, by_slot), _rel(by_slot, hand)
    return Axis(
        "B4", "Is the answer on a distinct-material section unchanged?",
        f"material-name sugar and zone slots agree to {d1:.2e}, and "
        f"both agree with the hand-built section to {d2:.2e}.  Only "
        f"the shared-object fixture may move under a re-keying.",
        d1 < TOL and d2 < TOL)


# ==================================================================
#  B5 — closed-form transformed properties (assembly II)
# ==================================================================

def axis_b5():
    r"""Homogenised area and inertia, assembled by hand."""
    conc = Concrete(fck=35.0, gamma_c=1.5, long_term_reduction_factor=0.85)
    conc.name = "C35"
    ps = PrestressingSteel(f_p01d=1391.3, Ep=EP)
    ps.name = "Y1860"
    sec = GenericSection(
        polygon=_outline(), bulk_material=conc,
        rebars=_rebars(_steel()),
        tendons=[Tendon(y=Y_T, Ap=AP, material=ps, x=150.0,
                        eps_pe=0.006, name="P1")],
        mesh_size=MESH)
    props = sls_transformed_properties(sec, _state(3),
                                       moduli={"C35": E_SINGLE})

    n_s, n_p = ES / E_SINGLE, EP / E_SINGLE
    area = B * H + (n_s - 1.0) * (AS1 + AS2) + (n_p - 1.0) * AP
    # First moment about the soffit, then the centroid and I_x.
    s_x = (B * H * H / 2.0 + (n_s - 1.0) * (AS1 * Y_S1 + AS2 * Y_S2)
           + (n_p - 1.0) * AP * Y_T)
    y_g = s_x / area
    i_x = (B * H ** 3 / 12.0 + B * H * (H / 2.0 - y_g) ** 2
           + (n_s - 1.0) * (AS1 * (Y_S1 - y_g) ** 2
                            + AS2 * (Y_S2 - y_g) ** 2)
           + (n_p - 1.0) * AP * (Y_T - y_g) ** 2)
    d_a = abs(props.area - area) / area
    d_i = abs(props.Ix - i_x) / i_x
    return Axis(
        "B5", "Do the transformed properties match the closed-form "
              "homogenised section?",
        f"A_id {props.area:.1f} vs {area:.1f} mm2 ({d_a:.2e}); "
        f"I_x {props.Ix:.4e} vs {i_x:.4e} mm4 ({d_i:.2e}).",
        d_a < 1.0e-9 and d_i < 1.0e-9)


# ==================================================================
#  B6 — positional consumption on a masked stage view
# ==================================================================

def axis_b6():
    r"""
    With element 0 out of the view, view element 0 is union element 1
    and must read *its own* modulus — the sharp case for
    ``_union_index``.
    """
    conc = Concrete(fck=35.0, gamma_c=1.5, long_term_reduction_factor=0.85)
    conc.name = "C35"
    ps = PrestressingSteel(f_p01d=1391.3, Ep=EP)
    ps.name = "Y1860"
    sec = GenericSection(
        polygon=_outline(), bulk_material=conc,
        rebars=_rebars(_steel()),
        tendons=[Tendon(y=Y_T, Ap=AP, material=ps, x=150.0,
                        eps_pe=0.006, name="P1")],
        mesh_size=MESH)
    per_elem = {("elem", 0): 100000.0, ("elem", 1): 210000.0,
                ("tendon", "P1"): 190000.0}
    mm = resolve_moduli(sec, SlsPolicy(),
                        explicit={"C35": E_SINGLE, **per_elem})
    view = materialize_view(sec, _state(3, active=[False, True, True]))
    vw = sls_view(view, mm)
    got = [float(vw.rebars[0].material.E),
           float(vw.tendons[0].material.E)]
    want = [210000.0, 190000.0]
    ok = got == want
    return Axis(
        "B6", "Does a masked stage view read each element's own slot?",
        f"view holds union elements {list(getattr(view, '_union_index'))}"
        f" and reads {got} MPa; expected {want} (a positional bug would "
        f"read {100000.0:.0f} on the surviving rebar).",
        ok)


# ==================================================================
#  B7 — the retired key forms
# ==================================================================

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception:                                   # noqa: BLE001
        return False
    return False


def axis_b7():
    r"""Every ambiguous or retired key form must fail loud."""
    sec = _two_zone(shared=True)
    conc = sec.bulk_material
    good = {("zone", 0): E_SUB, ("zone", 1): E_TOP}

    class _Hashable:
        name = "exotic"

    probes = {
        "bare int (the id() form)":
            lambda: resolve_sls_moduli(sec, {id(conc): E_SUB}),
        "hashable material object":
            lambda: resolve_sls_moduli(sec, {_Hashable(): E_SUB}),
        "dead name override":
            lambda: resolve_sls_moduli(sec, {**good, "Ghost": 1.0e4}),
        "two slot keys on one slot":
            lambda: resolve_sls_moduli(
                sec, {**good, ("rebar", "A1"): ES, ("elem", 0): ES}),
        "unknown zone name":
            lambda: resolve_sls_moduli(sec, {("zone", "deck"): E_SUB}),
        "boolean reference":
            lambda: resolve_sls_moduli(sec, {("zone", True): E_SUB}),
        "shape mismatch in the container":
            lambda: MaterialMap(bulk=(LinearElastic(E=E_SUB),),
                                elem=(),
                                labels_bulk=("base", "topping"),
                                labels_elem=()),
        "a number where a material belongs":
            lambda: MaterialMap(bulk=(E_SUB,), elem=(),
                                labels_bulk=("base",), labels_elem=()),
        "non-positive modulus":
            lambda: MaterialMap(bulk=(LinearElastic(E=-1.0),), elem=(),
                                labels_bulk=("base",), labels_elem=()),
    }
    failed = [name for name, fn in probes.items() if not _raises(fn)]
    mm = resolve_sls_moduli(sec, good)
    # Two axes, two mechanisms: the materials are a tuple and cannot be
    # assigned into at all; the derived moduli are an array with its
    # write flag cleared.  Both must refuse, or "frozen" is a promise
    # the object breaks through whichever field it holds by reference.
    # A tuple has no ``__setitem__`` at all, so the refusal arrives as
    # AttributeError rather than TypeError.  Both are accepted: what is
    # asserted is that the assignment does not go through, not which
    # exception Python picks to say so.
    read_only = (_raises(lambda: mm.bulk.__setitem__(0, 1.0),
                         (TypeError, AttributeError))
                 and _raises(lambda: mm.E_bulk.__setitem__(0, 1.0)))
    if not read_only:
        failed.append("read-only arrays")
    return Axis(
        "B7", "Does every retired or ambiguous key form fail loud?",
        (f"{len(probes) + 1 - len(failed)} / {len(probes) + 1} probes "
         f"raise") + ("." if not failed else f"; silent: {failed}."),
        not failed)


# ==================================================================
#  B8 — substitution dedup
# ==================================================================

def axis_b8():
    r"""One substituted law per distinct ``(original name, E)``."""
    sec = _two_zone(shared=True)
    split = sls_view(sec, resolve_sls_moduli(
        sec, {("zone", 0): E_SUB, ("zone", 1): E_TOP}))
    merged = sls_view(sec, resolve_sls_moduli(sec, {"C45": E_SUB}))

    def _laws(view):
        return [view.bulk_material] + [m for _p, m in view.bulk_materials] \
            + [r.material for r in view.rebars]

    n_split = len({id(m) for m in _laws(split)})
    n_merged = len({id(m) for m in _laws(merged)})
    # Independently: |{(name, E)}| over the same slots.
    want_split = len({("C45", E_SUB), ("C45", E_TOP), ("B450C", ES)})
    want_merged = len({("C45", E_SUB), ("B450C", ES)})
    ok = (n_split == want_split and n_merged == want_merged)
    return Axis(
        "B8", "Is there exactly one substituted law per distinct "
              "(name, E)?",
        f"two moduli -> {n_split} law(s) (expected {want_split}); one "
        f"modulus -> {n_merged} law(s) (expected {want_merged}).  The "
        f"two rebars keep one shared law, so the solver's identity "
        f"grouping is preserved.",
        ok)


# ==================================================================
#  Driver
# ==================================================================

def main():
    """
    Run every axis.

    Returns
    -------
    int
    """
    axes = [axis_b1(), axis_b2(), axis_b3(), axis_b4(),
            axis_b5(), axis_b6(), axis_b7(), axis_b8()]
    return _report(axes)


if __name__ == "__main__":
    sys.exit(main())
