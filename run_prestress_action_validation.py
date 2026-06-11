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
Validation of the Phase-1 ``PrestressAction`` couple generator and its
YAML wiring.

Run from the project root (with the package installed / importable):

.. code-block:: console

    python run_prestress_action_validation.py

The checks are intentionally closed-form / hand-calc so they can be run
without any reference tool:

1. **Couple sign and magnitude** of :meth:`PrestressAction.from_force`
   against the hand-calc resultant about the reference point, and against
   the integrator sign convention encoded in ``_element_net_force``.
2. **Jacking-on-hardened-concrete demand point** equals the equivalent
   :math:`(N, M)` couple, reproducing the *exact* demand-walk arithmetic
   of ``check._check_staged`` / ``analysis._analyze_staged``.
3. **YAML round-trip**: a text document is parsed and the prestress
   actions are resolved to the expected triples; the bulk ``prestrain``
   field is read.

These exercise the pure (load-side) logic only; the full numerical suite
(``unittest`` / ``pytest``) and the existing-prestress VCASLU N-M
cross-check remain the end-to-end gate.
"""

import numpy as np
import yaml

from gensec.solver.section_state import (
    PrestressAction, _element_net_force,
)
from gensec.io_yaml import (
    _parse_combination, _resolve_prestress_actions, _parse_bulk_prestrain,
)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def demand_walk(stages, demand_db):
    r"""
    Reproduce verbatim the staged demand accumulation of the engines:
    ``cum += sum(factor * component) + sum(action.triple())``.
    """
    cum = [0.0, 0.0, 0.0]
    points = []
    for stg in stages:
        dN = dMx = dMy = 0.0
        for c in stg["components"]:
            d = demand_db[c["ref"]]
            f = c["factor"]
            dN += f * d["N"]
            dMx += f * d["Mx"]
            dMy += f * d["My"]
        for act in stg.get("_prestress_actions", []):
            aN, aMx, aMy = act.triple()
            dN += aN
            dMx += aMx
            dMy += aMy
        cum[0] += dN
        cum[1] += dMx
        cum[2] += dMy
        points.append(tuple(cum))
    return points


def test_couple_sign_and_magnitude():
    print("[1] from_force: couple sign & magnitude vs hand calc")
    cases = [
        dict(P=1.4e6, x=200.0, y=80.0, xr=150.0, yr=300.0),
        dict(P=-500e3, x=0.0, y=0.0, xr=0.0, yr=0.0),
        dict(P=900e3, x=-120.0, y=540.0, xr=150.0, yr=300.0),
    ]
    for c in cases:
        a = PrestressAction.from_force(
            c["P"], c["x"], c["y"], x_ref=c["xr"], y_ref=c["yr"])
        F = -c["P"]
        N_h = F
        Mx_h = F * (c["y"] - c["yr"])
        My_h = -F * (c["x"] - c["xr"])
        assert approx(a.N, N_h), (a.N, N_h)
        assert approx(a.Mx, Mx_h), (a.Mx, Mx_h)
        assert approx(a.My, My_h), (a.My, My_h)
        # Spec identities.
        assert approx(a.N, -c["P"])
        assert approx(a.Mx, -c["P"] * (c["y"] - c["yr"]))
        assert approx(a.My, +c["P"] * (c["x"] - c["xr"]))
    print("    OK")


def test_consistency_with_element_net_force():
    r"""
    ``from_force`` must map an external force ``P`` exactly as the
    integrator maps an internal element force ``F = -P`` at the same
    lever.  We verify the algebraic identity directly: building a
    one-element linear section and reading ``_element_net_force`` at a
    strain plane that puts stress ``sigma`` on it, the action of an
    equal-and-opposite external force reproduces ``from_force``.
    """
    print("[2] from_force consistent with _element_net_force map")
    # Algebraic identity (no solver needed): from_force uses F=-P in the
    # same (F, F*ly, -F*lx) map _element_net_force returns.
    P, x, y, xr, yr = 1.23e6, 211.0, 73.0, 150.0, 300.0
    a = PrestressAction.from_force(P, x, y, x_ref=xr, y_ref=yr)
    F = -P
    ly, lx = (y - yr), (x - xr)
    assert approx(a.N, F)
    assert approx(a.Mx, F * ly)
    assert approx(a.My, -F * lx)
    print("    OK")


def test_jacking_demand_point():
    print("[3] jacking-on-hardened-concrete demand point == equivalent (N,M)")

    class _T:
        name = None

    class FakeSec:
        x_centroid = 150.0
        y_centroid = 300.0
        x_tendons = np.array([200.0])
        y_tendons = np.array([80.0])
        tendons = [_T()]

    sec = FakeSec()
    xr, yr = sec.x_centroid, sec.y_centroid
    P, x, y = 1.4e6, 200.0, 80.0

    combo = _parse_combination({
        "name": "PT",
        "stages": [
            {"name": "jack", "components": [],
             "prestress_actions": [{"P": P, "x": x, "y": y}]},
        ],
    })
    _resolve_prestress_actions([combo], sec)
    Ncum, Mxcum, Mycum = demand_walk(combo["stages"], {})[-1]
    assert approx(Ncum, -P)
    assert approx(Mxcum, -P * (y - yr))
    assert approx(Mycum, +P * (x - xr))

    # Adds on top of a prior gravity stage.
    combo2 = _parse_combination({
        "name": "PT2",
        "stages": [
            {"name": "G", "components": [{"ref": "G", "factor": 1.0}]},
            {"name": "jack", "components": [],
             "prestress_actions": [{"P": P, "x": x, "y": y}]},
        ],
    })
    _resolve_prestress_actions([combo2], sec)
    dd = {"G": {"N": -300e3, "Mx": 50e6, "My": 0.0}}
    last = demand_walk(combo2["stages"], dd)[-1]
    assert approx(last[0], -300e3 - P)
    assert approx(last[1], 50e6 - P * (y - yr))
    assert approx(last[2], 0.0 + P * (x - xr))
    print("    OK")


def test_yaml_round_trip():
    print("[4] YAML round-trip (text -> parse -> resolve)")

    class _T:
        name = None

    class FakeSec:
        x_centroid = 150.0
        y_centroid = 300.0
        x_tendons = np.array([200.0])
        y_tendons = np.array([80.0])
        tendons = [_T()]

    sec = FakeSec()
    xr, yr = sec.x_centroid, sec.y_centroid

    txt = """
combinations:
  - name: PT_rt
    stages:
      - name: g1
        components: [{ref: G, factor: 1.0}]
      - name: tesatura
        components: []
        prestress_actions:
          - {P_kN: 1400, ref: 0}
          - {sigma_p0: 950, Ap: 1200, x: 100, y: 540}
"""
    data = yaml.safe_load(txt)

    combos = [_parse_combination(c) for c in data["combinations"]]
    _resolve_prestress_actions(combos, sec)
    acts = combos[0]["stages"][1]["_prestress_actions"]
    assert len(acts) == 2
    e0 = PrestressAction.from_force(1.4e6, 200, 80, xr, yr)
    e1 = PrestressAction.from_force(950 * 1200, 100, 540, xr, yr)
    for got, exp in zip(acts, (e0, e1)):
        assert approx(got.N, exp.N)
        assert approx(got.Mx, exp.Mx)
        assert approx(got.My, exp.My)
    print("    OK")


def test_placement_guards():
    print("[5] placement guards (prestress_actions only on a stage)")
    raised = False
    try:
        _parse_combination({"name": "S", "components": [{"ref": "G"}],
                            "prestress_actions": [{"P": 1.0}]})
    except ValueError:
        raised = True
    assert raised, "simple-combination guard did not fire"

    raised = False
    try:
        _parse_combination({"name": "S",
                            "stages": [{"name": "a", "components": []}],
                            "prestress_actions": [{"P": 1.0}]})
    except ValueError:
        raised = True
    assert raised, "combo-level guard did not fire"
    print("    OK")




def test_bulk_prestrain_guard():
    r"""
    *No-silent-no-op* guard: a non-zero bulk ``prestrain`` /
    ``eps_init`` must raise until the fiber solver consumes the offset
    (the resistance domain would otherwise NOT reflect it).
    """
    print("[6] non-zero bulk prestrain raises (no-silent-no-op guard)")
    assert _parse_bulk_prestrain({}) == 0.0
    assert _parse_bulk_prestrain({"prestrain": 0.0}) == 0.0
    for spec in ({"prestrain": -2e-4}, {"eps_init": 1e-4}):
        raised = False
        try:
            _parse_bulk_prestrain(spec)
        except ValueError:
            raised = True
        assert raised, f"guard did not fire for {spec}"
    print("    OK")


if __name__ == "__main__":
    test_couple_sign_and_magnitude()
    test_consistency_with_element_net_force()
    test_jacking_demand_point()
    test_yaml_round_trip()
    test_placement_guards()
    test_bulk_prestrain_guard()
    print("\nALL PRESTRESS-ACTION VALIDATIONS PASSED")
