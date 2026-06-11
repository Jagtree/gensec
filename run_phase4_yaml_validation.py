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
Validation of the Phase-4 step *"unified stage model reachable from
YAML"*: the ``section_ops`` / ``time`` / ``report`` stage schema, the
union-index + name reference mechanism, and the gated engine wiring.

Run from the project root (with the package installed / importable):

.. code-block:: console

    python run_phase4_yaml_validation.py

Checks, all self-contained (closed-form or self-referential — no
external baseline tool needed):

1. **Parser guards** — pure value-level checks on
   ``io_yaml._parse_combination`` / ``_parse_section_ops_spec``:
   unknown keys, misplacement of stage-only keys, malformed blocks,
   the ``bulk_eps`` *no-silent-no-op* Phase-5 guard, ``time`` sign and
   monotonicity.
2. **Reference mechanism** — ``_union_name_map`` /
   ``_resolve_element_id``: names resolve to union indices across both
   element populations, duplicate names raise, unknown names raise,
   out-of-range indices raise, YAML booleans are rejected as indices.
3. **YAML ≡ API identity** — the headline check: a YAML staged-
   construction toy (group B of rebars deactivated at stage 0,
   re-activated by *name* at stage 1) run through the YAML-wired
   ``VerificationEngine`` must reproduce, **dict-for-dict**, the
   API-driven Phase-3 result (explicit integer ``section_ops`` on the
   same loaded section).  Same section objects → same capacity hashes
   → exact equality, not tolerance equality.
4. **Capacity-frozen regression anchor** — the same YAML with the
   ``section_ops`` blocks removed makes ``staged_ops_present`` return
   ``False``; the gate then constructs the engine exactly as before
   Phase 4 and the result dict equals the explicitly-legacy
   construction byte-for-byte (and carries no ``domain_hash`` /
   ``domain_reset`` provenance keys).
5. **Static-engine refusal** — a ``section_ops``-carrying combination
   on an engine *without* a staged manager raises instead of silently
   freezing the capacity (both ``VerificationEngine`` and
   ``AnalysisEngine``).
6. **``time`` carry-through** — lands on ``SectionState.time_days``,
   carries forward when omitted, and never perturbs the capacity hash.
7. **``report`` echo** — the opaque payload reappears verbatim in the
   per-stage result of both engines.
"""

import copy
import json
import tempfile

import numpy as np

from gensec.io_yaml import (
    load_yaml, staged_ops_present,
    _parse_combination, _parse_section_ops_spec,
    _union_name_map, _resolve_element_id,
)
from gensec.solver import FiberSolver, NMDiagram, VerificationEngine
from gensec.solver.analysis import AnalysisEngine
from gensec.solver.section_state import StagedDomainManager

#: Small/fast domain generator settings, used for BOTH the engine
#: pre-computed domain and the staged manager so that every compared
#: quantity is generated at identical resolution.
GEN_KW = dict(n_angles=24, n_points_per_angle=40)

#: The staged-construction toy.  Same section as the Phase-3 validation
#: script (300x500, C25/30, fyk 450, 4 bars of 400 mm²), expressed in
#: YAML, with named rebars so the stage ops can exercise the name
#: reference path (stage 0 deactivates by name, stage 1 re-activates by
#: union index — both spellings of the same mechanism).
TOY_YAML = """
materials:
  conc:  {type: concrete_ec2_gen1, class: 'C25/30'}
  steel: {type: steel, fyk: 450, gamma_s: 1.15}

section:
  shape: rect
  params: {B: 300, H: 500}
  bulk_material: conc
  mesh_size: 20
  rebars:
    - {x: 40,  y: 40,  As: 400, material: steel, name: A1}
    - {x: 260, y: 40,  As: 400, material: steel, name: A2}
    - {x: 40,  y: 460, As: 400, material: steel, name: B1}
    - {x: 260, y: 460, As: 400, material: steel, name: B2}

demands:
  - {name: G, N_kN: -500, Mx_kNm: 20}
  - {name: E, Mx_kNm: 70}

combinations:
  - name: staged-build
    stages:
      - name: build-A
        components: [{ref: G, factor: 1.0}]
        section_ops: {deactivate: [B1, B2]}
        time: 0
      - name: add-B
        components: [{ref: E, factor: 1.0}]
        section_ops: {activate: [2, 3]}
        time: 28
        report: {note: "B grouted"}
      - name: service
        components: [{ref: E, factor: 0.5}]

output:
  eta_norm: true
  eta_norm_beta: true
  eta_path_norm_ray: true
"""

FLAGS = {"eta_norm": True, "eta_norm_beta": True,
         "eta_path_norm_ray": True}


def _load_toy(yaml_text=TOY_YAML):
    """Round-trip the toy through a temp file and ``load_yaml``."""
    with tempfile.NamedTemporaryFile(
            "w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        path = f.name
    return load_yaml(path)


def _raises(fn, *args, exc=ValueError, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    return False


# ==================================================================
#  1. Parser guards
# ==================================================================

def test_parser_guards():
    print("[1] parser guards (structure, misplacement, bulk_eps, time)")

    # Unknown section_ops key.
    assert _raises(_parse_section_ops_spec, "C", "s",
                   {"activate": [0], "activte": [1]}), "unknown key"
    # Malformed blocks.
    assert _raises(_parse_section_ops_spec, "C", "s", [1, 2]), "non-dict"
    assert _raises(_parse_section_ops_spec, "C", "s",
                   {"activate": 0}), "activate not a list"
    assert _raises(_parse_section_ops_spec, "C", "s",
                   {"eps_override": [0]}), "eps_override not a mapping"
    assert _raises(_parse_section_ops_spec, "C", "s",
                   {"release": "yes"}), "release not a bool"

    # bulk_eps: zero passes (inert), non-zero raises (Phase-5 guard).
    assert _parse_section_ops_spec("C", "s",
                                   {"bulk_eps": 0.0}) == {"bulk_eps": 0.0}
    for v in (1e-4, -2e-4):
        assert _raises(_parse_section_ops_spec, "C", "s",
                       {"bulk_eps": v}), f"bulk_eps={v} must raise"

    # Misplacement: stage-only keys on a simple combination ...
    for key, val in (("section_ops", {"activate": [0]}),
                     ("time", 28), ("report", {})):
        assert _raises(_parse_combination,
                       {"name": "S", "components": [{"ref": "G"}],
                        key: val}), f"simple-combo guard: {key}"
    # ... and at the staged-combination level.
    for key, val in (("section_ops", {"activate": [0]}),
                     ("time", 28), ("report", {})):
        assert _raises(_parse_combination,
                       {"name": "S",
                        "stages": [{"name": "a", "components": []}],
                        key: val}), f"combo-level guard: {key}"

    # time: negative and decreasing raise; equal is allowed.
    assert _raises(_parse_combination,
                   {"name": "S", "stages": [
                       {"name": "a", "components": [], "time": -1}]})
    assert _raises(_parse_combination,
                   {"name": "S", "stages": [
                       {"name": "a", "components": [], "time": 28},
                       {"name": "b", "components": [], "time": 7}]})
    c = _parse_combination(
        {"name": "S", "stages": [
            {"name": "a", "components": [], "time": 28},
            {"name": "b", "components": [], "time": 28}]})
    assert [s["time"] for s in c["stages"]] == [28.0, 28.0]
    print("    OK")


# ==================================================================
#  2. Reference mechanism (union index + names)
# ==================================================================

def test_reference_mechanism():
    print("[2] union-index + name reference mechanism")
    data = _load_toy()
    sec = data["section"]

    nm = _union_name_map(sec)
    assert nm == {"A1": 0, "A2": 1, "B1": 2, "B2": 3}, nm

    # Resolution semantics + guards.
    assert _resolve_element_id("B1", 4, nm, "w") == 2
    assert _resolve_element_id(3, 4, nm, "w") == 3
    assert _raises(_resolve_element_id, "Bx", 4, nm, "w"), "unknown name"
    assert _raises(_resolve_element_id, 4, 4, nm, "w"), "out of range"
    assert _raises(_resolve_element_id, -1, 4, nm, "w"), "negative"
    assert _raises(_resolve_element_id, True, 4, nm, "w"), \
        "YAML bool must not be accepted as index 1"

    # Duplicate names across the union set raise.
    sec_dup = copy.copy(sec)
    sec_dup.rebars = list(sec.rebars)
    sec_dup.rebars[1] = copy.copy(sec.rebars[1])
    sec_dup.rebars[1].name = "A1"
    assert _raises(_union_name_map, sec_dup), "duplicate name"

    # The loader resolved the YAML ops: names -> union indices.
    stages = data["combinations"][0]["stages"]
    assert stages[0]["section_ops"] == {"deactivate": [2, 3]}
    assert stages[1]["section_ops"] == {"activate": [2, 3]}
    assert "_section_ops_spec" not in stages[0]
    print("    OK")


# ==================================================================
#  3. YAML staged-construction toy == API-driven Phase-3 result
# ==================================================================

def test_yaml_equals_api():
    print("[3] YAML staged-construction toy == API-driven Phase-3 run")
    data = _load_toy()
    sec = data["section"]
    demand_db = {d["name"]: d for d in data["demands"]}

    solver = FiberSolver(sec)
    nm_gen = NMDiagram(solver)
    cloud = nm_gen.generate_biaxial(**GEN_KW)

    # --- Path A: YAML-driven (the loader's resolved combination) ---
    mgr_a = StagedDomainManager(sec, biaxial=True, gen_kwargs=GEN_KW)
    eng_a = VerificationEngine(cloud, nm_gen, FLAGS,
                               staged_manager=mgr_a)
    res_a = eng_a.check_combination(data["combinations"][0], demand_db)

    # --- Path B: API-driven (explicit integer section_ops, same
    #     loaded section/materials -> same capacity hashes) ---
    stages_b = [
        {"name": "build-A",
         "components": [{"ref": "G", "factor": 1.0}],
         "section_ops": {"deactivate": [2, 3]}, "time": 0.0},
        {"name": "add-B",
         "components": [{"ref": "E", "factor": 1.0}],
         "section_ops": {"activate": [2, 3]}, "time": 28.0,
         "report": {"note": "B grouted"}},
        {"name": "service",
         "components": [{"ref": "E", "factor": 0.5}]},
    ]
    mgr_b = StagedDomainManager(sec, biaxial=True, gen_kwargs=GEN_KW)
    eng_b = VerificationEngine(cloud, nm_gen, FLAGS,
                               staged_manager=mgr_b)
    res_b = eng_b._check_staged("staged-build", stages_b, demand_db)

    assert res_a == res_b, (
        "YAML-driven and API-driven staged results diverge:\n"
        f"  yaml={json.dumps(res_a, default=str)}\n"
        f"  api ={json.dumps(res_b, default=str)}")

    # Structural expectations (mirror of run_phase3_validation case 3).
    st = res_a["stages"]
    assert st[0]["domain_hash"] != st[1]["domain_hash"], \
        "activation must change the hash"
    assert st[1]["domain_hash"] == st[2]["domain_hash"], \
        "unchanged section must keep the hash"
    assert [s["domain_reset"] for s in st] == [True, True, False]
    assert np.isfinite(res_a["eta_governing"])
    print(f"    identical dicts; resets "
          f"{[s['domain_reset'] for s in st]}, "
          f"eta_gov={res_a['eta_governing']}: OK")


# ==================================================================
#  4. Capacity-frozen YAML == legacy engine, byte-for-byte
# ==================================================================

def test_capacity_frozen_identity():
    print("[4] capacity-frozen staged YAML == legacy engine output")
    frozen_yaml = TOY_YAML.replace(
        "        section_ops: {deactivate: [B1, B2]}\n", "").replace(
        "        section_ops: {activate: [2, 3]}\n", "")
    data = _load_toy(frozen_yaml)
    sec = data["section"]
    demand_db = {d["name"]: d for d in data["demands"]}
    combos = data["combinations"]

    # The gate must report no ops ...
    assert not staged_ops_present(combos)
    # ... so the wiring constructs exactly the legacy engine.
    solver = FiberSolver(sec)
    nm_gen = NMDiagram(solver)
    cloud = nm_gen.generate_biaxial(**GEN_KW)

    staged_mgr = None
    if staged_ops_present(combos):          # the cli/api gate, verbatim
        staged_mgr = StagedDomainManager(sec, biaxial=True,
                                         gen_kwargs=GEN_KW)
    eng_gated = VerificationEngine(cloud, nm_gen, FLAGS,
                                   staged_manager=staged_mgr)
    eng_legacy = VerificationEngine(cloud, nm_gen, FLAGS)

    r_gated = eng_gated.check_combination(combos[0], demand_db)
    r_legacy = eng_legacy.check_combination(combos[0], demand_db)
    assert r_gated == r_legacy, "gated != legacy on frozen input"
    for sr in r_gated["stages"]:
        assert "domain_hash" not in sr and "domain_reset" not in sr, \
            "frozen run must not carry staged provenance keys"
    print("    OK (no manager built, dicts identical, no provenance "
          "keys)")


# ==================================================================
#  5. Static-engine refusal (no-silent-no-op)
# ==================================================================

def test_static_engine_refusal():
    print("[5] section_ops without a staged_manager raises")
    data = _load_toy()
    sec = data["section"]
    demand_db = {d["name"]: d for d in data["demands"]}
    combo = data["combinations"][0]

    solver = FiberSolver(sec)
    nm_gen = NMDiagram(solver)
    cloud = nm_gen.generate_biaxial(**GEN_KW)

    eng = VerificationEngine(cloud, nm_gen, FLAGS)      # no manager
    assert _raises(eng.check_combination, combo, demand_db,
                   exc=RuntimeError), "VerificationEngine must refuse"

    an = AnalysisEngine(solver)                          # no manager
    assert _raises(an.analyze_combinations, [combo], demand_db,
                   exc=RuntimeError), "AnalysisEngine must refuse"
    print("    OK")


# ==================================================================
#  6. time carry-through (never hashed)
# ==================================================================

def test_time_carry_through():
    print("[6] stage 'time' -> SectionState.time_days, never hashed")
    data = _load_toy()
    sec = data["section"]
    stages = data["combinations"][0]["stages"]

    mgr = StagedDomainManager(sec, biaxial=True, gen_kwargs=GEN_KW)
    states, hashes, _b, _d = mgr.resolve_stages(stages)
    # Stage 2 omits 'time' -> carries the stage-1 value forward.
    assert [s.time_days for s in states] == [0.0, 28.0, 28.0], \
        [s.time_days for s in states]

    # Same ops, different time -> identical capacity hash.
    st_a = [{"name": "a", "components": [], "time": 0.0},
            {"name": "b", "components": [], "time": 365.0}]
    _s, h2, _b2, _d2 = mgr.resolve_stages(st_a)
    assert h2[0] == h2[1], "time alone must never change the hash"
    print(f"    time_days={[s.time_days for s in states]}, "
          "hash time-invariant: OK")


# ==================================================================
#  7. report echo (both engines)
# ==================================================================

def test_report_echo():
    print("[7] per-stage 'report' echoed verbatim by both engines")
    data = _load_toy()
    sec = data["section"]
    demand_db = {d["name"]: d for d in data["demands"]}
    combo = data["combinations"][0]

    solver = FiberSolver(sec)
    nm_gen = NMDiagram(solver)
    cloud = nm_gen.generate_biaxial(**GEN_KW)

    mgr = StagedDomainManager(sec, biaxial=True, gen_kwargs=GEN_KW)
    res = VerificationEngine(cloud, nm_gen, FLAGS, staged_manager=mgr
                             ).check_combination(combo, demand_db)
    assert res["stages"][1]["report"] == {"note": "B grouted"}
    assert "report" not in res["stages"][0]
    assert "report" not in res["stages"][2]

    mgr2 = StagedDomainManager(sec, biaxial=False,
                               gen_kwargs={"n_points": 60})
    ares = AnalysisEngine(solver, staged_manager=mgr2
                          ).analyze_combinations([combo], demand_db)[0]
    assert ares["stages"][1]["report"] == {"note": "B grouted"}
    assert "report" not in ares["stages"][0]
    print("    OK")


if __name__ == "__main__":
    test_parser_guards()
    test_reference_mechanism()
    test_yaml_equals_api()
    test_capacity_frozen_identity()
    test_static_engine_refusal()
    test_time_carry_through()
    test_report_echo()
    print("\nALL PHASE-4 YAML VALIDATIONS PASSED")
