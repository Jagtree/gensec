r"""
Phase 36 — one mechanism for unknown keys (Z-17, Z-15), and the
observation inventory (E34-1).

Why these fixtures fail on the incoming tree
--------------------------------------------
Before this phase a key nobody reads loaded **in silence** at three
levels: inside ``section:`` (Z-17), on a combination and on a stage
(Z-15).  A silent fallback is outside the filter of ``00_BOOTSTRAP``
§0.4, so there was no "warn instead" option to weigh: the key is
refused, by name, with the valid set printed beside it.

Every fixture below is written so that its RED on the pristine tree is
the **fact** and not the shape of the answer (E35-1).  None imports the
helper that did not yet exist; each loads or parses a document carrying
an unknown key and requires a ``ValueError``.  On the incoming tree the
call returns normally, which is the defect itself; a repair that
declared a helper and never called it would leave the red red.

The denominators these fixtures rest on were re-measured on the
incoming tree, not copied:

===============  ====================================================
level            measured on the 24 distributed examples
===============  ====================================================
``section:``     15 keys read (AST, 20 sites); 12 written by the
                 examples, all inside the read set; 0 unknown
combination      19 present; 0 unknown
stage            17 present; 0 unknown
===============  ====================================================

So the refusal breaks no distributed document, and the last fixture
holds that fact still.
"""

from __future__ import annotations

import copy
import os

import pytest
import yaml

from gensec.io_yaml import _parse_combination, load_yaml

# The valid-key tuples are imported **inside** the fixtures that need
# them, never at module scope.  At module scope an incoming tree that
# does not define them yet turns every fixture in this file into a
# collection error, and a collection error measures the absence of a
# name -- which is exactly the failure E35-1 records.  The fixtures
# that measure the defect must go red on the fact: the document loads
# when it should have been refused.

# ----------------------------------------------------------------------
#  Documents
# ----------------------------------------------------------------------

#: A minimal document that loads on the incoming tree.  Deliberately not
#: one of the distributed examples: a fixture that edits a shipped file
#: in place would couple this test to whatever a later phase does to
#: that example.
_DOC = """
code: ec2
materials:
  C30:
    type: concrete_ec2_gen1_custom
    fck: 30.0
    gamma_c: 1.5
    long_term_reduction_factor: 0.85
section:
  shape: rect
  params: {B: 300.0, H: 500.0}
  bulk_material: C30
  mesh_size: 50.0
demands:
  - name: G
    N_kN: -100.0
    Mx_kNm: 20.0
    My_kNm: 0.0
combinations:
  - name: SLU
    components:
      - {ref: G, factor: 1.35}
"""


def _repo_root() -> str:
    r"""The tree root, from this file's own location."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write(tmp_path, doc: dict, name="doc.yaml") -> str:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return str(p)


@pytest.fixture()
def doc() -> dict:
    return copy.deepcopy(yaml.safe_load(_DOC))


# ----------------------------------------------------------------------
#  The null control: the document must load untouched
# ----------------------------------------------------------------------

def test_the_untouched_document_still_loads(tmp_path, doc):
    r"""
    Null control.

    Without it a refusal that rejected every document would satisfy
    every other fixture in this file.
    """
    data = load_yaml(_write(tmp_path, doc))
    assert data["section"] is not None


# ----------------------------------------------------------------------
#  section:  — Z-17
# ----------------------------------------------------------------------

def test_unknown_key_inside_section_is_refused(tmp_path, doc):
    r"""
    RED on the incoming tree: this load **returns**.

    The failure being measured is that the document declares something
    and is verified anyway, which reports a verdict on a model its
    author did not write.
    """
    doc["section"]["mesh_szie"] = 50.0
    with pytest.raises(ValueError) as exc:
        load_yaml(_write(tmp_path, doc))
    msg = str(exc.value)
    assert "mesh_szie" in msg, "the refusal must name the offending key"
    assert "mesh_size" in msg, "and print the valid set beside it"
    assert "section" in msg, "and say which block it was found in"


def test_the_section_valid_set_is_what_the_parser_reads(tmp_path, doc):
    r"""
    Every name in :data:`VALID_SECTION_KEYS` must be accepted.

    A valid set larger than the read set would let a typo through under
    a name that is only *nearly* right; smaller, and it would refuse a
    key the parser honours.  Measured: 15 keys over 20 AST sites.
    """
    from gensec.io_yaml import VALID_SECTION_KEYS

    assert len(VALID_SECTION_KEYS) == 15
    for key in ("shape", "params", "B", "H", "bulk_material", "mesh_size",
                "mesh_method", "n_fibers", "n_fibers_x", "n_fibers_y",
                "rebars", "tendons", "material_zones", "prestrain",
                "eps_init"):
        assert key in VALID_SECTION_KEYS


# ----------------------------------------------------------------------
#  combination and stage — Z-15
# ----------------------------------------------------------------------

def test_unknown_key_on_a_combination_is_refused():
    r"""RED on the incoming tree: this parse returns a combination."""
    with pytest.raises(ValueError) as exc:
        _parse_combination({"name": "SLU",
                            "components": [{"ref": "G"}],
                            "componets": [{"ref": "Q"}]})
    msg = str(exc.value)
    assert "componets" in msg
    assert "components" in msg


def test_unknown_key_on_a_stage_is_refused():
    r"""RED on the incoming tree: this parse returns a staged combo."""
    with pytest.raises(ValueError) as exc:
        _parse_combination({
            "name": "SLU",
            "stages": [{"name": "s0", "components": [],
                        "section_op": {"activate": []}}]})
    msg = str(exc.value)
    assert "section_op" in msg
    assert "section_ops" in msg


def test_a_misplaced_stage_key_keeps_its_own_diagnosis():
    r"""
    ``section_ops`` at combination level is **not** an unknown key.

    It is a known key in the wrong place, and the refusal that says so
    is more useful than one that lists the valid set.  The unknown-key
    check therefore has to carry the misplaced names in its valid set
    and let the specific refusal fire — which is the one way this
    repair could have made a message worse instead of better.
    """
    with pytest.raises(ValueError) as exc:
        _parse_combination({"name": "SLU",
                            "components": [{"ref": "G"}],
                            "section_ops": {"activate": []}})
    from gensec.io_yaml import VALID_COMBINATION_KEYS

    assert "only valid on a stage" in str(exc.value)
    assert "section_ops" in VALID_COMBINATION_KEYS


def test_internal_stage_keys_are_not_input_keys():
    r"""
    The parser injects ``_prestress_action_specs`` and
    ``_section_ops_spec`` into the stage it **returns**; they are never
    read from the stage it is **given**.

    Naming them in :data:`VALID_STAGE_KEYS` would advertise two private
    names as part of the document schema.  Keeping them out is only
    safe because the check reads the input, and this fixture is what
    holds that distinction still.
    """
    from gensec.io_yaml import VALID_STAGE_KEYS

    assert "_prestress_action_specs" not in VALID_STAGE_KEYS
    assert "_section_ops_spec" not in VALID_STAGE_KEYS
    out = _parse_combination({
        "name": "C",
        "stages": [{"name": "s", "components": [],
                    "prestress_actions": [{"P_kN": 100.0, "x": 0.0,
                                           "y": 0.0}]}]})
    assert "_prestress_action_specs" in out["stages"][0]


# ----------------------------------------------------------------------
#  The denominator, held still
# ----------------------------------------------------------------------

def test_every_distributed_example_still_loads():
    r"""
    0 unknown keys across the 24 distributed documents, at all three
    levels, measured on the incoming tree and required to stay so.

    This is the fixture that would have stopped the repair had the
    denominator been wrong: a refusal that broke a shipped example
    would be a defect found, not an obstacle to work around.
    """
    import glob
    import os

    from gensec.io_yaml import (VALID_COMBINATION_KEYS,
                                VALID_SECTION_KEYS, VALID_STAGE_KEYS)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = sorted(glob.glob(os.path.join(root, "examples", "*.yaml")))
    assert len(files) == 24
    unknown = []
    for path in files:
        data = yaml.safe_load(open(path, encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        sec = data.get("section")
        if isinstance(sec, dict):
            unknown += [(path, k) for k in sec
                        if k not in VALID_SECTION_KEYS]
        for combo in (data.get("combinations") or []):
            unknown += [(path, k) for k in combo
                        if k not in VALID_COMBINATION_KEYS]
            for stage in (combo.get("stages") or []):
                unknown += [(path, k) for k in stage
                            if k not in VALID_STAGE_KEYS]
    assert unknown == []


# ----------------------------------------------------------------------
#  The observation inventory — E34-1
# ----------------------------------------------------------------------

def test_every_module_carries_an_observation_declaration():
    r"""
    E34-1, turned into a gate.

    A merge read INERT on all three fingerprints not because it was
    right but because no probe looked at it.  The gate below refuses a
    module of ``src/gensec`` with no declaration, so a module arriving
    unwatched is a diff instead of a silence.

    RED on the incoming tree by construction: neither the tool nor the
    declarations existed.
    """
    import subprocess
    import sys

    root = _repo_root()
    proc = subprocess.run(
        [sys.executable,
         os.path.join(root, "tools", "observation_inventory.py"), "--gate"],
        cwd=root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_probe_debt_is_pinned_at_its_measured_value():
    r"""
    Nine modules produce or render numbers that no key watches.

    Pinned, not asserted to be zero.  The debt is real and each phase
    that lowers it does so with one designed bench, not with a batch of
    probes written to make the number look better.  What the pin buys
    is that the debt cannot **grow** unnoticed, which is the only way it
    ever got to thirteen.

    Phase 36 took ``output/report.py`` off (Z-12).  Phase 37 took two
    off: ``solver/prestress_transfer.py``, reached by the two
    ``d1c.divergence.*.max_rel_sigma_closed_form`` keys, of which the
    linear one is exact to the last bit; and ``cli.py``, reached by
    the ``cli.*`` bench that holds the run verb's wiring.

    Phase 38 took a third off, and by a route the earlier two did not
    use: ``_presets.py`` was not observed, it was **deleted**.  The
    module was imported by nothing anywhere in the tree, and the eight
    resolution parameters it carried were a second, dead home for the
    ones the YAML ``output`` block already declares -- disagreeing with
    it, too: ``n_angles_mx_my`` was 36 there and 144 here (38_2 section
    7.4).  The debt went with the module.

    That route needs guarding, because it is the one that could be
    abused: deleting a module is a legitimate way to lower this number
    only when the module was dead, and dead is a measurement, not an
    impression.  Lower it this way only with the grep that shows no
    importer.

    Raise this number only together with the reason a module joined the
    list; lower it only together with the bench that took one off, or
    with the measurement that showed there was nothing there.
    """
    import sys

    sys.path.insert(0, os.path.join(_repo_root(), "tools"))
    import observation_reasons as R

    debt = sorted(m for m, r in R.UNOBSERVED.items()
                  if r.lstrip().lstrip("*").startswith("DEBT"))
    assert len(R.OBSERVED) + len(R.UNOBSERVED) == 51
    assert len(debt) == 9, debt
    # Removed in Phase 38 rather than observed.  Asserted absent from
    # both tables, not merely off the debt list: a module that came back
    # into the declarations without coming back into the tree would be a
    # declaration about nothing.
    assert "_presets.py" not in R.UNOBSERVED
    assert "_presets.py" not in R.OBSERVED
    assert not os.path.exists(
        os.path.join(_repo_root(), "src", "gensec", "_presets.py"))
    # The one that was not a renderer and not an entry point.  A solver
    # module with no observer is E34-1's exact shape; its repair was a
    # visible edit to this fixture, which is why the name stays here as
    # a negative assertion instead of disappearing.
    assert "solver/prestress_transfer.py" not in debt
    assert "solver/prestress_transfer.py" in R.OBSERVED
    assert "cli.py" not in debt
    assert "cli.py" in R.OBSERVED
