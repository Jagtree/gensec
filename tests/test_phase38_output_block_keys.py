#!/usr/bin/env python3
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
The ``output`` block refuses a key it does not know.

The hole, and how it was found
------------------------------
Phase 36 unified six hand-written unknown-key refusals into
``_refuse_unknown_keys`` and wired it to ``section``,
``section.material_zones`` and the top level.  The ``output`` block was
not among them, and ``_parse_output_flags`` opens with::

    flags = dict(output_spec)

Every key the document wrote survives into the flags dictionary.  One
that nothing downstream reads is carried, ignored, and never mentioned.

This was not deduced.  Phase 38 inventoried every ``output`` block in
the tree -- 24 documents, 23 distinct keys -- while classifying the
observation debt (``38_2`` section 7.4), and found **three keys that no
module reads**:

``preset`` (``examples/example_tee.yaml``)
    Addressed ``_presets.py``, a module nothing imported.  Removed in
    Phase 38: the resolution parameters have one home, the ``output``
    block itself.
``nm_diagram`` (``examples/example_aci318.yaml``)
    Read by nothing anywhere in ``src``.
``eta_3D`` (``examples/example_aci318.yaml``)
    The trap of the three.  ``eta_3D`` **is** the name the demand
    architecture gives one of its four utilisation types, so a document
    writing ``eta_3D: true`` is asking for something real -- and the
    flags that implement it are called ``eta_norm``, ``eta_norm_beta``
    and ``eta_norm_ray``.  The document asks correctly, in the
    architecture's own vocabulary, and gets silence.

A shipped example carrying a key that does nothing is the strongest
argument the hole could have made for itself: if the documents this
project distributes drifted from the loader without anyone noticing,
a user's document can.

What is asserted
----------------
Two facts, and they are different.  That an unknown key **raises**, and
that the valid set is exactly the keys the loader and the CLI actually
consume -- a refusal whose admissible set is wrong refuses the right
thing for the wrong reason, and would let ``n_levels_mode`` through on a
technicality rather than because ``cli.py`` reads it.
"""
import io

import pytest
import yaml

from gensec import io_yaml


#: Keys the loader defaults in ``_parse_output_flags``, plus the two the
#: CLI reads directly from the resolved flags.  Named here so that the
#: test states the contract rather than reading the implementation and
#: agreeing with itself.
CONSUMED = {
    # utilisation flags
    "eta_norm", "eta_norm_beta", "eta_norm_ray", "eta_2D",
    "eta_path_norm_ray", "eta_path_norm_beta", "eta_path_2D",
    "delta_N_tol",
    # tiered reporting
    "verification_top_k", "fiber_details_top_k",
    # domain generation
    "generate_mx_my", "generate_3d_surface",
    "n_angles_mx_my", "n_scan_mx_my", "n_chi_mx_my",
    # moment-curvature and ductility
    "generate_moment_curvature", "generate_polar_ductility",
    "generate_3d_moment_curvature",
    # read by cli.py from the resolved flags (gensec/cli.py:251-252)
    "n_levels_mode", "n_levels_count",
}


def test_unknown_output_key_raises():
    """
    A key nothing reads must stop the load, not ride along.
    """
    with pytest.raises(ValueError) as exc:
        io_yaml._parse_output_flags({"eta_norm": True, "presset": "rapid"})
    msg = str(exc.value)
    assert "output" in msg and "presset" in msg, (
        f"the refusal must name the block and the key; got: {msg}")


def test_the_three_dead_keys_are_refused():
    """
    The three keys measured in the shipped examples, each on its own.

    Listed one by one rather than in a loop over a set: when this goes
    red it should say *which* key stopped being refused.
    """
    for dead in ("preset", "nm_diagram", "eta_3D"):
        with pytest.raises(ValueError):
            io_yaml._parse_output_flags({dead: True})


def test_every_consumed_key_is_admissible():
    """
    The admissible set covers exactly what is consumed.

    The second half of the contract.  A refusal that also refused
    ``n_levels_mode`` would be a regression dressed as rigour: six
    shipped documents use it and ``cli.py`` reads it.
    """
    flags = io_yaml._parse_output_flags({k: None for k in CONSUMED})
    assert set(flags) >= CONSUMED


def test_shipped_examples_survive_the_refusal():
    """
    Every ``output`` block this project distributes must load.

    The loop is the point: the hole was found because the examples had
    drifted, so the examples are what pins the repair.
    """
    import glob
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad = []
    for path in sorted(glob.glob(os.path.join(root, "examples", "*.yaml"))):
        with io.open(path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        if not isinstance(doc, dict) or not isinstance(
                doc.get("output"), dict):
            continue
        try:
            io_yaml._parse_output_flags(doc["output"])
        except ValueError as exc:
            bad.append(f"{os.path.basename(path)}: {exc}")
    assert not bad, "shipped examples rejected by their own loader:\n" + \
        "\n".join(bad)
