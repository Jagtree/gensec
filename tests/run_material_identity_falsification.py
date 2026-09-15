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
Falsification harness for the material-identity decision (12_0d).

Purpose
-------
This harness is **not** a validator of a shipped feature.  It is the
rule-3 fixture that did not exist: an executable statement of the
behaviour the identity-keying retrofit must change, written *before*
the retrofit so that it cannot be shaped by it.

It is deliberately **RED today**.  Every axis reports the currently
observed behaviour next to the contract ratified in
``12_0d-GENSEC_MATERIAL_IDENTITY_DECISION.md``, and the process exits
non-zero for as long as any axis still exhibits the pre-retrofit
behaviour.  Phase 12 turns it green; the day it goes green is the day
the decision is implemented, and the exit code is the gate.

Design constraints obeyed here (12_0 §2, rules 1–4)
---------------------------------------------------
* **Rule 2** — no axis reads a quantity through the same call the
  implementation uses to produce it.  A5 re-derives the signature in a
  **separate interpreter process**; A3/A4 read the modulus back out of
  the *substituted view* (``LinearElastic.E``), not out of the map that
  was fed in.
* **Rule 3** — every concrete axis uses a real non-linear
  :class:`~gensec.materials.concrete.Concrete` with
  :math:`\gamma_c = 1.5`.  The flagship fixture
  (``example_composite_losses.yaml``) declares two *distinct* zone
  materials (``c_precast``, ``c_topping``) and therefore **cannot**
  exhibit A3/A4 at all; the shared-object section built here is the
  missing fixture.
* **Rule 4** — this file is meant to be *run*.  It is named
  ``run_*`` and is therefore not collected by pytest
  (``python_files = test_*.py``), exactly like
  ``aci209_falsification.py`` and the ``run_phase8_task*`` validators.

Axes
----
======  ==========================================================
Axis    Question
======  ==========================================================
A1      Can a material instance be used as a mapping key at all?
A2      Does material equality distinguish two differently-named
        materials with identical mechanical parameters?
A3      Do two bulk zones that share one material object receive
        their own SLS moduli?
A4      Does an uncast-zone placeholder stay confined to the zone
        being cast?
A5      Is the capacity signature reproducible across processes?
A6      Is the capacity signature invariant under
        :func:`copy.deepcopy`?
A7      Is ``id()`` the *only* source of signature
        irreproducibility?
======  ==========================================================

Usage
-----
.. code-block:: console

    $ python tests/run_material_identity_falsification.py
    $ PYTHONPATH=src python tests/run_material_identity_falsification.py

Exit status
-----------
``0`` when every axis matches the ratified target (retrofit complete),
``1`` otherwise.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# --- Make the in-tree package importable when run from the repo root. ---
_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np                                          # noqa: E402
from shapely.geometry import Polygon                         # noqa: E402

from gensec.geometry.fiber import RebarLayer                 # noqa: E402
from gensec.geometry.section import GenericSection           # noqa: E402
from gensec.materials.concrete import Concrete               # noqa: E402
from gensec.materials.steel import Steel                     # noqa: E402
from gensec.solver.losses import (                           # noqa: E402
    _UNCAST_PLACEHOLDER_E, _zone_material,
)
from gensec.solver.section_state import geometry_signature   # noqa: E402
from gensec.solver.sls import resolve_sls_moduli, sls_view   # noqa: E402


# ==================================================================
#  Fixture — rule 3: real non-linear concrete, gamma_c = 1.5
# ==================================================================

#: Substrate modulus fed to the resolver [MPa].  Order of magnitude of
#: :math:`E_{cm}(90\,\mathrm{d})` for a C45 (12_0 §1 measured table).
E_SUBSTRATE = 36283.0

#: Topping modulus fed to the resolver [MPa].  Order of magnitude of
#: :math:`E_{cm}(4\,\mathrm{d})` for the same class — the two differ by
#: ~39 %, which is what makes a collapse observable rather than
#: cosmetic.
E_TOPPING = 22000.0


def _shared_material_section():
    r"""
    Two-zone section whose zones share **one** material object.

    This is reachable from the public YAML surface with no trickery:
    ``io_yaml.load_yaml`` stores materials in a ``{name: object}``
    mapping and hands ``materials[zone_mat_name]`` to every zone, so
    declaring the same material name for the base bulk and for a
    ``material_zones`` entry yields the identical object in both
    slots.

    Returns
    -------
    tuple
        ``(section, concrete, steel)``.  The caller needs the
        material objects to build override maps.
    """
    conc = Concrete(fck=45.0, gamma_c=1.5, long_term_reduction_factor=0.85)      # real, non-linear law
    conc.name = "C45"
    steel = Steel(fyk=450.0)
    steel.name = "B450C"

    outline = Polygon([(0.0, 0.0), (300.0, 0.0),
                       (300.0, 600.0), (0.0, 600.0)])
    topping = Polygon([(0.0, 500.0), (300.0, 500.0),
                       (300.0, 600.0), (0.0, 600.0)])

    sec = GenericSection(
        polygon=outline,
        bulk_material=conc,
        rebars=[RebarLayer(y=50.0, As=628.0, material=steel, x=150.0)],
        mesh_size=25.0,
        bulk_materials=[(topping, conc, "topping")],
    )
    return sec, conc, steel


def _distinct_material_section():
    r"""
    Single-zone reference section, for the signature axes.

    Returns
    -------
    gensec.geometry.section.GenericSection
    """
    conc = Concrete(fck=45.0, gamma_c=1.5, long_term_reduction_factor=0.85)
    conc.name = "C45"
    steel = Steel(fyk=450.0)
    steel.name = "B450C"
    return GenericSection(
        polygon=Polygon([(0.0, 0.0), (300.0, 0.0),
                         (300.0, 600.0), (0.0, 600.0)]),
        bulk_material=conc,
        rebars=[RebarLayer(y=50.0, As=628.0, material=steel, x=150.0)],
        mesh_size=25.0,
    )


def _zone_key(section, z, n_zones=2):
    r"""
    The key form a per-zone modulus write uses on **this** tree.

    Before the 12_0d S1 retrofit the per-zone loops of ``losses.py``
    and ``timeline.py`` write ``id(material)``; after it they write a
    zone slot, and a bare integer is rejected by contract.  Probing
    which form the tree accepts keeps A3 and A4 **runnable on both
    trees**, so they measure the behaviour they were written to
    measure instead of crashing on a retired key form.

    Only the input side moves.  Each axis still reads its verdict out
    of the substituted view, and each can still fail.

    Parameters
    ----------
    section : GenericSection
    z : int
        Zone index.
    n_zones : int, optional
        Zones the probe must cover for a resolution to succeed.

    Returns
    -------
    tuple or int
        ``("zone", z)`` post-retrofit, ``id(material)`` before it.
    """
    probe = {("zone", zz): 1.0e4 for zz in range(n_zones)}
    try:
        resolve_sls_moduli(section, probe)
    except Exception:                                    # noqa: BLE001
        return id(_zone_material(section, z))
    return ("zone", z)


# ==================================================================
#  Axis result plumbing
# ==================================================================

class Axis:
    """
    One falsification axis.

    Parameters
    ----------
    tag : str
        Short identifier, ``"A1"`` … ``"A7"``.
    question : str
        The question the axis answers, in one line.
    observed : str
        What the tree does **now**.
    target : str
        What the ratified decision requires.
    on_target : bool
        ``True`` when ``observed`` already satisfies ``target``.
    """

    __slots__ = ("tag", "question", "observed", "target", "on_target")

    def __init__(self, tag, question, observed, target, on_target):
        self.tag = tag
        self.question = question
        self.observed = observed
        self.target = target
        self.on_target = bool(on_target)


def _report(axes):
    """
    Print the axis table and return the process exit status.

    Parameters
    ----------
    axes : list of Axis

    Returns
    -------
    int
        ``0`` if every axis is on target, else ``1``.
    """
    width = 74
    print("=" * width)
    print("MATERIAL IDENTITY — FALSIFICATION HARNESS  (12_0d)")
    print("=" * width)
    for ax in axes:
        mark = "GREEN" if ax.on_target else " RED "
        print(f"\n[{mark}] {ax.tag}  {ax.question}")
        for label, text in (("observed", ax.observed),
                            ("target  ", ax.target)):
            body = textwrap.wrap(text, width - 14)
            print(f"    {label} : {body[0] if body else ''}")
            for cont in body[1:]:
                print(f"               {cont}")
    green = sum(1 for a in axes if a.on_target)
    print("\n" + "-" * width)
    print(f"{green} / {len(axes)} axes on target")
    if green == len(axes):
        print("RETROFIT COMPLETE — the 12_0d contract holds on this tree.")
        return 0
    print("INCOMPLETE — the 12_0d contract does NOT hold on this tree.")
    print("Phase 12 has landed: a red axis here is a regression, not "
          "the expected state.")
    print("-" * width)
    return 1


# ==================================================================
#  A1 — is a material usable as a mapping key?
# ==================================================================

def axis_a1():
    r"""
    Probe whether a material instance can key a mapping.

    ``resolve_sls_moduli``'s docstring advertises resolution *"by
    **instance** (``moduli`` key is the material object itself)"*.
    Every material in the library is an unfrozen ``@dataclass`` with
    the default ``eq=True``, which sets ``__hash__ = None`` — so the
    caller cannot construct that mapping at all, and the documented
    branch is unreachable.  Under the ratified decision the
    unhashability is **retained on purpose**: it makes the condemned
    pattern unrepresentable rather than merely discouraged, and the
    override contract moves to slot keys.

    Returns
    -------
    Axis
    """
    _sec, conc, _steel = _shared_material_section()
    try:
        _ = {conc: E_SUBSTRATE}
        hashable = True
    except TypeError:
        hashable = False

    doc = resolve_sls_moduli.__doc__ or ""
    promises = "by **instance**" in doc

    # Positive probe: the successor contract has to *work*.  Without
    # it this axis would go green on a docstring edit alone, which is
    # the weakest gate a retrofit could be given.
    probe_sec, _c2, _s2 = _shared_material_section()
    try:
        resolve_sls_moduli(probe_sec, {("zone", 0): E_SUBSTRATE,
                                       ("zone", 1): E_TOPPING})
        slots_live = True
    except Exception:                                    # noqa: BLE001
        slots_live = False

    observed = (
        f"{type(conc).__name__} hashable: {hashable}; "
        f"resolve_sls_moduli's docstring still advertises a "
        f"by-instance branch: {promises}; slot keys ('zone', n) "
        f"accepted: {slots_live}."
        + ("" if not (promises and not hashable) else
           "  The advertised branch cannot be exercised by any caller "
           "— a documented capability that does not exist.")
    )
    return Axis(
        "A1", "Does the override contract match what callers can do?",
        observed,
        "Materials stay unhashable by contract — the condemned "
        "{material: E} pattern is unrepresentable, not merely "
        "discouraged — and the docstring advertises slot keys "
        "instead of a by-instance branch.",
        (not hashable) and (not promises) and slots_live,
    )


# ==================================================================
#  A2 — does equality distinguish named materials?
# ==================================================================

def axis_a2():
    r"""
    Probe that material identity is **content-derived**, not nominal.

    Rewritten for ``12_7`` §4.  The previous form asked whether
    ``name`` participates in equality and called ``True`` a defect.
    That assertion contradicts the ratified design: identity is a
    BLAKE2b digest over *declared content* (P1), a name is a label an
    engineer chose for a report, and two zones that declare the same
    concrete **are** the same material however they are labelled --
    which is precisely what lets the modulus cache be reused
    correctly.  Asserting the opposite could not be satisfied by any
    correct implementation, so the axis measured the wrong thing.

    What replaces it is the property the design actually claims, in
    both directions:

    * same declared content, different labels
      :math:`\Rightarrow` one identity;
    * any difference in declared content, same label
      :math:`\Rightarrow` two identities.

    The trap the old axis worried about is real and stays documented:
    a name is not identity.  It is pinned here rather than left
    latent.

    Returns
    -------
    Axis
    """
    a = Concrete(fck=45.0, gamma_c=1.5, long_term_reduction_factor=0.85)
    a.name = "C45 precast"
    b = Concrete(fck=45.0, gamma_c=1.5, long_term_reduction_factor=0.85)
    b.name = "C45 topping"
    c = Concrete(fck=50.0, gamma_c=1.5, long_term_reduction_factor=0.85)
    c.name = "C45 precast"          # same label, different content

    same_token = a.content_token() == b.content_token()
    diff_token = a.content_token() != c.content_token()
    label_free = bool(a == b)

    observed = (
        f"content_token('C45 precast') == content_token('C45 topping') "
        f"-> {same_token} (same declared content, different labels). "
        f"content_token(fck=45) != content_token(fck=50) under the same "
        f"label -> {diff_token}. Value equality follows the token "
        f"-> {label_free}. The name is a report label and is "
        f"deliberately absent from identity."
    )
    return Axis(
        "A2", "Is material identity content-derived, not nominal?",
        observed,
        "One identity per declared content, in both directions: same "
        "content under different labels collapses to one key, any "
        "content difference under the same label does not. The name "
        "is not identity, and that is pinned here rather than latent.",
        same_token and diff_token and label_free,
    )


# ==================================================================
#  A3 — zone/material axis collapse
# ==================================================================

def axis_a3():
    r"""
    Probe the modulus map for a zone-axis collapse.

    The SLS/datum modulus of a bulk zone is
    :math:`E_{c,z}(t_z)` — a function of the **zone's** age, not of
    the material's declared parameters.  Keying the map on the
    material makes the two zones of a shared-object section
    indistinguishable, and the per-zone write loop
    (``losses.py`` ~882-888, ``timeline.py`` ~786-800) silently
    overwrites: the surviving value is whichever zone the loop
    visited last.

    Read-back is deliberately taken from the substituted view's
    ``LinearElastic.E`` rather than from the map that was fed in
    (rule 2: the reference must not share the implementation's
    inputs).

    Returns
    -------
    Axis
    """
    sec, _conc, _steel = _shared_material_section()

    # Shape of the per-zone loops in losses.py / timeline.py, written
    # in the key form this tree accepts (see _zone_key).  Only the
    # input side of the axis moved with the retrofit; the predicate --
    # read back out of the substituted view -- did not.
    mod = {}
    per_zone = {0: E_SUBSTRATE, 1: E_TOPPING}
    for z in (0, 1):
        mod[_zone_key(sec, z)] = per_zone[z]

    view = sls_view(sec, resolve_sls_moduli(sec, mod))
    e_base = float(view.bulk_material.E)
    e_topping = float(view.bulk_materials[0][1].E)
    separated = (abs(e_base - E_SUBSTRATE) < 1.0
                 and abs(e_topping - E_TOPPING) < 1.0)

    observed = (
        f"fed zone 0 = {E_SUBSTRATE:.0f} MPa and zone 1 = "
        f"{E_TOPPING:.0f} MPa; the map holds {len(mod)} entr"
        f"{'y' if len(mod) == 1 else 'ies'} and the substituted view "
        f"reads zone 0 = {e_base:.0f} MPa, zone 1 = {e_topping:.0f} "
        f"MPa (error on zone 0: "
        f"{100.0 * (e_base - E_SUBSTRATE) / E_SUBSTRATE:+.1f} %)."
    )
    return Axis(
        "A3", "Do two zones sharing one material object keep "
              "their own moduli?",
        observed,
        "The modulus map is indexed by bulk-zone slot, so the two "
        "zones carry independent moduli whether or not they share a "
        "material object.",
        separated,
    )


# ==================================================================
#  A4 — uncast-zone placeholder containment
# ==================================================================

def axis_a4():
    r"""
    Probe whether the uncast placeholder stays inside its own zone.

    ``ConstructionTimeline._service_datum`` writes
    :data:`~gensec.solver.losses._UNCAST_PLACEHOLDER_E` (1 MPa) for the
    zone being cast and the zone's real :math:`E_c(t)` for every active
    substrate zone.  With a shared material object and the cast zone
    visited **after** the substrate — which the ascending
    ``for z in range(n_zones)`` walk guarantees for a topping — the
    placeholder overwrites the substrate's modulus.  The service
    casting datum is then drawn on a section four orders of magnitude
    too soft, and it converges, so nothing complains.

    The C5 perturbation defence ("perturb the placeholder by
    :math:`10^6` and check the answer does not move") does not reach
    this: it varies the placeholder in a fixture whose zones hold
    *distinct* material objects, where the leak cannot occur.

    Returns
    -------
    Axis
    """
    sec, _conc, _steel = _shared_material_section()
    zi = 1                                  # the zone being cast

    # Shape of timeline._service_datum's zone loop, written in the key
    # form this tree accepts (see _zone_key).
    mod = {}
    for z in range(2):
        if z == zi:
            mod[_zone_key(sec, z)] = _UNCAST_PLACEHOLDER_E
            continue
        mod[_zone_key(sec, z)] = E_SUBSTRATE

    view = sls_view(sec, resolve_sls_moduli(sec, mod))
    e_base = float(view.bulk_material.E)
    contained = abs(e_base - E_SUBSTRATE) < 1.0

    observed = (
        f"casting zone {zi}; substrate fed {E_SUBSTRATE:.0f} MPa and "
        f"the cast zone fed the {_UNCAST_PLACEHOLDER_E:.0f} MPa "
        f"placeholder. The substituted substrate law reads "
        f"{e_base:.1f} MPa"
        + ("." if contained else
           f" — the placeholder has leaked into the load-bearing "
           f"substrate, a factor "
           f"{E_SUBSTRATE / max(e_base, 1e-30):.0f} too soft.")
    )
    return Axis(
        "A4", "Does the uncast-zone placeholder stay in its own zone?",
        observed,
        "The placeholder is written into one slot of a zone-indexed "
        "array and cannot reach another zone, whatever the material "
        "objects are.",
        contained,
    )


# ==================================================================
#  A5 — cross-process reproducibility of the capacity signature
# ==================================================================

_CHILD_PROBE = r"""
import sys
sys.path.insert(0, {src!r})
from shapely.geometry import Polygon
from gensec.geometry.fiber import RebarLayer
from gensec.geometry.section import GenericSection
from gensec.materials.concrete import Concrete
from gensec.materials.steel import Steel
from gensec.solver.section_state import geometry_signature

conc = Concrete(fck=45.0, gamma_c=1.5,
                long_term_reduction_factor=0.85); conc.name = "C45"
steel = Steel(fyk=450.0); steel.name = "B450C"
sec = GenericSection(
    polygon=Polygon([(0.0, 0.0), (300.0, 0.0), (300.0, 600.0), (0.0, 600.0)]),
    bulk_material=conc,
    rebars=[RebarLayer(y=50.0, As=628.0, material=steel, x=150.0)],
    mesh_size=25.0)
sig = geometry_signature(sec)
print(hash(sig))
"""


def _child_signature(seed):
    """
    Compute the geometry signature in a fresh interpreter.

    Parameters
    ----------
    seed : str
        Value for ``PYTHONHASHSEED`` in the child environment.

    Returns
    -------
    str
        The child's ``hash(geometry_signature(section))``, or a
        diagnostic string starting with ``"ERR"``.
    """
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    src = str(_SRC) if _SRC.is_dir() else ""
    try:
        out = subprocess.run(
            [sys.executable, "-c", _CHILD_PROBE.format(src=src)],
            capture_output=True, text=True, env=env, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:   # pragma: no cover
        return f"ERR {exc}"
    if out.returncode != 0:                                # pragma: no cover
        return f"ERR rc={out.returncode} {out.stderr.strip()[:120]}"
    return out.stdout.strip()


def axis_a5():
    r"""
    Probe whether the capacity signature survives a process boundary.

    Two child interpreters compute the signature of the *same* section
    from the *same* source, with ``PYTHONHASHSEED`` pinned to ``0`` so
    that every seed-randomised term is neutralised.  Any residual
    difference is attributable to ``id()`` alone.

    This is the axis the existing
    ``test_geometry_signature_stable_and_hashable`` cannot fail: it
    compares two calls inside one process, where ``id()`` is constant
    by construction — the shape 12_0 §2 names ("the defence was built
    so that it could not fail").

    Returns
    -------
    Axis
    """
    a = _child_signature("0")
    b = _child_signature("0")
    ok = (not a.startswith("ERR")) and (not b.startswith("ERR")) and a == b
    observed = (
        f"two fresh interpreters, PYTHONHASHSEED=0, identical source: "
        f"{a} vs {b} -> "
        + ("equal." if a == b else "different (id()-derived term).")
    )
    return Axis(
        "A5", "Is the capacity signature reproducible across "
              "processes?",
        observed,
        "The signature is a stable digest over declared content and is "
        "byte-identical across processes and machines, so a serialised "
        "domain hash in the Phase-13 JSON dump is comparable.",
        ok,
    )


# ==================================================================
#  A6 — deepcopy invariance
# ==================================================================

def axis_a6():
    r"""
    Probe signature invariance under :func:`copy.deepcopy`.

    A deep copy of a section is, by content, the same section: same
    mesh, same declared constitutive parameters, same names.  Under
    ``id()`` keying it hashes differently, so an in-process cache
    misses (harmless) *and* a serialised hash disagrees with itself
    (not harmless).  ``copy`` is already imported in
    ``section_state.py`` and ``rheology.py``; nothing asserts against
    this.

    Returns
    -------
    Axis
    """
    sec = _distinct_material_section()
    h_orig = hash(geometry_signature(sec))
    h_copy = hash(geometry_signature(copy.deepcopy(sec)))
    ok = h_orig == h_copy
    observed = (
        f"original {h_orig} vs deepcopy {h_copy} -> "
        + ("equal." if ok else
           "different, although the two sections are identical by "
           "content. No assertion intercepts it.")
    )
    return Axis(
        "A6", "Is the capacity signature invariant under deepcopy?",
        observed,
        "Content-derived digest: a deep copy hashes identically, and "
        "the cache reuse it enables is correct rather than merely "
        "safe.",
        ok,
    )


# ==================================================================
#  A7 — is id() the only irreproducibility source?
# ==================================================================

def axis_a7():
    r"""
    Probe that **GenSec's** signature is seed-independent.

    Rewritten for ``12_7`` §4, extended for Phase 13.  The original
    form ran ``hash(ndarray.tobytes())`` in two default-seed
    subprocesses and reported the difference.  That measures CPython's
    SipHash keying, not GenSec: it is red on every tree ever built,
    before and after any retrofit, so it could not distinguish a fixed
    engine from a broken one.

    The question the axis exists to answer is whether the signature
    *this tree* produces is reproducible across processes.  Three terms
    are measured directly, in two subprocesses whose
    ``PYTHONHASHSEED`` is explicitly forced apart.  Forced, not left to
    the default randomisation: two random seeds may coincide, and an
    axis that can pass by luck is not an axis.

    1. the array digest of
       :func:`~gensec.solver.section_state.geometry_signature`;
    2. :meth:`~gensec.materials.base.Material.content_token`;
    3. **the content digest of the machine-readable timeline dump**
       (:func:`~gensec.output.timeline_dump.build_timeline_dump`).

    The third is the Phase-13 addition, and it is what makes the claim
    *"the dump's domain hashes go through BLAKE2b, never builtin
    ``hash()``"* a measurement rather than a statement.  Before it, the
    axis measured two functions the dump merely *resembles*: a
    serialiser that let a ``hash()`` in -- directly, or through a
    ``set`` iterated without ordering, which is the realistic way it
    happens -- would have left this axis green.  Now it turns it red,
    which is the designed signal.

    Returns
    -------
    Axis
    """
    probe = (
        "import numpy as np;"
        "from gensec.materials.base import digest_bytes;"
        "from gensec.materials.concrete import Concrete;"
        "from gensec.io_yaml import load_yaml;"
        "from gensec.solver.timeline import ConstructionTimeline;"
        "from gensec.output.timeline_dump import build_timeline_dump;"
        "a = np.ascontiguousarray([1.0, 2.0, 3.0]);"
        "m = Concrete(fck=45.0, gamma_c=1.5, "
        "             long_term_reduction_factor=0.85);"
        "d = load_yaml(%r);"
        "tl = ConstructionTimeline.from_document(d);"
        "db = {x['name']: {'N': x.get('N', 0.0), 'Mx': x.get('Mx', 0.0),"
        " 'My': x.get('My', 0.0)} for x in d['demands']};"
        "res = tl.resolve(d['section'], db);"
        "dump = build_timeline_dump(tl, res, d['section'], db,"
        " stem='a7', combinations=d['combinations']);"
        "print(digest_bytes(a.tobytes()).hex(), m.content_token().hex(),"
        " dump['content_digest'])"
    ) % str(_ROOT / "examples" / "example_composite_losses.yaml")
    vals = []
    for seed in ("0", "12345"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        env["MPLBACKEND"] = "Agg"
        env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
        out = subprocess.run([sys.executable, "-c", probe],
                             capture_output=True, text=True,
                             env=env, timeout=300)
        vals.append(out.stdout.strip() or f"<error: {out.stderr.strip()[:160]}>")
    stable = bool(vals[0]) and vals[0] == vals[1] \
        and not vals[0].startswith("<error")
    observed = (
        f"GenSec digest terms (array, material, timeline dump) under "
        f"PYTHONHASHSEED=0 and =12345: "
        f"{vals[0][:24]}... vs {vals[1][:24]}... -> "
        + ("identical; every term enters through BLAKE2b, so a "
           "serialised signature -- including the versioned dump -- is "
           "comparable across processes and machines." if stable else
           "different -- a builtin hash() survives somewhere in the "
           "signature or the dump path.")
    )
    return Axis(
        "A7", "Is the signature reproducible across processes?",
        observed,
        "Every term of the signature is seed-independent: array "
        "content, material identity and the machine-readable timeline "
        "dump all enter through a cryptographic digest, not through "
        "builtin hash().",
        stable,
    )


# ==================================================================
#  Entry point
# ==================================================================

def main():
    """
    Run every axis and report.

    Returns
    -------
    int
        Process exit status.
    """
    axes = [axis_a1(), axis_a2(), axis_a3(), axis_a4(),
            axis_a5(), axis_a6(), axis_a7()]
    return _report(axes)


if __name__ == "__main__":
    sys.exit(main())
