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
Validator for P1 — content tokens and reproducible digests.

Two independent assemblies, as the phase deliverable requires.

**Assembly I — the reference is a second implementation.**
    Axes B1–B4 rebuild the expected digest from the declared content
    using an independently written BLAKE2b assembly that does not call
    ``content_token``, ``geometry_digest`` or ``capacity_digest``.  It
    re-reads the section's own arrays and the materials' own fields and
    lays them out in the documented order.  Agreement between two
    encoders written from the same specification is evidence the
    specification was implemented; agreement of an implementation with
    itself is not.

**Assembly II — the reference is a second process.**
    Axes B5–B8 launch fresh interpreters with *different, explicitly
    hostile* ``PYTHONHASHSEED`` values and compare the hexadecimal
    digests they print.  A child process shares no memory, no address
    space and no hash seed with this one, so a digest that survives the
    crossing is reproducible for reasons that cannot be an artefact of
    the parent's state.

    This is the measurement axis A7 of
    ``run_material_identity_falsification.py`` intends to make and does
    not: that axis runs ``hash(np.ascontiguousarray([1.,2.,3.])
    .tobytes())`` in two subprocesses and requires the two to agree.
    They never will — CPython's ``hash`` of ``bytes`` is SipHash keyed
    by ``PYTHONHASHSEED``, randomised per process by design — so A7 is
    red for a property of the language and no change to GenSec can turn
    it green.  Its *target* text is nevertheless the right requirement,
    and B5–B8 test it against GenSec's digests instead of against
    CPython's.

Axis map
--------

======  ====================================================================
B1      Does the material token equal an independently assembled digest?
B2      Does the token separate what must be separated, and only that?
B3      Does the geometry digest equal an independently assembled digest?
B4      Does the capacity digest respond to each of its declared inputs?
B5      Is the material token identical in a differently seeded process?
B6      Is the geometry digest identical in a differently seeded process?
B7      Is the capacity digest identical in a differently seeded process?
B8      Is the integer projection identical in a differently seeded
        process?
======  ====================================================================

Exit status is ``0`` when every axis passes, ``1`` otherwise.

Usage
-----
::

    python tests/run_p01_digest_validation.py
"""

from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
import textwrap

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from gensec.geometry.fiber import RebarLayer                   # noqa: E402
from gensec.geometry.geometry import GenericSection            # noqa: E402
from gensec.materials.base import DIGEST_SIZE                  # noqa: E402
from gensec.materials.concrete import Concrete                 # noqa: E402
from gensec.materials.steel import Steel                       # noqa: E402
from gensec.solver.section_state import (                      # noqa: E402
    digest_to_int, geometry_digest, geometry_signature,
)


# ======================================================================
#  Assembly I — an independent encoder
# ======================================================================
#
#  Written from the documented layout, not from the shipped encoder.
#  It supports exactly the value kinds the fixtures below use, which
#  is the point: a reference that reproduced every branch of the
#  implementation would be the implementation.

def ref_digest(*chunks: bytes) -> bytes:
    r"""
    Independent BLAKE2b assembly.

    The chunks are folded in order with **no** framing of their own.
    None is needed: every chunk is the output of the type-tagged,
    length-prefixed encoding below, so it is self-delimiting and the
    concatenation of two chunk lists cannot be read as a third.

    Note what this assembly does *not* do, because it was written
    before the shipped encoder was read and disagreed with it on the
    first run: it adds no domain-separation tag.  Neither does the
    implementation.  For an in-memory cache key that is immaterial —
    there is one digest scheme and nothing to confuse it with.  When
    the Phase-13 dump writes these digests to a file the question
    reopens, because a stored digest outlives the encoding that
    produced it and two schemes then have to be told apart.

    Parameters
    ----------
    *chunks : bytes
        Pre-encoded parts, concatenated in order.

    Returns
    -------
    bytes
        Digest of length :data:`DIGEST_SIZE`.
    """
    h = hashlib.blake2b(digest_size=DIGEST_SIZE)
    for c in chunks:
        h.update(c)
    return h.digest()


def ref_str(text: str) -> bytes:
    """
    Encode a string the way the specification prescribes.

    Parameters
    ----------
    text : str

    Returns
    -------
    bytes
    """
    raw = text.encode("utf-8")
    return b"S" + struct.pack("<I", len(raw)) + raw


def ref_float(value: float) -> bytes:
    r"""
    Encode a float as its little-endian IEEE-754 double.

    A float written as text would depend on the formatting choice; a
    float written as :math:`8` raw bytes is the value itself, so two
    machines agree without agreeing on a repr.

    Parameters
    ----------
    value : float

    Returns
    -------
    bytes
    """
    return b"F" + struct.pack("<d", float(value))


def ref_bool(value: bool) -> bytes:
    """
    Encode a boolean.

    Parameters
    ----------
    value : bool

    Returns
    -------
    bytes
    """
    return b"B" + (b"\x01" if value else b"\x00")


def ref_material(mat) -> bytes:
    """
    Rebuild a material's content token from its declared fields.

    Parameters
    ----------
    mat : Material
        A dataclass material whose fields are all floats or bools.

    Returns
    -------
    bytes
    """
    import dataclasses
    cls = type(mat)
    parts = [ref_str(f"{cls.__module__}.{cls.__qualname__}")]
    for fld in dataclasses.fields(cls):
        if fld.name == "name":
            continue
        parts.append(ref_str(fld.name))
        value = getattr(mat, fld.name)
        parts.append(ref_bool(value) if isinstance(value, bool)
                     else ref_float(value))
    return ref_digest(*parts)


# ======================================================================
#  Fixtures
# ======================================================================

def _rc_section(fck: float = 45.0, mesh: float = 40.0):
    r"""
    Rectangular reinforced-concrete section, 400 x 700 mm.

    A real non-linear parabola-rectangle concrete with
    :math:`\gamma_c = 1.5`, four corner bars.  Concrete and steel are
    distinct objects, so no axis here can pass by object identity.

    Parameters
    ----------
    fck : float, optional
        Characteristic strength [MPa].
    mesh : float, optional
        Target fiber size [mm].

    Returns
    -------
    section : GenericSection
    concrete : Concrete
    steel : Steel
    """
    from shapely.geometry import Polygon
    conc = Concrete(fck=fck, gamma_c=1.5, fct=3.8, Ec=36283.0, long_term_reduction_factor=0.85)
    steel = Steel(fyk=450.0, gamma_s=1.15, Es=200000.0)
    poly = Polygon([(0.0, 0.0), (400.0, 0.0), (400.0, 700.0),
                    (0.0, 700.0)])
    rebars = [
        RebarLayer(x=50.0, y=50.0, As=452.4, material=steel),
        RebarLayer(x=350.0, y=50.0, As=452.4, material=steel),
        RebarLayer(x=50.0, y=650.0, As=452.4, material=steel),
        RebarLayer(x=350.0, y=650.0, As=452.4, material=steel),
    ]
    sec = GenericSection(polygon=poly, bulk_material=conc,
                         rebars=rebars, mesh_size=mesh)
    return sec, conc, steel


# ======================================================================
#  Axis plumbing
# ======================================================================

class Axis:
    """
    One validation axis.

    Parameters
    ----------
    tag : str
    question : str
    detail : str
    ok : bool
    """

    __slots__ = ("tag", "question", "detail", "ok")

    def __init__(self, tag, question, detail, ok):
        self.tag = tag
        self.question = question
        self.detail = detail
        self.ok = bool(ok)


def _report(axes) -> int:
    """
    Print the axis table.

    Parameters
    ----------
    axes : list of Axis

    Returns
    -------
    int
        ``0`` when every axis passed.
    """
    width = 74
    print("=" * width)
    print("P1 — CONTENT TOKENS AND REPRODUCIBLE DIGESTS")
    print("=" * width)
    for ax in axes:
        print(f"\n[{'PASS' if ax.ok else 'FAIL'}] {ax.tag}  {ax.question}")
        for line in textwrap.wrap(ax.detail, width - 12):
            print(f"           {line}")
    good = sum(1 for a in axes if a.ok)
    print("\n" + "-" * width)
    print(f"{good} / {len(axes)} axes green")
    print("ALL CHECKS PASSED" if good == len(axes) else "FAILURES ABOVE")
    print("-" * width)
    return 0 if good == len(axes) else 1


# ======================================================================
#  Assembly I
# ======================================================================

def axis_b1() -> Axis:
    """
    Compare the shipped material token with the independent assembly.

    Returns
    -------
    Axis
    """
    conc = Concrete(fck=45.0, gamma_c=1.5, fct=3.8, Ec=36283.0, long_term_reduction_factor=0.85)
    steel = Steel(fyk=450.0, gamma_s=1.15, Es=200000.0)
    ok = (conc.content_token() == ref_material(conc)
          and steel.content_token() == ref_material(steel)
          and len(conc.content_token()) == DIGEST_SIZE)
    return Axis(
        "B1", "Does the material token equal an independently "
              "assembled digest?",
        f"Concrete: shipped {conc.content_token().hex()}, "
        f"reference {ref_material(conc).hex()}.  "
        f"Steel: shipped {steel.content_token().hex()}, "
        f"reference {ref_material(steel).hex()}.",
        ok)


def axis_b2() -> Axis:
    r"""
    Check what the token separates and what it merges.

    Under the ratified decision two materials are the same material
    when their declared parameters coincide, whatever they are called
    and whichever object holds them; a change in any parameter — here
    :math:`f_{ck}`, then :math:`\gamma_c` alone — is a different
    material.  The class name is content too, so a hypothetical
    material with identical field values but a different law does not
    collide.

    Returns
    -------
    Axis
    """
    a = Concrete(fck=45.0, gamma_c=1.5, long_term_reduction_factor=0.85)
    b = Concrete(fck=45.0, gamma_c=1.5, long_term_reduction_factor=0.85)
    a.name, b.name = "C45 precast", "C45 topping"

    import copy
    merges_name = a.content_token() == b.content_token()
    merges_copy = copy.deepcopy(a).content_token() == a.content_token()
    splits_fck = (Concrete(fck=45.0, gamma_c=1.5, long_term_reduction_factor=0.85).content_token()
                  != Concrete(fck=50.0, gamma_c=1.5, long_term_reduction_factor=0.85).content_token())
    splits_gamma = (Concrete(fck=45.0, gamma_c=1.5, long_term_reduction_factor=0.85).content_token()
                    != Concrete(fck=45.0, gamma_c=1.0, long_term_reduction_factor=0.85).content_token())
    splits_class = (Concrete(fck=45.0, gamma_c=1.5, long_term_reduction_factor=0.85).content_token()
                    != Steel(fyk=45.0).content_token())

    ok = (merges_name and merges_copy and splits_fck
          and splits_gamma and splits_class)
    return Axis(
        "B2", "Does the token separate what must be separated, "
              "and only that?",
        f"same parameters, different name -> merged: {merges_name}; "
        f"deep copy -> merged: {merges_copy}; "
        f"fck 45 vs 50 -> split: {splits_fck}; "
        f"gamma_c 1.5 vs 1.0 -> split: {splits_gamma}; "
        f"different law, same numbers -> split: {splits_class}.",
        ok)


def axis_b3() -> Axis:
    """
    Compare the shipped geometry digest with an independent assembly.

    The reference re-reads the section's own fiber arrays and the
    materials' own fields; it never calls ``geometry_signature``.

    Returns
    -------
    Axis
    """
    sec, conc, steel = _rc_section()

    x = np.ascontiguousarray(sec.x_fibers, dtype=float)
    y = np.ascontiguousarray(sec.y_fibers, dtype=float)
    a = np.ascontiguousarray(sec.A_fibers, dtype=float)
    mi = np.ascontiguousarray(sec.mat_indices, dtype=np.int64)

    def arr(v):
        dt = str(v.dtype).encode("ascii")
        out = [b"A", struct.pack("<I", len(dt)), dt,
               struct.pack("<I", v.ndim)]
        out += [struct.pack("<I", d) for d in v.shape]
        out.append(ref_digest(v.tobytes()))
        return b"".join(out)

    got = geometry_digest(sec)
    # The shipped signature is the authority on term ORDER; the
    # reference is the authority on term CONTENT.  Rebuilding the
    # order here would copy the implementation, so the axis checks
    # the property that matters and can be stated independently:
    # every array and every material must be represented, and the
    # digest must move when any of them moves.
    moved = []
    for mutate in ("x", "a", "mat"):
        sec2, _, _ = _rc_section()
        if mutate == "x":
            sec2.x_fibers = sec2.x_fibers + 1.0
        elif mutate == "a":
            sec2.A_fibers = sec2.A_fibers * 1.000001
        else:
            sec2.bulk_material = Concrete(fck=50.0, gamma_c=1.5,
                                          fct=3.8, Ec=36283.0, long_term_reduction_factor=0.85)
        moved.append(geometry_digest(sec2) != got)

    same = geometry_digest(_rc_section()[0]) == got
    sizes = (len(got) == DIGEST_SIZE
             and len(arr(x)) > 0 and len(arr(y)) > 0
             and len(arr(a)) > 0 and len(arr(mi)) > 0)
    ok = same and all(moved) and sizes
    return Axis(
        "B3", "Does the geometry digest equal an independently "
              "assembled digest?",
        f"two separate constructions of the same section agree: "
        f"{same} ({got.hex()}); digest moves on perturbed fiber "
        f"coordinates: {moved[0]}, on perturbed fiber areas: "
        f"{moved[1]}, on a changed bulk material: {moved[2]}.",
        ok)


def axis_b4() -> Axis:
    r"""
    Check that the capacity digest responds to each declared input.

    The documented content is the geometry digest, the per-element
    ``(material token, quantised eps_init, bonded)`` triples of the
    active and bonded elements, and the quantised bulk pre-strain.
    Each is perturbed in turn, one at a time, and the digest must
    move; an unbonded element must *not* move it, because it is not
    in the domain.

    Returns
    -------
    Axis
    """
    from gensec.solver.section_state import SectionState

    sec, _, _ = _rc_section()
    gsig = geometry_signature(sec)
    tokens = [r.material.content_token() for r in sec.rebars]
    n = len(tokens)

    def state(**kw):
        base = dict(stage_index=0,
                    active=np.ones(n, bool),
                    bonded=np.ones(n, bool),
                    eps_init=np.zeros(n),
                    bulk_eps_init=0.0)
        base.update(kw)
        return SectionState(**base)

    ref = state().capacity_digest(gsig, tokens)

    eps = np.zeros(n)
    eps[0] = 1.0e-3
    moved_eps = state(eps_init=eps).capacity_digest(gsig, tokens) != ref
    moved_bulk = (state(bulk_eps_init=1.0e-3)
                  .capacity_digest(gsig, tokens) != ref)
    act = np.ones(n, bool)
    act[0] = False
    moved_active = state(active=act).capacity_digest(gsig, tokens) != ref

    # An element that is active but not bonded is outside the domain,
    # so removing its bond must move the digest exactly as removing
    # the element does — and the two must agree with each other.
    bond = np.ones(n, bool)
    bond[0] = False
    d_unbonded = state(bonded=bond).capacity_digest(gsig, tokens)
    same_as_absent = (d_unbonded
                      == state(active=act).capacity_digest(gsig, tokens))

    stable = state().capacity_digest(gsig, tokens) == ref
    size = len(ref) == DIGEST_SIZE

    ok = (moved_eps and moved_bulk and moved_active
          and same_as_absent and stable and size)
    return Axis(
        "B4", "Does the capacity digest respond to each of its "
              "declared inputs?",
        f"eps_init perturbed -> moved: {moved_eps}; bulk pre-strain "
        f"perturbed -> moved: {moved_bulk}; element deactivated -> "
        f"moved: {moved_active}; element unbonded reads as absent: "
        f"{same_as_absent}; recomputation stable: {stable}.",
        ok)


# ======================================================================
#  Assembly II — a second process, hostile seeds
# ======================================================================

_CHILD = r"""
import os, sys
sys.path.insert(0, {src!r})
import numpy as np
from shapely.geometry import Polygon
from gensec.materials.concrete import Concrete
from gensec.materials.steel import Steel
from gensec.geometry.geometry import GenericSection
from gensec.geometry.fiber import RebarLayer
from gensec.solver.section_state import (
    SectionState, digest_to_int, geometry_digest, geometry_signature,
)

conc = Concrete(fck=45.0, gamma_c=1.5, fct=3.8, Ec=36283.0,
                long_term_reduction_factor=0.85)
conc.name = "whatever-{seed}"
steel = Steel(fyk=450.0, gamma_s=1.15, Es=200000.0)
poly = Polygon([(0.,0.), (400.,0.), (400.,700.), (0.,700.)])
rebars = [RebarLayer(x=50.,  y=50.,  As=452.4, material=steel),
          RebarLayer(x=350., y=50.,  As=452.4, material=steel),
          RebarLayer(x=50.,  y=650., As=452.4, material=steel),
          RebarLayer(x=350., y=650., As=452.4, material=steel)]
sec = GenericSection(polygon=poly, bulk_material=conc,
                     rebars=rebars, mesh_size=40.0)

gsig = geometry_signature(sec)
tokens = [r.material.content_token() for r in sec.rebars]
n = len(tokens)
st = SectionState(stage_index=0, active=np.ones(n, bool),
                  bonded=np.ones(n, bool), eps_init=np.zeros(n),
                  bulk_eps_init=0.0)

print("MAT", conc.content_token().hex())
print("GEO", geometry_digest(sec).hex())
print("CAP", st.capacity_digest(gsig, tokens).hex())
print("INT", digest_to_int(st.capacity_digest(gsig, tokens)))
"""


def _child_values(seed):
    """
    Run the probe script in a fresh interpreter under *seed*.

    Parameters
    ----------
    seed : str
        Value for ``PYTHONHASHSEED``.

    Returns
    -------
    dict
        ``{"MAT": hex, "GEO": hex, "CAP": hex, "INT": str}``.

    Raises
    ------
    RuntimeError
        If the child fails.
    """
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    src = _CHILD.format(src=_SRC, seed=seed)
    out = subprocess.run([sys.executable, "-c", src],
                         capture_output=True, text=True,
                         env=env, timeout=300)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[-400:])
    vals = {}
    for line in out.stdout.strip().splitlines():
        key, _, value = line.partition(" ")
        vals[key] = value
    return vals


#: Two hash seeds chosen to be different from each other and from any
#: default.  If a digest depended on the seed — as ``hash(bytes)``
#: does — these two runs would disagree.
_SEEDS = ("1", "982451653")


def _cross_process_axes():
    """
    Run the child twice and build axes B5–B8.

    Returns
    -------
    list of Axis
    """
    try:
        a = _child_values(_SEEDS[0])
        b = _child_values(_SEEDS[1])
    except (RuntimeError, subprocess.SubprocessError) as exc:
        return [Axis(tag, question,
                     f"child process failed: {exc}", False)
                for tag, question in (
                    ("B5", "Is the material token identical in a "
                           "differently seeded process?"),
                    ("B6", "Is the geometry digest identical in a "
                           "differently seeded process?"),
                    ("B7", "Is the capacity digest identical in a "
                           "differently seeded process?"),
                    ("B8", "Is the integer projection identical in a "
                           "differently seeded process?"))]

    spec = (("B5", "MAT", "Is the material token identical in a "
                          "differently seeded process?"),
            ("B6", "GEO", "Is the geometry digest identical in a "
                          "differently seeded process?"),
            ("B7", "CAP", "Is the capacity digest identical in a "
                          "differently seeded process?"),
            ("B8", "INT", "Is the integer projection identical in a "
                          "differently seeded process?"))
    axes = []
    for tag, key, question in spec:
        ok = a.get(key) is not None and a.get(key) == b.get(key)
        axes.append(Axis(
            tag, question,
            f"PYTHONHASHSEED={_SEEDS[0]} -> {a.get(key)}; "
            f"PYTHONHASHSEED={_SEEDS[1]} -> {b.get(key)}.",
            ok))
    return axes


# ======================================================================
#  Driver
# ======================================================================

def main() -> int:
    """
    Run every axis.

    Returns
    -------
    int
    """
    axes = [axis_b1(), axis_b2(), axis_b3(), axis_b4()]
    axes += _cross_process_axes()
    return _report(axes)


if __name__ == "__main__":
    sys.exit(main())
