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
Phase 8, Task 1 patcher — engine-level bulk staging + per-zone
locked-in datum planes.

Implements the Task-1 slice of the Phase-8 timeline master plan
(``10_0`` / ``10_0b``, decisions B2-engine, B3–B8) per the primer
``10_1-GENSEC_TASK1_ENGINE_PRIMER.md``:

- **Named bulk zones** (B8): ``bulk_materials`` accepts
  ``(Polygon, Material, name)`` 3-tuples (2-tuples stay legal and
  auto-name); strict key validation on the YAML ``material_zones``
  parser (silent-ignore gap closed).
- **Per-zone activity + locked-in planes** (B2-engine):
  ``SectionState.bulk_active`` / ``SectionState.bulk_planes``, the
  atomic ``with_bulk_activated`` constructor-op, capacity-hash terms
  for mask and quantized planes (B4, curvature quantum
  ``QUANT_EPS / max(H, B)``).
- **Containment invariant + staging parent** (B3): geometric parents
  from ``mat_indices_*``; optional ``Tendon.parent`` override, legal
  only with ``embedded=False``; ``active[i] ⇒ bulk_active[parent(i)]``
  enforced per stage in ``resolve_stages``.
- **Re-slice masked view** (B5): exact active geometry via Shapely,
  ``_bounds``/``H``/``B``/``ideal_gross_area`` overridden per view,
  reference point pinned to the full-polygon centroid; all-True mask
  with zero planes = byte-identical fast path.
- **Per-fiber kernel offset field** (B6): ``_bulk_eps_by_group``
  retired; ``None`` fast path keeps the scalar branch numerics.
- **``Tendon.system`` retired** (B7): parser raises with a migration
  message; shipped example YAMLs updated.

Deliverable conventions (``10_0b`` §F): idempotent, CRLF-preserving,
uniqueness-checked string surgery; every edit either applies cleanly,
is detected as already applied, or fails loud.

**Partial-application safety.** Every ``old`` anchor must occur
*exactly once* in its file; a file in any intermediate manual state
fails the anchor check rather than silently no-op'ing.  The YAML
parser edit set keeps the existing guard property of
``_parse_section_ops_spec`` (unknown keys raise), so no intermediate
state can silently drop an ``activate_bulk`` block: until the parser
edit lands the key is *rejected*, never ignored.

Usage
-----
Run from anywhere inside the repository (the file resolver walks the
tree)::

    python apply_phase8_task1.py            # apply
    python apply_phase8_task1.py --check    # dry-run: report only

Exit code 0 iff every edit is applied (or already applied) and the
patched files compile.
"""

import argparse
import py_compile
import sys
from pathlib import Path

# ==================================================================
#  Patch infrastructure (CRLF-preserving, uniqueness-checked)
# ==================================================================

#: Directories never walked when resolving target files.
_NOISE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache",
    "node_modules", "build", "dist", ".mypy_cache", ".ruff_cache",
    "docs", "web", ".claude", ".github", "gensec.egg-info",
}


def _find_file(root: Path, filename: str, hint: str = "") -> Path:
    r"""
    Locate *filename* under *root*, excluding noise directories.

    Parameters
    ----------
    root : pathlib.Path
        Repository root (or any ancestor of the target).
    filename : str
        Bare file name to locate.
    hint : str, optional
        Required substring of the POSIX-style path **relative to
        root**, used to disambiguate names that legitimately exist in
        more than one package directory (e.g. ``geometry.py``).  Noise
        directories are matched on the relative parts only, so an
        absolute-prefix component (username, mount point) never
        triggers exclusion.

    Returns
    -------
    pathlib.Path

    Raises
    ------
    FileNotFoundError
        No candidate found.
    RuntimeError
        More than one candidate found (ambiguous target — the patcher
        refuses to guess).
    """
    hits = []
    for p in root.rglob(filename):
        # Noise is filtered on the path *relative to root*: components
        # of the absolute prefix (a username, a mount point) must
        # never trip the exclusion — only directories genuinely below
        # the search root do.
        rel = p.relative_to(root)
        if any(part in _NOISE_DIRS for part in rel.parts):
            continue
        if hint and hint not in rel.as_posix():
            continue
        hits.append(p)
    if not hits:
        raise FileNotFoundError(
            f"{filename!r} (hint={hint!r}) not found under {root}."
        )
    if len(hits) > 1:
        raise RuntimeError(
            f"{filename!r} (hint={hint!r}) is ambiguous under {root}: "
            f"{[str(h) for h in hits]}."
        )
    return hits[0]


def _package_root(root: Path) -> Path:
    r"""
    Locate the canonical ``src/gensec`` package directory.

    Resolution is anchored to the canonical ``src`` layout rather
    than walking the whole tree, so any shadow copies of the sources
    (e.g. iteration snapshots archived under a sibling directory) are
    excluded **by construction**, regardless of how that directory is
    named.  The ``<root>/src/gensec`` path is preferred; a filtered
    search (shallowest match, noise directories excluded) is the
    fallback for non-standard checkouts.

    Parameters
    ----------
    root : pathlib.Path
        Repository root (the ``--root`` argument).

    Returns
    -------
    pathlib.Path
        The ``src/gensec`` directory to resolve Python targets under.

    Raises
    ------
    FileNotFoundError
        No ``src/gensec`` package found under *root*.
    """
    direct = root / "src" / "gensec"
    if direct.is_dir():
        return direct
    cands = []
    for ini in root.rglob("__init__.py"):
        if any(part in _NOISE_DIRS for part in ini.parts):
            continue
        pkg = ini.parent
        if pkg.name == "gensec" and pkg.parent.name == "src":
            cands.append(pkg)
    if not cands:
        raise FileNotFoundError(
            f"canonical package src/gensec not found under {root}."
        )
    return min(cands, key=lambda p: len(p.parts))


def _examples_root(root: Path) -> Path:
    r"""
    Locate the canonical ``examples`` directory.

    Anchored to ``<root>/examples`` (sibling of ``src``); a filtered
    search is the fallback.  Anchoring is what disambiguates the
    shipped example YAMLs from any archived copies.

    Parameters
    ----------
    root : pathlib.Path
        Repository root.

    Returns
    -------
    pathlib.Path

    Raises
    ------
    FileNotFoundError
        No ``examples`` directory found under *root*.
    """
    direct = root / "examples"
    if direct.is_dir():
        return direct
    for cand in root.rglob("examples"):
        if cand.is_dir() and not any(
                part in _NOISE_DIRS for part in cand.parts):
            return cand
    raise FileNotFoundError(
        f"canonical examples/ directory not found under {root}."
    )


class PatchFile:
    r"""
    One target file: bytes in, LF-normalized text edits, original EOL
    back out.

    The dominant end-of-line convention is detected on read
    (``\r\n`` if any CRLF pair is present, else ``\n``) and restored
    verbatim on write, so a CRLF repository stays CRLF byte-for-byte
    on untouched lines.

    Parameters
    ----------
    path : pathlib.Path
    """

    def __init__(self, path: Path):
        self.path = path
        raw = path.read_bytes()
        self.crlf = b"\r\n" in raw
        self.text = raw.decode("utf-8").replace("\r\n", "\n")
        self.applied = 0
        self.skipped = 0
        self.failed = []

    def edit(self, label: str, old: str, new: str):
        r"""
        Replace *old* with *new*, exactly once.

        Idempotency contract:

        - *old* present exactly once and *new* absent → apply;
        - *old* absent and *new* present → already applied, skip;
        - anything else → hard failure (ambiguous or unknown state).

        Parameters
        ----------
        label : str
            Edit identifier for the report.
        old, new : str
            LF-normalized exact source fragments.
        """
        n_old = self.text.count(old)
        n_new = self.text.count(new)
        # ``old`` may be a substring of ``new`` (insertion-style
        # edits): in the applied state ``old`` then still occurs,
        # exactly ``n_new * new.count(old)`` times.  The
        # already-applied test must account for that, or a re-run
        # would duplicate the insertion.
        old_in_new = new.count(old)
        if n_new >= 1 and n_old == n_new * old_in_new:
            self.skipped += 1
            print(f"    [=] {label}: already applied")
            return
        if n_old == 1 and n_new == 0:
            self.text = self.text.replace(old, new, 1)
            self.applied += 1
            print(f"    [+] {label}: applied")
            return
        self.failed.append(
            f"{label}: anchor count old={n_old}, new={n_new} "
            f"(expected old=1/new=0 to apply, old=0/new>=1 to skip). "
            f"File is in an unknown intermediate state — refusing to "
            f"guess."
        )
        print(f"    [!] {label}: FAILED anchor check "
              f"(old={n_old}, new={n_new})")

    def save(self, check_only: bool):
        r"""Write back with the original EOL convention."""
        if check_only or not self.applied:
            return
        out = self.text
        if self.crlf:
            out = out.replace("\n", "\r\n")
        self.path.write_bytes(out.encode("utf-8"))


# ==================================================================
#  Edit definitions
# ==================================================================

def edits_geometry(pf: PatchFile):
    r"""``gensec/geometry/geometry.py`` — named zones, staging parents."""

    # G1 — bulk_materials parameter docstring: 2-/3-tuple contract.
    pf.edit(
        "G1 bulk_materials docstring",
        """    bulk_materials : list of tuple, optional
        Additional bulk material zones. Each tuple is
        ``(Polygon, Material)``. Fibers inside each zone use
        that zone's material instead of ``bulk_material``.
        Zones are checked in order; first match wins.
        Default empty (single-material section).
""",
        """    bulk_materials : list of tuple, optional
        Additional bulk material zones.  Each entry is either
        ``(Polygon, Material)`` or ``(Polygon, Material, name)``.
        Fibers inside each zone use that zone's material instead of
        ``bulk_material``.  Zones are checked in order; first match
        wins.  Default empty (single-material section).

        Zone *names* (Phase 8) are the stable staging references of
        the ``section_ops`` ``activate_bulk`` schema.  Unnamed zones
        (2-tuples, or 3-tuples with ``name=None``) are auto-named
        ``zone_<k>`` with *k* the 1-based position in this list; the
        implicit zone ``0`` (``bulk_material``) is named ``base`` and
        is always active.  Names must be unique, non-numeric strings
        distinct from ``'base'`` (a numeric name would be ambiguous
        with the 1-based integer zone reference).  After construction
        the normalized 2-tuples are stored back on this attribute and
        the names are exposed on :attr:`zone_names`, index-aligned
        with ``mat_indices`` values.
""",
    )

    # G2 — retire the stale "integrator does not yet consume" note on
    # bulk_eps_init (the Phase-5 bulk-kernel patch wired it in; the
    # note survived).  Documentation-accuracy fix, no behaviour.
    pf.edit(
        "G2 stale bulk_eps_init note",
        """        .. note::
           As of this phase the value is parsed, stored, hashed and
           propagated to every materialized view, but the integrator
           does **not yet** evaluate the bulk constitutive law at the
           offset argument :math:`\\varepsilon_{\\text{sec}} +
           \\varepsilon_{b,0}`.  A non-zero ``bulk_eps_init`` therefore
           shifts the domain *identity* (cache key) without yet shifting
           the domain *geometry*.  Wiring the offset into
           :meth:`~gensec.solver.integrator.FiberSolver.strain_field`
           and the displaced-bulk subtractions is a deliberate,
           separately-validated kernel change (losses/creep phase); see
           the deliverable note.  Sections that never set this field are
           unaffected.
""",
        """        .. note::
           Since the Phase-5 bulk-kernel patch the fiber integrator
           **consumes** this offset: the bulk constitutive law is
           evaluated at :math:`\\varepsilon_{\\text{sec}} +
           \\varepsilon_{b,0}` at the scalar, tangent and batch sites
           (validated by ``run_bulk_prestrain_validation_new.py``), so
           a non-zero value moves the resistance domain, not only the
           cache identity.  As of Phase 8 the scalar is one term of
           the general per-fiber offset field: it is added on top of
           the per-zone locked-in datum planes carried by
           :attr:`~gensec.solver.section_state.SectionState.bulk_planes`.
           Sections that never set this field are unaffected.
""",
    )

    # G3 — normalize zones before meshing (meshing calls
    # ``_material_index`` which relies on the normalized 2-tuples).
    pf.edit(
        "G3 __post_init__ zone normalization call",
        """        self._bounds = (minx, miny, maxx, maxy)

        # ---- Mesh ----
""",
        """        self._bounds = (minx, miny, maxx, maxy)

        # ---- Bulk zone normalization (Phase 8: named zones) ----
        # Must precede meshing: ``_material_index`` (called per fiber
        # during the mesh walk) unpacks the normalized 2-tuples.
        self._normalize_bulk_zones()

        # ---- Mesh ----
""",
    )

    # G4 — normalization + zone-reference helpers.
    pf.edit(
        "G4 zone helpers",
        """    def _material_index(self, x, y):
""",
        '''    def _normalize_bulk_zones(self):
        r"""
        Normalize ``bulk_materials`` entries and build the zone-name
        table.

        Accepts, per entry, either the legacy 2-tuple
        ``(Polygon, Material)`` or the Phase-8 3-tuple
        ``(Polygon, Material, name)``.  Entries are stored back as
        2-tuples (the internal contract every downstream consumer
        unpacks) and the names — auto-generated ``zone_<k>`` where not
        given — are collected on :attr:`zone_names`, index-aligned
        with ``mat_indices`` values (``zone_names[0] == 'base'``).

        Raises
        ------
        ValueError
            Malformed entry; non-string explicit name; the reserved
            name ``'base'``; a purely numeric name (ambiguous with the
            1-based integer zone reference of the staging schema); or
            a duplicate name.
        """
        norm, names = [], []
        for k, entry in enumerate(self.bulk_materials):
            entry_t = tuple(entry)
            if len(entry_t) == 2:
                zone_poly, zone_mat = entry_t
                name = None
            elif len(entry_t) == 3:
                zone_poly, zone_mat, name = entry_t
            else:
                raise ValueError(
                    f"bulk_materials[{k}]: expected (Polygon, Material)"
                    f" or (Polygon, Material, name), got a "
                    f"{len(entry_t)}-tuple."
                )
            if name is None:
                name = f"zone_{k + 1}"
            elif not isinstance(name, str):
                raise ValueError(
                    f"bulk_materials[{k}]: zone name must be a string, "
                    f"got {name!r}. Integer zone references are the "
                    f"1-based positions in this list and need no name."
                )
            if name == "base":
                raise ValueError(
                    f"bulk_materials[{k}]: 'base' is the reserved name "
                    f"of the implicit zone 0 (the primary "
                    f"bulk_material) and cannot name an explicit zone."
                )
            stripped = name.strip().lstrip("+-")
            if stripped.isdigit():
                raise ValueError(
                    f"bulk_materials[{k}]: zone name {name!r} is "
                    f"purely numeric and would be ambiguous with the "
                    f"1-based integer zone reference. Use a "
                    f"non-numeric name."
                )
            if name in names:
                raise ValueError(
                    f"bulk_materials[{k}]: duplicate zone name "
                    f"{name!r} — names used as staging references must "
                    f"be unique."
                )
            norm.append((zone_poly, zone_mat))
            names.append(name)
        self.bulk_materials = norm
        self.zone_names = ["base"] + names

    @property
    def n_zones(self):
        r"""
        Number of bulk zones, including the implicit base zone.

        Returns
        -------
        int
            ``1 + len(bulk_materials)`` — index-aligned with
            ``mat_indices`` values and :attr:`zone_names`.
        """
        return 1 + len(self.bulk_materials)

    def zone_index(self, ref):
        r"""
        Resolve a bulk-zone reference to its zone index.

        The staging schema references a zone either by *name*
        (:attr:`zone_names`; ``'base'`` is zone 0) or by its **1-based
        integer position** in the ``bulk_materials`` list (``0`` is
        the base zone).  Whether a given zone may be the target of an
        operation (e.g. zone 0 is never activatable) is enforced by
        the operation, not here.

        Parameters
        ----------
        ref : str or int
            Zone name or zone index.

        Returns
        -------
        int
            Zone index in ``[0, n_zones)``.

        Raises
        ------
        ValueError
            Unknown name, index out of range, or unsupported type.
            Booleans are rejected explicitly (YAML ``true`` is a
            :class:`bool`, an :class:`int` subclass — accepting it as
            zone 1 would mask an input error).
        """
        if isinstance(ref, bool):
            raise ValueError(
                f"zone reference must be a zone name (str) or a zone "
                f"index (int), got the boolean {ref!r}."
            )
        if isinstance(ref, str):
            try:
                return self.zone_names.index(ref)
            except ValueError:
                raise ValueError(
                    f"unknown bulk zone name {ref!r}. Known zones: "
                    f"{self.zone_names}."
                ) from None
        if isinstance(ref, (int, np.integer)):
            zi = int(ref)
            if 0 <= zi < self.n_zones:
                return zi
            raise ValueError(
                f"bulk zone index {zi} out of range: the section has "
                f"{self.n_zones} zone(s) (0 = 'base', 1..N = "
                f"material_zones order)."
            )
        raise ValueError(
            f"zone reference must be a zone name (str) or a zone "
            f"index (int), got {ref!r}."
        )

    def _material_index(self, x, y):
''',
    )

    # G5 — staging parents for tendons (populated branch).
    pf.edit(
        "G5 _setup_tendons staging parents",
        """            self.mat_indices_tendon = np.array(
                [self._material_index(t.x, t.y)
                 for t in self.tendons],
                dtype=int)
        else:
""",
        """            self.mat_indices_tendon = np.array(
                [self._material_index(t.x, t.y)
                 for t in self.tendons],
                dtype=int)
            self.staging_parent_tendon = \\
                self._resolve_tendon_parents()
        else:
""",
    )

    # G6 — staging parents empty branch.
    pf.edit(
        "G6 _setup_tendons empty parents",
        """            self.embedded_tendons = np.empty(0, dtype=bool)
            self.mat_indices_tendon = np.empty(0, dtype=int)
""",
        """            self.embedded_tendons = np.empty(0, dtype=bool)
            self.mat_indices_tendon = np.empty(0, dtype=int)
            self.staging_parent_tendon = np.empty(0, dtype=int)
""",
    )

    # G7 — parent-override resolver.
    pf.edit(
        "G7 _resolve_tendon_parents",
        """    # ------------------------------------------------------------------
    #  Geometric properties
    # ------------------------------------------------------------------
""",
        '''    def _resolve_tendon_parents(self):
        r"""
        Resolve each tendon's **staging parent** zone.

        The staging parent is the bulk zone whose activity gates the
        tendon in the per-stage containment invariant

        .. math::

            \\mathrm{active}[i] \\;\\Rightarrow\\;
            \\mathrm{bulk\\_active}[\\,\\mathrm{parent}(i)\\,]

        enforced by
        :meth:`~gensec.solver.section_state.StagedDomainManager.resolve_stages`.
        By default it is the geometric containing zone
        (``mat_indices_tendon``).  A tendon may override it via
        :attr:`~gensec.geometry.fiber.Tendon.parent` **only** when it
        is not embedded: an embedded tendon physically displaces the
        zone that contains it, and the displaced-bulk subtraction —
        which always uses the geometric zone — would contradict a
        different staging parent.  The override exists for
        non-embedded elements whose structural anchorage belongs to a
        zone other than the one their coordinates happen to fall in
        (e.g. an external tendon routed across a void).

        Returns
        -------
        numpy.ndarray of int
            Per-tendon staging-parent zone index.

        Raises
        ------
        ValueError
            ``parent`` set on an embedded tendon (the message carries
            the coordinates and both zones), or an unresolvable zone
            reference (propagated from :meth:`zone_index`).
        """
        parents = self.mat_indices_tendon.copy()
        for j, t in enumerate(self.tendons):
            override = getattr(t, "parent", None)
            if override is None:
                continue
            zi = self.zone_index(override)
            if t.embedded:
                geo = int(parents[j])
                raise ValueError(
                    f"Tendon {j} ('{t.name}') at "
                    f"(x={self.x_tendons[j]:.1f}, "
                    f"y={self.y_tendons[j]:.1f}): staging 'parent' "
                    f"override ({override!r} -> zone "
                    f"'{self.zone_names[zi]}') is legal only with "
                    f"embedded=False. An embedded tendon displaces "
                    f"the zone that geometrically contains it (zone "
                    f"'{self.zone_names[geo]}'), and its staging "
                    f"parent must coincide with it."
                )
            parents[j] = zi
        return parents

    # ------------------------------------------------------------------
    #  Geometric properties
    # ------------------------------------------------------------------
''',
    )


def edits_fiber(pf: PatchFile):
    r"""``gensec/geometry/fiber.py`` — retire ``system``, add ``parent``."""

    # F1 — typing import (Union for the parent reference type).
    pf.edit(
        "F1 typing import",
        "from typing import Optional\n",
        "from typing import Optional, Union\n",
    )

    # F2 — docstring: replace the ``system`` block with ``parent``.
    pf.edit(
        "F2 Tendon docstring system->parent",
        """    system : str, optional
        Construction-system tag (``'pre'`` / ``'post'``).  **Stored only,
        no behavioural effect.**  Retained for I/O round-tripping; per
        the element-vs-load taxonomy the modeling behaviour is fixed by
        ``bonded`` and by the element's placement, so this field is
        redundant and slated for removal — do not branch on it.
""",
        """    parent : int or str or None, optional
        **Staging-parent zone override** (Phase 8).  Default ``None``:
        the staging parent is the bulk zone that geometrically
        contains the tendon (``mat_indices_tendon``), which gates the
        tendon in the per-stage containment invariant
        :math:`\\mathrm{active}[i] \\Rightarrow
        \\mathrm{bulk\\_active}[\\mathrm{parent}(i)]`.  A zone name or
        1-based zone index overrides the *staging* parent only — the
        displaced-bulk subtraction keeps using the geometric zone —
        and is legal **only** with ``embedded=False`` (enforced at
        section assembly, where the zone map exists).  The former
        ``system`` tag (``'pre'``/``'post'``) is retired: the
        construction system is derived from the staging timeline
        (ordering of stressing vs casting events), never declared.
""",
    )

    # F3 — field swap.
    pf.edit(
        "F3 Tendon field system->parent",
        """    embedded: bool = True
    bonded: bool = True
    system: str = "pre"
    n_strands: int = 1
""",
        """    embedded: bool = True
    bonded: bool = True
    parent: Optional[Union[int, str]] = None
    n_strands: int = 1
""",
    )


def edits_io_yaml(pf: PatchFile):
    r"""``gensec/io_yaml.py`` — strict zones, ``activate_bulk``, migration."""

    # Y1 — strict material_zones parser with name support.  Closes the
    # silent-ignore gap (B8): a misspelled key must never be dropped.
    pf.edit(
        "Y1 material_zones strict parser",
        """        # Optional multi-material zones
        bulk_materials = []
        for zone_spec in sec_spec.get("material_zones", []):
            zone_poly = _SHAPE_FACTORIES[
                zone_spec["shape"].lower()](zone_spec.get("params", {}))
            zone_mat_name = zone_spec["material"]
            if zone_mat_name not in materials:
                raise ValueError(
                    f"Zone material '{zone_mat_name}' not found."
                )
            bulk_materials.append((zone_poly, materials[zone_mat_name]))
""",
        """        # Optional multi-material zones.  Key validation is
        # strict: an unknown key (a typo) must raise, never be
        # silently ignored — it would change the model without
        # telling (fail-loud policy, Phase-8 gap closure).
        bulk_materials = []
        _zone_keys = ("shape", "params", "material", "name")
        for iz, zone_spec in enumerate(
                sec_spec.get("material_zones", [])):
            unknown = sorted(set(zone_spec) - set(_zone_keys))
            if unknown:
                raise ValueError(
                    f"section.material_zones[{iz}]: unknown key(s) "
                    f"{unknown}. Valid: {list(_zone_keys)}."
                )
            for req in ("shape", "material"):
                if req not in zone_spec:
                    raise ValueError(
                        f"section.material_zones[{iz}]: missing "
                        f"required key '{req}'."
                    )
            zone_poly = _SHAPE_FACTORIES[
                zone_spec["shape"].lower()](zone_spec.get("params", {}))
            zone_mat_name = zone_spec["material"]
            if zone_mat_name not in materials:
                raise ValueError(
                    f"Zone material '{zone_mat_name}' not found."
                )
            # 3-tuple (Polygon, Material, name).  name=None gets the
            # positional auto-name zone_<k> at section construction
            # (GenericSection._normalize_bulk_zones), which also
            # enforces uniqueness and the reserved/numeric-name rules.
            bulk_materials.append((zone_poly,
                                   materials[zone_mat_name],
                                   zone_spec.get("name")))
""",
    )

    # Y2 — tendon parser: reject the retired ``system`` key with a
    # migration message; parse the new ``parent`` override.
    pf.edit(
        "Y2 tendon parser system->parent",
        """    tendons = []
    for t_spec in sec_spec.get("tendons", []):
        mat_name = t_spec["material"]
        if mat_name not in materials:
            raise ValueError(
                f"Tendon material '{mat_name}' not found in materials."
            )
        tendons.append(Tendon(
            y=float(t_spec["y"]),
            material=materials[mat_name],
            Ap=float(t_spec.get("Ap", 0)),
            eps_pe=float(t_spec.get("eps_pe", 0.0)),
            x=float(t_spec["x"]) if "x" in t_spec else None,
            system=t_spec.get("system", "pre"),
            bonded=bool(t_spec.get("bonded", True)),
""",
        """    tendons = []
    for t_spec in sec_spec.get("tendons", []):
        if "system" in t_spec:
            raise ValueError(
                f"Tendon spec (y={t_spec.get('y')!r}, "
                f"name={t_spec.get('name')!r}): the 'system' key is "
                f"retired. Pre-/post-tensioning is derived from the "
                f"staging timeline (ordering of stressing vs casting "
                f"events), never declared per tendon — remove the "
                f"key."
            )
        mat_name = t_spec["material"]
        if mat_name not in materials:
            raise ValueError(
                f"Tendon material '{mat_name}' not found in materials."
            )
        tendons.append(Tendon(
            y=float(t_spec["y"]),
            material=materials[mat_name],
            Ap=float(t_spec.get("Ap", 0)),
            eps_pe=float(t_spec.get("eps_pe", 0.0)),
            x=float(t_spec["x"]) if "x" in t_spec else None,
            parent=t_spec.get("parent"),
            bonded=bool(t_spec.get("bonded", True)),
""",
    )

    # Y3 — tendon parser docstring: drop the ``system`` line from the
    # YAML example, document ``parent``.
    pf.edit(
        "Y3 tendon docstring example",
        """        tendons:
          - y: 80
            x: 200
            material: ps_1
            Ap: 1400
            eps_pe: 0.0065
            system: post
            bonded: true
            name: T_bottom     # optional; usable as prestress_actions ref
""",
        """        tendons:
          - y: 80
            x: 200
            material: ps_1
            Ap: 1400
            eps_pe: 0.0065
            bonded: true
            name: T_bottom     # optional; usable as prestress_actions ref
            # parent: <zone>   # staging-parent override; legal only
            #                  # with embedded: false (Phase 8)

    The retired ``system`` key ('pre'/'post') raises with a migration
    message: the construction system is derived from the staging
    timeline, never declared per tendon.
""",
    )

    # Y4 — section_ops key set: activate_bulk / deactivate_bulk.
    pf.edit(
        "Y4 _SECTION_OPS_KEYS",
        """_SECTION_OPS_KEYS = ("activate", "deactivate", "eps_override",
                     "bulk_eps", "release")
""",
        """_SECTION_OPS_KEYS = ("activate", "deactivate", "eps_override",
                     "bulk_eps", "release", "activate_bulk",
                     "deactivate_bulk")
""",
    )

    # Y5 — value-level validation of activate_bulk (parse time, no
    # section needed); deactivate_bulk reserved (demolition, D2).
    pf.edit(
        "Y5 _parse_section_ops_spec activate_bulk",
        """    if "bulk_eps" in ops:
        out["bulk_eps"] = float(ops["bulk_eps"])
    return out
""",
        """    if "bulk_eps" in ops:
        out["bulk_eps"] = float(ops["bulk_eps"])
    if "deactivate_bulk" in ops:
        raise NotImplementedError(
            f"{where}: bulk deactivation not yet supported. "
            f"Demolition requires the released-stress resultant of a "
            f"bulk region (the bulk analog of deactivation_actions) — "
            f"deferred beyond the prestress arc."
        )
    if "activate_bulk" in ops:
        val = ops["activate_bulk"]
        # Casting event: activate a bulk zone with its mandatory
        # locked-in datum plane (eps0, chi_x, chi_y).  The datum is
        # mandatory-explicit at engine level: writing zeros is legal,
        # omitting a component is not (a defaulted datum would be a
        # silent reconciliation — the with_grouted failure mode).
        if not isinstance(val, dict) or not val:
            raise ValueError(
                f"{where}: section_ops 'activate_bulk' must be a "
                f"non-empty mapping "
                f"{{zone_ref: {{eps0, chi_x, chi_y}}}}, got {val!r}."
            )
        out_ab = {}
        for zref, datum in val.items():
            zwhere = f"{where}: activate_bulk[{zref!r}]"
            if not isinstance(datum, dict):
                raise ValueError(
                    f"{zwhere}: datum must be a mapping with the "
                    f"three keys eps0, chi_x, chi_y, got "
                    f"{type(datum).__name__}."
                )
            unknown_d = sorted(set(datum)
                               - {"eps0", "chi_x", "chi_y"})
            if unknown_d:
                raise ValueError(
                    f"{zwhere}: unknown datum key(s) {unknown_d}. "
                    f"Valid: ['eps0', 'chi_x', 'chi_y']."
                )
            missing = [kk for kk in ("eps0", "chi_x", "chi_y")
                       if kk not in datum]
            if missing:
                raise ValueError(
                    f"{zwhere}: missing datum key(s) {missing}. The "
                    f"casting datum plane is mandatory-explicit at "
                    f"engine level; write zeros explicitly if that "
                    f"is the intent."
                )
            out_ab[zref] = {kk: float(datum[kk])
                            for kk in ("eps0", "chi_x", "chi_y")}
        out["activate_bulk"] = out_ab
    return out
""",
    )

    # Y6 — name-level resolution of zone refs.
    pf.edit(
        "Y6 _resolve_section_ops activate_bulk",
        """            if "release" in spec:
                ops["release"] = spec["release"]
            if "bulk_eps" in spec:
                ops["bulk_eps"] = spec["bulk_eps"]
            stage["section_ops"] = ops
""",
        """            if "release" in spec:
                ops["release"] = spec["release"]
            if "bulk_eps" in spec:
                ops["bulk_eps"] = spec["bulk_eps"]
            if "activate_bulk" in spec:
                ab = {}
                for zref, datum in spec["activate_bulk"].items():
                    zwhere = f"{where}.activate_bulk"
                    try:
                        zi = section.zone_index(zref)
                    except AttributeError:
                        raise ValueError(
                            f"{zwhere}: the section does not expose "
                            f"bulk zones (no zone_index); "
                            f"activate_bulk needs a GenericSection "
                            f"with material_zones."
                        ) from None
                    except ValueError as exc:
                        raise ValueError(f"{zwhere}: {exc}") from None
                    if zi == 0:
                        raise ValueError(
                            f"{zwhere}: zone 0 ('base') is always "
                            f"active and not activatable."
                        )
                    if zi in ab:
                        raise ValueError(
                            f"{zwhere}: zone {zref!r} resolves to "
                            f"zone index {zi}, already targeted in "
                            f"this stage (name/index double "
                            f"reference)."
                        )
                    ab[zi] = (datum["eps0"], datum["chi_x"],
                              datum["chi_y"])
                ops["activate_bulk"] = ab
            stage["section_ops"] = ops
""",
    )

    # Y7 — resolve_stages docstring pointer in the loader (the ops
    # documentation block of _parse_combination).
    pf.edit(
        "Y7 _parse_combination ops doc",
        """      ``activate`` / ``deactivate`` (lists of element references),
""",
        """      ``activate`` / ``deactivate`` (lists of element references),
      ``activate_bulk`` (``{zone_ref: {eps0, chi_x, chi_y}}`` — cast a
      bulk zone with its mandatory locked-in datum plane; Phase 8),
""",
    )


def edits_section_state(pf: PatchFile):
    r"""``gensec/solver/section_state.py`` — state fields, hash, view."""

    # S1 — SectionState docstring: new parameters.
    pf.edit(
        "S1 SectionState docstring",
        """    bulk_eps_init : float, optional
        Uniform bulk pre-strain [-] (e.g. shrinkage), 0 by default.
""",
        """    bulk_eps_init : float, optional
        Uniform bulk pre-strain [-] (e.g. shrinkage), 0 by default.
    bulk_active : numpy.ndarray of bool or None, optional
        Per-zone activity mask over ``n_zones = 1 +
        len(bulk_materials)`` (Phase 8 bulk staging).  Zone 0 (the
        base ``bulk_material``) is always active.  ``None`` (default,
        legacy direct construction) means "all zones active"; states
        built by :meth:`StagedDomainManager.initial_state` always
        carry the explicit array.  The per-fiber mask is always
        *derived*, never stored:
        ``fiber_active = bulk_active[mat_indices]``.
    bulk_planes : numpy.ndarray or None, optional
        Per-zone locked-in datum planes, shape ``(n_zones, 3)``:
        :math:`(\\varepsilon_{0,z}, \\chi_{x,z}, \\chi_{y,z})` per
        zone, evaluated with the sign convention of
        :meth:`~gensec.solver.integrator.FiberSolver.strain_field`
        about the solver reference point (the full-polygon centroid,
        pinned across stages).  The casting datum of a staged zone:
        the zone is stress-free on the plane :math:`-\\,
        \\mathrm{plane}_z` (linear-equivalence identity, master plan
        §3).  The legacy scalar ``bulk_eps_init`` is *not* folded in
        here — it remains a separate uniform term added to every
        active zone by the integrator's offset field (one internal
        mechanism, two inputs; ``with_bulk_eps`` keeps working
        unchanged).
""",
    )

    # S2 — dataclass fields.
    pf.edit(
        "S2 SectionState fields",
        """    stage_index: int
    active: np.ndarray
    bonded: np.ndarray
    eps_init: np.ndarray
    bulk_eps_init: float = 0.0
    time_days: float = 0.0
    label: str = ""
""",
        """    stage_index: int
    active: np.ndarray
    bonded: np.ndarray
    eps_init: np.ndarray
    bulk_eps_init: float = 0.0
    bulk_active: Optional[np.ndarray] = None
    bulk_planes: Optional[np.ndarray] = None
    time_days: float = 0.0
    label: str = ""
""",
    )

    # S3 — capacity_hash: signature, terms, docstring.
    pf.edit(
        "S3 capacity_hash",
        """    def capacity_hash(self, geom_sig: Tuple[Any, ...],
                      union_materials: List[int]) -> int:
        r\"\"\"
        Capacity state hash: identity of the resistance domain.

        Composed of

        1. the fixed geometry/mesh signature *geom_sig*;
        2. for every **active and bonded** element, in ascending
           index order, the triple
           ``(material_id, quantize(eps_init), bonded)``;
        3. the quantized bulk pre-strain.

        Active-but-unbonded elements are excluded (they are not in the
        domain).  Applied loads / ``PrestressAction`` are excluded by
        construction — they never reach this method.

        Parameters
        ----------
        geom_sig : tuple
            Output of :func:`geometry_signature` for the base section.
        union_materials : list of int
            ``id(material)`` for each element in the union set, in the
            canonical ``rebars + tendons`` order.

        Returns
        -------
        int
            Python hash of the canonical state tuple.
        \"\"\"
        elem_terms = []
        idx = np.nonzero(self.active & self.bonded)[0]
        for i in idx:
            elem_terms.append(
                (union_materials[int(i)],
                 _quantize(float(self.eps_init[int(i)])),
                 True)
            )
        return hash((
            geom_sig,
            tuple(elem_terms),
            _quantize(float(self.bulk_eps_init)),
        ))
""",
        """    def capacity_hash(self, geom_sig: Tuple[Any, ...],
                      union_materials: List[int],
                      chi_quantum: float = QUANT_EPS) -> int:
        r\"\"\"
        Capacity state hash: identity of the resistance domain.

        Composed of

        1. the fixed geometry/mesh signature *geom_sig*;
        2. for every **active and bonded** element, in ascending
           index order, the triple
           ``(material_id, quantize(eps_init), bonded)``;
        3. the quantized bulk pre-strain;
        4. (Phase 8, when :attr:`bulk_active` is set) the byte hash of
           the zone activity mask, and — per **active** zone in index
           order — the quantized locked-in plane triple

           .. math::

               \\bigl(\\,q(\\varepsilon_{0,z}),\\;
               q_\\chi(\\chi_{x,z}),\\; q_\\chi(\\chi_{y,z})\\,\\bigr)

           with the curvature quantum
           :math:`q_\\chi = \\texttt{QUANT\\_EPS} / D`,
           :math:`D = \\max(H, B)` of the base section, so the
           bucketing error on the extreme-fiber strain
           :math:`\\chi \\cdot D` stays :math:`\\le` ``QUANT_EPS`` —
           coherent with the documented ``QUANT_EPS`` trap.

        Active-but-unbonded elements are excluded (they are not in the
        domain).  Applied loads / ``PrestressAction`` are excluded by
        construction — they never reach this method.  States without
        zone arrays (``bulk_active is None``, legacy direct
        construction) hash exactly as before Phase 8; a manager mixes
        the two forms only in the *safe* direction (a missed cache
        reuse, never a wrong one).

        Parameters
        ----------
        geom_sig : tuple
            Output of :func:`geometry_signature` for the base section.
        union_materials : list of int
            ``id(material)`` for each element in the union set, in the
            canonical ``rebars + tendons`` order.
        chi_quantum : float, optional
            Curvature bucket width [1/mm].  Deterministic per manager:
            :class:`StagedDomainManager` computes it once from the
            base section and passes it down.  Default
            :data:`QUANT_EPS` (dimensionally a fallback for direct
            callers only).

        Returns
        -------
        int
            Python hash of the canonical state tuple.
        \"\"\"
        elem_terms = []
        idx = np.nonzero(self.active & self.bonded)[0]
        for i in idx:
            elem_terms.append(
                (union_materials[int(i)],
                 _quantize(float(self.eps_init[int(i)])),
                 True)
            )
        terms = [
            geom_sig,
            tuple(elem_terms),
            _quantize(float(self.bulk_eps_init)),
        ]
        if self.bulk_active is not None:
            ba = np.ascontiguousarray(self.bulk_active, dtype=bool)
            terms.append(hash(ba.tobytes()))
            if self.bulk_planes is None:
                planes = np.zeros((ba.size, 3), dtype=float)
            else:
                planes = np.asarray(self.bulk_planes, dtype=float)
            zone_terms = []
            for z in np.nonzero(ba)[0]:
                e0, cx, cy = planes[int(z)]
                zone_terms.append((
                    _quantize(float(e0)),
                    _quantize(float(cx), chi_quantum),
                    _quantize(float(cy), chi_quantum),
                ))
            terms.append(tuple(zone_terms))
        return hash(tuple(terms))
""",
    )

    # S4 — copy_advanced propagates the zone arrays.
    pf.edit(
        "S4 copy_advanced",
        """        return SectionState(
            stage_index=stage_index,
            active=self.active.copy(),
            bonded=self.bonded.copy(),
            eps_init=self.eps_init.copy(),
            bulk_eps_init=self.bulk_eps_init,
            time_days=self.time_days,
            label=label or self.label,
        )
""",
        """        return SectionState(
            stage_index=stage_index,
            active=self.active.copy(),
            bonded=self.bonded.copy(),
            eps_init=self.eps_init.copy(),
            bulk_eps_init=self.bulk_eps_init,
            bulk_active=(None if self.bulk_active is None
                         else self.bulk_active.copy()),
            bulk_planes=(None if self.bulk_planes is None
                         else self.bulk_planes.copy()),
            time_days=self.time_days,
            label=label or self.label,
        )
""",
    )

    # S5 — with_bulk_activated, after with_grouted (anchor: the end of
    # with_grouted plus the PrestressAction section header).
    pf.edit(
        "S5 with_bulk_activated",
        """        s = self.copy_advanced(self.stage_index, self.label)
        s.active[idx] = True
        s.bonded[idx] = True
        for i in idx:
            s.eps_init[int(i)] = float(eps_init_map[int(i)])
        return s

# ==================================================================
#  PrestressAction (Phase-3 interface; fleshed out in prestress v1)
# ==================================================================
""",
        '''        s = self.copy_advanced(self.stage_index, self.label)
        s.active[idx] = True
        s.bonded[idx] = True
        for i in idx:
            s.eps_init[int(i)] = float(eps_init_map[int(i)])
        return s

    def with_bulk_activated(self, zones, plane_map) -> "SectionState":
        r"""
        New state with the bulk *zones* **cast**: made active with
        their locked-in datum planes set, **atomically**.

        The bulk analog of :meth:`with_grouted` (same single-side
        invariant): a zone enters the resistance domain only together
        with an **explicit** casting datum plane.  Casting a zone
        while leaving whatever plane the array happened to hold would
        be a *silent* reconciliation — precisely the failure mode the
        prestress driver forbids for tendons.  A missing entry
        therefore raises rather than defaulting;
        :math:`(0, 0, 0)` is legal but must be written.

        The datum plane :math:`(\\varepsilon_{0,z}, \\chi_{x,z},
        \\chi_{y,z})` is expressed about the solver reference point
        (full-polygon centroid) with the sign convention of
        :meth:`~gensec.solver.integrator.FiberSolver.strain_field`.
        Physically: the zone is **stress-free** on the section strain
        plane :math:`-\\,\\mathrm{plane}_z`, so the datum of a zone
        cast on a deformed substrate is the *negated* substrate plane
        at casting (linear incremental ≡ one-shot equivalence, master
        plan §3).  Producing that plane automatically (``auto``) is
        the Task-2 timeline resolution walk; at engine level the datum
        is an input (demand purity of ``resolve_stages``).

        Parameters
        ----------
        zones : sequence of int
            Zone indices to activate (1-based zone list positions;
            zone 0 = ``'base'`` is always active and not activatable).
        plane_map : dict
            ``{zone_index: (eps0, chi_x, chi_y)}`` — the locked-in
            datum plane for **every** index in *zones*.

        Returns
        -------
        SectionState
            A new state; the hash changes (mask flip + plane terms),
            triggering an automatic domain rebuild.

        Raises
        ------
        KeyError
            If any zone in *zones* has no entry in *plane_map*
            (atomicity guard, ``with_grouted``-style).
        ValueError
            Zone 0 targeted; zone index out of range; state built
            without zone arrays; malformed or non-finite plane.
        """
        if self.bulk_active is None:
            raise ValueError(
                "with_bulk_activated: this state carries no zone "
                "arrays (bulk_active is None — legacy direct "
                "construction). Derive states from "
                "StagedDomainManager.initial_state(), which sizes "
                "bulk_active/bulk_planes on the section's zones."
            )
        zs = [int(z) for z in zones]
        missing = [z for z in zs if z not in
                   {int(k) for k in plane_map}]
        if missing:
            raise KeyError(
                f"with_bulk_activated: no locked-in datum plane for "
                f"zone(s) {missing}. Casting requires an explicit "
                f"(eps0, chi_x, chi_y) for every activated zone "
                f"(single-side invariant: a zone enters the "
                f"resistance domain with its casting datum, never a "
                f"stale array value; (0, 0, 0) is legal but must be "
                f"written)."
            )
        n_zones = int(self.bulk_active.size)
        s = self.copy_advanced(self.stage_index, self.label)
        if s.bulk_planes is None:
            s.bulk_planes = np.zeros((n_zones, 3), dtype=float)
        plane_by_int = {int(k): v for k, v in plane_map.items()}
        for z in zs:
            if z == 0:
                raise ValueError(
                    "with_bulk_activated: zone 0 ('base') is always "
                    "active and not activatable."
                )
            if not (0 < z < n_zones):
                raise ValueError(
                    f"with_bulk_activated: zone index {z} out of "
                    f"range (state has {n_zones} zone(s))."
                )
            plane = np.asarray(plane_by_int[z], dtype=float).ravel()
            if plane.size != 3 or not np.all(np.isfinite(plane)):
                raise ValueError(
                    f"with_bulk_activated: datum plane for zone {z} "
                    f"must be three finite floats "
                    f"(eps0, chi_x, chi_y), got "
                    f"{plane_by_int[z]!r}."
                )
            s.bulk_active[z] = True
            s.bulk_planes[z, :] = plane
        return s

# ==================================================================
#  PrestressAction (Phase-3 interface; fleshed out in prestress v1)
# ==================================================================
''',
    )

    # S6 — staging-parents helper, before materialize_view.
    pf.edit(
        "S6 _staging_parents helper",
        """def materialize_view(base_section, state: SectionState):
""",
        '''def _staging_parents(section) -> np.ndarray:
    r"""
    Per-union-element staging-parent zone indices.

    Concatenates, in the canonical ``rebars + tendons`` order, the
    geometric containing zones (``mat_indices_rebar`` /
    ``mat_indices_tendon``) with the tendon override already resolved
    by the section (:attr:`staging_parent_tendon` — legal only for
    non-embedded tendons, see
    :meth:`~gensec.geometry.geometry.GenericSection._resolve_tendon_parents`).
    Sections built before the zone machinery fall back to zone 0 for
    every element.

    Parameters
    ----------
    section : GenericSection

    Returns
    -------
    numpy.ndarray of int
        Shape ``(n_union,)``.
    """
    n_reb = int(getattr(section, "x_rebars", np.empty(0)).size)
    n_ten = int(getattr(section, "x_tendons", np.empty(0)).size)
    par_r = getattr(section, "mat_indices_rebar", None)
    if par_r is None:
        par_r = np.zeros(n_reb, dtype=int)
    par_t = getattr(section, "staging_parent_tendon", None)
    if par_t is None:
        par_t = getattr(section, "mat_indices_tendon", None)
        if par_t is None:
            par_t = np.zeros(n_ten, dtype=int)
    return np.concatenate([np.asarray(par_r, dtype=int),
                           np.asarray(par_t, dtype=int)])


def _apply_bulk_staging(base_section, state: SectionState, view):
    r"""
    Apply the Phase-8 bulk-staging state to a materialized *view*.

    Three regimes (master plan §1 / primer §2, with the fast-path
    condition corrected to *mask all-True **and** planes all-zero* —
    an all-active state may still carry non-zero locked-in planes at
    the final stage of a composite, and those must reach the solver):

    1. **Trivial** (``bulk_active`` is ``None``, or all-True with
       zero planes): return immediately — no attribute is set, the
       single-bulk pipeline is byte-identical by construction.
    2. **All-active, non-zero planes**: bulk arrays stay shared by
       reference; only ``view.bulk_planes_active`` is attached.
    3. **Masked**: re-slice the bulk fiber arrays to the active zones
       (mask-in-kernel was rejected as a correctness bug — masked
       fibers would still veto planes in ``strains_within_limits``),
       enforce the containment invariant on the kept point elements,
       override the stale geometry attributes from the **exact**
       active polygon (base polygon minus the union of inactive zone
       polygons), and attach the planes.

    The reference point is **pinned**: ``x_centroid`` / ``y_centroid``
    are properties of the *shared full polygon*, which this function
    never replaces — the demand path requires a constant moment
    reference across stages.  ``bbox`` reads the plain attribute
    ``_bounds``, so the per-instance override takes effect without
    touching the class.

    Parameters
    ----------
    base_section : GenericSection
    state : SectionState
    view : GenericSection
        The shallow copy being materialized (mutated in place).

    Raises
    ------
    ValueError
        Zone 0 inactive; a kept (active & bonded) point element whose
        staging parent zone is inactive; empty active bulk.
    """
    ba = state.bulk_active
    planes = state.bulk_planes
    has_planes = planes is not None and bool(
        np.any(np.asarray(planes, dtype=float)))
    masked = ba is not None and not bool(np.all(ba))
    if not masked and not has_planes:
        return                                    # regime 1: fast path

    if masked:                                    # regime 3
        if not bool(ba[0]):
            raise ValueError(
                "materialize_view: zone 0 ('base') is inactive. The "
                "base bulk zone is always active by contract; bulk "
                "deactivation (demolition) is not supported."
            )
        mi = getattr(base_section, "mat_indices", None)
        if mi is None:
            mi = np.zeros(int(base_section.n_fibers), dtype=int)
        mi = np.asarray(mi, dtype=int)
        keep = np.nonzero(ba[mi])[0]
        if keep.size == 0:
            raise ValueError(
                "materialize_view: the active bulk is empty (no fiber "
                "belongs to an active zone). A stage with no bulk "
                "material is meaningless."
            )
        view.x_fibers = base_section.x_fibers[keep]
        view.y_fibers = base_section.y_fibers[keep]
        view.A_fibers = base_section.A_fibers[keep]
        view.mat_indices = mi[keep]
        view.n_fibers = int(keep.size)

        # Containment invariant on the elements kept in this view
        # (resolve_stages enforces it for manager-built walks; this
        # protects direct materialize_view callers too).
        parents = _staging_parents(base_section)
        resist = state.active & state.bonded
        bad = np.nonzero(resist & ~ba[parents])[0]
        if bad.size:
            names = getattr(base_section, "zone_names", None)
            i0 = int(bad[0])
            z0 = int(parents[i0])
            zlab = (names[z0] if names and z0 < len(names)
                    else str(z0))
            raise ValueError(
                f"materialize_view: union element {i0} is active but "
                f"its staging-parent bulk zone '{zlab}' (index {z0}) "
                f"is inactive — an element cannot exist before the "
                f"zone that carries it is cast."
            )

        # Exact active geometry: base polygon minus inactive zones.
        from shapely.ops import unary_union
        inactive_polys = [
            base_section.bulk_materials[z - 1][0]
            for z in np.nonzero(~ba)[0] if z > 0
        ]
        active_poly = base_section.polygon.difference(
            unary_union(inactive_polys))
        if active_poly.is_empty:
            raise ValueError(
                "materialize_view: the active geometry is empty "
                "(inactive zones cover the whole section)."
            )
        minx, miny, maxx, maxy = active_poly.bounds
        view._bounds = (float(minx), float(miny),
                        float(maxx), float(maxy))
        view.B = float(maxx - minx)
        view.H = float(maxy - miny)
        view.ideal_gross_area = float(active_poly.area)

    # Regimes 2 and 3: the solver consumes the per-zone planes
    # through this attribute (index-aligned with mat_indices values,
    # which the re-slice above preserves).
    view.bulk_planes_active = (
        np.zeros((int(ba.size), 3), dtype=float) if planes is None
        else np.array(planes, dtype=float, copy=True))


def materialize_view(base_section, state: SectionState):
''',
    )

    # S7 — hook _apply_bulk_staging into materialize_view.
    pf.edit(
        "S7 materialize_view hook",
        """    view.bulk_eps_init = float(state.bulk_eps_init)
    view._ideal_gross_props_cache = None
    return view
""",
        """    view.bulk_eps_init = float(state.bulk_eps_init)

    # ---- Bulk staging (Phase 8, Task 1) ----
    # Re-slice / plane attachment / geometry overrides, or a strict
    # no-op on the trivial state (single-bulk byte-identity anchor).
    _apply_bulk_staging(base_section, state, view)

    view._ideal_gross_props_cache = None
    return view
""",
    )

    # S8 — manager __init__: curvature quantum, staging parents, zone
    # bookkeeping (deterministic per manager, passed down to the
    # hash).
    pf.edit(
        "S8 manager __init__",
        """        self._geom_sig = geometry_signature(base_section)
        self._union_materials = self._collect_union_materials(
            base_section)
        self._cache: Dict[int, DomainBundle] = {}
""",
        """        self._geom_sig = geometry_signature(base_section)
        self._union_materials = self._collect_union_materials(
            base_section)
        self._cache: Dict[int, DomainBundle] = {}

        # ---- Phase 8: bulk staging bookkeeping ----
        # Curvature quantum for the plane terms of the capacity hash:
        # QUANT_EPS / max(H, B), so the bucketing error on the
        # extreme-fiber strain chi * D stays <= QUANT_EPS.  Computed
        # once per manager (deterministic cache identity).
        _D = max(float(getattr(base_section, "H", 0.0)),
                 float(getattr(base_section, "B", 0.0)))
        self._chi_quantum = (QUANT_EPS / _D) if _D > 0 else QUANT_EPS
        self._n_zones = 1 + len(getattr(base_section,
                                        "bulk_materials", []) or [])
        self._staging_parents = _staging_parents(base_section)
        self._mat_indices = getattr(base_section, "mat_indices", None)
""",
    )

    # S9 — hash_of passes the manager's curvature quantum.
    pf.edit(
        "S9 hash_of",
        """        return state.capacity_hash(self._geom_sig,
                                   self._union_materials)
""",
        """        # The curvature quantum is a deterministic property of the
        # manager (set in __init__).  ``getattr`` keeps hash_of usable
        # on partially-built managers (a documented test idiom builds
        # them via ``__new__`` to hash states without ever building a
        # domain); the fallback is capacity_hash's own documented
        # default, and states without zone arrays never consume it.
        return state.capacity_hash(
            self._geom_sig,
            self._union_materials,
            chi_quantum=getattr(self, "_chi_quantum", QUANT_EPS))
""",
    )

    # S10 — initial_state builds the zone arrays.
    pf.edit(
        "S10 initial_state",
        """            bulk_eps_init=float(getattr(sec, "bulk_eps_init", 0.0)),
            label="stage0",
        )
""",
        """            bulk_eps_init=float(getattr(sec, "bulk_eps_init", 0.0)),
            # Phase 8: every zone active with zero locked-in planes —
            # the trivial state, byte-identical to the pre-staging
            # pipeline through the materialize_view fast path.
            bulk_active=np.ones(self._n_zones, dtype=bool),
            bulk_planes=np.zeros((self._n_zones, 3), dtype=float),
            label="stage0",
        )
""",
    )


def edits_resolve_stages(pf: PatchFile):
    r"""``resolve_stages`` — pre-scan, activate_bulk op, invariants."""

    # R1 — docstring: the activate_bulk / deactivate_bulk keys.
    pf.edit(
        "R1 resolve_stages docstring",
        """            indices), ``eps_override`` (``{idx: eps}``), ``bulk_eps``
            (float) and ``release`` (bool; whether deactivations are
            force-released vs cleanly removed).""",
        """            indices), ``eps_override`` (``{idx: eps}``), ``bulk_eps``
            (float), ``release`` (bool; whether deactivations are
            force-released vs cleanly removed) and — Phase 8 —
            ``activate_bulk``
            (``{zone_index: (eps0, chi_x, chi_y)}``: cast a bulk zone
            with its **mandatory** locked-in datum plane).  A zone
            targeted by any stage's ``activate_bulk`` starts
            **inactive** (activation-declarative pre-scan: casting a
            zone at stage *k* declares it not yet cast before *k*);
            re-activating an active zone raises.  The keyword-only
            ``initially_inactive`` (sequence of zone indices) marks
            zones as not-yet-cast **without** an ``activate_bulk``
            in this stage list — required by a timeline compiler
            emitting a prefix anchored *before* a zone's casting
            event (Task 2), where the pre-scan has nothing to see.
            ``deactivate_bulk`` (demolition) raises
            ``NotImplementedError`` — it needs the released-stress
            resultant of a bulk region, the bulk analog of
            :meth:`deactivation_actions`.

            Two invariants are enforced per stage (fail-loud):
            the **containment invariant**
            :math:`\\mathrm{active}[i] \\Rightarrow
            \\mathrm{bulk\\_active}[\\mathrm{parent}(i)]` for every
            union element (which subsumes "reject
            stage(tendon) < stage(bulk)" and protects API-built
            stages, not only YAML), and a **non-empty active bulk**
            (a stage whose active bulk holds no fiber is
            meaningless).""",
    )

    # R2 — loop body: pre-scan + op wiring + invariants.
    pf.edit(
        "R2 resolve_stages body",
        """    def resolve_stages(self, stages):
""",
        """    def resolve_stages(self, stages, *, initially_inactive=None):
""",
    )

    pf.edit(
        "R2b resolve_stages body",
        """        states, hashes, bundles, deact = [], [], [], []
        cur = self.initial_state()
        for k, stage in enumerate(stages):
            ops = stage.get("section_ops", {}) or {}
            deact_idx = list(ops.get("deactivate", []) or [])
            release = bool(ops.get("release", True))

            if ops.get("activate"):
                cur = cur.with_activated(ops["activate"])
""",
        """        states, hashes, bundles, deact = [], [], [], []
        cur = self.initial_state()

        # ---- Bulk-staging pre-scan (Phase 8) ---------------------
        # A zone activated at some stage is, by declaration, not yet
        # cast before it: collect every activate_bulk target and
        # start those zones inactive.  Activation-declarative — no
        # separate "initially inactive" list to keep in sync — and a
        # double activation becomes a hard error below.
        planned = set()
        for z in (initially_inactive or ()):
            zi = int(z)
            if not (1 <= zi < self._n_zones):
                raise ValueError(
                    f"resolve_stages: initially_inactive zone index "
                    f"{zi} out of range (section has "
                    f"{self._n_zones} zone(s); zone 0 = 'base' is "
                    f"always active)."
                )
            planned.add(zi)
        for stage in stages:
            ops = stage.get("section_ops", {}) or {}
            if "deactivate_bulk" in ops:
                raise NotImplementedError(
                    "bulk deactivation not yet supported (demolition "
                    "needs the released-stress resultant of a bulk "
                    "region, the bulk analog of deactivation_actions)."
                )
            for z in (ops.get("activate_bulk") or {}):
                zi = int(z)
                if not (1 <= zi < self._n_zones):
                    raise ValueError(
                        f"resolve_stages: activate_bulk zone index "
                        f"{zi} out of range (section has "
                        f"{self._n_zones} zone(s); zone 0 = 'base' "
                        f"is always active and not activatable)."
                    )
                planned.add(zi)
        if planned:
            cur.bulk_active[sorted(planned)] = False

        for k, stage in enumerate(stages):
            ops = stage.get("section_ops", {}) or {}
            deact_idx = list(ops.get("deactivate", []) or [])
            release = bool(ops.get("release", True))

            if ops.get("activate_bulk"):
                ab = ops["activate_bulk"]
                already = [int(z) for z in ab
                           if cur.bulk_active[int(z)]]
                if already:
                    raise ValueError(
                        f"resolve_stages: stage {k} "
                        f"('{stage.get('name', '')}') re-activates "
                        f"already-active bulk zone(s) {already}. A "
                        f"zone is cast exactly once; a second "
                        f"activation would silently overwrite its "
                        f"locked-in datum plane."
                    )
                cur = cur.with_bulk_activated(
                    [int(z) for z in ab],
                    {int(z): tuple(p) for z, p in ab.items()})
            if ops.get("activate"):
                cur = cur.with_activated(ops["activate"])
""",
    )

    # R3 — per-stage invariants, before the bundle build.
    pf.edit(
        "R3 resolve_stages invariants",
        """            if "time" in stage:
                cur.time_days = float(stage["time"])
            cur.stage_index = k

            h, bundle, _built = self.get_bundle(cur)
""",
        """            if "time" in stage:
                cur.time_days = float(stage["time"])
            cur.stage_index = k

            # ---- Containment invariant (Phase 8) -----------------
            # active[i] => bulk_active[parent(i)] for every union
            # element: nothing can be anchored in a zone that is not
            # yet cast.  Checked after all of the stage's ops, so the
            # order of ops within a stage is immaterial.
            if cur.bulk_active is not None:
                bad = np.nonzero(
                    cur.active
                    & ~cur.bulk_active[self._staging_parents])[0]
                if bad.size:
                    names = getattr(self.base_section,
                                    "zone_names", None)
                    i0 = int(bad[0])
                    z0 = int(self._staging_parents[i0])
                    zlab = (names[z0] if names and z0 < len(names)
                            else str(z0))
                    raise ValueError(
                        f"resolve_stages: stage {k} "
                        f"('{stage.get('name', '')}'): union element "
                        f"{i0} is active but its staging-parent bulk "
                        f"zone '{zlab}' (index {z0}) is inactive. "
                        f"Deactivate the element until the zone is "
                        f"cast (activate_bulk), then activate/grout "
                        f"it."
                    )
                # ---- Non-empty active bulk -----------------------
                if (self._mat_indices is not None
                        and not bool(np.any(
                            cur.bulk_active[self._mat_indices]))):
                    raise ValueError(
                        f"resolve_stages: stage {k} "
                        f"('{stage.get('name', '')}'): the active "
                        f"bulk is empty (no fiber belongs to an "
                        f"active zone). A stage with no bulk "
                        f"material is meaningless."
                    )

            h, bundle, _built = self.get_bundle(cur)
""",
    )


def edits_integrator(pf: PatchFile):
    r"""``gensec/solver/integrator.py`` — per-fiber offset field (B6)."""

    # I1 — __init__: retire the per-group scalar list, build the
    # per-fiber / per-element offset fields.
    pf.edit(
        "I1 __init__ offset fields",
        """        self._bulk_eps_init = float(
            getattr(self.sec, 'bulk_eps_init', 0.0))
        # Per-group offsets, index-aligned with self._bulk_groups.
        # Today every group carries the single section-level value;
        # differential shrinkage per zone will populate distinct
        # values here WITHOUT touching the kernel sites again.
        self._bulk_eps_by_group = [self._bulk_eps_init
                                   for _ in self._bulk_groups]
""",
        """        self._bulk_eps_init = float(
            getattr(self.sec, 'bulk_eps_init', 0.0))
        # Per-fiber / per-element imposed-strain offset fields
        # (Phase 8): the uniform scalar above plus the per-zone
        # locked-in datum planes carried by the view
        # (``bulk_planes_active``).  ``None`` on the fast path — the
        # consumption sites then add the plain scalar, keeping the
        # legacy numeric stream unchanged (never ``eb + zeros(n)``).
        self._build_offset_fields()
""",
    )

    # I2 — offset-field builder + accessors, before the
    # single-configuration section header.
    pf.edit(
        "I2 offset helpers",
        """    # ==================================================================
    #  Single-configuration integration
    # ==================================================================
""",
        '''    def _build_offset_fields(self):
        r"""
        Precompute the imposed-strain offset fields (Phase 8).

        The bulk constitutive law of fiber *i* (zone
        :math:`z(i) = \\texttt{mat\\_indices}[i]`) is evaluated at the
        offset argument
        :math:`\\varepsilon_{\\mathrm{sec},i} +
        \\varepsilon^{\\mathrm{off}}_i` with

        .. math::

           \\varepsilon^{\\mathrm{off}}_i \\;=\\; \\varepsilon_{b,0}
           \\; + \\; \\varepsilon_{0,z(i)}
           \\; + \\; \\chi_{x,z(i)}\\,(y_i - y_{\\mathrm{ref}})
           \\; - \\; \\chi_{y,z(i)}\\,(x_i - x_{\\mathrm{ref}})

        — the exact sign convention of :meth:`strain_field` — where
        :math:`\\varepsilon_{b,0}` is the legacy uniform scalar and
        :math:`(\\varepsilon_{0,z}, \\chi_{x,z}, \\chi_{y,z})` the
        per-zone locked-in plane from the view attribute
        ``bulk_planes_active`` (attached by
        :func:`~gensec.solver.section_state.materialize_view`).
        Companion fields are evaluated at the rebar and tendon
        coordinates for the displaced-bulk subtraction: the displaced
        concrete of zone *z* carries zone *z*'s plane at the element
        point, with the zone given by the **geometric** containment
        (``mat_indices_rebar`` / ``mat_indices_tendon`` — a staging
        ``parent`` override never touches the physics).

        **Fast path**: with no planes (or all-zero planes) every field
        is ``None`` and the consumption sites add the plain scalar
        ``_bulk_eps_init`` — byte-identity with the pre-Phase-8
        pipeline by construction (the NumPy instruction stream is
        unchanged), not by IEEE luck.

        Raises
        ------
        ValueError
            ``bulk_planes_active`` present with a shape inconsistent
            with the section's zone indices.
        """
        planes = getattr(self.sec, 'bulk_planes_active', None)
        if planes is None or not np.any(
                np.asarray(planes, dtype=float)):
            self._bulk_eps_field = None
            self._rebar_eps_field = None
            self._tendon_eps_field = None
            return
        planes = np.asarray(planes, dtype=float)
        mi = getattr(self.sec, 'mat_indices', None)
        if mi is None:
            mi = np.zeros(int(self.sec.n_fibers), dtype=int)
        mi = np.asarray(mi, dtype=int)
        mir = getattr(self.sec, 'mat_indices_rebar', None)
        if mir is None:
            mir = np.zeros(self._ly_rebar.size, dtype=int)
        mir = np.asarray(mir, dtype=int)
        mit = getattr(self.sec, 'mat_indices_tendon', None)
        if mit is None:
            mit = np.zeros(self._ly_tendon.size, dtype=int)
        mit = np.asarray(mit, dtype=int)
        zmax = max([int(mi.max(initial=0)),
                    int(mir.max(initial=0)),
                    int(mit.max(initial=0))])
        if planes.ndim != 2 or planes.shape[1] != 3 \\
                or planes.shape[0] <= zmax:
            raise ValueError(
                f"bulk_planes_active has shape {planes.shape}; "
                f"expected (n_zones >= {zmax + 1}, 3) for this "
                f"section's zone indices."
            )
        eb0 = self._bulk_eps_init
        self._bulk_eps_field = (eb0 + planes[mi, 0]
                                + planes[mi, 1] * self._ly_bulk
                                - planes[mi, 2] * self._lx_bulk)
        self._rebar_eps_field = (eb0 + planes[mir, 0]
                                 + planes[mir, 1] * self._ly_rebar
                                 - planes[mir, 2] * self._lx_rebar)
        self._tendon_eps_field = (eb0 + planes[mit, 0]
                                  + planes[mit, 1] * self._ly_tendon
                                  - planes[mit, 2] * self._lx_tendon)

    def _bulk_offset(self, idx=None):
        r"""
        Imposed-strain offset at bulk fibers.

        Parameters
        ----------
        idx : numpy.ndarray or None, optional
            Fiber subset; ``None`` for all fibers.

        Returns
        -------
        float or numpy.ndarray
            The scalar ``_bulk_eps_init`` on the fast path (fields
            ``None``), else the per-fiber field (sliced).  Either
            broadcasts transparently against 1-D and batch strain
            arrays at every consumption site.
        """
        if self._bulk_eps_field is None:
            return self._bulk_eps_init
        return (self._bulk_eps_field if idx is None
                else self._bulk_eps_field[idx])

    def _rebar_offset(self, idx):
        r"""
        Displaced-bulk offset at rebar locations (fiber-subset *idx*).

        Returns
        -------
        float or numpy.ndarray
        """
        if self._rebar_eps_field is None:
            return self._bulk_eps_init
        return self._rebar_eps_field[idx]

    def _tendon_offset(self, idx):
        r"""
        Displaced-bulk offset at tendon locations (subset *idx*).

        Returns
        -------
        float or numpy.ndarray
        """
        if self._tendon_eps_field is None:
            return self._bulk_eps_init
        return self._tendon_eps_field[idx]

    # ==================================================================
    #  Single-configuration integration
    # ==================================================================
''',
    )

    # I3 — _tendon_forces: batch displaced bulk.
    pf.edit(
        "I3 tendon batch displaced bulk",
        """                s_bulk = bulk_mat.stress_array(
                    e_sec_g + self._bulk_eps_init)
                s_net = s_tendon.copy()
                s_net[:, emb] -= s_bulk[:, emb]
""",
        """                s_bulk = bulk_mat.stress_array(
                    e_sec_g + self._tendon_offset(idx))
                s_net = s_tendon.copy()
                s_net[:, emb] -= s_bulk[:, emb]
""",
    )

    # I4 — _tendon_forces: scalar displaced bulk.
    pf.edit(
        "I4 tendon scalar displaced bulk",
        """            s_bulk = bulk_mat.stress_array(
                e_sec_g + self._bulk_eps_init)
            s_net = s_tendon.copy()
            s_net[emb] -= s_bulk[emb]
""",
        """            s_bulk = bulk_mat.stress_array(
                e_sec_g + self._tendon_offset(idx))
            s_net = s_tendon.copy()
            s_net[emb] -= s_bulk[emb]
""",
    )

    # I5 — _tendon_forces: tangent displaced bulk.
    pf.edit(
        "I5 tendon tangent displaced bulk",
        """                Et_bulk = bulk_mat.tangent_array(
                    e_sec_g + self._bulk_eps_init)
""",
        """                Et_bulk = bulk_mat.tangent_array(
                    e_sec_g + self._tendon_offset(idx))
""",
    )

    # I6 — integrate: bulk contribution.
    pf.edit(
        "I6 integrate bulk",
        """        if not self._is_multi_material:
            # Fast path: single material on all fibers
            sb = self.sec.bulk_material.stress_array(
                eb + self._bulk_eps_init)
        else:
            # Multi-material: evaluate each group at its own offset
            sb = np.zeros_like(eb)
            for (mat, idx), eps_b in zip(self._bulk_groups,
                                         self._bulk_eps_by_group):
                sb[idx] = mat.stress_array(eb[idx] + eps_b)
""",
        """        if not self._is_multi_material:
            # Fast path: single material on all fibers
            sb = self.sec.bulk_material.stress_array(
                eb + self._bulk_offset())
        else:
            # Multi-material: evaluate each group at its own offset
            sb = np.zeros_like(eb)
            for mat, idx in self._bulk_groups:
                sb[idx] = mat.stress_array(
                    eb[idx] + self._bulk_offset(idx))
""",
    )

    # I7 — integrate: rebar displaced bulk.
    pf.edit(
        "I7 integrate rebar displaced bulk",
        """            er_g = er[idx]
            s_rebar = mat.stress_array(er_g)
            sb_at_rebars = bulk_mat.stress_array(
                er_g + self._bulk_eps_init)
            a = self.sec.A_rebars[idx]
""",
        """            er_g = er[idx]
            s_rebar = mat.stress_array(er_g)
            sb_at_rebars = bulk_mat.stress_array(
                er_g + self._rebar_offset(idx))
            a = self.sec.A_rebars[idx]
""",
    )

    # I8 — integrate_with_tangent: bulk stress + tangent.
    pf.edit(
        "I8 tangent bulk",
        """        if not self._is_multi_material:
            sb = self.sec.bulk_material.stress_array(
                eb + self._bulk_eps_init)
            Et_b = self.sec.bulk_material.tangent_array(
                eb + self._bulk_eps_init)
        else:
            sb = np.zeros_like(eb)
            Et_b = np.zeros_like(eb)
            for (mat, idx), eps_b in zip(self._bulk_groups,
                                         self._bulk_eps_by_group):
                sb[idx] = mat.stress_array(eb[idx] + eps_b)
                Et_b[idx] = mat.tangent_array(eb[idx] + eps_b)
""",
        """        if not self._is_multi_material:
            sb = self.sec.bulk_material.stress_array(
                eb + self._bulk_offset())
            Et_b = self.sec.bulk_material.tangent_array(
                eb + self._bulk_offset())
        else:
            sb = np.zeros_like(eb)
            Et_b = np.zeros_like(eb)
            for mat, idx in self._bulk_groups:
                e_off = eb[idx] + self._bulk_offset(idx)
                sb[idx] = mat.stress_array(e_off)
                Et_b[idx] = mat.tangent_array(e_off)
""",
    )

    # I9 — integrate_with_tangent: rebar displaced bulk (stress +
    # tangent at the same offset argument).
    pf.edit(
        "I9 tangent rebar displaced bulk",
        """            sb_at_rebars = bulk_mat.stress_array(
                er_g + self._bulk_eps_init)
            Et_bulk_r = bulk_mat.tangent_array(
                er_g + self._bulk_eps_init)
""",
        """            sb_at_rebars = bulk_mat.stress_array(
                er_g + self._rebar_offset(idx))
            Et_bulk_r = bulk_mat.tangent_array(
                er_g + self._rebar_offset(idx))
""",
    )

    # I10 — integrate_batch: bulk (broadcast (n_fibers,) over the
    # (n, n_fibers) strain matrix).
    pf.edit(
        "I10 batch bulk",
        """        if not self._is_multi_material:
            sb = self.sec.bulk_material.stress_array(
                eb + self._bulk_eps_init)
        else:
            sb = np.zeros_like(eb)
            for (mat, idx), eps_b in zip(self._bulk_groups,
                                         self._bulk_eps_by_group):
                sb[:, idx] = mat.stress_array(eb[:, idx] + eps_b)
""",
        """        if not self._is_multi_material:
            sb = self.sec.bulk_material.stress_array(
                eb + self._bulk_offset())
        else:
            sb = np.zeros_like(eb)
            for mat, idx in self._bulk_groups:
                sb[:, idx] = mat.stress_array(
                    eb[:, idx] + self._bulk_offset(idx))
""",
    )

    # I11 — integrate_batch: rebar displaced bulk.
    pf.edit(
        "I11 batch rebar displaced bulk",
        """            er_g = er[:, idx]
            s_rebar = mat.stress_array(er_g)
            sb_at_rebars = bulk_mat.stress_array(
                er_g + self._bulk_eps_init)
""",
        """            er_g = er[:, idx]
            s_rebar = mat.stress_array(er_g)
            sb_at_rebars = bulk_mat.stress_array(
                er_g + self._rebar_offset(idx))
""",
    )

    # I12 — get_fiber_results: bulk.
    pf.edit(
        "I12 fiber_results bulk",
        """        # Bulk stresses
        if not self._is_multi_material:
            sb = self.sec.bulk_material.stress_array(
                eb + self._bulk_eps_init)
        else:
            sb = np.zeros_like(eb)
            for (mat, idx), eps_b in zip(self._bulk_groups,
                                         self._bulk_eps_by_group):
                sb[idx] = mat.stress_array(eb[idx] + eps_b)
""",
        """        # Bulk stresses
        if not self._is_multi_material:
            sb = self.sec.bulk_material.stress_array(
                eb + self._bulk_offset())
        else:
            sb = np.zeros_like(eb)
            for mat, idx in self._bulk_groups:
                sb[idx] = mat.stress_array(
                    eb[idx] + self._bulk_offset(idx))
""",
    )

    # I13 — get_fiber_results: rebar displaced bulk.
    pf.edit(
        "I13 fiber_results rebar displaced bulk",
        """            sr_ideal_gross[idx] = mat.stress_array(er_g)
            sb_at_rebars[idx] = bulk_mat.stress_array(
                er_g + self._bulk_eps_init)
""",
        """            sr_ideal_gross[idx] = mat.stress_array(er_g)
            sb_at_rebars[idx] = bulk_mat.stress_array(
                er_g + self._rebar_offset(idx))
""",
    )

    # I14 — get_fiber_results: tendon displaced bulk.
    pf.edit(
        "I14 fiber_results tendon displaced bulk",
        """                sb_at_tendons[idx] = bulk_mat.stress_array(
                    e_sec_g + self._bulk_eps_init)
""",
        """                sb_at_tendons[idx] = bulk_mat.stress_array(
                    e_sec_g + self._tendon_offset(idx))
""",
    )

    # I15 — strains_within_limits: bulk group loop (limits apply to
    # the offset argument of the constitutive law, per zone).
    pf.edit(
        "I15 strains_within_limits",
        """        for (mat, idx), eps_b in zip(self._bulk_groups,
                                     self._bulk_eps_by_group):
            e_group = eb[idx] + eps_b
""",
        """        for mat, idx in self._bulk_groups:
            e_group = eb[idx] + self._bulk_offset(idx)
""",
    )


def edits_sls(pf: PatchFile):
    r"""``gensec/solver/sls.py`` — fail-loud bulk-staging guard.

    Deviation from the primer's "zero expected edits" prediction,
    documented in the Task-1 recap: without a guard, a bulk-staged
    state entering :func:`verify_sls_staged` fails with an obscure
    accumulator shape mismatch (``S_bulk`` is sized on the base
    fibers, the per-stage views on the masked fibers) — or, worse for
    plane-only transitions, is silently misattributed by the
    ``eps_changed`` classifier, which knows nothing of per-zone
    planes.  The composite SLS basis is exactly fork C4 (Task 3).
    """
    pf.edit(
        "L1 verify_sls_staged bulk-staging guard",
        """        resist = _resist_mask(state, n_union)
""",
        """        # ---- Phase-8 bulk staging: deferred to Task 3 (C4) ----
        # The staged fiber accumulation below is sized on the base
        # fiber set and its transition taxonomy predates per-zone
        # locked-in planes; accepting a bulk-staged state here would
        # either crash on an accumulator shape mismatch or silently
        # misattribute a plane change.  Fail loud until the composite
        # SLS basis lands (decision fork C4).
        _ba = getattr(state, "bulk_active", None)
        if _ba is not None and not bool(np.all(_ba)):
            raise NotImplementedError(
                f"Stage '{name}': SLS verification across bulk "
                f"staging (inactive zones) is not yet supported — "
                f"composite SLS basis deferred (fork C4, Task 3)."
            )
        _bp = getattr(state, "bulk_planes", None)
        if _bp is not None and bool(
                np.any(np.asarray(_bp, dtype=float))):
            raise NotImplementedError(
                f"Stage '{name}': per-zone locked-in datum planes in "
                f"the SLS staged walk are not yet supported — the "
                f"stage-attribution taxonomy predates them (fork C4, "
                f"Task 3)."
            )
        resist = _resist_mask(state, n_union)
""",
    )


def edits_yaml_prestress(pf: PatchFile):
    r"""``examples/example_prestress.yaml`` — retired ``system`` key."""
    pf.edit(
        "X1 example_prestress system removal",
        """      eps_pe: 0.0065  # effective prestrain after losses (tension +)
      system: post    # 'pre' or 'post' (stored; no effect in Phase 1)
      bonded: true    # Phase 1 supports bonded only
""",
        """      eps_pe: 0.0065  # effective prestrain after losses (tension +)
      bonded: true    # Phase 1 supports bonded only
""",
    )


def edits_yaml_reference(pf: PatchFile):
    r"""``examples/yaml_reference_example.yaml`` — ``system`` comments."""
    pf.edit(
        "X2 yaml_reference system comment",
        """  # system: 'pre'/'post' -- stored only, no behavioural effect.
""",
        """  # parent: staging-parent zone override (name or 1-based zone
  #   index); legal only with embedded: false.  The former 'system'
  #   key ('pre'/'post') is retired: the construction system is
  #   derived from the staging timeline, never declared per tendon.
""",
    )
    pf.edit(
        "X4 yaml_reference stale prestrain guard comment",
        """  # PHASE-5 GUARD: the field is parsed, hashed and propagated, but
  # the fiber kernel does not consume it yet -- a NON-ZERO value
  # RAISES (no-silent-no-op) until the kernel change lands.
""",
        """  # Since the Phase-5 bulk-kernel patch the fiber kernel CONSUMES
  # this offset (bulk law evaluated at eps_sec + prestrain); as of
  # Phase 8 it is one term of the per-fiber offset field, on top of
  # the per-zone locked-in datum planes of staged casting.
""",
    )
    pf.edit(
        "X3 yaml_reference system in commented example",
        """  #     eps_pe: 0.00715       # ~= 1395 MPa / 195000 MPa
  #     system: pre
  #     bonded: true          # (default; false raises by design)
""",
        """  #     eps_pe: 0.00715       # ~= 1395 MPa / 195000 MPa
  #     bonded: true          # (default; false raises by design)
""",
    )


#: New shipped example: engine-level composite staged casting.
_COMPOSITE_EXAMPLE = """\
# ---------------------------------------------------------------------------
# GenSec example -- composite section, engine-level bulk staging (Phase 8).
#
# A precast web (cast first) receives a topping (cast later) on the
# already-deformed substrate.  The topping zone is *activated* at its
# casting stage with an explicit locked-in datum plane: the negated
# substrate strain plane at casting, so the topping is stress-free at
# the instant it is cast (linear incremental == one-shot equivalence).
#
# Engine-level schema: the datum plane (eps0, chi_x, chi_y) is
# mandatory-explicit -- zeros are legal but must be written.  The
# timeline-owned automatic datum ('auto') is the Task-2 resolution
# walk.  Zone references: 'base' (zone 0, always active) or the zone
# 'name' declared in material_zones (or its 1-based position).
# ---------------------------------------------------------------------------

materials:
  c_precast: {type: concrete_ec2_gen1, class: 'C45/55'}
  c_topping: {type: concrete_ec2_gen1, class: 'C30/37'}
  steel:     {type: steel, fyk: 450, gamma_s: 1.15}

section:
  shape: rect
  params: {B: 600, H: 1400}       # composite envelope
  bulk_material: c_precast         # zone 0 ('base'): the precast web
  mesh_size: 50
  material_zones:
    - shape: custom                        # topping slab, y in [1200, 1400]
      params:
        exterior: [[0, 1200], [600, 1200], [600, 1400], [0, 1400]]
      material: c_topping
      name: topping
  rebars:
    - {x:  60, y:   60, As: 800, material: steel, name: A1}
    - {x: 540, y:   60, As: 800, material: steel, name: A2}
    - {x:  60, y: 1340, As: 400, material: steel, name: T1}
    - {x: 540, y: 1340, As: 400, material: steel, name: T2}

# Sign convention: compression-negative N; Mx = sum F * (y - y_ref),
# so a sagging moment (tension at the bottom bars A1/A2) is NEGATIVE.
demands:
  - {name: G1, N_kN: -300, Mx_kNm: -220}   # self weight on the precast
  - {name: G2, Mx_kNm: -180}               # topping weight + finishes
  - {name: Q,  Mx_kNm: -260}               # service load on the composite

combinations:
  - name: staged-composite
    stages:
      - name: precast-alone
        components: [{ref: G1, factor: 1.35}]
        # T1/T2 sit in the topping: they cannot exist before their
        # parent zone is cast (containment invariant).
        section_ops: {deactivate: [T1, T2], release: false}
      - name: cast-topping
        components: [{ref: G2, factor: 1.35}]
        section_ops:
          # Engine-level explicit datum (illustrative values); with
          # the Task-2 timeline this becomes datum: auto.
          activate_bulk:
            topping: {eps0: 2.1e-4, chi_x: 3.4e-7, chi_y: 0.0}
          activate: [T1, T2]
      - name: service
        components: [{ref: Q, factor: 1.5}]

output:
  generate_mx_my: false
"""


# ==================================================================
#  Driver
# ==================================================================

#: (function, filename, path hint) — the hint disambiguates module
#: names that exist in more than one package directory.
_TARGETS = (
    (edits_geometry,       "geometry.py",      "geometry/geometry.py"),
    (edits_fiber,          "fiber.py",         "geometry/fiber.py"),
    (edits_io_yaml,        "io_yaml.py",       ""),
    (edits_section_state,  "section_state.py", "solver/section_state.py"),
    (edits_resolve_stages, "section_state.py", "solver/section_state.py"),
    (edits_integrator,     "integrator.py",    "solver/integrator.py"),
    (edits_sls,            "sls.py",           "solver/sls.py"),
    (edits_yaml_prestress, "example_prestress.yaml", ""),
    (edits_yaml_reference, "yaml_reference_example.yaml", ""),
)


def main(argv=None) -> int:
    r"""
    Apply the Task-1 edit set.

    Returns
    -------
    int
        0 on success (all edits applied or already applied, patched
        modules compile); 1 on any anchor failure.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--root", default=".",
                    help="Repository root (default: cwd).")
    ap.add_argument("--check", action="store_true",
                    help="Dry run: report, write nothing.")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    print(f"Phase 8 / Task 1 patcher — root: {root}")
    # Anchor resolution to the canonical layout: shadow trees (e.g.
    # iteration snapshots archived under a sibling directory) are
    # excluded by construction, not by name-matching.  A missing
    # canonical root is a hard, explicit failure.
    try:
        pkg_root = _package_root(root)
        ex_root = _examples_root(root)
    except FileNotFoundError as exc:
        print(f"  [!] {exc}")
        print("\nSummary: 0 applied (canonical layout not found — "
              "nothing written)")
        return 1
    print(f"  package : {pkg_root}")
    print(f"  examples: {ex_root}")
    ok = True
    unresolved = []
    open_files = {}
    for fn, name, hint in _TARGETS:
        is_yaml = name.endswith((".yaml", ".yml"))
        search_root = ex_root if is_yaml else pkg_root
        try:
            path = _find_file(search_root, name, hint)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"  [!] {name}: {exc}")
            unresolved.append(name)
            ok = False
            continue
        pf = open_files.get(path)
        if pf is None:
            pf = open_files[path] = PatchFile(path)
            print(f"  -- {path.relative_to(root)} "
                  f"({'CRLF' if pf.crlf else 'LF'})")
        fn(pf)

    for path, pf in open_files.items():
        if pf.failed:
            ok = False
            for msg in pf.failed:
                print(f"  [!] {path.name}: {msg}")

    if ok:
        for path, pf in open_files.items():
            pf.save(args.check)
        # New shipped example (created only if absent — idempotent).
        new_ex = ex_root / "example_composite_topping.yaml"
        if new_ex.exists():
            print(f"    [=] {new_ex.name}: already present")
        elif not args.check:
            new_ex.write_bytes(_COMPOSITE_EXAMPLE.encode("utf-8"))
            print(f"    [+] {new_ex.name}: created")

        # Compile gate on every patched Python module.
        if not args.check:
            for path in open_files:
                if path.suffix == ".py":
                    try:
                        py_compile.compile(str(path), doraise=True)
                    except py_compile.PyCompileError as exc:
                        print(f"  [!] compile check failed: {exc}")
                        ok = False

    n_app = sum(pf.applied for pf in open_files.values())
    n_skip = sum(pf.skipped for pf in open_files.values())
    n_fail = sum(len(pf.failed) for pf in open_files.values())
    n_unres = len(unresolved)
    if n_unres:
        print(f"  [!] {n_unres} target(s) could not be located: "
              f"{unresolved}")
    tail = " (dry run — nothing written)" if args.check else (
        "" if ok else " — ATOMIC ABORT: nothing written")
    print(f"\nSummary: {n_app} applied, {n_skip} already applied, "
          f"{n_fail} anchor-failed, {n_unres} unresolved{tail}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
