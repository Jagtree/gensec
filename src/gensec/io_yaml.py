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
# along with GenSec. If not, see <https://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------

r"""
YAML input loader for GenSec.

Reads a YAML file describing materials, section geometry, and
(optionally) load demands, and returns fully constructed GenSec
objects ready for analysis.

Section geometry format
-----------------------
The ``section`` block supports two modes:

**Legacy rectangular** (backward-compatible):

.. code-block:: yaml

    section:
      B: 300
      H: 600
      bulk_material: concrete_1
      n_fibers_y: 100
      n_fibers_x: 1
      rebars:
        - y: 40
          As: 942.5
          material: steel_1

**Generic section** (new):

.. code-block:: yaml

    section:
      shape: tee            # or: rect, circle, annulus, h, box,
                             #     single_tee, double_tee, custom
      params:
        bf: 800
        hf: 150
        bw: 300
        hw: 450
      bulk_material: concrete_1
      mesh_size: 15
      mesh_method: grid      # or: triangle
      rebars:
        - y: 40
          x: 150
          As: 942.5
          material: steel_1

**Custom polygon** (arbitrary vertex list):

.. code-block:: yaml

    section:
      shape: custom
      params:
        exterior: [[0,0], [300,0], [300,600], [0,600]]
        holes:
          - [[50,50], [250,50], [250,150], [50,150]]
      bulk_material: concrete_1
      mesh_size: 10
      mesh_method: triangle
      rebars: []

The YAML parser detects which mode to use:

- If ``shape`` is present → generic section.
- If ``B`` and ``H`` are present without ``shape`` → legacy
  rectangular (wrapped via :class:`RectSection`).
"""

import yaml
import numpy as np

from .materials import Concrete, Steel, TabulatedMaterial
from .geometry.fiber import RebarLayer, Tendon
from .geometry.section import RectSection
from .geometry.geometry import GenericSection
from .geometry import primitives as prim
from .solver.section_state import PrestressAction
from .materials.ec2_bridge import (
    concrete_from_class, concrete_from_ec2,
    prestress_from_ec2, prestress_from_class,
)


# ---- Material builders (unchanged) ----

_MATERIAL_BUILDERS = {
    "concrete_ec2_gen1_custom": {
        "cls": Concrete,
        "params": ["fck", "gamma_c", "alpha_cc", "n_parabola",
                    "eps_c2", "eps_cu2", "fct", "Ec"],
    },
    "steel": {
        "cls": Steel,
        "params": ["fyk", "gamma_s", "Es", "k_hardening", "eps_su",
                    "works_in_compression"],
    },
    "tabulated": {
        "cls": TabulatedMaterial,
        "params": ["strains", "stresses", "name"],
    },
}

# Backward-compatible aliases for material type names.
_MATERIAL_ALIASES = {
    "concrete": "concrete_ec2_gen1_custom",
    "concrete_ec2": "concrete_ec2_gen1",
}


def _build_material(name, spec):
    """
    Build a Material instance from a YAML specification dict.

    Supported types: ``concrete_ec2_gen1_custom``,
    ``concrete_ec2_gen1``, ``steel``, ``tabulated``.
    Legacy aliases ``concrete`` and ``concrete_ec2`` are also
    accepted.

    Parameters
    ----------
    name : str
        Key used in the YAML ``materials`` block.
    spec : dict
        Must contain a ``'type'`` key.

    Returns
    -------
    Material

    Raises
    ------
    ValueError
        Unknown material type.
    """
    mat_type = spec.get("type", "").lower()

    # Resolve backward-compatible aliases.
    mat_type = _MATERIAL_ALIASES.get(mat_type, mat_type)

    if mat_type == "prestressing_steel_ec2":
        # EC2 §3.3 prestressing steel. Either a standard designation
        # ('class', e.g. 'Y1860S7') or explicit characteristic values
        # (f_p01k, f_pk, eps_uk). The partial factor derives from the
        # limit state / national annex, exactly as for concrete.
        common = dict(
            ls=spec.get("ls", "F"),
            NA=spec.get("NA", "EC2"),
            eps_ud_factor=float(spec.get("eps_ud_factor", 0.9)),
            gamma_s_override=spec.get("gamma_s_override"),
            diagram=spec.get("diagram", "horizontal"),
            works_in_compression=bool(
                spec.get("works_in_compression", True)),
        )
        ps_class = spec.get("class")
        if ps_class:
            return prestress_from_class(ps_class, **common)
        f_p01k = spec.get("f_p01k")
        f_pk = spec.get("f_pk")
        eps_uk = spec.get("eps_uk")
        if f_p01k is None or f_pk is None or eps_uk is None:
            raise ValueError(
                f"Material '{name}': prestressing_steel_ec2 requires "
                f"'class' (e.g. 'Y1860S7') or all of "
                f"'f_p01k', 'f_pk', 'eps_uk'."
            )
        return prestress_from_ec2(
            f_p01k=float(f_p01k), f_pk=float(f_pk),
            eps_uk=float(eps_uk),
            Ep=float(spec.get("Ep", 195000.0)),
            **common,
        )

    if mat_type == "concrete_ec2_gen1":
        # Tension branch flags (common to both class-based and fck-based).
        enable_tension = bool(spec.get("enable_tension", False))
        tension_fct = spec.get("tension_fct", "fctd")

        conc_class = spec.get("class")
        if conc_class:
            return concrete_from_class(
                conc_class,
                ls=spec.get("ls", "F"),
                loadtype=spec.get("loadtype", "slow"),
                TypeConc=spec.get("TypeConc", "R"),
                NA=spec.get("NA", "French"),
                time=spec.get("time", 28),
                enable_tension=enable_tension,
                tension_fct=tension_fct,
            )
        fck = spec.get("fck")
        if fck is None:
            raise ValueError(
                f"Material '{name}': concrete_ec2_gen1 requires "
                f"'class' (e.g. 'C30/37') or 'fck'."
            )
        return concrete_from_ec2(
            fck=float(fck),
            ls=spec.get("ls", "F"),
            loadtype=spec.get("loadtype", "slow"),
            TypeConc=spec.get("TypeConc", "R"),
            NA=spec.get("NA", "French"),
            time=spec.get("time", 28),
            enable_tension=enable_tension,
            tension_fct=tension_fct,
        )

    if mat_type not in _MATERIAL_BUILDERS:
        raise ValueError(
            f"Unknown material type '{mat_type}' for '{name}'. "
            f"Valid: {list(_MATERIAL_BUILDERS.keys())} "
            f"+ 'concrete_ec2_gen1'"
        )

    builder = _MATERIAL_BUILDERS[mat_type]
    cls = builder["cls"]
    kwargs = {}
    for p in builder["params"]:
        if p in spec:
            val = spec[p]
            if isinstance(val, list):
                val = np.array(val, dtype=float)
            kwargs[p] = val
    return cls(**kwargs)


# ---- Shape factory dispatch ----

_SHAPE_FACTORIES = {
    "rect": lambda p: prim.rect_poly(p["B"], p["H"]),
    "circle": lambda p: prim.circle_poly(
        p["D"], resolution=p.get("resolution", 64)),
    "annulus": lambda p: prim.annulus_poly(
        p["D_ext"], p["D_int"],
        resolution=p.get("resolution", 64)),
    "tee": lambda p: prim.tee_poly(
        p["bf"], p["hf"], p["bw"], p["hw"]),
    "inv_tee": lambda p: prim.inv_tee_poly(
        p["bf"], p["hf"], p["bw"], p["hw"]),
    "h": lambda p: prim.h_poly(
        p["bf"], p["hf_top"], p["hf_bot"], p["bw"], p["hw"]),
    "box": lambda p: prim.box_poly(
        p["B"], p["H"], p["tw"], p["tf_top"],
        tf_bot=p.get("tf_bot")),
    "single_tee": lambda p: prim.single_tee_slab_poly(
        p["b_top"], p["h_top"], p["bw"], p["hw"]),
    "double_tee": lambda p: prim.double_tee_slab_poly(
        p["b_top"], p["h_top"], p["bw"], p["hw"],
        p["stem_spacing"]),
    "custom": lambda p: prim.custom_poly(
        p["exterior"], holes=p.get("holes")),
}


def _build_polygon(sec_spec):
    r"""
    Build a Shapely polygon from the ``section`` YAML block.

    Parameters
    ----------
    sec_spec : dict
        The ``section`` block from YAML.

    Returns
    -------
    shapely.geometry.Polygon

    Raises
    ------
    ValueError
        If the shape type is not recognized.
    """
    shape = sec_spec["shape"].lower()
    params = sec_spec.get("params", {})

    if shape not in _SHAPE_FACTORIES:
        raise ValueError(
            f"Unknown section shape '{shape}'. "
            f"Valid: {list(_SHAPE_FACTORIES.keys())}"
        )

    return _SHAPE_FACTORIES[shape](params)


# ---- Main loader ----

def load_yaml(filepath):
    r"""
    Load a GenSec input file and return constructed objects.

    Detects whether the section block uses the legacy rectangular
    format (``B`` + ``H``) or the new generic format (``shape``).

    Parameters
    ----------
    filepath : str or pathlib.Path

    Returns
    -------
    dict
        Keys: ``'materials'``, ``'section'`` (GenericSection or
        RectSection), ``'demands'``, ``'combinations'``,
        ``'output_options'``.
    """
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    # ---- Materials ----
    materials = {}
    for mat_name, mat_spec in data.get("materials", {}).items():
        mat = _build_material(mat_name, mat_spec)
        mat.name = mat_name
        materials[mat_name] = mat

    # ---- Section ----
    sec_spec = data["section"]
    bulk_name = sec_spec["bulk_material"]
    if bulk_name not in materials:
        raise ValueError(
            f"Bulk material '{bulk_name}' not found in materials."
        )

    # Parse rebars (common to both modes)
    rebars = _parse_rebars(sec_spec, materials)

    # Parse tendons (prestress, common to both modes)
    tendons = _parse_tendons(sec_spec, materials)

    if "shape" in sec_spec:
        # ---- New generic mode ----
        polygon = _build_polygon(sec_spec)

        # Optional multi-material zones
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

        section = GenericSection(
            polygon=polygon,
            bulk_material=materials[bulk_name],
            rebars=rebars,
            mesh_size=float(sec_spec.get("mesh_size", 10)),
            mesh_method=sec_spec.get("mesh_method", "grid"),
            bulk_materials=bulk_materials,
            tendons=tendons,
        )
    else:
        # ---- Legacy rectangular mode ----
        section = RectSection(
            B=float(sec_spec["B"]),
            H=float(sec_spec["H"]),
            bulk_material=materials[bulk_name],
            rebars=rebars,
            n_fibers_y=int(sec_spec.get("n_fibers_y",
                            sec_spec.get("n_fibers", 100))),
            n_fibers_x=int(sec_spec.get("n_fibers_x", 1)),
            tendons=tendons,
        )

    # ---- Bulk pre-strain (resistance-side imposed-strain offset) ----
    # Accept ``prestrain`` (canonical) or ``eps_init`` (alias).  Stored
    # on the section; defaults to 0.0 so sections without the field are
    # unaffected.  See ``GenericSection.bulk_eps_init``.
    section.bulk_eps_init = _parse_bulk_prestrain(sec_spec)

    # ---- Demands ----
    demands = [_parse_demand(d) for d in data.get("demands", [])]

    # ---- Combinations (v2.1: components / stages) ----
    combinations = [_parse_combination(c)
                    for c in data.get("combinations", [])]

    # ---- Prestress actions (demand-side loads) ----
    # Resolve each stage's raw ``prestress_actions`` specs into
    # ``PrestressAction`` objects now that the section (hence its
    # reference point and tendon geometry) is known, and attach them as
    # ``_prestress_actions`` for the staged engines to sum into the
    # demand.  A no-op for combinations that declare none.
    _resolve_prestress_actions(combinations, section, sec_spec)

    # ---- Envelopes ----
    envelopes = [_parse_envelope(e)
                 for e in data.get("envelopes", [])]

    # ---- Output options (with v2.1 flag defaults) ----
    output_opts = _parse_output_flags(data.get("output", {}))

    return {
        "materials": materials,
        "section": section,
        "demands": demands,
        "combinations": combinations,
        "envelopes": envelopes,
        "output_options": output_opts,
    }


def _parse_rebars(sec_spec, materials):
    """
    Parse the ``rebars`` list from a section YAML block.

    If ``As`` is omitted but ``diameter`` is given, the area is
    computed automatically as
    :math:`A_s = n_{\\text{bars}} \\cdot \\pi/4 \\cdot d^2`.
    If both are given, ``As`` takes precedence.

    Parameters
    ----------
    sec_spec : dict
        Section specification dict.
    materials : dict
        Material name → Material mapping.

    Returns
    -------
    list of RebarLayer
    """
    rebars = []
    for rb_spec in sec_spec.get("rebars", []):
        mat_name = rb_spec["material"]
        if mat_name not in materials:
            raise ValueError(
                f"Rebar material '{mat_name}' not found in materials."
            )
        rebars.append(RebarLayer(
            y=float(rb_spec["y"]),
            As=float(rb_spec.get("As", 0)),
            material=materials[mat_name],
            x=float(rb_spec["x"]) if "x" in rb_spec else None,
            embedded=bool(rb_spec.get("embedded", True)),
            n_bars=int(rb_spec.get("n_bars", 1)),
            diameter=float(rb_spec.get("diameter", 0)),
        ))
    return rebars


def _parse_tendons(sec_spec, materials):
    r"""
    Parse the ``tendons`` list from a section YAML block (prestress).

    Each tendon entry specifies a location, a prestressing-steel
    material, an area (directly via ``Ap`` or via ``n_strands`` and
    ``area_strand``), and an effective prestrain ``eps_pe`` (positive
    = tension).  Phase 1 supports bonded tendons only.

    .. code-block:: yaml

        tendons:
          - y: 80
            x: 200
            material: ps_1
            Ap: 1400
            eps_pe: 0.0065
            system: post
            bonded: true

    Parameters
    ----------
    sec_spec : dict
        Section specification dict.
    materials : dict
        Material name → Material mapping.

    Returns
    -------
    list of Tendon
    """
    tendons = []
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
            embedded=bool(t_spec.get("embedded", True)),
            n_strands=int(t_spec.get("n_strands", 1)),
            area_strand=float(t_spec.get("area_strand", 0)),
        ))
    return tendons


def _parse_demand(d_spec):
    """
    Parse a single demand triple from YAML.

    Accepts ``Mx_kNm`` / ``My_kNm`` (canonical) or legacy
    ``M_kNm`` (Mx only, My=0).

    Parameters
    ----------
    d_spec : dict

    Returns
    -------
    dict
        Keys: ``name``, ``N`` [N], ``Mx`` [N*mm], ``My`` [N*mm].
    """
    N = float(d_spec.get("N_kN", 0)) * 1e3

    if "Mx_kNm" in d_spec:
        Mx = float(d_spec["Mx_kNm"]) * 1e6
        My = float(d_spec.get("My_kNm", 0)) * 1e6
    elif "M_kNm" in d_spec:
        Mx = float(d_spec["M_kNm"]) * 1e6
        My = 0.0
    else:
        Mx = 0.0
        My = 0.0

    return {
        "name": d_spec.get("name", "unnamed"),
        "N": N,
        "Mx": Mx,
        "My": My,
    }


# ---- Combination parser (v2.1) ----

def _parse_combination(c_spec):
    r"""
    Parse a combination from YAML.

    A combination has **either** ``components`` (simple factored sum)
    **or** ``stages`` (sequential accumulation), never both.

    Simple form:

    .. code-block:: yaml

        - name: SLU_1
          components:
            - {ref: G, factor: 1.3}
            - {ref: Q1, factor: 1.5}

    Staged form:

    .. code-block:: yaml

        - name: SLU_sismico
          stages:
            - name: gravitazionale
              components:
                - {ref: G, factor: 1.0}
            - name: sisma
              components:
                - {ref: Ex, factor: 1.0}

    A stage may additionally carry a ``prestress_actions`` block of
    demand-side prestressing loads (post-tension / external / jacking
    on hardened concrete).  Each entry gives a force — ``P`` [N],
    ``P_kN`` [kN], or ``sigma_p0`` [MPa] ``+`` ``Ap`` [mm²] — and a
    position — ``x`` / ``y`` [mm] or a ``ref`` to a declared tendon's
    geometry (index or ``name``):

    .. code-block:: yaml

        - name: PT_jacking
          stages:
            - name: peso_proprio
              components: [{ref: G, factor: 1.0}]
            - name: tesatura
              components: []
              prestress_actions:
                - {P_kN: 1400, x: 200, y: 80}
                - {sigma_p0: 1000, Ap: 1400, ref: 0}

    The raw specs are carried unresolved here (the section reference
    point is not yet known) and resolved by
    :func:`_resolve_prestress_actions` in :func:`load_yaml`.

    Parameters
    ----------
    c_spec : dict
        Raw YAML dict for one combination entry.

    Returns
    -------
    dict
        Parsed combination with ``name`` and either ``components``
        or ``stages``.

    Raises
    ------
    ValueError
        If both ``components`` and ``stages`` are present, or neither.
    """
    name = c_spec.get("name", "unnamed")
    has_components = "components" in c_spec
    has_stages = "stages" in c_spec

    if has_components and has_stages:
        raise ValueError(
            f"Combination '{name}': cannot have both 'components' "
            f"and 'stages'."
        )
    if not has_components and not has_stages:
        raise ValueError(
            f"Combination '{name}': must have 'components' or "
            f"'stages'."
        )

    if has_components:
        if "prestress_actions" in c_spec:
            raise ValueError(
                f"Combination '{name}': 'prestress_actions' is only "
                f"valid on a stage of a staged combination, not on a "
                f"simple (components-only) combination."
            )
        return {
            "name": name,
            "components": _parse_component_list(c_spec["components"]),
        }

    # Staged.
    if "prestress_actions" in c_spec:
        raise ValueError(
            f"Combination '{name}': place 'prestress_actions' on an "
            f"individual stage, not at the combination level."
        )
    stages = []
    for i, s_spec in enumerate(c_spec["stages"]):
        stage = {
            "name": s_spec.get("name", f"stage_{i}"),
            "components": _parse_component_list(
                s_spec.get("components", [])),
        }
        # Carry the raw prestress-action specs forward unresolved; the
        # main loader resolves them against the built section (it needs
        # the reference point and any tendon geometry referenced).
        if "prestress_actions" in s_spec:
            stage["_prestress_action_specs"] = list(
                s_spec["prestress_actions"])
        stages.append(stage)
    return {"name": name, "stages": stages}


def _parse_component_list(comp_list):
    """
    Parse a list of component references with optional factors.

    Parameters
    ----------
    comp_list : list of dict
        Each dict has ``ref`` (str) and optionally ``factor``
        (float, default 1.0).

    Returns
    -------
    list of dict
        ``[{"ref": str, "factor": float}, ...]``
    """
    parsed = []
    for c in comp_list:
        parsed.append({
            "ref": c["ref"],
            "factor": float(c.get("factor", 1.0)),
        })
    return parsed


# ---- Bulk pre-strain + prestress-action resolution ----

def _parse_bulk_prestrain(sec_spec):
    r"""
    Read the section bulk pre-strain from a ``section`` YAML block.

    Accepts ``prestrain`` (canonical) or ``eps_init`` (alias) as a
    uniform locked-in bulk strain [-], tension positive.  Returns
    ``0.0`` when neither key is present, so a section that does not
    declare one is unaffected.

    Parameters
    ----------
    sec_spec : dict
        The ``section`` block from YAML.

    Returns
    -------
    float
        Bulk pre-strain [-].

    Raises
    ------
    ValueError
        If both ``prestrain`` and ``eps_init`` are present with
        different values (ambiguous).
    """
    has_p = "prestrain" in sec_spec
    has_e = "eps_init" in sec_spec
    if has_p and has_e:
        if float(sec_spec["prestrain"]) != float(sec_spec["eps_init"]):
            raise ValueError(
                "section: 'prestrain' and 'eps_init' both given with "
                "different values; they are aliases — set only one."
            )
    if has_p:
        return float(sec_spec["prestrain"])
    if has_e:
        return float(sec_spec["eps_init"])
    return 0.0


def _resolve_prestress_actions(combinations, section, sec_spec):
    r"""
    Resolve raw ``prestress_actions`` specs into
    :class:`~gensec.solver.section_state.PrestressAction` objects.

    Walks every staged combination and replaces each stage's deferred
    ``_prestress_action_specs`` (carried by :func:`_parse_combination`)
    with a list of resolved actions under the key ``_prestress_actions``
    — the key the staged engines
    (:meth:`~gensec.solver.check.VerificationEngine._check_staged`,
    :meth:`~gensec.solver.analysis.AnalysisEngine._analyze_staged`)
    consume and sum into the demand.

    Each action is taken about the section reference point
    (``x_centroid`` / ``y_centroid``), which is the point the demand
    path and the integrator both use, so the resolved triple is
    directly additive to the cumulative demand.

    Mutates *combinations* in place.

    Parameters
    ----------
    combinations : list of dict
        Parsed combinations (output of :func:`_parse_combination`).
    section : GenericSection
        Built section (supplies the reference point and tendon
        geometry for ``ref`` resolution).
    sec_spec : dict
        Raw ``section`` block; its ``tendons`` list is used to resolve
        a string ``ref`` against an optional tendon ``name``.

    Raises
    ------
    ValueError
        If a ``prestress_actions`` block is declared on a simple
        (non-staged) combination, or an entry is malformed.

    Notes
    -----
    Prestress actions are routed **per stage**.  A jacking event on
    hardened concrete (post-tension / external / unbonded) is therefore
    expressed as a stage carrying the action — physically a construction
    step — and never as a section element (a bonded ``Tendon``); this
    keeps the resistance/demand separation intact (the action never
    reaches the capacity hash).
    """
    x_ref = float(section.x_centroid)
    y_ref = float(section.y_centroid)
    name_map = _tendon_name_map(sec_spec)

    for combo in combinations:
        if "stages" not in combo:
            continue
        for stage in combo["stages"]:
            specs = stage.pop("_prestress_action_specs", None)
            if not specs:
                continue
            stage["_prestress_actions"] = [
                _resolve_single_prestress_action(
                    spec, section, x_ref, y_ref, name_map)
                for spec in specs
            ]


def _tendon_name_map(sec_spec):
    r"""
    Map a tendon ``name`` (if declared in YAML) to its list index.

    Tendons are referenced by integer index by default; this allows an
    optional human-readable ``name`` key on a tendon spec to be used as
    a ``ref`` instead.

    Parameters
    ----------
    sec_spec : dict
        Raw ``section`` block.

    Returns
    -------
    dict
        ``{name: index}`` for every tendon spec that declares a
        ``name``.
    """
    out = {}
    for i, t in enumerate(sec_spec.get("tendons", [])):
        nm = t.get("name") if isinstance(t, dict) else None
        if nm is not None:
            out[str(nm)] = i
    return out


def _resolve_single_prestress_action(spec, section, x_ref, y_ref,
                                     name_map):
    r"""
    Resolve one ``prestress_actions`` entry into a
    :class:`~gensec.solver.section_state.PrestressAction`.

    Force magnitude (tension positive [N]) comes from **either**

    - ``P`` [N] or ``P_kN`` [kN] (explicit force), **or**
    - ``sigma_p0`` [MPa] :math:`\times` ``Ap`` [mm²] (stress
      :math:`\times` area).

    Position comes from **either** explicit ``x`` / ``y`` [mm] **or** a
    ``ref`` to a declared tendon's geometry — an integer index or a
    string matching a tendon ``name`` — in which case the section's
    resolved tendon coordinates are used.

    Parameters
    ----------
    spec : dict
        One raw entry from a stage's ``prestress_actions`` list.
    section : GenericSection
        Built section (tendon coordinate arrays for ``ref``).
    x_ref, y_ref : float
        Section reference point [mm].
    name_map : dict
        ``{tendon_name: index}`` from :func:`_tendon_name_map`.

    Returns
    -------
    PrestressAction

    Raises
    ------
    ValueError
        If the force or the position cannot be resolved, or a ``ref``
        is out of range / unknown.
    """
    # ---- Force [N], tension positive ----
    if "P" in spec:
        P = float(spec["P"])
    elif "P_kN" in spec:
        P = float(spec["P_kN"]) * 1e3
    elif "sigma_p0" in spec and "Ap" in spec:
        P = float(spec["sigma_p0"]) * float(spec["Ap"])
    else:
        raise ValueError(
            "prestress_actions entry: provide 'P' [N], 'P_kN' [kN], "
            "or both 'sigma_p0' [MPa] and 'Ap' [mm^2]. "
            f"Got keys: {sorted(spec)}."
        )

    # ---- Position [mm] ----
    if "ref" in spec:
        ref = spec["ref"]
        if isinstance(ref, str):
            if ref not in name_map:
                raise ValueError(
                    f"prestress_actions entry: ref '{ref}' does not "
                    f"match any tendon 'name'. Known: {sorted(name_map)}."
                )
            idx = name_map[ref]
        else:
            idx = int(ref)
        n_ten = int(getattr(section, "x_tendons", np.empty(0)).size)
        if not (0 <= idx < n_ten):
            raise ValueError(
                f"prestress_actions entry: tendon ref index {idx} out "
                f"of range (section has {n_ten} tendon(s))."
            )
        x = float(section.x_tendons[idx])
        y = float(section.y_tendons[idx])
        # Explicit x/y, if also given, override the referenced geometry.
        x = float(spec.get("x", x))
        y = float(spec.get("y", y))
    elif "x" in spec and "y" in spec:
        x = float(spec["x"])
        y = float(spec["y"])
    else:
        raise ValueError(
            "prestress_actions entry: provide 'x' and 'y' [mm], or a "
            f"'ref' to a declared tendon. Got keys: {sorted(spec)}."
        )

    return PrestressAction.from_force(
        P, x, y, x_ref=x_ref, y_ref=y_ref,
        label=str(spec.get("label", "")),
        origin="prestress",
    )


# ---- Envelope parser ----

def _parse_envelope(e_spec):
    r"""
    Parse an envelope from YAML.

    Members can be references to demands/combinations or inline
    demand points:

    .. code-block:: yaml

        - name: Envelope_1
          members:
            - {ref: SLU_1}
            - {ref: G, factor: 1.2}
            - {N_kN: -2500, Mx_kNm: 100, My_kNm: 50}

    Parameters
    ----------
    e_spec : dict
        Raw YAML dict for one envelope entry.

    Returns
    -------
    dict
        ``{"name": str, "members": list}``.
    """
    name = e_spec.get("name", "unnamed")
    members = []

    for i, m_spec in enumerate(e_spec.get("members", [])):
        member = {}
        if "ref" in m_spec:
            member["ref"] = m_spec["ref"]
        else:
            # Inline demand.  Keep raw kN/kNm for the engine
            # to convert.
            member["N_kN"] = float(m_spec.get("N_kN", 0))
            member["Mx_kNm"] = float(m_spec.get("Mx_kNm", 0))
            member["My_kNm"] = float(m_spec.get("My_kNm", 0))
            member["name"] = m_spec.get("name", f"{name}[{i}]")

        if "factor" in m_spec:
            member["factor"] = float(m_spec["factor"])

        members.append(member)

    return {"name": name, "members": members}


# ---- Output flags parser (v2.1 defaults) ----

def _parse_output_flags(output_spec):
    r"""
    Parse the ``output`` block with v2.1 flag defaults.

    Parameters
    ----------
    output_spec : dict
        Raw YAML ``output`` block.

    Returns
    -------
    dict
        All original keys preserved, plus guaranteed defaults for
        the v2.1 utilization flags.
    """
    # Start with all original keys.
    flags = dict(output_spec)

    # Utilization flag defaults.
    flags.setdefault("eta_norm", True)        # principal: linear distance to boundary (alpha)
    flags.setdefault("eta_norm_beta", True)   # composite ratio (sensitivity to perturbation)
    flags.setdefault("eta_norm_ray", False)   # ray-cast from origin in normalised space
    flags.setdefault("eta_2D", False)         # ray-cast in (Mx,My) plane at fixed N
    flags.setdefault("eta_path_norm_ray", False)       # ray-cast staged in normalised space
    flags.setdefault("eta_path_norm_beta", False)  # composite ratio along stage segment
    flags.setdefault("eta_path_2D", False)
    flags.setdefault("delta_N_tol", 0.03)

    # Tiered reporting defaults.
    flags.setdefault("verification_top_k", 10)
    flags.setdefault("fiber_details_top_k", 5)

    # Domain generation defaults.
    flags.setdefault("generate_mx_my", False)
    flags.setdefault("generate_3d_surface", False)
    flags.setdefault("n_angles_mx_my", 144)
    flags.setdefault("n_scan_mx_my", 120)
    flags.setdefault("n_chi_mx_my", 14)

    # Moment-curvature and ductility generation defaults.
    flags.setdefault("generate_moment_curvature", True)
    flags.setdefault("generate_polar_ductility", True)
    flags.setdefault("generate_3d_moment_curvature", True)

    return flags
