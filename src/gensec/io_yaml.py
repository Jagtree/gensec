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
from .solver.losses import LossModel
from .materials.rheology import (
    EC2RheologicalModel, TabulatedRheologicalModel,
)
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

        # Optional multi-material zones.  Key validation is
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
    # reference point and tendon geometry/names) is known, and attach
    # them as ``_prestress_actions`` for the staged engines to sum into
    # the demand.  A no-op for combinations that declare none.
    _resolve_prestress_actions(combinations, section)

    # ---- Section ops (capacity-side stage operations, Phase 3) ----
    # Resolve each stage's raw ``section_ops`` specs (element names ->
    # union indices) into the form ``StagedDomainManager.resolve_stages``
    # consumes, attached under ``section_ops``.  A no-op for
    # combinations that declare none.
    _resolve_section_ops(combinations, section)

    # ---- Envelopes ----
    envelopes = [_parse_envelope(e)
                 for e in data.get("envelopes", [])]

    # ---- Output options (with v2.1 flag defaults) ----
    output_opts = _parse_output_flags(data.get("output", {}))

    # Phase-8 Task-2: the single construction timeline (G-D1).
    # Carried raw (a list of single-key event mappings) for
    # gensec.solver.timeline.ConstructionTimeline.from_block. Absent
    # -> None and the whole timeline machinery is inert (the run is
    # byte-identical to the pre-Task-2 behaviour).
    construction_history = data.get("construction_history")

    # Phase-5 / C5: the rheological loss models an ``interval`` may
    # reference.  Absent -> {} and the whole machinery is inert.
    losses_models = _parse_losses_models(data.get("losses_models"))

    return {
        "materials": materials,
        "section": section,
        "demands": demands,
        "combinations": combinations,
        "envelopes": envelopes,
        "output_options": output_opts,
        "construction_history": construction_history,
        "losses_models": losses_models,
    }


#: Provider constructors reachable from a YAML ``losses_models`` block.
#: A norm enters GenSec by adding one line here and one class in
#: :mod:`gensec.materials.rheology` -- nothing in the container moves.
_RHEO_PROVIDERS = {
    "ec2": EC2RheologicalModel,
    "ntc": EC2RheologicalModel,      # NTC 2018 adopts the EC2 formulae
    "tabulated": TabulatedRheologicalModel,
}
# 'aci' is deliberately absent.  A rheology is declarable only for a norm
# GenSec can actually design to, and there is no ACI material -- no bridge,
# no partial factors, no limit states.  The ACI provider exists, as a
# falsification fixture, in aci209_falsification.py, outside the package.
#: Keys of a ``losses_models`` entry that belong to the **container**
#: (:class:`~gensec.solver.losses.LossModel`), not to the provider.
_LOSS_CONTAINER_KEYS = ("provider", "chi", "relaxation_reduction",
                        "n_steps", "steps")


def _parse_losses_models(block):
    r"""
    Parse the top-level ``losses_models`` block into
    :class:`~gensec.solver.losses.LossModel` objects (Phase 5 / C5).

    ::

        losses_models:
          rheo_precast:
            provider: ec2               # | ntc | tabulated
            fck: 45                     # -> the provider's constructor
            cement_class: R
            RH: 70
            chi: lump                   # -> the container: lump | from_J | float
            relaxation_reduction: 0.8   # -> the container
            steps: [1, 7, 30, 365]      # -> the container (opt-in)

    Every key that is **not** a container knob is forwarded verbatim to
    the provider's constructor, so a new norm needs no change here beyond
    an entry in :data:`_RHEO_PROVIDERS` — the split between *mechanics*
    (container) and *normative content* (provider) is enforced by the
    parser itself.

    The drying geometry :math:`(A_c, u)` is deliberately **not** a YAML
    key: the container binds it per concrete zone from the section's own
    polygons, because in a composite section every zone has its own
    exposed perimeter.  A provider constructed with an explicit ``A_c``
    and ``u`` keeps them (an engineer overriding the exposed perimeter of
    a topping cast onto a web).

    Parameters
    ----------
    block : dict or None
        The raw ``losses_models`` mapping.

    Returns
    -------
    dict
        ``{name: LossModel}``.  Empty when the block is absent, in which
        case the whole rheology machinery is inert and the run is
        byte-identical to the pre-C5 behaviour.

    Raises
    ------
    ValueError
        Malformed block; unknown provider; a provider constructor that
        rejects its arguments (re-raised with the model's name attached).
    """
    if not block:
        return {}
    if not isinstance(block, dict):
        raise ValueError(
            f"'losses_models' must be a mapping {{name: spec}}, got "
            f"{type(block).__name__}."
        )
    out = {}
    for name, spec in block.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"losses_models['{name}'] must be a mapping, got "
                f"{type(spec).__name__}."
            )
        pname = str(spec.get("provider", "")).lower()
        if pname not in _RHEO_PROVIDERS:
            raise ValueError(
                f"losses_models['{name}']: unknown provider "
                f"{spec.get('provider')!r}. Valid: "
                f"{sorted(set(_RHEO_PROVIDERS))}."
            )
        pkwargs = {k: v for k, v in spec.items()
                   if k not in _LOSS_CONTAINER_KEYS}
        try:
            provider = _RHEO_PROVIDERS[pname](name=name, **pkwargs)
        except TypeError as exc:
            raise ValueError(
                f"losses_models['{name}']: the '{pname}' provider does "
                f"not accept these keys ({sorted(pkwargs)}) -- {exc}."
            ) from exc
        ckwargs = {k: spec[k] for k in _LOSS_CONTAINER_KEYS
                   if k in spec and k != "provider"}
        out[name] = LossModel(provider=provider, name=name, **ckwargs)
    return out


def _parse_rebars(sec_spec, materials):
    """
    Parse the ``rebars`` list from a section YAML block.

    If ``As`` is omitted but ``diameter`` is given, the area is
    computed automatically as
    :math:`A_s = n_{\\text{bars}} \\cdot \\pi/4 \\cdot d^2`.
    If both are given, ``As`` takes precedence.

    An optional ``name`` (symmetric to the tendon ``name``) makes the
    layer referenceable by name from a stage's ``section_ops`` block;
    see :func:`_union_name_map`.

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
            name=rb_spec.get("name"),
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
            bonded: true
            name: T_bottom     # optional; usable as prestress_actions ref
            # parent: <zone>   # staging-parent override; legal only
            #                  # with embedded: false (Phase 8)

    The retired ``system`` key ('pre'/'post') raises with a migration
    message: the construction system is derived from the staging
    timeline, never declared per tendon.

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
            embedded=bool(t_spec.get("embedded", True)),
            n_strands=int(t_spec.get("n_strands", 1)),
            area_strand=float(t_spec.get("area_strand", 0)),
            name=t_spec.get("name"),
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

    A stage may also carry (Phase-3 unified stage model):

    - ``section_ops`` — capacity-state operations applied *at* the
      stage, consumed by
      :meth:`~gensec.solver.section_state.StagedDomainManager.resolve_stages`:
      ``activate`` / ``deactivate`` (lists of element references),
      ``activate_bulk`` (``{zone_ref: {eps0, chi_x, chi_y}}`` — cast a
      bulk zone with its mandatory locked-in datum plane; Phase 8),
      ``eps_override`` (``{element_ref: eps}``, prestrain override [-]),
      ``bulk_eps`` (float [-]; consumed by the integrator as of
      Phase 5, see :func:`_parse_section_ops_spec`), ``release`` (bool, default
      ``True``; whether deactivations are force-released).  An element
      reference is a **union index** (integer position in the canonical
      ``rebars + tendons`` order of the section) or an element ``name``
      (:attr:`~gensec.geometry.fiber.RebarLayer.name` /
      :attr:`~gensec.geometry.fiber.Tendon.name`); names resolve to
      union indices in :func:`_resolve_section_ops`.
    - ``time`` — cumulative time since stage 0 [days], carried onto
      :attr:`~gensec.solver.section_state.SectionState.time_days`.
      Informational only: it never enters the capacity hash (losses act
      through ``eps_override``, never through time itself).  Omitted →
      the previous stage's value carries forward.
    - ``report`` — opaque per-stage reporting payload, echoed verbatim
      into the stage's result dict by the engines for the reporting
      layer.  No solver-side effect.

    .. code-block:: yaml

        - name: costruzione
          stages:
            - name: fase_1
              components: [{ref: G1, factor: 1.0}]
              section_ops:
                deactivate: [B_top, 3]     # names and union indices mix
              time: 0
            - name: fase_2
              components: [{ref: G2, factor: 1.0}]
              section_ops:
                activate: [B_top, 3]
                eps_override: {T_bottom: 0.0058}
              time: 28
              report: {note: "getto completato"}

    Like ``prestress_actions``, the raw ``section_ops`` spec is carried
    unresolved (``_section_ops_spec``) and resolved against the built
    section by :func:`_resolve_section_ops` in :func:`load_yaml`;
    value-level validation that needs no section (structure, unknown
    keys, ``time`` monotonicity) happens here.

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
        If both ``components`` and ``stages`` are present, or neither;
        if a stage-only key (``prestress_actions``, ``section_ops``,
        ``time``, ``report``) is misplaced on a simple combination or
        at the staged-combination level; if a ``section_ops`` block is
        malformed (see :func:`_parse_section_ops_spec`); if ``time``
        is negative or decreases across stages.
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
        for key in ("section_ops", "time", "report"):
            if key in c_spec:
                raise ValueError(
                    f"Combination '{name}': '{key}' is only valid on "
                    f"a stage of a staged combination, not on a simple "
                    f"(components-only) combination."
                )
        combo = {
            "name": name,
            "components": _parse_component_list(c_spec["components"]),
        }
        # Phase-8 Task-2: a components-based combination may anchor at
        # a construction-timeline point. The anchor metadata is passed
        # verbatim to the timeline compiler; its presence does not
        # conflict with the components/stages exclusivity (the compiler
        # emits the stages). Combinations without 'at' are unaffected.
        for _tl_key in ("at", "history_factors", "gamma_P"):
            if _tl_key in c_spec:
                combo[_tl_key] = c_spec[_tl_key]
        return combo

    # Staged.
    if "prestress_actions" in c_spec:
        raise ValueError(
            f"Combination '{name}': place 'prestress_actions' on an "
            f"individual stage, not at the combination level."
        )
    for key in ("section_ops", "time", "report"):
        if key in c_spec:
            raise ValueError(
                f"Combination '{name}': place '{key}' on an "
                f"individual stage, not at the combination level."
            )
    stages = []
    prev_time = None
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
        # Carry the raw section-ops spec forward unresolved (element
        # names need the built section); value-level validation that
        # needs no section happens now, in _parse_section_ops_spec.
        if "section_ops" in s_spec:
            stage["_section_ops_spec"] = _parse_section_ops_spec(
                name, stage["name"], s_spec["section_ops"])
        # Cumulative time at this stage [days].  Carry-through only:
        # it lands on SectionState.time_days and never enters the
        # capacity hash.  Guarded for sign and monotonicity here.
        if "time" in s_spec:
            t = s_spec["time"]
            try:
                t = float(t)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Combination '{name}', stage '{stage['name']}': "
                    f"'time' must be a number [days], got {t!r}."
                )
            if t < 0.0:
                raise ValueError(
                    f"Combination '{name}', stage '{stage['name']}': "
                    f"'time' must be non-negative, got {t:g}."
                )
            if prev_time is not None and t < prev_time:
                raise ValueError(
                    f"Combination '{name}', stage '{stage['name']}': "
                    f"'time' decreases ({prev_time:g} -> {t:g} days); "
                    f"stage times are cumulative since stage 0 and "
                    f"must be non-decreasing."
                )
            prev_time = t
            stage["time"] = t
        # Opaque per-stage reporting payload, echoed verbatim into the
        # per-stage result by the engines.  No solver-side effect.
        if "report" in s_spec:
            stage["report"] = s_spec["report"]
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
        different values (they are aliases — set only one).

    Notes
    -----
    A non-zero value is now **consumed** by the fiber integrator, which
    evaluates the bulk constitutive law at ``eps_section + bulk_eps_init``
    (batch, scalar and tangent sites), so the resistance domain reflects
    the offset.  The earlier *no-silent-no-op* guard that rejected any
    non-zero value — a stopgap for when the kernel ignored the offset —
    has been retired; the kernel consumption is validated end-to-end by
    ``run_bulk_prestrain_validation_new.py``.
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
        value = float(sec_spec["prestrain"])
    elif has_e:
        value = float(sec_spec["eps_init"])
    else:
        return 0.0

    return value


def _resolve_prestress_actions(combinations, section):
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
        Built section (supplies the reference point, the tendon
        coordinate arrays, and the tendon names for ``ref``
        resolution).

    Raises
    ------
    ValueError
        If an entry is malformed (see
        :func:`_resolve_single_prestress_action`) or tendon names are
        ambiguous (see :func:`_tendon_name_map`).

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
    name_map = _tendon_name_map(section)

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


def _tendon_name_map(section):
    r"""
    Map a tendon ``name`` (if set) to its index in the section's
    tendon list.

    Tendons are referenced by integer index by default; this allows the
    optional :attr:`~gensec.geometry.fiber.Tendon.name` to be used as a
    ``ref`` instead.  Reading from the **built** :class:`Tendon` objects
    (rather than the raw YAML spec) makes name resolution work
    identically for API-constructed sections.

    Parameters
    ----------
    section : GenericSection
        Built section.

    Returns
    -------
    dict
        ``{name: index}`` for every tendon with a non-``None`` name.

    Raises
    ------
    ValueError
        If two tendons share the same name (the reference would be
        ambiguous).
    """
    out = {}
    for i, t in enumerate(getattr(section, "tendons", [])):
        nm = getattr(t, "name", None)
        if nm is None:
            continue
        nm = str(nm)
        if nm in out:
            raise ValueError(
                f"section: duplicate tendon name '{nm}' (tendons "
                f"{out[nm]} and {i}) — names used as refs must be "
                f"unique."
            )
        out[nm] = i
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


# ---- Section-ops parsing + resolution (Phase-3 staged YAML) ----

#: Keys accepted in a stage's ``section_ops`` block.  Anything else is
#: a typo and raises (no-silent policy: a misspelled op must never be
#: silently dropped — it would change the model without telling).
_SECTION_OPS_KEYS = ("activate", "deactivate", "eps_override",
                     "bulk_eps", "release", "activate_bulk",
                     "deactivate_bulk", "bulk_plane_delta")


def _parse_section_ops_spec(combo_name, stage_name, ops):
    r"""
    Value-level validation of one stage's ``section_ops`` block.

    Runs at parse time (no section needed): structure, unknown keys,
    type checks.  Element
    references (union indices or names) are **not** resolved here —
    that needs the built section and happens in
    :func:`_resolve_section_ops`.

    Parameters
    ----------
    combo_name, stage_name : str
        For error messages.
    ops : dict
        Raw ``section_ops`` block: any of ``activate`` / ``deactivate``
        (lists of int index or str name), ``eps_override``
        (``{ref: eps}``), ``bulk_eps`` (float), ``release`` (bool).

    Returns
    -------
    dict
        The validated raw spec (same keys, lists/dicts shallow-copied).

    Raises
    ------
    ValueError
        Malformed block or unknown key.

    Notes
    -----
    ``bulk_eps`` is parsed and returned verbatim.  As of Phase 5 the
    fiber integrator **consumes** the offset (it evaluates the bulk
    constitutive law at ``eps_section + bulk_eps_init`` at the batch,
    scalar and tangent sites), so a non-zero value moves the resistance
    domain, not only the cache identity — exactly like the section-level
    ``prestrain`` / ``eps_init`` field (see :func:`_parse_bulk_prestrain`).
    The earlier *no-silent-no-op* rejection has been retired; the kernel
    consumption is validated by ``run_bulk_prestrain_validation_new.py``.
    """
    where = f"Combination '{combo_name}', stage '{stage_name}'"
    if not isinstance(ops, dict):
        raise ValueError(
            f"{where}: 'section_ops' must be a mapping, "
            f"got {type(ops).__name__}."
        )
    unknown = sorted(set(ops) - set(_SECTION_OPS_KEYS))
    if unknown:
        raise ValueError(
            f"{where}: unknown section_ops key(s) {unknown}. "
            f"Valid: {list(_SECTION_OPS_KEYS)}."
        )

    out = {}
    for key in ("activate", "deactivate"):
        if key in ops:
            val = ops[key]
            if not isinstance(val, list):
                raise ValueError(
                    f"{where}: section_ops '{key}' must be a list of "
                    f"element references (union index or name), got "
                    f"{type(val).__name__}."
                )
            out[key] = list(val)
    if "eps_override" in ops:
        val = ops["eps_override"]
        if not isinstance(val, dict):
            raise ValueError(
                f"{where}: section_ops 'eps_override' must be a "
                f"mapping {{element_ref: eps}}, got "
                f"{type(val).__name__}."
            )
        out["eps_override"] = dict(val)
    if "release" in ops:
        val = ops["release"]
        if not isinstance(val, bool):
            raise ValueError(
                f"{where}: section_ops 'release' must be a boolean, "
                f"got {val!r}."
            )
        out["release"] = val
    if "bulk_eps" in ops:
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


def _union_name_map(section):
    r"""
    Map an element ``name`` to its **union index** in the canonical
    ``rebars + tendons`` order of the section.

    This is the one reference mechanism of the staged YAML schema:
    the union index is the primitive (it is the index every
    :class:`~gensec.solver.section_state.SectionState` array is defined
    over), and a name — :attr:`~gensec.geometry.fiber.RebarLayer.name`
    or :attr:`~gensec.geometry.fiber.Tendon.name` — is an optional
    alias for it.  Same semantics and guards as the Step-A
    ``prestress_actions`` ``ref`` (:func:`_tendon_name_map`): reading
    from the **built** element objects makes resolution identical for
    API-constructed sections, and a duplicate name raises because the
    reference would be ambiguous.  The uniqueness scope here is the
    whole union set (a rebar and a tendon may not share a name), which
    is strictly wider than the tendon-only scope of
    :func:`_tendon_name_map` — necessarily so, since a ``section_ops``
    name may target either population.

    Parameters
    ----------
    section : GenericSection
        Built section.

    Returns
    -------
    dict
        ``{name: union_index}`` for every named element.

    Raises
    ------
    ValueError
        If two elements share the same name.
    """
    out = {}
    elements = list(getattr(section, "rebars", []))
    elements += list(getattr(section, "tendons", []))
    for i, el in enumerate(elements):
        nm = getattr(el, "name", None)
        if nm is None:
            continue
        nm = str(nm)
        if nm in out:
            raise ValueError(
                f"section: duplicate element name '{nm}' (union "
                f"elements {out[nm]} and {i}) — names used as refs "
                f"must be unique across the rebars + tendons union set."
            )
        out[nm] = i
    return out


def _resolve_element_id(eid, n_union, name_map, where):
    r"""
    Resolve one element reference to a union index.

    Parameters
    ----------
    eid : int or str
        Union index (integer position in the canonical
        ``rebars + tendons`` order) or element name.
    n_union : int
        Size of the union element set.
    name_map : dict
        ``{name: union_index}`` from :func:`_union_name_map`.
    where : str
        Context prefix for error messages.

    Returns
    -------
    int
        Union index in ``[0, n_union)``.

    Raises
    ------
    ValueError
        Unknown name, index out of range, or unsupported type
        (booleans are rejected explicitly: YAML ``true`` is a
        :class:`bool`, which is an :class:`int` subclass — accepting it
        as index 1 would mask an input error).
    """
    if isinstance(eid, str):
        if eid not in name_map:
            raise ValueError(
                f"{where}: element ref '{eid}' does not match any "
                f"rebar/tendon 'name'. Known: {sorted(name_map)}."
            )
        return name_map[eid]
    if isinstance(eid, bool) or not isinstance(eid, int):
        raise ValueError(
            f"{where}: element ref must be a union index (int) or an "
            f"element name (str), got {eid!r}."
        )
    if not (0 <= eid < n_union):
        raise ValueError(
            f"{where}: element ref index {eid} out of range (section "
            f"has {n_union} union element(s): rebars + tendons)."
        )
    return eid


def _resolve_section_ops(combinations, section):
    r"""
    Resolve raw ``section_ops`` specs into the form consumed by
    :meth:`~gensec.solver.section_state.StagedDomainManager.resolve_stages`.

    The exact counterpart of :func:`_resolve_prestress_actions` for the
    capacity side: walks every staged combination and replaces each
    stage's deferred ``_section_ops_spec`` (carried by
    :func:`_parse_combination`) with a resolved dict under the key
    ``section_ops`` — element names replaced by union indices,
    ``eps_override`` values coerced to float.  Mutates *combinations*
    in place; a no-op for combinations that declare none.

    Parameters
    ----------
    combinations : list of dict
        Parsed combinations (output of :func:`_parse_combination`).
    section : GenericSection
        Built section (supplies the union element count and the
        element names).

    Raises
    ------
    ValueError
        Unknown name, duplicate name, or out-of-range index (see
        :func:`_resolve_element_id` / :func:`_union_name_map`).

    Notes
    -----
    The union name map is built (and its duplicate guard enforced)
    **unconditionally**, mirroring the Step-A behaviour of
    :func:`_resolve_prestress_actions` /
    :func:`_tendon_name_map`: an ambiguous reference name is an input
    error at declaration, whether or not the current file happens to
    dereference it.
    """
    name_map = _union_name_map(section)
    n_union = (int(getattr(section, "x_rebars", np.empty(0)).size)
               + int(getattr(section, "x_tendons", np.empty(0)).size))

    for combo in combinations:
        if "stages" not in combo:
            continue
        for stage in combo["stages"]:
            spec = stage.pop("_section_ops_spec", None)
            if spec is None:
                continue
            where = (f"Combination '{combo['name']}', stage "
                     f"'{stage['name']}', section_ops")
            ops = {}
            for key in ("activate", "deactivate"):
                if key in spec:
                    ops[key] = [
                        _resolve_element_id(e, n_union, name_map,
                                            f"{where}.{key}")
                        for e in spec[key]
                    ]
            if "eps_override" in spec:
                ops["eps_override"] = {
                    _resolve_element_id(e, n_union, name_map,
                                        f"{where}.eps_override"):
                        float(v)
                    for e, v in spec["eps_override"].items()
                }
            if "release" in spec:
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


def staged_ops_present(combinations):
    r"""
    Whether any staged combination carries ``section_ops``.

    This is the single gate the engine-wiring layers (``cli``, ``api``)
    use to decide whether to build a
    :class:`~gensec.solver.section_state.StagedDomainManager`: when it
    returns ``False`` the engines are constructed exactly as before
    Phase 3 (no manager — the legacy capacity-frozen run is byte-
    identical); when ``True``, a manager is mandatory, because the
    engines refuse to run a ``section_ops``-carrying combination
    without one (no-silent-no-op: the ops would otherwise be dropped
    and the capacity silently frozen).

    Parameters
    ----------
    combinations : list of dict
        Parsed combinations (after :func:`_resolve_section_ops`).

    Returns
    -------
    bool
    """
    for combo in combinations:
        for stage in combo.get("stages", []):
            if "section_ops" in stage:
                return True
    return False


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
