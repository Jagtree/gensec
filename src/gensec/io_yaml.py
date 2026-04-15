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
from .geometry.fiber import RebarLayer
from .geometry.section import RectSection
from .geometry.geometry import GenericSection
from .geometry import primitives as prim
from .materials.ec2_bridge import concrete_from_class, concrete_from_ec2


# ---- Material builders (unchanged) ----

_MATERIAL_BUILDERS = {
    "concrete": {
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


def _build_material(name, spec):
    """
    Build a Material instance from a YAML specification dict.

    Supported types: ``concrete``, ``concrete_ec2``, ``steel``,
    ``tabulated``.

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

    if mat_type == "concrete_ec2":
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
                f"Material '{name}': concrete_ec2 requires 'class' "
                f"(e.g. 'C30/37') or 'fck'."
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
            f"Valid: {list(_MATERIAL_BUILDERS.keys())} + 'concrete_ec2'"
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
        materials[mat_name] = _build_material(mat_name, mat_spec)

    # ---- Section ----
    sec_spec = data["section"]
    bulk_name = sec_spec["bulk_material"]
    if bulk_name not in materials:
        raise ValueError(
            f"Bulk material '{bulk_name}' not found in materials."
        )

    # Parse rebars (common to both modes)
    rebars = _parse_rebars(sec_spec, materials)

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
        )

    # ---- Demands ----
    demands = [_parse_demand(d) for d in data.get("demands", [])]

    # ---- Combinations (v2.1: components / stages) ----
    combinations = [_parse_combination(c)
                    for c in data.get("combinations", [])]

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
            As=float(rb_spec["As"]),
            material=materials[mat_name],
            x=float(rb_spec["x"]) if "x" in rb_spec else None,
            embedded=bool(rb_spec.get("embedded", True)),
            n_bars=int(rb_spec.get("n_bars", 1)),
            diameter=float(rb_spec.get("diameter", 0)),
        ))
    return rebars


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
        return {
            "name": name,
            "components": _parse_component_list(c_spec["components"]),
        }

    # Staged.
    stages = []
    for i, s_spec in enumerate(c_spec["stages"]):
        stages.append({
            "name": s_spec.get("name", f"stage_{i}"),
            "components": _parse_component_list(
                s_spec.get("components", [])),
        })
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
    flags.setdefault("eta_3D", True)
    flags.setdefault("eta_2D", False)
    flags.setdefault("eta_path", True)
    flags.setdefault("eta_path_2D", False)
    flags.setdefault("delta_N_tol", 0.03)

    # Domain generation defaults.
    flags.setdefault("generate_mx_my", False)
    flags.setdefault("generate_3d_surface", False)
    flags.setdefault("n_angles_mx_my", 144)

    return flags
