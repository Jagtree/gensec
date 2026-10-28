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
from .materials.base import (
    DeclaredRule,
)
from .materials.codes import UNDECLARED, known_codes, resolve_code
from .materials.verification_limits import ec2_stress_limits
from .materials.ec2_bridge import (
    concrete_from_class, concrete_from_ec2,
    prestress_from_ec2, prestress_from_class,
)


# ---- Material builders (unchanged) ----

_MATERIAL_BUILDERS = {
    "concrete_ec2_gen1_custom": {
        "cls": Concrete,
        "params": ["fck", "gamma_c", "long_term_reduction_factor",
                    "n_parabola", "eps_c2", "eps_cu2", "fct", "Ec"],
        # Declared or refused.  These three multiply into f_cd, so a
        # value the input does not state can only come from a
        # container-side default -- which is exactly what this phase
        # took away.  Listed here rather than left to the dataclass so
        # the refusal can name the material and offer a remedy.
        "required": ["fck", "gamma_c", "long_term_reduction_factor"],
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

#: Input keys that name a normative clause instead of the neutral
#: container field, mapped to the field they are trying to reach.
#: Refused rather than silently accepted *or* silently ignored: an
#: ignored key is the worse of the two, because the author believes
#: they declared something.
_CLAUSE_NAMED_KEYS = {
    "alpha_cc": ("long_term_reduction_factor",
                 "EN 1992-1-1 §3.1.6(1)P"),
}

# Backward-compatible aliases for material type names.
_MATERIAL_ALIASES = {
    "concrete": "concrete_ec2_gen1_custom",
    "concrete_ec2": "concrete_ec2_gen1",
}


def _declarable():
    """
    The code names an input file may actually declare.

    ``undeclared`` is a registry entry because the resolver has to be
    able to *return* the absence of a declaration; it is not something
    a reader can write, so offering it in a refusal would be advice
    that does not work.

    Returns
    -------
    str
        Comma-separated, in registration order.
    """
    return ", ".join(k for k in known_codes() if k != "undeclared")


def _build_material(name, spec, model_code=None):
    """
    Build a Material instance from a YAML specification dict.

    The material is stamped with the :class:`NormativeCode` that
    governs it, resolved by :func:`~gensec.materials.codes.resolve_code`
    from the material's own ``code:``, the model's top-level ``code:``,
    or the material type when the type names one.  Everything code
    dependent -- the conventional modulus, the rheological provider --
    is derived from that one declaration rather than re-decided at each
    call site.

    Supported types: ``concrete_ec2_gen1_custom``,
    ``concrete_ec2_gen1``, ``steel``, ``tabulated``.
    Legacy aliases ``concrete`` and ``concrete_ec2`` are also
    accepted.

    Parameters
    ----------
    name : str
        Key used in the YAML ``materials`` block.
    spec : dict
        Must contain a ``'type'`` key.  An optional ``'code'`` key
        overrides the model-level declaration for this material alone.
    model_code : str or NormativeCode, optional
        The model's top-level ``code:`` declaration.

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

    # The normative declaration is resolved BEFORE the material is
    # built, and its refusal therefore comes first.  The order is not
    # cosmetic: a model that declares no code has nothing to say about
    # which document the material constants should have come from, so
    # complaining that a constant is missing would answer the second
    # question while the first is still open.  Phase 21 established
    # that this refusal is the outermost one; Phase 23 kept it there.
    code = resolve_code(material_spec=spec, model_code=model_code)
    if code is UNDECLARED:
        raise ValueError(
            f"material {name!r}: no normative code is declared. Add a\n"
            f"top-level 'code:' to the input file, or a 'code:' key to\n"
            f"this material, or declare 'code: custom' if the input\n"
            f"supplies every code-dependent value itself.\n"
            f"    Declarable codes: {_declarable()}\n"
            f"GenSec derives nothing from an undeclared model. The\n"
            f"material type is not read as a declaration: it names a\n"
            f"constitutive shape, not a verification code, and nothing\n"
            f"says a GenSec section is concrete at all.")

    material = _construct_material(name, spec, mat_type)
    # One home for the declaration.  Stamped here, on the way out of
    # the loader, so that no construction path can forget it and no
    # consumer has to re-resolve it.
    material.code = code
    return material


_LS_VALUES = ("F", "A", "QP", "C")
_LS_WHAT = ("It selects the limit state whose partial factor and "
            "alpha_cc apply; F gives gamma_c = 1.5 and QP gives 1.0, "
            "a fifty per cent difference in f_cd.")
_LOADTYPE_VALUES = ("slow", "fast")
_LOADTYPE_WHAT = ("It selects the loading rate the EN 1992 strength "
                  "development is evaluated for.")
_TYPECONC_VALUES = ("R", "N", "S")
_TYPECONC_WHAT = ("It selects the cement class of EN 1992-1-1 3.1.2, "
                  "which drives the strength development with age.")
_NA_WHAT = ("It selects the National Annex the partial factors come "
            "from.  Measured in Phase 37: the annex sets implemented "
            "for concrete and for prestressing steel are disjoint, so "
            "the two materials of one document declare different "
            "values today.")
_TIME_WHAT = ("It is the concrete age in days at which the EN 1992 "
              "properties are evaluated; 28 is a reference age, not a "
              "value GenSec may assume.")


def _declared(spec, key, name, what, admissible=None):
    r"""
    Read a normative parameter that the document must declare itself.

    GenSec derives nothing from an undeclared model.  A parameter that
    enters an EN 1992 expression is part of that declaration, so the
    loader reads it and never supplies it: a missing ``ls`` used to
    become ``"F"`` here, and the document that omitted it received
    :math:`\gamma_c = 1.5` and :math:`f_{cd} = 17.0` MPa where the same
    section under a quasi-permanent combination gets
    :math:`\gamma_c = 1.0` and :math:`f_{cd} = 25.5` MPa.  A fifty per
    cent difference in design strength is not a default's business.

    The refusal deliberately does **not** name a value to write.
    Listing the admissible set is naming the alternatives; suggesting
    one of them would be the silent default with an extra step.

    Parameters
    ----------
    spec : dict
        The material's YAML mapping.
    key : str
        The key that must be present.
    name : str
        Material name, for the message.
    what : str
        One clause saying what the parameter governs.
    admissible : sequence of str, optional
        The admissible values, listed in the message when the set is
        closed and short.  Omitted where the set is open (a time, a
        modulus).

    Returns
    -------
    object
        The declared value, unconverted.

    Raises
    ------
    ValueError
        When the key is absent.
    """
    if key not in spec:
        tail = ("" if admissible is None
                else f"\n    Admissible: {list(admissible)}")
        raise ValueError(
            f"material {name!r}: {key!r} is not declared, and GenSec\n"
            f"supplies no value for it. {what}{tail}")
    return spec[key]


def _construct_material(name, spec, mat_type):
    """
    Build the Material instance itself, unstamped.

    Split out of :func:`_build_material` so the code stamp has exactly
    one application point rather than one per ``return``.

    Parameters
    ----------
    name : str
    spec : dict
    mat_type : str
        Type name with aliases already resolved.

    Returns
    -------
    Material
    """

    if mat_type == "prestressing_steel_ec2":
        # EC2 §3.3 prestressing steel. Either a standard designation
        # ('class', e.g. 'Y1860S7') or explicit characteristic values
        # (f_p01k, f_pk, eps_uk). The partial factor derives from the
        # limit state / national annex, exactly as for concrete.
        common = dict(
            ls=_declared(spec, "ls", name,
                         "It selects the limit state whose partial "
                         "factor applies.",
                         admissible=("F", "A", "QP", "C")),
            NA=_declared(spec, "NA", name,
                         "It selects the National Annex the partial "
                         "factor and eps_ud come from.  Measured in "
                         "Phase 37: the annex sets implemented for "
                         "concrete and for prestressing steel are "
                         "disjoint, so the two materials of one "
                         "document declare different values today."),
            eps_ud_factor=float(
                _declared(spec, "eps_ud_factor", name,
                          "It scales eps_uk to the design ultimate "
                          "strain; EN 1992-1-1 recommends 0.9 and a "
                          "National Annex may not.")),
            gamma_s_override=spec.get("gamma_s_override"),
            diagram=_declared(spec, "diagram", name,
                              "It selects the top branch of the "
                              "design stress-strain diagram, EN "
                              "1992-1-1 3.3.6(7).",
                              admissible=("horizontal", "inclined")),
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
            Ep=float(_declared(spec, "Ep", name,
                               "It is the tendon modulus of "
                               "elasticity in MPa; EN 1992-1-1 3.3.6 "
                               "gives an indicative value only.")),
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
                ls=_declared(spec, "ls", name, _LS_WHAT,
                             admissible=_LS_VALUES),
                loadtype=_declared(spec, "loadtype", name, _LOADTYPE_WHAT,
                                   admissible=_LOADTYPE_VALUES),
                TypeConc=_declared(spec, "TypeConc", name, _TYPECONC_WHAT,
                                   admissible=_TYPECONC_VALUES),
                NA=_declared(spec, "NA", name, _NA_WHAT),
                time=_declared(spec, "time", name, _TIME_WHAT),
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
            ls=_declared(spec, "ls", name, _LS_WHAT,
                         admissible=_LS_VALUES),
            loadtype=_declared(spec, "loadtype", name, _LOADTYPE_WHAT,
                           admissible=_LOADTYPE_VALUES),
            TypeConc=_declared(spec, "TypeConc", name, _TYPECONC_WHAT,
                           admissible=_TYPECONC_VALUES),
            NA=_declared(spec, "NA", name, _NA_WHAT),
            time=_declared(spec, "time", name, _TIME_WHAT),
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

    # A key that names a normative clause is refused before anything
    # else, and told which neutral field it was aiming at.  Letting it
    # through unread would be a silent fallback in the most dangerous
    # direction: the input states a value, GenSec ignores it, and the
    # author has no way to know.
    for clause_key, (field, where) in _CLAUSE_NAMED_KEYS.items():
        if clause_key in spec:
            raise ValueError(
                f"material {name!r}: {clause_key!r} is the symbol "
                f"{where} gives its own value, and it names a clause "
                f"rather than a quantity. GenSec is normatively "
                f"agnostic, so the container field is called "
                f"{field!r}. Rename the key. If the model is declared "
                f"under that document, the value the clause recommends "
                f"is what belongs there; under any other document, "
                f"supply that document's value, or 1.0 as an explicit "
                f"proxy when it applies no material-side reduction.")

    kwargs = {}
    for p in builder["params"]:
        if p in spec:
            val = spec[p]
            if isinstance(val, list):
                val = np.array(val, dtype=float)
            kwargs[p] = val

    missing = [p for p in builder.get("required", ()) if p not in kwargs]
    if missing:
        raise ValueError(
            f"material {name!r} of type {mat_type!r}: "
            f"{', '.join(repr(m) for m in missing)} "
            f"{'is' if len(missing) == 1 else 'are'} not declared. "
            f"GenSec derives no material constant it was not given: "
            f"these values enter f_cd = k_lt * fck / gamma_c and a "
            f"number the input never stated could only have come from "
            f"a default in the container, which no longer exists.\n"
            f"    Declare them on the material, or build it through a "
            f"code's own module -- 'type: concrete_ec2_gen1' with a "
            f"'class:' (e.g. 'C30/37') reads all three off "
            f"EN 1992-1-1 and is the intended route for a Eurocode "
            f"model.")
    return cls(**kwargs)


# ---- Shape factory dispatch ----

def _declared_segments_per_quadrant(spec, shape):
    r"""
    Read the mandatory discretisation of a curved boundary.

    A parameter that is mandatory in
    :mod:`gensec.geometry.primitives` and defaulted in the loader is
    not mandatory, so the refusal is repeated here rather than left to
    a default that would never reach the library.

    Parameters
    ----------
    spec : dict
        The shape block as written in the YAML file.
    shape : str
        The shape name, for the message.

    Returns
    -------
    int
        The declared number of straight segments per quadrant.

    Raises
    ------
    ValueError
        If the key is absent.  The message states the magnitude the
        choice governs, because a user who has never had to make it
        needs to know what they are choosing.
    """
    if "resolution" in spec:
        raise ValueError(
            f"shape '{shape}': 'resolution' was renamed to "
            f"'segments_per_quadrant' because it counts segments per "
            f"quadrant, not vertices on the boundary: the polygon "
            f"receives 4 x segments_per_quadrant vertices.  Rename the "
            f"key; the value keeps its meaning."
        )
    if "segments_per_quadrant" not in spec:
        raise ValueError(
            f"shape '{shape}': 'segments_per_quadrant' must be "
            f"declared.  It selects which section is analysed, not how "
            f"finely a fixed one is integrated: the boundary becomes a "
            f"polygon with 4 x segments_per_quadrant vertices, whose "
            f"sides depart from the curve by R*(1-cos(pi/n)) with "
            f"n = 4 x segments_per_quadrant.  GenSec does not choose "
            f"this for you."
        )
    return spec["segments_per_quadrant"]



_SHAPE_FACTORIES = {
    "rect": lambda p: prim.rect_poly(p["B"], p["H"]),
    "circle": lambda p: prim.circle_poly(
        p["D"],
        segments_per_quadrant=_declared_segments_per_quadrant(p, "circle")),
    "annulus": lambda p: prim.annulus_poly(
        p["D_ext"], p["D_int"],
        segments_per_quadrant=_declared_segments_per_quadrant(p, "annulus")),
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

    # ---- Z-7: the top-level gate --------------------------------------
    # First, before the schema gate and before anything is built.  A
    # document with a misspelt block should hear about the misspelling,
    # not about a schema contract it may satisfy perfectly.
    _check_unknown_top_level(data, filepath)

    # ---- D7: the versioned-schema gate --------------------------------
    # Refused here, before anything is built, so that a document whose
    # contract is unknown never reaches a solve.
    _check_schema_version(data, filepath)

    # ---- Materials ----
    materials = {}
    # The model's normative declaration, read once.  Everything code
    # dependent derives from it; a material may override it for itself.
    model_code = data.get("code")
    for mat_name, mat_spec in data.get("materials", {}).items():
        mat = _build_material(mat_name, mat_spec, model_code=model_code)
        mat.name = mat_name
        materials[mat_name] = mat

    # ---- Section ----
    sec_spec = data["section"]
    # Z-17: until Phase 36 a key nobody reads inside 'section:' loaded
    # in silence.  Refused before anything is built, for the same
    # reason the top level is: a document with a misspelt key should
    # hear about the misspelling, not receive a verdict on a model its
    # author did not write.  Denominator measured, not assumed -- 15
    # keys read over 20 AST sites, 12 written by the 24 distributed
    # examples, all inside the set.
    _refuse_unknown_keys(sec_spec, where="section",
                         valid=VALID_SECTION_KEYS)
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
            _refuse_unknown_keys(
                zone_spec, where=f"section.material_zones[{iz}]",
                valid=_zone_keys)
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
    combinations = [_parse_combination(c, model_code=model_code)
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

    # Phase-12 / P3: the rheological identity of a zone, declared once,
    # outside any interval.  Absent -> {} and nothing changes.
    zone_rheology = _parse_zone_rheology(data.get("zone_rheology"),
                                         losses_models)

    # Phase-12 / P3: the explicit time-axis switch.  Validated here, at
    # parse, so a misspelt value is refused before any solve rather
    # than falling back to the default and quietly re-enabling a
    # rheology the model asked to turn off.
    rheology = _normalise_rheology_switch(data.get("rheology"))

    # Numerical tolerances.  Resolved input-document -> configuration
    # file -> shipped default, so a model states what it can and a
    # harness states what no document can name.  Unknown keys raise:
    # an override that is silently ignored is a silent error.
    from .tolerances import Tolerances, load_tolerance_file
    tolerances = Tolerances.from_mapping(data.get("tolerances"),
                                         base=load_tolerance_file())

    return {
        "materials": materials,
        "section": section,
        "demands": demands,
        "combinations": combinations,
        "envelopes": envelopes,
        "output_options": output_opts,
        "construction_history": construction_history,
        "losses_models": losses_models,
        "zone_rheology": zone_rheology,
        "rheology": rheology,
        "tolerances": tolerances,
    }


#: Top-level blocks that may not be written without a declared
#: ``schema_version`` (D7).
#:
#: The three of them, and not every block, because these are the only
#: ones that carry **hand-written numbers that are time-dependent or a
#: matter of judgement**: a zone's declared age, a numerical tolerance,
#: an engineer's rule extending a provider past its stated domain.  All
#: three travel into the machine-readable dump *as provenance*.  If the
#: document that carries them declares no contract, then what the dump
#: records is the provenance of an unknown format — which is not
#: provenance at all.
#:
#: ``materials``, ``section`` and ``demands`` describe objects whose
#: interpretation does not drift with time, so they are not gated: D7
#: is not a tax on the past.
VERSION_GATED_KEYS = ("zone_rheology", "tolerances", "declared")

#: Document-schema MAJOR this build reads.
DOCUMENT_SCHEMA_MAJOR = 1


#: Every top-level key :func:`load_yaml` reads, and the **only** ones it
#: accepts (Phase 32, Task A / finding Z-7).
#:
#: Why this constant exists
#: ------------------------
#: ``section.material_zones`` refuses an unknown key by name, citing the
#: fail-loud policy in its own comment.  The **top level** of the same
#: document applied no such policy: these thirteen keys were read with
#: scattered ``data.get(...)`` calls and anything else fell on the floor
#: without a word.  Measured on the incoming tree, three injections into
#: a distributed example, all three accepted in silence -- including
#: ``custom_zone_strengths``, the name ratified for the Phase-32
#: per-zone strengths.  Before this refusal, writing a ratified block
#: name in a document was indistinguishable from writing a typo.
#:
#: Why a constant and not a second list
#: ------------------------------------
#: The refusal needs the list, and a refusal carrying its own copy
#: diverges from the readers the first time a block is added.  One home
#: for one fact: this tuple is the home, ``_check_unknown_top_level``
#: is its only consumer, and ``tests/test_phase32_top_level_keys.py``
#: checks it against the readers in this very file so that the two
#: cannot drift apart in silence.
#:
#: Adding a key here is a **decision**, not a side effect: a fixture
#: pins the membership, so a fourteenth block arrives with a diff that
#: says so.
VALID_TOP_LEVEL_KEYS = (
    "code",
    "materials",
    "section",
    "demands",
    "combinations",
    "envelopes",
    "output",
    "construction_history",
    "losses_models",
    "zone_rheology",
    "rheology",
    "tolerances",
    "schema_version",
)


#: Every key :func:`_parse_section` and its helpers read from a
#: ``section:`` block, and the **only** ones it accepts (Phase 36,
#: finding Z-17).
#:
#: Measured on the incoming tree by AST walk over ``sec_spec``: **20
#: read sites, 15 distinct keys** -- ``load_yaml`` 10,
#: ``_parse_bulk_prestrain`` 6, ``_build_polygon`` 2, ``_parse_rebars``
#: 1, ``_parse_tendons`` 1.  The 24 distributed examples write 12 of
#: them, all inside this set, so the refusal breaks no shipped
#: document; ``tests/test_phase36_unknown_keys.py`` holds that fact
#: still.
#:
#: ``n_fibers`` is the legacy alias of ``n_fibers_y`` and stays in the
#: set because the parser honours it.  It is an alias, not a fallback:
#: it names the same quantity rather than supplying one nobody wrote.
VALID_SECTION_KEYS = (
    "shape",
    "params",
    "B",
    "H",
    "bulk_material",
    "mesh_method",
    "mesh_size",
    "n_fibers",
    "n_fibers_x",
    "n_fibers_y",
    "material_zones",
    "rebars",
    "tendons",
    "prestrain",
    "eps_init",
)

#: Every key :func:`_parse_combination` reads at the **combination**
#: level (Phase 36, finding Z-15).
#:
#: ``prestress_actions``, ``section_ops``, ``time`` and ``report`` are
#: in this set although they are *refused* at this level: they are
#: known keys in the wrong place, and the refusal that says so is more
#: useful than one that prints the valid set.  Leaving them out would
#: have replaced a good diagnosis with a worse one -- the one way this
#: repair could have made a message worse instead of better, and the
#: reason a fixture pins it.
VALID_COMBINATION_KEYS = (
    "name",
    "components",
    "stages",
    "sls",
    "at",
    "history_factors",
    "gamma_P",
    "prestress_actions",
    "section_ops",
    "time",
    "report",
)

#: Every key :func:`_parse_combination` reads from a **stage**
#: (Phase 36, finding Z-15).
#:
#: ``_prestress_action_specs`` and ``_section_ops_spec`` are
#: deliberately **absent**.  The parser injects them into the stage it
#: *returns*; it never reads them from the stage it is *given*, and the
#: check below reads the given one.  Naming them here would advertise
#: two private names as part of the document schema.
VALID_STAGE_KEYS = (
    "name",
    "components",
    "prestress_actions",
    "section_ops",
    "time",
    "report",
)


def _refuse_unknown_keys(mapping, *, where, valid, hint="") -> None:
    r"""
    Refuse a mapping carrying a key nobody reads.

    Why one helper and not one check per block
    ------------------------------------------
    Until Phase 36 this file carried **six** hand-written unknown-key
    refusals -- top level, ``section.material_zones``,
    ``zone_rheology``, ``losses_models[...].declared``, ``sls`` and
    ``sls.limits`` -- each with its own wording, its own ordering of
    the two lists, and its own decision about whether to name the
    block.  Six cases of one mechanism are six cases to maintain, and
    they had already drifted: some sorted the valid set, some printed
    it in declaration order; some said ``Known:``, some ``Valid:``,
    some ``Recognised:``.  A reader who has learnt to recognise one
    refusal does not recognise the next.

    Why refusing at all
    -------------------
    A key outside *valid* is not a harmless extra: it is a declaration
    the engine will not act on, and the writer has no way to find that
    out.  The document is verified anyway and reports a verdict on a
    model its author did not write.  A silent fallback is outside the
    filter of ``00_BOOTSTRAP`` §0.4, so there is no "warn instead"
    variant of this function.

    Parameters
    ----------
    mapping : object
        The block to check.  A non-mapping is left alone: it has no
        keys at all, and turning that into a report about unknown keys
        would replace one confusing failure with another.  Callers that
        require a mapping refuse it for their own reason, first.
    where : str
        The path of the block as a document writer would name it --
        ``"section"``, ``"zone_rheology['deck']"``, ``"Combination
        'SLU', stage 'casting'"``.  Named because a project loads many
        documents and a block name without a path sends the reader into
        the source.
    valid : sequence of str
        Every key the block's readers honour.  Printed sorted, so that
        two refusals from different blocks read the same way.
    hint : str, optional
        One sentence appended verbatim, for a block whose typical
        mistake has a known cure (the top level's "it may belong one
        level deeper").  Empty by default: a hint that fits every block
        fits none.

    Raises
    ------
    ValueError
        If any key of *mapping* is outside *valid*.  The message names
        **both** the offending keys and the valid set: the first tells
        the reader what went wrong, the second what to write instead.

    Examples
    --------
    >>> _refuse_unknown_keys({"shape": "rect"}, where="section",
    ...                      valid=("shape", "params"))
    >>> _refuse_unknown_keys({"shpae": "rect"}, where="section",
    ...                      valid=("shape", "params"))
    Traceback (most recent call last):
        ...
    ValueError: section: unknown key(s) ['shpae'].  Valid: ...
    """
    if not isinstance(mapping, dict):
        return
    unknown = sorted(str(k) for k in mapping if k not in valid)
    if not unknown:
        return
    raise ValueError(
        f"{where}: unknown key(s) {unknown}.  "
        f"Valid: {sorted(str(v) for v in valid)}.  "
        f"An unrecognised key is refused rather than ignored: it "
        f"carries an intent the engine would never act on, and a "
        f"document that declares something and is verified anyway "
        f"reports a verdict on a model its author did not write."
        + (f"  {hint}" if hint else "")
    )


def _check_unknown_top_level(data, filepath) -> None:
    r"""
    Refuse a document carrying a top-level key nobody reads.

    A key outside :data:`VALID_TOP_LEVEL_KEYS` is not a harmless extra:
    it is a declaration the engine will not act on, and the writer has
    no way to find that out.  Every intent expressed by such a key --
    a limit, a strength, a rheological model -- is silently discarded,
    and the run completes and reports a verdict.

    Parameters
    ----------
    data : object
        The parsed document.  Non-mappings are left alone: they have no
        top-level keys at all, and turning that into a report about
        unknown keys would replace one confusing failure with another.
        They fall through to the readers below, which already refuse
        them for their own reason.
    filepath : str or pathlib.Path
        Named in the refusal, because a project loads many documents.

    Raises
    ------
    ValueError
        If any top-level key is outside :data:`VALID_TOP_LEVEL_KEYS`.
        The message names **both** the offending keys and the valid
        set: naming only the first tells the reader what went wrong,
        naming the second tells them what to write instead, and a
        message with only the first sends them into the source.
    """
    if not isinstance(data, dict):
        return
    _refuse_unknown_keys(
        data, where=f"{filepath}, top level",
        valid=VALID_TOP_LEVEL_KEYS,
        hint="Check the spelling, or the block belongs one level "
             "deeper (per-zone rheology goes in 'zone_rheology', "
             "per-stage quantities go on a combination's stages).")


def _declared_block_present(data) -> bool:
    r"""
    Whether any ``losses_models`` entry carries a ``declared`` block.

    ``declared`` is nested one level down, unlike its two siblings, so
    it needs its own probe rather than a top-level key test.

    Parameters
    ----------
    data : dict

    Returns
    -------
    bool
    """
    for spec in (data.get("losses_models") or {}).values():
        if isinstance(spec, dict) and spec.get("declared"):
            return True
    return False


def _check_schema_version(data, filepath) -> None:
    r"""
    Enforce D7: a gated block requires a declared ``schema_version``.

    Parameters
    ----------
    data : dict
        The raw parsed document.
    filepath : str
        Named in the message, because a project has many models and
        the one that is missing a version is the useful fact.

    Raises
    ------
    ValueError
        A gated block is present without a version, the version is
        malformed, or its MAJOR is one this build does not read.

    Notes
    -----
    The rule is stated in ``handoff/13_0-GENSEC_DUMP_SCHEMA.md`` §1.1
    and is the input-document half of the same contract the dump obeys
    (:func:`gensec.output.timeline_dump.load_dump`).  One number, two
    gates, so a run cannot be half-versioned.
    """
    if not isinstance(data, dict):
        return
    gated = [k for k in VERSION_GATED_KEYS if data.get(k)]
    if _declared_block_present(data) and "declared" not in gated:
        gated.append("declared")
    version = data.get("schema_version")

    if version is None:
        if not gated:
            return
        raise ValueError(
            f"{filepath}: the document declares "
            f"{sorted(gated)} but no 'schema_version'.  These blocks "
            f"carry hand-written numbers that are time-dependent or a "
            f"matter of judgement -- a declared age, a numerical "
            f"tolerance, a rule extending a provider past its stated "
            f"domain -- and all three are serialised into the "
            f"machine-readable dump as provenance.  Provenance recorded "
            f"under an unstated contract is not provenance.  Add "
            f"'schema_version: {DOCUMENT_SCHEMA_MAJOR}' at the top "
            f"level.")

    try:
        major = int(str(version).split(".")[0])
    except (TypeError, ValueError):
        raise ValueError(
            f"{filepath}: 'schema_version' must be an integer MAJOR or "
            f"'MAJOR.MINOR', got {version!r}.") from None
    if major != DOCUMENT_SCHEMA_MAJOR:
        raise ValueError(
            f"{filepath}: this build reads document schema MAJOR "
            f"{DOCUMENT_SCHEMA_MAJOR}, the document declares {major} "
            f"(schema_version: {version!r}).  A MAJOR bump means a key "
            f"was removed, renamed or changed meaning, so reading it "
            f"with this build would be wrong rather than incomplete.")


def _normalise_rheology_switch(value):
    r"""
    Validate the top-level ``rheology`` switch at parse time.

    Delegates to
    :func:`gensec.solver.timeline._normalise_rheology_switch`, which is
    the rule's one home; this wrapper exists only to place the refusal
    at parse rather than at solve.

    Parameters
    ----------
    value : None or bool or str

    Returns
    -------
    str
        ``'declared'``, ``'on'`` or ``'off'``.

    Raises
    ------
    ValueError
        On an unrecognised value.
    """
    from .solver.timeline import _normalise_rheology_switch as _norm
    return _norm(value)


def _parse_zone_rheology(block, losses_models):
    r"""
    Parse the top-level ``zone_rheology`` block.

    A zone's rheological identity is a property of the **zone**: which
    model describes its concrete, how old that concrete is when the
    timeline first sees the zone, and when its curing ended.  Before
    Phase-12 that fact was declared inside ``interval.losses``, which
    made it unreachable to every event that is not an interval --- so a
    ``cast`` had no way to know the substrate's :math:`E_c(t)` and drew
    its datum on the design law instead (12_0 §1).  This block is its
    one home::

        zone_rheology:
          base:    {model: rheo_precast, age: 28, curing: 3}
          topping: {model: rheo_topping, age: 0,  curing: 1}

    ``age`` is the age of the zone **when the timeline first sees it**:
    at :math:`t = 0` for a zone that is there from the start, at its own
    ``cast`` event for a zone that is cast (normally ``0``; non-zero for
    a precast element placed some days after it was poured).  Every
    later age is derived by the walk --- an age is declared once, then
    derived, and a second declaration that disagrees is a modelling
    error, not a re-declaration.

    ``curing`` is the age at which curing ended, from which drying
    shrinkage is measured.  Optional here; a zone that never appears in
    an ``interval.losses`` never needs it, because nothing integrates
    shrinkage over an interval that does not exist.

    Parameters
    ----------
    block : dict or None
        The raw ``zone_rheology`` mapping, ``{zone name: spec}``.
    losses_models : dict
        The parsed ``losses_models``, used to check that every ``model``
        reference resolves.  A dangling reference is refused here rather
        than at the first interval that happens to use it.

    Returns
    -------
    dict
        ``{zone name: {"model": str, "age": float,
        "curing": float or None}}``.  Empty when *block* is absent.

    Raises
    ------
    ValueError
        Malformed block; missing or dangling ``model``; missing or
        negative ``age``.

    Examples
    --------
    >>> _parse_zone_rheology({"base": {"model": "m", "age": 28}},
    ...                      {"m": object()})["base"]["age"]
    28.0
    """
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(
            f"'zone_rheology' must be a mapping {{zone: spec}}, got "
            f"{type(block).__name__}."
        )
    out = {}
    for zone, spec in block.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"zone_rheology['{zone}'] must be a mapping, got "
                f"{type(spec).__name__}."
            )
        model = spec.get("model")
        if model is None:
            raise ValueError(
                f"zone_rheology['{zone}'] declares no 'model'.  The block "
                f"exists to say which rheological model describes the "
                f"zone; without one there is nothing to declare."
            )
        if model not in losses_models:
            raise ValueError(
                f"zone_rheology['{zone}'] references unknown model "
                f"'{model}'.  Declared in 'losses_models': "
                f"{sorted(losses_models)}."
            )
        if "age" not in spec:
            raise ValueError(
                f"zone_rheology['{zone}'] declares no 'age'.  E_c(t), "
                f"creep and shrinkage are all functions of how old the "
                f"concrete is; the timeline cannot know, and assuming "
                f"would be a silent normative choice."
            )
        age = float(spec["age"])
        if age < 0.0:
            raise ValueError(
                f"zone_rheology['{zone}']: age {age:g} d is negative.  "
                f"Concrete does not exist before it is cast."
            )
        curing = spec.get("curing")
        _refuse_unknown_keys(
            spec, where=f"zone_rheology['{zone}']",
            valid=("model", "age", "curing"))
        out[str(zone)] = {"model": str(model), "age": age,
                          "curing": None if curing is None
                          else float(curing)}
    return out


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
        declared = _parse_declared_rules(name, pkwargs.pop("declared", None))
        try:
            provider = _RHEO_PROVIDERS[pname](name=name, **pkwargs)
        except TypeError as exc:
            raise ValueError(
                f"losses_models['{name}']: the '{pname}' provider does "
                f"not accept these keys ({sorted(pkwargs)}) -- {exc}."
            ) from exc
        if declared:
            provider = provider.with_declared(**declared)
        ckwargs = {k: spec[k] for k in _LOSS_CONTAINER_KEYS
                   if k in spec and k != "provider"}
        out[name] = LossModel(provider=provider, name=name, **ckwargs)
    return out


def _parse_declared_rules(model_name, block):
    r"""
    Parse a provider's ``declared`` block into
    :class:`~gensec.materials.base.DeclaredRule` objects.

    Every provider accepts this block, whichever norm it implements:
    it is the single, provider-agnostic way an engineer extends a
    source beyond the ages that source states.  A rule taken from a
    national annex, a later code, a paper or a lab campaign enters
    here identically.

    .. code-block:: yaml

        losses_models:
          rheo_topping:
            provider: ec2
            fck: 30
            declared:
              linearity_limit:
                value:  1.0                 # MPa, or a list of pairs
                domain: [0.0, 3.0]          # days
                source: >
                  Project judgement outside EN 1992-1-1 §3.1.2(5).

    ``value`` is either a scalar or a list of ``[t, value]`` pairs,
    which is linearly interpolated in *t* and held at the ends — a
    research curve or a measured dataset needs no code to enter.

    Parameters
    ----------
    model_name : str
        For the error messages.
    block : dict or None
        The raw ``declared`` mapping.

    Returns
    -------
    dict
        ``{quantity: DeclaredRule}``, empty when *block* is falsy.

    Raises
    ------
    ValueError
        Malformed block, or a rule with no ``source``.

    Notes
    -----
    ``source`` is mandatory by design.  A number whose provenance is
    not recorded cannot be told apart from one GenSec invented, and
    the point of the whole mechanism is that GenSec never invents one.
    """
    if not block:
        return {}
    if not isinstance(block, dict):
        raise ValueError(
            f"losses_models['{model_name}']: 'declared' must be a "
            f"mapping {{quantity: rule}}, got {type(block).__name__}."
        )
    out = {}
    for quantity, spec in block.items():
        if not isinstance(spec, dict):
            raise ValueError(
                f"losses_models['{model_name}'].declared['{quantity}'] "
                f"must be a mapping with 'value', 'source' and an "
                f"optional 'domain', got {type(spec).__name__}."
            )
        _refuse_unknown_keys(
            spec,
            where=f"losses_models['{model_name}']"
                  f".declared['{quantity}']",
            valid=("value", "source", "domain"))
        if "value" not in spec:
            raise ValueError(
                f"losses_models['{model_name}'].declared['{quantity}'] "
                f"has no 'value'."
            )
        raw = spec["value"]
        if isinstance(raw, (list, tuple)):
            pairs = [(float(a), float(b)) for a, b in raw]
            if len(pairs) < 2:
                raise ValueError(
                    f"losses_models['{model_name}']."
                    f"declared['{quantity}']: a tabulated value needs "
                    f"at least two [t, value] pairs, got {len(pairs)}."
                )
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            if any(b <= a for a, b in zip(xs, xs[1:])):
                raise ValueError(
                    f"losses_models['{model_name}']."
                    f"declared['{quantity}']: the t column must be "
                    f"strictly increasing, got {xs}."
                )
            value = _interp_rule(xs, ys)
        else:
            value = float(raw)
        out[quantity] = DeclaredRule(
            value=value, source=spec.get("source"),
            domain=spec.get("domain"))
    return out


def _interp_rule(xs, ys):
    r"""
    Build ``f(t)`` interpolating *ys* over *xs*, holding the ends.

    Parameters
    ----------
    xs, ys : list of float

    Returns
    -------
    callable
    """
    def rule(t):
        return float(np.interp(float(t), xs, ys))
    rule.__qualname__ = f"declared_table[{len(xs)}]"
    return rule


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

#: Stress-limit providers reachable from a YAML ``sls.limits`` block,
#: keyed by the normative code that owns the clauses.  A norm enters
#: GenSec by adding one line here and one provider in
#: :mod:`gensec.materials.verification_limits` -- nothing else moves.
#:
#: ``ntc18`` maps to the EN 1992-1-1 provider because NTC 2018 adopts
#: those clauses verbatim and supplies its own coefficients through the
#: same ``overrides`` hook.  This is a wiring statement made in the
#: wiring layer; neither provider names the other.
#:
#: ``aci318-19`` is deliberately absent, as it is in
#: :data:`_RHEO_PROVIDERS`: there is no ACI stress-limit provider, and
#: a document declaring that code is refused by name rather than served
#: with another document's coefficients.
_LIMIT_PROVIDERS = {
    "ec2": ec2_stress_limits,
    "ntc18": ec2_stress_limits,
}

#: The two facts an ``sls`` block on a combination carries.  One home
#: for the membership, so a third sub-block arrives as a decision and
#: not as a side effect (the rule Phase 32 established for
#: :data:`VALID_TOP_LEVEL_KEYS`).
VALID_SLS_KEYS = ("moduli", "limits")

#: The fields of ``sls.limits``.  Anything else is refused by name.
VALID_SLS_LIMIT_KEYS = (
    "code", "kind", "fyk", "fpk", "decompression",
    "custom_zone_strengths", "overrides",
)


def _parse_sls_moduli(combo_name, block):
    r"""
    Parse the per-zone SLS moduli of an ``sls`` block.

    Why the moduli live on the combination
    --------------------------------------
    An SLS modulus is not a property of the zone alone.  The effective
    modulus of the same concrete is

    .. math::

        E_{c,\mathrm{eff}} = \frac{E_{cm}}{1 + \varphi(t, t_0)}

    under a quasi-permanent combination and :math:`E_{cm}` under a
    characteristic one -- two numbers for one zone, and what
    distinguishes them is *which verification is being performed*.
    That is the combination.  Declaring them on the zone would put a
    combination-dependent quantity on the material axis, where it
    cannot be correct for more than one combination at a time.

    GenSec derives neither (Phase-7 D3,
    :class:`~gensec.solver.sls.SlsPolicy`): a modulus is a normative,
    time-dependent choice, and the engine never reaches into a
    normative bridge on its own.  The caller computes it -- through the
    concrete's ``ec2`` bridge with its ``time`` argument, or otherwise
    -- and declares it here.

    Parameters
    ----------
    combo_name : str
        Named in every refusal, because a document holds many
        combinations.
    block : dict
        ``{zone name: modulus [MPa]}``.

    Returns
    -------
    dict
        ``{zone name: float}``, ready to be keyed as
        ``('zone', name)`` by the driver.

    Raises
    ------
    ValueError
        If the block is not a mapping, or a value is not a finite,
        strictly positive modulus.

    Notes
    -----
    A zone name that matches nothing is **not** checked here.  The
    section does not exist yet at parse time, and
    :func:`gensec.solver.sls.resolve_moduli` already refuses a dead
    override by name -- *"the caller believes a modulus is in force
    when it is not"*.  Checking it twice would be a second home for one
    rule, and the two copies would drift.
    """
    if not isinstance(block, dict):
        raise ValueError(
            f"Combination '{combo_name}': 'sls.moduli' must be a "
            f"mapping {{zone name: modulus [MPa]}}, got "
            f"{type(block).__name__}."
        )
    if not block:
        raise ValueError(
            f"Combination '{combo_name}': 'sls.moduli' is empty.  Every "
            f"bulk zone needs an SLS modulus: GenSec derives none, "
            f"because E_c,eff is a normative and time-dependent choice "
            f"(Phase-7 D3).  An empty mapping declares nothing and the "
            f"engine would refuse one zone at a time instead of saying "
            f"so here."
        )
    out = {}
    for zone, value in block.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Combination '{combo_name}': sls.moduli['{zone}'] "
                f"must be a number [MPa], got {value!r}."
            )
        if not np.isfinite(v) or v <= 0.0:
            raise ValueError(
                f"Combination '{combo_name}': sls.moduli['{zone}'] "
                f"must be a finite, strictly positive modulus [MPa]; "
                f"got {value!r}."
            )
        out[str(zone)] = v
    return out


def _parse_stress_limits(combo_name, block, model_code):
    r"""
    Build a :class:`~gensec.materials.verification_limits.StressLimits`
    from an ``sls.limits`` block.

    The block declares which normative document owns the coefficients
    and which family of limits is wanted; the provider for that
    document turns the pair into the container.  Nothing is assumed:

    .. code-block:: yaml

        sls:
          moduli: {base: 36000, topping: 31000}
          limits:
            kind: characteristic          # mandatory, no default
            fyk: 450.0
            fpk: 1860.0
            decompression: true
            custom_zone_strengths:
              base: {f_ck: 28.0}          # f_ck(t) at transfer lives here
            overrides: {k1: 0.6}          # national annex / NTC hook

    Which code governs
    ------------------
    ``code`` on the block, else the model's top-level ``code``, and
    **no third source** -- the same precedence
    :func:`~gensec.materials.codes.resolve_code` applies to a material,
    reused rather than restated.  A document that declares neither is
    refused: a stress-limit coefficient without a document behind it is
    a number with no authority, and picking one would be the silent
    normative choice this project removes.

    Parameters
    ----------
    combo_name : str
        Named in every refusal.
    block : dict
        The raw ``sls.limits`` mapping.
    model_code : str or None
        The document's top-level ``code``, or ``None``.

    Returns
    -------
    StressLimits

    Raises
    ------
    ValueError
        If the block is not a mapping; if a field is outside
        :data:`VALID_SLS_LIMIT_KEYS`; if no ``kind`` is declared; if
        no code resolves, or the code that resolves has no stress-limit
        provider; or if the provider itself refuses (an unknown
        ``kind``, an unknown override, ``kind='transfer'`` without
        declared strengths).

    Notes
    -----
    ``kind='transfer'`` reaches the provider unchanged and is refused
    there without ``custom_zone_strengths``.  That refusal is **not**
    worked around here: at transfer the governing strength is
    :math:`f_{ck}(t)` at the age the tendons are released, which no
    zone material knows, and deriving it from the 28-day value would
    raise the limit above what the concrete carries on the day it is
    loaded -- a silent fallback in the unsafe direction.  The YAML
    surface is a way *to* the provider, never a way *around* it.
    """
    if not isinstance(block, dict):
        raise ValueError(
            f"Combination '{combo_name}': 'sls.limits' must be a "
            f"mapping, got {type(block).__name__}."
        )
    _refuse_unknown_keys(
        block, where=f"Combination '{combo_name}', sls.limits",
        valid=VALID_SLS_LIMIT_KEYS,
        hint="Accepting a field nothing reads would leave the "
             "verification running on the provider's values while the "
             "document says otherwise.")
    if "kind" not in block:
        raise ValueError(
            f"Combination '{combo_name}': sls.limits declares no "
            f"'kind'.  The kind selects which family of limits the "
            f"governing document prescribes -- at transfer, at the "
            f"characteristic combination, at the quasi-permanent one "
            f"-- and they are different coefficients on different "
            f"strengths.  There is no default: guessing one would "
            f"report a verdict against limits the document never "
            f"asked for."
        )
    code = resolve_code(material_spec=block, model_code=model_code)
    if code is UNDECLARED:
        raise ValueError(
            f"Combination '{combo_name}': sls.limits resolves to no "
            f"normative code.  Declare 'code' on the block or at the "
            f"top level of the document.  A stress-limit coefficient "
            f"is a prescription of a document; without one there is "
            f"nothing to look it up in, and GenSec will not fall back "
            f"to a default here."
        )
    provider = _LIMIT_PROVIDERS.get(code.key)
    if provider is None:
        raise ValueError(
            f"Combination '{combo_name}': code '{code.key}' has no "
            f"stress-limit provider in GenSec.  Providers exist for: "
            f"{sorted(_LIMIT_PROVIDERS)}.  Serving another document's "
            f"coefficients under this name would be a normative "
            f"substitution nothing in the output could reveal."
        )
    overrides = block.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError(
            f"Combination '{combo_name}': sls.limits['overrides'] must "
            f"be a mapping of coefficient names to values, got "
            f"{type(overrides).__name__}."
        )
    return provider(
        block["kind"],
        fyk=block.get("fyk"),
        fpk=block.get("fpk"),
        decompression=bool(block.get("decompression", False)),
        custom_zone_strengths=block.get("custom_zone_strengths"),
        **overrides,
    )


def _parse_sls_block(combo_name, block, model_code):
    r"""
    Parse the ``sls`` block of one combination.

    The block is the combination's **verification identity**: which
    effective stiffness the section is analysed with, and which limits
    the resulting stresses are held to.  The two travel together
    because they are decided together -- a quasi-permanent check uses
    the creep-reduced modulus *and* the quasi-permanent coefficient,
    and a document that could state one without the other would be
    able to describe a verification nobody performs.

    Parameters
    ----------
    combo_name : str
    block : dict
        The raw ``sls`` mapping.
    model_code : str or None
        The document's top-level ``code``.

    Returns
    -------
    dict
        ``{"moduli": {zone: float}, "limits": StressLimits}``.

    Raises
    ------
    ValueError
        If the block is not a mapping, carries a key outside
        :data:`VALID_SLS_KEYS`, or omits either sub-block.
    """
    if not isinstance(block, dict):
        raise ValueError(
            f"Combination '{combo_name}': 'sls' must be a mapping with "
            f"'moduli' and 'limits', got {type(block).__name__}."
        )
    _refuse_unknown_keys(
        block, where=f"Combination '{combo_name}', sls",
        valid=VALID_SLS_KEYS)
    for required in VALID_SLS_KEYS:
        if required not in block:
            raise ValueError(
                f"Combination '{combo_name}': the 'sls' block declares "
                f"no '{required}'.  Both are needed and neither has a "
                f"default: 'moduli' fixes the stiffness the section is "
                f"analysed with, 'limits' fixes what the resulting "
                f"stresses are held to, and a block with one of them "
                f"describes half a verification."
            )
    return {
        "moduli": _parse_sls_moduli(combo_name, block["moduli"]),
        "limits": _parse_stress_limits(combo_name, block["limits"],
                                       model_code),
    }


def _parse_combination(c_spec, *, model_code=None):
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
    # Z-15: the combination level accepted unknown keys in silence.
    # Checked first, so that a misspelt 'componets' is reported as a
    # misspelling instead of as "must have 'components' or 'stages'",
    # which is true and unhelpful.  The stage-only names are inside
    # VALID_COMBINATION_KEYS on purpose: they are known keys in the
    # wrong place, and the refusals below diagnose them better than a
    # list of valid names would.
    _refuse_unknown_keys(c_spec, where=f"Combination '{name}'",
                         valid=VALID_COMBINATION_KEYS)
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
        # Phase 33: the SLS verification identity of this combination.
        # Optional, and its absence is reported rather than assumed --
        # gensec.solver.timeline_run.run_sls emits an explicit skipped
        # record for a combination that declares none, so "no stress
        # check was requested" and "no stress check exists" are
        # different lines in the report instead of the same silence.
        if "sls" in c_spec:
            combo["sls"] = _parse_sls_block(name, c_spec["sls"],
                                            model_code)
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
        # Z-15, one level down.  The check reads the stage as GIVEN;
        # the two private keys this loop injects into the stage it
        # RETURNS (_prestress_action_specs, _section_ops_spec) are
        # therefore not in VALID_STAGE_KEYS, and must not be: they are
        # not part of the document schema.
        _refuse_unknown_keys(
            s_spec,
            where=f"Combination '{name}', stage "
                  f"'{s_spec.get('name', f'stage_{i}')}'"
                  if isinstance(s_spec, dict)
                  else f"Combination '{name}', stage {i}",
            valid=VALID_STAGE_KEYS)
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
    staged = {"name": name, "stages": stages}
    # Phase 33: the SLS block sits on the combination, never on a
    # stage.  One verification identity per combination -- the modulus
    # and the coefficient are chosen together, and a per-stage override
    # would let one stage of a quasi-permanent check be verified as a
    # characteristic one.
    if "sls" in c_spec:
        staged["sls"] = _parse_sls_block(name, c_spec["sls"], model_code)
    return staged


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

#: Every key the ``output`` block admits.  The set is the union of what
#: :func:`_parse_output_flags` defaults below and what ``cli.py`` reads
#: from the resolved flags -- not a superset kept generous "just in
#: case", because a key admitted here and read by nobody is exactly the
#: silence this tuple exists to end.
#:
#: ``n_levels_mode`` and ``n_levels_count`` are not defaulted here: they
#: are read in ``gensec/cli.py`` (lines 251-252) with a default of their
#: own.  That is a second home for an output default and it is recorded
#: as such; admitting them is correct today, and unifying the two homes
#: is a separate decision.
_OUTPUT_VALID = (
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
    # consumed by the CLI, not by this function
    "n_levels_mode", "n_levels_count",
    # rebar annotation density on section/state figures
    "show_rebar_labels",
)


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
    # Refuse before defaulting.  Until Phase 38 this function opened
    # with ``dict(output_spec)`` and nothing else: every key the
    # document wrote survived into the flags dictionary, and one that
    # nothing downstream read was carried, ignored, and never
    # mentioned.  Phase 36 closed that hole for ``section``,
    # ``section.material_zones`` and the top level and did not reach
    # here; three keys in the project's own shipped examples had
    # already drifted through it (38_2 section 7.4).
    #
    # The order matters.  Refusing after ``setdefault`` would compare
    # the document against a mapping this function has just filled in,
    # and every unknown key would still be unknown but no longer
    # distinguishable from one the loader supplied itself.
    _refuse_unknown_keys(output_spec, where="output",
                         valid=_OUTPUT_VALID,
                         hint="resolution and reporting parameters are "
                              "declared here and nowhere else")

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

    # Per-bar annotations (name/number + strain or stress value) on
    # section_*.png and state_*.png.  A densely reinforced section
    # turns these labels into an unreadable smear; set to False to
    # keep the rebar markers but drop the text.
    flags.setdefault("show_rebar_labels", True)

    return flags
# GENSEC-COMPLETE phase=38 utc=2026-09-11T16:04:37Z sha256=bb062bbec502a8cbebcbd2c2ebdaf506ba5d7be0da9752f95039d7f8cb5e2792 body_lines=3013
