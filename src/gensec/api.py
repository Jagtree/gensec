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

"""
Public Python facade for GenSec.

All frontends (CLI, FastAPI server, pywebview desktop wrapper, notebooks)
call these functions.  Every function accepts YAML as a string or path and
returns Pydantic models that serialise cleanly to JSON.

The module deliberately knows nothing about HTTP, windowing, or rendering.
It orchestrates :mod:`gensec.io_yaml`, :mod:`gensec.solver`,
:mod:`gensec.solver.check`, and :mod:`gensec.output`.

Caching
-------
Heavy objects (parsed section, fiber solver, 3D resistance hull,
verification results) are memoised in-memory keyed on the SHA-256 hash
of the *normalised* YAML text.  Any meaningful change to the YAML
invalidates the cache automatically; whitespace, comments and key order
do not.  Cache size is bounded by ``SECTION_CACHE_SIZE``.

Memory profile
--------------
For a typical biaxial section the heavy artefacts are:

- the 3-D resistance point cloud from :meth:`NMDiagram.generate_biaxial`
  (hundreds of MB at default resolution),
- the per-N Mx-My contour cache held inside :class:`VerificationEngine`,
- the uniaxial N-My diagram (only useful when the N-My tab is opened).

To keep the GUI's RAM footprint reasonable:

- ``SECTION_CACHE_SIZE`` is small (2 by default) — desktop sessions
  rarely need more.
- ``nm_data_y`` is computed *lazily*, only when ``render_plot`` is
  invoked with ``kind='nm_y'``.
- The engine's contour cache is cleared after the verification table
  has been materialised; ``verify_point`` re-populates it on demand.
- ``_Session.build`` triggers a ``gc.collect()`` at the end so the
  intermediate NumPy buffers from ``generate_biaxial`` are reclaimed
  before control returns to the caller.

Examples
--------
>>> from gensec.api import analyze
>>> result = analyze(yaml_path="examples/biaxial_column.yaml")
>>> result.verification[0].eta_norm
0.41
"""

from __future__ import annotations

import base64
import gc
import hashlib
import io
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml as _yaml
from pydantic import BaseModel, Field

from gensec._version import __version__  # noqa: F401


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SECTION_CACHE_SIZE: int = 1
"""Maximum number of distinct sections held in memory.

Desktop GUI use: the user typically works on one YAML at a time.
Keeping only one slot guarantees the previous heavy NumPy buffers
are released as soon as a new YAML is analysed.  Larger values
accumulate hundreds of MB per session.
"""

DEFAULT_N_ANGLES_3D: int = 16
"""Default angular resolution of the 3-D resistance surface.

Kept deliberately low (every 22.5°) to bound RAM usage on large
sections.  Override per-YAML with ``output.n_angles_3d_surface``.
"""

DEFAULT_N_POINTS_3D: int = 20
"""Default points per angle for the 3-D scan.  See ``DEFAULT_N_ANGLES_3D``."""


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class MaterialInfo(BaseModel):
    """Derived properties of a material, for the GUI sidebar."""
    id: str
    kind: str
    cls: Optional[str] = None
    design_strength_MPa: Optional[float] = None
    modulus_MPa: Optional[float] = None
    eps_ultimate: Optional[float] = None


class RebarInfo(BaseModel):
    x: float
    y: float
    diameter: Optional[float] = None
    As_mm2: float
    material: str


class SectionInfo(BaseModel):
    B_mm: float
    H_mm: float
    bulk_material: str
    n_fibers_x: int
    n_fibers_y: int
    rebars: list[RebarInfo]


class DemandInfo(BaseModel):
    name: str
    N_kN: float
    Mx_kNm: float
    My_kNm: float


class CombinationInfo(BaseModel):
    name: str
    staged: bool
    resolved: DemandInfo
    stages: Optional[list[dict[str, Any]]] = None


class EnvelopeInfo(BaseModel):
    name: str
    members: list[dict[str, Any]]
    eta_max: Optional[float] = None


class VerificationRow(BaseModel):
    """One row of the verification table.

    Numeric eta fields cover the full v0.3 metric set; only those
    enabled in the YAML output flags will be populated.
    """
    kind: Literal["demand", "combination", "envelope"]
    name: str
    N_kN: Optional[float] = None
    Mx_kNm: Optional[float] = None
    My_kNm: Optional[float] = None
    # Point metrics
    eta_norm: Optional[float] = None
    eta_norm_beta: Optional[float] = None
    eta_norm_ray: Optional[float] = None
    eta_2D: Optional[float] = None
    # Path metrics (staged combinations only)
    eta_path_norm_ray: Optional[float] = None
    eta_path_norm_beta: Optional[float] = None
    eta_path_2D: Optional[float] = None
    # Worst of the enabled metrics
    eta_governing: Optional[float] = None
    status: Literal["ok", "warn", "fail"]
    staged: bool = False


class DomainPayload(BaseModel):
    """Numeric domain data for interactive plotting in the frontend."""
    nm: list[tuple[float, float]] = Field(
        default_factory=list,
        description="Uniaxial N-Mx points  [(N_kN, M_kNm), ...]",
    )
    nm_y: list[tuple[float, float]] = Field(
        default_factory=list,
        description="Uniaxial N-My points (biaxial only)",
    )
    mxmy: dict[str, list[tuple[float, float]]] = Field(
        default_factory=dict,
        description="Mx-My contours keyed by N_kN label",
    )
    surface: dict[str, Any] = Field(
        default_factory=dict,
        description="3D hull payload: {'N_kN': [...], 'Mx_kNm': [...], "
                    "'My_kNm': [...]}",
    )
    mchi: list[dict[str, Any]] = Field(
        default_factory=list,
        description="[{'N_kN': N, 'points': [[chi_1_per_mm, M_kNm], ...]}]",
    )


class Meta(BaseModel):
    gensec_version: str = __version__
    elapsed_ms: float = 0.0
    cached: bool = False
    warnings: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Full payload returned by :func:`analyze`."""
    materials: list[MaterialInfo]
    section: SectionInfo
    properties: Optional["SectionProperties"] = None
    demands: list[DemandInfo]
    combinations: list[CombinationInfo]
    envelopes: list[EnvelopeInfo]
    verification: list[VerificationRow]
    domain: DomainPayload
    meta: Meta


class SectionProperties(BaseModel):
    """Homogenized geometric & inertial properties of the section.

    Computed via :func:`gensec.geometry.properties.compute_section_properties`
    on the polygon + rebars *only*: no solver, no NMDiagram, no
    verification.  Cached on the :class:`_Session` so that an
    :func:`analyze` call following an :func:`inspect` does not
    recompute them.

    Lengths in mm, areas in mm², second moments in mm⁴, section
    moduli in mm³, moduli in MPa, angles in radians.  Fields that
    cannot be defined (e.g. plastic moduli when
    ``compute_plastic=False``) are returned as ``None``.
    """
    # Homogenization meta
    E_ref_MPa: float
    E_bulk_MPa: float
    n_bulk: float
    # Area and centroid
    area_mm2: float
    Sx_mm3: float
    Sy_mm3: float
    xg_mm: float
    yg_mm: float
    # Centroidal second moments (user frame)
    Ix_mm4: float
    Iy_mm4: float
    Ixy_mm4: float
    # Principal centroidal second moments
    I_xi_mm4: float
    I_eta_mm4: float
    alpha_rad: float
    # Radii of gyration
    rho_x_mm: float
    rho_y_mm: float
    rho_xi_mm: float
    rho_eta_mm: float
    I_polar_mm4: float
    is_convex: bool
    # Extreme-fiber distances
    c_y_top_mm: float
    c_y_bot_mm: float
    c_x_left_mm: float
    c_x_right_mm: float
    c_xi_pos_mm: float
    c_xi_neg_mm: float
    c_eta_pos_mm: float
    c_eta_neg_mm: float
    # Elastic section moduli
    W_x_top_mm3: Optional[float] = None
    W_x_bot_mm3: Optional[float] = None
    W_y_left_mm3: Optional[float] = None
    W_y_right_mm3: Optional[float] = None
    W_xi_pos_mm3: Optional[float] = None
    W_xi_neg_mm3: Optional[float] = None
    W_eta_pos_mm3: Optional[float] = None
    W_eta_neg_mm3: Optional[float] = None
    # Plastic section moduli
    Z_x_mm3: Optional[float] = None
    Z_y_mm3: Optional[float] = None
    Z_xi_mm3: Optional[float] = None
    Z_eta_mm3: Optional[float] = None
    # Torsional constant (placeholder; future St-Venant solver)
    I_t_mm4: Optional[float] = None


class InspectResult(BaseModel):
    """Light-weight response returned by :func:`inspect`.

    Contains everything derivable from the YAML without running the
    fiber solver: materials, section geometry, homogenized geometric
    & inertial properties, demands, factor-resolved combinations
    and envelopes.  *No* verification, *no* resistance domain.

    The GUI calls this immediately after the user loads a YAML, to
    populate the side panels and the "Properties" tab.  Hitting
    "Run" then triggers :func:`analyze`, which re-uses the same
    cache key, so the YAML is parsed only once.
    """
    materials: list[MaterialInfo]
    section: SectionInfo
    properties: SectionProperties
    demands: list[DemandInfo]
    combinations: list[CombinationInfo]
    envelopes: list[EnvelopeInfo]
    meta: Meta


AnalysisResult.model_rebuild()


class ContourResponse(BaseModel):
    N_kN: float
    points: list[tuple[float, float]]
    meta: Meta


class PointVerificationResponse(BaseModel):
    N_kN: float
    Mx_kNm: float
    My_kNm: float
    eta_norm: Optional[float] = None
    eta_norm_beta: Optional[float] = None
    eta_norm_ray: Optional[float] = None
    eta_2D: Optional[float] = None
    eta_governing: Optional[float] = None
    status: Literal["ok", "warn", "fail"]
    meta: Meta


PlotKindLit = Literal[
    "mxmy", "nm", "nm_y", "mchi", "surface", "polar", "section",
]


class PlotImageResponse(BaseModel):
    kind: str
    mime: Literal["image/png"]
    data_base64: str
    width_px: int
    height_px: int
    meta: Meta


# ---------------------------------------------------------------------------
# Input normalisation & caching
# ---------------------------------------------------------------------------

def _load_yaml_text(
    yaml_text: Optional[str] = None,
    yaml_path: Optional[Union[str, Path]] = None,
) -> str:
    """Return YAML content as a string; exactly one input must be given."""
    if (yaml_text is None) == (yaml_path is None):
        raise ValueError(
            "Provide exactly one of 'yaml_text' or 'yaml_path'."
        )
    if yaml_path is not None:
        p = Path(yaml_path)
        if not p.is_file():
            raise FileNotFoundError(f"YAML file not found: {p}")
        return p.read_text(encoding="utf-8")
    return yaml_text  # type: ignore[return-value]


def _normalise_yaml(text: str) -> str:
    """Canonicalise YAML so equivalent inputs hash identically."""
    data = _yaml.safe_load(text) or {}
    return _yaml.safe_dump(data, sort_keys=True, default_flow_style=False)


def yaml_key(text: str) -> str:
    """SHA-256 hex digest of the *normalised* YAML text."""
    norm = _normalise_yaml(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


@lru_cache(maxsize=SECTION_CACHE_SIZE)
def _get_session(key: str, normalised_yaml: str) -> "_Session":
    """Build or retrieve a cached :class:`_Session` for this YAML."""
    return _Session.build(normalised_yaml)


def _session_for(
    yaml_text: Optional[str], yaml_path: Optional[Union[str, Path]],
) -> tuple["_Session", bool]:
    """Resolve inputs -> (session, cached_flag).

    Used by every public entry point so caching is uniform.
    """
    text = _load_yaml_text(yaml_text, yaml_path)
    norm = _normalise_yaml(text)
    key = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    hits_before = _get_session.cache_info().hits
    session = _get_session(key, norm)
    cached = _get_session.cache_info().hits > hits_before
    return session, cached


# ---------------------------------------------------------------------------
# Session: holds parsed section, solver, precomputed domain, verification
# ---------------------------------------------------------------------------

class _Session:
    """Expensive, reusable per-YAML state.  Do not expose to frontends."""

    __slots__ = (
        "yaml_data", "section", "solver", "nmdiagram",
        "domain", "verification", "_properties",
    )

    def __init__(
        self, *, yaml_data, section, solver, nmdiagram,
        domain, verification,
    ):
        self.yaml_data    = yaml_data
        self.section      = section
        self.solver       = solver
        self.nmdiagram    = nmdiagram
        self.domain       = domain
        self.verification = verification

    # -- construction -------------------------------------------------------

    @classmethod
    def build(cls, normalised_yaml: str) -> "_Session":
        """Parse YAML, build section & solver, run the analysis.

        Heavy: runs once per unique YAML.  Memory hot-spots:

        - ``NMDiagram.generate_biaxial`` allocates a flat float64 array
          per Mx/My/N triple (size ``n_angles × n_points_per_angle``)
          and the inner integrator works on
          ``n_configs × n_fibers`` strain matrices.
        - ``VerificationEngine`` builds an ``MxMyContour`` for every
          distinct N in the demand set; we clear this cache below
          once the verification table has been materialised.
        """
        # Local imports keep this module lightweight at import time.
        from .io_yaml import load_yaml
        from .solver import FiberSolver, NMDiagram
        from .solver.check import VerificationEngine

        # load_yaml wants a path.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8",
        ) as f:
            f.write(normalised_yaml)
            tmp = f.name
        try:
            data = load_yaml(tmp)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

        section      = data["section"]
        demands      = data["demands"]
        combinations = data.get("combinations", [])
        envelopes    = data.get("envelopes", [])
        opts         = data.get("output_options", {})
        n_points     = int(opts.get("n_points", 400))
        n_angles_3d  = int(opts.get("n_angles_3d_surface",
                                     DEFAULT_N_ANGLES_3D))
        n_points_3d  = int(opts.get("n_points_3d_surface",
                                     DEFAULT_N_POINTS_3D))

        solver     = FiberSolver(section)
        is_biaxial = int(section.n_fibers_x) > 1

        nm_gen     = NMDiagram(solver)

        # ------------------------------------------------------------------
        # FULLY LAZY pipeline.
        #
        # The only mandatory eager step for ``analyze()`` is the
        # verification table — without it the GUI has nothing to show
        # in the bottom panel after Run.  Everything else is deferred:
        #
        #   nm_data          uniaxial N-Mx curve
        #                    ----> eager only for UNIAXIAL sections
        #                          (acts as the resistance domain).
        #                          For biaxial sections it's lazy.
        #   nm_data_y        N-My curve   ----> lazy.
        #   nm_3d            3-D point cloud
        #                    ----> eager for biaxial (domain), lazy never
        #                          re-used for the 3D plot (already kept).
        #   mx_my contours   per-N 2-D contours
        #                    ----> contour cache wiped after verification;
        #                          contour_at_N / verify_point re-fill it
        #                          on demand.
        #   M-chi diagrams   ----> lazy, only built by render_plot('mchi').
        # ------------------------------------------------------------------
        if is_biaxial:
            nm_data = None  # lazy: built only if the N-Mx plot is opened
            nm_3d = nm_gen.generate_biaxial(
                n_angles=n_angles_3d,
                n_points_per_angle=n_points_3d,
            )
            domain_data = nm_3d
        else:
            # Uniaxial: nm_data IS the resistance domain.  Cannot defer.
            nm_data = nm_gen.generate(n_points=n_points)
            nm_3d = None
            domain_data = nm_data

        engine = VerificationEngine(
            domain_data, nm_gen, opts,
            n_points=n_points // 2,
        )

        demand_db = {d["name"]: d for d in demands}

        demand_results = engine.check_demands(demands) if demands else []

        combination_results: list[dict] = []
        combination_db: dict[str, dict] = {}
        for combo in combinations:
            try:
                cr = engine.check_combination(combo, demand_db)
                combination_results.append(cr)
                combination_db[combo["name"]] = cr
            except KeyError:
                pass

        envelope_results: list[dict] = []
        for env in envelopes:
            try:
                envelope_results.append(
                    engine.check_envelope(env, demand_db, combination_db))
            except KeyError:
                pass

        # Free the engine's per-N MxMyContour cache: it filled during
        # check_demands but the verification rows are already
        # materialised.  verify_point repopulates it lazily on demand.
        try:
            engine._contour_cache.clear()
        except Exception:
            pass

        # Drop the raw point cloud kept inside DomainChecker for the
        # Chebyshev-radius LP: it's used only at construction time,
        # so several MB can be reclaimed here.  hull.equations is
        # what verify_point actually needs at runtime.
        try:
            engine.domain._pts_norm = None
        except Exception:
            pass

        session = cls(
            yaml_data=data,
            section=section,
            solver=solver,
            nmdiagram=nm_gen,
            domain=dict(
                nm_data=nm_data,        # None for biaxial (lazy)
                nm_data_y=None,         # always lazy
                nm_3d=nm_3d,            # set for biaxial, None for uniaxial
                is_biaxial=is_biaxial,
            ),
            verification=dict(
                demands=demand_results,
                combinations=combination_results,
                envelopes=envelope_results,
                engine=engine,
            ),
        )

        # Compute & memoize homogenized geometric properties.
        try:
            session._properties = _compute_section_properties(
                section, data["materials"])
        except Exception:
            session._properties = None
        # Reclaim intermediate NumPy buffers (mega-batch integrate
        # allocates several O(n_configs * n_fibers) float64 arrays
        # that become unreachable once we return).
        gc.collect()
        return session

    # -- lazy: 3D surface --------------------------------------------------

    def get_nm_3d(self, n_angles: int = DEFAULT_N_ANGLES_3D,
                  n_points_per_angle: int = DEFAULT_N_POINTS_3D):
        """Return the 3-D resistance point cloud, computing it on demand."""
        if self.domain.get("nm_3d") is None and self.domain.get("is_biaxial"):
            self.domain["nm_3d"] = self.nmdiagram.generate_biaxial(
                n_angles=n_angles,
                n_points_per_angle=n_points_per_angle,
            )
            gc.collect()
        return self.domain.get("nm_3d")

    # -- lazy: uniaxial N-Mx (biaxial sections only) ----------------------

    def get_nm_data(self):
        """Return the uniaxial N-Mx curve, computing it on first access.

        For uniaxial sections this is already eager (it's the domain).
        For biaxial sections it's deferred until the N-Mx plot tab
        is opened.
        """
        if self.domain.get("nm_data") is None:
            opts = self.yaml_data.get("output_options", {})
            n_points = int(opts.get("n_points", 400))
            self.domain["nm_data"] = self.nmdiagram.generate(
                n_points=n_points, direction="x")
            gc.collect()
        return self.domain.get("nm_data")

    # -- lazy: uniaxial N-My -----------------------------------------------

    def get_nm_data_y(self):
        """Return the uniaxial N-My curve, computing it on first access."""
        if (self.domain.get("nm_data_y") is None
                and self.domain.get("is_biaxial")):
            opts = self.yaml_data.get("output_options", {})
            n_points = int(opts.get("n_points", 400))
            self.domain["nm_data_y"] = self.nmdiagram.generate(
                n_points=n_points, direction="y")
            gc.collect()
        return self.domain.get("nm_data_y")


def clear_cache() -> None:
    """Evict every cached session.  Call after shutdown or in tests."""
    _get_session.cache_clear()
    gc.collect()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inspect(
    yaml_text: Optional[str] = None,
    yaml_path: Optional[Union[str, Path]] = None,
    *,
    compute_plastic: bool = True,
) -> InspectResult:
    """Light-weight YAML inspection: section + properties, no solver.

    Called by the GUI the moment the user loads a YAML.  Parses the
    document, builds the section geometry, computes the homogenized
    geometric & inertial properties, returns materials / section /
    demands / combinations / envelopes metadata.  It does **not**
    instantiate ``FiberSolver``, **not** build the resistance
    domain, **not** run any verification.  Typical cost: < 200 ms.

    A subsequent :func:`analyze` call on the same YAML re-uses the
    parsed section and the already-computed properties via the
    LRU cache (no double parsing).

    Parameters
    ----------
    yaml_text : str, optional
    yaml_path : str or Path, optional
        Exactly one must be provided.
    compute_plastic : bool, default True
        Whether to compute plastic section moduli ``Z_*``.  Cheap.

    Returns
    -------
    InspectResult
    """
    t0 = time.perf_counter()
    text = _load_yaml_text(yaml_text, yaml_path)
    norm = _normalise_yaml(text)

    from .io_yaml import load_yaml
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
    ) as f:
        f.write(norm)
        tmp = f.name
    try:
        data = load_yaml(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    section = data["section"]

    try:
        from .geometry.properties import (
            compute_section_properties, HomogenizedRebar,
        )
        poly = getattr(section, "polygon", None)
        bulk = section.bulk_material
        E_bulk = _material_modulus(bulk) or 1.0
        rebars_hom = []
        for r in getattr(section, "rebars", []) or []:
            x = float(r.x) if getattr(r, "x", None) is not None else 0.0
            E_s = _material_modulus(r.material) or E_bulk
            rebars_hom.append(HomogenizedRebar(
                x=x, y=float(r.y), area=float(r.As), E=float(E_s),
            ))
        props_dc = (compute_section_properties(
            polygon=poly, rebars=rebars_hom,
            E_bulk=float(E_bulk), E_ref=float(E_bulk),
            compute_plastic=compute_plastic)
            if poly is not None else None)
    except Exception:
        props_dc = None

    materials_info = [_material_info(mid, m)
                      for mid, m in data["materials"].items()]
    section_info = _section_info(section)
    properties_info = _section_properties_payload(props_dc)

    demands_info = [
        DemandInfo(name=str(d["name"]),
                   N_kN=d["N"] / 1e3,
                   Mx_kNm=d["Mx"] / 1e6,
                   My_kNm=d["My"] / 1e6)
        for d in data.get("demands", [])
    ]

    # Resolve combinations arithmetically (no solver needed).
    demand_db = {d["name"]: d for d in data.get("demands", [])}
    combinations_info: list[CombinationInfo] = []
    for c in data.get("combinations", []):
        if "stages" in c:
            staged = True
            N = Mx = My = 0.0
            stages_payload = []
            for s in c["stages"]:
                sN = sMx = sMy = 0.0
                for comp in s.get("components", []):
                    ref = comp["ref"]; f = float(comp.get("factor", 1.0))
                    d = demand_db.get(ref)
                    if d is None:
                        continue
                    sN += f * d["N"]; sMx += f * d["Mx"]; sMy += f * d["My"]
                N += sN; Mx += sMx; My += sMy
                stages_payload.append({
                    "name": str(s.get("name", "")),
                    "components": s.get("components", []),
                    "cumulative": {
                        "N_kN": N / 1e3, "Mx_kNm": Mx / 1e6,
                        "My_kNm": My / 1e6,
                    },
                })
        else:
            staged = False
            N = Mx = My = 0.0
            for comp in c.get("components", []):
                ref = comp["ref"]; f = float(comp.get("factor", 1.0))
                d = demand_db.get(ref)
                if d is None:
                    continue
                N += f * d["N"]; Mx += f * d["Mx"]; My += f * d["My"]
            stages_payload = None

        combinations_info.append(CombinationInfo(
            name=str(c["name"]),
            staged=staged,
            resolved=DemandInfo(
                name=str(c["name"]),
                N_kN=N / 1e3, Mx_kNm=Mx / 1e6, My_kNm=My / 1e6,
            ),
            stages=stages_payload,
        ))

    envelopes_info = [
        EnvelopeInfo(name=str(e.get("name", "")),
                     members=e.get("members", []),
                     eta_max=None)
        for e in data.get("envelopes", [])
    ]

    return InspectResult(
        materials=materials_info,
        section=section_info,
        properties=properties_info,
        demands=demands_info,
        combinations=combinations_info,
        envelopes=envelopes_info,
        meta=Meta(
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            cached=False,
        ),
    )


def analyze(
    yaml_text: Optional[str] = None,
    yaml_path: Optional[Union[str, Path]] = None,
) -> AnalysisResult:
    """Run the full GenSec analysis from YAML and return a structured payload.

    Parameters
    ----------
    yaml_text : str, optional
        Raw YAML content.
    yaml_path : str or Path, optional
        Path to a YAML file on disk.

    Returns
    -------
    AnalysisResult
        All JSON-serialisable.  Cached transparently by YAML hash.
    """
    t0 = time.perf_counter()
    session, cached = _session_for(yaml_text, yaml_path)
    result = _build_analysis_result(session)
    result.meta = Meta(
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        cached=cached,
    )
    return result


def contour_at_N(
    N_kN: float,
    yaml_text: Optional[str] = None,
    yaml_path: Optional[Union[str, Path]] = None,
    n_angles: int = 144,
    n_points_per_angle: int = 200,
) -> ContourResponse:
    """Return a single Mx-My interaction contour at a fixed axial force."""
    t0 = time.perf_counter()
    session, cached = _session_for(yaml_text, yaml_path)

    mx_my = session.nmdiagram.generate_mx_my(
        N_fixed=float(N_kN) * 1e3,
        n_angles=n_angles,
        n_points_per_angle=n_points_per_angle,
    )
    mx = _as_list(mx_my.get("Mx_kNm", mx_my.get("Mx", [])))
    my = _as_list(mx_my.get("My_kNm", mx_my.get("My", [])))
    if "Mx_kNm" not in mx_my:
        mx = [v / 1e6 for v in mx]
        my = [v / 1e6 for v in my]
    points = list(zip(mx, my))

    return ContourResponse(
        N_kN=float(N_kN),
        points=points,
        meta=Meta(
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            cached=cached,
        ),
    )


def verify_point(
    N_kN: float,
    Mx_kNm: float,
    My_kNm: float,
    yaml_text: Optional[str] = None,
    yaml_path: Optional[Union[str, Path]] = None,
) -> PointVerificationResponse:
    """Verify an ad-hoc demand against the cached resistance domain."""
    t0 = time.perf_counter()
    session, cached = _session_for(yaml_text, yaml_path)

    demand = {
        "name": "probe",
        "N":  float(N_kN)  * 1e3,
        "Mx": float(Mx_kNm) * 1e6,
        "My": float(My_kNm) * 1e6,
    }
    engine = session.verification["engine"]
    rows = engine.check_demands([demand])
    row = rows[0] if rows else {}

    eta_gov = _governing_eta(row)
    return PointVerificationResponse(
        N_kN=float(N_kN),
        Mx_kNm=float(Mx_kNm),
        My_kNm=float(My_kNm),
        eta_norm=row.get("eta_norm"),
        eta_norm_beta=row.get("eta_norm_beta"),
        eta_norm_ray=row.get("eta_norm_ray"),
        eta_2D=row.get("eta_2D"),
        eta_governing=eta_gov,
        status=_status_from_eta(eta_gov),
        meta=Meta(
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            cached=cached,
        ),
    )


def render_plot(
    kind: PlotKindLit,
    yaml_text: Optional[str] = None,
    yaml_path: Optional[Union[str, Path]] = None,
    width_px: int = 1200,
    height_px: int = 800,
    dpi: int = 150,
    **kwargs: Any,
) -> PlotImageResponse:
    """Render a matplotlib plot as a base64-encoded PNG."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    from .output import (
        plot_nm_diagram, plot_mx_my_diagram, plot_moment_curvature,
        plot_3d_surface, plot_polar_ductility, plot_section,
    )

    t0 = time.perf_counter()
    session, cached = _session_for(yaml_text, yaml_path)
    dom = session.domain
    nm_gen = session.nmdiagram

    fig = None
    try:
        if kind == "nm":
            fig = plot_nm_diagram(session.get_nm_data())
        elif kind == "nm_y":
            if not dom["is_biaxial"]:
                raise ValueError("N-My plot requires a biaxial section.")
            # Lazy: compute only now if not done before.
            nm_data_y = session.get_nm_data_y()
            fig = plot_nm_diagram(nm_data_y,
                                  title="N-My Interaction Diagram")
        elif kind == "surface":
            nm_3d = session.get_nm_3d()
            if nm_3d is None:
                raise ValueError("3D surface requires a biaxial section.")
            fig = plot_3d_surface(
                nm_3d,
                demands=session.yaml_data.get("demands", []),
            )
        elif kind == "section":
            fig = plot_section(session.section, title="Section geometry")
        elif kind == "mxmy":
            N_kN = float(kwargs.get("N_kN", 0.0))
            data = nm_gen.generate_mx_my(
                N_fixed=N_kN * 1e3,
                n_angles=int(kwargs.get("n_angles", 144)),
                n_points_per_angle=int(kwargs.get("n_points_per_angle",
                                                  200)),
            )
            fig = plot_mx_my_diagram(data)
        elif kind == "mchi":
            N_kN = float(kwargs.get("N_kN", 0.0))
            direction = kwargs.get("direction", "x")
            data = nm_gen.generate_moment_curvature(
                N_fixed=N_kN * 1e3,
                n_points=int(kwargs.get("n_points", 400)),
                direction=direction,
            )
            fig = plot_moment_curvature(data)
        elif kind == "polar":
            if not dom["is_biaxial"]:
                raise ValueError("Polar plot requires a biaxial section.")
            N_kN = float(kwargs.get("N_kN", 0.0))
            fig = plot_polar_ductility(
                nm_gen,
                N_fixed=N_kN * 1e3,
                n_angles=int(kwargs.get("n_angles", 144)),
                n_points=int(kwargs.get("n_points", 400)),
            )
        else:
            raise ValueError(f"Unknown plot kind: {kind}")

        fig.set_size_inches(width_px / dpi, height_px / dpi)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        data_b64 = base64.b64encode(buf.read()).decode("ascii")

    finally:
        if fig is not None:
            plt.close(fig)

    return PlotImageResponse(
        kind=kind,
        mime="image/png",
        data_base64=data_b64,
        width_px=width_px,
        height_px=height_px,
        meta=Meta(
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            cached=cached,
        ),
    )


# ---------------------------------------------------------------------------
# Private helpers: payload assembly
# ---------------------------------------------------------------------------

def _compute_section_properties(section, materials_dict):
    """Run the geometry/properties pipeline on a parsed section.

    Returns the dataclass ``SectionProperties`` from
    :mod:`gensec.geometry.properties`, or ``None`` if any required
    attribute is missing.  Reference modulus is the bulk material's
    ``Ec`` if available, else 1.0 (pure-geometry mode).
    """
    from .geometry.properties import (
        compute_section_properties, HomogenizedRebar,
    )

    poly = getattr(section, "polygon", None)
    if poly is None:
        return None

    bulk = section.bulk_material
    E_bulk = _material_modulus(bulk) or 1.0
    E_ref = E_bulk

    rebars_hom = []
    for r in getattr(section, "rebars", []) or []:
        x = float(r.x) if getattr(r, "x", None) is not None else 0.0
        E_s = _material_modulus(r.material) or E_bulk
        rebars_hom.append(HomogenizedRebar(
            x=x, y=float(r.y), area=float(r.As), E=float(E_s),
        ))

    return compute_section_properties(
        polygon=poly,
        rebars=rebars_hom,
        E_bulk=float(E_bulk),
        E_ref=float(E_ref),
        compute_plastic=True,
    )


def _material_modulus(mat) -> Optional[float]:
    """Return the elastic modulus [MPa] of a material, if exposed."""
    for attr in ("Es", "Ec", "ecm"):
        v = getattr(mat, attr, None)
        if v is not None and float(v) > 0.0:
            return float(v)
    ec2 = getattr(mat, "ec2", None)
    if ec2 is not None:
        v = getattr(ec2, "ecm", None)
        if v is not None and float(v) > 0.0:
            return float(v)
    return None


def _nan_to_none(v) -> Optional[float]:
    """NaN-safe float coercion for JSON-serialisable Pydantic fields."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:        # NaN check
        return None
    if f in (float("inf"), float("-inf")):
        return None
    return f


def _section_properties_payload(p) -> Optional["SectionProperties"]:
    """Convert the dataclass to its Pydantic image, with NaN scrubbed."""
    if p is None:
        return None
    return SectionProperties(
        E_ref_MPa=float(p.E_ref),
        E_bulk_MPa=float(p.E_bulk),
        n_bulk=float(p.n_bulk),
        area_mm2=float(p.area),
        Sx_mm3=float(p.Sx),
        Sy_mm3=float(p.Sy),
        xg_mm=float(p.xg),
        yg_mm=float(p.yg),
        Ix_mm4=float(p.Ix),
        Iy_mm4=float(p.Iy),
        Ixy_mm4=float(p.Ixy),
        I_xi_mm4=float(p.I_xi),
        I_eta_mm4=float(p.I_eta),
        alpha_rad=float(p.alpha),
        rho_x_mm=float(p.rho_x),
        rho_y_mm=float(p.rho_y),
        rho_xi_mm=float(p.rho_xi),
        rho_eta_mm=float(p.rho_eta),
        I_polar_mm4=float(p.I_polar),
        is_convex=bool(p.is_convex),
        c_y_top_mm=float(p.c_y_top),
        c_y_bot_mm=float(p.c_y_bot),
        c_x_left_mm=float(p.c_x_left),
        c_x_right_mm=float(p.c_x_right),
        c_xi_pos_mm=float(p.c_xi_pos),
        c_xi_neg_mm=float(p.c_xi_neg),
        c_eta_pos_mm=float(p.c_eta_pos),
        c_eta_neg_mm=float(p.c_eta_neg),
        W_x_top_mm3=_nan_to_none(p.W_x_top),
        W_x_bot_mm3=_nan_to_none(p.W_x_bot),
        W_y_left_mm3=_nan_to_none(p.W_y_left),
        W_y_right_mm3=_nan_to_none(p.W_y_right),
        W_xi_pos_mm3=_nan_to_none(p.W_xi_pos),
        W_xi_neg_mm3=_nan_to_none(p.W_xi_neg),
        W_eta_pos_mm3=_nan_to_none(p.W_eta_pos),
        W_eta_neg_mm3=_nan_to_none(p.W_eta_neg),
        Z_x_mm3=_nan_to_none(p.Z_x),
        Z_y_mm3=_nan_to_none(p.Z_y),
        Z_xi_mm3=_nan_to_none(p.Z_xi),
        Z_eta_mm3=_nan_to_none(p.Z_eta),
        I_t_mm4=_nan_to_none(p.I_t),
    )


_POINT_ETA_KEYS = ("eta_norm", "eta_norm_beta", "eta_norm_ray", "eta_2D")
_PATH_ETA_KEYS = ("eta_path_norm_ray", "eta_path_norm_beta", "eta_path_2D")


def _governing_eta(row: dict) -> Optional[float]:
    """Worst (max) of the enabled etas in a verification row."""
    vals = [row.get(k) for k in _POINT_ETA_KEYS + _PATH_ETA_KEYS
            if row.get(k) is not None]
    return max(vals) if vals else None


def _status_from_eta(eta: Optional[float]) -> Literal["ok", "warn", "fail"]:
    """Map an η value to a traffic-light status."""
    if eta is None:
        return "ok"
    if eta >= 1.0:
        return "fail"
    if eta >= 0.85:
        return "warn"
    return "ok"


def _as_list(arr) -> list[float]:
    """NumPy array or list -> list[float]."""
    if arr is None:
        return []
    if hasattr(arr, "tolist"):
        return [float(x) for x in arr.tolist()]
    return [float(x) for x in arr]


def _material_info(mid: str, mat) -> MaterialInfo:
    """Extract display props from a Concrete/Steel/Tabulated instance."""
    info = MaterialInfo(id=mid, kind=type(mat).__name__.lower())

    if hasattr(mat, "fcd") and mat.fcd is not None:
        info.design_strength_MPa = float(mat.fcd)
    elif hasattr(mat, "fck") and mat.fck is not None:
        info.design_strength_MPa = float(mat.fck)
    if hasattr(mat, "Ec") and mat.Ec is not None and info.modulus_MPa is None:
        info.modulus_MPa = float(mat.Ec)
    if hasattr(mat, "eps_cu2") and mat.eps_cu2 is not None:
        info.eps_ultimate = float(mat.eps_cu2)
    if hasattr(mat, "ec2") and getattr(mat, "ec2", None) is not None:
        info.cls = getattr(mat.ec2, "name", None)

    if hasattr(mat, "fyd") and mat.fyd is not None:
        info.design_strength_MPa = float(mat.fyd)
    if hasattr(mat, "Es") and mat.Es is not None:
        info.modulus_MPa = float(mat.Es)
    if hasattr(mat, "eps_su") and mat.eps_su is not None:
        info.eps_ultimate = float(mat.eps_su)

    return info


def _section_info(section) -> SectionInfo:
    """Works for both RectSection and GenericSection (both expose B/H)."""
    B = float(section.B)
    H = float(section.H)
    rebars = []
    for r in getattr(section, "rebars", []) or []:
        x = float(r.x) if getattr(r, "x", None) is not None else B / 2.0
        rebars.append(RebarInfo(
            x=x,
            y=float(r.y),
            diameter=(float(r.diameter)
                      if getattr(r, "diameter", 0) else None),
            As_mm2=float(r.As),
            material=type(r.material).__name__.lower(),
        ))
    return SectionInfo(
        B_mm=B, H_mm=H,
        bulk_material=type(section.bulk_material).__name__.lower(),
        n_fibers_x=int(section.n_fibers_x),
        n_fibers_y=int(section.n_fibers_y),
        rebars=rebars,
    )


def _domain_payload(dom: dict) -> DomainPayload:
    """Serialise the numeric domain into the public shape.

    Only fields that are *already* computed are returned.  In the
    fully-lazy flow, ``analyze()`` ships an essentially empty domain
    payload — the front-end requests each curve through
    ``render_plot`` (or ``contour_at_N``) the moment the user opens
    the matching tab.  This keeps both the response size and the
    server-side RAM footprint to a minimum.
    """
    nm_points: list[tuple[float, float]] = []
    if dom.get("nm_data") is not None:
        nm = dom["nm_data"]
        nm_points = list(zip(_as_list(nm.get("N_kN")),
                             _as_list(nm.get("M_kNm"))))

    nm_y_points: list[tuple[float, float]] = []
    if dom.get("nm_data_y") is not None:
        nmy = dom["nm_data_y"]
        nm_y_points = list(zip(_as_list(nmy.get("N_kN")),
                               _as_list(nmy.get("M_kNm"))))

    surface: dict[str, Any] = {}
    if dom.get("nm_3d") is not None:
        s = dom["nm_3d"]
        for out_key, candidates in (
            ("N_kN",   ("N_kN",  "N")),
            ("Mx_kNm", ("Mx_kNm", "Mx")),
            ("My_kNm", ("My_kNm", "My")),
        ):
            for c in candidates:
                if c in s:
                    vals = _as_list(s[c])
                    if c == "N":
                        vals = [v / 1e3 for v in vals]
                    elif c in ("Mx", "My"):
                        vals = [v / 1e6 for v in vals]
                    surface[out_key] = vals
                    break

    return DomainPayload(
        nm=nm_points, nm_y=nm_y_points,
        mxmy={}, surface=surface, mchi=[],
    )


def _build_verification_row(
    kind: Literal["demand", "combination", "envelope"],
    name: str,
    raw: dict,
    *,
    staged: bool = False,
    forces: Optional[dict] = None,
) -> VerificationRow:
    """Build a VerificationRow from a raw VerificationEngine result dict."""
    f = forces or {}
    eta_gov = _governing_eta(raw)
    return VerificationRow(
        kind=kind,
        name=str(name),
        N_kN=f.get("N_kN") if f else raw.get("N_kN"),
        Mx_kNm=f.get("Mx_kNm") if f else raw.get("Mx_kNm"),
        My_kNm=f.get("My_kNm") if f else raw.get("My_kNm"),
        eta_norm=raw.get("eta_norm"),
        eta_norm_beta=raw.get("eta_norm_beta"),
        eta_norm_ray=raw.get("eta_norm_ray"),
        eta_2D=raw.get("eta_2D"),
        eta_path_norm_ray=raw.get("eta_path_norm_ray"),
        eta_path_norm_beta=raw.get("eta_path_norm_beta"),
        eta_path_2D=raw.get("eta_path_2D"),
        eta_governing=eta_gov,
        status=_status_from_eta(eta_gov),
        staged=staged,
    )


def _build_analysis_result(session: "_Session") -> AnalysisResult:
    """Convert a built Session into the public payload.

    All ``name`` fields read from raw YAML are coerced via ``str(...)``
    so that integer keys like ``name: 1`` (legitimate YAML, but
    rejected by Pydantic v2 ``str`` fields) round-trip correctly.
    """
    raw = session.yaml_data
    ver = session.verification

    materials_info = [_material_info(mid, m)
                      for mid, m in raw["materials"].items()]
    section_info = _section_info(session.section)

    demands_info = [
        DemandInfo(name=str(d["name"]),
                   N_kN=d["N"] / 1e3,
                   Mx_kNm=d["Mx"] / 1e6,
                   My_kNm=d["My"] / 1e6)
        for d in raw.get("demands", [])
    ]

    combinations_info: list[CombinationInfo] = []
    for cr in ver["combinations"]:
        res = cr.get("resultant", {}) or {}
        combinations_info.append(CombinationInfo(
            name=str(cr["name"]),
            staged=("stages" in cr),
            resolved=DemandInfo(
                name=str(cr["name"]),
                N_kN=float(res.get("N_kN", 0.0)),
                Mx_kNm=float(res.get("Mx_kNm", 0.0)),
                My_kNm=float(res.get("My_kNm", 0.0)),
            ),
            stages=cr.get("stages"),
        ))

    envelopes_info = [
        EnvelopeInfo(name=str(er["name"]),
                     members=er.get("members", []),
                     eta_max=er.get("eta_max"))
        for er in ver["envelopes"]
    ]

    rows: list[VerificationRow] = []
    for r in ver["demands"]:
        rows.append(_build_verification_row(
            "demand", r["name"], r,
            forces={
                "N_kN": r.get("N_kN"),
                "Mx_kNm": r.get("Mx_kNm"),
                "My_kNm": r.get("My_kNm"),
            },
        ))
    for cr in ver["combinations"]:
        res = cr.get("resultant", {}) or {}
        rows.append(_build_verification_row(
            "combination", cr["name"], cr,
            staged=("stages" in cr),
            forces={
                "N_kN": res.get("N_kN"),
                "Mx_kNm": res.get("Mx_kNm"),
                "My_kNm": res.get("My_kNm"),
            },
        ))
    for er in ver["envelopes"]:
        rows.append(VerificationRow(
            kind="envelope",
            name=str(er["name"]),
            eta_governing=er.get("eta_max"),
            status=_status_from_eta(er.get("eta_max")),
        ))

    return AnalysisResult(
        materials=materials_info,
        section=section_info,
        properties=_section_properties_payload(
            getattr(session, "_properties", None)),
        demands=demands_info,
        combinations=combinations_info,
        envelopes=envelopes_info,
        verification=rows,
        domain=_domain_payload(session.domain),
        meta=Meta(),  # overwritten in analyze()
    )
