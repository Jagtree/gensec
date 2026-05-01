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
do not.  Cache size is bounded by ``SECTION_CACHE_SIZE`` (default 32).

Examples
--------
>>> from gensec.api import analyze
>>> result = analyze(yaml_path="examples/biaxial_column.yaml")
>>> result.verification[0].eta_3D
0.41
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional, Union
from gensec._version import __version__  # noqa: F401

import yaml as _yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SECTION_CACHE_SIZE: int = 32
"""Maximum number of distinct sections held in the in-memory cache."""

#__version__ = "0.3.0"


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
    """One row of the verification table."""
    kind: Literal["demand", "combination", "envelope"]
    name: str
    N_kN: Optional[float] = None
    Mx_kNm: Optional[float] = None
    My_kNm: Optional[float] = None
    eta_3D: Optional[float] = None
    eta_2D: Optional[float] = None
    eta_path: Optional[float] = None
    eta_path_2D: Optional[float] = None
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
    demands: list[DemandInfo]
    combinations: list[CombinationInfo]
    envelopes: list[EnvelopeInfo]
    verification: list[VerificationRow]
    domain: DomainPayload
    meta: Meta


class ContourResponse(BaseModel):
    N_kN: float
    points: list[tuple[float, float]]
    meta: Meta


class PointVerificationResponse(BaseModel):
    N_kN: float
    Mx_kNm: float
    My_kNm: float
    eta_3D: Optional[float] = None
    eta_2D: Optional[float] = None
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
        "domain", "verification", "nm_3d",
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
        """Parse YAML, build section & solver, precompute full domain.

        Mirrors the pipeline in ``gensec.cli._run`` but keeps results in
        memory.  Heavy: runs once per unique YAML.
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

        solver    = FiberSolver(section)
        is_biaxial = int(section.n_fibers_x) > 1

        nm_gen    = NMDiagram(solver)
        n_points      = int(opts.get("n_points", 400))
        n_angles_3d   = int(opts.get("n_angles_3d_surface", 72))

        nm_data   = nm_gen.generate(n_points=n_points)
        nm_data_y = nm_gen.generate(n_points=n_points, direction="y") if is_biaxial else None

        domain_data = nm_data   # use uniaxial diagram for verification
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
                # Unresolved reference — skip silently; surface later
                # via warnings if needed.
                pass

        envelope_results: list[dict] = []
        for env in envelopes:
            try:
                envelope_results.append(
                    engine.check_envelope(env, demand_db, combination_db))
            except KeyError:
                pass

        return cls(
            yaml_data=data,
            section=section,
            solver=solver,
            nmdiagram=nm_gen,
            domain=dict(
                nm_data=nm_data,
                nm_data_y=nm_data_y,
                nm_3d=None,
                is_biaxial=is_biaxial,
            ),
            verification=dict(
                demands=demand_results,
                combinations=combination_results,
                envelopes=envelope_results,
                engine=engine,
            ),
        )

def get_nm_3d(self, n_angles: int = 36, n_points_per_angle: int = 50):
    """Lazy: compute the 3D surface the first time it's asked."""
    if self.domain.get("nm_3d") is None and self.domain.get("is_biaxial"):
        self.domain["nm_3d"] = self.nmdiagram.generate_biaxial(
            n_angles=n_angles,
            n_points_per_angle=n_points_per_angle,
        )
    return self.domain.get("nm_3d")

def clear_cache() -> None:
    """Evict every cached session.  Call after shutdown or in tests."""
    _get_session.cache_clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    """Return a single Mx-My interaction contour at a fixed axial force.

    Fast path used by the N-slider in the GUI.  Reuses the cached
    section/hull; only the slice is recomputed.

    Parameters
    ----------
    N_kN : float
        Axial force [kN].  Sign: negative = compression.
    n_angles : int, default 144
    n_points_per_angle : int, default 200
    """
    t0 = time.perf_counter()
    session, cached = _session_for(yaml_text, yaml_path)

    mx_my = session.nmdiagram.generate_mx_my(
        N_fixed=float(N_kN) * 1e3,
        n_angles=n_angles,
        n_points_per_angle=n_points_per_angle,
    )
    # generate_mx_my returns dict with "Mx_kNm" / "My_kNm" arrays.
    mx = _as_list(mx_my.get("Mx_kNm", []))
    my = _as_list(mx_my.get("My_kNm", []))
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

    # Reuse the same engine the Session built.  VerificationEngine
    # expects demands in N / N*mm with the CLI's dict shape.
    demand = {
        "name": "probe",
        "N":  float(N_kN)  * 1e3,
        "Mx": float(Mx_kNm) * 1e6,
        "My": float(My_kNm) * 1e6,
    }
    engine = session.verification["engine"]
    rows = engine.check_demands([demand])
    row = rows[0] if rows else {}

    eta_3D = row.get("eta_3D")
    eta_2D = row.get("eta_2D")
    status = _status_from_eta(eta_3D if eta_3D is not None else eta_2D)

    return PointVerificationResponse(
        N_kN=float(N_kN),
        Mx_kNm=float(Mx_kNm),
        My_kNm=float(My_kNm),
        eta_3D=eta_3D,
        eta_2D=eta_2D,
        status=status,
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
    """Render a matplotlib plot as a base64-encoded PNG.

    Reuses the functions in :mod:`gensec.output` so images here are
    identical to CLI output.

    Parameters
    ----------
    kind : {'mxmy', 'nm', 'nm_y', 'mchi', 'surface', 'polar', 'section'}
    width_px, height_px, dpi : raster controls.
    **kwargs : forwarded to the underlying plotter
        (e.g. ``N_kN=-2000`` for 'mxmy' / 'polar' / 'mchi').
    """
    # Matplotlib Agg backend — no display needed.
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
            fig = plot_nm_diagram(dom["nm_data"])
        elif kind == "nm_y":
            if dom["nm_data_y"] is None:
                raise ValueError("N-My plot requires a biaxial section.")
            fig = plot_nm_diagram(
                dom["nm_data_y"], title="N-My Interaction Diagram")
        elif kind == "surface":
            if dom["nm_3d"] is None:
                raise ValueError("3D surface requires a biaxial section.")
            fig = plot_3d_surface(
                dom["nm_3d"],
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

    # Concrete-ish: fcd, Ec, eps_cu2, optional EC2 class.
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

    # Steel-ish: fyd, Es, eps_su.
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
    """Serialise the numeric domain into the public shape."""
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
        # Units of the returned arrays are whatever generate_biaxial
        # emits; the CLI treats "_kN" / "_kNm" suffixes as already
        # converted, plain keys as SI.  We expose both when present.
        for out_key, candidates in (
            ("N_kN",  ("N_kN",  "N")),
            ("Mx_kNm", ("Mx_kNm", "Mx")),
            ("My_kNm", ("My_kNm", "My")),
        ):
            for c in candidates:
                if c in s:
                    vals = _as_list(s[c])
                    # If coming from raw SI, convert.
                    if c in ("N",):
                        vals = [v / 1e3 for v in vals]
                    elif c in ("Mx", "My"):
                        vals = [v / 1e6 for v in vals]
                    surface[out_key] = vals
                    break

    return DomainPayload(
        nm=nm_points, nm_y=nm_y_points,
        mxmy={}, surface=surface, mchi=[],
    )


def _build_analysis_result(session: "_Session") -> AnalysisResult:
    """Convert a built Session into the public payload.  Pure data shuffling."""
    raw = session.yaml_data
    ver = session.verification

    materials_info = [_material_info(mid, m)
                      for mid, m in raw["materials"].items()]
    section_info = _section_info(session.section)

    demands_info = [
        DemandInfo(name=d["name"],
                   N_kN=d["N"] / 1e3,
                   Mx_kNm=d["Mx"] / 1e6,
                   My_kNm=d["My"] / 1e6)
        for d in raw.get("demands", [])
    ]

    combinations_info: list[CombinationInfo] = []
    for cr in ver["combinations"]:
        res = cr.get("resultant", {}) or {}
        combinations_info.append(CombinationInfo(
            name=cr["name"],
            staged=("stages" in cr),
            resolved=DemandInfo(
                name=cr["name"],
                N_kN=float(res.get("N_kN", 0.0)),
                Mx_kNm=float(res.get("Mx_kNm", 0.0)),
                My_kNm=float(res.get("My_kNm", 0.0)),
            ),
            stages=cr.get("stages"),
        ))

    envelopes_info = [
        EnvelopeInfo(name=er["name"],
                     members=er.get("members", []),
                     eta_max=er.get("eta_max"))
        for er in ver["envelopes"]
    ]

    rows: list[VerificationRow] = []
    for r in ver["demands"]:
        eta_gov = r.get("eta_3D") if r.get("eta_3D") is not None \
                  else r.get("eta_2D")
        rows.append(VerificationRow(
            kind="demand", name=r["name"],
            N_kN=r.get("N_kN"), Mx_kNm=r.get("Mx_kNm"),
            My_kNm=r.get("My_kNm"),
            eta_3D=r.get("eta_3D"), eta_2D=r.get("eta_2D"),
            status=_status_from_eta(eta_gov),
        ))
    for cr in ver["combinations"]:
        res = cr.get("resultant", {}) or {}
        eta_gov = cr.get("eta_governing")
        rows.append(VerificationRow(
            kind="combination", name=cr["name"],
            N_kN=res.get("N_kN"), Mx_kNm=res.get("Mx_kNm"),
            My_kNm=res.get("My_kNm"),
            eta_3D=cr.get("eta_3D"), eta_2D=cr.get("eta_2D"),
            eta_path=cr.get("eta_path"),
            eta_path_2D=cr.get("eta_path_2D"),
            status=_status_from_eta(eta_gov),
            staged=("stages" in cr),
        ))
    for er in ver["envelopes"]:
        rows.append(VerificationRow(
            kind="envelope", name=er["name"],
            eta_3D=er.get("eta_max"),
            status=_status_from_eta(er.get("eta_max")),
        ))

    return AnalysisResult(
        materials=materials_info,
        section=section_info,
        demands=demands_info,
        combinations=combinations_info,
        envelopes=envelopes_info,
        verification=rows,
        domain=_domain_payload(session.domain),
        meta=Meta(),  # overwritten in analyze()
    )
