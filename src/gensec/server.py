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

# ---------------------------------------------------------------------------
# GenSec -- Copyright (c) 2026 Andrea Albero
# AGPL-3.0-or-later. See LICENSE file.
# ---------------------------------------------------------------------------

"""
FastAPI server exposing the GenSec analysis engine over HTTP.

The server is the HTTP adapter for :mod:`gensec.api` -- every endpoint
is a thin wrapper that forwards to one API function and returns its
Pydantic response as JSON.

No computation lives here.  All heavy lifting (parsing, solving,
verification, plotting) happens in :mod:`gensec.api`, which means the
server and a notebook and a desktop wrapper all behave identically.

Layout
------
The companion static frontend lives under ``web/`` at the repo root
and is mounted at ``/``.  The canonical user flow is::

    uv run gensec-gui
    # opens http://127.0.0.1:8765 in the browser

The GUI fetches ``/api/analyze``, ``/api/contour`` and friends.

Running manually
----------------
For development with auto-reload::

    uv run uvicorn gensec.server:app --reload --port 8765

For a one-shot production launch that also opens the browser::

    uv run gensec-gui
"""

from __future__ import annotations

import logging
import webbrowser
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gensec import api

log = logging.getLogger("gensec.server")


# ---------------------------------------------------------------------------
# Request body models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Body for ``POST /api/analyze``."""
    yaml_text: str = Field(..., description="Full YAML input as a string.")


class ContourRequest(BaseModel):
    """Body for ``POST /api/contour``."""
    yaml_text: str
    N_kN: float = Field(..., description="Axial force [kN].  Sign: "
                                          "negative = compression.")
    n_angles: int = Field(144, ge=8, le=720,
                          description="Angular resolution of the contour.")


class VerifyPointRequest(BaseModel):
    """Body for ``POST /api/verify-point``."""
    yaml_text: str
    N_kN: float
    Mx_kNm: float
    My_kNm: float = 0.0


class PlotRequest(BaseModel):
    """Body for ``POST /api/plot/{kind}``."""
    yaml_text: str
    width_px: int = 1200
    height_px: int = 800
    dpi: int = 150
    # Forwarded verbatim to the underlying plotting function.
    # Known keys: N_kN (for 'mxmy'), direction ('x'/'y' for 'mchi'), ...
    options: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def _resolve_web_dir() -> Optional[Path]:
    """Locate the static ``web/`` directory next to the repo root.

    Returns ``None`` if not found (headless deployments).  The lookup
    walks up from this file's location and from the current working
    directory, so the server works whether launched from the repo root
    or from an installed wheel with a bundled frontend.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent / "web",   # repo root / web
        Path.cwd() / "web",                   # cwd / web
    ]
    for p in candidates:
        if p.is_dir() and (p / "index.html").is_file():
            return p
    return None


def create_app(enable_cors: bool = False) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters
    ----------
    enable_cors : bool, default False
        When True, install a permissive CORS middleware.  Only needed
        if the frontend is served from a different origin (e.g. Vite
        dev server on :3000 during development).  Production deploys
        that bundle the frontend under ``web/`` should leave this off.
    """
    app = FastAPI(
        title="GenSec API",
        version=api.__version__,
        description=(
            "HTTP facade for the GenSec fiber-based cross-section "
            "analysis engine.  All endpoints accept YAML input as "
            "text and return JSON-serialisable results."
        ),
    )

    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ---- API routes ----

    @app.get("/api/health", tags=["meta"])
    def health() -> dict:
        """Liveness check.  Returns the GenSec API version."""
        cache = api._get_session.cache_info()
        return {
            "status": "ok",
            "gensec_version": api.__version__,
            "cache": {
                "size": cache.currsize,
                "max": cache.maxsize,
                "hits": cache.hits,
                "misses": cache.misses,
            },
        }

    @app.post("/api/analyze",
              response_model=api.AnalysisResult, tags=["analysis"])
    def analyze_endpoint(req: AnalyzeRequest) -> api.AnalysisResult:
        """Run the full GenSec analysis from YAML text.

        See :func:`gensec.api.analyze` for the response shape.
        """
        try:
            return api.analyze(yaml_text=req.yaml_text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("analyze failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/contour",
              response_model=api.ContourResponse, tags=["analysis"])
    def contour_endpoint(req: ContourRequest) -> api.ContourResponse:
        """Return a single Mx-My contour at a fixed axial force.

        Fast path used by the N-slider.  The underlying session is
        cached so successive slider positions are cheap.
        """
        try:
            return api.contour_at_N(
                N_kN=req.N_kN,
                yaml_text=req.yaml_text,
                n_angles=req.n_angles,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("contour failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/verify-point",
              response_model=api.PointVerificationResponse, tags=["analysis"])
    def verify_point_endpoint(
        req: VerifyPointRequest,
    ) -> api.PointVerificationResponse:
        """Verify an ad-hoc (N, Mx, My) demand without editing the YAML."""
        try:
            return api.verify_point(
                N_kN=req.N_kN,
                Mx_kNm=req.Mx_kNm,
                My_kNm=req.My_kNm,
                yaml_text=req.yaml_text,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("verify_point failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/plot/{kind}",
              response_model=api.PlotImageResponse, tags=["plots"])
    def render_plot_endpoint(
        kind: str, req: PlotRequest,
    ) -> api.PlotImageResponse:
        """Render a matplotlib plot as a base64-encoded PNG.

        ``kind`` is one of: ``mxmy``, ``nm``, ``mchi``, ``surface``,
        ``polar``, ``section``.  Extra options (e.g. ``N_kN`` for the
        Mx-My plot) are passed through the ``options`` body field.
        """
        try:
            return api.render_plot(
                kind=kind,                       # type: ignore[arg-type]
                yaml_text=req.yaml_text,
                width_px=req.width_px,
                height_px=req.height_px,
                dpi=req.dpi,
                **req.options,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("render_plot failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ---- Static frontend (mounted last so /api/* wins) ----

    web_dir = _resolve_web_dir()
    if web_dir is not None:
        app.mount(
            "/",
            StaticFiles(directory=str(web_dir), html=True),
            name="web",
        )
        log.info("Serving frontend from %s", web_dir)
    else:
        log.warning(
            "No web/ directory found next to the repo root. "
            "API routes work but there is no GUI to serve."
        )

    return app


# Module-level app for ``uvicorn gensec.server:app``.
app = create_app(enable_cors=False)


# ---------------------------------------------------------------------------
# Console entry point: ``gensec-gui``
# ---------------------------------------------------------------------------

def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    reload: bool = False,
) -> None:
    """Launch the server and (optionally) open the GUI in a browser.

    This is the function bound to the ``gensec-gui`` console script
    in ``pyproject.toml``.

    Parameters
    ----------
    host : str, default "127.0.0.1"
        Loopback by default -- desktop app use case, no LAN exposure.
        Pass "0.0.0.0" to allow LAN access.
    port : int, default 8765
    open_browser : bool, default True
        Open ``http://host:port/`` in the default browser after a
        short delay, so the user lands on the GUI automatically.
    reload : bool, default False
        Enable uvicorn's auto-reload on code changes.  Developer mode.
    """
    import argparse
    import threading
    import time

    import uvicorn

    # argparse overrides when called as a script with flags.
    parser = argparse.ArgumentParser(
        prog="gensec-gui",
        description="Launch the GenSec graphical interface.",
    )
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open the browser automatically.")
    parser.add_argument("--reload", action="store_true",
                        help="Developer mode: reload on code change.")
    args, _ = parser.parse_known_args()

    url = f"http://{args.host}:{args.port}/"

    if open_browser and not args.no_browser:
        def _open_when_ready() -> None:
            time.sleep(1.2)   # give uvicorn time to bind
            try:
                webbrowser.open(url)
            except Exception:
                log.warning("Could not open browser; navigate to %s", url)
        threading.Thread(target=_open_when_ready, daemon=True).start()

    print(f"\n  GenSec GUI -> {url}")
    print(f"  Press Ctrl+C to stop.\n")

    uvicorn.run(
        "gensec.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload or reload,
        log_level="info",
    )


if __name__ == "__main__":
    serve()
