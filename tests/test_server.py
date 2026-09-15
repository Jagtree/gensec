"""
Tests for the FastAPI server (gensec.server).

These tests hit the HTTP layer with a TestClient -- no real uvicorn
process is spawned.  Heavy solver work is monkeypatched away so the
suite stays fast; separate integration tests cover the real pipeline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from gensec import api, server

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture
def client():
    """Plain TestClient over the module-level app."""
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# Contract tests (no solver)
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "gensec_version" in body
    assert "cache" in body


def test_analyze_rejects_empty_body(client):
    r = client.post("/api/analyze", json={})
    assert r.status_code == 422  # Pydantic validation


def test_analyze_400_on_malformed_yaml(client, monkeypatch):
    monkeypatch.setattr(api, "analyze",
                        MagicMock(side_effect=ValueError("bad YAML")))
    r = client.post("/api/analyze", json={"yaml_text": "::: not yaml"})
    assert r.status_code == 400
    assert "bad YAML" in r.json()["detail"]


def test_contour_validates_range(client):
    r = client.post("/api/contour", json={
        "yaml_text": "a: 1", "N_kN": 0.0, "n_angles": 4,
    })
    # n_angles < 8 is out of range
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Integration smoke (skipped if example missing)
# ---------------------------------------------------------------------------

@pytest.mark.timeout(60)
def test_analyze_round_trip(client):
    """End-to-end: YAML text -> JSON response with valid fields."""
    src = EXAMPLES / "biaxial_column.yaml"
    if not src.exists():
        pytest.skip("examples/biaxial_column.yaml missing")

    import yaml as _yaml
    data = _yaml.safe_load(src.read_text())
    data.setdefault("output", {})
    # Phase 38: ``data["output"]["n_points"] = 40`` used to sit here, put
    # there to keep this smoke test quick.  It never did anything --
    # nothing reads ``n_points`` from the output block; the solver takes
    # it from CLI arguments and gen_kwargs -- so the five flags below are
    # what has been making this test fast all along.  The line is gone
    # rather than wired, because wiring it would change what this test
    # computes.  It was found by the refusal added in this phase, which
    # is the refusal doing its job on the project's own suite.
    data["output"]["generate_3d_surface"] = False
    data["output"]["generate_mx_my"] = False
    data["output"]["generate_moment_curvature"] = False
    data["output"]["generate_polar_ductility"] = False
    data["output"]["generate_3d_moment_curvature"] = False
    yaml_text = _yaml.safe_dump(data)

    r = client.post("/api/analyze", json={"yaml_text": yaml_text})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["section"]["B_mm"] == 400
    assert body["section"]["H_mm"] == 600
    assert len(body["demands"]) >= 1
    assert "verification" in body
    assert "domain" in body
    assert "meta" in body


def test_second_analyze_is_cached(client):
    """Repeat analyze with same YAML should hit the LRU cache."""
    src = EXAMPLES / "biaxial_column.yaml"
    if not src.exists():
        pytest.skip("examples/biaxial_column.yaml missing")

    import yaml as _yaml
    data = _yaml.safe_load(src.read_text())
    data.setdefault("output", {})
    # Phase 38: ``data["output"]["n_points"] = 40`` used to sit here, put
    # there to keep this smoke test quick.  It never did anything --
    # nothing reads ``n_points`` from the output block; the solver takes
    # it from CLI arguments and gen_kwargs -- so the five flags below are
    # what has been making this test fast all along.  The line is gone
    # rather than wired, because wiring it would change what this test
    # computes.  It was found by the refusal added in this phase, which
    # is the refusal doing its job on the project's own suite.
    data["output"]["generate_3d_surface"] = False
    data["output"]["generate_mx_my"] = False
    data["output"]["generate_moment_curvature"] = False
    data["output"]["generate_polar_ductility"] = False
    data["output"]["generate_3d_moment_curvature"] = False
    yaml_text = _yaml.safe_dump(data)

    client.post("/api/analyze", json={"yaml_text": yaml_text})
    r2 = client.post("/api/analyze", json={"yaml_text": yaml_text})
    assert r2.status_code == 200
    assert r2.json()["meta"]["cached"] is True
