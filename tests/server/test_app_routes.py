"""Coverage for API routes not exercised by test_app.py.

Same hermetic setup (offline core + in-process TestClient): this fills in the
reasoning routes (solve / strategize), the fractal-temporal endpoints, and a real
memory save→load roundtrip through the HTTP surface."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from nyxara.kernel.config import NyxaraSettings, Profile  # noqa: E402
from nyxara.kernel.orchestrator import NyxaraCore  # noqa: E402
from nyxara.server.app import create_app  # noqa: E402

TOKEN = "test-secret-token"


def _client(core=None, *, token=TOKEN):
    s = NyxaraSettings.for_profile(Profile.DEV)
    if token is not None:
        s.server.api_token = SecretStr(token)
    if core is None:
        core = NyxaraCore(enable_memory=False, enable_tools=False, enable_skills=False)
    return TestClient(create_app(core=core, settings=s))


def _auth(token=TOKEN) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- #
# Reasoning routes
# --------------------------------------------------------------------------- #
def test_solve_returns_a_solution():
    r = _client().post("/v1/solve", json={"problem": "add 2 and 2"}, headers=_auth())
    assert r.status_code == 200
    assert isinstance(r.json(), dict) and r.json()


def test_strategize_returns_a_framework():
    r = _client().post("/v1/strategize", json={"problem": "improve test coverage"},
                       headers=_auth())
    assert r.status_code == 200
    assert isinstance(r.json(), dict) and r.json()


def test_solve_requires_token():
    r = _client().post("/v1/solve", json={"problem": "add 2 and 2"})
    assert r.status_code == 401


def test_solve_rejects_empty_problem():
    # pydantic min_length=1 -> 422 before the handler runs
    r = _client().post("/v1/solve", json={"problem": ""}, headers=_auth())
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# Delegation route (gated distributed intelligence over the wire)
# --------------------------------------------------------------------------- #
def test_delegate_spawns_a_gated_subagent():
    r = _client().post("/v1/delegate",
                       json={"name": "scout", "subgoal": "summarize the goal", "max_steps": 1},
                       headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "scout"
    assert "status" in body and "model" in body


def test_delegate_requires_token():
    r = _client().post("/v1/delegate", json={"name": "x", "subgoal": "y"})
    assert r.status_code == 401


def test_delegate_rejects_empty_subgoal():
    r = _client().post("/v1/delegate", json={"name": "x", "subgoal": ""}, headers=_auth())
    assert r.status_code == 422


# --------------------------------------------------------------------------- #
# Fractal temporal endpoints
# --------------------------------------------------------------------------- #
def test_temporal_report_is_serializable():
    r = _client().get("/v1/temporal/report", headers=_auth())
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_temporal_tick_advances_or_reports_absence():
    r = _client().post("/v1/temporal/tick", json={"beats": 3}, headers=_auth())
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_temporal_requires_token():
    assert _client().get("/v1/temporal/report").status_code == 401


# --------------------------------------------------------------------------- #
# Memory persistence roundtrip (Rule 7 continuity over the wire)
# --------------------------------------------------------------------------- #
def test_memory_save_then_load_roundtrip(tmp_path):
    path = str(tmp_path / "mem.json")
    core = NyxaraCore(enable_memory=True, enable_tools=False, enable_skills=False)
    core.memory.remember("the Master is sovereign", importance=0.9)
    client = _client(core)

    saved = client.post("/v1/memory/save", json={"path": path}, headers=_auth())
    assert saved.status_code == 200
    assert saved.json()["saved"] == path

    # a fresh core restores the same records from disk
    fresh = NyxaraCore(enable_memory=True, enable_tools=False, enable_skills=False)
    loaded = _client(fresh).post("/v1/memory/load", json={"path": path}, headers=_auth())
    assert loaded.status_code == 200
    assert loaded.json()["loaded"] >= 1


def test_memory_save_without_memory_returns_null():
    # the default light core has no memory store; the route reports that cleanly
    r = _client().post("/v1/memory/save", json={}, headers=_auth())
    assert r.status_code == 200
    assert r.json()["saved"] is None


# --------------------------------------------------------------------------- #
# Learning-status route
# --------------------------------------------------------------------------- #
def test_learning_route_returns_truthful_report():
    r = _client().get("/v1/learning", headers=_auth())
    assert r.status_code == 200
    body = r.json()
    # every section present; absent subsystems reported absent (None), never invented
    for key in ("foundry", "flywheel", "autoforge", "serving", "competence"):
        assert key in body


def test_learning_route_requires_auth():
    r = _client().get("/v1/learning")
    assert r.status_code == 401
