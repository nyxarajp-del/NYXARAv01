"""Tests for the HTTP/WebSocket API server (nyxara/server/app.py).

Hermetic: the core runs on the deterministic offline reasoner (no network), and the app is
driven through FastAPI's in-process TestClient (no real sockets)."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from nyxara.kernel.config import NyxaraSettings, Profile  # noqa: E402
from nyxara.kernel.orchestrator import NyxaraCore  # noqa: E402
from nyxara.server.app import create_app  # noqa: E402

TOKEN = "test-secret-token"


def _core() -> NyxaraCore:
    # light core: transport is what's under test, not the heavy faculties
    return NyxaraCore(enable_memory=False, enable_tools=False, enable_skills=False)


def _client(*, token=TOKEN, profile=Profile.DEV, enable_control=True, core=None):
    s = NyxaraSettings.for_profile(profile)
    if token is not None:
        s.server.api_token = SecretStr(token)
    s.server.enable_control = enable_control
    app = create_app(core=core or _core(), settings=s)
    return TestClient(app)


def _auth(token=TOKEN) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- #
# Health & auth
# --------------------------------------------------------------------------- #
def test_health_is_open():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chat_requires_token():
    r = _client().post("/v1/chat", json={"message": "hello"})
    assert r.status_code == 401


def test_chat_rejects_bad_token():
    r = _client().post("/v1/chat", json={"message": "hi"}, headers=_auth("wrong"))
    assert r.status_code == 401


def test_chat_with_token_runs_a_turn():
    r = _client().post("/v1/chat", json={"message": "hello NYXARA"}, headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert "disposition" in body and "response" in body
    assert body["response"]


def test_open_mode_allows_unauthenticated_chat():
    # no token configured (dev convenience) -> no header required
    r = _client(token=None).post("/v1/chat", json={"message": "hi"})
    assert r.status_code == 200


# --------------------------------------------------------------------------- #
# Report / agent
# --------------------------------------------------------------------------- #
def test_report():
    r = _client().get("/v1/report", headers=_auth())
    assert r.status_code == 200
    assert "control" in r.json()


def test_agent_runs_and_caps_steps():
    r = _client().post("/v1/agent", json={"goal": "say hello", "max_steps": 99},
                       headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert "status" in body and "final_answer" in body
    assert body["n_steps"] <= 8  # capped by server.max_agent_steps default


# --------------------------------------------------------------------------- #
# Sovereign control
# --------------------------------------------------------------------------- #
def test_control_pause_resume_scram():
    c = _client()
    assert c.post("/v1/control/pause", json={}, headers=_auth()).json()["control"] == "paused"
    assert c.post("/v1/control/resume", json={}, headers=_auth()).json()["control"] == "running"
    out = c.post("/v1/control/scram", json={"reason": "test"}, headers=_auth()).json()
    assert out["control"] == "stopped"


def test_unknown_control_action_404():
    r = _client().post("/v1/control/spin", json={}, headers=_auth())
    assert r.status_code == 404


def test_control_can_be_disabled():
    r = _client(enable_control=False).post("/v1/control/pause", json={}, headers=_auth())
    assert r.status_code == 404  # route not registered


# --------------------------------------------------------------------------- #
# WebSocket
# --------------------------------------------------------------------------- #
def test_websocket_chat():
    with _client().websocket_connect(f"/v1/ws?token={TOKEN}") as ws:
        ws.send_json({"message": "hello over the socket"})
        data = ws.receive_json()
        assert "disposition" in data and data["response"]


def test_websocket_rejects_bad_token():
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with _client().websocket_connect("/v1/ws?token=nope") as ws:
            ws.receive_json()


# --------------------------------------------------------------------------- #
# Fail-closed posture
# --------------------------------------------------------------------------- #
def test_prod_refuses_to_start_without_token():
    s = NyxaraSettings.for_profile(Profile.PROD)
    s.server.api_token = None
    with pytest.raises(RuntimeError):
        create_app(core=_core(), settings=s)


# --------------------------------------------------------------------------- #
# Boot provisioning — every front door, not just the one a person sits at
# --------------------------------------------------------------------------- #
def test_the_server_provisions_her_own_brains_on_startup(monkeypatch):
    """The daemon and the HTTP server are how she runs unattended, and they never bootstrapped.

    ``ensure_primary_model`` — install the cortex runtime, fetch the weights, warm them, forge the
    self model — ran only in ``__main__.main``, the interactive console. Nothing failed without
    it: the ladder simply walked past the missing rung and answered on the n-gram floor, correctly
    and silently, for the whole life of the process. "Boot her and it just works" has to mean
    every front door.
    """
    calls: list[str] = []
    monkeypatch.setattr("nyxara.growth.bootstrap.ensure_primary_model",
                        lambda **kw: calls.append("provisioned"))
    with _client():
        pass
    assert calls == ["provisioned"]


def test_a_failed_provision_never_blocks_the_port(monkeypatch):
    """A cold or offline box must still serve, one rung lower, rather than not serve at all."""
    def _boom(**kw):
        raise RuntimeError("no network, no weights, no runtime")

    monkeypatch.setattr("nyxara.growth.bootstrap.ensure_primary_model", _boom)
    with _client() as client:
        assert client.get("/health").status_code in (200, 401)
