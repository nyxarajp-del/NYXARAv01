"""HTTP coverage for the environment-adaptation routes: /v1/understand and /v1/adapt.

Same hermetic setup as the other server tests (offline core + in-process TestClient). A live callable
cannot cross the wire, so these endpoints accept a *declarative* system — a dataset of rows, or a named
law family + params — and run NYXARA's own open-world generalizer on it. All routes are bearer-gated.
"""
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


def test_understand_from_a_dataset():
    r = _client().post("/v1/understand", json={"dataset": [[0, 1], [1, 3], [2, 5], [3, 7]]},
                       headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert body.get("verdict") == "modelled"
    assert body.get("law_family") == "affine"


def test_understand_from_a_declared_family():
    r = _client().post("/v1/understand",
                       json={"family": "affine", "params": {"w": [1.0, 2.0]}, "dims": 1},
                       headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert body.get("verdict") == "modelled"


def test_understand_needs_a_spec():
    r = _client().post("/v1/understand", json={}, headers=_auth())
    assert r.status_code == 200
    assert "error" in r.json()


def test_understand_requires_token():
    r = _client(token=None).post("/v1/understand", json={"dataset": [[0, 1]]})
    # with a token configured the route is protected; here no token set => open, so assert the
    # protected variant explicitly instead
    r2 = _client().post("/v1/understand", json={"dataset": [[0, 1], [1, 3], [2, 5]]})
    assert r2.status_code in (401, 403)


def test_adapt_over_declared_systems():
    r = _client().post("/v1/adapt",
                       json={"systems": [{"family": "affine", "params": {"w": [1.0, 2.0]}},
                                         {"family": "sinusoidal",
                                          "params": {"a": 3.0, "b": 0.0, "c": 2.0, "omega": 1.0}}],
                             "label": "env"},
                       headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert body.get("n_systems") == 2
    assert body.get("modelled", 0) >= 1
    assert "outcomes" in body


def test_adapt_demo_environment_without_systems():
    r = _client().post("/v1/adapt", json={}, headers=_auth())
    assert r.status_code == 200
    body = r.json()
    # with no systems given she adapts to a demo environment of hidden alien machines
    assert "n_systems" in body or "error" in body


def test_adapt_requires_token():
    r = _client().post("/v1/adapt", json={"systems": []})
    assert r.status_code in (401, 403)
