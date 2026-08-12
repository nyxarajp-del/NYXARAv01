"""The abyss engines over HTTP (`/v1/abyss/*`).

Hermetic: an offline core driven through FastAPI's in-process TestClient (no real sockets). These
routes are the last leg of the wiring — the engines were built on every boot and reachable only
from the embodied loop's private planner, so nothing outside the process could ask NYXARA what she
sees ahead. What matters here is that they answer, and that they stay honest when she cannot see:
a blind pass must come back as a reported refusal, never as a confident ranking of nothing.
"""

from __future__ import annotations

import random

import pytest
from pydantic import SecretStr

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from nyxara.kernel.config import NyxaraSettings, Profile  # noqa: E402
from nyxara.kernel.orchestrator import NyxaraCore  # noqa: E402
from nyxara.server.app import create_app  # noqa: E402

TOKEN = "test-secret-token"


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def core():
    return NyxaraCore(enable_memory=False, enable_tools=False, enable_skills=False)


@pytest.fixture(scope="module")
def client(core):
    s = NyxaraSettings.for_profile(Profile.DEV)
    s.server.api_token = SecretStr(TOKEN)
    return TestClient(create_app(core=core, settings=s))


def _teach(core, n: int = 60) -> None:
    rng = random.Random(0)
    for _ in range(n):
        x = rng.uniform(-5.0, 5.0)
        core.world_model.observe((x,), "read", (x + 1.0,), reward=1.0)
        core.world_model.observe((x,), "write", (x - 1.0,), reward=-1.0)


# -------------------- the routes exist and are guarded -------------------- #
def test_both_routes_require_auth(client):
    assert client.post("/v1/abyss/timelines", json={"actions": ["a", "b"]}).status_code == 401
    assert client.post("/v1/abyss/butterfly", json={}).status_code == 401


def test_timelines_reports_blindness_rather_than_ranking_nothing(client):
    got = client.post("/v1/abyss/timelines",
                      json={"actions": ["read", "write"], "state": [0.0]},
                      headers=_auth())
    assert got.status_code == 200
    body = got.json()
    assert body["ok"] is False and body["blind"] is True and "transitions" in body["reason"]


def test_butterfly_reports_blindness_too(client):
    got = client.post("/v1/abyss/butterfly", json={"state": [0.0]}, headers=_auth())
    assert got.status_code == 200
    body = got.json()
    assert body["ok"] is False and body["blind"] is True


def test_a_single_action_is_refused(client):
    got = client.post("/v1/abyss/timelines", json={"actions": ["alone"]}, headers=_auth())
    assert got.status_code == 200 and got.json()["ok"] is False


def test_the_branch_ceiling_is_validated_not_obeyed(client):
    got = client.post("/v1/abyss/timelines",
                      json={"actions": ["a", "b"], "branches": 10_000_000}, headers=_auth())
    assert got.status_code == 422           # rejected at the edge, never handed to the engine


# -------------------- once the model has learned -------------------- #
def test_timelines_ranks_the_options_once_she_has_lived_something(client, core):
    _teach(core)
    got = client.post("/v1/abyss/timelines",
                      json={"actions": ["read", "write"], "state": [0.0],
                            "branches": 200, "horizon": 4},
                      headers=_auth())
    assert got.status_code == 200
    body = got.json()
    assert body["ok"] is True and body["blind"] is False
    assert body["best_action"] == "read"
    assert body["total_branches"] == 200
    assert {"confidence", "expected_outcome", "action_rankings"} <= set(body)


def test_butterfly_ranks_the_dimensions_of_now(client, core):
    _teach(core)
    got = client.post("/v1/abyss/butterfly",
                      json={"state": [0.0], "horizon": 4, "labels": ["position"]},
                      headers=_auth())
    assert got.status_code == 200
    body = got.json()
    if not body["ok"]:                      # no embodied agent in this build
        pytest.skip(body["reason"])
    assert body["ranking"] and "amplification" in body["most_sensitive"]
