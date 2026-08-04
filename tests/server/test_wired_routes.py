"""HTTP routes for the layers that were wired into the mind but had no API surface.

Every faculty here existed and was unit-tested; none of it was reachable over HTTP, which for
a daemon-shaped process means it was not reachable at all. These tests pin the routes and,
more importantly, pin the *honesty* of what they return: an undemonstrated capability must not
come back as a confident yes, a moral reading must expose disagreement rather than hide it, and
a training call with no trained brain must say why instead of raising or pretending.

Follows ``tests/server/test_new_layer_routes.py`` in shape.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="the API server needs the [server] extra")
from fastapi.testclient import TestClient  # noqa: E402

from nyxara.server.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


# --------------------------------------------------------------------------- #
# The defensive layer
# --------------------------------------------------------------------------- #
def test_guard_status_reports_every_defensive_faculty(client: TestClient) -> None:
    body = client.get("/v1/guard").json()
    assert {"anomaly", "containment", "network", "survival", "auth", "immune"} <= set(body)
    assert body["posture"]


def test_guard_anomalies_returns_the_watchtower_log(client: TestClient) -> None:
    body = client.get("/v1/guard/anomalies").json()
    assert body["enabled"] is True
    assert isinstance(body["recent"], list)


def test_contain_then_release_round_trips(client: TestClient) -> None:
    contained = client.post("/v1/guard/contain",
                            json={"component": "network", "action": "isolate"})
    assert contained.status_code == 200

    status = client.get("/v1/guard").json()
    assert status["containment"]["by_state"], "containment reported no state change"

    released = client.post("/v1/guard/contain",
                           json={"component": "network", "action": "release"})
    assert released.status_code == 200


def test_contain_rejects_an_unknown_component_as_400(client: TestClient) -> None:
    """A bad component name is the caller's error, not a server fault."""
    resp = client.post("/v1/guard/contain", json={"component": "no_such_thing"})
    assert resp.status_code == 400


def test_network_policy_denies_a_host(client: TestClient) -> None:
    resp = client.post("/v1/guard/network", json={"host": "hostile.example", "action": "deny"})
    assert resp.status_code == 200
    assert resp.json()["host"] == "hostile.example"


def test_phagocytose_turns_a_sample_into_an_antibody(client: TestClient) -> None:
    before = client.get("/v1/guard").json()["immune"]["antibodies"]
    resp = client.post("/v1/guard/phagocytose",
                       json={"sample": "import os; os.system('rm -rf /')", "kind": "malware"})
    assert resp.status_code == 200
    assert resp.json()["antibodies"] > before


def test_survival_reports_that_correction_outranks_it(client: TestClient) -> None:
    body = client.get("/v1/guard/survival").json()
    assert body["enabled"] is True
    assert body["correction_dominates_survival"] is True


# --------------------------------------------------------------------------- #
# Self-knowledge & provenance
# --------------------------------------------------------------------------- #
def test_capabilities_lists_what_she_can_do(client: TestClient) -> None:
    body = client.get("/v1/capabilities").json()
    assert body["enabled"] is True
    assert "can_do" in body and "calibration" in body


def test_can_i_does_not_overclaim(client: TestClient) -> None:
    """The whole point of the registry: no confident yes without demonstrated evidence."""
    unknown = client.post("/v1/capabilities/can_i", json={"capability": "time_travel"}).json()
    assert unknown.get("can") is not True

    known = client.post("/v1/capabilities/can_i", json={"capability": "reason"}).json()
    assert known["known"] is True
    assert "untested" in known["claim"].lower() or known.get("verified") is True


def test_lineage_chain_verifies(client: TestClient) -> None:
    body = client.get("/v1/lineage").json()
    assert body["enabled"] is True
    assert body["chain_verified"] is True


def test_seed_returns_an_identity_not_the_bundle(client: TestClient) -> None:
    """The signed bundle must never cross the wire — only its identity and size."""
    body = client.post("/v1/seed").json()
    assert body["seed"] and body["bytes"] > 0
    assert "bundle" not in body


# --------------------------------------------------------------------------- #
# The training stack — honest failure, not exceptions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path,payload", [
    ("/v1/train/dpo", {}),
    ("/v1/train/expand", {}),
    ("/v1/speculate", {"prompt": "hello"}),
])
def test_training_routes_report_why_they_cannot_run(client: TestClient, path: str,
                                                    payload: dict) -> None:
    resp = client.post(path, json=payload)
    assert resp.status_code == 200, "a missing brain is data, not a server error"
    body = resp.json()
    assert body.get("ok") is False and body.get("error")


def test_module_load_refuses_a_path_outside_the_forged_root(client: TestClient) -> None:
    body = client.post("/v1/module/load", json={"path": "/etc/passwd"}).json()
    assert body.get("ok") is False


# --------------------------------------------------------------------------- #
# Delegation & the mesh
# --------------------------------------------------------------------------- #
def test_agents_status(client: TestClient) -> None:
    body = client.get("/v1/agents").json()
    assert body["enabled"] is True
    assert "max_agents" in body


def test_cluster_is_single_node_until_a_peer_is_paired(client: TestClient) -> None:
    body = client.get("/v1/cluster").json()
    assert body["enabled"] is True
    assert body["single_node"] is True


# --------------------------------------------------------------------------- #
# Taste, register, conscience, foresight, determinism
# --------------------------------------------------------------------------- #
def test_aesthetic_judges_across_dimensions(client: TestClient) -> None:
    body = client.post("/v1/aesthetic", json={"artifact": "def f(x): return x * 2"}).json()
    assert "dimensions" in body or "score" in body


def test_aesthetic_rejects_an_unknown_artifact_kind(client: TestClient) -> None:
    resp = client.post("/v1/aesthetic", json={"artifact": "x", "kind": "not_a_kind"})
    assert resp.status_code == 400


def test_mode_sets_the_register(client: TestClient) -> None:
    body = client.post("/v1/mode", json={"mode": "analyst"}).json()
    assert body


def test_mode_rejects_an_unknown_name_with_the_valid_ones(client: TestClient) -> None:
    resp = client.post("/v1/mode", json={"mode": "wizard"})
    assert resp.status_code == 400
    assert "expected one of" in resp.json()["detail"]


def test_moral_surfaces_disagreement_rather_than_hiding_it(client: TestClient) -> None:
    body = client.post("/v1/moral",
                       json={"action": "delete a user's backups without asking"}).json()
    assert isinstance(body["contested"], bool)
    assert body["summary"]


def test_foresight_projects_a_future_self(client: TestClient) -> None:
    body = client.post("/v1/foresight", json={"horizon_days": 30}).json()
    assert body


def test_thermal_reports_a_compute_budget(client: TestClient) -> None:
    body = client.get("/v1/thermal").json()
    assert body["enabled"] is True
    assert 0.0 <= float(body["budget"]) <= 1.0


def test_replay_status_and_save(client: TestClient, tmp_path) -> None:
    status = client.get("/v1/replay").json()
    assert status["enabled"] is True
    saved = client.post("/v1/replay/save", json={"path": str(tmp_path / "tape.jsonl")})
    assert saved.status_code == 200
    assert (tmp_path / "tape.jsonl").exists()
