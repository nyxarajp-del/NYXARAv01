"""NYX V.02 over HTTP — the nineteen, reachable, and still unable to act by surprise.

Every faculty in ``nyxara/nyx/`` was unit-tested and none of it was reachable over the wire,
which for a daemon-shaped process means it was not reachable at all. These tests pin the routes
and — the part that matters — pin their **honesty**: a claim that admits no proof must come back
INEXPRESSIBLE rather than with a verdict, an unidentified causal pair must say so rather than
returning its co-occurrence weight wearing the word *cause*, and the two routes that can touch
the world must say what she *would* do unless the caller explicitly asks otherwise.

Follows ``tests/server/test_wired_routes.py`` in shape.
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
# What was actually asked for
# --------------------------------------------------------------------------- #
def test_intent_returns_the_structure_of_the_instruction(client: TestClient) -> None:
    body = client.post("/v1/nyx/intent",
                       json={"text": "fix the code but first run the tests"}).json()
    assert body["kind"] == "command"
    assert body["ordering"]                     # the half V.01 dropped entirely


def test_intent_carries_the_question_it_would_ask(client: TestClient) -> None:
    body = client.post("/v1/nyx/intent", json={"text": "delete some files"}).json()
    assert "clarifying_question" in body


def test_intent_reads_hinglish_as_a_command(client: TestClient) -> None:
    body = client.post("/v1/nyx/intent",
                       json={"text": "mera code fix kar do lekin pehle test chala"}).json()
    assert body["kind"] == "command"


# --------------------------------------------------------------------------- #
# Reasoning, with its tier on the answer
# --------------------------------------------------------------------------- #
def test_reason_names_the_tier_that_answered(client: TestClient) -> None:
    body = client.post("/v1/nyx/reason", json={
        "text": "apple -> APPLE!\nmango -> MANGO!\ncar -> ?"}).json()
    assert body["answered"] is True
    assert body["tier"] == "induced"
    assert "verifiable" in body


def test_reason_that_declines_names_the_tiers_it_tried(client: TestClient) -> None:
    """A reasoned abstention is a real result, and it is not a fabricated answer."""
    body = client.post("/v1/nyx/reason",
                       json={"text": "what did the Master eat for breakfast"}).json()
    assert body["answered"] is False
    assert "no tier could answer" in body["why"]


def test_learn_induces_a_procedure_and_answers_the_probe(client: TestClient) -> None:
    body = client.post("/v1/nyx/learn", json={
        "text": "apple -> APPLE!\nmango -> MANGO!\ncar -> ?"}).json()
    assert body["answer"] == "CAR!"


def test_learn_refuses_outside_its_operation_set_rather_than_guessing(
        client: TestClient) -> None:
    body = client.post("/v1/nyx/learn", json={"text": "hello there"}).json()
    assert not body.get("answer")


# --------------------------------------------------------------------------- #
# The proof core — the most important reply here is the one with no verdict
# --------------------------------------------------------------------------- #
def test_prove_returns_a_real_verdict_where_one_exists(client: TestClient) -> None:
    body = client.post("/v1/nyx/prove", json={"text": "for all real x, x*x >= 0"}).json()
    assert body["verdict"] == "proved" and body["proved"] is True


def test_prove_refuses_a_claim_that_is_not_a_formula(client: TestClient) -> None:
    """"Eliminate hallucination mathematically" is not claimed, and this is why."""
    body = client.post("/v1/nyx/prove", json={"text": "gravity pulls the apple down"}).json()
    assert body["verdict"] == "inexpressible"
    assert body["proved"] is False


def test_prove_does_not_call_a_contingent_claim_false(client: TestClient) -> None:
    body = client.post("/v1/nyx/prove", json={"text": "x > 5"}).json()
    assert body["verdict"] == "contingent"


# --------------------------------------------------------------------------- #
# The Skeptic
# --------------------------------------------------------------------------- #
def test_debate_attacks_the_claim_and_reports_what_landed(client: TestClient) -> None:
    body = client.post("/v1/nyx/debate",
                       json={"claim": "all cars are always fast", "confidence": 0.8}).json()
    assert body["debated"] is True and body["objections"]
    assert body["confidence_after"] < body["confidence_before"]


def test_debate_leaves_a_defensible_claim_standing(client: TestClient) -> None:
    body = client.post("/v1/nyx/debate", json={
        "claim": "the flywheel smooths the engine because its inertia resists a speed change",
        "confidence": 0.6}).json()
    assert body["survived"] is True and body["abstained"] is False


# --------------------------------------------------------------------------- #
# Causation, kept apart from association
# --------------------------------------------------------------------------- #
def test_causal_prints_the_intervention_beside_the_co_occurrence(client: TestClient) -> None:
    body = client.post("/v1/nyx/causal", json={"cause": "rain", "effect": "wet"}).json()
    assert "effect_size" in body and "correlation" in body


def test_an_unidentified_pair_is_not_dressed_up_as_a_cause(client: TestClient) -> None:
    body = client.post("/v1/nyx/causal", json={"cause": "alpha", "effect": "beta"}).json()
    assert body["causal"] is False
    assert body["strategy"] in ("", "none")


def test_a_causal_question_missing_a_side_is_refused(client: TestClient) -> None:
    body = client.post("/v1/nyx/causal", json={"cause": "rain", "effect": ""}).json()
    assert body["causal"] is False and "two things" in body["note"]


# --------------------------------------------------------------------------- #
# Writing code — a gauntlet, and a refusal that names itself
# --------------------------------------------------------------------------- #
def test_author_returns_every_stage_so_the_caller_sees_where_it_stopped(
        client: TestClient) -> None:
    body = client.post("/v1/nyx/author",
                       json={"spec": "a function that returns the nth triangular number",
                             "load": False}).json()
    assert "stopped_at" in body and "refusal" in body


def test_author_refuses_outside_its_families_rather_than_shipping_something_plausible(
        client: TestClient) -> None:
    body = client.post("/v1/nyx/author",
                       json={"spec": "a distributed database", "load": False}).json()
    assert body["ok"] is False and body["refusal"]


# --------------------------------------------------------------------------- #
# The two routes that can touch the world
# --------------------------------------------------------------------------- #
def test_act_says_what_she_would_do_without_doing_it(client: TestClient) -> None:
    """``dry_run`` defaults to true — reaching because you can is the failure this prevents."""
    body = client.post("/v1/nyx/act", json={"request": "list the files in this directory"}).json()
    assert body["ran"] is False
    assert "verdict" in body                    # the consequence gate's decision rides along


def test_act_does_not_reach_when_the_turn_did_not_want_a_tool(client: TestClient) -> None:
    body = client.post("/v1/nyx/act", json={"request": "what is the capital of france"}).json()
    assert body["wanted"] is False and body["ran"] is False


def test_goal_decomposes_and_prices_without_running(client: TestClient) -> None:
    body = client.post("/v1/nyx/goal",
                       json={"request": "fix the code but first run the tests"}).json()
    assert body["run"] is None
    assert body["plan"]["size"] == 2
    assert all(node["rollback"] for node in body["plan"]["nodes"])
    assert all(node["status"] == "pending" for node in body["plan"]["nodes"])


def test_goal_reports_itself_incomplete_rather_than_done(client: TestClient) -> None:
    body = client.post("/v1/nyx/goal", json={
        "request": "write a program that solves the halting problem", "run": True}).json()
    assert body["run"]["complete"] is False
    assert "rather than calling it done" in body["run"]["stopped_because"]


# --------------------------------------------------------------------------- #
# Her own systems, her own constants, her own agenda
# --------------------------------------------------------------------------- #
def test_axiom_checks_the_system_rather_than_asserting_it(client: TestClient) -> None:
    body = client.post("/v1/nyx/axiom",
                       json={"problem": "a transitive relation with a cyclic pair"}).json()
    assert "promoted" in body or "system" in body or "note" in body


def test_evolve_returns_the_step_and_the_ledger_behind_it(client: TestClient) -> None:
    body = client.post("/v1/nyx/evolve", json={"force": True}).json()
    assert "step" in body and "stats" in body
    assert "note" in body["stats"]              # and the honest framing with it


def test_agenda_reports_goals_and_what_she_is_stuck_on(client: TestClient) -> None:
    body = client.get("/v1/nyx/agenda").json()
    assert "goals" in body and "stats" in body
    assert isinstance(body["goals"], list)


def test_stream_digest_compresses_at_every_scale(client: TestClient) -> None:
    body = client.get("/v1/nyx/stream/digest").json()
    assert set(body["digests"]) == {"seconds", "minutes", "hours", "days"}
    assert isinstance(body["briefing"], str)


def test_the_stream_route_is_not_swallowed_by_the_faculty_catch_all(
        client: TestClient) -> None:
    """``/v1/nyx/{faculty}`` is registered last on purpose; a literal path after it is dead."""
    assert "digests" in client.get("/v1/nyx/stream/digest").json()
    assert "note" in client.get("/v1/nyx/stream").json()


# --------------------------------------------------------------------------- #
# Read-only faculty state
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("faculty", ["lingua", "semantics", "icl", "intent", "reason", "hands",
                                     "author", "consequence", "axiom", "omega", "telepathy",
                                     "prover", "vsa", "causal", "weaver", "dialectic",
                                     "stream", "goals"])
def test_every_v02_faculty_reports_its_own_state(client: TestClient, faculty: str) -> None:
    body = client.get(f"/v1/nyx/{faculty}").json()
    assert isinstance(body, dict) and body


def test_every_v02_faculty_states_its_own_limit(client: TestClient) -> None:
    """The honesty rule, checked over the wire: ``stats()`` carries the limit, not just numbers."""
    for faculty in ("semantics", "author", "prover", "causal", "dialectic", "stream", "goals"):
        assert client.get(f"/v1/nyx/{faculty}").json().get("note")


def test_an_unknown_faculty_is_named_rather_than_guessed(client: TestClient) -> None:
    body = client.get("/v1/nyx/telekinesis").json()
    assert body["available"] is False and "telekinesis" in body["reason"]


# --------------------------------------------------------------------------- #
# Meaning as structure — and no mind reading
# --------------------------------------------------------------------------- #
def test_a_frame_goes_straight_in_without_the_tokenizer(client: TestClient) -> None:
    with client.websocket_connect("/v1/nyx/telepathy") as socket:
        socket.send_json({"frame": {"concepts": ["gurutvakarshan", "seb"],
                                    "bindings": {"agent": "gurutvakarshan",
                                                 "patient": "seb"}}})
        got = socket.receive_json()
    assert got.get("concepts") or got.get("bindings") or got.get("note")


def test_she_emits_her_own_state_as_a_frame(client: TestClient) -> None:
    with client.websocket_connect("/v1/nyx/telepathy") as socket:
        socket.send_json({"emit": True})
        got = socket.receive_json()
    assert "frame" in got


def test_a_message_that_is_neither_says_what_to_send(client: TestClient) -> None:
    with client.websocket_connect("/v1/nyx/telepathy") as socket:
        socket.send_json({"nonsense": 1})
        got = socket.receive_json()
    assert "error" in got


# --------------------------------------------------------------------------- #
# Disabled faculties degrade rather than raise
# --------------------------------------------------------------------------- #
@pytest.fixture
def disabled(monkeypatch):
    """A client whose brain was built with faculties off.

    ``NyxaraCore`` reads the cached settings singleton, so setting the env var is not enough —
    the singleton has to be rebuilt before the core is constructed, and restored afterwards.
    """
    from nyxara.kernel.config import reload_settings

    def _build(**flags) -> TestClient:
        for name, value in flags.items():
            monkeypatch.setenv(f"NYXARA_NYX__{name.upper()}", str(value))
        reload_settings()
        return TestClient(create_app())

    yield _build
    monkeypatch.undo()
    reload_settings()


def test_a_disabled_faculty_returns_a_reason_rather_than_an_error(disabled) -> None:
    client = disabled(causal_enabled=False, goals_enabled=False)
    body = client.post("/v1/nyx/causal", json={"cause": "a", "effect": "b"}).json()
    assert body["available"] is False and "CAUSAL_ENABLED" in body["reason"]
    body = client.post("/v1/nyx/goal", json={"request": "do a thing"}).json()
    assert body["available"] is False and "GOALS_ENABLED" in body["reason"]


def test_a_disabled_stream_says_so_over_the_digest_route(disabled) -> None:
    body = disabled(stream_enabled=False).get("/v1/nyx/stream/digest").json()
    assert body["available"] is False and "STREAM_ENABLED" in body["reason"]
