"""Tests for nyxara.social.common_ground."""

from __future__ import annotations

from nyxara.social.common_ground import (
    CommonGround,
    Entity,
    GroundStatus,
    Proposition,
    ReferenceForm,
)


def _cg():
    return CommonGround(["NYXARA", "JP"])


# -------------------- grounding -------------------- #
def test_propose_not_grounded_alone():
    cg = _cg()
    cg.propose("NYXARA", "x is true")
    assert not cg.is_grounded("x is true")


def test_propose_then_acknowledge_grounds():
    cg = _cg()
    cg.propose("NYXARA", "x is true")
    cg.acknowledge("JP", "x is true")
    assert cg.is_grounded("x is true")


def test_known_by_tracks_speakers():
    cg = _cg()
    p = cg.propose("NYXARA", "fact")
    assert "NYXARA" in p.known_by
    cg.acknowledge("JP", "fact")
    assert {"NYXARA", "JP"} <= p.known_by


def test_assert_mutual_grounds_immediately():
    cg = _cg()
    cg.assert_mutual("both saw the alert")
    assert cg.is_grounded("both saw the alert")


def test_case_insensitive_keys():
    cg = _cg()
    cg.propose("NYXARA", "The Sky Is Blue")
    cg.acknowledge("JP", "the sky is blue")
    assert cg.is_grounded("THE SKY IS BLUE")


def test_contest_prevents_grounding():
    cg = _cg()
    cg.propose("NYXARA", "system secure")
    cg.acknowledge("JP", "system secure")
    cg.contest("JP", "system secure")
    assert not cg.is_grounded("system secure")
    assert any(p.content == "system secure" for p in cg.contested())


def test_acknowledge_unknown_creates():
    cg = _cg()
    cg.acknowledge("JP", "new fact")  # JP acks something not yet proposed
    assert "JP" in cg._props["new fact"].known_by


# -------------------- given vs new -------------------- #
def test_is_given_and_new():
    cg = _cg()
    cg.assert_mutual("known")
    assert cg.is_given("known")
    assert cg.is_new("unknown")


def test_should_explain():
    cg = _cg()
    cg.assert_mutual("already known")
    assert cg.should_explain("already known") is False
    assert cg.should_explain("brand new") is True


def test_split_given_new():
    cg = _cg()
    cg.assert_mutual("a")
    split = cg.split_given_new(["a", "b", "c"])
    assert split["given"] == ["a"]
    assert set(split["new"]) == {"b", "c"}


# -------------------- queries -------------------- #
def test_shared():
    cg = _cg()
    cg.assert_mutual("fact1")
    cg.propose("NYXARA", "fact2")  # not grounded
    shared = [p.content for p in cg.shared()]
    assert "fact1" in shared
    assert "fact2" not in shared


def test_known_to():
    cg = _cg()
    cg.propose("NYXARA", "nyxara fact")
    assert any(p.content == "nyxara fact" for p in cg.known_to("NYXARA"))
    assert not any(p.content == "nyxara fact" for p in cg.known_to("JP"))


# -------------------- discourse entities -------------------- #
def test_introduce_entity():
    cg = _cg()
    e = cg.introduce_entity("intruder", by="NYXARA")
    assert isinstance(e, Entity)
    assert cg.is_introduced("intruder")
    assert e.mentions == 1


def test_reference_form_new_is_indefinite():
    cg = _cg()
    assert cg.reference_form("never_seen") is ReferenceForm.INDEFINITE


def test_reference_form_salient_is_pronoun():
    cg = _cg()
    cg.introduce_entity("intruder")
    assert cg.reference_form("intruder") is ReferenceForm.PRONOUN


def test_reference_form_old_is_definite():
    cg = _cg()
    cg.introduce_entity("intruder")
    cg.introduce_entity("server")
    cg.introduce_entity("log")  # push intruder out of the salience window
    assert cg.reference_form("intruder") is ReferenceForm.DEFINITE


def test_mention_entity_increments():
    cg = _cg()
    cg.introduce_entity("x")
    cg.mention_entity("x")
    assert cg._entities["x"].mentions == 2


def test_mention_unknown_returns_none():
    cg = _cg()
    assert cg.mention_entity("ghost") is None


# -------------------- presupposition -------------------- #
def test_presupposition_definite():
    cg = _cg()
    pres = cg.presuppositions("Close the door.")
    assert any("door" in p for p in pres)


def test_presupposition_factive():
    cg = _cg()
    pres = cg.presuppositions("I know that the attack failed.")
    assert any("attack failed" in p for p in pres)


def test_presupposition_again():
    cg = _cg()
    pres = cg.presuppositions("The alarm went off again.")
    assert any("before" in p for p in pres)


def test_presupposition_stopped():
    cg = _cg()
    pres = cg.presuppositions("The service stopped responding.")
    assert any("previously" in p for p in pres)


def test_presupposition_none():
    cg = _cg()
    pres = cg.presuppositions("Hello.")
    assert pres == []


# -------------------- status -------------------- #
def test_status():
    cg = _cg()
    cg.assert_mutual("grounded fact")
    cg.introduce_entity("thing")
    s = cg.status()
    assert "grounded fact" in s["grounded"]
    assert "thing" in s["entities"]


def test_proposition_to_dict():
    p = Proposition("x", known_by={"A"}, status=GroundStatus.PROPOSED)
    d = p.to_dict()
    assert d["content"] == "x" and d["status"] == "proposed"
