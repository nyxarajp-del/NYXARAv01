"""Tests for nyxara.social.persons."""

from __future__ import annotations

import pytest

from nyxara.kernel.errors import AuthorizationError
from nyxara.social.persons import (
    ALLY_CAP,
    CommStyle,
    InteractionKind,
    Person,
    Roster,
    TrustLevel,
)


# -------------------- TrustLevel -------------------- #
def test_trust_from_score():
    assert TrustLevel.from_score(0.0, is_owner=True) is TrustLevel.OWNER
    assert TrustLevel.from_score(0.7) is TrustLevel.ALLY
    assert TrustLevel.from_score(0.4) is TrustLevel.NEUTRAL
    assert TrustLevel.from_score(0.2) is TrustLevel.SUSPECT
    assert TrustLevel.from_score(0.05) is TrustLevel.THREAT


# -------------------- Person -------------------- #
def test_owner_person_trust_locked():
    p = Person("JP", is_owner=True)
    assert p.trust_score == 1.0
    assert p.trust_level is TrustLevel.OWNER


def test_non_owner_default_low_trust():
    p = Person("Bob")
    assert p.trust_score < 0.35
    assert p.trust_level in (TrustLevel.THREAT, TrustLevel.SUSPECT)


def test_is_threat():
    assert Person("Bob").is_threat()
    assert not Person("JP", is_owner=True).is_threat()


def test_note_preference():
    p = Person("Bob")
    p.note_preference("likes:coffee", True)
    assert p.likes("coffee") is True
    assert p.likes("tea") is None


def test_recent_sentiment():
    from nyxara.social.persons import Interaction
    p = Person("Bob")
    p.history.append(Interaction(InteractionKind.MESSAGE, sentiment=0.5))
    p.history.append(Interaction(InteractionKind.MESSAGE, sentiment=0.7))
    assert p.recent_sentiment() == pytest.approx(0.6)


def test_aliases_lowercased():
    p = Person("Robert", aliases={"Bob", "Bobby"})
    assert "bob" in p.aliases
    assert "robert" in p.all_names()


# -------------------- CommStyle -------------------- #
def test_comm_style_running_average():
    cs = CommStyle()
    cs.observe(formality=0.2)
    cs.observe(formality=0.4)
    assert cs.formality == pytest.approx(0.3)
    assert cs.samples == 2


def test_comm_style_language():
    cs = CommStyle()
    cs.observe(language="hinglish")
    assert cs.language == "hinglish"


# -------------------- Roster: owner -------------------- #
def test_owner_created():
    r = Roster(owner_name="Jaypal Khoja", owner_aliases=["JP"])
    assert r.owner.is_owner
    assert r.owner.trust_score == 1.0


def test_owner_alias_resolves():
    r = Roster(owner_aliases=["JP", "boss"])
    assert r.get("JP") is r.owner
    assert r.get("boss") is r.owner


def test_owner_trust_immutable():
    r = Roster()
    r.record("JP", InteractionKind.HOSTILE, sentiment=-1.0)
    assert r.owner.trust_score == 1.0


def test_second_owner_refused():
    r = Roster()
    with pytest.raises(AuthorizationError):
        r.designate_owner("Impostor")


# -------------------- Roster: persons -------------------- #
def test_get_or_create():
    r = Roster()
    p = r.get_or_create("Alice")
    assert p.name == "Alice"
    assert r.get_or_create("Alice") is p  # same instance


def test_get_unknown_returns_none():
    assert Roster().get("nobody") is None


def test_resolve_alias():
    r = Roster()
    r.get_or_create("Robert", aliases=["Bob"])
    assert r.get("Bob").name == "Robert"


def test_roster_len():
    r = Roster()  # owner only
    assert len(r) == 1
    r.get_or_create("X")
    assert len(r) == 2


# -------------------- trust dynamics -------------------- #
def test_good_interactions_raise_trust():
    r = Roster()
    before = r.get_or_create("Ally").trust_score
    for _ in range(10):
        r.record("Ally", InteractionKind.HELP, sentiment=0.8)
    assert r.get("Ally").trust_score > before


def test_trust_capped_below_owner():
    r = Roster()
    for _ in range(50):
        r.record("Ally", InteractionKind.HELP, sentiment=1.0)
    p = r.get("Ally")
    assert p.trust_score <= ALLY_CAP
    assert p.trust_score < r.owner.trust_score


def test_trust_reaches_ally():
    r = Roster()
    for _ in range(30):
        r.record("Ally", InteractionKind.HELP, sentiment=0.9)
    assert r.get("Ally").trust_level is TrustLevel.ALLY


def test_hostile_collapses_trust():
    r = Roster()
    for _ in range(30):
        r.record("Frenemy", InteractionKind.HELP, sentiment=0.9)
    high = r.get("Frenemy").trust_score
    r.record("Frenemy", InteractionKind.HOSTILE, sentiment=-0.9)
    assert r.get("Frenemy").trust_score < high - 0.3


def test_negative_sentiment_lowers_trust():
    r = Roster()
    r.get_or_create("Bob")
    r.record("Bob", InteractionKind.MESSAGE, sentiment=-0.8)
    assert r.get("Bob").trust_score <= 0.15


def test_record_updates_rapport():
    r = Roster()
    r.record("Bob", InteractionKind.MESSAGE, sentiment=0.6)
    assert r.get("Bob").rapport > 0


def test_record_observes_comm_style():
    r = Roster()
    r.record("Bob", InteractionKind.MESSAGE, sentiment=0.5,
             comm={"formality": 0.2, "language": "hinglish"})
    p = r.get("Bob")
    assert p.comm_style.language == "hinglish"
    assert p.comm_style.samples == 1


def test_record_creates_person_if_missing():
    r = Roster()
    r.record("NewPerson", InteractionKind.MESSAGE, sentiment=0.5)
    assert r.get("NewPerson") is not None


def test_interaction_history_recorded():
    r = Roster()
    r.record("Bob", InteractionKind.QUERY, sentiment=0.3, summary="asked about weather")
    p = r.get("Bob")
    assert len(p.history) == 1
    assert p.history[-1].summary == "asked about weather"


# -------------------- assessment -------------------- #
def test_unknown_is_threat():
    assert Roster().assess("stranger") is TrustLevel.THREAT


def test_threats_and_allies_lists():
    r = Roster()
    r.get_or_create("Suspect")  # low trust -> threat list
    for _ in range(30):
        r.record("Friend", InteractionKind.HELP, sentiment=0.9)
    assert any(p.name == "Suspect" for p in r.threats())
    assert any(p.name == "Friend" for p in r.allies())


def test_owner_not_in_threats():
    r = Roster()
    assert r.owner not in r.threats()


def test_status():
    r = Roster()
    s = r.status()
    assert "owner" in s and "threats" in s and "allies" in s
