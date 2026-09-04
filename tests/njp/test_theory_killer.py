"""Phase 4 ⑧ — a falsifier that is a test rather than a sentence.

``Belief.falsifier`` has been written on every belief since the ledger existed and
``Belief.falsifiable`` is ``bool(falsifier)``, so it was true of everything: measured over a
thirty-six turn session, **36 beliefs, 36 falsifiable, 0 with hard evidence**. A property true of
every member of a set discriminates nothing.

Worse, the template that filled it in produced strings nothing could act on — *"an observation in
which fire does not causes heat"*, *"an observation in which plant does not water 2"* — and
nothing anywhere ever went and looked for what they named.

The falsifier for this mechanism is :func:`test_the_hunt_can_kill`: if a belief her own record
flatly contradicts still survives, the search is a counter going up.
"""

from __future__ import annotations

import pytest

from nyxara.njp.falsify import Falsifier, Killer, TheoryKiller


class _Ledger:
    """Just the surface the killer touches."""

    def __init__(self) -> None:
        self.beliefs = {}
        self.retracted = []

    def hold(self, claim, confidence=0.6):
        self.beliefs[claim] = type("B", (), {
            "claim": claim, "confidence": confidence, "outcome": None})()
        return self.beliefs[claim]

    def retract(self, claim, why=""):
        self.retracted.append((claim, why))
        belief = self.beliefs.get(claim)
        if belief is not None:
            belief.confidence = 0.0
            belief.outcome = False
        return belief


class _Triple:
    def __init__(self, obj, superseded=False):
        self.object, self.superseded = obj, superseded


class _Grounder:
    def __init__(self, rows=None):
        self.rows = rows or {}

    def _lookup(self, subject, predicate):
        return self.rows.get((subject, predicate), [])


class _Link:
    def __init__(self, cause_total, together):
        self.cause_total, self.together = cause_total, together


class _World:
    def __init__(self, links=None):
        self.links = links or {}

    def link(self, cause, effect):
        return self.links.get((cause, effect), _Link(0, 0))


class _Brain:
    def __init__(self, *, beliefs=None, grounder=None, world=None, universe=None):
        self.beliefs, self.grounder, self.world, self.universe = beliefs, grounder, world, universe


# --------------------------------------------------------------------------- #
# The falsifier for the mechanism
# --------------------------------------------------------------------------- #
def test_the_hunt_can_kill():
    """A causal belief the world record flatly contradicts must not survive the search."""
    ledger = _Ledger()
    ledger.hold("smoke causes rainbows", confidence=0.8)
    world = _World({("smoke", "rainbows"): _Link(cause_total=12, together=0)})
    killer = TheoryKiller(_Brain(beliefs=ledger, world=world))

    verdicts = killer.hunt()
    assert len(verdicts) == 1
    verdict = verdicts[0]
    assert verdict.checked and verdict.found and verdict.retracted
    assert not verdict.survived
    assert "12" in verdict.evidence and "0" in verdict.evidence
    assert ledger.beliefs["smoke causes rainbows"].confidence == 0.0
    assert killer.stats()["kill_rate"] == pytest.approx(1.0)


def test_a_belief_the_record_supports_survives_the_same_search():
    """The control. If everything dies the search is not discriminating, it is a shredder."""
    ledger = _Ledger()
    ledger.hold("fire causes heat", confidence=0.8)
    world = _World({("fire", "heat"): _Link(cause_total=12, together=11)})
    killer = TheoryKiller(_Brain(beliefs=ledger, world=world))

    verdict = killer.hunt()[0]
    assert verdict.checked and not verdict.found and verdict.survived
    assert not verdict.retracted
    assert ledger.retracted == []
    assert killer.stats()["kill_rate"] == pytest.approx(0.0)


def test_survived_and_unsearched_are_never_the_same_number():
    """`checked` is why `survived` means anything — the same discipline as asked vs scored."""
    ledger = _Ledger()
    ledger.hold("smoke causes rainbows", confidence=0.8)
    # A record too thin to say anything: looked at, not answerable.
    world = _World({("smoke", "rainbows"): _Link(cause_total=2, together=0)})
    killer = TheoryKiller(_Brain(beliefs=ledger, world=world))

    verdict = killer.hunt()[0]
    assert verdict.falsifier.checkable, "it names a record; that is not the same as reading one"
    assert not verdict.checked
    assert not verdict.survived, "too few occurrences is not a belief that withstood scrutiny"
    stats = killer.stats()
    assert stats["checkable"] == 1 and stats["checked"] == 0 and stats["survived"] == 0
    assert stats["kill_rate"] is None


# --------------------------------------------------------------------------- #
# The three searches
# --------------------------------------------------------------------------- #
def test_a_functional_predicate_dies_on_a_different_value():
    ledger = _Ledger()
    ledger.hold("ravi works_at acme", confidence=0.8)
    grounder = _Grounder({("ravi", "works_at"): [_Triple("globex")]})
    killer = TheoryKiller(_Brain(beliefs=ledger, grounder=grounder))

    verdict = killer.hunt()[0]
    assert verdict.found and verdict.retracted
    assert "globex" in verdict.evidence


def test_a_superseded_triple_cannot_kill_twice():
    """Already revised away, so it is not fresh evidence against anything."""
    ledger = _Ledger()
    ledger.hold("ravi works_at acme", confidence=0.8)
    grounder = _Grounder({("ravi", "works_at"): [_Triple("globex", superseded=True)]})
    killer = TheoryKiller(_Brain(beliefs=ledger, grounder=grounder))
    assert not killer.hunt()[0].found


def test_two_properties_are_not_a_contradiction():
    """Only functional predicates can be killed this way — a thing may have many properties."""
    ledger = _Ledger()
    ledger.hold("sparrow has_property small", confidence=0.9)
    grounder = _Grounder({("sparrow", "has_property"): [_Triple("brown")]})
    killer = TheoryKiller(_Brain(beliefs=ledger, grounder=grounder))

    verdict = killer.hunt()[0]
    assert verdict.falsifier.kind == Killer.NONE
    assert not verdict.checked and not verdict.found


def test_an_arrow_dies_when_the_reverse_is_established():
    from nyxara.njp.universe import InternalUniverse, Orientation

    universe = InternalUniverse()
    universe.declare("b", "a", sign=1, orientation=Orientation.VERIFIED)
    ledger = _Ledger()
    ledger.hold("a causes b", confidence=0.8)
    killer = TheoryKiller(_Brain(beliefs=ledger, universe=universe))

    verdict = killer.hunt()[0]
    verdict.falsifier = Falsifier(kind=Killer.REVERSED, subject="a", predicate="causes", object="b")
    found, evidence, searched = killer._look(verdict.falsifier)
    assert searched and found and "verified" in evidence


# --------------------------------------------------------------------------- #
# Deriving the killer from the claim
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("claim,kind", [
    ("fire causes heat", Killer.ABSENT_EFFECT),
    ("ravi works_at acme", Killer.DIFFERENT_VALUE),
    ("sparrow is_a bird", Killer.NONE),
    ("birds fly", Killer.NONE),
    ("", Killer.NONE),
])
def test_the_killer_is_read_off_the_claim(claim, kind):
    assert Falsifier.of(claim).kind == kind


def test_the_stated_falsifier_is_well_formed():
    """The template it replaces produced 'an observation in which fire does not causes heat'."""
    assert Falsifier.of("fire causes heat").stated() == \
        "fire occurring repeatedly without heat following"
    assert "does not causes" not in Falsifier.of("fire causes heat").stated()
    assert Falsifier.of("ravi works_at acme").stated() == \
        "an observation giving ravi a different works at than acme"


def test_checkable_is_not_the_same_property_as_falsifiable():
    """`falsifiable` was true of everything; `checkable` is the one that discriminates."""
    stated = Falsifier.of("sparrow is_a bird")
    assert stated.stated()                  # she did say what would end it
    assert not stated.checkable             # and no record could be searched for it


def test_the_field_writes_a_checkable_falsifier():
    """The source of the malformed strings — one template, now one shared object."""
    import os
    os.environ.setdefault("NYXARA_NJP__EVOLVE_ENABLED", "false")
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    brain.think("fire causes heat")
    held = [b for b in brain.beliefs.beliefs.values() if "fire" in b.claim]
    assert held, "the belief should have been recorded"
    assert held[0].falsifier == "fire occurring repeatedly without heat following"


# --------------------------------------------------------------------------- #
# Housekeeping
# --------------------------------------------------------------------------- #
def test_only_beliefs_she_is_committed_to_are_hunted():
    """The strongest, on the adversary's reasoning: a weak claim she already doubts."""
    ledger = _Ledger()
    ledger.hold("weak causes thing", confidence=0.2)
    ledger.hold("strong causes thing", confidence=0.9)
    killer = TheoryKiller(_Brain(beliefs=ledger, world=_World()))
    assert [v.claim for v in killer.hunt()] == ["strong causes thing"]


def test_a_retracted_belief_is_not_hunted_again():
    ledger = _Ledger()
    ledger.hold("smoke causes rainbows", confidence=0.8)
    world = _World({("smoke", "rainbows"): _Link(cause_total=12, together=0)})
    killer = TheoryKiller(_Brain(beliefs=ledger, world=world))
    assert killer.hunt()[0].retracted
    assert killer.hunt() == [], "outcome False, and confidence is now under the threshold"


def test_a_brain_with_no_ledger_hunts_nothing():
    assert TheoryKiller(_Brain()).hunt() == []
    assert TheoryKiller(None).stats()["examined"] == 0


def test_counters_survive_a_round_trip():
    ledger = _Ledger()
    ledger.hold("smoke causes rainbows", confidence=0.8)
    world = _World({("smoke", "rainbows"): _Link(cause_total=12, together=0)})
    killer = TheoryKiller(_Brain(beliefs=ledger, world=world))
    killer.hunt()
    revived = TheoryKiller(_Brain())
    revived.load_dict(killer.to_dict())
    assert revived.found == killer.found and revived.retracted == killer.retracted
