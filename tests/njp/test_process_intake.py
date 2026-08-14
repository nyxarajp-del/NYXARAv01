"""The intake that was structurally pinned at zero, and the laws it now builds.

The measurement that motivated this: over a real 20-turn session in the Master's own Hinglish,
``world.events``, ``world.observations`` and ``world.causal_links`` were all exactly 0, and
``grounding.facts`` was 3. Not because the world model was weak — because nothing reached it.
Every test here is a sentence that used to extract nothing.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.grounding import Grounder
from nyxara.njp.world import WorldView


@pytest.mark.parametrize("sentence,expected", [
    ("pani ubalne se bhaap banti hai", ("causes", "bhaap")),
    ("aag se garmi milti hai", ("causes", "garmi")),
    ("garmi se pani ubalta hai", ("causes", "pani")),
    ("aag garmi deti hai", ("produces", "garmi")),
    ("paudhe ko pani chahiye", ("requires", "pani")),
    ("bina pani paudha marta hai", ("requires", "pani")),
    ("plants need water", ("requires", "water")),
    ("engine produces heat", ("produces", "heat")),
])
def test_process_statements_extract(sentence, expected):
    triples = Grounder().ground(sentence).triples
    assert triples, f"{sentence!r} extracted nothing"
    predicate, obj = expected
    assert (triples[0].predicate, triples[0].object) == (predicate, obj)


@pytest.mark.parametrize("sentence,subject", [
    ("pani 100 degree par ubalta hai", "pani"),
    ("water boils at 100 degrees", "water"),
])
def test_a_threshold_keeps_its_number(sentence, subject):
    """"water boils" is true, useless, and unusable by any simulation. The value is the point."""
    triple = Grounder().ground(sentence).triples[0]
    assert (triple.subject, triple.predicate, triple.object) == (subject, "boils", "100")


def test_hinglish_and_english_collapse_onto_one_predicate():
    """Recording "ubalta" and "boils" separately halves the evidence for each."""
    hindi = Grounder().ground("barf pighalti hai").triples[0]
    english = Grounder().ground("the ice melts").triples[0]
    assert hindi.predicate == english.predicate == "melts"


def test_facts_still_win_over_processes():
    """A happening must never shadow a fact — the ordering in the pattern list is load-bearing."""
    grounder = Grounder()
    assert grounder.ground("mera naam NJP hai").triples[0].predicate == "has_name"
    assert grounder.ground("python is a language").triples[0].predicate == "is_a"


def test_a_stated_law_becomes_causal_from_the_first_telling():
    """Testimony about a mechanism is not a sample from it; waiting for three is the wrong rule."""
    world = WorldView()
    link = world.state_law("aag", "garmi")
    assert link.causal and link.stated == 1
    assert link.strength > 0.5
    assert any(l.cause == "aag" for l in world.links(causal_only=True))


def test_a_stated_law_dies_the_moment_it_is_refuted():
    """Admitted earlier than coincidence, and falsified harder. That is the whole trade."""
    world = WorldView()
    world.state_law("aag", "garmi")
    world.refute_law("aag", "garmi")
    link = world.link("aag", "garmi")
    assert not link.causal and link.strength == 0.0
    assert not [l for l in world.links(causal_only=True) if l.cause == "aag"]


def test_repeating_a_law_never_reaches_certainty():
    world = WorldView()
    for _ in range(20):
        world.state_law("aag", "garmi")
    assert world.link("aag", "garmi").strength <= 0.95


def test_a_law_is_not_an_event_on_the_timeline():
    world = WorldView()
    world.state_law("aag", "garmi")
    assert world.stats()["events"] == 0
    assert world.stats()["stated_laws"] == 1


def test_the_previously_zero_counters_move_in_a_real_session():
    """The regression this whole change exists to prevent."""
    brain = NJPBrain()
    for line in ("aag se garmi milti hai", "garmi se pani ubalta hai",
                 "pani 100 degree par ubalta hai", "paudhe ko pani chahiye",
                 "bina pani paudha marta hai", "suraj se roshni aati hai",
                 "roshni se paudhe badhte hai", "dog is a animal", "cat is a animal",
                 "dog needs food", "cat needs food", "barf pighalti hai"):
        brain.think(line)

    stats = brain.stats()
    assert stats["grounding"]["facts"] > 5
    assert stats["world"]["stated_laws"] > 5
    assert stats["world"]["causal_links"] > 0
    assert stats["discover"]["episodes"] > 5
    assert stats["concepts"]["observations"] > 5
    assert stats["beliefs"]["beliefs"] > 5
    assert stats["field"]["cycles"] == 12


def test_world_laws_survive_a_round_trip():
    world = WorldView()
    world.state_law("aag", "garmi")
    world.refute_law("x", "y")
    revived = WorldView()
    revived.load_dict(world.to_dict())
    assert revived.link("aag", "garmi").stated == 1
    assert revived.link("x", "y").refuted == 1
