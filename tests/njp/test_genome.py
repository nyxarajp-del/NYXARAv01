"""The Reasoning Genome: does she notice that she keeps re-deriving the same thing?

What matters here is not that traces are stored — it is that the tallies cannot be won cheaply.
A shape that is frequent and wrong must not be promoted, a shape that is short must not look like
a saving, and a shape learned from her own imagination must never become a primitive she then
applies to real questions.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.core import Derivation
from nyxara.njp.genome import ReasoningGenome, Shape, shape_of

CHAINS = [("aag", "garmi", "pasina"), ("baadal", "baarish", "keechad"),
          ("dhuan", "khansi", "dard"), ("thand", "kapkapi", "bukhar")]


def _chain(hops: int = 2, predicate: str = "causes") -> Derivation:
    support = [(f"s{i}", predicate, f"s{i + 1}") for i in range(hops)]
    return Derivation(answer="x", kind="composed", confidence=0.44, support=support)


# --------------------------------------------------------------------------- #
# The shape is the reusable part
# --------------------------------------------------------------------------- #
def test_the_same_reasoning_through_different_subjects_is_one_shape():
    """Strip the entities and what is left is the form — the thing worth naming once."""
    a = Derivation(kind="composed", support=[("aag", "causes", "garmi"),
                                             ("garmi", "causes", "pasina")])
    b = Derivation(kind="composed", support=[("garmi", "causes", "pasina"),
                                             ("pasina", "causes", "pyaas")])
    assert shape_of(a) == shape_of(b) == ("causes", "causes")


def test_a_one_hop_lookup_is_not_a_shape():
    """A direct answer's "chain" is one edge; counting it would swamp every tally."""
    genome = ReasoningGenome()
    assert genome.record(Derivation(kind="direct", support=[("a", "causes", "b")])) is None
    assert genome.stats()["shapes"] == 0


# --------------------------------------------------------------------------- #
# What "worth compressing" costs
# --------------------------------------------------------------------------- #
def test_a_shape_must_recur_before_it_is_a_pattern():
    genome = ReasoningGenome(min_seen=3)
    for _ in range(2):
        genome.record(_chain())
    assert not genome.candidates()
    genome.record(_chain())
    assert [s.shape for s in genome.candidates()] == [("causes", "causes")]


def test_the_saving_is_arithmetic_rather_than_a_threshold_someone_liked():
    """``n·k − (k + n)``: re-deriving n times at k steps, versus naming once and applying n times."""
    shape = Shape(shape=("causes", "causes"), seen=5, total_depth=10)
    assert shape.depth == pytest.approx(2.0)
    assert shape.saving == pytest.approx(5 * 2 - (2 + 5))

    # Twice at two steps saves nothing, and says so rather than being quietly promoted.
    assert Shape(shape=("causes", "causes"), seen=2, total_depth=4).saving == pytest.approx(0.0)


def test_a_frequent_shape_that_keeps_being_wrong_is_never_promoted():
    """Naming it would only teach her to make the same mistake faster."""
    genome = ReasoningGenome(min_seen=3)
    for _ in range(6):
        trace = genome.record(_chain())
        genome.grade(trace, correct=False)
    assert not genome.candidates(), "a shape with a failing record became a candidate"
    assert [s.shape for s in genome.liabilities()] == [("causes", "causes")]


def test_a_liability_is_reported_rather_than_merely_excluded():
    """That it is *frequent* is why it matters, not a reason to stay quiet about it."""
    genome = ReasoningGenome(min_seen=3)
    for _ in range(5):
        genome.grade(genome.record(_chain()), correct=False)
    assert genome.stats()["liabilities"] == 1
    assert genome.stats()["worst"]


def test_grading_is_recorded_once_and_only_once():
    genome = ReasoningGenome()
    trace = genome.record(_chain())
    genome.grade(trace, correct=True)
    genome.grade(trace, correct=False)
    shape = genome.shapes[("causes", "causes")]
    assert (shape.graded, shape.correct) == (1, 1)


def test_an_ungraded_shape_is_not_treated_as_a_failure():
    """None and 0.0 are different things, and averaging them lies."""
    genome = ReasoningGenome(min_seen=2)
    for _ in range(3):
        genome.record(_chain())
    assert genome.shapes[("causes", "causes")].success is None
    assert genome.candidates(), "an ungraded shape was penalised as though it had failed"


# --------------------------------------------------------------------------- #
# Simulation, one rung further up
# --------------------------------------------------------------------------- #
def test_a_shape_learned_only_from_simulation_is_never_promoted():
    """A reasoning form learned from her own imagination, then applied to real questions, is the
    same closed loop `truth.py` refuses one level down."""
    genome = ReasoningGenome(min_seen=3)
    for _ in range(6):
        genome.grade(genome.record(_chain(), simulated=True), correct=True)
    assert not genome.candidates()
    # Recorded honestly, though: the uses happened.
    assert genome.shapes[("causes", "causes")].seen == 6
    assert genome.shapes[("causes", "causes")].real == 0


# --------------------------------------------------------------------------- #
# Wired, and durable
# --------------------------------------------------------------------------- #
def test_the_brain_records_the_shape_of_what_it_derives():
    """The material existed on every derived answer and survived exactly one turn."""
    brain = NJPBrain()
    for head, mid, tail in CHAINS:
        brain.think(f"{head} se {mid} hoti hai")
        brain.think(f"{mid} se {tail} hoti hai")
        brain.think(f"why {tail}")
    stats = brain.stats()["genome"]
    assert stats["recorded"] >= len(CHAINS)
    assert stats["shapes"] == 1, "four different chains should share one reasoning form"
    assert stats["candidates"] == 1
    assert stats["total_saving"] > 0


def test_the_genome_survives_a_restart():
    """A record of how she reasons is worthless if it starts empty every morning."""
    brain = NJPBrain()
    for head, mid, tail in CHAINS:
        brain.think(f"{head} se {mid} hoti hai")
        brain.think(f"{mid} se {tail} hoti hai")
        brain.think(f"why {tail}")
    before = brain.stats()["genome"]

    revived = NJPBrain()
    revived.load_dict(brain.to_dict())
    after = revived.stats()["genome"]
    assert (after["shapes"], after["recorded"]) == (before["shapes"], before["recorded"])
    assert after["candidates"] == before["candidates"]
