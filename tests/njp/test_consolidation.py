"""Growth without consolidation is accumulation, and both halves of it were unreachable.

Two separate findings, measured over 1,200 corpus pairs.

**The fabric never compressed.** ``consolidate`` fires on the count ceilings — 250,000 cells or
4,000,000 synapses — and the run ended at 136,085 synapses, 3.4% of that. Working as configured,
and configured against the wrong bound: the count ceilings are a *memory* limit, while what makes
a fabric unusable first is *time*. Cost per expansion grows roughly linearly with the wiring
(measured: 2.4 ms at 4.5k synapses, 12.4 at 45k, 25.3 at 128k, 40.0 at 213k), so the declared
capacity is reached long after the turn has stopped being affordable.

**Nothing reached semantic memory.** Promotion is gated on the same claim recurring, and claim
identity was a bag of the memory's words — so two sentences asserting one relation were two
different claims, and recurrence was undetectable outside verbatim repetition. 386 of 391
memories sat in ``episodic`` forever. The fix is not a lower bar: it is a claim identity that can
see a repeat, plus counting *use* as the evidence it actually is.
"""

from __future__ import annotations

import random

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.fabric import Fabric
from nyxara.njp.levels import HierarchicalMemory, Level, LevelPolicy


# --------------------------------------------------------------------------- #
# The fabric: compressing on cost, not only on count
# --------------------------------------------------------------------------- #
def _grow(fabric: Fabric, turns: int, *, width: int = 24, span: int = 4000) -> None:
    rng = random.Random(0)
    for _ in range(turns):
        fabric.stimulate([rng.randrange(0, span) for _ in range(width)])
        fabric.expand(outcome=1.0, result=fabric.settle())


def test_a_fabric_that_has_grown_expensive_compresses():
    """The trigger the count ceilings could not reach at any realistic size.

    Grown for real rather than with injected costs, because the claim being made is that a fabric
    left to grow trips this on its own. ``soft_latency_ms`` is 1.0 — the lowest the constructor
    accepts, since a sub-millisecond bound on a whole expansion is not a bound on anything — and
    the measured cost passes it somewhere past 4k synapses.
    """
    fabric = Fabric(seed=7, soft_latency_ms=1.0, latency_floor_synapses=512)
    _grow(fabric, 220)
    assert fabric.n_synapses > 512, "the test needs a fabric past the floor"
    assert fabric.consolidated_on_latency > 0
    assert fabric.stats()["consolidations"] > 0


def test_a_small_fabric_never_compresses_however_slow_the_machine():
    """Below the floor the cost is the interpreter, and dropping synapses would only shrink her."""
    fabric = Fabric(seed=7, soft_latency_ms=1.0, latency_floor_synapses=10_000_000)
    _grow(fabric, 220)
    assert fabric.consolidated_on_latency == 0


def test_one_slow_turn_does_not_trigger_compression():
    """A window is required, so a garbage collection cannot cost her structure."""
    fabric = Fabric(seed=7, soft_latency_ms=1.0, latency_floor_synapses=256)
    _grow(fabric, 60)                       # past the size floor, so only the window is in play
    assert fabric.n_synapses >= 256
    fabric._expand_ms = [10_000.0]          # one catastrophically slow turn
    assert not fabric._too_slow(), "one sample is not a measurement"
    fabric._expand_ms = [10_000.0] * 8
    assert fabric._too_slow(), "a sustained window is"


def test_the_bound_is_not_settable_below_a_millisecond():
    """A sub-millisecond bound on a whole expansion would compress on measurement noise."""
    assert Fabric(seed=7, soft_latency_ms=0.0001).soft_latency_ms >= 1.0


def test_the_default_fabric_reports_its_cost_and_its_bound():
    """"She is getting slower" has to be a number, not something the Master notices."""
    fabric = Fabric(seed=7)
    _grow(fabric, 20)
    stats = fabric.stats()
    assert stats["mean_expand_ms"] is not None
    assert stats["soft_latency_ms"] > 0
    assert stats["consolidations"] == 0, "a cheap fabric has no reason to compress"


def test_compression_clears_the_window_it_acted_on():
    """Otherwise the pre-compression costs keep it true and she shrinks herself away."""
    fabric = Fabric(seed=7, soft_latency_ms=1.0, latency_floor_synapses=512)
    fabric._expand_ms = [10_000.0] * 32
    fabric.consolidate()
    assert fabric._expand_ms == []
    assert not fabric._too_slow(), "a cleared window cannot re-trigger on costs already acted on"


# --------------------------------------------------------------------------- #
# Levels: a claim identity that can see a repeat
# --------------------------------------------------------------------------- #
def test_two_wordings_of_one_relation_are_one_claim():
    """The whole reason recurrence was undetectable."""
    memory = HierarchicalMemory()
    memory.remember("t1", "noted: deep learning part of machine learning",
                    claim="deep learning|part_of|machine learning")
    memory.remember("t2", "ML contains deep learning, as it happens",
                    claim="deep learning|part_of|machine learning")
    report = memory.consolidate()
    assert report.promoted >= 1
    assert len(memory.levels[Level.SEMANTIC]) >= 1


def test_surface_text_remains_the_fallback_when_no_claim_is_given():
    """Text nobody parsed still gets the bag-of-words identity it always had."""
    memory = HierarchicalMemory()
    memory.remember("t1", "the sky is blue today")
    memory.remember("t2", "today the sky is blue")
    assert memory.consolidate().promoted >= 1


def test_one_telling_is_still_an_episode():
    """The bar was not lowered. A thing said once is a thing said once."""
    memory = HierarchicalMemory()
    memory.remember("t1", "noted: x causes y", claim="x|causes|y")
    memory.consolidate()
    assert not memory.levels[Level.SEMANTIC]


# --------------------------------------------------------------------------- #
# Levels: use is evidence too
# --------------------------------------------------------------------------- #
def test_a_memory_that_is_used_is_promoted():
    """Retrieval shows a memory earned its keep — stronger than the world saying it twice."""
    memory = HierarchicalMemory()
    memory.remember("t1", "noted: x causes y", claim="x|causes|y")
    memory.touch("t1")
    memory.touch("t1")
    report = memory.consolidate()
    assert report.promoted == 1
    assert memory.promoted_by_use == 1


def test_a_memory_used_once_is_not_promoted():
    memory = HierarchicalMemory()
    memory.remember("t1", "noted: x causes y", claim="x|causes|y")
    memory.touch("t1")
    memory.consolidate()
    assert not memory.levels[Level.SEMANTIC]


def test_use_promotion_is_off_where_the_policy_says_so():
    memory = HierarchicalMemory(policies={
        Level.WORKING: LevelPolicy(capacity=7, promotes_to=Level.EPISODIC, promote_after=1),
        Level.EPISODIC: LevelPolicy(capacity=100, promotes_to=Level.SEMANTIC,
                                    promote_after=2, promote_on_uses=0),
        Level.SEMANTIC: LevelPolicy(capacity=100),
        Level.PROCEDURAL: LevelPolicy(capacity=100),
        Level.AUTOBIOGRAPHICAL: LevelPolicy(capacity=100, protected=True),
    })
    memory.remember("t1", "noted: x causes y", claim="x|causes|y")
    for _ in range(5):
        memory.touch("t1")
    memory.consolidate()
    assert not memory.levels[Level.SEMANTIC]


def test_the_two_routes_are_counted_apart():
    """A high use count with low recurrence says her intake never repeats itself."""
    memory = HierarchicalMemory()
    memory.remember("a1", "noted: a causes b", claim="a|causes|b")
    memory.remember("a2", "b comes of a", claim="a|causes|b")
    memory.remember("c1", "noted: c causes d", claim="c|causes|d")
    memory.touch("c1")
    memory.touch("c1")
    memory.consolidate()
    assert memory.promoted_by_use == 1
    assert memory.stats()["promoted_by_recurrence"] == 1


# --------------------------------------------------------------------------- #
# The join: retrieval has to reach consolidation
# --------------------------------------------------------------------------- #
def test_touching_a_claim_reaches_every_memory_asserting_it():
    memory = HierarchicalMemory()
    memory.remember("t1", "noted: x causes y", claim="x|causes|y")
    memory.remember("t2", "y follows x", claim="x|causes|y")
    assert memory.touch_claim("x|causes|y") == 2


def test_touching_a_claim_reaches_a_turn_that_asserted_several():
    """A turn stating three relations is filed under all of them joined."""
    memory = HierarchicalMemory()
    memory.remember("t1", "noted: several things",
                    claim="a|is_a|b ; c|is_a|d")
    assert memory.touch_claim("c|is_a|d") == 1


def test_an_unknown_claim_touches_nothing():
    memory = HierarchicalMemory()
    memory.remember("t1", "noted: x causes y", claim="x|causes|y")
    assert memory.touch_claim("p|causes|q") == 0


def test_retrieval_in_a_whole_turn_rehearses_the_memory_it_used():
    """The arrow that was missing: a fact used a hundred times looked unread to consolidation."""
    brain = NJPBrain()
    brain.think("Deep learning is a subset of machine learning that uses neural networks.")
    before = sum(e.rehearsals for e in brain.levels.entries.values())
    answer = brain.think("What is deep learning?")
    assert answer.recalled, "this turn must go through retrieval for the test to mean anything"
    after = sum(e.rehearsals for e in brain.levels.entries.values())
    assert after > before


def test_a_turn_records_its_relations_as_the_claim():
    brain = NJPBrain()
    brain.think("Deep learning is a subset of machine learning.")
    claims = [e.claim for e in brain.levels.entries.values() if e.claim]
    assert any("part_of" in c for c in claims), claims
