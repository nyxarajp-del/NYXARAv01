"""Five kinds of memory, earned promotion, and forgetting that is actually real.

One undifferentiated pool is enough to *find* things and not enough to *organise* them: a passing
"hi NYXARA" and the Master's name shared a shelf, competed in the same recall, and decayed at the
same rate.
"""

from __future__ import annotations

import time

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.levels import HierarchicalMemory, Level

_DAY = 86400.0


@pytest.fixture()
def memory() -> HierarchicalMemory:
    return HierarchicalMemory()


# ---- levels ----------------------------------------------------------------- #
def test_a_memory_lands_at_the_level_it_was_filed_at(memory: HierarchicalMemory):
    memory.remember("k", "something", level=Level.PROCEDURAL)
    assert [e.key for e in memory.at(Level.PROCEDURAL)] == ["k"]


def test_recall_is_not_partitioned_by_level(memory: HierarchicalMemory):
    """Levels govern durability, not reachability. A filing cabinet you must already know the
    drawer of is not a memory."""
    memory.remember("deep", "the Master lives in Mumbai", level=Level.SEMANTIC)
    got = memory.recall("where does the Master live")
    assert getattr(got, "hit", None) is not None


def test_working_memory_is_small_on_purpose(memory: HierarchicalMemory):
    """A working memory that does not forget is just a slow episodic one."""
    for i in range(30):
        memory.remember(f"w{i}", f"item number {i}", level=Level.WORKING)
    assert len(memory.at(Level.WORKING)) <= memory.policies[Level.WORKING].capacity


# ---- promotion has to be earned --------------------------------------------- #
def test_a_claim_that_recurs_is_promoted_to_semantic(memory: HierarchicalMemory):
    memory.remember("e1", "the Master lives in Mumbai", level=Level.EPISODIC, source="turn-1")
    memory.remember("e2", "the Master lives in Mumbai", level=Level.EPISODIC, source="turn-9")
    report = memory.consolidate()
    assert report.promoted == 1
    assert [e.key for e in memory.at(Level.SEMANTIC)] == ["e1"]


def test_a_claim_seen_once_is_not_a_fact(memory: HierarchicalMemory):
    """Promoting on a single occurrence would make "semantic" a synonym for "old"."""
    memory.remember("e1", "a thing that happened exactly once", level=Level.EPISODIC)
    assert memory.consolidate().promoted == 0
    assert not memory.at(Level.SEMANTIC)


def test_promotion_keeps_the_episodes_that_evidenced_it(memory: HierarchicalMemory):
    """Deleting the witnesses would erase why she believes it."""
    memory.remember("e1", "the Master lives in Mumbai", level=Level.EPISODIC)
    memory.remember("e2", "the Master lives in Mumbai", level=Level.EPISODIC)
    memory.consolidate()
    promoted = memory.entries["e1"]
    assert promoted.sources == ["e2"]
    assert "e2" in memory.entries


def test_promotion_records_where_it_came_from(memory: HierarchicalMemory):
    memory.remember("e1", "recurring claim here", level=Level.EPISODIC)
    memory.remember("e2", "recurring claim here", level=Level.EPISODIC)
    memory.consolidate()
    assert memory.entries["e1"].promoted_from == Level.EPISODIC


# ---- forgetting is real ------------------------------------------------------ #
def test_an_untouched_memory_decays(memory: HierarchicalMemory):
    memory.remember("old", "never mentioned again", level=Level.EPISODIC)
    assert memory.retention("old") > 0.9
    assert memory.retention("old", now=time.time() + 400 * _DAY) < 0.05


def test_a_decayed_memory_is_actually_dropped(memory: HierarchicalMemory):
    memory.remember("old", "never mentioned again", level=Level.EPISODIC)
    report = memory.consolidate(now=time.time() + 400 * _DAY)
    assert report.forgotten == 1
    assert "old" not in memory.entries


def test_rehearsal_makes_a_memory_durable(memory: HierarchicalMemory):
    """Spaced repetition: the interval a memory survives extends each time it is retrieved."""
    memory.remember("used", "recalled often", level=Level.EPISODIC)
    before = memory.entries["used"].stability
    for _ in range(5):
        memory.touch("used")
    assert memory.entries["used"].stability > before
    assert memory.retention("used", now=time.time() + 5 * _DAY) > 0.5


def test_autobiographical_memory_is_never_forgotten(memory: HierarchicalMemory):
    """The level whose loss would change who she is. Not a memory-management decision."""
    memory.remember("self", "the Master is Jay", level=Level.AUTOBIOGRAPHICAL)
    report = memory.consolidate(now=time.time() + 10_000 * _DAY)
    assert report.forgotten == 0
    assert memory.retention("self", now=time.time() + 10_000 * _DAY) == 1.0
    assert "self" in memory.entries


def test_promotion_runs_before_forgetting(memory: HierarchicalMemory):
    """Forgetting first would discard the episodes whose recurrence is the evidence, losing the
    fact and the reason to believe it in the same pass."""
    memory.remember("e1", "durable recurring claim", level=Level.EPISODIC)
    memory.remember("e2", "durable recurring claim", level=Level.EPISODIC)
    report = memory.consolidate(now=time.time() + 400 * _DAY)
    assert report.promoted == 1
    assert "e1" in memory.entries, "the promoted fact was forgotten in the same pass"


def test_eviction_drops_the_least_retained_not_the_oldest(memory: HierarchicalMemory):
    """An old memory still being used is exactly the one worth keeping."""
    memory.policies[Level.EPISODIC].capacity = 3
    for i in range(3):
        memory.remember(f"e{i}", f"claim {i}", level=Level.EPISODIC)
    for _ in range(6):
        memory.touch("e0")               # the oldest, but heavily rehearsed
    memory.remember("e3", "newest claim", level=Level.EPISODIC)
    assert "e0" in memory.entries, "a well-rehearsed memory was evicted for being old"


# ---- persistence -------------------------------------------------------------- #
def test_levels_survive_a_reload(memory: HierarchicalMemory):
    memory.remember("self", "the Master is Jay", level=Level.AUTOBIOGRAPHICAL)
    fresh = HierarchicalMemory()
    fresh.load_dict(memory.to_dict())
    assert [e.key for e in fresh.at(Level.AUTOBIOGRAPHICAL)] == ["self"]


# ---- wired --------------------------------------------------------------------- #
def test_a_fact_about_the_master_is_filed_autobiographically():
    brain = NJPBrain()
    brain.think("my name is Jay")
    assert [e.key for e in brain.levels.at(Level.AUTOBIOGRAPHICAL)] == ["turn-1"]


def test_the_brain_reports_its_levels():
    brain = NJPBrain()
    assert set(brain.stats()["levels"]["levels"]) == set(Level.ALL)


def test_an_organ_that_is_off_is_absent():
    class _Off:
        levels_enabled = False

    brain = NJPBrain(_Off())
    assert brain.levels is None
    assert "levels" not in brain.stats()


def test_the_brain_still_remembers_without_levels():
    """The fallback has to work — an absent organ must degrade, not delete the capability."""
    class _Off:
        levels_enabled = False

    brain = NJPBrain(_Off())
    brain.think("my name is Jay")
    assert len(brain.memory) > 0
