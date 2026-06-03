"""Tests for nyxara.kernel.stream — default-mode cognition."""

from __future__ import annotations

import asyncio
import random

from nyxara.kernel.stream import (
    ConceptGraph,
    DefaultModeStream,
    OpenProblem,
    Seed,
    SeedPool,
    Thought,
    ThoughtType,
)
from nyxara.kernel.workspace import Content, GlobalWorkspace


def _stream(seed=1, **kw):
    rng = random.Random(seed)
    sp = SeedPool()
    sp.add_text("the owner", category="goal", tags=["owner"])
    sp.add_text("a packet", category="memory", tags=["net"])
    sp.add_text("chess", category="concept", tags=["chess"])
    cg = ConceptGraph()
    cg.associate("chess", "strategy", 2.0)
    cg.associate("strategy", "defense", 1.0)
    return DefaultModeStream(seeds=sp, concepts=cg, rng=rng, **kw)


# -------------------- SeedPool -------------------- #
def test_seedpool_add_and_len():
    sp = SeedPool()
    sp.add_text("a", "concept")
    sp.add_text("b", "memory")
    assert len(sp) == 2
    assert set(sp.categories()) == {"concept", "memory"}


def test_seedpool_weighted_sample_reproducible():
    sp = SeedPool()
    for i in range(20):
        sp.add_text(f"s{i}", weight=1.0)
    a = sp.sample(random.Random(3), k=5)
    b = sp.sample(random.Random(3), k=5)
    assert [s.text for s in a] == [s.text for s in b]
    assert len(a) == 5
    # without replacement: unique
    assert len({s.text for s in a}) == 5


def test_seedpool_sample_empty():
    assert SeedPool().sample(random.Random(), k=3) == []


def test_seedpool_ring_eviction():
    sp = SeedPool(per_category=3)
    for i in range(5):
        sp.add_text(f"s{i}", "concept")
    assert len(sp) == 3  # only last 3 kept


# -------------------- ConceptGraph -------------------- #
def test_concept_graph_associate_symmetric():
    cg = ConceptGraph()
    cg.associate("a", "b", 1.5)
    assert cg.neighbors("a")["b"] == 1.5
    assert cg.neighbors("b")["a"] == 1.5


def test_concept_graph_walk_stays_on_graph():
    cg = ConceptGraph()
    cg.associate("a", "b")
    cg.associate("b", "c")
    path = cg.walk("a", random.Random(0), steps=5)
    assert path[0] == "a"
    for node in path:
        assert node in {"a", "b", "c"}


def test_concept_graph_walk_dead_end():
    cg = ConceptGraph()
    path = cg.walk("lonely", random.Random(0), steps=3)
    assert path == ["lonely"]


# -------------------- OpenProblem incubation -------------------- #
def test_open_problem_eventually_solves():
    p = OpenProblem("hard thing", difficulty=0.3)
    rng = random.Random(5)
    solved = False
    for _ in range(100):
        if p.incubate(rng):
            solved = True
            break
    assert solved
    assert p.progress > 0.6
    assert p.attempts > 0


def test_open_problem_attempts_increment():
    p = OpenProblem("x")
    rng = random.Random(0)
    p.incubate(rng)
    p.incubate(rng)
    assert p.attempts == 2


# -------------------- Thought -> Content -------------------- #
def test_thought_to_content_tags_and_source():
    t = Thought(type=ThoughtType.WANDER, text="hi", tags=frozenset({"x"}))
    c = t.to_content()
    assert isinstance(c, Content)
    assert c.source == "stream"
    assert "wander" in c.tags and "spontaneous" in c.tags and "x" in c.tags


def test_insight_content_is_loud():
    t = Thought(type=ThoughtType.INSIGHT, text="aha")
    c = t.to_content()
    assert c.activation == 1.0
    assert c.source_priority >= 0.8


# -------------------- stream dynamics -------------------- #
def test_idle_produces_thoughts():
    s = _stream(base_rate=2.0)
    total = sum(len(s.tick(engagement=0.0)) for _ in range(10))
    assert total > 0


def test_engaged_is_quiet():
    s = _stream(base_rate=2.0)
    total = sum(len(s.tick(engagement=0.95)) for _ in range(10))
    assert total == 0


def test_thought_count_scales_with_idleness():
    s = _stream(base_rate=4.0)
    # average more thoughts at low engagement than high (but below busy threshold)
    rng_state = []
    low = sum(len(s.tick(engagement=0.1)) for _ in range(30))
    s2 = _stream(base_rate=4.0)
    high = sum(len(s2.tick(engagement=0.6)) for _ in range(30))
    assert low > high


def test_busy_resets_boredom_clock():
    s = _stream()
    s.tick(engagement=0.95, now=1000.0)
    assert s._idle_seconds(1000.0) == 0.0


def test_temperature_rises_with_idle():
    s = _stream()
    cool = s._temperature(0.0)
    warm = s._temperature(120.0)
    assert warm > cool
    assert 0.0 <= cool <= 1.0 and 0.0 <= warm <= 1.0


def test_stream_feeds_workspace():
    ws = GlobalWorkspace(access_threshold=0.1)
    s = _stream(base_rate=3.0, workspace=ws)
    produced = []
    for _ in range(5):
        produced += s.tick(engagement=0.0)
    assert len(ws.pending()) > 0 or ws.metrics.submitted == 0
    # at least as many submitted as produced
    assert ws.metrics.submitted == len(produced)


def test_incubation_yields_insight_through_stream():
    s = _stream(incubation_chance=1.0)
    s.add_problem("defend the network", difficulty=0.2, tags=["defense"])
    found = False
    for _ in range(80):
        for t in s.tick(engagement=0.1):
            if t.type is ThoughtType.INSIGHT:
                found = True
        if found:
            break
    assert found
    assert s.metrics.insights >= 1


def test_add_problem_registers_and_seeds():
    s = _stream()
    before = len(s.seeds)
    p = s.add_problem("solve X", difficulty=0.5, tags=["x"])
    assert p.problem_id in s.problems
    assert len(s.seeds) == before + 1


def test_inner_monologue_format():
    s = _stream(base_rate=3.0)
    for _ in range(5):
        s.tick(engagement=0.0)
    mono = s.inner_monologue(3)
    assert all(line.startswith("[") for line in mono)


def test_metrics_snapshot():
    s = _stream(base_rate=2.0)
    for _ in range(5):
        s.tick(engagement=0.0)
    snap = s.metrics.snapshot()
    assert snap["ticks"] == 5
    assert "by_type" in snap


def test_async_run_max_ticks():
    s = _stream(base_rate=2.0)

    async def go():
        await s.run(lambda: 0.0, interval=0.0, max_ticks=5)

    asyncio.run(go())
    assert s.metrics.ticks == 5
