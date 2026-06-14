"""Tests for nyxara.kernel.workspace — the Global Workspace."""

from __future__ import annotations

import asyncio

from nyxara.kernel.workspace import (
    Broadcast,
    Content,
    GlobalWorkspace,
    SalienceWeights,
)


def _c(source, **kw):
    return Content(payload=source, source=source, tags=frozenset(kw.pop("tags", set())), **kw)


def test_content_signature_and_refresh():
    c = Content(payload=1, source="s", tags=frozenset({"b", "a"}))
    assert c.signature() == "s:a,b"
    c.activation = 0.2
    c.refresh(0.9)
    assert c.activation == 0.9


def test_salience_weights_score_components():
    w = SalienceWeights()
    c = _c("x", activation=1.0, urgency=1.0, goal_relevance=1.0)
    s_low = w.score(c, goal_bias=0.0)
    s_high = w.score(c, goal_bias=2.0)
    assert s_high > s_low  # top-down bias raises salience


def test_basic_competition_winner():
    ws = GlobalWorkspace(access_threshold=0.5)
    ws.submit(_c("weak", activation=0.3, source_priority=0.1))
    ws.submit(_c("strong", activation=1.0, urgency=1.0, source_priority=0.9))
    out = ws.cycle()
    assert out
    assert out[0].winner.source == "strong"


def test_threshold_starvation():
    ws = GlobalWorkspace(access_threshold=100.0)  # nothing can cross
    ws.submit(_c("x", activation=1.0))
    out = ws.cycle()
    assert out == []
    assert ws.metrics.starved_cycles == 1


def test_empty_cycle_is_starved():
    ws = GlobalWorkspace()
    assert ws.cycle() == []
    assert ws.metrics.starved_cycles == 1


def test_listener_receives_broadcast():
    ws = GlobalWorkspace(access_threshold=0.1)
    seen = []
    ws.add_listener(lambda b: seen.append(b))
    ws.submit(_c("a", activation=1.0))
    ws.cycle()
    assert len(seen) == 1
    assert isinstance(seen[0], Broadcast)


def test_listener_exception_does_not_break_broadcast():
    ws = GlobalWorkspace(access_threshold=0.1)
    ok = []

    def bad(b):
        raise RuntimeError("boom")

    ws.add_listener(bad)
    ws.add_listener(lambda b: ok.append(1))
    ws.submit(_c("a", activation=1.0))
    ws.cycle()
    assert ok == [1]


def test_top_down_goal_bias_changes_winner():
    ws = GlobalWorkspace(access_threshold=0.1)
    ws.submit(_c("threat", activation=0.9, urgency=0.5, tags={"threat"}))
    ws.submit(_c("owner", activation=0.9, tags={"owner"}))
    ws.set_goals({"owner": 5.0})  # strongly prioritise owner
    out = ws.cycle()
    assert out[0].winner.source == "owner"


def test_winner_leaves_competition():
    ws = GlobalWorkspace(access_threshold=0.1)
    ws.submit(_c("a", activation=1.0))
    ws.submit(_c("b", activation=0.2))
    ws.cycle()
    pending_sources = {c.source for c in ws.pending()}
    assert "a" not in pending_sources  # winner consumed


def test_loser_decays_and_is_dropped():
    ws = GlobalWorkspace(access_threshold=100.0, decay=0.5, activation_floor=0.3)
    ws.submit(_c("a", activation=1.0))
    ws.cycle()  # 1.0 -> 0.5
    assert ws.pending()[0].activation == 0.5
    ws.cycle()  # 0.5 -> 0.25 < floor -> dropped
    assert ws.pending() == []
    assert ws.metrics.dropped_decayed == 1


def test_inhibition_of_return_suppresses_repeat():
    ws = GlobalWorkspace(access_threshold=0.1, ior_strength=1.0, ior_window=2)
    ws.submit(_c("a", activation=1.0, tags={"t"}))
    ws.submit(_c("b", activation=0.9, tags={"u"}))
    first = ws.cycle()[0].winner.source
    assert first == "a"
    # re-submit a; it should be suppressed by IOR so b wins
    ws.submit(_c("a", activation=1.0, tags={"t"}))
    ws.submit(_c("b", activation=0.9, tags={"u"}))
    second = ws.cycle()[0].winner.source
    assert second == "b"


def test_habituation_reduces_novelty_over_repeats():
    ws = GlobalWorkspace(access_threshold=0.1, habituation=0.0, ior_strength=0.0)
    c1 = _c("n", activation=0.1, novelty=1.0, tags={"x"})
    ws.submit(c1)
    s_first = ws._salience(c1)
    ws.cycle()  # records signature
    c2 = _c("n", activation=0.1, novelty=1.0, tags={"x"})
    s_repeat = ws._salience(c2)
    assert s_repeat < s_first  # habituated


def test_coalition_beats_lonely_item():
    ws = GlobalWorkspace(access_threshold=1.5, coalition_synergy=0.5)
    ws.submit(_c("a", activation=0.9, tags={"plan"}))
    ws.submit(_c("b", activation=0.9, tags={"plan"}))
    ws.submit(_c("c", activation=1.0, tags={"lonely"}))
    out = ws.cycle()
    assert out
    assert "plan" in out[0].winner.tags
    assert len(out[0].coalition.members) == 2


def test_capacity_allows_multiple_winners():
    ws = GlobalWorkspace(access_threshold=0.1, capacity=2)
    ws.submit(_c("a", activation=1.0, tags={"x"}))
    ws.submit(_c("b", activation=1.0, tags={"y"}))
    out = ws.cycle()
    assert len(out) == 2


def test_submit_same_id_refreshes():
    ws = GlobalWorkspace()
    c = _c("a", activation=0.2)
    ws.submit(c)
    c2 = Content(payload="a", source="a", activation=0.9, item_id=c.item_id)
    ws.submit(c2)
    assert len(ws.pending()) == 1
    assert ws.pending()[0].activation == 0.9


def test_conscious_and_stream():
    ws = GlobalWorkspace(access_threshold=0.1)
    ws.submit(_c("a", activation=1.0))
    ws.cycle()
    ws.submit(_c("b", activation=1.0))
    ws.cycle()
    assert ws.conscious.winner.source == "b"
    assert [s.winner.source for s in ws.stream()] == ["a", "b"]


def test_bus_integration_publishes():
    from nyxara.kernel.bus import EventBus

    async def go():
        bus = EventBus(workers=1)
        received = []
        bus.subscribe("workspace.broadcast", lambda ev: received.append(ev.payload))
        ws = GlobalWorkspace(access_threshold=0.1, bus=bus)
        async with bus:
            ws.submit(_c("a", activation=1.0))
            ws.cycle()
            await bus.drain()
        return received

    received = asyncio.run(go())
    assert len(received) == 1
    assert isinstance(received[0], Broadcast)
    assert received[0].winner.source == "a"


def test_metrics_snapshot_keys():
    ws = GlobalWorkspace(access_threshold=0.1)
    ws.submit(_c("a", activation=1.0))
    ws.cycle()
    m = ws.metrics.snapshot()
    assert m["cycles"] == 1 and m["broadcasts"] == 1 and m["submitted"] == 1
