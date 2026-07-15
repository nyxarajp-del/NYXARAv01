"""Tests for the kernel wiring of native reasoning + dynamic causal learning.

The orchestrator must (a) hand the reasoner her runtime-learned causal model and
knowledge graph; (b) feed every turn's stimulus TOPICS into causal discovery (runtime
learning about the world, beyond training data); (c) refresh causal links incrementally
every turn instead of only every N-th; and (d) count native answers as HER OWN mind in
the handoff meter.
"""

from __future__ import annotations

from nyxara.kernel.orchestrator import Candidate, NyxaraCore


def _core(**kw):
    return NyxaraCore(**kw)


def test_reasoner_receives_the_causal_model_and_knowledge_graph():
    nyx = _core()
    assert nyx.causal_world_model is not None
    assert getattr(nyx.reasoner, "causal_model", None) is nyx.causal_world_model
    assert getattr(nyx.reasoner, "knowledge_graph", None) is nyx.knowledge_graph


def test_turns_feed_causal_events_including_stimulus_topics():
    nyx = _core()
    nyx.process("the weather system is fascinating today")
    stats = nyx.causal_world_model.stats()
    assert stats["events"] > 0
    labels = nyx.causal_world_model.labels()
    assert any(lab.startswith("topic:") for lab in labels)


def test_stimulus_topics_can_be_disabled():
    from nyxara.kernel.config import get_settings
    nyx = _core()
    before = get_settings().causal.observe_stimulus_topics
    try:
        get_settings().causal.observe_stimulus_topics = False
        assert nyx._stimulus_topics("the weather system is fascinating") == []
    finally:
        get_settings().causal.observe_stimulus_topics = before


def test_incremental_discovery_runs_every_turn_not_only_on_the_cadence():
    nyx = _core()
    cwm = nyx.causal_world_model
    calls = []
    original = cwm.update_links_for
    cwm.update_links_for = lambda labels: calls.append(list(labels)) or original(labels)
    nyx.process("hello")
    # the per-turn incremental refresh ran, with this turn's labels (act/outcome/topics)
    assert len(calls) == 1
    assert any(lab.startswith("act:") for lab in calls[0])
    assert any(lab.startswith("outcome:") for lab in calls[0])


def test_incremental_discovery_can_be_disabled():
    from nyxara.kernel.config import get_settings
    nyx = _core()
    cwm = nyx.causal_world_model
    calls = []
    original = cwm.update_links_for
    cwm.update_links_for = lambda labels: calls.append(list(labels)) or original(labels)
    before = get_settings().causal.incremental_discovery
    try:
        get_settings().causal.incremental_discovery = False
        nyx.process("hello")
        assert calls == []
    finally:
        get_settings().causal.incremental_discovery = before


def test_native_answers_count_as_her_own_mind_in_the_handoff_meter():
    nyx = _core()
    cand = Candidate(text="ok", kind="respond",
                     rationale="system_1 via her own native reasoning (causal; 4 steps)")
    assert nyx._classify_answer_source(cand) == "native"
    nyx._tally_handoff(cand)
    rep = nyx._handoff_report()
    assert rep["counts"].get("native") == 1
    assert rep["own_turns"] == rep["turns"]  # native is NOT a teacher handoff


def test_report_surfaces_native_reasoning():
    nyx = _core()
    nyx.process("what is 12 * 12?")
    rep = nyx.report()
    assert "native_reasoning" in rep
    assert rep["native_reasoning"]["last_trace_steps"] > 0
