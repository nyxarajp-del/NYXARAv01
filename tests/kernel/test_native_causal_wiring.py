"""Tests for the kernel wiring of native reasoning + dynamic causal learning.

The orchestrator must (a) hand the reasoner her runtime-learned causal model and
knowledge graph; (b) feed every turn's stimulus TOPICS into causal discovery (runtime
learning about the world, beyond training data); (c) refresh causal links incrementally
every turn instead of only every N-th; and (d) count native answers as HER OWN mind in
the handoff meter.
"""

from __future__ import annotations
from pathlib import Path


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
    """Her own chain of thought reaches the report — with the law store cleared first.

    This is not tidying: a discovered-law store changes whether `_native_answer` produces a trace
    at all, and the dependency is measured. In one process::

        fresh core                          -> 6 trace steps
        after one core.idle_maintenance()   -> 0
        after deleting the two law files    -> 5

    `idle_maintenance` runs law discovery and writes `data/law_discovery.json` and
    `data/law_notebook.jsonl`; every core built afterwards loads them and then *defers* on this
    question, so `last_native_trace` is never assigned and the report says she did no native
    reasoning. Whether deferring to a discovered law is the right answer is a separate question —
    answering from a law she derived is arguably better than re-deriving it — but reporting it as
    **no trace** is wrong either way, and that is a finding about `nyxara/mind/nyxara_reasoner.py`
    rather than about this test.

    So the store is cleared to make this test measure the thing in its name. The dependency is
    written down here rather than hidden, because it was invisible for as long as no test ran
    `idle_maintenance` before this one.
    """
    from nyxara.kernel.config import get_settings

    data_dir = Path(get_settings().paths.data_dir)
    for name in ("law_discovery.json", "law_notebook.jsonl"):
        target = data_dir / name
        if target.exists():
            target.unlink()

    nyx = _core()
    nyx.process("what is 12 * 12?")
    rep = nyx.report()
    assert "native_reasoning" in rep
    assert rep["native_reasoning"]["last_trace_steps"] > 0
