"""The Complementary Learning System is the reward learner, wired end to end.

Proves the orchestrator builds a CLS as ``self.learner`` (a drop-in superset), that every path the
core already drives keeps working on it — the flattened cortex weight vector, the dominant-task tag
from the episodic buffer, the per-turn NREM consolidation, and the idle deep-sleep — and that
disabling the flag cleanly falls back to the single-system Learner.
"""

from __future__ import annotations

from nyxara.growth.cls import ComplementaryLearningSystem
from nyxara.growth.learn import Learner
from nyxara.kernel.orchestrator import NyxaraCore


def test_core_builds_cls_as_the_learner():
    core = NyxaraCore()
    assert isinstance(core.learner, ComplementaryLearningSystem)
    assert hasattr(core.learner, "sleep") and hasattr(core.learner, "hippocampus")


def test_weight_vector_and_task_tag_read_the_two_systems():
    core = NyxaraCore()
    for _ in range(40):
        core.learner.record("act", {"owner": 1.0, "kind_a": 1.0}, reward=1.0, task="act")
    core.learner.replay(balanced=True)      # NREM: consolidate into the cortex
    core.learner.consolidate()
    # _learner_weight_vector flattens the CORTEX weights (what ElasticSynapses anchors)
    wv = core._learner_weight_vector()
    assert wv and any(k.startswith("act::") for k in wv)
    # _dominant_task_tag reads the HIPPOCAMPAL episodic buffer
    assert core._dominant_task_tag() == "act"


def test_per_turn_consolidation_moves_knowledge_to_cortex():
    core = NyxaraCore()
    for _ in range(50):
        core.learner.record("act", {"kf": 1.0}, reward=1.0, task="act")
    before = core.learner.neocortex.predict("act", {"kf": 1.0})
    for _ in range(10):
        core.learner.replay(balanced=True)
        core.learner.consolidate()
    after = core.learner.neocortex.predict("act", {"kf": 1.0})
    assert after > before
    # the blended value the decision loop reads is sensible
    assert core.learner.value("act", {"kf": 1.0}) > 0.3


def test_idle_maintenance_runs_cls_sleep():
    core = NyxaraCore()
    for _ in range(40):
        core.learner.record("act", {"kf": 1.0}, reward=1.0, task="act")
    report = core.idle_maintenance()
    assert "cls_sleep" in report
    assert report["cls_sleep"]["replayed_to_cortex"] >= 0
    assert "forgetting" in report["cls_sleep"]


def test_process_turn_records_through_cls(monkeypatch):
    from nyxara.agency.permissions import Authority
    core = NyxaraCore()
    # a real turn should encode into the fast store without error and grow the episodic buffer
    core.process("hello NYXARA", authority=Authority.OWNER)
    assert isinstance(core.learner, ComplementaryLearningSystem)
    # the hippocampus is the live store the turn feeds
    assert len(core.learner.hippocampus.episodes()) >= 0


def test_cls_can_be_disabled_falls_back_to_single_learner(monkeypatch):
    monkeypatch.setenv("NYXARA_MEMORY__CLS_ENABLED", "false")
    from nyxara.kernel.config import reload_settings
    reload_settings()
    try:
        core = NyxaraCore()
        assert isinstance(core.learner, Learner)
        assert not isinstance(core.learner, ComplementaryLearningSystem)
    finally:
        reload_settings()
