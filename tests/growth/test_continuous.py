"""Tests for the always-on continuous-learning engine (nyxara.growth.continuous)."""

from __future__ import annotations

from nyxara.growth.continuous import ContinuousLearner
from nyxara.growth.learn import Learner


def _twoskill_learner() -> Learner:
    L = Learner(base_lr=0.2, ewc_lambda=1.0, task_reserve=16, der_alpha=0.1, seed=1)
    for _ in range(60):
        L.record("greet_warmly", {"with_master": 1.0}, reward=1.0, task="greet")
    for _ in range(60):
        L.record("be_terse", {"task_urgent": 1.0}, reward=1.0, task="urgent")
    return L


def test_background_ticks_rehearse_and_consolidate():
    L = _twoskill_learner()
    cl = ContinuousLearner(L, consolidate_every=4, replay_batch=16)
    consolidations = 0
    for _ in range(12):
        rep = cl.tick()
        consolidations += 1 if rep["consolidated"] else 0
    assert cl._ticks == 12
    assert cl._replayed_total > 0                    # learning happened with NO new turns
    assert consolidations >= 2
    assert cl._probes                                # retention probes registered on consolidation


def test_forgetting_drift_triggers_self_defense():
    L = _twoskill_learner()
    cl = ContinuousLearner(L, consolidate_every=4, replay_batch=16, drift_trigger=0.05)
    for _ in range(8):
        cl.tick()                                    # consolidate + register probes
    lam_before = L.model.ewc_lambda
    der_before = L.der_alpha
    # hammer one skill so the other drifts away from its consolidated target
    for _ in range(200):
        L.record("greet_warmly", {"with_master": 1.0}, reward=-1.0, task="greet")
    defended = False
    for _ in range(6):
        defended = defended or cl.tick()["defended"]
    assert defended, "drift did not trigger a self-defense"
    assert L.model.ewc_lambda > lam_before           # EWC protection raised
    assert L.der_alpha >= der_before
    assert cl.stats()["defenses_applied"] >= 1


def test_stats_shape():
    L = _twoskill_learner()
    cl = ContinuousLearner(L, consolidate_every=2)
    for _ in range(4):
        cl.tick()
    s = cl.stats()
    for key in ("ticks", "replayed_total", "consolidations", "probes",
                "last_drift", "defenses_applied"):
        assert key in s


def test_missing_learner_is_safe():
    cl = ContinuousLearner(None)
    rep = cl.tick()                                  # must never raise
    assert rep["replayed"] == 0 and rep["consolidated"] is False


def test_wired_into_core_idle_maintenance():
    """A real core runs continuous learning on the idle clock (hermetic under pytest)."""
    from nyxara.kernel.orchestrator import NyxaraCore
    core = NyxaraCore()
    if core.learner is None:
        return
    # keep the idle pass cheap: null the ~10s dream/consolidation faculties so the test exercises
    # the continuous-learning block, not the whole dream pipeline (covered elsewhere).
    core.dream_session = None
    core.consolidator = None
    core._continuous_consolidate_every = 2
    for _ in range(40):
        core.learner.record("greet_warmly", {"with_master": 1.0}, reward=1.0, task="greet")
        core.learner.record("be_terse", {"task_urgent": 1.0}, reward=1.0, task="urgent")
    replayed_any = consolidated_any = False
    for _ in range(4):
        rep = core.idle_maintenance()
        cont = rep.get("continuous")
        if cont:
            replayed_any = replayed_any or cont.get("replayed", 0) > 0
            consolidated_any = consolidated_any or cont.get("consolidated", False)
    assert replayed_any, "continuous learning did not rehearse in the idle loop"
    assert consolidated_any, "continuous learning did not consolidate in the idle loop"
    stats = core.learning_report().get("continuity", {}).get("continuous")
    assert stats and stats["ticks"] >= 1
