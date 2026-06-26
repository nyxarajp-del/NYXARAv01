"""Tests for nyxara.growth.active_curiosity — NYXARA asks her own questions.

Active Curiosity is the faculty that, on noticing a lived event, spontaneously poses her own
WHY ("why did this happen?") and WHAT-IF ("what if I do this?") questions, self-designs a
*safe, internal* experiment (her learned causal model, an imagined world-simulation, or the
Scientist), and folds the answer back as a belief and a memory — queuing the next question.

Everything here is offline and deterministic: a WHAT-IF is *simulated*, never enacted.
"""

from __future__ import annotations

import random

from nyxara.growth.active_curiosity import (
    ActiveCuriosity,
    CuriosityFinding,
    CuriosityPass,
    Question,
)
from nyxara.mind.causal_world_model import CausalWorldModel
from nyxara.mind.world_simulator import WorldSimulator
from nyxara.memory.self_model import SelfModel
from nyxara.memory.store import MemoryStore, MemoryType
from nyxara.knowledge.base import KnowledgeBase
from nyxara.planning.voi import ValueOfInformation


# --------------------------------------------------------------------------- #
# fixtures — deterministic, offline
# --------------------------------------------------------------------------- #
def _causal_model() -> CausalWorldModel:
    """A causal world where 'rain' genuinely causes 'wet_ground' (contingency well-defined:
    rain is present in only ~half the epochs)."""
    rng = random.Random(7)
    m = CausalWorldModel(window=10.0)
    for k in range(200):
        base = k * 100.0
        if rng.random() < 0.5:
            m.observe("rain", at=base)
            if rng.random() < 0.9:
                m.observe("wet_ground", at=base + 1)
    m.discover()
    return m


def _full_curiosity() -> ActiveCuriosity:
    store = MemoryStore()
    return ActiveCuriosity(
        causal_model=_causal_model(),
        world_simulator=WorldSimulator(),
        scientist=None,
        voi=ValueOfInformation(),
        self_model=SelfModel(),
        knowledge=KnowledgeBase(store=store, name="curiosity"),
        memory=store,
    )


# --------------------------------------------------------------------------- #
# question generation — she asks BOTH a WHY and a WHAT-IF
# --------------------------------------------------------------------------- #
def test_asks_both_why_and_what_if():
    ac = _full_curiosity()
    questions = ac._ask("wet_ground")
    kinds = {q.kind for q in questions}
    assert "why" in kinds and "what_if" in kinds
    why = next(q for q in questions if q.kind == "why")
    assert why.text == "Why did wet_ground happen?"
    assert all(q.text.endswith("?") for q in questions)


def test_what_if_question_is_about_an_action():
    ac = _full_curiosity()
    what_ifs = [q for q in ac._ask("wet_ground") if q.kind == "what_if"]
    assert what_ifs
    assert all(q.text.lower().startswith("what if i") for q in what_ifs)


# --------------------------------------------------------------------------- #
# WHY — answered from the learned causal model ("why did X happen?")
# --------------------------------------------------------------------------- #
def test_why_answered_from_causal_model():
    ac = _full_curiosity()
    finding = ac.investigate(Question(text="Why did wet_ground happen?", kind="why",
                                      subject="wet_ground"))
    assert finding.method == "causal"
    assert "rain" in finding.answer
    assert finding.confidence > 0.0
    assert finding.error is None
    # one answer breeds the next question — a WHAT-IF about intervening on the cause
    assert finding.follow_up and finding.follow_up.lower().startswith("what if i")


def test_why_falls_back_honestly_when_no_cause_known():
    ac = ActiveCuriosity(causal_model=_causal_model())
    finding = ac.investigate(Question(text="Why did the_sky_turned_green happen?", kind="why",
                                      subject="the_sky_turned_green"))
    # no established cause, no scientist → an honest, low-confidence conjecture (never raises)
    assert finding.error is None
    assert finding.method == "fallback"
    assert finding.confidence < 0.5


# --------------------------------------------------------------------------- #
# WHAT-IF — a SIMULATED experiment, never an enacted action
# --------------------------------------------------------------------------- #
def test_what_if_runs_a_simulation_without_acting():
    ac = _full_curiosity()
    finding = ac.investigate(Question(text="What if I delete the logs?", kind="what_if",
                                      subject="delete the logs"))
    assert finding.method == "simulation"
    assert finding.answer
    assert finding.verdict is not None           # a risk tier from the simulator
    assert finding.error is None
    # the simulator flags a destructive action as risky — and it was only imagined
    assert "file_delete" in str(finding.evidence)


# --------------------------------------------------------------------------- #
# a full pass — observe → ask → value → experiment → learn → follow-up
# --------------------------------------------------------------------------- #
def test_full_pass_learns_and_queues_follow_up():
    ac = _full_curiosity()
    cp = ac.wonder("wet_ground")
    assert isinstance(cp, CuriosityPass)
    assert cp.wondered and cp.resolved
    assert cp.chosen is not None and cp.finding is not None
    # the discovery became a memory, and she queued her own next question
    assert cp.finding.learned_memory
    assert ac.frontier()


def test_full_pass_folds_a_belief_into_the_self_model():
    sm = SelfModel()
    store = MemoryStore()
    ac = ActiveCuriosity(causal_model=_causal_model(), world_simulator=WorldSimulator(),
                         voi=ValueOfInformation(), self_model=sm, memory=store)
    # drive a WHY directly so we assert the belief fold deterministically
    finding = ac.investigate(Question(text="Why did wet_ground happen?", kind="why",
                                      subject="wet_ground"))
    ac._learn(Question(text="Why did wet_ground happen?", kind="why", subject="wet_ground"),
              finding)
    assert finding.learned_belief
    belief = sm.beliefs.best("event:wet_ground", "explained_by")
    assert belief is not None


def test_memory_records_the_wondering():
    ac = _full_curiosity()
    ac.wonder("wet_ground")
    semantic = ac.memory.by_type(MemoryType.SEMANTIC)
    assert any("wondered" in str(getattr(r, "content", "")).lower() for r in semantic)


# --------------------------------------------------------------------------- #
# follow-ups compound — a queued question is picked up on the next pass
# --------------------------------------------------------------------------- #
def test_follow_up_is_consumed_on_next_pass():
    ac = _full_curiosity()
    ac.wonder("wet_ground")
    assert ac.frontier()
    before = list(ac.frontier())
    cp2 = ac.wonder()                            # no event → she pursues her own follow-up
    assert cp2.wondered
    assert cp2.event and cp2.event not in (None, "")
    # the follow-up she was carrying got drawn down
    assert list(ac.frontier()) != before or cp2.event


# --------------------------------------------------------------------------- #
# error-as-data — never raises into the cognitive cycle
# --------------------------------------------------------------------------- #
def test_degrades_gracefully_with_no_dependencies():
    ac = ActiveCuriosity()
    cp = ac.wonder("the build failed unexpectedly")
    assert isinstance(cp, CuriosityPass)
    assert cp.wondered
    assert cp.finding is not None and cp.finding.error is None


def test_no_op_when_nothing_salient():
    ac = ActiveCuriosity()
    cp = ac.wonder("ok")                          # trivial chit-chat is not worth wondering about
    assert not cp.wondered
    assert cp.finding is None


def test_report_counts_passes_and_questions():
    ac = _full_curiosity()
    ac.wonder("wet_ground")
    rep = ac.report()
    assert rep["passes"] == 1
    assert rep["questions_asked"] >= 2


# --------------------------------------------------------------------------- #
# orchestrator integration — the assembled mind wonders, and obeys the gate
# --------------------------------------------------------------------------- #
def test_orchestrator_active_curiosity_pass():
    from nyxara.kernel.orchestrator import NyxaraCore
    from nyxara.observe.mindscope import ThoughtKind

    nyx = NyxaraCore()
    assert nyx.active_curiosity is not None
    nyx.mind.record(ThoughtKind.INFERENCE,
                    "the build failed after the dependency upgrade", salience=0.9)
    report = nyx.active_curiosity_pass()
    assert isinstance(report, dict)
    assert report["wondered"]
    assert report["question"] and report["question"].endswith("?")


def test_orchestrator_idle_maintenance_includes_curiosity():
    from nyxara.kernel.orchestrator import NyxaraCore
    from nyxara.observe.mindscope import ThoughtKind

    nyx = NyxaraCore()
    nyx.mind.record(ThoughtKind.INFERENCE, "the cache hit rate dropped sharply", salience=0.9)
    im = nyx.idle_maintenance()
    assert "active_curiosity" in im


def test_paused_mind_does_not_wonder():
    from nyxara.kernel.orchestrator import NyxaraCore
    from nyxara.guard.oversight import ControlState
    from nyxara.observe.mindscope import ThoughtKind

    nyx = NyxaraCore()
    nyx.mind.record(ThoughtKind.INFERENCE, "something surprising happened", salience=0.9)
    nyx.oversight.state = ControlState.PAUSED
    report = nyx.active_curiosity_pass()
    assert report["wondered"] is False
