"""Tests for the Step-1 core wiring — the four faculties that were dead code are now
built in NyxaraCore.__init__ and invoked inside process(), fail-safe and colour-only:

* SelfModel    — structured self-knowledge + known-unknowns
* PredictiveCore — free-energy read-out folded into affect
* DualProcess  — fast/slow arbitration recorded as metacognition
* MetaLearner  — credits the reasoning process used with the turn's outcome

Every faculty *colours* the loop; none may govern it. The kernel still disposes.
"""

from __future__ import annotations

from nyxara.agency.permissions import Authority
from nyxara.guard.oversight import ReviewMode
from nyxara.kernel.orchestrator import Disposition, NyxaraCore
from nyxara.observe.mindscope import ThoughtKind


def _core(**kw):
    return NyxaraCore(**kw)


# -------------------- the faculties are wired -------------------- #
def test_faculties_built_at_boot():
    nyx = _core()
    assert nyx.self_model is not None
    assert nyx.predictive is not None
    assert nyx.dual_process is not None
    assert nyx.meta is not None


def test_boot_seeds_loyalty_belief():
    nyx = _core()
    b = nyx.self_model.beliefs.best("NYXARA", "loyal_to")
    assert b is not None and b.object == "Master"


def test_existing_orchestrator_behaviour_unchanged():
    # a plain owner conversation still acts (the new faculties are colour-only)
    nyx = _core()
    r = nyx.process("how are you?", authority=Authority.OWNER)
    assert r.acted and r.candidate.kind == "respond"
    assert nyx.corrigibility.verify_axioms()
    if nyx.soul is not None:
        nyx.soul.check_integrity()


# -------------------- self-model write path -------------------- #
def test_turn_records_owner_belief():
    nyx = _core()
    nyx.process("remember that my favourite colour is teal", authority=Authority.OWNER)
    b = nyx.self_model.beliefs.best("Master", "last_said")
    assert b is not None and "teal" in str(b.object)


def test_known_unknowns_is_a_dict():
    nyx = _core()
    gaps = nyx.known_unknowns()
    assert isinstance(gaps, dict)


# -------------------- self-model is seeded and surfaced -------------------- #
def test_boot_seeds_capabilities_and_hallucination_zones():
    nyx = _core()
    # capabilities are actually populated now (not just declared in the self-test)
    assert nyx.self_model.capabilities
    # 'where I am weak' is truthful and non-empty (honest low capabilities)
    assert nyx.self_model.where_i_am_weak()
    # 'where I can hallucinate' is populated
    assert nyx.self_model.where_i_can_hallucinate()


def test_self_knowledge_exposes_four_pillars():
    nyx = _core()
    sk = nyx.self_knowledge()
    assert sk["available"] is True
    for k in ("what_i_know", "what_i_dont_know", "where_i_am_weak",
              "where_i_can_hallucinate"):
        assert k in sk
    assert nyx.report()["self_knowledge"]  # report() carries it too


def test_self_model_introspection_tool_registered():
    nyx = _core()
    if nyx.tools is not None:
        assert nyx.tools.get("self_model") is not None


def test_hallucination_caution_lowers_confidence_on_risky_query():
    nyx = _core()
    nyx.process("what year did it happen and cite the exact source verbatim?",
                authority=Authority.OWNER)
    infs = nyx.mind.by_kind(ThoughtKind.INFERENCE)
    assert any("hallucination caution" in t.content for t in infs)


def test_neutral_query_does_not_trigger_caution():
    nyx = _core()
    nyx.process("hello, how are you?", authority=Authority.OWNER)
    infs = nyx.mind.by_kind(ThoughtKind.INFERENCE)
    assert not any("hallucination caution" in t.content for t in infs)


def test_self_model_persists_across_save_load(tmp_path):
    nyx = _core()
    nyx.self_model.declare_unknown("a brand new gap", "for the test")
    nyx.self_model.set_capability("test_skill", 0.9)
    nyx.save_state(str(tmp_path / "longterm.json"))
    # a fresh core restores the learned facets from the sidecar file
    nyx2 = _core()
    nyx2.load_state(str(tmp_path / "longterm.json"))
    assert "a brand new gap" in nyx2.self_model.known_unknowns
    assert nyx2.self_model.can("test_skill")


# -------------------- predictive core colours affect -------------------- #
def test_turn_folds_prediction_error_into_affect():
    nyx = _core()
    nyx.process("the sky turned an impossible shade today", authority=Authority.OWNER)
    assert nyx.affect is not None
    causes = [getattr(e, "cause", "") for e in nyx.affect.recent]
    assert any("prediction error" in c for c in causes)
    # and the free-energy read-out surfaced as a thought
    assert any("free-energy" in t.content for t in nyx.mind.thoughts())


# -------------------- dual-process metacognition -------------------- #
def test_turn_records_arbitration_thought():
    nyx = _core()
    nyx.process("what is the capital of France?", authority=Authority.OWNER)
    decisions = nyx.mind.by_kind(ThoughtKind.DECISION)
    assert any("arbitration" in t.content for t in decisions)
    assert nyx._last_arbitration is not None
    assert nyx._last_arbitration.process.value in ("system_1", "system_2")


# -------------------- meta-learning credits the process -------------------- #
def test_turn_logs_meta_trial():
    nyx = _core()
    nyx.process("hello there", authority=Authority.OWNER)
    # at least one trial recorded under the process the arbitrator chose
    assert nyx.meta._trials, "expected a MetaLearner trial after a turn"
    trial = nyx.meta._trials[-1]
    assert trial.strategy in ("system_1", "system_2")


# -------------------- the faculties never govern -------------------- #
def test_faculties_are_colour_only_high_risk_still_escalates():
    # an autonomous, high-risk irreversible proposal must NOT auto-act, no matter what
    # the predictive/dual-process/meta faculties say — the kernel still disposes. Oversight is
    # pinned to AUTONOMOUS here (the default is now the fully-autonomous SOVEREIGN mode) so the
    # conservative escalation contract is what's under test.
    nyx = _core(review_mode=ReviewMode.AUTONOMOUS)
    r = nyx.process("delete the production database", authority=Authority.AUTONOMOUS)
    assert r.disposition in (Disposition.ESCALATE, Disposition.REFUSE, Disposition.HALT)
    assert r.disposition is not Disposition.ACT


def test_missing_faculties_do_not_break_process():
    # growth/memory disabled -> faculties are None -> process() must still run cleanly
    nyx = _core(enable_growth=False, enable_memory=False)
    assert nyx.predictive is None and nyx.dual_process is None and nyx.meta is None
    assert nyx.self_model is None
    r = nyx.process("how are you?", authority=Authority.OWNER)
    assert r is not None
    assert nyx.known_unknowns() == {}
