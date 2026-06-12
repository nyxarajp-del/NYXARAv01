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
    # the predictive/dual-process/meta faculties say — the kernel still disposes.
    nyx = _core()
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
