"""Tests for nyxara.mind.native_reasoner — HER OWN chain of thought (no LLM anywhere).

Every conclusion here is produced by NYXARA's real faculties — the learned causal graph,
the knowledge graph, verifiable computation — with an inspectable step trace, and every
question she cannot verify is an honest abstain (None), never a fabrication.
"""

from __future__ import annotations

import json
import random

from nyxara.mind.causal_world_model import CausalWorldModel
from nyxara.mind.native_reasoner import (
    NativeReasoner,
    NativeReasoningImprover,
    RungBandit,
    classify_intent,
)


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def _chain_model(seed: int = 7, n: int = 300) -> CausalWorldModel:
    """rain → wet_ground → slippery, discovered from lived episodes."""
    rng = random.Random(seed)
    m = CausalWorldModel(window=10.0)
    for k in range(n):
        base = k * 100.0
        if rng.random() < 0.5:
            m.observe("rain", at=base)
            if rng.random() < 0.9:
                m.observe("wet_ground", at=base + 1)
                if rng.random() < 0.85:
                    m.observe("slippery", at=base + 2)
        if rng.random() < 0.3:
            m.observe("birds_singing", at=base + rng.uniform(0, 90))
    m.discover()
    return m


def _confounded_model(seed: int = 2, n: int = 400) -> CausalWorldModel:
    rng = random.Random(seed)
    m = CausalWorldModel(window=10.0)
    for k in range(n):
        base = k * 100.0
        if rng.random() < 0.5:
            m.observe("summer", at=base)
            if rng.random() < 0.85:
                m.observe("ice_cream", at=base + 1)
            if rng.random() < 0.8:
                m.observe("drowning", at=base + 2)
        else:
            if rng.random() < 0.1:
                m.observe("ice_cream", at=base + 1)
            if rng.random() < 0.05:
                m.observe("drowning", at=base + 2)
    m.discover()
    return m


def _graph():
    from nyxara.memory.graph import KnowledgeGraph
    kg = KnowledgeGraph()
    kg.add_triple("NYXARA", "owned_by", "Jaypal")
    kg.add_triple("Jaypal", "created", "NYXARA")
    return kg


def _reasoner(tmp_path, **kw) -> NativeReasoner:
    return NativeReasoner(data_dir=str(tmp_path), **kw)


# --------------------------------------------------------------------------- #
# intent parsing
# --------------------------------------------------------------------------- #
def test_classify_intent_shapes():
    assert classify_intent("why did the ground get wet?") == "causal_why"
    assert classify_intent("does rain cause slippery?") == "causal_pair"
    assert classify_intent("what happens if rain?") == "causal_whatif"
    assert classify_intent("what if rain had not happened?") == "counterfactual"
    assert classify_intent("how can I prevent slippery?") == "intervention"
    assert classify_intent("what is 12 * 12?") == "computational"
    assert classify_intent("who is the owner of NYXARA?") == "factual"
    assert classify_intent("hello, how are you today?") == "chat"


# --------------------------------------------------------------------------- #
# causal answers — why / pair / what-if / counterfactual, from LIVED evidence
# --------------------------------------------------------------------------- #
def test_why_question_answers_from_the_causal_graph(tmp_path):
    nr = _reasoner(tmp_path, causal_model=_chain_model())
    c = nr.reason("why is the ground slippery?")
    assert c is not None and c.source == "causal" and c.verified
    assert "wet_ground" in c.answer or "rain" in c.answer
    assert len(c.steps) >= 2
    assert any(s.kind.value == "reason" for s in c.steps)


def test_causal_pair_yes_and_no(tmp_path):
    nr = _reasoner(tmp_path, causal_model=_chain_model())
    yes = nr.reason("does rain cause slippery?")
    assert yes is not None and yes.answer.startswith("Yes")
    no = nr.reason("does birds_singing cause slippery?")
    assert no is not None and no.answer.startswith("No")


def test_confounded_pair_is_answered_honestly(tmp_path):
    nr = _reasoner(tmp_path, causal_model=_confounded_model())
    c = nr.reason("does ice_cream cause drowning?")
    assert c is not None and c.answer.startswith("No")
    assert "confounded" in c.answer or "common cause" in c.answer


def test_whatif_propagates_do_through_the_graph(tmp_path):
    nr = _reasoner(tmp_path, causal_model=_chain_model())
    c = nr.reason("what happens if rain?")
    assert c is not None and "slippery" in c.answer
    assert any(s.kind.value == "simulate" for s in c.steps)


def test_counterfactual_answers_with_a_structural_contrast(tmp_path):
    nr = _reasoner(tmp_path, causal_model=_chain_model())
    c = nr.reason("what if rain had not happened?")
    assert c is not None and "Counterfactual" in c.answer


def test_unknown_cause_is_an_honest_no_cause_verdict(tmp_path):
    nr = _reasoner(tmp_path, causal_model=_chain_model())
    c = nr.reason("why did birds_singing happen?")
    assert c is not None
    assert "no established cause" in c.answer


# --------------------------------------------------------------------------- #
# intervention planning — the do-operator used prescriptively
# --------------------------------------------------------------------------- #
def test_intervention_names_an_upstream_lever_with_the_path(tmp_path):
    nr = _reasoner(tmp_path, causal_model=_chain_model())
    c = nr.reason("how can I prevent slippery?")
    assert c is not None and c.source == "intervention"
    assert "wet_ground" in c.answer or "rain" in c.answer
    assert "→" in c.answer  # the causal path is shown


def test_intervention_for_unknown_goal_abstains(tmp_path):
    nr = _reasoner(tmp_path, causal_model=_chain_model())
    assert nr.reason("how can I prevent quantum bananas?") is None


# --------------------------------------------------------------------------- #
# multi-hop blackboard — sub-engines COMPOSE, pronouns resolve across hops
# --------------------------------------------------------------------------- #
def test_compound_question_composes_causal_and_intervention(tmp_path):
    nr = _reasoner(tmp_path, causal_model=_chain_model())
    c = nr.reason("why is the ground slippery and how can I prevent it?")
    assert c is not None
    assert "causal" in c.source and "intervention" in c.source
    assert "because" in c.answer and "prevent" in c.answer.lower()
    # the pronoun 'it' was resolved against the first hop's subject
    assert any("pronoun" in s.summary for s in c.steps)


# --------------------------------------------------------------------------- #
# knowledge graph + computation
# --------------------------------------------------------------------------- #
def test_factual_question_answers_from_the_knowledge_graph(tmp_path):
    nr = _reasoner(tmp_path, knowledge_graph=_graph())
    c = nr.reason("who is the owner of NYXARA?")
    assert c is not None and c.source == "graph"
    assert "Jaypal" in c.answer


def test_computational_question_is_exactly_computed(tmp_path):
    nr = _reasoner(tmp_path)
    c = nr.reason("what is 12 * 12?")
    assert c is not None and "144" in c.answer
    assert c.verified


def test_open_chat_is_an_honest_abstain(tmp_path):
    nr = _reasoner(tmp_path, causal_model=_chain_model(),
                   knowledge_graph=_graph())
    assert nr.reason("hello, how are you today?") is None


def test_trace_renders_step_by_step(tmp_path):
    nr = _reasoner(tmp_path, causal_model=_chain_model())
    c = nr.reason("why is the ground slippery?")
    rendered = c.render(with_trace=True)
    assert "my own reasoning" in rendered
    assert "1. [perceive]" in rendered
    assert c.render(with_trace=False) == c.answer


# --------------------------------------------------------------------------- #
# calibration — lived failures teach her to abstain
# --------------------------------------------------------------------------- #
def test_repeated_failures_drive_marginal_confidence_to_abstain(tmp_path):
    nr = _reasoner(tmp_path, causal_model=_chain_model())
    c = nr.reason("what happens if rain?")  # raw confidence 0.75
    assert c is not None
    for _ in range(30):
        nr.reason("what happens if rain?")
        nr.record_outcome(False)        # the answers keep turning out wrong
    assert nr.reason("what happens if rain?") is None  # calibrated → abstain


# --------------------------------------------------------------------------- #
# bandit — she learns WHICH engine pays off, and it persists
# --------------------------------------------------------------------------- #
def test_bandit_demotes_a_losing_engine_and_persists(tmp_path):
    path = str(tmp_path / "rungs.json")
    b = RungBandit(path)
    for _ in range(6):
        b.record("computational", "chain", False)
        b.record("computational", "compute", True)
    order = b.order("computational", ["chain", "compute"])
    assert order[0] == "compute"
    fresh = RungBandit(path)  # a new instance loads the learned state
    assert fresh.order("computational", ["chain", "compute"])[0] == "compute"


# --------------------------------------------------------------------------- #
# compounding knowledge — verified conclusions become graph knowledge
# --------------------------------------------------------------------------- #
def test_verified_causal_answer_promotes_causes_triples(tmp_path):
    kg = _graph()
    nr = _reasoner(tmp_path, causal_model=_chain_model(), knowledge_graph=kg)
    before = len(kg)
    c = nr.reason("why is the ground slippery?")
    assert c is not None and c.verified
    assert len(kg) > before
    from nyxara.memory.graph import GraphQuery
    hits = kg.match(GraphQuery(predicate="causes"))
    assert any(t for t in hits)


# --------------------------------------------------------------------------- #
# verified-answer replay gate — only genuinely verified work may answer repeats
# --------------------------------------------------------------------------- #
def test_teacher_baked_chat_triple_never_replays(tmp_path):
    """A chat utterance that an old epistemic-distill pass baked must not replay forever."""
    from nyxara.memory.graph import KnowledgeGraph
    kg = KnowledgeGraph()
    kg.add_triple("tum kon ho", "has_verified_answer",
                  "Yes — it is valid under every truth assignment.",
                  confidence=0.78, attributes={"verdict": "graded", "source": "teacher"})
    nr = _reasoner(tmp_path, knowledge_graph=kg)
    assert nr.reason("tum kon ho") is None      # chat → honest abstain, never the stale bake


def test_unproven_teacher_triple_is_not_replayable(tmp_path):
    from nyxara.memory.graph import KnowledgeGraph
    kg = KnowledgeGraph()
    kg.add_triple("what is the airspeed of a swallow", "has_verified_answer",
                  "an unverified teacher guess",
                  confidence=0.9, attributes={"verdict": "graded", "source": "teacher"})
    nr = _reasoner(tmp_path, knowledge_graph=kg)
    assert nr._cached_answer("what is the airspeed of a swallow") is None


def test_proven_or_native_triples_still_replay(tmp_path):
    from nyxara.memory.graph import KnowledgeGraph
    kg = KnowledgeGraph()
    kg.add_triple("what is 6 times 7", "has_verified_answer", "42",
                  confidence=1.0, attributes={"verdict": "proven", "source": "teacher"})
    kg.add_triple("what is 12 times 12", "has_verified_answer", "144",
                  confidence=0.9, attributes={"source": "native_reasoner"})
    nr = _reasoner(tmp_path, knowledge_graph=kg)
    assert nr._cached_answer("what is 6 times 7") == ("42", 1.0)
    assert nr._cached_answer("what is 12 times 12") == ("144", 0.9)


# --------------------------------------------------------------------------- #
# self-improvement — measure, propose, PROVE on replay, apply (bounded)
# --------------------------------------------------------------------------- #
def _write_replay(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def test_improver_raises_the_floor_when_marginal_answers_keep_failing(tmp_path):
    replay = str(tmp_path / "replay.jsonl")
    tuning = str(tmp_path / "tuning.json")
    # mostly-wrong answers at confidence 0.5: answering them costs, abstaining wins
    _write_replay(replay, [
        {"intent": "factual", "engine": "graph", "confidence": 0.5,
         "verified": False, "outcome": (i % 4 == 0)} for i in range(30)])
    imp = NativeReasoningImprover(replay_path=replay, tuning_path=tuning)
    report = imp.improve()
    assert report["applied"] is True
    assert report["min_confidence"] > 0.5
    assert report["score_after"] > report["score_before"]
    with open(tuning, encoding="utf-8") as fh:
        assert json.load(fh)["min_confidence"] == report["min_confidence"]


def test_improver_rejects_when_current_parameters_already_replay_best(tmp_path):
    replay = str(tmp_path / "replay.jsonl")
    tuning = str(tmp_path / "tuning.json")
    # confident answers that keep being RIGHT — nothing to fix, no mutation applied
    _write_replay(replay, [
        {"intent": "factual", "engine": "graph", "confidence": 0.9,
         "verified": True, "outcome": True} for _ in range(30)])
    imp = NativeReasoningImprover(replay_path=replay, tuning_path=tuning)
    report = imp.improve()
    assert report["applied"] is False
    import os
    assert not os.path.exists(tuning)


def test_improver_needs_lived_evidence(tmp_path):
    replay = str(tmp_path / "replay.jsonl")
    _write_replay(replay, [
        {"intent": "factual", "engine": "graph", "confidence": 0.5,
         "verified": False, "outcome": False} for _ in range(5)])
    imp = NativeReasoningImprover(replay_path=replay,
                                  tuning_path=str(tmp_path / "t.json"))
    assert imp.improve()["applied"] is False


def test_improver_respects_hard_bounds(tmp_path):
    replay = str(tmp_path / "replay.jsonl")
    tuning = str(tmp_path / "tuning.json")
    # EVERY answer wrong at high confidence → the best floor is the hard ceiling, and
    # the applied value never leaves the safe envelope
    _write_replay(replay, [
        {"intent": "factual", "engine": "graph", "confidence": 0.85,
         "verified": False, "outcome": False} for _ in range(40)])
    imp = NativeReasoningImprover(replay_path=replay, tuning_path=tuning)
    report = imp.improve()
    assert report["applied"] is True
    assert report["min_confidence"] == 0.9  # clamped at the ceiling, never beyond


def test_reasoner_applies_its_own_proven_tuning(tmp_path):
    nr = _reasoner(tmp_path, causal_model=_chain_model())
    # her lived record: what-if answers at ~0.75 confidence keep failing
    for _ in range(30):
        nr.reason("what happens if rain?")
        rec = nr._pending
        assert rec is not None
        nr._pending = dict(rec, outcome=False)
        nr._flush_pending()
    report = nr.self_improve()
    assert report["applied"] is True and report["min_confidence"] > 0.75
    # a fresh reasoner over the same data_dir loads the tuning and abstains now
    fresh = _reasoner(tmp_path, causal_model=_chain_model())
    assert fresh._floor("causal_whatif") > 0.75
    assert fresh.reason("what happens if rain?") is None


# --------------------------------------------------------------------------- #
# honesty invariants
# --------------------------------------------------------------------------- #
def test_no_causal_model_means_causal_questions_abstain(tmp_path):
    nr = _reasoner(tmp_path)
    assert nr.reason("why is the ground slippery?") is None
    assert nr.reason("how can I prevent slippery?") is None


def test_reason_never_raises_on_garbage(tmp_path):
    nr = _reasoner(tmp_path, causal_model=_chain_model())
    for junk in ("", "   ", "why", "does cause?", "what if", None):
        assert nr.reason(junk) is None or nr.reason(junk) is not None  # no exception


def test_disabled_flag_turns_the_whole_engine_off(tmp_path):
    from nyxara.kernel.config import NyxaraSettings, Profile
    s = NyxaraSettings.for_profile(Profile.TEST)
    s.native_reasoning.enabled = False
    nr = NativeReasoner(causal_model=_chain_model(), settings=s,
                        data_dir=str(tmp_path))
    assert nr.reason("why is the ground slippery?") is None
