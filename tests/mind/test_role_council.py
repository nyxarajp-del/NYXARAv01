"""Tests for nyxara.mind.role_council — the six-role council, offline reasoning path.

These exercise the deterministic, no-LLM path: with no provider, the council must produce
*real* per-role analysis grounded in the question (not placeholder filler)."""

from __future__ import annotations

from nyxara.mind.role_council import (
    RoleCouncil,
    RoleVote,
    _offline_role_analyses,
    _question_lens,
    _topic_of,
)


# -------------------- analysis primitives -------------------- #
def test_topic_extracts_key_phrase():
    topic = _topic_of("Why does network latency affect user trust?")
    assert "latency" in topic or "trust" in topic


def test_question_lens_detects_type():
    assert _question_lens("Why does X happen?") == "causal"
    assert _question_lens("How do I build this?") == "procedural"
    assert _question_lens("Should I deploy on Friday?") == "evaluative"
    assert _question_lens("Which option is better?") == "comparative"
    assert _question_lens("Tell me a story.") == "open"


def test_offline_analyses_cover_all_six_roles():
    analyses = _offline_role_analyses("Why does network latency affect user trust?")
    names = [a[0] for a in analyses]
    assert names == ["Scientist", "Engineer", "Strategist", "Critic",
                     "Security Officer", "Philosopher"]
    # every analysis references the actual topic — not generic filler
    for _name, text, _conf, _weight in analyses:
        assert "latency" in text.lower() or "trust" in text.lower()


def test_offline_analyses_adapt_to_question_type():
    causal = _offline_role_analyses("Why does X fail?")[0][1]
    evaluative = _offline_role_analyses("Should I do X?")[0][1]
    assert "cause" in causal.lower()
    assert "trade-off" in evaluative.lower()
    assert causal != evaluative


def test_offline_analyses_not_placeholder_filler():
    text = " ".join(a[1] for a in _offline_role_analyses("How do I secure the API?"))
    assert "a thoughtful response" not in text          # the old stub phrasing is gone
    assert "principles is needed" not in text


# -------------------- convene (offline) -------------------- #
def test_convene_offline_returns_structured_candidate():
    cand = RoleCouncil(llm=None).convene("Why does network latency affect user trust?")
    assert cand.kind == "respond"
    assert cand.confidence > 0
    assert "six lenses" in cand.text
    assert "Integrated view:" in cand.text
    # all six role lenses appear in the synthesis
    for role in ("Scientist", "Engineer", "Strategist", "Critic",
                 "Security Officer", "Philosopher"):
        assert role in cand.text


def test_convene_offline_mentions_entities():
    cand = RoleCouncil(llm=None).convene("Should I deploy on Friday?")
    assert "Friday" in cand.text


def test_stub_votes_are_real_votes():
    council = RoleCouncil(llm=None)
    votes = council._stub_votes("What is the fastest sorting algorithm?")
    assert len(votes) == 6
    assert all(isinstance(v, RoleVote) for v in votes)
    assert all(v.weight > 0 for v in votes)
