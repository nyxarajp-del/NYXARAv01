"""Tests for nyxara.mind.recursive_improver — the offline RSI loop.

These focus on the no-LLM (rule-based) path: it must *genuinely* improve a
candidate — repair degenerate text, calibrate confidence, and derive a real
rationale — deterministically and without fabricating content.
"""

from __future__ import annotations

from nyxara.mind.recursive_improver import (
    RecursiveImprover,
    _clean_text,
    _derive_rationale,
)


# --------------------------------------------------------------------------- #
# A minimal Candidate-like stand-in (the improver is duck-typed)
# --------------------------------------------------------------------------- #
class _Cand:
    def __init__(self, text="", confidence=0.7, rationale="", kind="respond"):
        self.text = text
        self.confidence = confidence
        self.rationale = rationale
        self.kind = kind
        self.risk = "low"
        self.reversible = True


def _improver():
    return RecursiveImprover(llm=None, n_iterations=5)


# --------------------------------------------------------------------------- #
# _clean_text — repairs degeneration, never fabricates
# --------------------------------------------------------------------------- #
def test_clean_text_destutters_repeated_words():
    assert _clean_text("The the the capital is Paris.") == "The capital is Paris."


def test_clean_text_collapses_whitespace():
    assert _clean_text("Paris   is\tthe  capital.") == "Paris is the capital."


def test_clean_text_drops_duplicated_adjacent_sentences():
    assert _clean_text("Paris is the capital. Paris is the capital.") == "Paris is the capital."


def test_clean_text_is_idempotent_on_clean_input():
    clean = "The capital of France is Paris."
    assert _clean_text(clean) == clean
    assert _clean_text(_clean_text(clean)) == clean


def test_clean_text_does_not_invent_content():
    # cleaning only ever removes redundancy; output words are a subset of input words
    src = "Yes yes the answer is forty two two."
    out = _clean_text(src)
    assert set(out.lower().split()) <= set(src.lower().split())


# --------------------------------------------------------------------------- #
# _derive_rationale — real, content-aware, not canned filler
# --------------------------------------------------------------------------- #
def test_derive_rationale_is_content_aware():
    rat = _derive_rationale("The capital of France is Paris.", iteration=2)
    assert "paris" in rat.lower() or "france" in rat.lower() or "capital" in rat.lower()
    assert "iter 2" in rat
    assert len(rat.split()) >= 4                       # never the old sparse stub


def test_derive_rationale_reports_word_count():
    rat = _derive_rationale("one two three four five", iteration=1)
    assert "5-word" in rat


# --------------------------------------------------------------------------- #
# improve() — the rule-based loop genuinely improves and never regresses
# --------------------------------------------------------------------------- #
def test_improve_repairs_degenerate_text_offline():
    c = _Cand(text="The the capital is is Paris.", confidence=0.7, rationale="x")
    out = _improver().improve("What is the capital of France?", c)
    assert "the the" not in out.text.lower()
    assert "is is" not in out.text.lower()


def test_improve_clamps_overconfidence():
    c = _Cand(text="Paris.", confidence=0.99, rationale="direct")
    out = _improver().improve("Capital of France?", c)
    assert out.confidence <= 0.90


def test_improve_floors_underconfidence_for_real_response():
    c = _Cand(text="A clear, substantive answer to the question.", confidence=0.1,
              rationale="reasoned")
    out = _improver().improve("question?", c)
    assert out.confidence >= 0.35


def test_improve_enriches_sparse_rationale():
    c = _Cand(text="The capital of France is Paris.", confidence=0.7, rationale="")
    out = _improver().improve("Capital of France?", c)
    assert len(out.rationale.split()) >= 4


def test_improve_never_mutates_immutable_fields():
    c = _Cand(text="x x x", confidence=0.99, rationale="")
    out = _improver().improve("q", c)
    assert out.kind == "respond"
    assert out.risk == "low"
    assert out.reversible is True


def test_improve_is_deterministic_offline():
    a = _improver().improve("q", _Cand(text="The the answer.", confidence=0.97)).text
    b = _improver().improve("q", _Cand(text="The the answer.", confidence=0.97)).text
    assert a == b


def test_improve_never_lowers_score_below_initial():
    from nyxara.mind.recursive_improver import _score_candidate

    c = _Cand(text="The the the capital is is Paris.", confidence=0.99, rationale="")
    initial = _score_candidate(c)
    out = _improver().improve("Capital of France?", c)
    assert _score_candidate(out) >= initial
