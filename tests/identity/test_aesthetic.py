"""Tests for nyxara.identity.aesthetic."""

from __future__ import annotations

from nyxara.identity.aesthetic import (
    AestheticDimensions,
    AestheticEvaluation,
    AestheticJudge,
    ArtifactKind,
)


def _judge():
    return AestheticJudge()


_ELEGANT_CODE = '''
def factorial(n):
    """Return n! for a non-negative integer n."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
'''

_UGLY_CODE = '''
def f(n):
    x = 1
    if n > 0:
        if n > 1:
            for i in range(0, n):
                if i > 0:
                    if i > 1:
                        x = x * i
                        x = x * 1
                        x = x * 1
    return x
'''


# -------------------- kind detection -------------------- #
def test_detect_code():
    assert _judge().detect_kind("def foo(): return 1") is ArtifactKind.CODE


def test_detect_structure():
    assert _judge().detect_kind({"a": 1}) is ArtifactKind.STRUCTURE
    assert _judge().detect_kind([1, 2, 3]) is ArtifactKind.STRUCTURE


def test_detect_text():
    assert _judge().detect_kind("A lovely day for a walk.") is ArtifactKind.TEXT


# -------------------- code aesthetics -------------------- #
def test_elegant_beats_ugly_code():
    j = _judge()
    assert j.judge(_ELEGANT_CODE).score > j.judge(_UGLY_CODE).score


def test_ugly_code_flags_nesting():
    crit = _judge().judge(_UGLY_CODE).critique
    assert any("nest" in c for c in crit)


def test_ugly_code_flags_duplication():
    crit = _judge().judge(_UGLY_CODE).critique
    assert any("dup" in c.lower() for c in crit)


def test_code_evaluation_shape():
    ev = _judge().judge(_ELEGANT_CODE)
    assert isinstance(ev, AestheticEvaluation)
    assert ev.kind is ArtifactKind.CODE
    assert 0.0 <= ev.score <= 1.0
    assert isinstance(ev.dimensions, AestheticDimensions)


def test_empty_code():
    ev = _judge().judge("", kind=ArtifactKind.CODE)
    assert "empty" in ev.critique


def test_docstring_suggestion_for_undocumented():
    code = "def g(x):\n    return x + 1\n"
    sugg = _judge().judge(code).suggestions
    assert any("docstring" in s for s in sugg)


# -------------------- text aesthetics -------------------- #
def test_clear_beats_rambling_text():
    j = _judge()
    clear = ("The kernel is sovereign. It decides what runs. Every faculty proposes, "
             "and the kernel disposes.")
    rambling = ("The thing is that basically the system which is a system that does "
                "things does the things that the system does because the system is a "
                "system that does things and things are done by the system.")
    assert j.judge(clear).score > j.judge(rambling).score


def test_rambling_flags_repetition():
    rambling = ("system system system that does things things things that the system "
                "system does things things does system system things things")
    crit = _judge().judge(rambling).critique
    assert any("repetitive" in c for c in crit)


def test_long_sentences_flagged():
    long_text = " ".join(["word"] * 60) + "."
    crit = _judge().judge(long_text, kind=ArtifactKind.TEXT).critique
    assert any("long" in c for c in crit)


def test_empty_text():
    ev = _judge().judge("", kind=ArtifactKind.TEXT)
    assert "empty" in ev.critique


# -------------------- structure aesthetics -------------------- #
def test_flat_beats_nested_structure():
    j = _judge()
    flat = {"a": 1, "b": 2, "c": 3}
    nested = {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}}
    assert j.judge(flat).score > j.judge(nested).score


def test_deep_structure_flagged():
    nested = {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}}
    crit = _judge().judge(nested).critique
    assert any("nest" in c for c in crit)


def test_structure_kind():
    ev = _judge().judge({"x": [1, 2, 3]})
    assert ev.kind is ArtifactKind.STRUCTURE


# -------------------- comparison -------------------- #
def test_compare_picks_more_beautiful():
    j = _judge()
    winner, w_eval, l_eval = j.compare(_ELEGANT_CODE, _UGLY_CODE)
    assert winner is _ELEGANT_CODE
    assert w_eval.score >= l_eval.score


def test_compare_symmetric():
    j = _judge()
    w1, _, _ = j.compare(_ELEGANT_CODE, _UGLY_CODE)
    w2, _, _ = j.compare(_UGLY_CODE, _ELEGANT_CODE)
    assert w1 is _ELEGANT_CODE and w2 is _ELEGANT_CODE


# -------------------- birkhoff measure -------------------- #
def test_birkhoff_measure_higher_for_elegant():
    j = _judge()
    assert j.judge(_ELEGANT_CODE).measure > j.judge(_UGLY_CODE).measure


def test_measure_positive():
    assert _judge().judge(_ELEGANT_CODE).measure > 0


# -------------------- dimensions -------------------- #
def test_dimensions_in_range():
    ev = _judge().judge(_ELEGANT_CODE)
    for v in ev.dimensions.to_dict().values():
        assert 0.0 <= v <= 1.0


def test_elegant_higher_simplicity():
    j = _judge()
    assert (j.judge(_ELEGANT_CODE).dimensions.simplicity
            > j.judge(_UGLY_CODE).dimensions.simplicity)


def test_to_dict():
    d = _judge().judge(_ELEGANT_CODE).to_dict()
    assert "score" in d and "dimensions" in d and "measure" in d


# -------------------- taste -------------------- #
def test_custom_taste_weights():
    from nyxara.identity.aesthetic import _Taste
    plain = AestheticJudge()
    simple_lover = AestheticJudge(taste=_Taste(simplicity=5.0, elegance=0.1, clarity=0.1,
                                               coherence=0.1, parsimony=0.1, harmony=0.1))
    # both still rank elegant over ugly, but scores differ
    assert plain.judge(_ELEGANT_CODE).score != simple_lover.judge(_ELEGANT_CODE).score
