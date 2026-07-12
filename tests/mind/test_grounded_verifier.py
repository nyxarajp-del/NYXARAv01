"""Tests for nyxara.mind.grounded_verifier — select search by *truth*, not fluency.

The verifier must (a) on a decidable prompt, score a correct answer far above a fluent-but-wrong
one (the ceiling-break), recognising a correct answer even when wrapped in prose; (b) on a
non-decidable prompt, behave *identically* to the intrinsic answer_quality verifier (no
regression on open-ended turns); and (c) never raise on any input. These run fully offline: the
exact faculty oracle decides arithmetic without any model, network, or weights.
"""

from __future__ import annotations

from nyxara.mind.grounded_verifier import (answer_carries_truth, ground_truth,
                                           grounded_verifier)
from nyxara.mind.router import answer_quality


def test_decidable_correct_beats_fluent_wrong():
    v = grounded_verifier()
    # the faculty oracle decides "2 + 2"; the terse-correct answer must win over the fluent-wrong.
    s_right = v("2 + 2", "4")
    s_wrong = v("2 + 2", "The answer is quite clearly five, on careful reflection.")
    assert s_right > s_wrong
    assert s_right >= 0.9        # a match is certain
    assert s_wrong <= 0.15       # a contradiction is capped far below any match


def test_correct_answer_recognised_inside_prose():
    v = grounded_verifier()
    assert v("2 + 2", "After adding them up I get 4 in total.") >= 0.9
    # a different number in prose is still wrong
    assert v("2 + 2", "After adding them up I get 7 in total.") <= 0.15


def test_open_prompt_matches_intrinsic_exactly():
    # No oracle can decide an open-ended prompt → grounded score == intrinsic score, bit for bit.
    v = grounded_verifier()
    p = "Describe the mood of a rainy afternoon."
    a = "A hush settles in; the rain softens the light and slows the whole street down."
    assert abs(v(p, a) - answer_quality(p, a)) < 1e-9


def test_empty_answer_scores_zero():
    v = grounded_verifier()
    assert v("2 + 2", "") == 0.0
    assert v("anything at all", "   ") == 0.0


def test_never_raises_on_garbage():
    v = grounded_verifier()
    for p, a in [("", ""), (None, None), ("x" * 5, "y" * 5), ("1/0 =", "nan"),
                 ("2+2", 4), (object(), object())]:
        _ = v(p, a)  # type: ignore[arg-type]  # must not raise


def test_ground_truth_and_carries_truth_helpers():
    # arithmetic is decidable offline; an unanswerable prose prompt is not.
    assert ground_truth("2 + 2") is not None
    assert ground_truth("What does love feel like?") is None
    assert answer_carries_truth("4", "the total is 4")
    assert not answer_carries_truth("4", "the total is 5")
    assert not answer_carries_truth("", "anything")
