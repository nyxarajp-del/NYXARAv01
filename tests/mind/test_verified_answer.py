"""Tests for nyxara.mind.verified_answer — inference-time search + verify (ceiling-break)."""

from __future__ import annotations

from nyxara.mind.verified_answer import (VerifiedAnswer, best_verified_answer,
                                         faculty_oracle)


def test_oracle_answer_wins_outright():
    """A provably-correct oracle answer is returned over any sampled candidate."""
    va = best_verified_answer("2 + 2", lambda: "five", oracle=lambda p: ("4", 1.0))
    assert isinstance(va, VerifiedAnswer)
    assert va.text == "4"
    assert va.method == "oracle"
    assert va.confidence == 1.0


def test_majority_vote_rescues_noisy_model():
    """With no oracle, the most-agreed answer across samples is chosen (self-consistency)."""
    bag = iter(["Paris", "London", "Paris", "Lyon", "Paris"])
    va = best_verified_answer("capital of France?", lambda: next(bag),
                              samples=5, oracle=lambda p: None)
    assert va is not None
    assert va.text == "Paris"
    assert va.method == "self_consistency"
    assert va.confidence == 0.6           # 3 of 5 agreed
    assert va.samples == 5


def test_near_matches_cluster_together():
    """Equivalent answers differing only in punctuation/spacing count as the same vote."""
    bag = iter(["x = 2", "x=2.", "x  =  2", "y = 9"])
    va = best_verified_answer("solve for x", lambda: next(bag),
                              samples=4, oracle=lambda p: None)
    assert va is not None
    assert va.method == "self_consistency"
    assert va.confidence == 0.75          # 3 of 4 are the same answer
    assert va.samples == 4


def test_no_candidates_and_no_oracle_returns_none():
    """When nothing can be produced and no oracle applies, return None so the caller falls back."""
    va = best_verified_answer("x", lambda: "", samples=3, oracle=lambda p: None)
    assert va is None


def test_single_candidate_is_returned_as_single():
    bag = iter(["only one", "", ""])
    va = best_verified_answer("q", lambda: next(bag), samples=3, oracle=lambda p: None)
    assert va is not None
    assert va.text == "only one"
    assert va.method == "single"


def test_faculty_oracle_solves_exact_arithmetic():
    """The default oracle uses NYXARA's exact reasoning faculties on a decidable prompt."""
    hit = faculty_oracle("12 * 12")
    assert hit is not None
    answer, conf = hit
    assert answer.strip() == "144"
    assert conf > 0.0


def test_faculty_oracle_abstains_on_open_prompt():
    """On a non-decidable prompt the oracle returns None (the neural mind then handles it)."""
    assert faculty_oracle("Describe your feelings about autumn.") is None


def test_generator_exceptions_are_tolerated():
    """A sampler that sometimes raises just lowers the effective count; selection still works."""
    seq = iter(["good", "good"])

    def _gen() -> str:
        try:
            return next(seq)
        except StopIteration:
            raise RuntimeError("sampler exhausted")

    va = best_verified_answer("q", _gen, samples=5, oracle=lambda p: None)
    assert va is not None
    assert va.text == "good"
