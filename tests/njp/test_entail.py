"""What she worked out from FLAN's reasoning, and the honest size of it."""

from __future__ import annotations

import pytest

from nyxara.njp.entail import LABELS, Reasoner, probe, read_pairs
from nyxara.njp.entailschool import examine, split
from nyxara.njp.induce import cover


@pytest.fixture(scope="module")
def pairs():
    return read_pairs()


@pytest.fixture(scope="module")
def taught(pairs):
    learn, _held = split(pairs)
    reasoner = Reasoner()
    reasoner.learn_from(learn)
    return reasoner


# --------------------------------------------------------------------------------------------- #
#  the corpus and what was read out of it
# --------------------------------------------------------------------------------------------- #
def test_the_prompt_is_read_as_well_as_the_answer(pairs):
    """Only the targets would have been a quarter of the data."""
    assert len(pairs) > 5000


def test_every_pair_carries_one_of_the_three_answers(pairs):
    assert {p.label for p in pairs} == set(LABELS)


def test_no_pair_appears_twice(pairs):
    assert len({p.key for p in pairs}) == len(pairs)


def test_the_rationale_is_kept_and_not_learned_from(pairs):
    """Learning from it would be learning the dataset's words, not the regularity underneath."""
    assert any(p.rationale for p in pairs)
    reading = probe(pairs[0].premise, pairs[0].hypothesis)
    assert all(not isinstance(v, str) or v in ("none", "one", "few", "many",
                                               "all", "most", "some")
               for v in reading.values())


# --------------------------------------------------------------------------------------------- #
#  the probes are senses
# --------------------------------------------------------------------------------------------- #
def test_no_probe_names_an_answer():
    reading = probe("A band plays as a crowd gathers.", "A band is playing for a crowd.")
    for name, value in reading.items():
        assert name not in LABELS and value not in LABELS


def test_a_hypothesis_inside_the_premise_overlaps_completely():
    reading = probe("A large group of kids listening to a lady in a blue dress.",
                    "A lady is in a blue dress.")
    assert reading["overlap"] == "all"
    assert reading["adds nothing"] is True


def test_a_hypothesis_that_adds_something_says_so():
    reading = probe("A brown dog is on the ground growling.",
                    "A dog is warning a stranger.")
    assert reading["adds nothing"] is False
    assert reading["added"] != "none"


# --------------------------------------------------------------------------------------------- #
#  what she learned, and what she did not
# --------------------------------------------------------------------------------------------- #
def test_she_finds_the_containment_rule(taught):
    """A hypothesis made only of the premise's own words usually follows from it.

    Asserted on the claim rather than on which probe carries it. Two readings say the same thing —
    ``overlap is all`` and ``adds nothing`` — and the induction takes whichever covers more; on
    7,226 pairs that was the first and on 36,302 it is the second. Pinning the test to one of them
    would have failed on a corpus five times the size for no reason at all.
    """
    yes = [r for r in taught.rules if r.label == "yes"]
    assert yes, "nothing learned about entailment"
    containment = {("overlap", "all"), ("adds nothing", True), ("added", "none")}
    assert any((name, value) in containment for r in yes for name, value in r.terms)


def test_every_rule_carries_the_rate_it_was_kept_at(taught):
    for rule in taught.rules:
        assert 0.0 < rule.purity <= 1.0
        assert rule.purity >= taught.purity


def test_demanding_exactness_leaves_her_with_nothing(pairs):
    """Language is not a decision procedure, and this is the measurement that says so."""
    learn, _held = split(pairs)
    strict = Reasoner(purity=1.0)
    strict.learn_from(learn)
    assert strict.rules == []


def test_the_hard_labels_are_reported_as_near_misses_not_rounded_up(taught):
    """Every candidate for `no` has hundreds of counterexamples. That is a finding, not a rule."""
    misses = {rule.label for rule in taught.near_misses}
    assert "no" in misses
    for rule in taught.near_misses:
        assert rule.counterexamples > 0


def test_she_is_silent_where_nothing_covers_the_pair(taught):
    said, why = taught.answer("Colourless green ideas sleep furiously.",
                              "The committee approved the amendment on Tuesday.")
    assert said == "unknown" and why


def test_the_fallback_is_not_folded_into_the_answer(taught):
    """Guessing the commonest label is what a caller gets, not something she worked out."""
    weird = ("Colourless green ideas sleep furiously.",
             "The committee approved the amendment on Tuesday.")
    assert taught.answer(*weird)[0] == "unknown"
    assert taught.guess(*weird)[0] == taught.commonest


# --------------------------------------------------------------------------------------------- #
#  the numbers
# --------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def marked():
    return examine()


def test_when_she_answers_she_is_well_above_the_base_rate(marked):
    assert marked["taught"].when_answered > marked["base_rate"].accuracy + 0.25


def test_but_she_answers_only_a_small_share_of_them(marked):
    """The honest shape of the result: one real rule, and it covers a sliver."""
    assert marked["taught"].coverage < 0.25
    assert marked["taught"].accuracy < marked["base_rate"].accuracy


def test_the_rules_do_lift_a_majority_guesser(marked):
    assert marked["with_fallback"].accuracy > marked["base_rate"].accuracy


def test_with_learning_off_she_answers_nothing(marked):
    assert marked["no_rules"].rules == 0
    assert marked["no_rules"].coverage == 0.0


# --------------------------------------------------------------------------------------------- #
#  the shared induction
# --------------------------------------------------------------------------------------------- #
def test_the_cover_does_not_stop_at_the_first_seed_it_cannot_explain():
    """Stopping there meant exactly one of three labels ever got a rule."""
    positives = [{"a": 1, "b": i % 2} for i in range(40)] + [{"a": 9, "b": 9}] * 20
    negatives = [{"a": 9, "b": 9}] * 10 + [{"a": 0, "b": 0}] * 10
    rules, misses = cover(positives, negatives, label="X", min_support=5, min_share=0.1)
    assert rules and any(r.terms == (("a", 1),) for r in rules)
    assert misses


def test_purity_below_one_keeps_a_rule_that_is_usually_right():
    positives = [{"a": 1}] * 9
    negatives = [{"a": 1}] * 1 + [{"a": 0}] * 20
    assert cover(positives, negatives, label="X", min_support=4, purity=1.0)[0] == []
    kept, _misses = cover(positives, negatives, label="X", min_support=4, purity=0.85)
    assert kept and kept[0].purity == pytest.approx(0.9)
