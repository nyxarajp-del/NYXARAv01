"""What she worked out from FLAN's reasoning, and the honest size of it."""

from __future__ import annotations

import pytest

from nyxara.njp.entail import (
    LABELS, Pair, Reasoner, as_asked, probe, read_pairs, relation_of,
)
from nyxara.njp.entailschool import EXAMINE_PAIRS, examine, shuffled, split
from nyxara.njp.induce import cover


@pytest.fixture(scope="module")
def corpus():
    """Every pair there is. Only the tests that are *about* the corpus read all of it."""
    return read_pairs()


@pytest.fixture(scope="module")
def pairs(corpus):
    """A deterministic slice of the same mixture — what the mechanism tests learn from.

    The broad corpus is 564,166 pairs and one induction over seven tenths of it is ten minutes.
    That is the right cost for :func:`~nyxara.njp.entailschool.run`, which is measuring the
    corpus, and the wrong cost for a test, which is measuring whether a mechanism works. The
    slice comes off :func:`~nyxara.njp.entailschool.shuffled`, so it is the same mixture as the
    whole: what changes is how much is read, not what.
    """
    return shuffled(corpus)[:EXAMINE_PAIRS]


@pytest.fixture(scope="module")
def taught(pairs):
    learn, _held = split(pairs)
    reasoner = Reasoner()
    reasoner.learn_from(learn)
    return reasoner


# --------------------------------------------------------------------------------------------- #
#  the corpus and what was read out of it
# --------------------------------------------------------------------------------------------- #
def test_the_prompt_is_read_as_well_as_the_answer(corpus):
    """Only the targets would have been a quarter of the data."""
    assert len(corpus) > 5000


def test_every_pair_carries_one_of_the_three_answers(corpus):
    # The broad corpus carries a 0.8% tail of rows whose source labelled them "entailment",
    # "contradiction" or "neutral" instead. Named rather than filtered away silently.
    assert set(LABELS) <= {p.label for p in corpus}
    assert sum(1 for p in corpus if p.label in LABELS) > 0.98 * len(corpus)


def test_no_pair_appears_twice(corpus):
    assert len({p.key for p in corpus}) == len(corpus)


def test_the_rationale_is_kept_and_not_learned_from(corpus):
    """Learning from it would be learning the dataset's words, not the regularity underneath."""
    assert any(p.rationale for p in corpus)
    reading = probe(corpus[0].premise, corpus[0].hypothesis)
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


# --------------------------------------------------------------------------------------------- #
#  two vocabularies, three relations
# --------------------------------------------------------------------------------------------- #
def test_the_two_spellings_of_a_relation_fold_onto_one():
    """`entailment` and `yes` are one relation asked in two ways, not two classes."""
    assert relation_of("entailment") == relation_of("yes") == "yes"
    assert relation_of("contradiction") == relation_of("no") == "no"
    assert relation_of("neutral") == relation_of("it is not possible to tell")


def test_folding_is_case_and_space_insensitive():
    assert relation_of("  Entailment ") == "yes"


def test_a_label_it_has_never_seen_is_left_alone():
    """The map is a fold, not a filter. Anything outside it passes through as itself."""
    assert relation_of("maybe") == "maybe"


def test_the_vocabulary_the_question_asked_in_is_given_back():
    """Answering the right relation in the wrong words is still a wrong answer."""
    assert as_asked("yes", like="entailment") == "entailment"
    assert as_asked("yes", like="no") == "yes"
    assert as_asked("it is not possible to tell", like="neutral") == "neutral"


def test_rendering_round_trips_through_the_relation():
    for spelled in ("entailment", "contradiction", "neutral"):
        assert as_asked(relation_of(spelled), like=spelled) == spelled


def test_the_reasoner_learns_relations_not_spellings():
    """A corpus written in both vocabularies must induce over three classes, not six."""
    rows = []
    for i in range(120):
        # The same pair, half of it spelled one way and half the other.
        yes = i % 2 == 0
        spelled = ("entailment" if i % 4 == 0 else "yes") if yes else (
            "contradiction" if i % 4 == 1 else "no")
        rows.append(Pair(premise="a cat sat on a mat and it was warm",
                         hypothesis="a cat sat down" if yes else "no cat sat down",
                         label=spelled))
    reasoner = Reasoner(purity=0.6, min_support=4, min_share=0.05)
    reasoner.learn_from(rows)
    assert {r.label for r in reasoner.rules} <= {"yes", "no", "it is not possible to tell"}
    assert reasoner.commonest in ("yes", "no")
