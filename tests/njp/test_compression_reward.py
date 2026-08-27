"""Phase 3 ④ — compression progress as the curiosity reward, and what would falsify it.

The plan's own rule for this phase: *"Each mechanism ships with the measurement that would falsify
it: ... a curiosity reward that never reorders the questions ... If the number does not move, the
mechanism comes out."*

So the load-bearing tests here are the ones that could fail:

* :func:`test_reward_reorders_the_queue` — the falsifier. One organ yielding and the other stalled
  must put a different question at the top than the reverse. If it does not, this is a constant
  added to every structural gap and it changes nothing.
* :func:`test_level_alone_buys_nothing` — arriving already compressed is not learning.
* :func:`test_oscillation_pays_once` — breaking and re-forming the same structure must not pay
  twice, or the reward is a reason to churn.
* :func:`test_reward_never_outranks_stakes` — a tidy gap in a fast-moving region must not
  outrank a prediction that keeps failing.
"""

from __future__ import annotations

import pytest

from nyxara.njp.curiosity import Curiosity, Gap, Question, _REWARD_CAP
from nyxara.njp.progress import CompressionProgress, Source, _PER_PASS_CAP


class _Organ:
    """A compressing organ whose ratio the test drives directly."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def compression(self) -> float:
        return self.value


class _Brain:
    """Just the two attributes :data:`nyxara.njp.progress.SOURCES` reads."""

    def __init__(self, genesis: float = 1.0, discoverer: float = 1.0) -> None:
        self.genesis = _Organ(genesis)
        self.discoverer = _Organ(discoverer)


def _curiosity(brain: _Brain) -> Curiosity:
    # `voi=None` so the fallback pricing is used: it is arithmetic on the question's own fields,
    # which makes the base value predictable and the reward's contribution readable.
    curiosity = Curiosity(brain, voi=None)
    curiosity.progress = CompressionProgress(brain)
    return curiosity


def _ask(curiosity: Curiosity, *questions: Question) -> None:
    for question in questions:
        curiosity._raise(question)


def _genesis_gap() -> Question:
    return Question(text="what is fire's is a?", gap=Gap.MISSING_RELATION,
                    subject="fire", predicate="is_a", uncertainty=0.7, stakes=0.3, cost=0.2)


def _discoverer_gap() -> Question:
    return Question(text="what usually follows rain?", gap=Gap.THIN_COVERAGE,
                    subject="rain", predicate="consequence", uncertainty=0.7, stakes=0.3,
                    cost=0.2)


# --------------------------------------------------------------------------- #
# The falsifier
# --------------------------------------------------------------------------- #
def test_reward_reorders_the_queue():
    """Which organ is still yielding must decide which question is on top.

    Two gaps priced identically by everything else — same uncertainty, same stakes, same cost —
    so the *only* thing that can separate them is what closing them would let her compress. If
    this test's two halves agree, the reward is a constant and the mechanism comes out.
    """
    # Concepts moving, abstractions stalled.
    brain = _Brain(genesis=1.0, discoverer=1.0)
    curiosity = _curiosity(brain)
    curiosity.progress.sample()                     # first sample sets both marks
    brain.genesis.value = 1.5                       # a real new record for concepts only
    _ask(curiosity, _genesis_gap(), _discoverer_gap())
    curiosity.progress.sample()
    curiosity._appraise()
    first = curiosity.top()
    assert first is not None
    assert first.gap == Gap.MISSING_RELATION, "the yielding organ's gap should be on top"

    # Now the reverse, from a fresh queue.
    brain = _Brain(genesis=1.0, discoverer=1.0)
    curiosity = _curiosity(brain)
    curiosity.progress.sample()
    brain.discoverer.value = 1.5                    # abstractions moving, concepts stalled
    _ask(curiosity, _genesis_gap(), _discoverer_gap())
    curiosity.progress.sample()
    curiosity._appraise()
    second = curiosity.top()
    assert second is not None
    assert second.gap == Gap.THIN_COVERAGE, "the other organ's gap should now be on top"

    assert first.gap != second.gap, "progress never reordered the queue — the mechanism is inert"


def test_reordering_survives_the_full_wonder_pass():
    """The same flip, through `wonder()` rather than by calling `_appraise` directly.

    `wonder` samples progress before appraising, and a sample taken after the prices are set
    arrives one pass too late to change them. This is the test that would catch that ordering
    being reversed.
    """
    brain = _Brain(genesis=1.0, discoverer=1.0)
    curiosity = _curiosity(brain)
    _ask(curiosity, _genesis_gap(), _discoverer_gap())
    curiosity.wonder()                              # sets the marks; nothing is yielding yet
    assert all(q.reward == 0.0 for q in curiosity.open_questions())

    brain.discoverer.value = 2.0
    curiosity.wonder()
    top = curiosity.top()
    assert top is not None and top.gap == Gap.THIN_COVERAGE


# --------------------------------------------------------------------------- #
# The three ways of cheating it
# --------------------------------------------------------------------------- #
def test_level_alone_buys_nothing():
    """An organ that arrives compressing beautifully and stands still scores zero."""
    brain = _Brain(genesis=8.0, discoverer=9.0)     # excellent, and going nowhere
    curiosity = _curiosity(brain)
    _ask(curiosity, _genesis_gap(), _discoverer_gap())
    curiosity.wonder()
    curiosity.wonder()
    curiosity.wonder()
    assert curiosity.progress.rate("genesis") == 0.0
    assert curiosity.progress.rate("discoverer") == 0.0
    assert all(q.reward == 0.0 for q in curiosity.open_questions())


def test_oscillation_pays_once():
    """Falling back and climbing to the same place again is not a second thing learned.

    Progress is measured against a high-water mark rather than the previous sample precisely so
    that breaking structure and re-forming it cannot be farmed for reward.
    """
    source = Source(name="genesis")
    source.sample(2.0)                              # sets the mark
    first = source.sample(3.0)                      # a real gain
    assert first > 0.0
    assert source.sample(2.0) == 0.0                # fell back: no reward for falling
    assert source.sample(3.0) == 0.0, "climbing back to a beaten record must pay nothing"
    assert source.records == 1


def test_one_pass_cannot_buy_more_than_one_pass():
    """A single enormous jump must not keep the region looking live for a dozen passes."""
    source = Source(name="discoverer")
    source.sample(1.0)
    source.sample(6.0)                              # the unified corpus's real jump, 1.0 -> 6.0
    assert source.trace == pytest.approx(_PER_PASS_CAP)
    # And it decays away rather than sitting at the cap.
    for _ in range(6):
        source.sample(6.0)
    assert source.trace < _PER_PASS_CAP / 4.0


def test_reward_never_outranks_stakes():
    """A tidy gap in a fast-moving region must not beat a prediction that keeps failing.

    The cap exists for this. A reward that can outrank stakes is not curiosity, it is
    distraction, and being repeatedly wrong is a worse problem than being incompletely organised.
    """
    brain = _Brain(genesis=1.0, discoverer=1.0)
    curiosity = _curiosity(brain)
    curiosity.progress.sample()
    brain.genesis.value = 100.0                     # absurdly fast, far past any real pass
    failure = Question(text="why does my world model keep failing?", gap=Gap.REPEATED_FAILURE,
                       subject="world_model", predicate="failure_cause",
                       uncertainty=1.0, stakes=0.7, cost=0.15)
    _ask(curiosity, _genesis_gap(), failure)
    curiosity.progress.sample()
    curiosity._appraise()
    top = curiosity.top()
    assert top is not None and top.gap == Gap.REPEATED_FAILURE
    assert all(q.reward <= _REWARD_CAP + 1e-9 for q in curiosity.open_questions())


def test_only_gaps_that_feed_a_compressing_organ_are_paid():
    """A known-unknown and an overconfident belief hand nothing to anything that compresses."""
    brain = _Brain(genesis=1.0, discoverer=1.0)
    curiosity = _curiosity(brain)
    curiosity.progress.sample()
    brain.genesis.value = 3.0
    brain.discoverer.value = 3.0
    unknown = Question(text="how do I solve this?", gap=Gap.UNKNOWN, subject="this",
                       uncertainty=0.9, stakes=0.4, cost=0.6)
    overconfident = Question(text="what would settle whether X is true?", gap=Gap.OVERCONFIDENT,
                             subject="X", predicate="evidence", uncertainty=0.8, stakes=0.6,
                             cost=0.15)
    _ask(curiosity, unknown, overconfident, _genesis_gap())
    curiosity.progress.sample()
    curiosity._appraise()
    by_gap = {q.gap: q for q in curiosity.open_questions()}
    assert by_gap[Gap.UNKNOWN].reward == 0.0
    assert by_gap[Gap.OVERCONFIDENT].reward == 0.0
    assert by_gap[Gap.MISSING_RELATION].reward > 0.0


# --------------------------------------------------------------------------- #
# Housekeeping the mechanism depends on
# --------------------------------------------------------------------------- #
def test_reward_is_auditable_and_separable():
    """`value` is base + reward, and both halves must be readable apart."""
    brain = _Brain(genesis=1.0, discoverer=1.0)
    curiosity = _curiosity(brain)
    curiosity.progress.sample()
    brain.genesis.value = 1.4
    question = _genesis_gap()
    _ask(curiosity, question)
    curiosity.progress.sample()
    curiosity._appraise()
    base, _action = curiosity._value_of(question)
    assert question.reward > 0.0
    assert question.value == pytest.approx(base + question.reward)
    assert question.to_dict()["reward"] == pytest.approx(round(question.reward, 5))


def test_high_water_marks_survive_a_restart():
    """Dropping the marks would make her find a region she had learned out interesting again."""
    brain = _Brain(genesis=1.0, discoverer=1.0)
    curiosity = _curiosity(brain)
    curiosity.wonder()
    brain.genesis.value = 4.0
    curiosity.wonder()
    saved = curiosity.to_dict()

    revived = _curiosity(_Brain(genesis=4.0, discoverer=1.0))
    revived.load_dict(saved)
    assert revived.progress.sources["genesis"].best == pytest.approx(4.0)
    # And the region is correctly dull: 4.0 is not a new record.
    revived.wonder()
    assert revived.progress.sources["genesis"].records == 1


def test_a_stalled_tracker_reports_nothing_yielding():
    tracker = CompressionProgress(_Brain(genesis=2.0, discoverer=2.0))
    tracker.sample()
    tracker.sample()
    assert tracker.yielding() == []
    assert tracker.stats()["passes"] == 2


def test_missing_organs_are_absent_rather_than_zero():
    """A brain without the compressing organs must price gaps, not crash."""

    class _Bare:
        genesis = None
        discoverer = None

    curiosity = Curiosity(_Bare(), voi=None)
    _ask(curiosity, _genesis_gap())
    curiosity.wonder()
    question = curiosity.open_questions()[0]
    assert question.reward == 0.0
    assert question.value > 0.0


def test_nan_and_infinity_are_not_records():
    source = Source(name="genesis")
    source.sample(2.0)
    assert source.sample(float("inf")) == 0.0
    assert source.sample(float("nan")) == 0.0
    assert source.best == pytest.approx(2.0)
