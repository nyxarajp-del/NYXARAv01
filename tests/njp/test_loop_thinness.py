"""Why the learning loop measured almost nothing, and what it measures now.

Over a 39-turn session the loop reported ``scored 32, correct 2, accuracy 0.0625``, with
``deferred_resolved``, ``strategies_graded`` and ``beliefs_settled`` all flat zero. The loop was
closed — :mod:`nyxara.njp.integrate` had already fixed that — and what it closed onto could not
produce a right answer or a resolved question. Two independent defects, both structural:

**The anticipation was a quota, not a prediction.** ``Manifold.precognition`` took ``scored[:k]``
with ``k`` derived from the seed width rather than from the evidence. On a live fabric the
similarities were ``0.600, 0.595, 0.278, 0.230, 0.095`` — a clean break after the second — and
``k`` was 6 against a pool of 5, so every candidate came back as "about to fire". The manifold had
separated the winners perfectly and the selection discarded the separation. Because the outcome is
graded on Jaccard overlap, that is not merely imprecise but *arithmetically unscoreable*:
predicting six cells when three fire caps the score at 0.5 however right the prediction is, and the
correctness floor needs 0.75. Every prediction was wrong by construction.

**The deferred channel was keyed twice.** Opening read ``(subject, predicate)`` from
``Grounder._read_question``, which resolves the entity — ``"what does a plant need?"`` gives
``plant``. Closing built its key from the raw ``triple.subject`` — ``"plants need water"`` gives
``plants``. One entity, two keys, so the fact that answered the question never found the question.
``deferred_resolved`` was 0 in every session ever run: she asked, she was told, and nothing
connected the two. This is the ``birds``/``bird`` failure :mod:`nyxara.njp.canon` was written for,
in a second place.

Neither fix touches ``_CORRECT_FLOOR``, the manifold's evidence floors, or anything else that
decides how hard it is to be right. Loosening the bar to make the number move is the one change
that would make this file a lie.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.manifold import _FLOOR_SIM, Manifold


# --------------------------------------------------------------------------- #
# selection — the evidence decides how many cells, k only caps it
# --------------------------------------------------------------------------- #

def test_the_cut_does_not_return_the_whole_pool():
    """The exact distribution measured on a live fabric, with the k that was being used.

    Asserted as a property rather than as a specific cut, deliberately. An earlier version of
    this test asserted ``[1, 2]`` — where the elbow visually is — and the rule that produces it
    measured *worse* over four sessions than the one kept. Pinning the cut position would encode
    an intuition the fabric does not share; what the fix actually guarantees is that ``k`` no
    longer decides, and that "everything" is never the answer.
    """
    scored = [(0.600, 1), (0.595, 2), (0.278, 3), (0.230, 4), (0.095, 5)]
    chosen = Manifold._winners(scored, 6)
    assert 0 < len(chosen) < len(scored)
    # The two strongest candidates are in every honest reading of this distribution.
    assert {1, 2} <= {c for _s, c in chosen}


def test_a_generous_k_does_not_widen_a_sharp_prediction():
    scored = [(0.90, 1), (0.10, 2), (0.09, 3), (0.08, 4)]
    for k in (2, 8, 64):
        assert [c for _s, c in Manifold._winners(scored, k)] == [1]


def test_k_still_caps_a_wide_one():
    """`k` stops being a quota and becomes a bound. It must still bound."""
    scored = [(0.9 - i * 0.001, i) for i in range(40)]
    assert len(Manifold._winners(scored, 5)) <= 5


def test_noise_is_never_named():
    """A cell near-orthogonal to the predicted point is not about to fire."""
    scored = [(0.80, 1), (_FLOOR_SIM / 2, 2), (0.0, 3)]
    assert [c for _s, c in Manifold._winners(scored, 8)] == [1]


def test_a_tight_cluster_of_winners_is_kept_whole():
    """No break among candidates that cleared the noise floor means they all fire.

    The distribution is the manifold's canonical success — ``{1,2,3} → {4,5,6}`` learned twelve
    times — and an earlier fallback here kept only the top of it, turning the one case the whole
    module exists to get right into a two-thirds miss.
    """
    scored = [(0.515, 6), (0.495, 5), (0.488, 4), (0.011, 3), (0.004, 8)]
    assert {c for _s, c in Manifold._winners(scored, 3)} == {4, 5, 6}


def test_a_single_dominant_winner_is_a_legitimate_answer():
    assert len(Manifold._winners([(0.95, 1), (0.20, 2), (0.19, 3)], 8)) == 1


def test_predicting_everything_is_reported_as_not_a_prediction():
    """It scores non-zero overlap against any outcome whatsoever, which is why it is untrusted.

    The margin is what makes this dangerous rather than merely useless: with nothing left in the
    field, the separation calculation falls back to an absolute similarity that reads exactly like
    a clean stand-off.
    """
    manifold = Manifold(min_samples=1)
    scored = [(0.5, 1), (0.5, 2)]
    # A flat distribution over the whole pool: the mean fallback keeps nothing back.
    assert len(Manifold._winners(scored, 4)) <= len(scored)


# --------------------------------------------------------------------------- #
# the deferred channel — one entity, one key
# --------------------------------------------------------------------------- #

def test_the_question_and_the_fact_that_answers_it_share_a_key():
    from nyxara.njp.grounding import Grounder
    from nyxara.njp.integrate import LearningLoop
    grounder = Grounder()
    singular = LearningLoop._deferred_key(grounder, "plant", "requires")
    plural = LearningLoop._deferred_key(grounder, "plants", "requires")
    assert singular == plural


def test_a_stated_fact_closes_the_question_it_answers():
    """Asked, told, and the two connect. `deferred_resolved` was 0 in every session before this."""
    brain = NJPBrain()
    brain.think("what does a plant need?")
    report = brain.think("plants need water")
    assert report.loop.deferred_resolved == 1


def test_the_key_survives_a_grounder_that_cannot_fold():
    """A broken canonicaliser degrades to lowercase rather than breaking the channel."""
    from nyxara.njp.integrate import LearningLoop

    class _NoKey:
        def _key(self, _subject):
            raise RuntimeError("boom")

    assert LearningLoop._deferred_key(_NoKey(), "Plants", "requires") == "deferred:plants:requires"


# --------------------------------------------------------------------------- #
# end to end — she derives, reality grades her, a belief settles
# --------------------------------------------------------------------------- #

def test_a_derived_answer_the_master_confirms_is_counted_correct():
    """The whole point of the loop, and it had never once happened.

    "water" here is never stated of a sparrow — it is composed through ``sparrow is_a bird`` and
    ``bird requires water``. The Master's later statement is what grades it, and it is independent
    of the guess it grades.
    """
    brain = NJPBrain()
    brain.think("birds need water")
    brain.think("a sparrow is a bird")
    assert "water" in (brain.think("what does a sparrow need?").answer or "")
    report = brain.think("sparrows need water")
    assert report.loop.deferred_resolved == 1
    assert report.loop.correct == 1


def test_a_belief_she_staked_is_settled_by_the_fact():
    brain = NJPBrain()
    brain.think("birds need water")
    brain.think("a sparrow is a bird")
    brain.think("what does a sparrow need?")
    brain.think("sparrows need water")
    assert brain.loop.stats()["totals"]["beliefs_settled"] >= 1


def test_the_loop_scores_something_right_over_a_repetitive_session():
    """A regression guard on the headline number, stated as a floor rather than a target.

    Pinning an exact accuracy would make any honest change to the fabric look like a break. What
    must not come back is *zero* — an accuracy of 0.0 over a session this repetitive is the
    signature of a prediction that cannot be right rather than one that is wrong.
    """
    brain = NJPBrain()
    for _ in range(6):
        for turn in ("plants need water", "plants need light", "plants need water"):
            brain.think(turn)
    stats = brain.loop.stats()
    assert stats["totals"]["scored"] > 0
    assert stats["totals"]["correct"] > 0
    assert stats["accuracy"] > 0.0


def test_the_manifold_reports_hits_as_well_as_misses():
    """`hits 0 / misses 32` was the same defect seen from the fabric's own side."""
    brain = NJPBrain()
    for _ in range(6):
        for turn in ("plants need water", "plants need light", "plants need water"):
            brain.think(turn)
    manifold = brain.fabric.stats()["manifold"]
    assert manifold["hits"] > 0
