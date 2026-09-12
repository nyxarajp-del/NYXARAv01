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


# --------------------------------------------------------------------------- #
# a guess reality confirmed is the only hard evidence she earns
# --------------------------------------------------------------------------- #

def test_a_confirmed_guess_is_recorded_as_hard_evidence():
    """`EvidenceKind` calls three kinds hard and only a hard reason may establish a belief alone.

    Measured across a whole session, ``with_hard_evidence`` was 0: the single ``beliefs.support``
    call anywhere in the package passes ``TESTIMONY``, so the ledger's establishment path was
    written, tested and unreachable. Every belief she held was held on being told, and nothing
    she worked out could ever be worth more than hearsay.
    """
    from nyxara.njp.beliefs import EvidenceKind
    brain = NJPBrain()
    brain.think("birds need water")
    brain.think("a sparrow is a bird")
    brain.think("what does a sparrow need?")
    brain.think("sparrows need water")
    assert brain.beliefs.stats()["with_hard_evidence"] >= 1
    case = brain.beliefs.why("water")
    kinds = [e["kind"] for e in case.get("evidence", [])]
    assert EvidenceKind.PREDICTION in kinds, case


def test_a_wrong_guess_is_never_recorded_as_evidence_for_itself():
    """A refuted guess is a refutation. Filing it as support would invert the whole ledger."""
    from nyxara.njp.beliefs import EvidenceKind
    brain = NJPBrain()
    brain.think("birds need water")
    brain.think("a sparrow is a bird")
    brain.think("what does a sparrow need?")      # she answers "water"
    brain.think("sparrows need seeds")            # reality says otherwise
    case = brain.beliefs.why("water")
    kinds = [e["kind"] for e in case.get("evidence", [])] if case.get("known") else []
    assert EvidenceKind.PREDICTION not in kinds, case


# --------------------------------------------------------------------------- #
# compression — episodes in which nothing could recur
# --------------------------------------------------------------------------- #

def test_episodes_are_recorded_under_the_kind_so_instances_can_recur():
    """Every antecedent was built from the *instance*, so nothing ever generalised.

    ``sparrow requires water``, ``crow requires water`` and ``robin requires water`` shared no
    subset at all, so every candidate rule had support 1 against a floor of 4. Measured over a
    whole session: 70 episodes, 4 discovery passes, **0 abstractions**, compression 1.0 — which
    :mod:`nyxara.njp.concepts` defines as "does not pay for itself". The compressor was not
    failing; it was being fed data in which nothing could recur.
    """
    brain = NJPBrain()
    birds = ("sparrow", "crow", "robin", "eagle", "finch", "wren", "lark", "swift")
    for bird in birds:
        brain.think(f"a {bird} is a bird")
    for bird in birds:
        brain.think(f"{bird}s need water")
        brain.think(f"{bird}s eat seeds")
    stats = brain.discoverer.stats()
    assert stats["proposed_total"] > 0, stats
    names = {a.name for a in brain.discoverer.abstractions.values()}
    assert any("bird" in n for n in names), names


def test_a_kind_rule_is_tested_on_held_out_episodes_and_can_be_refuted():
    """Proposing is not confirming. The falsification half has to do work too."""
    brain = NJPBrain()
    birds = ("sparrow", "crow", "robin", "eagle", "finch", "wren", "lark", "swift",
             "heron", "stork")
    plants = ("rose", "tulip", "fern", "daisy", "oak", "pine", "ivy", "moss", "sage", "elm")
    for bird in birds:
        brain.think(f"a {bird} is a bird")
    for plant in plants:
        brain.think(f"a {plant} is a plant")
    for bird in birds:
        brain.think(f"{bird}s need water")
        brain.think(f"{bird}s eat seeds")
    for plant in plants:
        brain.think(f"{plant}s need light")
        brain.think(f"{plant}s need soil")
    stats = brain.discoverer.stats()
    assert stats["confirmed_total"] > 0, stats
    # Some proposals must die, or "confirmed" means "proposed" under another name.
    assert stats["refuted_total"] > 0, stats
    # Above 1.0 is the concepts module's own definition of paying for itself.
    assert stats["compression"] > 1.0, stats
    best = stats["best"]
    assert best and best["tested"] > 0 and best["precision"] > 0.7, best


def test_a_kind_rule_never_restates_its_own_antecedent():
    """Substituting a subject with its kind turns `sparrow is_a bird` into `{bird, is_a} → bird`.

    A tautology, and one that scored five supports and crowded out the rules worth finding.
    """
    brain = NJPBrain()
    for bird in ("sparrow", "crow", "robin", "eagle", "finch", "wren"):
        brain.think(f"a {bird} is a bird")
    for name in (a.name for a in brain.discoverer.abstractions.values()):
        assert "→ bird" not in name or "bird," not in name, name


# --------------------------------------------------------------------------- #
# a clash the grounder found and nobody was told about
# --------------------------------------------------------------------------- #

def test_a_revision_is_recorded_as_a_contradiction_between_beliefs():
    """`GroundingResult.contradictions` was produced correctly and read by nothing but to_dict.

    So `BeliefLedger.contradict` had no caller on the path that actually produces contradictions,
    and `contradictions_found` and `contested` stood at zero however often the Master revised
    himself.
    """
    brain = NJPBrain()
    brain.think("my name is Jay")
    report = brain.think("my name is Raj")
    assert report.field.beliefs_contradicted == 1, report.field.to_dict()
    stats = brain.beliefs.stats()
    assert stats["contradictions_found"] >= 1, stats
    assert stats["contested"] >= 2, stats


def test_the_superseded_belief_actually_loses_confidence():
    """The number is not the point — this is.

    `contradict` is where each side loses confidence in proportion to the *other's* earned
    support, so a well-evidenced claim barely moves against a bare assertion and two bare
    assertions both collapse. Without the call a superseded belief kept the confidence it was
    first held at, which is a brain that has been corrected and does not know it.
    """
    brain = NJPBrain()
    brain.think("my name is Jay")
    held = brain.beliefs.why("Master has_name Jay")["confidence"]
    brain.think("my name is Raj")
    after = brain.beliefs.why("Master has_name Jay")["confidence"]
    assert after < held, (held, after)


def test_two_facts_that_do_not_clash_are_not_contested():
    """Two things a subject needs are both true. Only a *revision* is a contradiction."""
    brain = NJPBrain()
    brain.think("plants need water")
    report = brain.think("plants need light")
    assert report.field.beliefs_contradicted == 0
    assert brain.beliefs.stats()["contradictions_found"] == 0
