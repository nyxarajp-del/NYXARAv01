"""Learning *how to think* per kind of problem — the arms that were never registered.

``metareason.py``'s own docstring has claimed since it was written:

    Strategy selection learns. Which strategy wins for which kind of problem is not hardcoded
    past a sensible prior: **each (kind, strategy) pair is a bandit arm** scored by UCB1 over
    its real outcomes.

It was not true. :meth:`~nyxara.njp.metareason.MetaReasoner.choose` asks
``meta_learner.choose(f"strategy:{kind}")`` before anything else and
:meth:`~nyxara.njp.metareason.MetaReasoner.outcome` rewards the same arm — and nothing anywhere
registered an option under those arms. Measured on a live brain::

    MetaLearner arms: settle_steps, recall_k, reason_depth, seat:causal, seat:factual
    choose('strategy:causal') -> None

Every ``strategy:*`` arm absent, so ``choose`` returned ``None`` on every turn, the kind-blind
fallback decided everything, and ``reward`` found an empty ``_pending`` and credited nothing. The
``seat:*`` arms are present because :mod:`nyxara.njp.router` registers its options properly — the
pattern was already in the codebase, in a neighbouring organ.

**Why per-kind is the whole point.** :class:`~nyxara.njp.metareason.Strategy` carries one
``wins``/``trials`` pair spanning every kind it serves. ``derive`` is registered for factual,
causal and empirical, so a single number averages three different competences, and a strategy
excellent at one and useless at another reports the mean.
"""
from __future__ import annotations

from nyxara.njp.brain import NJPBrain
from nyxara.njp.metareason import MetaReasoner, ProblemKind
from nyxara.njp.selfmodel import MetaLearner


def _strategy_arms(learner: MetaLearner) -> dict:
    return {arm: sorted(bucket) for arm, bucket in learner.strategies.items()
            if arm.startswith("strategy:")}


# --------------------------------------------------------------------------- #
# The arms exist at all
# --------------------------------------------------------------------------- #
def test_every_strategy_is_declared_under_each_kind_it_serves():
    brain = NJPBrain()
    arms = _strategy_arms(brain.meta)
    assert arms, "the arm `choose` consults on every turn had no options registered"
    # `derive` is registered for three kinds and must appear under all three.
    serving_derive = [arm for arm, names in arms.items() if "derive" in names]
    assert len(serving_derive) >= 2


def test_the_bandit_now_returns_a_strategy_for_a_kind():
    brain = NJPBrain()
    picked = brain.meta.choose("strategy:causal")
    assert picked is not None
    assert picked.name in brain.metareason.strategies


def test_a_kind_with_one_strategy_still_gets_an_arm():
    brain = NJPBrain()
    arms = _strategy_arms(brain.meta)
    assert "strategy:introspective" in arms


def test_registering_without_a_learner_is_harmless():
    """The shared bandit is optional throughout this module and must stay optional."""
    reasoner = MetaReasoner()
    reasoner.register("x", (ProblemKind.CAUSAL,), lambda p, c: "y")
    assert "x" in reasoner.strategies


# --------------------------------------------------------------------------- #
# The same strategy is scored separately per kind
# --------------------------------------------------------------------------- #
def test_one_strategy_carries_a_different_record_for_each_kind():
    """The measured point: `derive` on factual and `derive` on causal are two questions."""
    learner = MetaLearner()
    reasoner = MetaReasoner(meta_learner=learner)
    reasoner.register("derive", (ProblemKind.FACTUAL, ProblemKind.CAUSAL), lambda p, c: "a")

    learner.reward("strategy:factual", 1.0, name="derive")
    learner.reward("strategy:factual", 1.0, name="derive")
    learner.reward("strategy:causal", 0.0, name="derive")

    factual = learner.strategies["strategy:factual"]["derive"]
    causal = learner.strategies["strategy:causal"]["derive"]
    assert factual.mean > causal.mean
    assert factual.trials == 2 and causal.trials == 1


def test_a_real_session_separates_the_kinds():
    brain = NJPBrain()
    for turn in ("a sparrow is a bird", "birds need water",
                 "what does a sparrow need?", "sparrow needs food",
                 "a crow is a bird", "what does a crow need?", "crow needs grain"):
        brain.think(turn)
    tried = {arm: {n: s.trials for n, s in bucket.items() if s.trials}
             for arm, bucket in brain.meta.strategies.items()
             if arm.startswith("strategy:")}
    assert any(tried.values()), "no (kind, strategy) arm was ever rewarded"


# --------------------------------------------------------------------------- #
# Credit goes to the strategy that earned it
# --------------------------------------------------------------------------- #
def test_a_named_reward_does_not_depend_on_what_was_chosen_last():
    """A strategy outcome can arrive many turns after the choice.

    `_pending` holds only the most recent choice for an arm, so any question of the same kind in
    between would hand the credit to whichever strategy was chosen most recently.
    """
    learner = MetaLearner()
    learner.register("strategy:causal", "derive", "derive")
    learner.register("strategy:causal", "simulate", "simulate")

    learner.choose("strategy:causal")          # moves `_pending`, whichever it picks
    learner.choose("strategy:causal")
    learner.reward("strategy:causal", 1.0, name="derive")

    assert learner.strategies["strategy:causal"]["derive"].trials == 1
    assert learner.strategies["strategy:causal"]["simulate"].trials == 0


def test_an_unnamed_reward_keeps_the_old_pending_behaviour():
    """Callers that reward immediately after choosing must be unaffected."""
    learner = MetaLearner()
    learner.register("knob", "a", "a")
    chosen = learner.choose("knob")
    learner.reward("knob", 1.0)
    assert chosen is not None and chosen.trials == 1


def test_rewarding_a_name_that_was_never_registered_does_nothing():
    learner = MetaLearner()
    learner.register("strategy:causal", "derive", "derive")
    learner.reward("strategy:causal", 1.0, name="nonexistent")
    assert learner.strategies["strategy:causal"]["derive"].trials == 0


def test_the_winner_for_a_kind_is_reported_once_it_has_been_tried_enough():
    learner = MetaLearner(min_trials=2)
    learner.register("strategy:causal", "derive", "derive")
    learner.register("strategy:causal", "simulate", "simulate")
    for _ in range(3):
        learner.reward("strategy:causal", 1.0, name="derive")
        learner.reward("strategy:causal", 0.0, name="simulate")
    best = learner.best("strategy:causal")
    assert best is not None and best.name == "derive"
