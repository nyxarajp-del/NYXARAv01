"""A world model that predicted and never checked itself.

``InternalUniverse.reconcile`` had no caller anywhere in the package. The field rolled the model
forward on every cycle — ``rollouts`` climbed — and nothing ever compared a rollout to what
happened next, so ``reconciled`` and ``surprises`` were structurally zero and ``mean_error`` was
``None``. "The model has been wrong six times" was a sentence the universe could not form, which
is also the only form in which a retirement decision can be made at all.

The caller that was missing is ``field._sync_world``, which already had the observation in hand
and was handing it straight to ``observe``. ``reconcile`` observes *and* grades, so the fix is
which of the two it calls — plus forwarding the stated word order, which is the only orientation
evidence a joint observation carries and would otherwise have been lost.
"""

from __future__ import annotations

from nyxara.njp import NJPBrain


def _measured(brain: NJPBrain, rows) -> NJPBrain:
    for name, water, growth in rows:
        brain.think(f"plant {name} got {water} litre water")
        brain.think(f"plant {name} grew {growth} cm")
    return brain


def test_a_rollout_is_scored_against_what_happened_next():
    brain = _measured(NJPBrain(), (("a", 2, 20), ("b", 4, 38), ("c", 1, 11),
                                   ("d", 6, 55), ("e", 3, 29), ("f", 5, 47),
                                   ("g", 7, 64), ("h", 8, 72)))
    stats = brain.universe.stats()
    assert stats["rollouts"] > 0, stats
    assert stats["reconciled"] > 0, stats
    assert stats["mean_error"] is not None, stats


def test_the_word_order_survives_being_graded():
    """`observe`'s `order` is the one thing in a joint observation that can say which way an
    arrow runs — five readings of water beside growth are Markov-equivalent without it."""
    brain = _measured(NJPBrain(), (("a", 2, 20), ("b", 4, 38), ("c", 1, 11),
                                   ("d", 6, 55), ("e", 3, 29), ("f", 5, 47)))
    assert brain.universe.stats()["oriented_relations"] > 0, brain.universe.stats()


def test_a_model_the_world_contradicts_records_a_surprise():
    """The count of times the world contradicted her is the single most useful number about a
    world model, and it is the one a model that never checks itself cannot produce."""
    brain = _measured(NJPBrain(), (("a", 2, 20), ("b", 4, 40), ("c", 6, 60),
                                   ("d", 8, 80), ("e", 10, 100), ("f", 3, 30)))
    steady = brain.universe.stats()
    # Now the relation inverts underneath the fitted law.
    _measured(brain, (("p", 2, 90), ("q", 4, 70), ("r", 6, 50),
                      ("s", 8, 30), ("t", 10, 10)))
    broken = brain.universe.stats()
    assert broken["surprises"] > steady["surprises"], (steady, broken)
    assert broken["mean_error"] > steady["mean_error"], (steady, broken)


def test_reconciling_does_not_double_count_the_observation():
    """`reconcile` hands the state to `observe` itself, so calling both would fit every row twice."""
    brain = _measured(NJPBrain(), (("a", 2, 20), ("b", 4, 38), ("c", 1, 11)))
    arrow = brain.universe.relations.get(("plant.water", "plant.growth"))
    assert arrow is not None, brain.universe.stats()
    # Three plants, each contributing one partial reading and one complete pair.
    assert arrow.n <= 3, arrow.n
