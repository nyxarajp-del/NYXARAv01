"""One surprise, shared — and the 113/0 rule that governs every prediction in this package.

`brain.py` carries the reason prediction-as-a-control-loop was tried and removed: the key was the
raw stimulus, so the only thing that could ever have scored it was the identical sentence arriving
again with the truth attached. **113 turns, 113 predictions registered, 0 scored.** A prediction
that cannot in principle be observed is not a prediction; it is a counter going up.

So the load-bearing test here is `test_predictions_are_scored_not_merely_registered`. Everything
else is about the three organs that should share the resulting signal and, before this, could not:

* `attention` weights `prediction_error` highest of its five terms and was filling it from the
  manifold's *settling residual* — a real number, and not the one the prediction loop produces;
* `curiosity` reads `predictor.outcomes` already, gated at `min_failures` so that once is noise;
* `memory` had **nowhere to put a salience at all**, which is why "keep what was surprising" could
  not be a policy.

The falsification for the memory hook is `test_salience_changes_what_survives`: if retention of a
surprising memory does not diverge from a dull one, the parameter is decoration and should go.
"""

from __future__ import annotations

import time

import pytest

from nyxara.njp.attention import Attention, Candidate
from nyxara.njp.brain import NJPBrain
from nyxara.njp.levels import HierarchicalMemory
from nyxara.njp.memory import HoloMemory

_TURNS = ["a mammal is warm blooded", "a seal is a mammal", "is a seal warm blooded",
          "fire causes heat", "what does fire cause", "water causes growth",
          "a bird is feathered", "what is a seal", "the plant grew", "what does water cause"]


@pytest.fixture(scope="module")
def lived():
    brain = NJPBrain()
    for turn in _TURNS:
        brain.think(turn)
    return brain


# --------------------------------------------------------------------------- #
# The 113/0 rule
# --------------------------------------------------------------------------- #
def test_predictions_are_scored_not_merely_registered(lived):
    """The direct descendant of the measurement that got the old loop deleted."""
    predictor = lived.predictor
    assert predictor.predictions > 0, "nothing predicted anything"
    assert predictor.scored > 0, (
        "predictions registered and none scored — this is the 113/0 failure returning")
    assert predictor.accuracy is not None


def test_an_unresolvable_prediction_would_show_up_as_an_unscored_one(lived):
    """The open queue is allowed to be non-empty — a deferred question waits on the Master — but
    it must not be the *whole* of it, which is exactly what the raw-stimulus key produced."""
    predictor = lived.predictor
    assert predictor.scored >= 1
    assert predictor.scored <= predictor.predictions


# --------------------------------------------------------------------------- #
# One surprise, named
# --------------------------------------------------------------------------- #
def test_attention_reads_the_scored_surprise_not_only_the_settling_one():
    class Outcome:
        key, surprise, organ = "njp-7:next-state", 0.9, "manifold"

    class Predictor:
        outcomes = [Outcome()]

    class Brain:
        predictor = Predictor()

    class Thought:
        cycle_id = "njp-7"

    got, source = Attention(Brain())._scored_surprise(Thought())
    assert got == pytest.approx(0.9)
    assert source == "manifold"


def test_an_outcome_from_another_turn_does_not_move_this_turns_focus():
    """Matched on `cycle_id` because `integrate` keys predictions that way. Yesterday's shock is
    not this turn's."""
    class Outcome:
        key, surprise, organ = "njp-3:next-state", 0.9, "manifold"

    class Brain:
        class predictor:
            outcomes = [Outcome()]

    class Thought:
        cycle_id = "njp-7"

    assert Attention(Brain())._scored_surprise(Thought()) == (0.0, "")


def test_a_brain_without_a_predictor_still_attends():
    class Thought:
        cycle_id = "njp-1"

    assert Attention(object())._scored_surprise(Thought()) == (0.0, "")


def test_the_error_source_is_recorded_so_the_two_can_be_told_apart():
    assert Candidate(prediction_error=0.4, error_source="manifold").to_dict()["error_source"] \
        == "manifold"


def test_prediction_error_still_carries_the_most_weight():
    """The weighting was already right; only the number under it was wrong."""
    only_error = Candidate(prediction_error=1.0).salience
    only_novelty = Candidate(novelty=1.0).salience
    assert only_error > only_novelty


# --------------------------------------------------------------------------- #
# The memory hook
# --------------------------------------------------------------------------- #
def test_salience_changes_what_survives():
    """The falsification: no divergence means the parameter is decoration."""
    memory = HierarchicalMemory()
    memory.remember("dull", "an ordinary fact", salience=0.0)
    memory.remember("shock", "a surprising fact", salience=1.0)
    later = time.time() + 3 * 86_400
    assert memory.retention("shock", now=later) > memory.retention("dull", now=later) * 2


def test_salience_reuses_the_repo_forgetting_law():
    """One rehearsal's worth of durability, not a multiplier invented here."""
    base = 1.0
    assert HierarchicalMemory._with_salience(base, 0.0) == base
    assert HierarchicalMemory._with_salience(base, 1.0) == \
        pytest.approx(HierarchicalMemory._next_stability(base))
    middle = HierarchicalMemory._with_salience(base, 0.5)
    assert base < middle < HierarchicalMemory._next_stability(base)


def test_salience_is_optional_and_defaults_to_no_change():
    memory = HierarchicalMemory()
    memory.remember("plain", "no salience given")
    assert memory.entries["plain"].stability == \
        HierarchicalMemory._with_salience(memory.entries["plain"].stability, 0.0)


def test_a_quiet_rewrite_does_not_erase_a_shock():
    """High-water mark: what a memory cost to learn is not undone by mentioning it again."""
    store = HoloMemory()
    store.remember("k", "learned in a shock", salience=0.9)
    store.remember("k", "mentioned again", salience=0.0)
    assert store.traces["k"].salience == pytest.approx(0.9)


def test_the_trace_reports_its_salience():
    store = HoloMemory()
    store.remember("k", "text", salience=0.5)
    assert store.traces["k"].to_dict()["salience"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Curiosity's channel
# --------------------------------------------------------------------------- #
def test_curiosity_reads_outcomes_and_is_gated_rather_than_dead(lived):
    """It fires on repeated failure of one kind, not on one miss. Once is noise."""
    curiosity = lived.curiosity
    assert curiosity.min_failures >= 2, "a single miss must not become a standing question"
    assert hasattr(curiosity, "_from_failures")


def test_curiosity_raises_a_question_once_a_failure_repeats():
    from nyxara.njp.curiosity import Gap

    class Diagnosis:
        kind = "world_model"

    class Outcome:
        correct = False
        diagnosis = Diagnosis()

    brain = NJPBrain()
    brain.predictor.outcomes = [Outcome() for _ in range(brain.curiosity.min_failures + 1)]
    raised = brain.curiosity._from_failures()
    assert raised, "repeated failures of one kind must become a question"
    assert raised[0].gap == Gap.REPEATED_FAILURE


# --------------------------------------------------------------------------- #
# The hook is filled, not merely present
# --------------------------------------------------------------------------- #
def test_a_lived_turn_writes_its_salience(lived):
    """A parameter nothing ever fills is decoration, which is the whole failure mode here."""
    traces = getattr(lived.memory, "traces", {}) or {}
    written = [t for k, t in traces.items() if k.startswith("turn-")]
    assert written, "no turn was remembered at all"
    assert any(getattr(t, "salience", 0.0) > 0.0 for t in written)


def test_salience_varies_between_turns(lived):
    """The falsification. A constant signal carries no information however it is plumbed."""
    entries = getattr(lived.levels, "entries", {}) or {}
    stabilities = {round(e.stability, 3) for k, e in entries.items() if k.startswith("turn-")}
    assert len(stabilities) > 1, (
        f"every turn was equally surprising ({stabilities}) — the signal is constant")


def test_the_brain_and_attention_agree_on_what_surprise_means(lived):
    """One definition, read from one place. Two that drift apart is the defect this phase closed."""
    import inspect

    source = inspect.getsource(type(lived)._turn_salience)
    assert "_scored_surprise" in source, "the brain must reuse attention's rule, not restate it"


# --------------------------------------------------------------------------- #
# The within-turn prediction — the one shape the 113/0 key could never have
# --------------------------------------------------------------------------- #
_UNANSWERABLE = ["what is a glarbex", "why does the zorbin flicker", "what causes vemthral",
                 "how does quorvin work", "what is the purpose of a thrumble",
                 "what does a blint require"]


@pytest.fixture(scope="module")
def deliberated():
    brain = NJPBrain()
    for turn in _UNANSWERABLE:
        brain.think(turn)
    return brain


def test_the_deliberation_prediction_resolves_in_the_same_turn(deliberated):
    """Nothing waits in the open queue: it is scored one call after it is registered."""
    outcomes = [o for o in deliberated.predictor.outcomes if ":deliberation" in str(o.key)]
    assert outcomes, "deliberation never ran, so nothing was predicted about it"
    assert all(o.actual in ("answered", "unanswered") for o in outcomes)


def test_it_raises_the_scored_count_which_is_the_condition_for_keeping_it(deliberated):
    """The plan's own gate: added only if `scored` demonstrably rises. This is that measurement."""
    predictor = deliberated.predictor
    delib = sum(1 for o in predictor.outcomes if ":deliberation" in str(o.key))
    assert delib > 0
    assert predictor.scored >= delib
    assert predictor.accuracy is not None


def test_the_prior_moves_with_the_evidence(deliberated):
    """Predict, be wrong, and expect differently next time — otherwise it is a fixed guess."""
    model = deliberated.self_model
    recorded = model.capabilities.get("reasoning:deliberation") if model else None
    assert recorded is not None, "the outcome never reached the self-model"
    assert recorded.observations >= 2
    assert recorded.level != 0.5, "the prior never moved off its untested default"


def test_an_answered_turn_predicts_nothing_about_deliberation():
    """The branch only runs when grounding came back empty; predicting otherwise would be a
    counter going up on turns where deliberation never happened."""
    brain = NJPBrain()
    brain.think("a mammal is warm blooded")
    assert not [o for o in brain.predictor.outcomes if ":deliberation" in str(o.key)]
