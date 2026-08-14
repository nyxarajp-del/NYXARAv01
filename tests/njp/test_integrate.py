"""The loop that makes the organs learn from each other.

Every organ NJP needs already existed. The world model took events, the predictor scored
expectations, the levels consolidated, the discoverer tested abstractions, the self-model took
observations, the readout took gradient steps — and over a real 113-turn session every one of
those counters was **zero**, because nothing ever called them.

These tests are about exactly one claim: the counters move on an ordinary conversation. Not that
she is right — :func:`test_a_closed_loop_is_not_a_correct_loop` is here to keep that distinction
visible — but that the signals now reach the organs that learn from them.
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.grounding import Grounder
from nyxara.njp.integrate import LearningLoop, LoopReport
from nyxara.njp.predict import ErrorKind, PredictionEngine
from nyxara.njp.world import _EVENT_PREDICATES, WorldView


@pytest.fixture()
def brain() -> NJPBrain:
    return NJPBrain()


def _teach(brain: NJPBrain) -> None:
    """A short, ordinary session — the kind of thing the Master would actually type."""
    for line in ("my name is Jay",
                 "Ravi is a person",
                 "Sara is a person",
                 "Ravi lives in Mumbai",
                 "Sara lives in Delhi",
                 "Ravi works at Acme",
                 "the glass broke",
                 "the server crashed because the disk failed",
                 "what is my name",
                 "where does Ravi live"):
        brain.think(line)


# ---- the loop exists and runs ----------------------------------------------- #
def test_the_brain_builds_the_loop(brain: NJPBrain):
    assert isinstance(brain.loop, LearningLoop)


def test_every_turn_closes_the_loop(brain: NJPBrain):
    thought = brain.think("my name is Jay")
    assert isinstance(thought.loop, LoopReport)
    assert thought.loop.turn == 1


def test_an_organ_that_is_off_is_absent_not_zeroed():
    class _Off:
        loop_enabled = False

    off = NJPBrain(_Off())
    assert off.loop is None
    assert "loop" not in off.stats()
    assert off.think("my name is Jay").loop is None      # and the turn still works


# ---- grounding: happenings become events ------------------------------------ #
def test_an_intransitive_happening_is_extracted():
    """"the glass broke" is a complete sentence and used to ground to nothing at all."""
    triples = Grounder().ground("the glass broke").triples
    assert [(t.subject, t.predicate) for t in triples] == [("glass", "broke")]


def test_a_happening_becomes_an_event_and_a_fact_does_not():
    grounder, world = Grounder(), WorldView()
    assert len(world.from_grounding(grounder.ground("the apple fell"))) == 1
    assert len(world.from_grounding(grounder.ground("Jay lives in Mumbai"))) == 0


def test_every_event_predicate_the_world_recognises_is_now_reachable():
    """The gap this closed: 21 predicates were listed and 1 of them could ever be produced."""
    grounder = Grounder()
    produced = set()
    for text in ("the apple fell", "the glass broke", "the server crashed",
                 "the build failed", "Jay went to Mumbai", "Jay bought a car",
                 "gravity causes rain"):
        produced.update(t.predicate for t in grounder.ground(text).triples)
    reachable = produced & _EVENT_PREDICATES
    assert len(reachable) >= 6, reachable


def test_because_grounds_both_halves_and_the_link_between_them():
    triples = Grounder().ground("the server crashed because the disk failed").triples
    got = {(t.subject, t.predicate, t.object) for t in triples}
    assert ("disk", "failed", "") in got
    assert ("server", "crashed", "") in got
    assert ("disk", "causes", "server") in got


def test_hinglish_states_a_cause_too():
    """The Master writes in Hinglish; an English-only causal reader is blind to half of it."""
    triples = Grounder().ground("the build failed kyunki the test broke").triples
    assert any(t.predicate == "causes" for t in triples)


def test_a_causal_sentence_does_not_swallow_a_plain_fact():
    triples = Grounder().ground("Jay lives in Mumbai").triples
    assert [(t.subject, t.predicate, t.object) for t in triples] == [
        ("Jay", "located_in", "Mumbai")]


# ---- the world stops being empty --------------------------------------------- #
def test_the_world_records_events_on_a_real_session(brain: NJPBrain):
    _teach(brain)
    stats = brain.stats()["world"]
    assert stats["events"] > 0
    assert stats["observations"] > 0


def test_the_dynamics_model_learns_transitions(brain: NJPBrain):
    _teach(brain)
    assert brain.stats()["world"]["dynamics"]["transitions"] > 0


def test_a_transition_is_recorded_once_per_turn_not_twice(brain: NJPBrain):
    """A repair that re-recorded the same transition reported 202 of them for 113 turns."""
    _teach(brain)
    transitions = brain.stats()["world"]["dynamics"]["transitions"]
    assert transitions <= brain.turns


# ---- prediction actually gets scored ------------------------------------------ #
def test_predictions_are_scored_not_merely_registered(brain: NJPBrain):
    _teach(brain)
    stats = brain.stats()["predict"]
    assert stats["predictions"] > 0
    assert stats["scored"] > 0
    assert stats["accuracy"] is not None


def test_the_open_queue_does_not_grow_without_bound(brain: NJPBrain):
    """Before this, every turn added an expectation that nothing could ever observe."""
    _teach(brain)
    stats = brain.stats()["predict"]
    assert stats["open"] < stats["predictions"] / 2


def test_the_next_state_prediction_is_scored_against_what_it_predicted(brain: NJPBrain):
    """The anticipation is computed pre-settle; the settle is what scores it. No circularity.

    What is checked is the identity of the two halves: what got scored is exactly the cell list
    the manifold produced *before* the fabric ran, and what it was scored against is exactly what
    fired. Nothing about the answer she composed afterwards is on either side.
    """
    # The manifold refuses to name any cell until it has learned enough transitions to have an
    # opinion — it reports "only 3 transitions learned, need 8" and predicts nothing. Run past
    # that floor first, so what is under test is the scoring and not the cold start.
    for i in range(12):
        brain.think(f"person{i} lives in Mumbai")
    thought = brain.think("Ravi lives in Mumbai")

    anticipated = thought.percept.anticipated
    assert anticipated is not None and anticipated.cells

    key = f"{thought.cycle_id}:next-state"
    outcome = next(o for o in brain.predictor.outcomes if o.key == key)
    assert tuple(outcome.expected) == tuple(anticipated.cells)
    assert tuple(outcome.actual) == tuple(thought.percept.settled.fired)


def test_a_closed_loop_is_not_a_correct_loop(brain: NJPBrain):
    """``scored > 0`` means she is measuring herself. It does not mean she is right."""
    _teach(brain)
    stats = brain.stats()["predict"]
    assert stats["scored"] > 0
    assert 0.0 <= stats["accuracy"] <= 1.0     # honestly reported, whatever it is


# ---- deferred answers: graded by the Master, never by herself ------------------ #
def test_an_unanswered_question_is_left_open(brain: NJPBrain):
    brain.think("where does Ravi live")
    assert brain.loop.stats()["deferred_open"] == 1
    assert brain.loop.totals["deferred_opened"] == 1


def test_a_question_never_answered_is_never_scored(brain: NJPBrain):
    """Nothing told her the answer, so there is nothing to grade. Silence is not a score."""
    for _ in range(4):
        brain.think("where does Ravi live")
    assert brain.loop.totals["deferred_resolved"] == 0


def test_the_masters_later_statement_grades_the_guess(brain: NJPBrain):
    brain.think("where does Ravi live")            # she does not know
    brain.think("Ravi lives in Mumbai")            # now she is told
    assert brain.loop.totals["deferred_resolved"] == 1


def test_being_told_something_she_got_wrong_counts_as_a_correction(brain: NJPBrain):
    brain.think("Ravi lives in Delhi")             # she is taught one thing
    brain.think("what is my name")                 # unrelated, keeps the session ordinary
    brain.loop._deferred.clear()
    brain.loop._open_deferred(
        _Stub(stimulus="where does Ravi live"), brain.grounder, brain.predictor,
        _StubGrounding(answer=_StubAnswer("Delhi")), LoopReport())
    brain.think("Ravi lives in Mumbai")            # the Master says otherwise
    assert brain.loop.totals["corrections"] == 1


# ---- consolidation, abstraction and curiosity run on turns, not on a clock ------ #
def test_memory_consolidates_on_a_turn_cadence(brain: NJPBrain):
    _teach(brain)
    assert brain.stats()["levels"]["consolidations"] > 0


def test_repetition_promotes_an_episode_to_semantic(brain: NJPBrain):
    for _ in range(3):
        brain.think("Ravi lives in Mumbai")
    for _ in range(10):
        brain.think("Sara is a person")
    assert brain.stats()["levels"]["promoted_total"] > 0


def test_abstraction_discovery_runs_and_tests_on_held_out_episodes(brain: NJPBrain):
    """A pattern needs to recur before there is anything to abstract from.

    The episodes are deliberately repeated: a run of sixteen *distinct* facts has no recurring
    antecedent set, so proposing nothing from it is correct behaviour rather than a dead organ.

    Three sentences rather than two, because :meth:`DiscoveryEngine._split` strides the episodes
    and a two-way alternation lines up exactly with an even stride — every "Sara" episode lands in
    the fit half and every "Ravi" one in the held-out half, so no rule is ever testable and
    everything stays conjectured forever. That is a property of the test data, not of the split.
    """
    for _ in range(12):
        brain.think("Ravi lives in Mumbai")
        brain.think("Sara lives in Delhi")
        brain.think("Amit lives in Pune")
    stats = brain.stats()["discover"]
    assert stats["passes"] > 1
    assert stats["proposed_total"] > 0
    # Whatever survived did so on episodes it was never fitted to — a rule that only describes
    # the episodes that suggested it is not a rule.
    assert stats["confirmed_total"] + stats["refuted_total"] > 0


def test_curiosity_raises_questions_from_measured_gaps(brain: NJPBrain):
    for i in range(20):
        brain.think(f"person{i} is a person")
        brain.think(f"person{i} lives in Mumbai")
    assert brain.stats()["curiosity"]["passes"] > 0


def test_the_cadence_is_configurable(brain: NJPBrain):
    loop = LearningLoop(brain, consolidate_every=1, discover_every=1, wonder_every=1)
    assert loop.consolidate_every == 1
    rep = LoopReport()
    brain.turns = 1
    loop._slow(rep)
    assert rep.consolidated


# ---- the self-model stops being untested ---------------------------------------- #
def test_capabilities_are_graded_from_real_turns(brain: NJPBrain):
    _teach(brain)
    caps = brain.stats()["self_model"]["capabilities"]
    assert caps["grounding"]["observations"] > 0
    assert caps["prediction"]["observations"] > 0


def test_a_capability_the_turn_did_not_exercise_is_not_graded(brain: NJPBrain):
    """Counts that mean nothing would destroy the only thing a Beta posterior is good for."""
    _teach(brain)
    caps = brain.stats()["self_model"]["capabilities"]
    assert caps["planning"]["observations"] == 0
    assert caps["tool_use"]["observations"] == 0


def test_she_can_now_say_which_faculty_is_weakest(brain: NJPBrain):
    _teach(brain)
    report = brain.stats()["self_model"]
    assert report["untested"] != list(report["capabilities"])   # not everything is untested now


# ---- gradients run on every turn, not only in a dream ----------------------------- #
def test_the_readout_takes_gradient_steps_during_ordinary_turns(brain: NJPBrain):
    _teach(brain)
    assert brain.stats()["readout"]["steps"] > 0


def test_the_first_turn_has_nothing_to_train_on(brain: NJPBrain):
    """The readout learns 'what fired then → what fires now'. Turn one has no 'then'."""
    thought = brain.think("my name is Jay")
    assert not thought.loop.trained


# ---- errors are diagnosed and routed to an organ that repairs them ---------------- #
def test_every_repairable_error_kind_has_a_repair_registered(brain: NJPBrain):
    for kind in (ErrorKind.PERCEPTION, ErrorKind.GROUNDING, ErrorKind.MEMORY,
                 ErrorKind.WORLD_MODEL, ErrorKind.REASONING):
        assert kind in brain.predictor.repairs


def test_a_diagnosed_miss_is_actually_repaired(brain: NJPBrain):
    _teach(brain)
    stats = brain.stats()["predict"]
    if sum(stats["by_kind"].values()):
        assert sum(stats["repaired"].values()) > 0


def test_the_world_repair_replays_only_a_genuine_before_after_pair(brain: NJPBrain):
    """The fabric overwrites its own previous snapshot mid-settle, so the loop keeps its own.

    Reading the fabric's ``_prev_snapshot`` instead made ``before is after`` on every single miss,
    so the replay was a no-op that still reported itself repaired.
    """
    brain.think("my name is Jay")
    brain.think("Ravi lives in Mumbai")
    loop = brain.loop

    # A real pair — two different settles — replays.
    loop._prev_snapshot, loop._snapshot = object(), object()
    replayed = {"n": 0}
    brain.fabric.manifold.learn_transition = lambda *a, **k: replayed.__setitem__(
        "n", replayed["n"] + 1)
    assert loop._repair_world(object()) is True
    assert replayed["n"] == 1

    # The degenerate pair the fabric would have handed it does not.
    same = object()
    loop._prev_snapshot = loop._snapshot = same
    assert loop._repair_world(object()) is False
    assert replayed["n"] == 1


def test_an_outcome_carries_the_evidence_its_repair_needs():
    engine = PredictionEngine()
    engine.predict("k", "Mumbai", stimulus="where does Ravi live")
    got = engine.observe("k", "Delhi")
    assert got is not None and got.context.get("stimulus") == "where does Ravi live"


def test_the_fabric_accepts_an_error_from_outside_the_settle():
    from nyxara.njp.fabric import Fabric

    fabric = Fabric()
    for _ in range(fabric.neurogenesis_window * 8):
        fabric.note_error(1.0)
    # Bounded exactly as the in-settle path is, so an organ cannot flood it into growing.
    assert len(fabric._errors) <= fabric.neurogenesis_window * 4


# ---- the reasoner is reachable from a turn ----------------------------------------- #
def test_an_unanswerable_question_descends_the_ladder(brain: NJPBrain):
    """A question about something she was never told — no structure can answer it.

    This used to use ``mera naam kya hai``, on the grounds that grounding had no answer for it.
    It has one now, so the example had to change: an answerable question must *not* pay for the
    ladder, and Hinglish being answerable is the point of the change that broke this test.
    """
    brain.think("my name is Jay")
    brain.think("who is Zoran")                    # never mentioned, in any language
    assert brain.stats()["reason"]["passes"] > 0


def test_an_answerable_question_does_not_pay_for_the_ladder(brain: NJPBrain):
    """A mind that runs a proof engine over 'what is my name' is broken, not careful."""
    brain.think("my name is Jay")
    brain.think("what is my name")
    assert brain.stats()["reason"]["passes"] == 0


# ---- the whole cycle, end to end ---------------------------------------------------- #
def test_the_counters_that_were_all_zero_are_no_longer_zero(brain: NJPBrain):
    """The one test this entire module exists for.

    Each of these was measured at exactly 0 over a 113-turn session before the integration layer
    existed, with every underlying algorithm already written and already passing its own tests.
    """
    for i in range(8):
        brain.think(f"person{i} is a person")
        brain.think(f"person{i} lives in Mumbai")
        # Repeated on purpose: promotion to semantic memory requires the same claim to recur from
        # independent episodes, which is the whole difference between "semantic" and "old".
        brain.think("Ravi lives in Mumbai")
    brain.think("the server crashed because the disk failed")
    brain.think("where does person0 live")

    stats = brain.stats()
    assert stats["world"]["events"] > 0,               "world was empty"
    assert stats["world"]["observations"] > 0,         "nothing was observed"
    assert stats["predict"]["scored"] > 0,             "predictions were never scored"
    assert stats["levels"]["consolidations"] > 0,      "memory never consolidated"
    assert stats["levels"]["promoted_total"] > 0,      "nothing was ever promoted"
    assert stats["discover"]["passes"] > 0,            "abstraction never ran"
    assert stats["curiosity"]["passes"] > 0,           "she never wondered"
    assert stats["readout"]["steps"] > 0,              "no gradient step was ever taken"
    assert stats["self_model"]["capabilities"]["grounding"]["observations"] > 0, "untested"


def test_the_loop_reports_whether_the_turn_changed_anything(brain: NJPBrain):
    _teach(brain)
    assert brain.loop.last is not None and brain.loop.last.learned


def test_the_loop_survives_an_organ_that_raises(brain: NJPBrain):
    """A learning loop that can break a reply is worse than one that occasionally learns nothing."""
    class _Angry:
        def __getattr__(self, _name):
            raise RuntimeError("organ down")

    brain.world = _Angry()
    brain.levels = _Angry()
    thought = brain.think("Ravi lives in Mumbai")     # must not raise
    assert isinstance(thought.loop, LoopReport)


def test_the_loop_stats_report_real_totals(brain: NJPBrain):
    _teach(brain)
    stats = brain.loop.stats()
    assert stats["closes"] == brain.turns
    assert stats["totals"]["scored"] > 0
    assert stats["accuracy"] is not None


# --- small stubs, for the one test that has to force a wrong standing answer --- #
class _StubAnswer:
    def __init__(self, text: str) -> None:
        self.text, self.confidence = text, 0.8


class _StubGrounding:
    def __init__(self, answer) -> None:
        self.answer, self.concepts, self.is_question = answer, [], True


class _Stub:
    def __init__(self, stimulus: str) -> None:
        self.stimulus = stimulus
