"""Phase 7: she changes what she is made of, and only where the change measures better.

The milestone is ``cognitive_rewires > 0`` **with measurable improvement**, and the second half is
the whole of it. A counter that went up when she *tried* something would be satisfied by a module
that rewires her at random, so every test here is really one test asked in different places:
*could this have been adopted without evidence?*

The bug it found on the way is the one worth reading. :meth:`nyxara.njp.field.RecursiveCognitiveField._expect`
observes the turn's grounded facts into the dynamics model; :meth:`nyxara.njp.integrate.LearningLoop._observe_transition`
observes a sixteen-bucket histogram of which cells fired into the *same* model. One n-gram, two
kinds of state, strictly alternating::

    [0.0, 0.0, 0.5, 0.0, 0.0, 0.5, …]
    garmi hui
    [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, …]
    pasina aaya

So every order-1 context for a fact was a histogram. Nothing here asserts which representation is
right — the held-out replay decides and the two batteries have to agree nothing broke.
"""

from __future__ import annotations

from types import SimpleNamespace

from nyxara.njp import NJPBrain
from nyxara.njp.evolution import (
    ENCODERS,
    CognitiveEvolution,
    Measurement,
    Mutation,
    Situation,
)
from nyxara.njp.metareason import MetaReasoner, ProblemKind
from nyxara.njp.selfmodel import MetaLearner

#: A session with enough distinct facts that a state sequence exists to be predicted at all.
SESSION = ["birds need water", "a sparrow is a bird", "what does a sparrow need?",
           "sparrows need water", "why do birds need water?", "a crow is a bird",
           "what does a crow need?", "crows need water", "aag lagi", "garmi hui",
           "pasina aaya", "what caused garmi?", "hello", "a robin is a bird",
           "what does a robin need?", "robins need water"]


def _lived(turns: int = 120, **config) -> NJPBrain:
    brain = NJPBrain(config=SimpleNamespace(**config) if config else None)
    for turn in range(turns):
        brain.think(SESSION[turn % len(SESSION)])
    return brain


# --------------------------------------------------------------------------- #
# the trace is raw material, not a record of one encoding
# --------------------------------------------------------------------------- #

def test_every_turn_leaves_raw_material_behind():
    brain = _lived(30)
    trace = brain.evolution.trace
    assert len(trace) == 30
    assert all(isinstance(s, Situation) for s in trace)


def test_the_trace_keeps_the_inputs_to_an_encoding_and_not_an_encoding():
    """A trace of encoded states can only re-score the encoding that produced it, which would
    make the sandbox an echo of the decision it is supposed to adjudicate."""
    brain = _lived(30)
    assert any(s.cells for s in brain.evolution.trace)
    assert any(s.facts for s in brain.evolution.trace)
    assert any(s.concepts for s in brain.evolution.trace)
    assert any(s.action for s in brain.evolution.trace)


def test_a_trace_too_short_to_hold_out_is_not_scored():
    box = CognitiveEvolution()
    for _ in range(4):
        box.observe(Situation(cells=(1, 2), facts=("a b c",), action="statement"))
    assert box.replay("facts") is None
    assert box.cycle(SimpleNamespace()).promoted is False


def test_the_trace_is_bounded():
    box = CognitiveEvolution(capacity=32)
    for i in range(200):
        box.observe(Situation(cells=(i,), action="statement"))
    assert len(box.trace) == 32


# --------------------------------------------------------------------------- #
# the measure, and what it refuses to say
# --------------------------------------------------------------------------- #

def test_the_measure_can_tell_the_representations_apart():
    """If every candidate scored the same, this whole module would be theatre with a counter."""
    brain = _lived(120)
    scores = {name: brain.evolution.replay(name) for name in ENCODERS}
    assert all(m is None or isinstance(m, Measurement) for m in scores.values())
    values = {round(m.value, 4) for m in scores.values() if m is not None}
    assert len(values) > 1, scores


def test_a_score_carries_how_many_rows_it_is_made_of():
    got = Measurement(0.5, 10)
    assert got.floor == 0.1
    assert got.required == 0.2
    assert Measurement(0.5, 0).floor == 1.0


def test_two_rows_not_one():
    """A gain of exactly one row out of fourteen is one episode landing the other way."""
    measure = Measurement(0.0714, 14)
    assert measure.floor < 0.0715          # one row is within reach of noise
    assert measure.required > 0.14         # two rows is what a rewire has to clear


def test_a_candidate_its_measure_cannot_see_is_refused_not_adopted():
    """The single most important line in `field.meta_cycle`, kept here because the change is
    structural and therefore worse to get wrong."""
    box = CognitiveEvolution(gates=False)
    box.generate = lambda brain: [Mutation(kind="representation", name="blind",
                                           apply=lambda b: True, revert=lambda b: True,
                                           score=lambda: None)]
    trial = box.cycle(SimpleNamespace())
    assert trial.promoted is False
    assert trial.unmeasurable is True
    assert box.unmeasurable == 1
    assert box.cognitive_rewires == 0


def test_a_tie_is_not_evidence_for_a_change():
    box = CognitiveEvolution(gates=False)
    applied = []
    box._baseline = lambda brain, kind: Measurement(0.8, 40)
    box.generate = lambda brain: [Mutation(kind="representation", name="same",
                                           apply=lambda b: applied.append(b) or True,
                                           revert=lambda b: True,
                                           score=lambda: Measurement(0.8, 40))]
    trial = box.cycle(SimpleNamespace())
    assert trial.promoted is False
    assert not applied, "a rejected candidate must never reach the live brain"
    assert box.cognitive_rewires == 0


def test_a_gain_inside_the_measures_resolution_is_refused():
    box = CognitiveEvolution(gates=False)
    box._baseline = lambda brain, kind: Measurement(0.5, 10)
    box.generate = lambda brain: [Mutation(kind="representation", name="thin",
                                           apply=lambda b: True, revert=lambda b: True,
                                           score=lambda: Measurement(0.6, 10))]   # one row
    assert box.cycle(SimpleNamespace()).promoted is False
    box2 = CognitiveEvolution(gates=False)
    box2._baseline = lambda brain, kind: Measurement(0.5, 10)
    box2.generate = lambda brain: [Mutation(kind="representation", name="real",
                                            apply=lambda b: True, revert=lambda b: True,
                                            score=lambda: Measurement(0.75, 10))]  # two and a half
    assert box2.cycle(SimpleNamespace()).promoted is True
    assert box2.cognitive_rewires == 1


def test_a_rejected_candidate_is_not_offered_again_from_the_same_state():
    box = CognitiveEvolution(gates=False)
    box._baseline = lambda brain, kind: Measurement(0.8, 40)
    made = Mutation(kind="representation", name="same", apply=lambda b: True,
                    revert=lambda b: True, score=lambda: Measurement(0.8, 40))
    box.generate = lambda brain: [m for m in [made] if m.signature not in box._rejected]
    box.cycle(SimpleNamespace())
    second = box.cycle(SimpleNamespace())
    assert "nothing structural to propose" in second.why


# --------------------------------------------------------------------------- #
# the gates
# --------------------------------------------------------------------------- #

def test_a_change_that_damages_the_language_surface_is_refused():
    """The batteries sit at their ceiling, which is useless as evidence of progress and exactly
    right as evidence that nothing was broken on the way to it."""
    brain = NJPBrain()
    harmful = Mutation(kind="representation", name="wreck",
                       apply=lambda b: (setattr(b, "grounder", None), True)[1],
                       revert=lambda b: True)
    adversarial, regression, note = brain.evolution._gates(brain, harmful)
    assert not (adversarial and regression)
    assert note


def test_a_harmless_change_clears_both_gates():
    brain = NJPBrain()
    benign = Mutation(kind="representation", name="noop",
                      apply=lambda b: True, revert=lambda b: True)
    adversarial, regression, note = brain.evolution._gates(brain, benign)
    assert adversarial and regression
    assert note == ""


def test_a_gate_that_cannot_run_refuses():
    """Fail-closed. A gate that errors is not a gate that passed."""
    box = CognitiveEvolution()
    exploding = Mutation(kind="representation", name="boom",
                         apply=lambda b: (_ for _ in ()).throw(RuntimeError("no")))
    adversarial, regression, note = box._gates(SimpleNamespace(), exploding)
    assert not adversarial and not regression
    # And the reason says what went wrong, because "a gate failed" is the report that gets a
    # gate quietly disabled by whoever reads it next.
    assert "RuntimeError" in note


def test_gates_off_is_not_a_way_through():
    """`gates=False` is for a caller that has already measured them. It skips the batteries and
    changes nothing about the benchmark a candidate still has to win."""
    box = CognitiveEvolution(gates=False)
    box._baseline = lambda brain, kind: Measurement(0.5, 10)
    box.generate = lambda brain: [Mutation(kind="representation", name="thin",
                                           apply=lambda b: True, revert=lambda b: True,
                                           score=lambda: Measurement(0.55, 10))]
    assert box.cycle(SimpleNamespace()).promoted is False


# --------------------------------------------------------------------------- #
# the milestone
# --------------------------------------------------------------------------- #

def test_a_structural_rewire_is_adopted_and_it_measured_better():
    """Phase 7's milestone, end to end and on real turns."""
    brain = _lived(120)
    evolution = brain.evolution
    assert evolution.cognitive_rewires > 0, evolution.stats()
    assert evolution.total_gain > 0.0
    adopted = evolution.adopted[-1]
    assert adopted["gain"] > 0.0
    assert adopted["mutation"]["kind"] in ("representation", "topology", "operator", "strategy")


def test_the_promotion_changes_what_the_loop_actually_feeds():
    """A rewire nothing acts on is a note in a report. The loop reads the promoted encoding."""
    brain = _lived(120)
    assert brain.evolution.cognitive_rewires > 0
    assert brain.loop.predictive_encoding == brain.evolution.encoding
    assert brain.loop.predictive_encoding != "buckets16"


def test_the_loop_reports_promotions_and_never_attempts():
    brain = _lived(120)
    totals = brain.loop.totals
    assert totals["evolution_trials"] >= totals["cognitive_rewires"]
    assert totals["cognitive_rewires"] == brain.evolution.cognitive_rewires


def test_the_arrow_is_open_until_something_survives_a_benchmark():
    fresh = NJPBrain().pipeline_report()
    assert fresh["weakness→rewire"]["state"] == "open"
    assert fresh["weakness→rewire"]["why"]
    assert _lived(120).pipeline_report()["weakness→rewire"]["state"] == "closed"


def test_the_organ_is_readable_from_the_brains_own_report():
    brain = _lived(30)
    block = brain.stats().get("evolution")
    assert isinstance(block, dict)
    assert block["trace"] == len(brain.evolution.trace)


def test_evolution_gated_off_is_absent_not_broken():
    brain = NJPBrain(config=SimpleNamespace(evolution_enabled=False))
    for line in SESSION[:6]:
        brain.think(line)
    assert brain.evolution is None
    assert "evolution" not in brain.stats()
    assert "weakness→rewire" not in brain.pipeline_report()
    assert brain.loop.predictive_encoding == "buckets16"


# --------------------------------------------------------------------------- #
# what the sandbox found
# --------------------------------------------------------------------------- #

def test_one_model_was_being_fed_two_kinds_of_state():
    """The defect the representation candidates exist to adjudicate, shown rather than asserted."""
    brain = NJPBrain(config=SimpleNamespace(evolution_enabled=False))
    for turn in range(24):
        brain.think(SESSION[turn % len(SESSION)])
    history = list(brain.predictive._history)
    histograms = [h for h in history if h.startswith("[")]
    facts = [h for h in history if not h.startswith("[")]
    assert histograms and facts, history


def test_the_promoted_representation_beat_the_written_one_when_it_was_adopted():
    """Judged on the trial that adopted it, not on the trace as it stands afterwards.

    The two are different questions and only the first is what the golden rule asks. She adopts
    at the turn the cycle runs, on the tail she had then; a hundred turns later the encodings can
    converge — as they do here, 0.95 against 1.00 — and re-deriving the decision from the final
    trace would be grading a choice on evidence that did not exist when it was made.
    """
    brain = _lived(120)
    promoted = [t for t in brain.evolution.trials if t.promoted]
    assert promoted, brain.evolution.stats()
    for trial in promoted:
        assert trial.gain >= max(brain.evolution.min_gain, 2.0 / max(1, trial.rows))
        assert trial.adversarial_passed and trial.regression_passed
    written = brain.evolution.replay("buckets16")
    now = brain.evolution.replay(brain.evolution.encoding)
    assert now.value >= written.value


# --------------------------------------------------------------------------- #
# the selection gates the bandit was walking through
# --------------------------------------------------------------------------- #

def test_the_bandit_may_not_return_a_strategy_the_kind_excludes():
    """Regression. `choose` computed the eligible pool and the act-permitted pool and then threw
    both away whenever the shared bandit had an opinion — the check was `name in self.strategies`,
    which every registered strategy passes."""
    learner = MetaLearner(explore=0.0)
    reasoner = MetaReasoner(meta_learner=learner)
    reasoner.register("simulate", (ProblemKind.CAUSAL,), lambda p, c: "sim")
    reasoner.register("introspect", (ProblemKind.INTROSPECTIVE,), lambda p, c: "intro")
    intruder = learner.register("strategy:introspective", "simulate", "simulate")
    intruder.trials, intruder.reward = 20, 20.0
    incumbent = learner.strategies["strategy:introspective"]["introspect"]
    incumbent.trials, incumbent.reward = 20, 2.0

    assert [s.name for s in reasoner._candidates(ProblemKind.INTROSPECTIVE)] == ["introspect"]
    assert reasoner.choose(ProblemKind.INTROSPECTIVE).name == "introspect"


def test_the_bandit_may_not_return_a_strategy_the_speech_act_forbids():
    """"A turn whose speech act permits no reasoning is a turn that gets none" — `_permitted`."""
    learner = MetaLearner(explore=0.0)
    reasoner = MetaReasoner(meta_learner=learner)
    reasoner.register("simulate", (ProblemKind.CAUSAL,), lambda p, c: "sim")
    reasoner.register("causal", (ProblemKind.CAUSAL,), lambda p, c: "cau")
    forbidden = learner.strategies["strategy:causal"]["simulate"]
    forbidden.trials, forbidden.reward = 20, 20.0
    allowed = learner.strategies["strategy:causal"]["causal"]
    allowed.trials, allowed.reward = 20, 1.0

    assert reasoner.choose(ProblemKind.CAUSAL).name == "simulate"          # unconstrained
    assert reasoner.choose(ProblemKind.CAUSAL, pathways=("reason",)).name == "causal"


def test_an_empty_permitted_pool_still_means_no_strategy():
    """The bandit fix must not turn "nothing is permitted" into "fall back to everything"."""
    learner = MetaLearner(explore=0.0)
    reasoner = MetaReasoner(meta_learner=learner)
    reasoner.register("simulate", (ProblemKind.CAUSAL,), lambda p, c: "sim")
    assert reasoner.choose(ProblemKind.CAUSAL, pathways=("relationship",)) is None


# --------------------------------------------------------------------------- #
# the sandbox has to be the same experiment as the live one
# --------------------------------------------------------------------------- #

def test_the_sandbox_reproduces_the_live_model_exactly():
    """**The guarantee the whole module rests on.** Re-fitting a brain from its own trace, in the
    representation it was already using, has to reproduce the model it actually built — same
    observations, same history, same counts. Anything less and a promotion is decided by a
    difference between the sandbox and the world rather than between two representations.

    Two fidelity bugs were found by exactly this comparison, and each one made a promotion look
    good and then measure worse live:

    * ``previous`` was carried across turns the encoding could not represent, inventing
      transitions the loop never makes — 58 observations where 48 lived turns produced 43.
    * ``field._predict_world`` labels its observation with ``brain._last_intent_kind``, which
      **nothing in the package ever assigned**, so every one of its transitions carried the empty
      action while the loop's carried a real one.
    """
    import copy

    brain = NJPBrain(config=SimpleNamespace(evolution_enabled=False))
    brain.evolution = CognitiveEvolution(gates=False)
    brain.loop.predictive_encoding = brain.evolution.encoding = "facts"
    for turn in range(48):
        brain.think(SESSION[turn % len(SESSION)])

    counts = copy.deepcopy(brain.predictive._counts)
    history = list(brain.predictive._history)
    observations = brain.predictive.observations

    assert brain.evolution.refit(brain) is True
    assert brain.predictive.observations == observations
    assert brain.predictive._history == history
    assert brain.predictive._counts == counts


def test_the_field_labels_its_observations_with_the_real_intent():
    """Regression: one read, no writes. `brain._last_intent_kind` was read by the field and
    assigned nowhere, so the two halves of one dynamics model disagreed about whether an action
    had happened at all."""
    brain = NJPBrain()
    brain.think("a sparrow is a bird")
    assert brain._last_intent_kind
    brain.think("what does a sparrow need?")
    actions = {key[2] for key in brain.predictive._counts}
    assert actions - {""}, "every transition is still unlabelled"


def test_a_turn_the_encoding_cannot_represent_breaks_the_chain():
    """The loop's ``previous`` is literally last turn. A representation that cannot express a
    turn ends the chain there rather than reaching over it."""
    box = CognitiveEvolution(gates=False, min_trace=4)
    for facts in (("a b c",), (), ("d e f",)):
        box.observe(Situation(cells=(1,), facts=facts, action="statement"))
    from nyxara.njp.predictive import PredictiveWorldModel

    model = PredictiveWorldModel()
    previous = None
    for situation in box.trace:
        current = ENCODERS["facts"](situation)
        if previous is not None and current is not None:
            model.observe(previous, situation.action, next_state=current)
        previous = current
    assert not any("a b c→d e f" in key[1] for key in model._counts), model._counts


def test_the_representation_she_adopted_predicts_better_held_constant():
    """The controlled comparison the pipeline is a proxy for: each representation held from turn
    zero, over identical turns, scored on the predictions she actually made."""
    scores = {}
    for encoding in ("buckets16", "facts"):
        brain = NJPBrain(config=SimpleNamespace(evolution_enabled=False))
        brain.loop.predictive_encoding = encoding
        for turn in range(120):
            brain.think(SESSION[turn % len(SESSION)])
        scores[encoding] = brain.predictive.accuracy
    assert scores["facts"] > scores["buckets16"], scores
