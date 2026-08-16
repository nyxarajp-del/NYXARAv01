"""Does the loop actually close?

The tests that matter here are the ones a system with all the right organs and none of the wiring
would fail. Every organ these exercise was already written, tested and reachable *in isolation*
before this file existed — what was missing was the caller and the return edge. So none of these
assert that a mechanism exists; they assert that asking a question reaches it, and that what
happened afterwards got back to it.

The numbers are the point. Each assertion is on a counter that was measured at exactly zero over a
real multi-turn session while every underlying algorithm worked perfectly when called by hand.
"""

from __future__ import annotations

import pytest

from nyxara.njp import NJPBrain
from nyxara.njp.grounding import _measurements

# The session every test here teaches from: one entity, two variables, an exact 2× relation and
# no noise. Exactness is deliberate — a fitted slope of 2.0 is a claim that can be checked to
# 1e-6, where "roughly correlated" would pass on a coincidence.
# Ordered so the last reading is the round one: an intervention is resolved against what was
# most recently observed, so the tail of this tuple is what "halve the water" halves.
_TAUGHT = (
    "the plant got 2 litres of water and grew 4 cm",
    "the plant got 5 litres of water and grew 10 cm",
    "the plant got 8 litres of water and grew 16 cm",
    "the plant got 3 litres of water and grew 6 cm",
    "the plant got 10 litres of water and grew 20 cm",
)


def _taught_brain() -> NJPBrain:
    brain = NJPBrain()
    for line in _TAUGHT:
        brain.think(line)
    return brain


# --------------------------------------------------------------------------- #
# Ingestion — the causal engine has to be fed before anything can reach it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sentence,expected", [
    ("the plant got 10 litres of water and grew 20 cm", [("water", 10.0), ("growth", 20.0)]),
    ("the plant got 5 litres of water and grew 10 cm", [("water", 5.0), ("growth", 10.0)]),
])
def test_a_measurement_sentence_yields_every_quantity_not_just_the_first(sentence, expected):
    # The whole point of the extractor: two variables observed together are what a coefficient is
    # fitted to, and reading one of the two destroys the pairing rather than halving it.
    assert _measurements(sentence) == expected


def test_a_standing_law_is_not_read_as_a_measurement():
    # "water boils at 100" is a claim about water in general, already read by the threshold
    # pattern. Treating it as an observation of some particular thing would put a law on the
    # simulator's evidence pile.
    brain = NJPBrain()
    triples = brain.think("water boils at 100 degrees").percept.grounding.triples
    assert not any(t.source == "measurement" for t in triples), [t.to_dict() for t in triples]


def test_an_unnameable_quantity_is_dropped_rather_than_stored_under_its_unit():
    # A bare count with no noun, no governing quantity-verb and no unit names no variable. Storing
    # it anyway is how a simulator ends up fitting arrows between things that were never measured.
    assert _measurements("the plant has 10 leaves") == []


def test_numeric_statements_reach_the_causal_engine():
    # Measured before this wiring existed: universe.state was {} and usable_relations() was []
    # after exactly this session, so what_if answered at confidence 0.0 for want of any data.
    brain = _taught_brain()
    relations = brain.universe.usable_relations()
    assert relations, brain.universe.stats()

    arrow = next((r for r in relations
                  if r.cause == "plant.water" and r.effect == "plant.growth"), None)
    assert arrow is not None, [(r.cause, r.effect) for r in relations]
    assert abs(arrow.slope - 2.0) < 1e-6, arrow.slope
    assert arrow.r2 > 0.99, arrow.r2


def test_the_do_operator_answers_from_what_a_conversation_taught():
    # The engine could always do this when handed numbers directly. The claim under test is that
    # a conversation is now a way of handing it numbers.
    brain = _taught_brain()
    out = brain.universe.what_if("plant.water", 1.0)
    assert abs(out.predicted["plant.growth"] - 2.0) < 1e-6, out.predicted
    assert out.confidence > 0.5, out.confidence
    assert not out.reason, out.reason


def test_a_counterfactual_far_outside_the_observed_range_loses_confidence():
    # The honesty property, and the one a plausible-looking simulator fails: reaching further than
    # the data has to cost something, or every extrapolation is stated as firmly as an observation.
    brain = _taught_brain()
    near = brain.universe.what_if("plant.water", 4.0)
    far = brain.universe.what_if("plant.water", 500.0)
    assert far.confidence < near.confidence, (near.confidence, far.confidence)


# --------------------------------------------------------------------------- #
# Routing — the question has to reach the organ that can answer it
# --------------------------------------------------------------------------- #
def test_a_counterfactual_reads_as_a_causal_query_not_a_lookup():
    # `CAUSAL_QUERY` is the only act whose pathways include SIMULATE, and nothing used to produce
    # it for "what if": the `why` cue does not match the phrase and the `what is` cue requires a
    # copula. So the one act that could reach the do-operator was unreachable by construction.
    from nyxara.njp.relevance import SpeechAct, SpeechActReader

    reader = SpeechActReader()
    for question in ("what if I halve the water", "what happens if I remove the water",
                     "agar main paani aadha kar doon"):
        assert reader.read(question).kind == SpeechAct.CAUSAL_QUERY, question


def test_an_intervention_resolves_to_the_variable_the_universe_actually_named():
    # The universe names variables `entity.attribute`; a question says "the water". Reconciling
    # them against observed variables rather than by guessing an entity is what stops her
    # simulating confidently over a subject the Master never mentioned.
    brain = _taught_brain()
    assert brain._counterfactual_context("what if I halve the water") == {
        "variable": "plant.water", "value": 5.0,          # half of the 10 last observed
    }
    assert brain._counterfactual_context("what if I halve the sunlight") == {}


def test_an_intervention_whose_size_is_not_stated_is_declined():
    # "reduce the water" names no magnitude. A factor she invented would come back through the
    # do-operator carrying the same confidence as one the Master gave, and nothing downstream
    # could tell them apart afterwards.
    brain = _taught_brain()
    assert brain._counterfactual_context("what if I reduce the water") == {}


def test_what_if_halving_the_water_gets_a_real_number():
    # The whole slice, end to end. Measured before it: the empty string, from the
    # `if is_question: return ""` in _compose, with a working do-operator two calls away.
    brain = _taught_brain()
    answer = brain.think("what if I halve the water").answer
    assert answer, "a counterfactual over an observed variable answered with nothing"
    assert "plant.growth" in answer, answer
    assert "10" in answer, answer                         # 20 → 10, the halved growth
    assert "confidence" in answer.lower(), answer


def test_a_counterfactual_that_changes_nothing_is_not_an_answer():
    # Setting a variable to the value it already holds entails no consequence, and reporting the
    # premise back as though it were a prediction is the counterfactual form of an echo.
    brain = _taught_brain()
    assert brain.think("what if the water were 10").answer == ""


def test_the_answer_states_the_confidence_the_do_operator_earned():
    # Not that a number is present, but that it *moves the right way*: a question inside the
    # observed range must come back more confident than one reaching well past it. This is the
    # assertion a template with a hard-coded confidence would fail.
    brain = _taught_brain()
    inside = brain.think("what if the water were 4").answer          # within the 2..10 observed
    outside = brain.think("what if the water were 90").answer        # far beyond it
    assert inside and outside, (inside, outside)

    def _confidence(text: str) -> float:
        return float(text.rsplit("confidence", 1)[1].strip(" )"))

    assert _confidence(inside) > _confidence(outside), (inside, outside)


def test_a_counterfactual_over_a_variable_never_observed_is_declined():
    # The honest failure. She has never measured sunlight, so there is no arrow to push on, and
    # inventing one is the single worst thing a simulator can do.
    brain = _taught_brain()
    assert brain.think("what if I halve the sunlight").answer == ""


def test_a_strategy_that_produces_nothing_yields_to_one_that_can():
    # `causal` (explanation) and `simulate` (intervention) are both eligible for a causal problem,
    # and `causal` has the higher prior. It produces nothing on an intervention — explanation is
    # not intervention — and abstaining there meant the organ that owns the question was never
    # asked. The retry is what makes the registry's second-best reachable.
    brain = _taught_brain()
    context = dict(brain._counterfactual_context("what if I halve the water"),
                   grounded=False, subject="", about_self=False)
    solution = brain.metareason.solve("what if I halve the water", context=context)
    assert solution.answered, solution.to_dict()
    assert solution.strategy == "simulate", solution.to_dict()
    assert "causal" in solution.attempts, solution.attempts


def test_an_ordinary_empirical_question_is_not_dragged_into_the_causal_path():
    # The regression guard for the classifier change. The causal boost is scored off the parsed
    # intervention, not off another keyword, so a question that names no intervention she can
    # perform is classified exactly as it was before.
    from nyxara.njp.metareason import ProblemKind, ProblemClassifier

    classifier = ProblemClassifier()
    plain = classifier.classify("what is the boiling point of mercury", context={"grounded": False})
    assert plain.kind == ProblemKind.EMPIRICAL, plain.to_dict()


# --------------------------------------------------------------------------- #
# The return edge — an outcome has to reach what staked a claim on it
# --------------------------------------------------------------------------- #
def _graded_brain(names=("Ravi", "Sita", "Amit", "Neha")) -> NJPBrain:
    """Ask before teaching, so the answer is deliberated and a later fact can grade it.

    The order is the whole point: she answers on one turn and the Master states the fact on a
    later one, so what does the grading is independent of what is being graded.
    """
    brain = NJPBrain()
    for name in names:
        brain.think(f"where does {name} live")
        brain.think(f"{name} lives in Delhi")
    return brain


def test_a_strategy_is_graded_by_reality_and_not_only_by_its_own_critic():
    # MetaReasoner.outcome exists to override the critic's provisional credit — its docstring
    # says so — and nothing had ever called it, so strategy selection was trained entirely on
    # the opinion of the critic that had just approved the answer.
    brain = _graded_brain()
    assert brain.loop.totals["strategies_graded"] > 0, brain.loop.totals


def test_an_answer_she_asserts_is_staked_as_a_belief_that_can_be_found_wrong():
    # The ledger only ever held the Master's testimony, so the one class of claim that could be
    # checked against an independent later fact was the class never entered.
    brain = NJPBrain()
    brain.think("Ravi lives in Pune")
    brain.think("where does Ravi live")
    held = [b for b in brain.beliefs.known() if "pune" in b.claim.lower()]
    assert held, [b.claim for b in brain.beliefs.known()]
    assert held[0].falsifier, held[0].to_dict()


def test_reliability_becomes_measurable_so_temper_stops_being_a_no_op():
    # The sharpest cascade in the audit: settle/retract were never called, so `_outcomes` stayed
    # empty, `reliability()` always reported zero samples, and `temper()` returned its input
    # unchanged for ever. An entire calibration path, written and tested and inert.
    brain = NJPBrain()
    for name in ("Ravi", "Sita", "Amit", "Neha", "Vikram", "Priya", "Arjun"):
        brain.think(f"{name} lives in Pune")
        brain.think(f"where does {name} live")
        brain.think(f"{name} lives in Delhi")       # the Master says otherwise

    reliability = brain.beliefs.reliability("located_in")
    assert reliability.samples >= 5, reliability.to_dict()
    assert brain.beliefs.temper(0.9, "located_in") < 0.9, reliability.to_dict()


def test_a_belief_is_retracted_rather_than_deleted():
    # Quarantine, not deletion. Driving a belief to zero and keeping it is what lets "this has
    # failed before" stay answerable; dropping the record makes the same mistake available again.
    brain = NJPBrain()
    brain.beliefs.hold("the sky is green", confidence=0.8, domain="colour",
                       falsifier="the sky is observed to be another colour")
    assert brain.beliefs.retract("the sky is green", why="contradicted by observation")
    assert brain.beliefs.stats()["retracted"] == 1

    # The tombstone: still answerable, still carrying its history, at zero confidence.
    tombstone = brain.beliefs.why("the sky is green")
    assert tombstone, "a retracted belief was deleted rather than quarantined"
    assert tombstone["confidence"] == 0.0, tombstone
    assert tombstone["history"], tombstone


# --------------------------------------------------------------------------- #
# Curiosity, and the experiment she designed but never ran
# --------------------------------------------------------------------------- #
def test_a_question_she_raised_is_closed_by_a_fact_that_answers_it():
    # `Curiosity.resolve` had no caller, so `resolved` never left zero and a question she raised
    # could only age out. Closed off any grounded fact, not off the deferral path: a deferral
    # exists where *she* was asked something, while these are questions she asked herself, and
    # the Master mentioning the answer in passing is what answers them.
    brain = NJPBrain()
    for line in ("Ravi lives in Pune", "Ravi works at Infosys",
                 "Sara lives in Delhi", "Sara works at TCS",
                 "Amit lives in Mumbai", "Neha lives in Chennai", "Neha works at Wipro"):
        brain.think(line)

    raised = brain.curiosity.wonder()
    assert raised, "no gap found where one entity lacks a property its peers have"
    assert any(q.subject == "amit" for q in raised), [q.text for q in raised]

    brain.think("Amit works at Zoho")
    assert brain.curiosity.stats()["resolved"] >= 1, brain.curiosity.stats()
    assert brain.loop.totals["questions_closed"] >= 1, brain.loop.totals


def test_a_past_tense_report_reaches_the_record():
    # The record is written from event predicates, and the past-tense forms the Master actually
    # uses extracted nothing at all — so "the plant grew" left no trace, `_counts` stayed empty,
    # and anything that reads the record could never clear its support floor.
    brain = NJPBrain()
    brain.think("the plant grew")
    assert brain.world._counts.get("grows"), brain.world._counts


def test_an_experiment_that_is_designed_is_eventually_run():
    # Measured before this: 8 experiments designed over 8 turns, 0 run, bits_gained pinned at
    # 0.0, because `observe_result` had no caller anywhere in the package. Curiosity that
    # computes the informative experiment and never performs it is a report about curiosity.
    brain = NJPBrain()
    for line in ["water causes growth"] + ["the plant grew", "the sun rose"] * 7:
        brain.think(line)

    stats = brain.designer.stats()
    assert stats["run"] > 0, stats
    assert stats["bits_gained"] > 0.0, stats
    assert brain.loop.totals["experiments_run"] > 0, brain.loop.totals


def test_the_record_outranks_the_testimony_that_raised_the_hypothesis():
    # She was *told* water causes growth, then watched growth happen seven times with no water
    # event preceding any of them. The observation settles it against the testimony, and the
    # losing hypothesis goes to exactly zero rather than being nudged down.
    brain = NJPBrain()
    for line in ["water causes growth"] + ["the plant grew", "the sun rose"] * 7:
        brain.think(line)

    probabilities = {n: h.probability for n, h in brain.designer.hypotheses.items()}
    assert probabilities.get("water→growth") == 0.0, probabilities


def test_an_experiment_is_not_settled_on_a_record_too_thin_to_call():
    # The honest non-answer. One sighting cannot decide what an effect depends on, and settling
    # a hypothesis against the wrong evidence is worse than leaving it open.
    brain = NJPBrain()
    for line in ("water causes growth", "the plant grew"):
        brain.think(line)
    assert brain.designer.stats()["run"] == 0, brain.designer.stats()


# --------------------------------------------------------------------------- #
# The field has to be a field — rivals from more than one voice
# --------------------------------------------------------------------------- #
def _flooded_brain() -> NJPBrain:
    """Two candidate causes for one effect, each seen about as often as the other."""
    brain = NJPBrain()
    for line in ["the rain fell", "the flood happened"] * 4 + \
                ["the river rose", "the flood happened"] * 3:
        brain.think(line)
    return brain


def test_the_causal_record_contributes_rivals_not_just_memory():
    # Every hypothesis used to come from one place — whatever memory recalled — so a "debate"
    # was three recollections of one conversation. The record holds genuinely different
    # candidates for why something happened, and none of them ever reached the pool.
    brain = _flooded_brain()
    conclusion = brain.reasoner.reason("why did the flood happen", uncertainty=0.8, stakes=0.8)
    sources = {h.source for h in conclusion.hypotheses}
    assert "world" in sources, [(h.source, h.claim) for h in conclusion.hypotheses]


def test_two_candidate_causes_leave_her_undecided_rather_than_picking_one():
    # The winner-is-not-truth discipline, at the point it actually bites. Rain and river are
    # equally supported; returning the first would be inventing a decision she has not made.
    brain = _flooded_brain()
    conclusion = brain.reasoner.reason("why did the flood happen", uncertainty=0.8, stakes=0.8)
    assert not conclusion.decided, conclusion.to_dict()
    assert conclusion.margin < 0.15, conclusion.to_dict()


def test_uncertainty_is_reported_in_bits_over_the_whole_field():
    # The margin cannot tell one strong reading against six weak ones from one strong reading
    # against one nearly as strong. Those are different states and this is the number that
    # separates them.
    brain = _flooded_brain()
    torn = brain.reasoner.reason("why did the flood happen", uncertainty=0.8, stakes=0.8)
    assert torn.uncertainty_bits > 1.0, torn.to_dict()

    # A pool with nothing live in it is not uncertain, it is empty — and reports 0.0, not a
    # number that would read as confidence.
    empty = brain.reasoner.reason("why did the quasar collapse", uncertainty=0.8, stakes=0.8)
    assert empty.uncertainty_bits == 0.0, empty.to_dict()


def test_a_correlate_is_offered_as_a_correlate_and_not_as_a_cause():
    # `world.why` already separates causes from correlates and nothing downstream read the
    # distinction. A correlate enters the pool at a discount and says what it is.
    brain = _flooded_brain()
    conclusion = brain.reasoner.reason("why did the flood happen", uncertainty=0.8, stakes=0.8)
    correlates = [h for h in conclusion.hypotheses if "preceded" in h.claim]
    for hypothesis in correlates:
        assert any("correlate" in e for e in hypothesis.evidence), hypothesis.to_dict()


def test_the_ladder_strategy_honours_the_ladder_s_own_refusal_to_commit():
    # One margin rule, one object, two callers — and only one of them enforced it. An undecided
    # conclusion could enter the meta-reasoner as a strategy result while the direct fallback
    # would have thrown the same conclusion away.
    brain = _flooded_brain()
    assert brain._strategy_ladder("why did the flood happen", {}) is None


# --------------------------------------------------------------------------- #
# The shadow model — ignorance as structure rather than as a sentence
# --------------------------------------------------------------------------- #
def _puzzled_brain() -> NJPBrain:
    brain = NJPBrain()
    for line in ("Ravi lives in Pune", "Ravi works at Infosys",
                 "Sara lives in Delhi", "Sara works at TCS", "Amit lives in Mumbai",
                 "what is a tachyon", "tell me about the tachyon", "what does a tachyon do"):
        brain.think(line)
    brain.curiosity.wonder()
    return brain


def test_a_word_she_keeps_meeting_and_cannot_place_becomes_a_gap():
    # The grounder computed this every turn since it was written; the only consumer rendered it
    # as a sentence on turns where she had nothing else to say, so on every turn she *did*
    # answer it was built and dropped.
    brain = _puzzled_brain()
    gaps = brain.shadow()["by_gap"]
    assert gaps.get("ungrounded_word", 0) >= 1, brain.shadow()


def test_the_relation_in_a_sentence_that_grounded_is_not_a_gap():
    # "Ravi lives in Pune" leaves "lives" unmatched as an entity because it is the relation, not
    # a thing she is missing. Counting it would fill the list with "what is lives?".
    brain = _puzzled_brain()
    words = {w for w, _n in brain.grounder.persistent_unknowns(min_times=1, limit=32)}
    assert "tachyon" in words, words
    assert "lives" not in words and "works" not in words, words


def test_a_belief_held_past_its_evidence_becomes_something_to_investigate():
    # `audit` is the richest ignorance structure in the package and it was 100% write-only — its
    # sole caller was its own stats(), which threw the rows away and kept the count.
    brain = _puzzled_brain()
    shadow = brain.shadow()
    assert shadow["by_gap"].get("overconfident", 0) >= 1, shadow["by_gap"]

    flagged = [g for g in shadow["gaps"] if g["gap"] == "overconfident"]
    # It has to be worth acting on. A belief resting entirely on testimony is uncertain to the
    # full height of its confidence, not to the difference of two numbers that happen to agree.
    assert any(g["value"] > 0.0 for g in flagged), flagged


def test_ignorance_about_the_world_and_about_herself_stay_separate():
    # Different repairs. "I have no node for tachyon" is something to go and find out; "I am bad
    # at recall" is not, and putting it in a queue of things to look up would be a category error.
    brain = _puzzled_brain()
    shadow = brain.shadow()
    assert "gaps" in shadow and "faculties" in shadow, shadow.keys()
    assert all(g["gap"] != "weak_faculty" for g in shadow["gaps"]), shadow["gaps"]
    # `certainty` — how much she knows about how good she is — had no reader anywhere.
    for record in shadow["faculties"]["measured"].values():
        assert "certainty" in record, record


def test_the_shadow_is_ranked_by_what_answering_it_would_buy():
    # Curiosity has already priced every question by expected information gain against cost.
    # Re-ranking here would be a second opinion with no new information behind it.
    brain = _puzzled_brain()
    values = [g["value"] for g in brain.shadow()["gaps"]]
    assert values == sorted(values, reverse=True), values


# --------------------------------------------------------------------------- #
# Homeostasis — a knob that is turned has to be scored
# --------------------------------------------------------------------------- #
def _graded_turns(n: int = 30) -> NJPBrain:
    brain = NJPBrain()
    for i in range(n):
        thought = brain.think(f"Ravi{i} lives in Pune")
        brain.resolve(thought, correct=1.0 if i % 3 else 0.0)
    return brain


def test_every_knob_that_is_spent_is_also_graded():
    # `_build_meta` registers three arms over real settings and says each "changes what a turn
    # costs". Only `settle_steps` was ever rewarded, so the other two accumulated choices and
    # never a single outcome — their means sat at the prior for ever and `best()` could never
    # name a winner. A knob that is turned and never scored is being varied, not learned about.
    brain = _graded_turns()
    for arm in ("settle_steps", "recall_k", "reason_depth"):
        assert brain.meta.best(arm) is not None, (arm, brain.meta.stats())
        trials = sum(s.trials for s in brain.meta.strategies[arm].values())
        assert trials > 0, (arm, trials)


def test_a_turn_spends_what_was_chosen_rather_than_a_constant():
    # The loop had no middle: choose() existed, reward() existed, and nothing read a choice back,
    # so every turn spent the same fixed config value and the bandit learned about a knob that
    # was never turned.
    brain = _graded_turns()
    spend = brain.budget()
    assert spend["learned"] is True, spend
    assert spend["settle_steps"] in (8, 24, 64), spend
    assert spend["recall_k"] in (3, 8), spend
    assert spend["reason_depth"] in ("association", "verification"), spend


def test_an_unregistered_knob_keeps_the_value_she_was_built_with():
    # The honest default. An untuned knob is the one she shipped with, not a guess.
    class _NoMeta:
        meta_enabled = False

    brain = NJPBrain(_NoMeta())
    assert brain.meta is None
    spend = brain.budget()
    assert spend["learned"] is False, spend
    assert spend["settle_steps"] == 24 and spend["recall_k"] == 5, spend


def test_the_depth_budget_caps_without_changing_what_the_question_needs():
    # A budget says what she may afford, not what this question requires. Conflating the two
    # would make an easy question expensive merely because she has been doing well.
    brain = NJPBrain()
    easy = brain.reasoner.depth_for(uncertainty=0.05, stakes=0.05, max_rung="verification")
    hard = brain.reasoner.depth_for(uncertainty=1.0, stakes=1.0, max_rung="verification")
    assert easy == "intuition", easy            # a generous budget does not inflate it
    assert hard == "verification", hard         # a tight budget does cap it

    from nyxara.njp.reason import Rung
    unbounded = brain.reasoner.depth_for(uncertainty=1.0, stakes=1.0)
    assert Rung.cost(unbounded) > Rung.cost(hard), (unbounded, hard)


# --------------------------------------------------------------------------- #
# The two stubs — a step that named a comparison, and a question nobody put
# --------------------------------------------------------------------------- #
def _rainy_model():
    from nyxara.njp.predictive import PredictiveWorldModel, WorldState

    model = PredictiveWorldModel()
    for _ in range(6):
        model.observe(WorldState.of(["rain"]))
        model.observe(WorldState.of(["flood"]))
        model.observe(WorldState.of(["rain"]))
        model.observe(WorldState.of(["drain"]))
    return model


class _Contested:
    """A ledger holding exactly one contested claim, about floods."""

    @staticmethod
    def known():
        return []

    @staticmethod
    def unknown():
        return [{"kind": "contested", "subject": "the flood claim"}]


def test_an_explanation_landing_on_a_contested_claim_is_discounted():
    # The step existed as a label and did nothing, so "which predicts best" in the trace named a
    # comparison that never happened: _conclude took the argmax of the model's raw prior and
    # everything _what_do_i_know gathered sat unused.
    from nyxara.njp.predictive import ThoughtState

    model = _rainy_model()
    plain = ThoughtState(context="what next").deliberate(model)
    contested = ThoughtState(context="what next").deliberate(model, beliefs=_Contested())

    def _p(state, outcome):
        return next(h.probability for h in state.hypotheses if h.predicts == outcome)

    assert _p(contested, "flood") < _p(plain, "flood"), (plain.to_dict(), contested.to_dict())
    # Discounted, not eliminated: contested means unsettled, not refuted, and treating the two
    # the same would be overclaiming in the other direction.
    assert _p(contested, "flood") > 0.0, contested.to_dict()


def test_the_reweighted_explanations_are_still_a_distribution():
    # _conclude's margin test compares two probabilities. Handing it a bag of scaled numbers
    # would silently change what the margin means.
    from nyxara.njp.predictive import ThoughtState

    state = ThoughtState(context="what next").deliberate(_rainy_model(), beliefs=_Contested())
    assert abs(sum(h.probability for h in state.hypotheses) - 1.0) < 1e-9, state.to_dict()
    ranked = [h.probability for h in state.hypotheses]
    assert ranked == sorted(ranked, reverse=True), ranked


def test_an_unrelated_contested_claim_changes_nothing():
    # The guard against the discount becoming a general confidence-lowering device: only an
    # explanation that actually lands on the contested claim may lose standing.
    from nyxara.njp.predictive import ThoughtState

    class _Elsewhere:
        @staticmethod
        def known():
            return []

        @staticmethod
        def unknown():
            return [{"kind": "contested", "subject": "the exchange rate"}]

    model = _rainy_model()
    plain = ThoughtState(context="what next").deliberate(model)
    other = ThoughtState(context="what next").deliberate(model, beliefs=_Elsewhere())
    assert [round(h.probability, 9) for h in plain.hypotheses] == \
           [round(h.probability, 9) for h in other.hypotheses]


def test_a_question_she_cannot_answer_is_put_to_the_master():
    # `Curiosity.ask` had no caller, so `asked` never incremented, so `stale` was permanently
    # False and the escalation behind it was unreachable. A queue of questions nobody ever puts
    # is not curiosity, it is a list.
    brain = _puzzled_brain()
    thought = brain.think("what does a muon weigh")
    assert thought.asked is not None, brain.shadow()
    assert thought.asked.asked == 1, thought.asked.to_dict()


def test_the_same_question_is_not_put_a_fourth_time():
    # Three attempts, then it goes quiet and she moves on to the next gap — `top` already
    # excluded stale questions, and nothing had ever been able to make one stale.
    brain = _puzzled_brain()
    first = brain.think("what does a muon weigh").asked
    for _ in range(2):
        brain.think("what does a muon weigh")
    assert first.stale, first.to_dict()

    later = brain.think("what does a muon weigh").asked
    assert later is None or later.key() != first.key(), later.to_dict()


def test_what_went_stale_is_reported_rather_than_going_quiet():
    # The escalation this makes reachable. "Asked repeatedly and never answered" reads as "no
    # longer curious" when nothing surfaces it, and it is the one state that wants a different
    # response rather than more patience.
    brain = _puzzled_brain()
    for _ in range(3):
        brain.think("what does a muon weigh")
    stale = brain.shadow()["stale"]
    assert stale, brain.shadow()
    assert all(q["asked"] >= 3 for q in stale), stale


def test_a_greeting_is_not_answered_with_a_question_of_her_own():
    # The relevance discipline, pointed outward. A greeting must not come back with
    # "what is a tachyon?" any more than it may come back with a pendulum law.
    brain = _puzzled_brain()
    assert brain.think("hello NYXARA").asked is None


def test_something_she_should_investigate_herself_is_not_counted_as_asked():
    # `ask` marks whatever is top. A gap curiosity priced as "go and find this out" is not one
    # she has put to him, and counting it would drive a question she never voiced to stale.
    brain = _puzzled_brain()
    for question in brain.curiosity.open_questions():
        if question.action != "ask":
            assert question.asked == 0, question.to_dict()


def test_a_greeting_still_cannot_reach_physics():
    # The failure `relevance.py` was written for, re-checked after making a new pathway live: a
    # greeting must not acquire a route to the world model just because one now exists.
    brain = _taught_brain()
    answer = brain.think("hello NYXARA").answer
    assert "plant" not in answer.lower(), answer
    assert "confidence" not in answer.lower(), answer
