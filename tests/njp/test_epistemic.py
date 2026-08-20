"""\"I don't know\" → \"what would let me know?\" — njp/epistemic.py.

The one rule this module adds to the repo is that an observation is worth making only if the
candidates DISAGREE about it. Everything here is that rule from a different angle, because the
failure it prevents is the one that feels most like progress: running a test that every surviving
hypothesis already predicts, watching it pass, and counting that as evidence.

``eval/intelligence.py`` names the same failure for predictions — *a mind that grades its guesses
against its own later guesses is not learning, it is agreeing with itself* — and confirmation costs
exactly as much effort as inquiry does.
"""
from __future__ import annotations

import pytest

from nyxara.njp.cortex import CortexHypothesis, HypothesisKind
from nyxara.njp.curiosity import Curiosity
from nyxara.njp.epistemic import (EpistemicCompiler, Observation, ObservationKind)


def _h(claim, predicts=(), forbids=()):
    return CortexHypothesis(claim=claim, predicts=list(predicts), forbids=list(forbids))


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #
def test_an_observation_every_candidate_predicts_discriminates_nothing():
    compiler = EpistemicCompiler()
    candidates = [_h("A", predicts=["the alarm fires"]),
                  _h("B", predicts=["the alarm fires"])]
    scored = compiler.discriminate(Observation(text="the alarm fires"), candidates)
    assert scored.power == 0.0
    assert not scored.discriminating
    assert "confirm, not discriminate" in scored.why


def test_an_observation_no_candidate_mentions_discriminates_nothing():
    compiler = EpistemicCompiler()
    scored = compiler.discriminate(Observation(text="the moon is up"),
                                   [_h("A", predicts=["x"]), _h("B", forbids=["x"])])
    assert scored.power == 0.0
    assert "could not eliminate any of them" in scored.why


def test_power_is_highest_when_the_split_is_even():
    compiler = EpistemicCompiler()
    even = compiler.discriminate(
        Observation(text="x"),
        [_h("A", predicts=["x"]), _h("B", predicts=["x"]),
         _h("C", forbids=["x"]), _h("D", forbids=["x"])])
    lopsided = compiler.discriminate(
        Observation(text="x"),
        [_h("A", predicts=["x"]), _h("B", predicts=["x"]),
         _h("C", predicts=["x"]), _h("D", forbids=["x"])])
    assert even.power > lopsided.power > 0.0


def test_a_candidate_that_both_predicts_and_forbids_something_is_silent_about_it():
    compiler = EpistemicCompiler()
    scored = compiler.discriminate(
        Observation(text="x"),
        [_h("A", predicts=["x"], forbids=["x"]), _h("B", predicts=["x"]), _h("C", forbids=["x"])])
    assert scored.predicted_by == ["B"] and scored.forbidden_by == ["C"]
    assert "A" in scored.silent


def test_compile_refuses_rather_than_returning_the_least_bad_confirmation():
    compiler = EpistemicCompiler()
    assert compiler.compile("why?", [_h("A", predicts=["x"]), _h("B", predicts=["x"])]) is None
    assert compiler.stats()["refused"] == 1


def test_one_candidate_is_not_a_set():
    compiler = EpistemicCompiler()
    assert compiler.compile("why?", [_h("A", predicts=["x"], forbids=["y"])]) is None
    assert compiler.compile("why?", []) is None


def test_the_chosen_experiment_is_one_the_candidates_disagree_about():
    compiler = EpistemicCompiler()
    experiment = compiler.compile("why?", [
        _h("A", predicts=["everyone agrees on this", "only A predicts this"],
           forbids=["only A forbids this"]),
        _h("B", predicts=["everyone agrees on this", "only A forbids this"],
           forbids=["only A predicts this"]),
    ])
    assert experiment is not None
    assert experiment.observation.text != "everyone agrees on this"
    assert experiment.discrimination.discriminating


# --------------------------------------------------------------------------- #
# The standing four rivals, and the do-operator that separates them
# --------------------------------------------------------------------------- #
def test_a_bare_causal_claim_gets_its_rivals_synthesised():
    rivals = EpistemicCompiler.rivals_for("aag", "garmi")
    kinds = {r.kind for r in rivals}
    assert kinds == {HypothesisKind.CAUSE, HypothesisKind.REVERSE,
                     HypothesisKind.CONFOUND, HypothesisKind.COINCIDENCE}
    assert all(r.discriminating for r in rivals)


def test_the_experiment_for_a_bare_causal_claim_is_an_intervention():
    """Only do(X) separates causation from correlation — nothing observational can."""
    compiler = EpistemicCompiler()
    experiment = compiler.compile("does aag cause garmi?",
                                  compiler.rivals_for("aag", "garmi"))
    assert experiment is not None
    assert experiment.observation.kind == ObservationKind.INTERVENE
    assert experiment.observation.text.startswith("do(")


def test_the_question_names_what_each_result_would_rule_out():
    compiler = EpistemicCompiler()
    experiment = compiler.compile("does aag cause garmi?",
                                  compiler.rivals_for("aag", "garmi"))
    assert "If it holds, that rules out:" in experiment.question
    assert "If it does not, that rules out:" in experiment.question


# --------------------------------------------------------------------------- #
# The join the adversary never had
# --------------------------------------------------------------------------- #
class _Attack:
    def __init__(self, landed=False):
        self.landed = landed


class _Report:
    def __init__(self, cause, effect, refuted=False):
        self.cause, self.effect = cause, effect
        self.attacks = [_Attack(landed=refuted)]

    @property
    def refuted_by(self):
        return [a for a in self.attacks if a.landed]


def test_an_undecided_attack_becomes_an_experiment():
    compiler = EpistemicCompiler()
    experiment = compiler.from_attack(_Report("rain", "wet ground"))
    assert experiment is not None
    assert "rain" in experiment.observation.text


def test_an_already_refuted_claim_needs_no_experiment():
    compiler = EpistemicCompiler()
    assert compiler.from_attack(_Report("rain", "wet ground", refuted=True)) is None


def test_an_empty_attack_report_yields_nothing():
    compiler = EpistemicCompiler()
    assert compiler.from_attack(_Report("", "")) is None


# --------------------------------------------------------------------------- #
# It lands somewhere that retires it
# --------------------------------------------------------------------------- #
def test_an_experiment_becomes_a_question_curiosity_can_retire():
    curiosity = Curiosity()
    compiler = EpistemicCompiler(curiosity=curiosity)
    experiment = compiler.compile("does aag cause garmi?",
                                  compiler.rivals_for("aag", "garmi"))
    question = compiler.to_question(experiment, subject="aag", predicate="causes")
    assert question is not None
    assert question.text == experiment.question
    assert question.action in ("act", "ask", "gather")


def test_with_no_curiosity_attached_nothing_raises():
    compiler = EpistemicCompiler()
    experiment = compiler.compile("does aag cause garmi?",
                                  compiler.rivals_for("aag", "garmi"))
    assert compiler.to_question(experiment) is None


def test_pricing_degrades_honestly_when_no_voi_engine_is_reachable():
    class _NoVOI(EpistemicCompiler):
        def _get_voi(self):
            return None

    compiler = _NoVOI()
    experiment = compiler.compile("does aag cause garmi?",
                                  compiler.rivals_for("aag", "garmi"))
    assert experiment is not None
    assert "priced on discriminating power alone" in experiment.rationale


def test_asking_the_master_is_the_dearest_way_to_find_out():
    """A mind that asks about everything has simply moved its work onto him."""
    costs = {kind: Observation(text="x", kind=kind).cost for kind in ObservationKind.ALL}
    assert costs[ObservationKind.ASK] > costs[ObservationKind.INTERVENE]
    assert costs[ObservationKind.INTERVENE] > costs[ObservationKind.OBSERVE]


def test_stats_count_refusals_as_well_as_experiments():
    compiler = EpistemicCompiler()
    compiler.compile("a", [_h("A", predicts=["x"]), _h("B", predicts=["x"])])
    compiler.compile("b", compiler.rivals_for("aag", "garmi"))
    stats = compiler.stats()
    assert stats["compiled"] == 2 and stats["experiments"] == 1 and stats["refused"] == 1
    assert stats["refusal_rate"] == pytest.approx(0.5)
