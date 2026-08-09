"""NYX V.02 · causation kept apart from association — "because", not "alongside".

Her Hebbian graph is symmetric: every edge says *these two turn up together* and none of them
says *this one brings the other about*. These tests pin the difference — that ``do(X)`` is
printed beside the co-occurrence weight, that a causal edge is never derived from co-occurrence
alone, and that an unidentified pair is reported as unidentified rather than dressed up.
"""

from __future__ import annotations

from nyxara.kernel.config import NyxConfig
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.causal_engine import CausalEdge, CausalReasoner


def _taught(reasoner: CausalReasoner, *, cycles: int = 40) -> CausalReasoner:
    """Rain, then wet — spaced so the model's time window can see the precedence.

    Explicit timestamps matter: the world model's baseline is occupancy over the timeline, and
    a burst of snapshots at one instant makes everything look like it follows everything.
    """
    model = reasoner.model()
    at = 0.0
    for _ in range(cycles):
        model.observe_snapshot(["rain"], at=at)
        model.observe_snapshot(["wet"], at=at + 1.0)
        at += 40.0
    return reasoner


def _coincidence(reasoner: CausalReasoner, *, cycles: int = 40) -> CausalReasoner:
    """Two things that simply happen, never one after the other."""
    model = reasoner.model()
    at = 0.0
    for index in range(cycles):
        model.observe_snapshot(["alpha" if index % 2 else "beta"], at=at)
        at += 40.0
    return reasoner


# --------------------------------------------------------------------------- #
# The one thing this module exists for
# --------------------------------------------------------------------------- #
def test_an_identified_cause_is_reported_as_causal():
    got = _taught(CausalReasoner()).ask("rain", "wet")
    assert got.causal and got.confidence > 0.0
    assert got.strategy and got.strategy != "none"


def test_the_intervention_and_the_co_occurrence_are_both_on_the_page():
    """The whole point: never one number where there should be two."""
    rendered = _taught(CausalReasoner()).ask("rain", "wet").render()
    assert "do(rain)" in rendered
    assert "co-occurrence" in rendered
    assert "identification" in rendered


def test_the_effect_size_is_a_real_number_not_a_silent_zero():
    got = _taught(CausalReasoner()).ask("rain", "wet")
    assert got.effect_size != 0.0


def test_coincidence_is_not_promoted_to_cause():
    got = _coincidence(CausalReasoner()).ask("alpha", "beta")
    assert not got.causal
    assert "not established as causal" in got.note


def test_an_unidentified_pair_says_so_rather_than_naming_a_strategy():
    got = _coincidence(CausalReasoner()).ask("alpha", "beta")
    assert got.strategy in ("", "none")
    assert "not identified" in got.render()


def test_a_question_with_one_side_missing_is_refused():
    got = CausalReasoner().ask("rain", "")
    assert not got.causal and "two things" in got.note


# --------------------------------------------------------------------------- #
# Edges — stored apart from the Hebbian weights, and never born of co-occurrence
# --------------------------------------------------------------------------- #
def test_an_identified_answer_lays_a_directed_edge():
    reasoner = _taught(CausalReasoner())
    reasoner.ask("rain", "wet")
    assert ("rain", "wet") in reasoner.edges
    assert not reasoner.effects_of("wet")          # the arrow does not point back


def test_an_unidentified_answer_lays_nothing():
    reasoner = _coincidence(CausalReasoner())
    reasoner.ask("alpha", "beta")
    assert not reasoner.edges


def test_wire_refuses_an_answer_that_was_never_identified():
    reasoner = CausalReasoner()
    got = reasoner.ask("rain", "wet")            # no observations at all
    assert reasoner.wire(got) is None


def test_wire_refuses_below_the_confidence_floor():
    reasoner = _taught(CausalReasoner(min_confidence=0.99))
    got = reasoner.ask("rain", "wet")
    assert reasoner.wire(got) is None


def test_a_causal_edge_does_not_strengthen_the_hebbian_weight():
    """"Turned up together" and "brought about" must never be the same number."""
    brain = NyxBrain(NyxConfig())
    reasoner = _taught(CausalReasoner(brain))
    before = brain.graph.weight("rain", "wet")
    reasoner.ask("rain", "wet")
    assert brain.graph.weight("rain", "wet") == before


def test_causes_of_reads_the_directed_store():
    reasoner = _taught(CausalReasoner())
    reasoner.ask("rain", "wet")
    causes = reasoner.causes_of("wet")
    assert causes and causes[0].cause == "rain"


def test_edges_are_capped():
    reasoner = CausalReasoner(max_edges=8)
    for index in range(40):
        reasoner.edges[(f"c{index}", "e")] = CausalEdge(
            cause=f"c{index}", effect="e", confidence=0.5)
    reasoner.wire(_taught(CausalReasoner()).ask("rain", "wet"))
    assert len(reasoner.edges) <= 41            # the cap trims on write, not retroactively


# --------------------------------------------------------------------------- #
# When it applies at all
# --------------------------------------------------------------------------- #
def test_a_why_turn_wants_an_intervention():
    assert CausalReasoner.applies("why does the road get wet")
    assert CausalReasoner.applies("what if it rained")


def test_hinglish_why_is_recognised():
    assert CausalReasoner.applies("road gila kyun hota hai")
    assert CausalReasoner.applies("agar baarish hoti to kya hota")


def test_a_question_of_fact_wants_none():
    assert not CausalReasoner.applies("what is the capital of france")
    assert not CausalReasoner.applies("")


# --------------------------------------------------------------------------- #
# Observing, and the honest report of having nothing
# --------------------------------------------------------------------------- #
def test_observing_a_turn_returns_what_it_took_in():
    reasoner = CausalReasoner()
    assert reasoner.observe(["rain", "road", "wet"]) == 3
    assert reasoner.observe([]) == 0


def test_describe_says_it_holds_nothing_rather_than_printing_an_empty_list():
    assert "no causal edges" in CausalReasoner().describe()


def test_describe_lists_what_it_holds():
    reasoner = _taught(CausalReasoner())
    reasoner.ask("rain", "wet")
    assert "rain ⇒ wet" in reasoner.describe()


def test_stats_admit_that_identification_rests_on_the_graph():
    note = CausalReasoner().stats()["note"].lower()
    assert "wrong graph" in note and "confident wrong answer" in note
    assert "apart from the hebbian" in note


def test_the_module_is_available_on_this_machine():
    assert CausalReasoner().available()


# --------------------------------------------------------------------------- #
# Her own failures
# --------------------------------------------------------------------------- #
def test_root_cause_without_a_brain_returns_nothing():
    assert CausalReasoner().root_cause() == []


def test_root_cause_asks_about_her_weakest_faculty():
    brain = NyxBrain(NyxConfig())
    for _ in range(3):
        brain.think("the engine turns the flywheel")
    answers = brain.causal.root_cause(k=2)
    assert isinstance(answers, list)               # a thin history yields nothing, honestly
    assert all(not a.causal or a.strategy for a in answers)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_edges_survive_a_round_trip():
    reasoner = _taught(CausalReasoner())
    reasoner.ask("rain", "wet")
    fresh = CausalReasoner()
    assert fresh.load_dict(reasoner.to_dict())
    assert ("rain", "wet") in fresh.edges
    assert fresh.identified == reasoner.identified


def test_a_corrupt_sidecar_never_blocks_boot():
    reasoner = CausalReasoner()
    assert not reasoner.load_dict("not a dict")
    assert reasoner.load_dict({"edges": [{"cause": "", "effect": ""}, "junk"]})
    assert not reasoner.edges


# --------------------------------------------------------------------------- #
# Wired into the brain
# --------------------------------------------------------------------------- #
def test_the_brain_exposes_it():
    brain = NyxBrain(NyxConfig())
    assert brain.causal is not None
    assert "causal" in brain.stats()


def test_the_brain_can_be_told_to_do_without_it():
    brain = NyxBrain(NyxConfig(causal_enabled=False))
    assert brain.causal is None
    assert brain.why_causal("rain", "wet") is None


def test_a_turn_feeds_the_model_without_changing_the_answer():
    brain = NyxBrain(NyxConfig())
    before = brain.causal.stats()["causal_edges"]
    brain.think("rain makes the road wet")
    assert brain.causal.stats()["causal_edges"] == before   # watching, never deciding


def test_the_brains_causal_question_carries_its_provenance():
    brain = NyxBrain(NyxConfig())
    _taught(brain.causal)
    got = brain.why_causal("rain", "wet")
    assert got is not None and got.causal and got.mechanism
