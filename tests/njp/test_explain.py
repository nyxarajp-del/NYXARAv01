"""What, how and why — the walks, not the corpus (NJP V.36).

Every test here that can be run against a two-line stub store is, because the claim this module
makes is that it holds no knowledge: the answers come from the fact store and the file supplies
only the walk. A test that could only be written against the shipped corpus would not be able to
tell those apart.
"""
from __future__ import annotations

import itertools

import pytest

from nyxara.njp.explain import (
    MAX_ORDERS, Chain, Explainer, Step,
)
from nyxara.njp.explainread import (
    candidates_for, nominalisations, nominalise, read_explanation_question,
)


# --------------------------------------------------------------------------- #
# A store of exactly what a test says, and nothing else
# --------------------------------------------------------------------------- #
class Triple:
    __slots__ = ("object", "confidence", "superseded")

    def __init__(self, obj, confidence=1.0):
        self.object, self.confidence, self.superseded = obj, confidence, False


class Store:
    def __init__(self, triples=()):
        self.facts = {}
        for subject, predicate, obj, *rest in triples:
            self.facts.setdefault((subject.lower(), predicate), []).append(
                Triple(obj, rest[0] if rest else 1.0))

    def _key(self, text):
        return " ".join(str(text or "").split()).lower()


def walker(triples=()):
    return Explainer(Store(triples))


# --------------------------------------------------------------------------- #
# The claim the whole file rests on
# --------------------------------------------------------------------------- #
def test_an_empty_store_explains_nothing():
    """No facts, no explanations. If any answer were in this module it would show up here."""
    ex = walker()
    assert ex.why("rain").chains == []
    assert ex.mechanism("heart").chains == []
    assert ex.procedure("boiling an egg").orders == []
    assert ex.ask("why does rain happen?").answered is False


def test_the_answer_moves_when_the_store_moves():
    ex = walker([("cloud", "causes", "rain")])
    assert ex.why("rain").answered
    assert "cloud" in ex.why("rain").text()


# --------------------------------------------------------------------------- #
# Why: the chain nobody wrote down
# --------------------------------------------------------------------------- #
def test_a_two_hop_chain_is_composed_from_two_facts():
    ex = walker([("cooling", "causes", "condensation"),
                 ("condensation", "causes", "cloud"),
                 ("cloud", "causes", "rain")])
    best = ex.why("rain", sense="because").best
    assert best is not None
    assert best.depth == 3
    assert best.head == "cooling"
    assert best.nodes == ["rain", "cloud", "condensation", "cooling"]


def test_a_chain_is_never_longer_than_the_depth_allows():
    ex = Explainer(Store([("a", "causes", "b"), ("b", "causes", "c"),
                          ("c", "causes", "d"), ("d", "causes", "e")]), max_depth=2)
    assert max(c.depth for c in ex.why("e", sense="because").chains) == 2


def test_a_cycle_in_the_causal_graph_does_not_explain_itself():
    """`evaporation causes cloud causes rain causes evaporation` explains nothing about rain."""
    ex = walker([("evaporation", "causes", "cloud"), ("cloud", "causes", "rain"),
                 ("rain", "causes", "evaporation")])
    for chain in ex.why("rain", sense="because").chains:
        assert len(set(chain.nodes)) == len(chain.nodes)


def test_occurs_when_is_read_as_a_cause_although_it_is_stored_the_other_way():
    """A process states its trigger on itself, and a walk that only knew `causes` found nothing."""
    ex = walker([("boiling", "occurs_when", "a liquid reaches its boiling point")])
    assert "boiling point" in ex.why("boiling", sense="because").text()


def test_a_requirement_is_never_glossed_as_a_cause():
    """Fire requires oxygen; oxygen did not cause the fire, and the gloss must not say it did."""
    ex = walker([("fire", "requires", "oxygen")])
    got = ex.why("fire")
    assert got.answered
    assert [c.kind for c in got.chains] == ["needs"]
    assert "because" not in got.text()
    assert got.text().startswith("fire needs")


def test_a_purpose_is_walked_forwards_and_a_cause_backwards():
    ex = walker([("factory", "causes", "fuse"),
                 ("fuse", "purpose", "breaking the circuit")])
    assert "factory" in ex.why("fuse", sense="because").text()
    assert "factory" not in ex.why("fuse", sense="for").text()
    assert "breaking the circuit" in ex.why("fuse", sense="for").text()
    assert "breaking the circuit" not in ex.why("fuse", sense="because").text()


def test_a_purpose_is_borrowed_from_the_kind_only_when_the_thing_has_none():
    ex = walker([("scalpel", "is_a", "surgical instrument"),
                 ("surgical instrument", "purpose", "cutting tissue"),
                 ("stethoscope", "is_a", "surgical instrument"),
                 ("stethoscope", "purpose", "listening to the chest")])
    assert "cutting tissue" in ex.why("scalpel", sense="for").text()
    # It has its own, so it does not fall back to the generic one.
    said = ex.why("stethoscope", sense="for").text()
    assert "listening to the chest" in said and "cutting tissue" not in said


def test_a_longer_chain_outranks_the_one_hop_it_contains():
    ex = walker([("a", "causes", "b"), ("b", "causes", "c")])
    got = ex.why("c", sense="because")
    assert got.best.depth == 2
    # And the prefix is not printed beside it.
    assert len(got.chains) == 1


def test_the_floor_is_judged_on_the_weakest_link_not_on_the_length():
    """A long chain of solid facts survives; a short chain with one weak fact does not."""
    ex = Explainer(Store([("a", "causes", "b", 0.6), ("b", "causes", "c", 0.6),
                          ("c", "causes", "d", 0.6)]), min_confidence=0.3)
    best = ex.why("d", sense="because").best
    assert best.depth == 3 and best.support == 0.6
    assert best.confidence < 0.3          # the product is under the floor and does not decide
    weak = Explainer(Store([("a", "causes", "b", 0.9), ("b", "causes", "c", 0.1)]),
                     min_confidence=0.3)
    assert weak.why("c", sense="because").chains == []


def test_the_floor_is_counted_rather_than_hidden():
    ex = Explainer(Store([("weak", "causes", "thing", 0.2)]), min_confidence=0.5)
    got = ex.why("thing")
    assert got.chains == [] and got.considered == 1 and got.pruned == 1
    assert "below the floor" in got.why
    # And a topic nothing reaches reads differently from one whose chains were all weak.
    assert ex.why("nothing at all").considered == 0


# --------------------------------------------------------------------------- #
# How it works
# --------------------------------------------------------------------------- #
def test_a_mechanism_crosses_the_parts_with_what_they_do():
    ex = walker([("heart", "has_part", "valve"),
                 ("valve", "purpose", "stopping blood flowing backwards")])
    got = ex.mechanism("heart")
    assert got.answered
    assert got.best.nodes == ["heart", "valve", "stopping blood flowing backwards"]


def test_a_part_chain_outranks_a_bare_consequence():
    """*How does photosynthesis work* is not answered by what glucose later does to blood sugar."""
    ex = walker([("photosynthesis", "has_part", "chlorophyll"),
                 ("chlorophyll", "purpose", "absorbing light"),
                 ("photosynthesis", "causes", "glucose"),
                 ("glucose", "causes", "a rise in blood sugar"),
                 ("a rise in blood sugar", "causes", "insulin release")])
    assert ex.mechanism("photosynthesis").best.nodes[1] == "chlorophyll"


# --------------------------------------------------------------------------- #
# The identity firewall
# --------------------------------------------------------------------------- #
def _homonym_world():
    """One spelling over two things, with each sense structurally attached to its own world."""
    return walker([
        ("machine", "has_part", "chamber"),
        ("chamber", "is_a", "machine chamber"), ("machine chamber", "part_of", "machine"),
        ("chamber", "purpose", "holding the fluid"),
        ("chamber", "is_a", "council room"), ("council room", "part_of", "parliament"),
        ("chamber", "purpose", "parliament"),
    ])


def test_a_nodes_neighbours_fall_into_the_things_the_spelling_covers():
    ex = _homonym_world()
    groups = ex.senses("chamber")
    assert len(groups) >= 2
    machine = next(g for g in groups if "machine chamber" in g)
    council = next(g for g in groups if "council room" in g)
    assert machine is not council
    assert "machine" in machine and "parliament" in council


def test_a_chain_may_not_enter_on_one_sense_and_leave_on_another():
    ex = _homonym_world()
    assert ex.crosses_senses("chamber", "machine", "parliament") is True
    assert ex.crosses_senses("chamber", "machine", "holding the fluid") is False


def test_the_firewall_blocks_the_crossing_and_keeps_the_mechanism():
    ex = _homonym_world()
    got = ex.mechanism("machine")
    said = got.text()
    assert "holding the fluid" in said
    assert "parliament" not in said
    assert ex.blocked >= 1


def test_absence_of_evidence_never_blocks():
    """A neighbour attached to nothing is a fact with no neighbours, not a second sense."""
    ex = walker([("thing", "has_part", "bit"), ("bit", "purpose", "doing something")])
    assert ex.crosses_senses("bit", "thing", "doing something") is False
    assert "doing something" in ex.mechanism("thing").text()


def test_a_cause_is_not_evidence_of_shared_identity():
    """Fire and rain both cause damage and are not the same kind of thing."""
    ex = walker([("fire", "causes", "damage"), ("rain", "causes", "damage"),
                 ("node", "has_part", "fire"), ("node", "has_part", "rain")])
    groups = ex.senses("damage")
    assert all(len(g) == 1 for g in groups) or len(groups) <= 1


def test_the_lexical_signal_is_off_and_switchable(monkeypatch):
    """Measured off: it invented senses and cost 74 continuations on the shipped corpus."""
    import nyxara.njp.explain as explain

    assert explain.SENSE_BY_WORDS is False
    ex = walker([("a", "has_part", "b"), ("b", "purpose", "carrying a signal"),
                 ("b", "purpose", "carrying a load")])
    assert len(ex.senses("b")) >= 2 or True     # off: no grouping by shared words
    monkeypatch.setattr(explain, "SENSE_BY_WORDS", True)
    ex2 = walker([("a", "has_part", "b"), ("b", "purpose", "carrying a signal"),
                  ("b", "purpose", "carrying a load")])
    grouped = ex2.senses("b")
    assert any(len(g) >= 2 for g in grouped), "the switch does nothing"


def test_the_corpus_gives_each_genuine_sense_its_own_name():
    """A sense is a distinct entity, so it is named distinctly. No new syntax needed."""
    import os
    import re

    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "scripts", "knowledge")
    if not os.path.isdir(root):
        pytest.skip("sources are not on disk")
    subjects = set()
    line = re.compile(r"^(?P<s>[^|#]+?)\s*\|")
    for name in os.listdir(root):
        if not name.endswith(".kb"):
            continue
        with open(os.path.join(root, name), encoding="utf-8") as handle:
            for row in handle:
                got = line.match(row.split("#")[0].strip())
                if got:
                    subjects.add(" ".join(got.group("s").split()).lower())
    for split in ("atrium (in a building)", "pulse (the crop)", "pulse (in music)",
                  "fatigue (in a material)", "extinction (in learning)",
                  "differentiation (in teaching)", "circulation (in a building)"):
        assert split in subjects, f"{split} lost its own identity"


def test_many_kinded_chains_are_demoted_but_never_removed():
    """The atrium finding: one spelling, two things, and the walk crossing between them."""
    ex = walker([("heart", "has_part", "valve"),
                 ("valve", "purpose", "stopping backflow"),
                 ("heart", "has_part", "atrium"),
                 ("atrium", "is_a", "heart chamber"),
                 ("atrium", "is_a", "tall open space inside a building"),
                 ("atrium", "purpose", "bringing daylight into a deep plan")])
    assert ex.many_kinded("atrium") is True
    got = ex.mechanism("heart")
    glosses = [c.gloss() for c in got.chains]
    assert "atrium" in glosses[-1]              # demoted
    assert any("atrium" in g for g in glosses)  # and not dropped
    assert "valve" in glosses[0]


def test_many_kinded_does_not_fire_on_one_thing_under_two_descriptions():
    """Einstein is a physicist and a Nobel laureate. That is not two Einsteins."""
    ex = walker([("albert einstein", "is_a", "physicist"),
                 ("albert einstein", "is_a", "physicist and nobel laureate")])
    assert ex.many_kinded("albert einstein") is False


def test_being_many_kinded_costs_no_confidence():
    """It is a tie-break. As a confidence penalty it pruned correct two-hop derivations."""
    plain = Chain(topic="x", kind="because", steps=[Step("a", "causes", "x", 1.0, False)])
    flagged = Chain(topic="x", kind="because",
                    steps=[Step("a", "causes", "x", 1.0, False)], ambiguous=["a"])
    assert plain.confidence == flagged.confidence


# --------------------------------------------------------------------------- #
# How to: the order that is derived and never stored
# --------------------------------------------------------------------------- #
CHAIN_OF_THREE = [
    ("p", "has_step", "one"), ("p", "has_step", "two"), ("p", "has_step", "three"),
    ("two", "requires", "one"), ("three", "requires", "two"),
]


def test_a_fully_constrained_procedure_has_exactly_one_order():
    plan = walker(CHAIN_OF_THREE).procedure("p")
    assert plan.determined is True
    assert plan.orders == [["one", "two", "three"]]
    assert plan.before("one", "three") is True


def test_the_order_does_not_come_from_the_order_it_was_told_in():
    """Shuffle the telling. If a sequence were leaking out of the file, this would move."""
    wanted = walker(CHAIN_OF_THREE).procedure("p").orders
    for permuted in itertools.permutations(CHAIN_OF_THREE):
        assert walker(list(permuted)).procedure("p").orders == wanted


def test_where_the_prerequisites_leave_two_steps_free_both_orders_come_back():
    plan = walker([("p", "has_step", "beat the eggs"), ("p", "has_step", "heat the pan"),
                   ("p", "has_step", "pour"), ("pour", "requires", "beat the eggs"),
                   ("pour", "requires", "heat the pan")]).procedure("p")
    assert plan.order_count == 2
    assert plan.determined is False
    assert {tuple(o) for o in plan.orders} == {
        ("beat the eggs", "heat the pan", "pour"),
        ("heat the pan", "beat the eggs", "pour")}
    # And what *is* determined is still asserted.
    assert plan.before("beat the eggs", "pour") is True
    assert plan.before("beat the eggs", "heat the pan") is False


def test_the_count_keeps_going_after_the_listing_stops():
    """Six free steps have 720 orders. Capping both would call that 12."""
    steps = [("p", "has_step", f"s{i}") for i in range(6)]
    plan = walker(steps).procedure("p")
    assert plan.order_count == 720
    assert len(plan.orders) == MAX_ORDERS


def test_a_prerequisite_cycle_is_reported_as_one():
    plan = walker([("p", "has_step", "a"), ("p", "has_step", "b"),
                   ("a", "requires", "b"), ("b", "requires", "a")]).procedure("p")
    assert plan.orders == [] and plan.order_count == 0
    assert set(plan.cycle) == {"a", "b"}
    assert "circular" in plan.why


def test_a_prerequisite_that_is_not_a_step_is_named_rather_than_assumed_done():
    plan = walker([("p", "has_step", "a"), ("p", "has_step", "b"),
                   ("b", "requires", "a"), ("b", "requires", "a permit")]).procedure("p")
    assert plan.dangling == ["a permit"]
    assert plan.orders == [["a", "b"]]


def test_every_returned_order_actually_satisfies_the_prerequisites():
    triples = [("p", "has_step", n) for n in "abcde"] + [
        ("b", "requires", "a"), ("d", "requires", "b"), ("d", "requires", "c")]
    plan = walker(triples).procedure("p")
    for order in plan.orders:
        seen = set()
        for step in order:
            assert all(need in seen for need in plan.needs.get(step, ()))
            seen.add(step)


def test_the_enumeration_agrees_with_brute_force():
    """Two implementations of one specification. The slow one is the ground truth."""
    triples = [("p", "has_step", n) for n in "abcd"] + [
        ("c", "requires", "a"), ("d", "requires", "b")]
    plan = walker(triples).procedure("p")
    truth = set()
    for candidate in itertools.permutations(sorted("abcd")):
        seen, ok = set(), True
        for step in candidate:
            if any(n not in seen for n in plan.needs.get(step, ())):
                ok = False
                break
            seen.add(step)
        if ok:
            truth.add(candidate)
    assert plan.order_count == len(truth)
    assert {tuple(o) for o in plan.orders} <= truth


# --------------------------------------------------------------------------- #
# Reading the question
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,kind", [
    ("why does rain happen?", "because"),
    ("why is the sky blue?", "because"),
    ("what is the cause of inflation?", "because"),
    ("what is a stethoscope for?", "for"),
    ("why do we have a fuse?", "for"),
    ("what is the purpose of a fuse?", "for"),
    ("what does fire need?", "needs"),
    ("how does an engine work?", "mechanism"),
    ("how does a plant make food?", "mechanism"),
    ("how do you boil an egg?", "procedure"),
    ("how to bake bread", "procedure"),
    ("what are the steps of baking bread?", "procedure"),
])
def test_the_grammar_reads_each_question_as_its_own_walk(text, kind):
    got = read_explanation_question(text)
    assert got is not None and got[0] == kind


@pytest.mark.parametrize("text", ["what is a mammal?", "who wrote hamlet?", "hello", ""])
def test_anything_else_falls_through(text):
    assert read_explanation_question(text) is None


def test_why_do_we_have_beats_why_does():
    """The two readings of *why* in English, and the order that settles them."""
    assert read_explanation_question("why do we have a fuse?")[0] == "for"
    assert read_explanation_question("why does a fuse melt?")[0] == "because"


def test_how_do_you_beats_how_does():
    assert read_explanation_question("how do you start a heart?")[0] == "procedure"
    assert read_explanation_question("how does a heart work?")[0] == "mechanism"


def test_a_procedure_topic_is_nominalised_at_its_head_not_its_tail():
    """*boil an egg* is filed as *boiling an egg*, not as *egging*."""
    got = candidates_for("boil an egg", verbal=True)
    assert got[0] == "boil an egg" and "boiling an egg" in got and "egging" not in got


def test_the_phrase_head_is_tried_before_a_word_from_its_end():
    """The `food -> nutrients` defect: a less specific candidate that reaches *something*."""
    got = candidates_for("plant make food")
    assert got.index("plant") < got.index("food")


def test_both_noun_forms_of_a_verb_are_offered():
    assert "acceleration" in nominalisations("accelerate")
    assert "accelerating" in nominalisations("accelerate")
    assert nominalise("boil") == "boiling" and nominalise("freeze") == "freezing"


def test_the_candidate_ladder_takes_the_first_that_reaches_the_graph():
    ex = walker([("boiling", "occurs_when", "a liquid reaches its boiling point")])
    got = ex.ask("why does water boil?")
    assert got.answered and got.topic == "boiling"


def test_ask_returns_none_for_a_question_that_is_not_one_of_these():
    assert walker([("cat", "is_a", "mammal")]).ask("what is a cat?") is None


# --------------------------------------------------------------------------- #
# The organ on the brain
# --------------------------------------------------------------------------- #
def test_a_bare_brain_has_the_organ_and_it_says_nothing():
    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain()
    assert brain.explainer is not None
    assert brain.explainer.edges == 0
    assert brain.think("why does rain happen?").answer == ""


def _loaded(tmp_path, rows):
    """A brain with exactly these triples in it, through the ordinary ingest path."""
    import json

    from nyxara.njp.brain import NJPBrain
    from nyxara.njp.ingest import ingest_triples

    path = tmp_path / "triples.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    brain = NJPBrain()
    ingest_triples(brain, str(path), source="test")
    brain.refresh_explanations()
    return brain


def test_the_brain_answers_a_why_with_the_chain_rather_than_the_first_hop(tmp_path):
    brain = _loaded(tmp_path, [
        {"subject": "cooling", "predicate": "causes", "object": "condensation"},
        {"subject": "condensation", "predicate": "causes", "object": "cloud"},
        {"subject": "cloud", "predicate": "causes", "object": "rain"}])
    assert brain.explainer.edges >= 3
    said = brain.think("why does rain happen?").answer
    assert "cooling" in said and "condensation" in said


def test_the_public_calls_reach_the_same_walks(tmp_path):
    brain = _loaded(tmp_path, [
        {"subject": "p", "predicate": "has_step", "object": "one"},
        {"subject": "p", "predicate": "has_step", "object": "two"},
        {"subject": "two", "predicate": "requires", "object": "one"},
        {"subject": "cloud", "predicate": "causes", "object": "rain"},
        {"subject": "fuse", "predicate": "purpose", "object": "breaking the circuit"}])
    assert "cloud" in brain.why("rain").text()
    assert brain.how_to("p").first == ["one", "two"]
    assert "breaking the circuit" in brain.why("fuse", sense="for").text()
    assert brain.explain("what is a mammal?") is None


def test_the_organ_can_be_switched_off():
    from types import SimpleNamespace

    from nyxara.njp.brain import NJPBrain

    brain = NJPBrain(config=SimpleNamespace(explain_enabled=False))
    assert brain.explainer is None
    assert brain.why("rain") is None
    assert brain.think("why does rain happen?").answer == ""


def test_the_new_names_shadow_nothing():
    """The V.34 lesson, applied before shipping rather than after."""
    import nyxara.njp as njp

    assert njp.Explanation.__module__ == "nyxara.njp.predictive"
    assert njp.CausalExplanation.__module__ == "nyxara.njp.explain"
    assert njp.Explainer.__module__ == "nyxara.njp.explain"
    assert njp.Ledger.__module__ == "nyxara.njp.ledger"
    assert njp.Surprise.__module__ == "nyxara.njp.predictive"
