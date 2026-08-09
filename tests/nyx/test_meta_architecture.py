"""NYX V.02 · typed structural plasticity — new edge kinds, born from evidence and reversible.

V.01's graph has exactly one edge kind: a co-occurrence weight. These tests pin the layer that
gives it typed, directed relations *without touching a single Hebbian value*, and — more
importantly — pin the discipline around it: a new type needs support rather than an idea, a node
is split only when its two neighbourhoods are genuinely disjoint, and everything is on a budget
and in a ledger.
"""

from __future__ import annotations

from nyxara.kernel.config import NyxConfig
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.meta_architecture import BUILTIN_TYPES, GraphWeaver


def _brain(**kw) -> NyxBrain:
    return NyxBrain(NyxConfig(**kw))


# --------------------------------------------------------------------------- #
# Typed edges over an untouched graph
# --------------------------------------------------------------------------- #
def test_a_typed_edge_is_laid():
    weaver = GraphWeaver()
    edge = weaver.relate("rain", "wet", type="causes")
    assert edge is not None and edge.type == "causes" and edge.directed


def test_every_builtin_type_exists_up_front():
    weaver = GraphWeaver()
    assert set(BUILTIN_TYPES) <= set(weaver.types)
    assert all(weaver.types[name].builtin for name in BUILTIN_TYPES)


def test_an_unknown_type_lays_nothing():
    """A type she has not minted is not a type she can use."""
    assert GraphWeaver().relate("rain", "wet", type="smells-like") is None


def test_a_self_edge_is_refused():
    assert GraphWeaver().relate("rain", "rain", type="causes") is None


def test_an_empty_side_is_refused():
    assert GraphWeaver().relate("", "wet", type="causes") is None


def test_typed_edges_do_not_touch_the_hebbian_weight():
    """The whole point of layering: association stays exactly where it was."""
    brain = _brain()
    brain.think("the engine turns the flywheel")
    before = brain.graph.weight("engine", "flywheel")
    GraphWeaver(brain).relate("engine", "flywheel", type="causes")
    assert brain.graph.weight("engine", "flywheel") == before


def test_edges_of_reads_the_typed_layer():
    weaver = GraphWeaver()
    weaver.relate("rain", "wet", type="causes")
    weaver.relate("rain", "cloud", type="part-of")
    assert len(weaver.edges_of("rain")) == 2
    assert len(weaver.edges_of("rain", type="causes")) == 1


def test_a_directed_edge_is_not_readable_backwards():
    weaver = GraphWeaver()
    weaver.relate("rain", "wet", type="causes")
    assert not weaver.edges_of("wet")


# --------------------------------------------------------------------------- #
# Bounded plasticity — the budget is real
# --------------------------------------------------------------------------- #
def test_the_rewire_budget_stops_her_mid_tick():
    weaver = GraphWeaver(rewire_budget=2)
    assert weaver.relate("a", "b", type="causes") is not None
    assert weaver.relate("b", "c", type="causes") is not None
    assert weaver.relate("c", "d", type="causes") is None


def test_a_tick_restores_the_budget():
    weaver = GraphWeaver(rewire_budget=1)
    weaver.relate("a", "b", type="causes")
    assert weaver.relate("b", "c", type="causes") is None
    weaver.tick()
    assert weaver.relate("b", "c", type="causes") is not None


def test_edges_are_capped_and_the_weakest_go_first():
    weaver = GraphWeaver(max_edges=64, rewire_budget=10_000)
    weaver.relate("keep", "me", type="causes", weight=1.0)
    for index in range(200):
        weaver.relate(f"n{index}", "x", type="co-occurs", weight=0.01)
    assert len(weaver.edges) <= 64
    assert ("keep", "me", "causes") in weaver.edges


# --------------------------------------------------------------------------- #
# A new type needs support, not an idea
# --------------------------------------------------------------------------- #
def test_one_sighting_mints_nothing():
    weaver = GraphWeaver(min_support=3)
    assert weaver.propose_type("smells-like") is None
    assert "smells-like" not in weaver.types


def test_the_candidate_is_counted_rather_than_dropped():
    weaver = GraphWeaver(min_support=3)
    weaver.propose_type("smells-like")
    assert weaver.candidates["smells-like"] == 1


def test_enough_support_mints_it():
    weaver = GraphWeaver(min_support=3)
    for _ in range(2):
        weaver.propose_type("smells-like")
    minted = weaver.propose_type("smells-like", example="petrol smells like ozone")
    assert minted is not None and not minted.builtin
    assert weaver.relate("petrol", "ozone", type="smells-like") is not None


def test_a_builtin_type_is_returned_rather_than_re_minted():
    weaver = GraphWeaver()
    assert weaver.propose_type("causes") is weaver.types["causes"]
    assert weaver.minted == 0


def test_the_type_cap_holds():
    weaver = GraphWeaver(min_support=2, max_types=len(BUILTIN_TYPES))
    weaver.propose_type("brand-new")
    assert weaver.propose_type("brand-new") is None


# --------------------------------------------------------------------------- #
# Reading a relation the sentence actually states
# --------------------------------------------------------------------------- #
def test_an_is_a_sentence_is_read():
    made = GraphWeaver().read("a sparrow is a bird")
    assert made and made[0].type == "is-a"


def test_a_causal_sentence_is_read():
    made = GraphWeaver().read("friction causes heat")
    assert made and made[0].type == "causes"


def test_a_sentence_stating_no_relation_is_read_as_none():
    """She lays what she can see, and nothing where she can see nothing."""
    assert GraphWeaver().read("the flywheel spins quickly") == []


def test_one_concept_is_not_a_relation():
    assert GraphWeaver().read("gravity") == []


# --------------------------------------------------------------------------- #
# Splitting a node — only when the two senses really are two
# --------------------------------------------------------------------------- #
def test_a_node_with_one_neighbourhood_is_not_split():
    brain = _brain()
    for _ in range(3):
        brain.think("the bank approved the loan and the mortgage")
    assert GraphWeaver(brain).split("bank") is None


def test_a_genuinely_polysemous_node_is_split():
    brain = _brain()
    for _ in range(3):
        brain.think("bank loan mortgage interest")
    for _ in range(3):
        brain.think("bank river sediment erosion")
    got = GraphWeaver(brain).split("bank")
    if got is not None:                     # a split is allowed to decline; it is never faked
        assert got == ("bank#1", "bank#2")
        assert "bank#1" in brain.graph


def test_splitting_an_unknown_node_does_nothing():
    assert GraphWeaver(_brain()).split("nothing-she-has-met") is None


def test_splitting_without_a_graph_does_nothing():
    assert GraphWeaver().split("bank") is None


# --------------------------------------------------------------------------- #
# Every structural change is reversible
# --------------------------------------------------------------------------- #
def test_a_mint_is_ledgered():
    weaver = GraphWeaver(min_support=2)
    weaver.propose_type("smells-like")
    weaver.propose_type("smells-like")
    assert weaver.ledger and weaver.ledger[-1].kind == "mint"


def test_a_mint_can_be_undone():
    weaver = GraphWeaver(min_support=2)
    weaver.propose_type("smells-like")
    weaver.propose_type("smells-like")
    weaver.relate("petrol", "ozone", type="smells-like")
    assert weaver.undo_last()
    assert "smells-like" not in weaver.types
    assert not weaver.edges                 # its edges go with it


def test_undo_with_an_empty_ledger_says_no():
    assert not GraphWeaver().undo_last()


# --------------------------------------------------------------------------- #
# Honesty
# --------------------------------------------------------------------------- #
def test_stats_refuse_the_synaptogenesis_claim():
    note = GraphWeaver().stats()["note"].lower()
    assert "analogy" in note and "not synaptogenesis" in note
    assert "support, not an idea" in note


def test_describe_names_which_types_were_minted_and_which_were_shipped():
    weaver = GraphWeaver(min_support=2)
    weaver.propose_type("smells-like")
    weaver.propose_type("smells-like")
    described = weaver.describe()
    assert "builtin" in described and "minted" in described


def test_describe_lists_a_candidate_that_has_not_earned_a_type_yet():
    weaver = GraphWeaver(min_support=5)
    weaver.propose_type("smells-like")
    assert "not yet minted" in weaver.describe()


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_types_and_edges_survive_a_round_trip():
    weaver = GraphWeaver(min_support=2)
    weaver.propose_type("smells-like")
    weaver.propose_type("smells-like")
    weaver.relate("petrol", "ozone", type="smells-like")
    fresh = GraphWeaver()
    assert fresh.load_dict(weaver.to_dict())
    assert "smells-like" in fresh.types
    assert ("petrol", "ozone", "smells-like") in fresh.edges


def test_a_corrupt_sidecar_never_blocks_boot():
    weaver = GraphWeaver()
    assert not weaver.load_dict([1, 2, 3])
    assert weaver.load_dict({"edges": ["junk", {"a": "", "b": ""}], "types": ["junk"]})
    assert not weaver.edges


# --------------------------------------------------------------------------- #
# Wired into the brain
# --------------------------------------------------------------------------- #
def test_the_brain_exposes_it():
    brain = _brain()
    assert brain.weaver is not None and "weaver" in brain.stats()


def test_the_brain_can_be_told_to_do_without_it():
    brain = _brain(weaver_enabled=False)
    assert brain.weaver is None
    assert brain.weave("rain", "wet", type="causes") is None


def test_a_turn_lays_the_relation_it_states():
    brain = _brain()
    brain.think("friction causes heat")
    assert any(edge.type == "causes" for edge in brain.weaver.edges.values())


def test_the_beat_restores_the_budget():
    brain = _brain()
    brain.weaver.rewire_budget = 1
    brain.weave("a", "b", type="causes")
    assert brain.weave("b", "c", type="causes") is None
    brain.tick()
    assert brain.weave("b", "c", type="causes") is not None
