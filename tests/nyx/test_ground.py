"""NYX V.01 · symbol grounding — words with referents, and honesty about the ones without."""

from __future__ import annotations

import pytest

from nyxara.kernel.config import get_settings
from nyxara.nyx.brain import NyxBrain
from nyxara.nyx.graph import DynamicNeuralGraph
from nyxara.nyx.ground import WorldGrounding


@pytest.fixture
def ground():
    g = WorldGrounding()
    if not g.available():
        pytest.skip("grounded lexicon unavailable")
    return g


# -------------------- a word with a referent -------------------- #
def test_a_grounded_word_fires_across_several_senses(ground):
    """The difference from a token: 'apple' has taste, sight, touch, physics, affordance."""
    got = ground.understanding("apple")
    assert got.grounded is True
    assert got.depth >= 3
    assert "taste" in got.senses and "vision" in got.senses


def test_grounding_carries_physics_not_just_appearance(ground):
    senses = ground.understanding("apple").senses
    assert "physics" in senses                    # it has weight, and it falls


def test_explanation_is_readable(ground):
    assert "apple" in ground.understanding("apple").explanation


# -------------------- meaning is geometry -------------------- #
def test_similar_things_are_near_and_unlike_things_are_far(ground):
    assert ground.similarity("apple", "orange") > ground.similarity("apple", "car")


def test_neighbours_are_semantic_not_alphabetical(ground):
    neighbours = [n for n, _ in ground.understanding("apple").neighbours]
    assert neighbours and any(n in {"orange", "cherry", "grape", "lemon", "strawberry"}
                              for n in neighbours)


def test_similarity_with_an_unknown_word_is_not_an_error(ground):
    assert 0.0 <= ground.similarity("apple", "zorblax") <= 1.0


# -------------------- honesty about not knowing -------------------- #
def test_an_unknown_word_reports_itself_ungrounded(ground):
    got = ground.understanding("zorblax")
    assert got.grounded is False and got.depth == 0
    assert ground.understands("zorblax") is False


def test_empty_input_is_ungrounded_not_an_error(ground):
    for junk in ("", "   ", None):
        assert ground.understanding(junk).grounded is False


# -------------------- learning a word by reading -------------------- #
def test_a_new_word_can_be_grounded_from_a_sentence(ground):
    assert ground.understands("mango") is False
    assert ground.learn("mango", "A mango is a sweet juicy yellow fruit that is edible.") is True
    got = ground.understanding("mango")
    assert got.grounded is True and got.depth >= 2


def test_learning_records_what_was_learned(ground):
    ground.learn("mango", "A mango is a sweet juicy yellow fruit that is edible.")
    assert "mango" in ground.stats()["recently_learned"]


def test_learning_from_nothing_fails_honestly(ground):
    assert ground.learn("mango", "") is False
    assert ground.learn("", "a description") is False


def test_a_definition_is_told_apart_from_a_passing_mention(ground):
    assert ground.defines("mango", "A mango is a sweet yellow fruit.") is True
    assert ground.defines("gravity", "an apple falls because of gravity") is False
    assert ground.defines("because", "an apple falls because of gravity") is False


def test_automatic_learning_requires_a_definition_not_a_mention(ground):
    """Without this, every word in every sentence gets grounded in whatever else it contained."""
    got = ground.understand(["because", "gravity"], text="an apple falls because of gravity")
    assert got.learned == []
    assert ground.understands("because") is False


def test_automatic_learning_fires_on_a_real_definition(ground):
    got = ground.understand(["mango"], text="A mango is a sweet juicy yellow fruit that is edible.")
    assert got.learned == ["mango"]
    assert ground.understands("mango") is True


def test_a_passing_mention_does_not_pollute_the_graph():
    """The junk grounding this prevents used to invent edges the text never contained."""
    b = NyxBrain(get_settings().nyx.model_copy(update={"holo_dim": 1024}))
    if b.ground is None:
        pytest.skip("grounding disabled")
    b.perceive("gravity pulls an apple toward the ground")
    b.perceive("an apple falls because of gravity")
    related = dict(b.related("gravity", k=8))
    assert related["apple"] == max(related.values())


def test_reading_can_be_switched_off():
    g = WorldGrounding(read_unknown=False)
    if not g.available():
        pytest.skip("grounded lexicon unavailable")
    got = g.understand(["zorblax"], text="A zorblax is a sweet red edible fruit.")
    assert got.ungrounded == ["zorblax"] and got.learned == []


# -------------------- a whole turn -------------------- #
def test_coverage_reports_how_much_of_a_turn_she_understands(ground):
    got = ground.understand(["apple", "zorblax"], text="")
    assert got.understood == ["apple"] and got.ungrounded == ["zorblax"]
    assert got.coverage == 0.5


def test_grounding_binds_features_into_the_graph(ground):
    """Understanding accretes in the same structure everything else does."""
    graph = DynamicNeuralGraph(max_nodes=200)
    ground.understand(["apple"], graph=graph)
    partners = {n for n, _ in graph.neighbours("apple", k=8)}
    assert partners & {"red", "sweet", "round", "edible"}


def test_binding_can_be_switched_off():
    g = WorldGrounding(bind_to_graph=False)
    if not g.available():
        pytest.skip("grounded lexicon unavailable")
    graph = DynamicNeuralGraph(max_nodes=200)
    g.understand(["apple"], graph=graph)
    assert graph.neighbours("apple") == []


def test_understand_never_raises_on_hostile_input(ground):
    for concepts in (None, [], "", ["", "  "], "!!! ???"):
        assert ground.understand(concepts) is not None


# -------------------- wired into the brain -------------------- #
@pytest.fixture
def brain():
    cfg = get_settings().nyx.model_copy(update={"holo_dim": 1024, "holo_capacity": 64})
    return NyxBrain(cfg)


def test_perceiving_grounds_the_turn(brain):
    if brain.ground is None:
        pytest.skip("grounding disabled")
    got = brain.perceive("the apple fell from the tree")
    assert got.grounded is not None and got.grounded.coverage > 0.0


def test_the_brain_exposes_what_it_understands(brain):
    if brain.ground is None:
        pytest.skip("grounding disabled")
    assert brain.understanding("apple").grounded is True
    assert brain.understanding("zorblax").grounded is False


def test_a_brain_without_grounding_still_thinks():
    cfg = get_settings().nyx.model_copy(update={"holo_dim": 1024, "ground_enabled": False})
    b = NyxBrain(cfg)
    assert b.ground is None
    assert b.understanding("apple") is None
    assert b.think("what pulls an apple down") is not None


def test_learned_groundings_round_trip(brain):
    if brain.ground is None:
        pytest.skip("grounding disabled")
    brain.ground.learn("mango", "A mango is a sweet juicy yellow fruit that is edible.")
    data = brain.to_dict()

    revived = NyxBrain(get_settings().nyx.model_copy(update={"holo_dim": 1024}))
    revived.load_dict(data)
    assert "mango" in revived.ground.learned
