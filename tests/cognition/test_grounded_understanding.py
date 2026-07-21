"""Tests for nyxara.cognition.grounded_understanding — perceptual symbol grounding.

These prove the headline requirement: NYXARA understands a word by *what it is in the
senses* — "apple" fires taste, vision, touch, physics and affordance simultaneously — and
that this grounded meaning is learned from reading, compared as geometry, and feeds the
abstraction taxonomy. The deterministic floor never needs a network or a model.
"""

from __future__ import annotations

from nyxara.cognition.grounded_understanding import (
    DESCRIPTOR_ONTOLOGY,
    GroundedActivation,
    GroundedConcept,
    GroundedLexicon,
    PerceptualSchema,
    Sense,
)


# -------------------- PerceptualSchema: stable, fixed-width -------------------- #
def test_schema_assigns_stable_dimensions():
    sch = PerceptualSchema()
    i_sweet = sch.index_of(Sense.TASTE, "sweet")
    i_red = sch.index_of(Sense.VISION, "red")
    assert sch.index_of(Sense.TASTE, "sweet") == i_sweet      # never moves
    assert sch.index_of(Sense.TASTE, "Sweet") == i_sweet      # case-insensitive
    assert i_sweet != i_red                                    # different (sense,feature)


def test_schema_vectorize_is_fixed_width_float():
    lex = GroundedLexicon()
    c = lex.get("apple")
    v = lex.schema.vectorize(c)
    assert len(v) == lex.schema.width()
    assert all(isinstance(x, float) for x in v)
    assert any(x != 0.0 for x in v)


# -------------------- the headline: many senses fire at once -------------------- #
def test_apple_activates_many_modalities_simultaneously():
    lex = GroundedLexicon()
    act = lex.activate("apple")
    assert isinstance(act, GroundedActivation)
    assert act.grounded is True
    # not a lone token — a world: taste + vision + physics all light up together
    assert act.fires(Sense.TASTE)
    assert act.fires(Sense.VISION)
    assert act.fires(Sense.PHYSICS)
    assert len(act.modalities) >= 3
    d = act.to_dict()
    assert d["word"] == "apple" and d["n_modalities"] >= 3


def test_plural_grounds_to_same_concept():
    lex = GroundedLexicon()
    assert lex.activate("apples").grounded          # singularised to "apple"
    assert lex.get("apples") is lex.get("apple")


def test_unknown_word_is_honest_not_hallucinated():
    lex = GroundedLexicon()
    act = lex.activate("zorblax")                    # no LLM configured → honest blank
    assert act.grounded is False
    assert act.modalities == []


# -------------------- learning grounded meaning from language -------------------- #
def test_learn_from_text_attaches_right_senses():
    lex = GroundedLexicon(seed=False)               # start empty: meaning purely emergent
    rep = lex.learn_from_text("Mangoes are sweet and yellow and soft.")
    assert "mango" in rep["grounded"]
    c = lex.get("mango")
    assert c.senses[Sense.TASTE].get("sweet", 0) > 0
    assert c.senses[Sense.VISION].get("yellow", 0) > 0
    assert c.senses[Sense.TOUCH].get("soft", 0) > 0


def test_learn_affordance_and_gravity_from_verbs():
    lex = GroundedLexicon(seed=False)
    lex.learn_from_text("You can eat a plumfruit. A plumfruit falls to the ground.")
    act = lex.activate("plumfruit")
    assert act.fires(Sense.AFFORDANCE)              # "eat" → edible
    assert act.fires(Sense.PHYSICS)                 # "falls" → gravity
    assert lex.get("plumfruit").senses[Sense.PHYSICS].get("gravity", 0) > 0


def test_negation_flips_feature_sign():
    lex = GroundedLexicon(seed=False)
    lex.learn_from_text("A limewedge is not sweet.")
    assert lex.get("limewedge").senses[Sense.TASTE]["sweet"] < 0


def test_is_a_category_is_captured():
    lex = GroundedLexicon(seed=False)
    lex.learn_from_text("A robin is a bird.")
    assert "bird" in lex.get("robin").categories


# -------------------- meaning is geometry -------------------- #
def test_grounded_similarity_fruit_closer_than_vehicle():
    lex = GroundedLexicon()
    s_fruit = lex.similarity("apple", "orange")
    s_car = lex.similarity("apple", "car")
    assert s_fruit > s_car
    assert s_fruit > 0.4                            # fruits genuinely overlap
    # symmetric and self-similar
    assert abs(lex.similarity("apple", "orange") - lex.similarity("orange", "apple")) < 1e-9
    assert lex.similarity("apple", "apple") > 0.99


def test_neighbors_are_perceptually_related():
    lex = GroundedLexicon()
    names = [n for n, _ in lex.neighbors("apple", top=4)]
    assert "orange" in names                        # another round, sweet, edible fruit
    assert "car" not in names


def test_similarity_zero_for_ungrounded():
    lex = GroundedLexicon()
    assert lex.similarity("apple", "zorblax") == 0.0


# -------------------- hyperdimensional geometry preserves the relations -------------- #
def test_hd_encoding_preserves_apple_orange_over_car():
    lex = GroundedLexicon()
    hd_fruit = lex.hd_similarity("apple", "orange")
    hd_car = lex.hd_similarity("apple", "car")
    assert hd_fruit > hd_car                        # same relation survives in 10,000-D


# -------------------- grounded perception feeds the taxonomy -------------------- #
def test_to_instance_and_abstraction_ladder_generalize():
    from nyxara.cognition.concept_formation import AbstractionLadder

    lex = GroundedLexicon()
    # include a text-learned fruit (no explicit weight) so the shared structure is the
    # genuine fruit commonality {edible, gravity}, not the seed-only weight feature.
    lex.learn_from_text("A mango is a fruit. Mangoes are sweet and yellow. "
                        "You can eat a mango and a mango falls.")
    ladder = AbstractionLadder()
    members = ["apple", "orange", "banana", "lemon", "mango"]
    assert lex.feed_abstraction_ladder(ladder, members) == 5
    concept = ladder.abstract(members)
    assert "affordance:edible" in concept.features
    assert "physics:gravity" in concept.features
    # a never-abstracted grape is recognised as the fruit-like concept by its grounding
    hit = ladder.classify(lex.feature_set("grape"))
    assert hit is not None and hit[1].name == concept.name


def test_to_instance_shape():
    lex = GroundedLexicon()
    inst = lex.to_instance("apple")
    assert inst is not None
    assert inst.name == "apple"
    assert "taste:sweet" in inst.features
    assert "affordance:edible" in inst.features


# -------------------- offline floor & introspection -------------------- #
def test_seed_lexicon_is_real_and_offline():
    lex = GroundedLexicon()                          # no llm, no network
    assert len(lex) >= 30
    # a spread of anchor concepts across domains is actually grounded
    for w in ("apple", "fire", "water", "dog", "car", "stone"):
        assert lex.activate(w).grounded


def test_descriptor_ontology_maps_qualities_to_senses():
    assert DESCRIPTOR_ONTOLOGY["sweet"][0] is Sense.TASTE
    assert DESCRIPTOR_ONTOLOGY["red"][0] is Sense.VISION
    assert DESCRIPTOR_ONTOLOGY["heavy"][0] is Sense.PHYSICS
    assert DESCRIPTOR_ONTOLOGY["light"][2] < 0       # bipolar opposite pole


def test_explain_and_reports_are_introspectable():
    lex = GroundedLexicon()
    text = lex.explain("apple")
    assert "apple" in text and ("sweet" in text or "edible" in text)
    assert "concepts" in lex.report()
    d = lex.to_dict()
    assert "apple" in d["concepts"] and d["concepts"]["apple"]["features"] > 0


def test_running_mean_averages_repeated_evidence():
    c = GroundedConcept(name="x")
    c.reinforce(Sense.TASTE, "sweet", 1.0)
    c.reinforce(Sense.TASTE, "sweet", 0.0)
    assert abs(c.senses[Sense.TASTE]["sweet"] - 0.5) < 1e-9


# -------------------- LLM ceiling degrades to the floor -------------------- #
def test_llm_ceiling_degrades_on_error():
    class _BoomLLM:
        def chosen_provider(self):
            class P:  # noqa: D401
                name = "real"
            return P()

        def generate_json(self, *a, **k):
            raise RuntimeError("no network")

    lex = GroundedLexicon(seed=False, llm=_BoomLLM())
    out = lex.learn_from_llm("dragonfruit")
    assert out["features_added"] == 0                # failed cleanly
    # and activation of an unknown word doesn't raise even with a broken ceiling
    assert lex.activate("dragonfruit").grounded is False


def test_mock_llm_is_not_used():
    class _MockLLM:
        def chosen_provider(self):
            class P:
                name = "native"
            return P()

    lex = GroundedLexicon(llm=_MockLLM())
    assert lex._use_llm() is False                   # mock never counts as a real ceiling


def test_learn_from_percepts_binds_features():
    lex = GroundedLexicon(seed=False)

    class _Percept:
        def __init__(self, content, modality="image", tags=None):
            self.content = content
            self.modality = modality
            self.tags = tags or []

    rep = lex.learn_from_percepts(
        "apple", [_Percept("a red round shiny thing", tags=["sweet"])])
    assert rep["features_added"] > 0
    assert lex.activate("apple").grounded


# -------------------- experience → meaning: grounding from perception -------------------- #
class _FramePercept:
    """A minimal stand-in for senses.binding.Percept (content/modality/entities/tags/data)."""

    def __init__(self, content, modality="image", entities=None, tags=None, data=None):
        self.content = content
        self.modality = modality
        self.entities = entities or []
        self.tags = tags or []
        self.data = data or {}


class _Frame:
    """A minimal stand-in for senses.binding.PerceptualFrame (percepts() is a method)."""

    def __init__(self, percepts):
        self._percepts = list(percepts)

    def percepts(self):
        return list(self._percepts)


def test_learn_from_frame_grounds_entities_from_perception():
    lex = GroundedLexicon(seed=False)               # nothing known: meaning purely perceived
    frame = _Frame([
        _FramePercept("a red round shiny thing", entities=["apple"]),
        _FramePercept("a big loud animal", modality="audio", entities=["truck"], tags=["loud"]),
    ])
    rep = lex.learn_from_frame(frame)
    assert "apple" in rep["grounded"] and "truck" in rep["grounded"]
    assert rep["via"] == "perception"
    # the apple was SEEN: colour/shape ground vision, and it activates vision
    apple = lex.activate("apple")
    assert apple.grounded and apple.fires(Sense.VISION)
    # the truck was HEARD: loudness grounds sound, and it activates sound
    truck = lex.activate("truck")
    assert truck.grounded and truck.fires(Sense.SOUND)


def test_learn_from_frame_channel_records_the_perceiving_sense():
    lex = GroundedLexicon(seed=False)
    # a percept with NO descriptor words at all — pure channel evidence
    lex.learn_from_frame(_FramePercept("", modality="image", entities=["blob"]))
    blob = lex.get("blob")
    assert blob is not None
    assert blob.senses[Sense.VISION].get("perceived", 0) > 0   # "I saw it" is itself grounding


def test_learn_from_frame_accepts_a_single_percept():
    lex = GroundedLexicon(seed=False)
    rep = lex.learn_from_frame(
        _FramePercept("a sweet soft yellow fruit", entities=["banana"]))
    assert "banana" in rep["grounded"]
    b = lex.get("banana")
    assert b.senses[Sense.TASTE].get("sweet", 0) > 0
    assert b.senses[Sense.TOUCH].get("soft", 0) > 0


def test_learn_from_frame_skips_percepts_without_entities():
    lex = GroundedLexicon(seed=False)
    rep = lex.learn_from_frame(_Frame([_FramePercept("red and round", entities=[])]))
    assert rep["grounded"] == [] and rep["features_added"] == 0


def test_pronoun_carries_the_subject_across_sentences():
    lex = GroundedLexicon(seed=False)
    lex.learn_from_text("A tiger is big. It is dangerous and fast.")
    tiger = lex.get("tiger")
    assert tiger is not None
    assert tiger.senses[Sense.VISION].get("size", 0) > 0        # "big" from sentence 1
    assert tiger.senses[Sense.EMOTION].get("fear", 0) > 0       # "dangerous" via "It"
    assert tiger.senses[Sense.ACTION].get("fast", 0) > 0        # "fast" via "It"
