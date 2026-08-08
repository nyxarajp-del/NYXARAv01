"""NYX V.02 · the semantic space — three rungs, and every answer naming the one it came from.

The measurement this file exists to invert: in V.01's embedder
``cos("car", "automobile") == 0.0`` while ``cos("car", "banana") == 0.27``. A synonym scored
*below* a random noun. The other half of the file pins the honesty rule that comes with
shipping no data — an untrained pair must report *I do not know*, never *unrelated*.
"""

from __future__ import annotations

import pytest

from nyxara.nyx.graph import DynamicNeuralGraph
from nyxara.nyx.semantics import RUNGS, SemanticSpace, Similarity

CAR_CORPUS = [
    "the car drove down the road quickly",
    "the automobile drove down the road quickly",
    "a car needs fuel and a driver",
    "an automobile needs fuel and a driver",
    "the car parked near the house",
    "the automobile parked near the house",
    "she ate a banana for breakfast",
    "the banana was ripe and sweet",
]


def _space(**kw) -> SemanticSpace:
    kw.setdefault("min_count", 2)
    kw.setdefault("train_every", 10_000)      # training is explicit in tests, never incidental
    return SemanticSpace(**kw)


@pytest.fixture(scope="module")
def read_space() -> SemanticSpace:
    """A space that has actually read something — one real SGNS pass, shared by the tests."""
    space = SemanticSpace(min_count=2, train_every=10_000, seed=7)
    for _ in range(4):
        for line in CAR_CORPUS:
            space.train_on(line, train=False)
    space.consolidate(budget_s=3.0)
    return space


# --------------------------------------------------------------------------- #
# The honesty rule: zero from an empty space is ignorance, not a measurement
# --------------------------------------------------------------------------- #
def test_an_untrained_pair_is_unknown_not_unrelated():
    sim = _space().similarity("car", "automobile")
    assert sim.known is False
    assert "not read" in sim.why


def test_describe_says_ignorance_out_loud():
    said = _space().describe("car", "automobile")
    assert "ignorance, not a measurement of zero" in said


def test_similarity_unpacks_as_score_rung_grade():
    score, rung, grade = _space().similarity("car", "cat")
    assert isinstance(score, float) and rung in RUNGS and 0.0 <= grade <= 1.0


def test_identity_is_the_one_free_answer():
    sim = _space().similarity("car", "car")
    assert sim.score == 1.0 and sim.known and sim.rung == "relational"


# --------------------------------------------------------------------------- #
# Rung 3 — subword: spelling, graded as spelling
# --------------------------------------------------------------------------- #
def test_subword_rung_catches_morphology():
    space = _space()
    for a, b in (("cat", "cats"), ("run", "running"), ("engine", "engines")):
        sim = space.similarity(a, b)
        assert sim.rung == "subword" and sim.known, f"{a}/{b}"
        assert "form, not meaning" in sim.why


def test_subword_rung_is_capped_because_spelling_is_not_meaning():
    """Even a perfect spelling match may not certify a claim about meaning."""
    sim = _space().similarity("cat", "cats")
    assert sim.grade <= 0.4


def test_subword_rung_does_not_invent_a_relation_from_a_shared_prefix():
    assert _space().similarity("part", "party").known is False
    assert _space().similarity("car", "cart").known is False


def test_subword_rung_is_script_agnostic():
    """The transliteration bridge, before any learning at all."""
    sim = _space().similarity("गुरुत्वाकर्षण", "gurutvakarshan")
    assert sim.score == pytest.approx(1.0)
    assert "two scripts" in sim.why


def test_a_synonym_is_not_reachable_by_spelling_and_says_so():
    sim = _space().similarity("car", "automobile")
    assert sim.rung == "subword" and sim.score == 0.0 and sim.known is False


# --------------------------------------------------------------------------- #
# Rung 1 — relational: what she was told, or read
# --------------------------------------------------------------------------- #
def test_being_told_is_the_strongest_evidence():
    space = _space()
    assert space.teach("car", "automobile")
    sim = space.similarity("car", "automobile")
    assert sim.rung == "relational" and sim.known and sim.score >= 0.85


def test_relations_are_symmetric():
    space = _space()
    space.teach("car", "automobile")
    assert space.similarity("automobile", "car").known


def test_she_reads_a_definition_out_of_a_sentence():
    space = _space()
    space.train_on("a car, an automobile, drove past the house", train=False)
    assert ("automobile", pytest.approx(0.85)) in [
        (label, pytest.approx(weight)) for label, weight in space.synonyms("car")]


def test_she_reads_a_hinglish_definition():
    space = _space()
    space.train_on("gaadi ka matlab car", train=False)
    assert space.similarity("gaadi", "car").rung == "relational"


def test_she_reads_a_devanagari_definition():
    space = _space()
    space.train_on("गाड़ी का मतलब कार", train=False)
    assert space.similarity("गाड़ी", "कार").known


def test_a_passing_mention_is_not_a_definition():
    space = _space()
    space.train_on("an apple falls because of gravity", train=False)
    assert space.synonyms("apple") == []


def test_grammar_never_becomes_a_relation():
    space = _space()
    space.train_on("this is the thing", train=False)
    assert space.relations("this") == []


def test_is_a_relates_without_equating():
    space = _space()
    space.teach("sparrow", "bird", kind="is-a", weight=0.6)
    assert space.synonyms("sparrow") == []          # a sparrow is not a synonym for bird
    assert space.similarity("sparrow", "bird").known


def test_the_transliteration_bridge_links_two_spellings_she_has_met():
    space = _space()
    space.train_on("गुरुत्वाकर्षण सेब को खींचता है", train=False)
    space.train_on("gurutvakarshan ek force hai", train=False)
    assert ("gurutvakarshan", pytest.approx(0.95)) in [
        (label, pytest.approx(weight)) for label, weight in space.synonyms("गुरुत्वाकर्षण")]


# --------------------------------------------------------------------------- #
# Rung 2 — distributional: real SGNS on her own corpus
# --------------------------------------------------------------------------- #
def test_reading_inverts_the_measured_v01_failure(read_space):
    """cos(car, automobile) must beat cos(car, banana). In V.01 it was 0.0 against 0.27."""
    synonym = read_space.similarity("car", "automobile")
    random_noun = read_space.similarity("car", "banana")
    assert synonym.rung == "distributional" and synonym.known
    assert synonym.score > random_noun.score
    assert synonym.score > 0.5


def test_the_distributional_rung_names_its_evidence(read_space):
    assert "her own reading" in read_space.similarity("car", "automobile").why


def test_similar_returns_the_rung_that_justified_each_neighbour(read_space):
    near = read_space.similar("car", k=3)
    assert near and near[0][0] == "automobile"
    assert all(rung in RUNGS for _label, _score, rung in near)


def test_vector_reports_which_rung_it_came_from(read_space):
    assert read_space.vector_rung("car") == "distributional"
    assert read_space.vector_rung("zzquux") == "subword"
    assert len(read_space.vector("car")) > 0


def test_an_unread_word_borrows_a_proven_synonym_vector(read_space):
    space = SemanticSpace(min_count=2, train_every=10_000)
    space.load_dict(read_space.to_dict())
    space.teach("gaadi", "car")
    assert space.vector_rung("gaadi") == "relational"
    assert space.vector("gaadi") == space.vector("car")


def test_stats_say_how_thin_the_evidence_is(read_space):
    stats = read_space.stats()
    assert stats["trained"] is True and stats["vocabulary"] > 0
    assert stats["rungs"]["distributional"] is True
    assert "no lexicon and no pretrained weights ship" in stats["note"]


def test_an_unread_space_admits_two_of_its_rungs_are_empty():
    rungs = _space().rungs_available()
    assert rungs == {"relational": False, "distributional": False, "subword": True}


# --------------------------------------------------------------------------- #
# The graph hook — a semantic prior over structure
# --------------------------------------------------------------------------- #
def _graph(space) -> DynamicNeuralGraph:
    g = DynamicNeuralGraph(max_nodes=64, max_edges=256, decay_rate=0.0)
    g.attach_semantics(space, birth_links=2, birth_weight=0.2, synonym_bridge=0.5)
    return g


def test_a_newborn_concept_is_wired_to_what_it_means():
    space = _space()
    space.teach("car", "automobile")
    g = _graph(space)
    g.activate("car road")
    g.activate("automobile")                 # born now, but not co-activated with "car"
    assert g.weight("car", "automobile") > 0.0


def test_the_semantic_prior_never_outweighs_a_lived_edge():
    space = _space()
    space.teach("car", "automobile")
    g = _graph(space)
    g.activate("car road")
    g.activate("automobile")
    assert g.weight("car", "road") > g.weight("car", "automobile")


def test_activation_crosses_a_synonym_boundary():
    space = _space()
    space.teach("car", "automobile")
    g = _graph(space)
    g.activate("automobile parked garage")
    g.activate("car road")
    reached = dict(g._spread(["car"]))
    assert "parked" in reached or "automobile" in reached


def test_neighbours_can_be_asked_for_meaning_as_well_as_association():
    space = _space()
    space.teach("car", "automobile")
    g = _graph(space)
    g.activate("car road")
    g.activate("automobile")
    assert "automobile" in dict(g.neighbours("car", semantic=True))


def test_a_graph_without_a_space_behaves_exactly_as_v01():
    g = DynamicNeuralGraph(max_nodes=64, max_edges=256, decay_rate=0.0)
    g.activate("car road")
    g.activate("automobile")
    assert g.weight("car", "automobile") == 0.0
    assert g.neighbours("car", semantic=True) == g.neighbours("car")


# --------------------------------------------------------------------------- #
# Node merge — bounded, ledgered, reversible
# --------------------------------------------------------------------------- #
def test_proven_synonyms_can_be_fused():
    space = _space()
    space.teach("car", "automobile", weight=0.95)
    g = _graph(space)
    g.activate("car road fuel")
    g.activate("automobile parked")
    assert g.auto_merge(limit=1)
    assert ("car" in g) != ("automobile" in g)     # exactly one label survives


def test_a_merge_folds_the_edges_rather_than_dropping_them():
    space = _space()
    space.teach("car", "automobile", weight=0.95)
    g = _graph(space)
    g.activate("car road")
    g.activate("automobile parked")
    g.auto_merge(limit=1)
    survivor = "car" if "car" in g else "automobile"
    assert g.weight(survivor, "road") > 0.0 and g.weight(survivor, "parked") > 0.0


def test_every_merge_is_in_the_ledger_and_reversible():
    space = _space()
    space.teach("car", "automobile", weight=0.95)
    g = _graph(space)
    g.activate("car road")
    g.activate("automobile parked")
    g.auto_merge(limit=1)
    assert g.merges and g.merges[-1]["reason"].startswith("synonym")
    assert g.unmerge_last()
    assert "car" in g and "automobile" in g
    assert g.weight("automobile", "parked") > 0.0


def test_merging_never_happens_on_distributional_closeness_alone(read_space):
    """'hot' sits next to 'cold'. Fusing on adjacency is a structural error she cannot see."""
    g = _graph(read_space)
    g.activate("the car drove down the road")
    g.activate("the automobile drove down the road")
    assert g.auto_merge(limit=2) == []          # nothing was *taught*, so nothing is fused


def test_merge_is_budgeted_per_call():
    space = _space()
    space.teach("a1", "a2", weight=0.95)
    space.teach("b1", "b2", weight=0.95)
    g = _graph(space)
    g.activate("a1 a2 b1 b2")
    assert len(g.auto_merge(limit=1)) == 1


# --------------------------------------------------------------------------- #
# Fail-soft and persistence
# --------------------------------------------------------------------------- #
def test_everything_is_fail_soft_on_junk():
    space = _space()
    assert space.similarity(None, None).known is False
    assert space.similar("") == []
    assert space.teach("", "") is False
    assert space.train_on(None)["tokens"] == 0
    assert space.read_relations(None) == 0
    assert space.synonyms(None) == []


def test_round_trips_through_a_sidecar():
    space = _space()
    space.teach("car", "automobile")
    space.train_on("the car drove past", train=False)
    other = _space()
    assert other.load_dict(space.to_dict())
    assert other.similarity("car", "automobile").known
    assert other.stats()["vocabulary"] == space.stats()["vocabulary"]


def test_a_corrupt_sidecar_never_raises():
    space = _space()
    assert space.load_dict(None) is False
    assert space.load_dict({"relations": [{"a": None}], "counts": "junk"}) is False


def test_relations_are_capped():
    space = SemanticSpace(max_relations=16, train_every=10_000)
    for i in range(40):
        space.teach(f"w{i}", f"v{i}")
    assert sum(len(v) for v in space._relations.values()) <= 16


def test_similarity_serialises():
    got = _space().similarity("car", "automobile").to_dict()
    assert set(got) == {"a", "b", "score", "rung", "grade", "known", "why"}
    assert isinstance(Similarity().to_dict(), dict)
