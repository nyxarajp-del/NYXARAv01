"""Concept Genesis: kinds invented from observations, and the compression that proves it.

Every test here asserts a *number that could have come out otherwise*. A concept system is easy
to write and easy to fool yourself about — it will always produce clusters — so the tests are
built around the two things that distinguish understanding from grouping: the description-length
win, and the ability to say something about a member that was never observed of it.
"""

from __future__ import annotations

from nyxara.njp.concepts import ConceptGenesis

ANIMALS = {
    "dog": ["living", "is_a:animal", "has:legs", "makes:sound"],
    "cat": ["living", "is_a:animal", "has:legs", "makes:sound"],
    "wolf": ["living", "is_a:animal", "has:legs", "makes:sound"],
}
PLANTS = {
    "oak": ["living", "is_a:plant", "has:roots", "needs:water"],
    "rose": ["living", "is_a:plant", "has:roots", "needs:water"],
    "fern": ["living", "is_a:plant", "has:roots", "needs:water"],
}


def _seed(genesis: ConceptGenesis, *tables: dict) -> ConceptGenesis:
    for table in tables:
        for subject, features in table.items():
            genesis.observe(subject, features)
    genesis.crystallise()
    return genesis


def test_concepts_are_invented_not_given():
    genesis = _seed(ConceptGenesis(), ANIMALS, PLANTS)
    assert genesis.concepts, "no concept formed from six clearly-kinded observations"
    # Nobody wrote "animal" anywhere in this module's inputs as a *concept* — it is a feature of
    # individuals. The concept is the thing that noticed several individuals share it.
    invariant_sets = [c.invariants for c in genesis.concepts.values()]
    assert any("is_a:animal" in s for s in invariant_sets)
    assert any("is_a:plant" in s for s in invariant_sets)


def test_compression_is_a_real_description_length_win():
    genesis = _seed(ConceptGenesis(), ANIMALS, PLANTS)
    # Six observations of four features each cost 24 symbols raw. If the concepts do not store
    # that in fewer, they have grouped without understanding.
    assert genesis.compression() > 1.5, genesis.stats()


def test_one_observation_compresses_nothing():
    """The guard against a concept per observation, which trivially 'explains' everything."""
    genesis = ConceptGenesis()
    genesis.observe("dog", ["living", "is_a:animal"])
    genesis.crystallise()
    assert genesis.compression() <= 1.0 + 1e-9


def test_hierarchy_emerges_by_subsumption():
    genesis = _seed(ConceptGenesis(), ANIMALS, PLANTS)
    parents = [c for c in genesis.concepts.values() if c.children]
    assert parents, "no concept subsumed another: the taxonomy is flat"
    parent = parents[0]
    for child_id in parent.children:
        child = genesis.concepts[child_id]
        # The defining property of the relation: the specific asserts everything the general
        # does, and more. Anything else is not a taxonomy.
        assert parent.invariants < child.invariants
        assert genesis.ancestors(child_id)[0] == parent.cid


def test_generalisation_says_something_never_observed():
    """The whole point of an abstraction — transfer to a member it was not learned from.

    Note what has to happen first. ``has:legs`` is *invariant* over the three animals, so a
    fourth animal lacking it does not satisfy the concept at all and is owed no inference. It is
    the restructure — recognising that the concept was over-claiming — that turns ``has:legs``
    from a claim into a typical feature, and only then is transferring it honest.
    """
    genesis = ConceptGenesis()
    for subject, features in ANIMALS.items():
        genesis.observe(subject, features)
    genesis.observe("lion", ["living", "is_a:animal", "makes:sound"])
    genesis.crystallise()

    coverage = genesis.explain("lion")
    assert coverage.gap == "violates" and "has:legs" in coverage.violated
    genesis.restructure(coverage)

    transferred = genesis.generalise("lion")
    assert "has:legs" in transferred, transferred


def test_unknown_and_violates_are_different_gaps():
    genesis = _seed(ConceptGenesis(), ANIMALS)
    # Nothing in common with anything: she has no kind for this.
    assert genesis.explain("quasar", ["emits:radio"]).gap == "unknown"
    # Nearly an animal, but breaks the concept's own claim.
    near = genesis.explain("snake", ["living", "is_a:animal", "makes:sound"])
    assert near.gap == "violates"
    assert "has:legs" in near.violated


def test_restructure_closes_a_violated_boundary_without_costing_compression():
    genesis = ConceptGenesis()
    for subject, features in ANIMALS.items():
        genesis.observe(subject, features)
    genesis.observe("horse", ["living", "is_a:animal", "has:legs"])   # no sound
    genesis.crystallise()

    coverage = genesis.explain("horse")
    assert coverage.gap == "violates"
    before = genesis.compression()

    genesis.restructure(coverage)

    assert genesis.explain("horse").covered, "the restructure did not actually cover the member"
    assert genesis.compression() >= before, "the repair cost compression elsewhere"


def test_restructure_refuses_a_generalisation_that_buys_nothing():
    """Loosening to swallow one outlier must not be accepted on no evidence."""
    genesis = ConceptGenesis()
    for subject, features in ANIMALS.items():
        genesis.observe(subject, features)
    genesis.crystallise()
    before_similarity = genesis.similarity
    before_ratio = genesis.compression()

    genesis.restructure(genesis.explain("quasar", ["emits:radio"]))

    assert genesis.compression() >= before_ratio
    # Whatever it settled on, it may not have walked the threshold outside its band.
    assert genesis.similarity >= 0.5 * before_similarity


def test_repairs_cannot_accumulate_into_an_unbounded_drift():
    """The failure this guard exists for: a run of outliers walking the threshold to the floor."""
    genesis = ConceptGenesis(similarity=0.34)
    for subject, features in ANIMALS.items():
        genesis.observe(subject, features)
    genesis.crystallise()
    for i in range(40):
        genesis.restructure(genesis.explain(f"alien{i}", [f"weird:{i}"]))
    assert genesis.similarity >= 0.5 * 0.34 - 1e-9, genesis.similarity


def test_triples_are_concept_forming_evidence():
    class _T:
        def __init__(self, s, p, o):
            self.subject, self.predicate, self.object = s, p, o

    genesis = ConceptGenesis()
    fed = genesis.observe_triples([_T("dog", "is_a", "animal"), _T("cat", "is_a", "animal")])
    assert fed == 2
    assert "is_a:animal" in genesis._by_subject["dog"].features


def test_a_numeric_object_becomes_a_quantity_not_a_category():
    """"boils:100" would make water and mercury unrelated kinds; what they share is HAVING one."""
    class _T:
        def __init__(self, s, p, o):
            self.subject, self.predicate, self.object = s, p, o

    genesis = ConceptGenesis()
    genesis.observe_triples([_T("water", "boils", "100"), _T("mercury", "boils", "357")])
    assert genesis._by_subject["water"].numbers == {"boils": 100.0}
    assert "boils" in genesis._by_subject["water"].features
    assert "boils:100" not in genesis._by_subject["water"].features


def test_state_survives_a_round_trip_and_is_rebuilt_from_observations():
    genesis = _seed(ConceptGenesis(), ANIMALS, PLANTS)
    before = genesis.compression()

    revived = ConceptGenesis()
    revived.load_dict(genesis.to_dict())

    assert len(revived.observations) == len(genesis.observations)
    assert abs(revived.compression() - before) < 1e-6
