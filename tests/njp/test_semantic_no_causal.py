"""Similarity may propose a concept. It may never establish a cause.

``njp/relevance.py`` was written after a reproduced failure: asked "How are you NYXARA?" she
returned a verified pendulum-period law and raised her confidence to 1.00. Every stage worked —
nothing ever asked whether a true thing had anything to do with the question. Teacher embeddings
are a larger version of the same trap, because proximity in a semantic space is genuinely
informative and reads exactly like a relation. These tests hold that line.
"""

from __future__ import annotations

from nyxara.njp.adversary import SelfAttacker
from nyxara.njp.concepts import ConceptGenesis
from nyxara.njp.shadow import GapKind, ShadowCognition
from nyxara.njp.universe import InternalUniverse


def test_a_semantic_gap_routes_to_concepts_and_nowhere_else():
    universe = InternalUniverse()
    shadow = ShadowCognition(concepts=ConceptGenesis(), universe=universe,
                             adversary=SelfAttacker())
    before = len(getattr(universe, "variables", {}) or {})
    reading = shadow.compare(
        "tell me about wolves",
        teacher_text="Wolves and foxes are canids, like dogs. Lattice, entropy, algebra.",
        njp_text="wolves are animals")
    semantic = reading.by_kind(GapKind.SEMANTIC)
    assert semantic, "words the teacher used and she did not must be found"
    for gap in semantic:
        assert gap.routed_to == "njp.concepts"
    # Co-occurring words did NOT become variables in the causal model.
    assert len(getattr(universe, "variables", {}) or {}) == before


def test_co_occurrence_never_produces_a_causal_gap():
    """The teacher listed words together. Listing is not causing, and no arrow may appear."""
    shadow = ShadowCognition(concepts=ConceptGenesis(), adversary=SelfAttacker())
    reading = shadow.compare("q", teacher_text="dog wolf fox canid lattice entropy", njp_text="")
    assert reading.by_kind(GapKind.CAUSAL) == []
    assert reading.claims_challenged == 0


def test_only_an_explicit_causal_sentence_becomes_a_claim_and_it_still_gets_attacked():
    shadow = ShadowCognition(concepts=ConceptGenesis(), adversary=SelfAttacker())
    reading = shadow.compare("q", teacher_text="Heat causes expansion.", njp_text="")
    causal = reading.by_kind(GapKind.CAUSAL)
    assert len(causal) == 1
    assert causal[0].subject == "heat -> expansion"
    # Challenged against her own record, and unsettled on an empty one.
    assert reading.claims_challenged == 1
    assert reading.claims_undecided == 1


def test_concepts_are_offered_as_observations_carrying_no_relation():
    """``ConceptGenesis`` decides for itself whether a kind is there and pays for it in
    description length. The teacher supplies words, never structure."""
    genesis = ConceptGenesis()
    shadow = ShadowCognition(concepts=genesis, adversary=SelfAttacker())
    reading = shadow.compare("q", teacher_text="wolves foxes and dogs are canids", njp_text="")
    gaps = reading.by_kind(GapKind.SEMANTIC)
    assert gaps and gaps[0].accepted
    assert "observation" in gaps[0].detail
