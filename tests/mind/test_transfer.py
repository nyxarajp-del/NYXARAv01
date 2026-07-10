"""Tests for nyxara.mind.transfer — cross-domain generalization by her OWN faculties.

The relational-transfer engine generalizes a new-domain query by structure-mapping it onto a
domain NYXARA already understands and projecting the known structure across — with no LLM
anywhere. These tests assert the projection is genuine (a held-out inference is recovered), that
it declines honestly when no structure maps, and that a learned domain transfers next time.
"""

from __future__ import annotations

from nyxara.mind.analogy import entity, relation
from nyxara.mind.transfer import (
    DomainSchema,
    DomainSchemaStore,
    RelationalTransferEngine,
    TransferResult,
    seed_library,
)


def _atom():
    e, r = entity, relation
    nucleus, electron = e("nucleus"), e("electron")
    return [r("attracts", nucleus, electron), r("revolves_around", electron, nucleus)]


def test_projects_held_out_higher_order_inference():
    """The atom is told only the two surface relations; the engine must project the deeper
    CAUSE relation by mapping from the solar system — the predictive payoff of analogy."""
    eng = RelationalTransferEngine()
    tr = eng.generalize("model of the atom", target_relations=_atom())
    assert isinstance(tr, TransferResult)
    assert tr.base_domain == "solar_system"
    assert tr.entity_mapping.get("sun") == "nucleus"
    assert tr.entity_mapping.get("planet") == "electron"
    inferred = " ".join(str(p) for p in tr.candidate_inferences)
    assert "causes" in inferred and "nucleus" in inferred and "electron" in inferred


def test_declines_when_no_structure_maps():
    """No extractable relational structure -> return None so the LLM path runs (no faking)."""
    eng = RelationalTransferEngine()
    assert eng.generalize("hi, how are you today?") is None
    assert eng.generalize("") is None


def test_min_score_gates_weak_mappings():
    """Raising the threshold above what any base can support forces an honest decline."""
    eng = RelationalTransferEngine(min_score=999.0)
    assert eng.generalize("atom", target_relations=_atom()) is None


def test_learned_domain_transfers_next_time():
    """A domain learned from lived structure is available to transfer from afterwards."""
    e, r = entity, relation
    eng = RelationalTransferEngine()
    spring, mass = e("spring"), e("mass")
    eng.learn_domain(
        "harmonic_oscillator",
        [r("pulls", spring, mass), r("oscillates", mass, spring),
         r("causes", r("pulls", spring, mass), r("oscillates", mass, spring))],
        keywords=["pendulum"],
    )
    lc, charge = e("inductor"), e("charge")
    tr = eng.generalize("an LC circuit",
                        target_relations=[r("pulls", lc, charge), r("oscillates", charge, lc)])
    assert tr is not None and tr.base_domain == "harmonic_oscillator"


def test_text_extraction_recovers_relations():
    """The conservative text extractor recovers relation(entity,entity) facts from prose using
    the store's known relation vocabulary, enough to drive a transfer without structured input."""
    eng = RelationalTransferEngine()
    tr = eng.generalize("the predator consumes prey and the predator reduces prey sharply")
    assert tr is not None
    assert tr.base_domain == "predator_prey"


def test_store_learn_merges_and_grows():
    store = DomainSchemaStore()
    n0 = len(store)
    e, r = entity, relation
    store.learn("d", [r("a", e("x"), e("y"))], keywords=["k"])
    assert len(store) == n0 + 1
    store.learn("d", [r("b", e("x"), e("z"))])   # merge, not clobber
    d = store.get("d")
    assert d is not None and len(d.relations) == 2


def test_seed_library_is_nonempty_and_structured():
    lib = seed_library()
    assert len(lib) >= 3
    for schema in lib:
        assert isinstance(schema, DomainSchema)
        assert schema.relations, "each seeded domain has relational structure"
        # every seed carries at least one higher-order (causal) relation to project
        assert any(any(hasattr(a, "args") for a in rel.args) for rel in schema.relations)


def test_render_is_honest_hypothesis_language():
    eng = RelationalTransferEngine()
    tr = eng.generalize("atom", target_relations=_atom())
    text = tr.render().lower()
    assert "analogy" in text
    assert "likely" in text or "hypothes" in text   # hedged, not asserted as fact
