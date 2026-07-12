"""Tests for nyxara.mind.domain_genesis — domain mastery FROM SCRATCH, by her OWN faculties.

The genesis engine models a genuinely alien field from its OWN internal structure (no known
base to borrow from, no LLM): it extracts the field's relations, induces the laws that hold
within it (transitivity, feedback loops, symmetry, inverse), and projects held-out predictions
by closure over those laws. These tests assert the induction is genuine (a held-out fact the
prompt never stated is recovered), that it declines honestly when there is no structure, and
that a modelled field is learned so it is recognised next time.
"""

from __future__ import annotations

from nyxara.mind.domain_genesis import DomainGenesisEngine, DomainLaw, DomainTheory
from nyxara.mind.transfer import RelationalTransferEngine


# An alien domain: fresh verbs, and a topology (a closed 3-cycle with a chained relation) that
# is described entirely in its own vocabulary.
_ALIEN = ("in this device the glorp zephs the drine and the drine zephs the quon "
          "and the quon flumps the glorp")


def test_extracts_a_relational_skeleton_from_alien_prose():
    theory = DomainGenesisEngine().build(_ALIEN)
    assert theory is not None
    rels = {str(p) for p in theory.relations}
    assert "zeph(glorp, drine)" in rels
    assert "zeph(drine, quon)" in rels
    assert "flump(quon, glorp)" in rels


def test_induces_transitivity_and_projects_the_held_out_fact():
    """zeph chains glorp→drine→quon, so zeph(glorp, quon) must be projected — a fact the prompt
    never stated, derived from the domain's own transitivity law."""
    theory = DomainGenesisEngine().build(_ALIEN)
    assert theory is not None
    kinds = {law.kind for law in theory.laws}
    assert "transitivity" in kinds
    proj = {str(p) for p in theory.inferences}
    assert "zeph(glorp, quon)" in proj


def test_induces_the_self_sustaining_feedback_loop():
    """glorp→drine→quon→glorp closes a cycle — a self-sustaining loop whose higher-order closure
    (the systematic glue the seed schemas carry) is projected, induced from the field itself."""
    theory = DomainGenesisEngine().build(_ALIEN)
    assert theory is not None
    assert any(law.kind == "feedback_loop" for law in theory.laws)
    assert any(str(p).startswith("sustains(") for p in theory.inferences)


def test_induces_symmetry_and_projects_the_mirror():
    """R(a,b) ∧ R(b,a) ⇒ R is symmetric; the mirror of an unmatched R-edge is then projected."""
    q = ("the arda meshes the boro and the boro meshes the arda "
         "and the arda meshes the cane")
    theory = DomainGenesisEngine().build(q)
    assert theory is not None
    assert any(law.kind == "symmetry" for law in theory.laws)
    # mesh(arda, cane) is stated but its mirror mesh(cane, arda) is not — so it is projected
    assert any(str(p) == "mesh(cane, arda)" for p in theory.inferences)


def test_generalize_answers_from_the_induced_theory():
    res = DomainGenesisEngine().generalize(_ALIEN)
    assert res is not None
    assert res.confidence >= 0.4
    # a moderate, honest confidence — always below a verified faculty
    assert res.confidence <= 0.66
    text = res.render()
    assert "from its own internal structure" in text
    # the projected held-out facts appear in the rendered answer
    assert "glorp" in text and "quon" in text


def test_declines_honestly_on_unstructured_chat():
    eng = DomainGenesisEngine()
    assert eng.generalize("hi how are you today") is None
    assert eng.build("hi how are you today") is None


def test_declines_when_too_little_structure():
    # a single relation is not enough to induce a law
    eng = DomainGenesisEngine(min_relations=2)
    assert eng.generalize("the widget frobs the gadget") is None


def test_masters_the_field_learns_it_into_the_shared_store():
    """Rule 4 growth: with a transfer engine, a field modelled from scratch is distilled into a
    DomainSchema and learned, so it is recognised (and transferable) next time."""
    shared = RelationalTransferEngine()
    eng = DomainGenesisEngine(transfer_engine=shared)
    res = eng.generalize(_ALIEN)
    assert res is not None
    label = res.theory.label
    assert shared.store.get(label) is not None, "the modelled field must enter the store"
    # the learned schema carries the field's own relations
    schema = shared.store.get(label)
    assert any(r.name in {"zeph", "flump"} for r in schema.relations)


def test_learning_can_be_disabled():
    shared = RelationalTransferEngine()
    eng = DomainGenesisEngine(transfer_engine=shared, learn_from_experience=False)
    res = eng.generalize(_ALIEN)
    assert res is not None
    assert shared.store.get(res.theory.label) is None


def test_multi_hop_transitive_closure_to_a_fixpoint():
    """Transitivity is iterated to a fixpoint: a→b→c→d yields the whole chain, not just one hop."""
    q = ("the alpha zorbs the beta and the beta zorbs the gamma "
         "and the gamma zorbs the delta")
    theory = DomainGenesisEngine().build(q)
    assert theory is not None
    proj = {str(p) for p in theory.inferences}
    assert "zorb(alpha, gamma)" in proj        # one hop
    assert "zorb(alpha, delta)" in proj        # two hops — only a fixpoint recovers this
    assert "zorb(beta, delta)" in proj


def test_induces_a_composition_law():
    """R(a,b) ∧ S(b,c) ∧ T(a,c) ⇒ T = R∘S, induced from the field's own structure."""
    q = ("the manager hires the worker and the worker builds the widget "
         "and the manager owns the widget")
    theory = DomainGenesisEngine().build(q)
    assert theory is not None
    assert any(law.kind == "composition" for law in theory.laws)


def test_theory_and_law_serialize():
    theory = DomainGenesisEngine().build(_ALIEN)
    assert theory is not None
    d = theory.to_dict()
    assert d["label"] and isinstance(d["relations"], list) and isinstance(d["laws"], list)
    assert isinstance(d["inferences"], list) and isinstance(d["entities"], list)
    law = DomainLaw("transitivity", "x chains", relation="x", support=2)
    assert law.to_dict()["kind"] == "transitivity"
