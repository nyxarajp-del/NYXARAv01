"""NYX V.02 · L-AXIOM-GENESIS — a formal system of her own, checked rather than asserted.

The layer's whole honesty lives in one distinction, and most of this file is about it: a model
FOUND proves consistency at that size; no model found proves nothing at all. Anything that
reported the second as "consistent" would be the exact lie this repo exists to avoid.
"""

from __future__ import annotations

import pytest

from nyxara.kernel.config import NyxConfig
from nyxara.nyx.axiom import PROPERTIES, Axiom, AxiomGenesis, Genesis
from nyxara.nyx.brain import NyxBrain

CYCLIC = "an ordering where a<b and b<a can both hold"
IMPOSSIBLE = "an irreflexive transitive ordering where a<b and b<a can both hold"

pytestmark = pytest.mark.skipif(not AxiomGenesis().available(), reason="z3 is not installed")


def _genesis(**kw) -> AxiomGenesis:
    kw.setdefault("bound", 4)
    return AxiomGenesis(**kw)


# --------------------------------------------------------------------------- #
# Reading a description into axioms
# --------------------------------------------------------------------------- #
def test_a_description_becomes_a_set_of_properties():
    axioms = _genesis().read(CYCLIC)
    names = {a.name for a in axioms}
    assert "cyclic-pair" in names and "transitive" in names


def test_a_negation_is_read_as_a_negated_axiom():
    axioms = _genesis().read("a relation that is not transitive")
    transitive = next(a for a in axioms if a.name == "transitive")
    assert transitive.asserted is False
    assert transitive.render() == "¬transitive"


def test_irreflexive_does_not_also_pull_in_reflexive():
    """Without a word boundary, "reflexiv" matches inside "irreflexive" and the system she
    writes down is not the one that was described."""
    names = {a.name for a in _genesis().read("an irreflexive relation")}
    assert names == {"irreflexive"}


def test_antisymmetric_does_not_also_pull_in_symmetric():
    names = {a.name for a in _genesis().read("an antisymmetric relation")}
    assert names == {"antisymmetric"}


def test_a_description_with_no_properties_is_refused_by_name():
    got = _genesis().invent("purple bananas")
    assert got.system is None
    assert "My axiom language is closed" in got.refused
    for name in PROPERTIES:
        assert name in got.refused


# --------------------------------------------------------------------------- #
# Consistency — a bounded model search, reported as one
# --------------------------------------------------------------------------- #
def test_a_consistent_system_reports_the_model_it_actually_found():
    got = _genesis().invent(CYCLIC)
    assert got.system.consistent is True
    assert got.system.model_size >= 2
    assert "R" in got.system.model                     # a real witness, not a claim
    assert "PROVEN at that size" in got.system.note


def test_no_model_is_never_reported_as_consistent():
    """irreflexive + transitive + a two-way pair genuinely has no model. She still will not
    call that a proof of inconsistency, and certainly not a proof of consistency."""
    got = _genesis().invent(IMPOSSIBLE)
    assert got.system.consistent is None
    assert "not a proof of inconsistency" in got.system.note
    assert "not a proof of consistency" in got.system.note


def test_the_bound_travels_with_every_result():
    got = _genesis(bound=3).invent(CYCLIC)
    assert got.system.bound == 3
    assert "bounded check to 3" in got.render()


def test_an_inconsistent_system_is_never_promoted():
    got = _genesis().invent(IMPOSSIBLE)
    assert got.promoted is False and got.conjecture is True


# --------------------------------------------------------------------------- #
# Independence — an axiom that follows from the rest is not a new axiom
# --------------------------------------------------------------------------- #
def test_independent_axioms_are_marked_independent():
    got = _genesis().invent(CYCLIC)
    assert any(a.independent for a in got.system.axioms)


def test_independence_is_shown_by_a_witness_not_asserted():
    got = _genesis().invent(CYCLIC)
    independent = [a for a in got.system.axioms if a.independent]
    assert independent and all("at size" in a.why for a in independent)


def test_a_system_with_no_independent_axiom_stays_a_conjecture(monkeypatch):
    """A system whose axioms all follow from each other is a restatement, not a discovery."""
    genesis = _genesis()
    monkeypatch.setattr(genesis, "_independent", lambda axioms, index: (False, "no witness"))
    got = genesis.invent(CYCLIC)
    assert got.system.consistent is True
    assert got.promoted is False and got.conjecture is True


# --------------------------------------------------------------------------- #
# Derivation — bounded, and labelled bounded
# --------------------------------------------------------------------------- #
def test_a_derived_theorem_carries_its_bound_and_says_it_is_not_a_proof():
    got = _genesis().invent(CYCLIC)
    held = [t for t in got.theorems if t.holds]
    assert held
    for theorem in held:
        assert theorem.bound == 4
        assert "bounded, not a proof" in theorem.why
        assert theorem.certificate.startswith("no counter-model")


def test_a_theorem_that_does_not_follow_shows_its_counter_model():
    got = _genesis().invent(CYCLIC)
    refuted = [t for t in got.theorems if not t.holds]
    assert refuted
    assert all("counter-model exists at size" in t.why for t in refuted)


def test_antisymmetry_does_not_follow_from_a_two_way_ordering():
    got = _genesis().invent(CYCLIC)
    antisym = next(t for t in got.theorems if t.statement == "antisymmetric")
    assert antisym.holds is False


def test_derivation_can_be_asked_for_a_specific_candidate():
    genesis = _genesis()
    got = genesis.invent("a transitive relation", derive=["reflexive"])
    assert [t.statement for t in got.theorems] == ["reflexive"]


# --------------------------------------------------------------------------- #
# Notation and promotion
# --------------------------------------------------------------------------- #
def test_a_promoted_system_gets_glyphs_of_her_own():
    brain = NyxBrain(NyxConfig())
    got = brain.invent_axioms(CYCLIC)
    assert got.promoted and got.notation


def test_promotion_requires_both_a_model_and_an_independent_axiom():
    got = _genesis().invent(CYCLIC)
    assert got.promoted
    assert got.system.consistent is True
    assert any(a.independent for a in got.system.axioms)


def test_a_promoted_system_is_kept_and_a_conjecture_is_not():
    genesis = _genesis()
    genesis.invent(CYCLIC, name="kept")
    genesis.invent(IMPOSSIBLE, name="not-kept")
    assert "kept" in genesis.systems and "not-kept" not in genesis.systems


# --------------------------------------------------------------------------- #
# Through the brain
# --------------------------------------------------------------------------- #
def test_invent_axioms_is_reachable_on_the_brain():
    got = NyxBrain(NyxConfig()).invent_axioms(CYCLIC)
    assert got is not None and got.system is not None


def test_disabling_the_layer_leaves_v01_behaviour():
    brain = NyxBrain(NyxConfig(axiom_enabled=False))
    assert brain.axiom is None and brain.invent_axioms("anything") is None


# --------------------------------------------------------------------------- #
# Honesty and fail-soft
# --------------------------------------------------------------------------- #
def test_stats_say_what_this_layer_is_and_is_not():
    note = _genesis().stats()["note"]
    assert "not new physical truths" in note
    assert "coherent, not true" in note
    assert "proves nothing" in note
    assert "will not prove Riemann" in note


def test_the_axiom_language_is_stated_as_closed():
    assert set(_genesis().stats()["language"]) == set(PROPERTIES)


def test_without_a_solver_the_layer_refuses_rather_than_guessing(monkeypatch):
    genesis = _genesis()
    monkeypatch.setattr(genesis, "available", lambda: False)
    got = genesis.invent(CYCLIC)
    assert got.system is None
    assert "nobody checked is not a result" in got.refused


def test_everything_is_fail_soft_on_junk():
    genesis = _genesis()
    assert genesis.invent(None).refused
    assert genesis.read(None) == []
    assert genesis.load_dict(None) is False
    assert Genesis().to_dict()["promoted"] is False


def test_the_record_survives_a_restart():
    genesis = _genesis()
    genesis.invent(CYCLIC, name="kept")
    other = _genesis()
    assert other.load_dict(genesis.to_dict())
    assert other.promoted == 1 and "kept" in other.systems


def test_an_axiom_renders_with_its_polarity():
    assert Axiom(name="transitive", asserted=False).render() == "¬transitive"
    assert Axiom(name="transitive").render() == "transitive"


# --------------------------------------------------------------------------- #
# A property she can express must be a property she can be asked for
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(PROPERTIES))
def test_every_property_answers_to_its_own_name(name):
    """``cyclic-pair`` was expressible, listed, and reachable by no phrasing at all.

    The hand-written cues covered "cycle" and "both ways", and ``\bcycle`` does not match
    "cyclic" — so a description naming it built a *weaker* system instead, found a model for
    that, and promoted it. She answered a question nobody asked and called it a success.
    """
    for phrasing in (name, name.replace("-", " ")):
        got = AxiomGenesis().read(f"a relation that is {phrasing}")
        assert name in {axiom.name for axiom in got}, f"{phrasing!r} did not select {name!r}"


def test_the_headline_case_reproduces_exactly():
    """The two systems the docs cite. Both were wrong while ``cyclic-pair`` was unreachable."""
    genesis = AxiomGenesis()

    ok = genesis.invent("a transitive relation with a cyclic pair")
    assert {a.name for a in ok.system.axioms} == {"transitive", "cyclic-pair"}
    assert ok.system.consistent is True and ok.system.model_size == 2

    none = genesis.invent("an irreflexive transitive relation with a cyclic pair")
    assert {a.name for a in none.system.axioms} == {"irreflexive", "transitive", "cyclic-pair"}
    assert none.system.consistent is None
    assert not none.promoted


def test_no_model_found_is_never_reported_as_inconsistent_or_consistent():
    """Gödel's respect is in this one line, so it is asserted rather than trusted."""
    rendered = AxiomGenesis().invent(
        "an irreflexive transitive relation with a cyclic pair").render().lower()
    assert "not a proof of inconsistency" in rendered
    assert "not a proof of consistency either" in rendered


def test_a_row_label_never_asserts_what_its_value_denies():
    """The row read ``consistent:`` while its value said no model had been found."""
    rendered = AxiomGenesis().invent(
        "an irreflexive transitive relation with a cyclic pair").render()
    assert "consistent:" not in rendered
