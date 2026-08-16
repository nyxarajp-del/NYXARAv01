"""The Cognitive Learning Core — does it actually learn, or does a counter go up?

Every test here is written against the *measured* failure it exists to prevent, because each of
these mechanisms was absent in a way that looked like presence: the organs were all built, all
reachable, and between them could not compose two facts, could not variabilise a rule, and could
not transfer a property to a thing they had never seen have it.

The rule the whole file keeps: **nothing is ever tested on what taught it.**
"""

from __future__ import annotations

import pytest

from nyxara.njp.brain import NJPBrain
from nyxara.njp.concepts import ConceptGenesis
from nyxara.njp.core import CognitiveLearningCore, _fold
from nyxara.njp.grounding import Grounder


def _fitted(sentences):
    """A grounder and a concept layer taught the same sentences, plus a Core over both."""
    grounder, concepts = Grounder(), ConceptGenesis(min_members=2)
    for sentence in sentences:
        result = grounder.ground(sentence)
        concepts.observe_triples(result.triples)
    concepts.crystallise()
    return CognitiveLearningCore(grounder=grounder, concepts=concepts, min_members=2)


_CHAIN = ("aag se garmi hoti hai", "garmi se pasina hota hai", "pasina se pyaas hoti hai")


# --------------------------------------------------------------------------- #
# Composition — the thing that did not exist anywhere in the package
# --------------------------------------------------------------------------- #

def test_two_stated_facts_compose_into_one_that_was_never_stated():
    """A grep for transitive closure across njp/ returned nothing before this."""
    core = _fitted(_CHAIN)
    derived = core.reach("aag", "causes")
    assert derived.ok
    assert derived.answer == "pasina"          # never said in any single sentence
    assert derived.kind == "composed"
    assert len(derived.support) >= 2


def test_a_composed_answer_is_never_stated_as_knowledge():
    """It follows from facts; it is not one. The ceiling is what keeps those apart."""
    core = _fitted(_CHAIN)
    derived = core.reach("aag", "causes")
    assert derived.defeasible
    assert derived.confidence <= 0.70


def test_a_longer_chain_is_worth_less_than_a_shorter_one():
    """Depth is priced, not merely capped — otherwise five hops assert as loudly as one."""
    core = _fitted(_CHAIN)
    near = core.connects("aag", "garmi")
    far = core.connects("aag", "pyaas")
    assert near.ok and far.ok
    assert far.confidence < near.confidence


def test_composition_does_not_walk_in_circles():
    """`a causes b` and `b causes a` are both sayable, and neither means a causes itself."""
    core = _fitted(("a se b hoti hai", "b se a hoti hai"))
    derived = core.reach("a", "causes")
    assert derived.answer != "a"


def test_an_unconnected_pair_is_not_connected():
    """The walk has to be able to say no, or 'yes' means nothing."""
    core = _fitted(_CHAIN)
    assert not core.connects("aag", "elephant").ok


# --------------------------------------------------------------------------- #
# Inheritance — the composition that actually rescues a lookup miss
# --------------------------------------------------------------------------- #

def test_a_property_is_inherited_through_a_stated_hierarchy():
    core = _fitted(("sparrow is a bird", "bird is a animal", "animal needs water"))
    derived = core.predict("sparrow", "needs")
    assert derived.answer == "water"
    assert derived.kind == "composed"
    assert derived.defeasible


def test_inheritance_folds_the_relation_name_the_way_the_store_does():
    """'needs' is stored as `requires`; a Core with its own naming rule would miss its own facts."""
    core = _fitted(("sparrow is a bird", "bird needs water"))
    assert core.predict("sparrow", "needs").answer == "water"
    assert core.predict("sparrow", "requires").answer == "water"


def test_a_stated_fact_always_beats_a_derived_one():
    """The ordering exists so a confident generalisation cannot bury the observation against it."""
    core = _fitted(("sparrow is a bird", "bird needs water", "sparrow needs nectar"))
    derived = core.predict("sparrow", "needs")
    assert derived.answer == "nectar"
    assert derived.kind == "direct"
    assert not derived.defeasible


# --------------------------------------------------------------------------- #
# Schemas — variabilised rules, which discover.py structurally cannot represent
# --------------------------------------------------------------------------- #

def _resources(n=8, *, withhold="potash"):
    names = ["water", "light", "fertilizer", "compost",
             "manure", "sunlight", "nitrogen", "potash"][:n]
    taught = [f"{name} is a resource" for name in names]
    taught += [f"{name} increases growth" for name in names if name != withhold]
    return names, taught


def test_a_schema_is_induced_over_a_kind_the_clustering_found():
    names, taught = _resources()
    core = _fitted(taught)
    report = core.represent()
    assert report.kept > 0
    assert any(s.predicate == "increases" and s.object == "growth"
               for s in core.schemas.values())


def test_a_schema_is_scored_only_on_facts_it_was_not_built_from():
    names, taught = _resources()
    core = _fitted(taught)
    core.represent()
    report = core.test()
    assert report.holdout_size > 0, "the split produced no held-out fold to score against"
    assert report.scored > 0
    # Everything in the fit fold must have been excluded from induction.
    for subject, predicate, obj, _c in core._edges(fit_only=True):
        assert not core._held_out(subject, predicate, obj)


def test_a_property_transfers_to_a_member_never_observed_to_have_it():
    """The claim the whole module exists for: unseen structure, successful prediction."""
    names, taught = _resources(withhold="potash")
    core = _fitted(taught)
    core.represent()
    core.test()
    derived = core.generalize("potash", "increases")
    assert derived.ok, "no transfer to a member that was never told the property"
    assert derived.answer == "growth"
    assert derived.kind == "schema"
    assert derived.defeasible


def test_transfer_refuses_a_subject_that_is_not_of_the_kind():
    names, taught = _resources()
    core = _fitted(taught + ["elephant is a mammal"])
    core.represent()
    core.test()
    assert not core.generalize("elephant", "increases").ok


def test_the_role_is_not_defined_by_the_property_being_predicted():
    """The circularity that made transfer unreachable by construction.

    Concepts are formed from features every member shares, so `increases:growth` lands among the
    *invariants* — and membership was then tested by requiring it. Asking whether a new thing
    belongs, in order to predict whether it increases growth, demanded it already be known to
    increase growth. Measured: seven resources clustered, the schema confirmed 3/3 on held-out
    facts, and `potash` got no concept and no prediction.
    """
    names, taught = _resources(withhold="potash")
    core = _fitted(taught)
    roles = core._roles_for_transfer("potash", "increases")
    assert roles, "the member matched no role once the predicted relation was excluded"
    # And with the relation left in, it is excluded from its own kind — the bug, reproduced.
    assert not core._role_of("potash")


# --------------------------------------------------------------------------- #
# Revision — a refuted schema changes something
# --------------------------------------------------------------------------- #

def test_a_refuted_schema_is_dropped_and_moves_its_predicate():
    core = _fitted(_CHAIN)
    from nyxara.njp.core import Schema
    dud = Schema(role="r", predicate="causes", object="nothing", hits=0, misses=6)
    core.schemas[dud.key] = dud
    before = core._transitivity("causes").value
    report = core.revise()
    assert report.dropped == 1
    assert dud.key not in core.schemas
    assert core._transitivity("causes").value < before, \
        "a schema can be refuted forever without the thing that proposed it learning anything"


def test_a_refuted_schema_becomes_a_question_she_can_actually_close():
    """Curiosity's own failure questions are terminal: "why does my grounding keep failing?"
    enters a queue priced by information gain and is either voiced or aged out, because nothing
    consumes `failure_cause`. A refuted schema is a better question — specific, and keyed on the
    very (subject, predicate) whose arrival resolves it."""
    from nyxara.njp.core import Schema
    brain = NJPBrain()
    brain.think("alpha is a widget")
    core = brain.learner
    dud = Schema(role="r", predicate="increases", object="speed", hits=0, misses=5,
                 refuted_by={"zeta": "drag"})
    core.schemas[dud.key] = dud

    revised = core.revise()
    core._wonder(type("_R", (), {"revised": revised})())

    assert core.questions_raised == 1
    raised = [q for q in brain.curiosity.open_questions()
              if q.subject == "zeta" and q.predicate == "increases"]
    assert raised, "the question was counted but never actually raised"
    assert "distinguishes" in raised[0].text


def test_the_specific_disagreement_is_recorded_not_just_the_miss():
    """"this generalisation is wrong" is not a question anyone can answer."""
    names, taught = _resources()
    core = _fitted(taught + ["potash is a resource", "potash increases drag"])
    core.represent()
    core.test()
    assert any(s.refuted_by for s in core.schemas.values()) or \
        all(s.misses == 0 for s in core.schemas.values())


def test_transitivity_is_learned_rather_than_tabled():
    core = _fitted(_CHAIN)
    start = core._transitivity("causes").value
    for _ in range(6):
        core._transitivity("causes").confirm()
    assert core._transitivity("causes").value > start


# --------------------------------------------------------------------------- #
# The held-out split itself
# --------------------------------------------------------------------------- #

def test_the_split_is_stable_across_calls():
    """A fold that reshuffles makes a schema's history meaningless."""
    keys = [f"s{i}|causes|o{i}" for i in range(50)]
    first = [_fold(k, share=0.3) for k in keys]
    second = [_fold(k, share=0.3) for k in keys]
    assert first == second


def test_the_split_is_roughly_the_share_it_claims():
    keys = [f"subject{i}|causes|object{i}" for i in range(2000)]
    held = sum(1 for k in keys if _fold(k, share=0.3))
    assert 0.25 <= held / len(keys) <= 0.35


# --------------------------------------------------------------------------- #
# Simulation without numbers
# --------------------------------------------------------------------------- #

def test_direction_propagates_where_no_coefficient_was_ever_fitted():
    """Between 'I have regression data' and 'I know nothing' is where she usually is."""
    core = _fitted(_CHAIN)
    down = core.simulate("aag", direction=-1.0)
    assert down.ok
    assert "garmi decreases" in down.answer
    assert "no fitted coefficient" in down.why


# --------------------------------------------------------------------------- #
# Wiring — the organ is reached on a real turn, and absent when off
# --------------------------------------------------------------------------- #

def test_the_core_is_built_and_cycles_on_a_real_turn():
    brain = NJPBrain()
    assert brain.learner is not None
    brain.think("aag se garmi hoti hai")
    assert brain.stats()["learner"]["cycles"] > 0


def test_the_core_shares_the_brains_own_fact_store():
    """A Core with a private copy would be a second brain to keep in step."""
    brain = NJPBrain()
    assert brain.learner.grounder is brain.grounder
    assert brain.learner.concepts is brain.genesis


def test_an_inherited_answer_reaches_the_master_through_a_whole_turn():
    brain = NJPBrain()
    for sentence in ("sparrow is a bird", "bird is a animal", "animal needs water"):
        brain.think(sentence)
    thought = brain.think("what does sparrow need?")
    assert "water" in thought.answer
    assert thought.derivation is not None
    assert thought.derivation.defeasible


def test_an_organ_that_is_off_is_absent_not_zeroed():
    class _Off:
        learner_enabled = False

    brain = NJPBrain(_Off())
    assert brain.learner is None
    assert "learner" not in brain.stats()


def test_switching_the_core_off_leaves_the_rest_of_the_brain_standing():
    """The ordering constraint that once took `metareason` down with it.

    `_build_metareason` registers a strategy bound to the Core, so it must be built first — and
    when it was not, the AttributeError was swallowed by the builder's fail-soft `except` and
    metareason silently became None. Six tests failed for a reason none of them named.
    """
    class _Off:
        learner_enabled = False

    brain = NJPBrain(_Off())
    assert brain.metareason is not None
    assert brain.calculator is not None


def test_persistence_round_trips_the_schemas():
    names, taught = _resources()
    core = _fitted(taught)
    core.represent()
    core.test()
    revived = CognitiveLearningCore(grounder=core.grounder, concepts=core.concepts)
    revived.load_dict(core.to_dict())
    assert len(revived.schemas) == len(core.schemas)


def test_a_corrupt_record_leaves_a_working_core():
    core = _fitted(_CHAIN)
    core.load_dict({"schemas": [{"nonsense": True}], "transitivity": "not a list"})
    assert core.stats()["turns"] >= 0


# --------------------------------------------------------------------------- #
# Reachable over the wire
# --------------------------------------------------------------------------- #

def test_the_core_is_readable_over_http():
    """An organ observable only from inside the process is a docstring, not a capability."""
    pytest.importorskip("fastapi", reason="the API server needs the [server] extra")
    from fastapi.testclient import TestClient

    from nyxara.server.app import create_app

    client = TestClient(create_app())
    body = client.get("/v1/njp/learner").json()
    assert "schemas" in body and "transitivity" in body

    calculator = client.get("/v1/njp/calculate").json()
    assert "evaluated" in calculator

    assert client.get("/v1/njp/nonsense").json()["available"] is False
