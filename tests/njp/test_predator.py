"""The process that goes after her own explanations (NJP V.39).

An explanation is a composition: every fact in it can be true and the composition can still be
false. These tests are about the four ways this module is allowed to say so, and the three things
it is not allowed to do.
"""
from __future__ import annotations

import pytest

from nyxara.njp.explain import Explainer
from nyxara.njp.predator import EXCLUDES, Attack, Predator, Survival


class Triple:
    __slots__ = ("object", "confidence", "superseded")

    def __init__(self, obj, confidence=1.0):
        self.object, self.confidence, self.superseded = obj, confidence, False


class Store:
    def __init__(self, triples=()):
        self.facts = {}
        for subject, predicate, obj, *rest in triples:
            self.facts.setdefault((subject.lower(), predicate), []).append(
                Triple(obj, rest[0] if rest else 1.0))

    def _key(self, text):
        return " ".join(str(text or "").split()).lower()


def walker(triples=(), **kwargs):
    return Explainer(Store(triples), **kwargs)


# --------------------------------------------------------------------------- #
# Exclusion: two answers that cannot both hold
# --------------------------------------------------------------------------- #
EXCLUSIVE = [("a", "causes", "t"), ("b", "causes", "t"),
             ("a", "excludes", "b"), ("b", "excludes", "a")]


def test_two_causes_stated_to_exclude_each_other_are_reported_as_a_dispute():
    got = walker(EXCLUSIVE).why("t", sense="because")
    assert got.conflict is True
    assert "cannot both hold" in got.text()
    # and both are still named, because the dispute is between two specific claims
    assert "a" in got.text() and "b" in got.text()


def test_the_dispute_is_reported_and_never_resolved():
    """Reporting "A and B cannot both hold" and then answering A is worse than not noticing."""
    got = walker(EXCLUSIVE).why("t", sense="because")
    attack = got.survival.of_kind("exclusion")[0]
    assert set(attack.about) == {"a", "b"}
    assert attack.evidence[1] in EXCLUDES


def test_two_causes_that_merely_differ_are_not_a_dispute():
    got = walker([("a", "causes", "t"), ("b", "causes", "t")]).why("t", sense="because")
    assert got.conflict is False and got.survival.survived


@pytest.mark.parametrize("relation", EXCLUDES)
def test_every_spelling_of_exclusion_is_read(relation):
    got = walker([("a", "causes", "t"), ("b", "causes", "t"),
                  ("a", relation, "b")]).why("t", sense="because")
    assert got.conflict is True


def test_an_exclusion_between_two_middles_is_not_reported():
    """The root is what the explanation offers as the answer; a middle may be irrelevant."""
    got = walker([("a", "causes", "m1"), ("m1", "causes", "t"),
                  ("b", "causes", "m2"), ("m2", "causes", "t"),
                  ("m1", "excludes", "m2")]).why("t", sense="because")
    assert got.conflict is False


# --------------------------------------------------------------------------- #
# Conjunction: chains that are not alternatives
# --------------------------------------------------------------------------- #
LEGS = [("a1", "causes", "a2"), ("a2", "causes", "t"),
        ("b1", "causes", "b2"), ("b2", "causes", "t"),
        ("t", "requires", "a2"), ("t", "requires", "b2")]


def test_chains_the_target_requires_are_reported_as_jointly_required():
    got = walker(LEGS).why("t", sense="because")
    assert got.joint is True
    assert "jointly required" in got.text()
    assert "a1" in got.text() and "b1" in got.text()


def test_alternatives_are_not_called_a_conjunction():
    got = walker([("a", "causes", "t"), ("b", "causes", "t")]).why("t", sense="because")
    assert got.joint is False


def test_one_requirement_is_not_a_conjunction():
    got = walker([("a1", "causes", "t"), ("b1", "causes", "t"),
                  ("t", "requires", "a1")]).why("t", sense="because")
    assert got.joint is False


# --------------------------------------------------------------------------- #
# Assumption and counterexample
# --------------------------------------------------------------------------- #
def test_a_weak_link_is_marked_and_the_chain_is_not_withdrawn():
    got = walker([("a", "causes", "b", 0.4), ("b", "causes", "t", 0.95)]).why("t",
                                                                              sense="because")
    assert got.answered
    marks = got.survival.of_kind("assumption")
    assert marks and "0.40" in marks[0].finding
    assert "a" in got.text()


def test_a_confident_chain_is_not_marked():
    got = walker([("a", "causes", "b", 0.95), ("b", "causes", "t", 0.95)]).why("t",
                                                                               sense="because")
    assert not got.survival.of_kind("assumption")


def test_a_stated_counterexample_is_found():
    got = walker([("a", "causes", "t"), ("t", "despite", "a")]).why("t", sense="because")
    found = got.survival.of_kind("counterexample")
    assert found and "despite" in found[0].finding


# --------------------------------------------------------------------------- #
# What it may not do
# --------------------------------------------------------------------------- #
def test_every_attack_rests_on_a_stated_fact():
    """An attack that needed a fact nobody stated would be confabulating in order to accuse."""
    explainer = walker(EXCLUSIVE + LEGS + [("a", "causes", "z", 0.3)])
    for topic in ("t", "z"):
        got = explainer.why(topic)
        for attack in (got.survival.attacks if got.survival else ()):
            assert attack.evidence, attack.finding
            subject, relation, obj = attack.evidence
            held = [o for o, _c in explainer._out(subject, relation)]
            assert any(explainer._key(o) == explainer._key(obj) for o in held), attack.evidence


def test_the_predator_runs_after_the_walk_and_never_inside_it():
    """A predator inside chain-building would suppress the evidence that corrects it."""
    with_hunt = walker(EXCLUSIVE).why("t", sense="because")
    without = walker(EXCLUSIVE, hunt=False).why("t", sense="because")
    assert [c.nodes for c in with_hunt.chains] == [c.nodes for c in without.chains]
    assert with_hunt.conflict is True and without.conflict is False


def test_an_explanation_with_no_chains_is_not_attacked():
    got = walker().why("nothing")
    assert got.chains == [] and got.survival is None
    assert got.conflict is False and got.joint is False


def test_the_finding_is_said_before_the_list_it_changes():
    got = walker(EXCLUSIVE).why("t", sense="because")
    first = got.text().splitlines()[0]
    assert "cannot both hold" in first


# --------------------------------------------------------------------------- #
# It is worth what it claims to be worth
# --------------------------------------------------------------------------- #
def test_the_gauntlet_papers_it_was_built_for_collapse_without_it():
    from nyxara.njp.explaingauntlet import Reply, run

    def blind(question, world):
        ex = Explainer(world, hunt=False)
        got = ex.ask(question)
        if got is None:
            return Reply()
        if hasattr(got, "orders"):
            return Reply(text=" → ".join(got.first) if got.first else "",
                         chains=[list(got.first)] if got.first else [])
        return Reply(text=got.text(), chains=[list(c.nodes) for c in got.chains],
                     conflict=bool(getattr(got, "conflict", False)),
                     joint=bool(getattr(got, "joint", False)))

    on, off = run(limit=16), run(limit=16, ask=blind)
    assert on.paper("contradiction").score == 1.0 and off.paper("contradiction").score == 0.0
    assert on.paper("legs").score == 1.0 and off.paper("legs").score == 0.0
    # and it changes nothing it was not built to change
    for untouched in ("entities", "distractors", "gap", "unknown"):
        assert on.paper(untouched).score == off.paper(untouched).score
