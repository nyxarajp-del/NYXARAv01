"""The gauntlet written to break the 0.990 (NJP V.37).

Most of this file tests the *instrument*. Four of the nine papers produced false passes or false
failures on their first run, and every one of those was the exam rather than her — so the tests
that matter here are the ones that assert a paper measures what it claims to.
"""
from __future__ import annotations

import pytest

from nyxara.njp.explaingauntlet import (
    ATTACKS, CONFLICT_PASS, SILENT_PASS, Fact, Gauntlet, Reply, Verdict, World,
    _deepest, _walk, render, run,
)


@pytest.fixture(scope="module")
def gauntlet():
    return Gauntlet(limit=12)


# --------------------------------------------------------------------------- #
# Nothing here comes from the corpus
# --------------------------------------------------------------------------- #
def test_every_world_is_minted_and_holds_only_its_own_item(gauntlet):
    """A shared store means an item's trap can be defused by an unrelated edge."""
    for attack in ATTACKS:
        for item in gauntlet.items(attack):
            assert item.world.facts_list, attack
            assert len(item.world.facts_list) <= 12, attack


def test_no_entity_in_the_gauntlet_is_in_the_shipped_corpus(gauntlet):
    from nyxara.njp.general import load_brain

    store = load_brain().grounder.facts
    known = {subject for (subject, _predicate) in store}
    for attack in ATTACKS:
        for item in gauntlet.items(attack):
            for fact in item.world.facts_list:
                assert fact.subject.lower() not in known, f"{attack}: {fact.subject}"


def test_the_same_seed_mints_the_same_gauntlet():
    first = Gauntlet(seed=7, limit=6).items("shape")
    second = Gauntlet(seed=7, limit=6).items("shape")
    assert [i.question for i in first] == [i.question for i in second]
    assert [i.question for i in Gauntlet(seed=8, limit=6).items("shape")] != \
        [i.question for i in first]


# --------------------------------------------------------------------------- #
# The four papers that were graded wrong
# --------------------------------------------------------------------------- #
def test_a_collider_accepts_either_cause_named_but_not_a_chain_between_them():
    """The gold was one of two equally right answers. Same rule as V.36's mechanism paper."""
    g = Gauntlet(limit=5)
    collider = [i for i in g.items("shape") if "collider" in i.note]
    assert collider, "no collider items minted"
    item = collider[0]
    a, b = item.both
    # Both named, no chain between them: a pass.
    assert g.grade(item, Reply(text=f"because {a}\nbecause {b}",
                               chains=[[a], [b]])).passed
    # One named only: not a pass.
    assert not g.grade(item, Reply(text=f"because {a}", chains=[[a]])).passed
    # Both named, but put on one chain: an invented edge, and wrong.
    bad = g.grade(item, Reply(text=f"because {a}, which causes {b}", chains=[[b, a]]))
    assert not bad.passed and bad.got is Verdict.WRONG and "independent" in bad.why


def test_the_gap_paper_asks_about_the_node_whose_support_was_deleted(gauntlet):
    """It asked about a node whose cause was still stated, and called a fact a confabulation."""
    for item in gauntlet.items("gap"):
        target = item.question.split()[2]
        # Nothing in the world causes the target.
        assert not any(f.predicate == "causes" and f.object.lower() == target.lower()
                       for f in item.world.facts_list), item.note
        # And the target still has edges, so there is something to invent from.
        assert any(f.subject.lower() == target.lower() for f in item.world.facts_list)
        assert item.want is Verdict.UNKNOWN


def test_the_distractors_are_on_the_relation_the_walk_actually_reads(gauntlet):
    """Injected on `purpose`/`has_part`/`is_a`, they were noise in a channel nobody listened to."""
    for item in gauntlet.items("distractors"):
        on_causes = [f for f in item.world.facts_list
                     if f.predicate == "causes" and f.object in item.forbidden]
        assert on_causes, item.note


def test_contradiction_needs_the_flag_and_not_merely_the_two_names(gauntlet):
    item = gauntlet.items("contradiction")[0]
    a, b = item.dispute
    listed = gauntlet.grade(item, Reply(text=f"because {a}\nbecause {b}", chains=[[a], [b]]))
    assert not listed.passed and "without noticing" in listed.why
    flagged = gauntlet.grade(item, Reply(text=f"{a} and {b} cannot both hold",
                                         chains=[[a], [b]], conflict=True))
    assert flagged.passed


def test_legs_needs_joint_necessity_and_not_a_list_of_alternatives(gauntlet):
    item = gauntlet.items("legs")[0]
    a, b = item.both
    alternatives = gauntlet.grade(item, Reply(text=f"because {a}\nbecause {b}",
                                              chains=[[a], [b]]))
    assert not alternatives.passed and "alternatives" in alternatives.why
    conjunction = gauntlet.grade(item, Reply(text=f"both {a} and {b} are required",
                                             chains=[[a], [b]], joint=True))
    assert conjunction.passed


def test_the_generated_graph_gold_is_the_longest_derivation_not_the_last_edge():
    facts = [Fact("a", "causes", "b"), Fact("b", "causes", "c"), Fact("x", "causes", "c")]
    assert _deepest(facts, "c") == ("c", "b", "a")


# --------------------------------------------------------------------------- #
# A forbidden node is wrong wherever it appears
# --------------------------------------------------------------------------- #
def test_a_distractor_named_beside_the_right_chain_is_still_wrong(gauntlet):
    item = gauntlet.items("distractors")[0]
    right = " ".join(item.chain)
    got = gauntlet.grade(item, Reply(text=f"{right} and also {item.forbidden[0]}",
                                     chains=[list(item.chain)]))
    assert not got.passed and got.got is Verdict.WRONG and "forbidden" in got.why


def test_silence_is_never_penalised_for_a_forbidden_node(gauntlet):
    item = gauntlet.items("distractors")[0]
    got = gauntlet.grade(item, Reply())
    assert got.got is Verdict.UNKNOWN and "forbidden" not in got.why


# --------------------------------------------------------------------------- #
# The report refuses to average across the divide
# --------------------------------------------------------------------------- #
def test_the_papers_that_reward_silence_are_not_in_the_score():
    report = run(limit=8)
    names = {p.name for p in report.answering}
    assert names.isdisjoint(set(SILENT_PASS) | set(CONFLICT_PASS))
    assert 0.0 <= report.score <= 1.0


def test_restraint_and_confabulation_are_reported_separately():
    report = run(limit=8)
    assert 0.0 <= report.restraint <= 1.0
    assert report.confabulated >= 0
    assert "restraint" in render(report) and "confabulation" in render(report)


def test_a_system_that_answers_everything_scores_worse_on_restraint():
    """The number that stops "answer more" from being the way to a higher score."""
    def always(question, world):
        return Reply(text="something", chains=[["something"]])

    honest = run(limit=8)
    loud = run(limit=8, ask=always)
    assert loud.restraint == 0.0 and honest.restraint > loud.restraint
    assert loud.confabulated > honest.confabulated


# --------------------------------------------------------------------------- #
# The floor, pinned
# --------------------------------------------------------------------------- #
def test_the_walk_is_measured_and_the_gaps_are_where_they_were_recorded():
    """Where she stands, pinned. This test is **expected to be edited** by each version.

    It is the record of which capabilities the gauntlet says are present and which are absent, and
    a version that closes one is supposed to come here and say so. What it must never do is get
    quietly relaxed: an assertion that a paper scores 0.0 turning into `>= 0.0` would hide the
    difference between fixing something and breaking the measurement.

    V.37 floor: wording 0.100, contradiction 0.000, homonym 0.000, legs 0.000 — 0.508 overall.
    V.38 now:   wording 1.000 on taught cues, homonym 1.000. contradiction and legs still absent.
    """
    report = run(limit=24)
    assert report.paper("entities").score == 1.0        # the control
    assert report.paper("gap").score == 1.0             # she does not invent a bridge
    assert report.paper("unknown").score == 1.0
    assert report.restraint == 1.0 and report.confabulated == 0

    # Closed at V.38.
    assert report.paper("wording").score == 1.0         # induced from demonstrations
    assert report.paper("homonym").score == 1.0         # the identity firewall

    # Still absent, and named rather than rounded away.
    assert report.paper("contradiction").score == 0.0   # no notion of a dispute
    assert report.paper("legs").score == 0.0            # no joint necessity
    # And the honest ceiling of an induced grammar: a cue nobody demonstrated is not guessed.
    assert report.paper("wording_new").score == 0.0
    assert report.paper("wording_new").silent == report.paper("wording_new").asked


def test_the_adapter_reports_no_conflict_and_no_joint_and_that_is_the_finding():
    world = World([Fact("a", "causes", "b")])
    reply = _walk("why does b happen?", world)
    assert reply.chains and reply.conflict is False and reply.joint is False
