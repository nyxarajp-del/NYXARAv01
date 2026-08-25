"""Goals she commits to and finishes, and the questions that could never be answered.

``goals_completed`` was zero over every session while ``goals_added`` climbed, and three separate
things had to be true at once for that to happen:

* **An evidence question could not be resolved by the only mechanism that resolves anything.**
  ``epistemic`` raises "what would settle whether birds requires water is true?" with its
  ``subject`` set to a whole *claim* and its ``predicate`` to the literal string ``evidence``.
  ``_close_curiosity`` matches a question's subject and predicate against an incoming triple's,
  so it compares ``birds requires water`` against ``birds``, and can never match.
* **Hard evidence was filed under one of the proposition's two names.** A confirmed guess supported
  the bare answer ("water") and not the claim shape ("sparrows requires water"), which is the key
  the evidence question asks about.
* **Only the top four questions by value were ever committed to**, and questions resolve
  independently of their rank, so the one that got answered was routinely one she had never
  undertaken.

And underneath all three, an ordering bug: closure ran before commitment, so a question raised and
answered inside one turn was already resolved when the tracker looked.
"""

from __future__ import annotations

from nyxara.njp import NJPBrain


def _with_an_evidence_question() -> tuple:
    """A brain and one open "what would settle whether X is true?" question.

    Driven until the question appears rather than assumed after N turns: `epistemic` raises these
    on a slow organ's own turn count, so how many turns it takes is an implementation detail this
    test has no business encoding.
    """
    brain = NJPBrain()
    session = ("birds need water", "a sparrow is a bird", "what does a sparrow need?",
               "sparrows need water")
    for turn in range(40):
        brain.think(session[turn % len(session)])
        question = next((q for q in brain.curiosity.open_questions()
                         if str(getattr(q, "predicate", "")) == "evidence"), None)
        if question is not None:
            return brain, question
    raise AssertionError("no evidence question was raised in 40 turns")


def _investigating() -> NJPBrain:
    """A session in which she derives answers and the Master then confirms them."""
    brain = NJPBrain()
    session = ["birds need water", "a sparrow is a bird", "what does a sparrow need?",
               "sparrows need water", "a crow is a bird", "what does a crow need?",
               "crows need water", "a robin is a bird", "what does a robin need?",
               "robins need water"]
    for _ in range(2):
        for line in session:
            brain.think(line)
    return brain


def test_a_question_she_committed_to_is_completed_when_it_is_answered():
    brain = _investigating()
    totals = brain.loop.stats()["totals"]
    assert totals["goals_added"] > 0, totals
    assert totals["goals_completed"] > 0, totals
    assert brain.goals.stats()["completed"] > 0, brain.goals.stats()


def test_work_is_committed_to_before_it_can_be_finished():
    """The ordering. Closure before commitment means a fast answer is never work at all."""
    import inspect
    from nyxara.njp.integrate import LearningLoop
    source = inspect.getsource(LearningLoop.close)
    assert source.index("_track_goals") < source.index("_close_curiosity"), \
        "curiosity is closed before goals are tracked, so a same-turn answer is never committed to"


def test_an_evidence_question_closes_when_the_evidence_arrives():
    """What settles "what would settle whether X is true?" is X acquiring hard evidence.

    Driven directly rather than through a session, because a session that earns hard evidence
    early never raises the question in the first place — which is the system working, and makes
    the end-to-end path a poor place to test the mechanism.

    A real condition being met, not a question retired for being old: staleness is a different
    state and `Curiosity.stale_questions` already owns it.
    """
    from nyxara.njp.beliefs import EvidenceKind
    brain, question = _with_an_evidence_question()
    claim = str(getattr(question, "subject", ""))
    assert not brain.beliefs.why(claim).get("hard_support")

    brain.beliefs.support(claim, EvidenceKind.PREDICTION,
                          detail="held on data it was not fitted to", source="test")
    brain.think("plants need light")            # any turn, so the loop runs again
    assert getattr(brain.curiosity.questions.get(question.key()), "resolved", False), \
        brain.curiosity.stats()


def test_an_evidence_question_stays_open_while_the_claim_rests_on_testimony():
    """Being told a thing does not settle whether it is true — that is the ledger's whole rule."""
    brain, question = _with_an_evidence_question()
    for _ in range(3):
        brain.think("birds need water")
    assert not getattr(brain.curiosity.questions.get(question.key()), "resolved", False)


def test_hard_evidence_is_filed_under_both_names_of_one_proposition():
    """`_stake_a_belief` files what she said; `field._record_beliefs` files the claim shape.

    One proposition, two keys, both held — so evidence filed against only the first leaves the
    second resting on testimony alone, and the question asking for exactly that evidence never
    sees it arrive.
    """
    brain = NJPBrain()
    brain.think("birds need water")
    brain.think("a sparrow is a bird")
    brain.think("what does a sparrow need?")
    brain.think("sparrows need water")
    bare = brain.beliefs.why("water")
    claim = brain.beliefs.why("sparrows requires water")
    assert bare.get("hard_support"), bare
    assert claim.get("hard_support"), claim


def test_commitment_follows_the_decision_not_the_ranking():
    """A cap on how much work is outstanding is a constraint; a cap on which she may notice
    is an accident of sorting."""
    from nyxara.njp.integrate import _TRACKED_QUESTIONS
    brain = _investigating()
    assert len(brain.loop._question_nodes) <= _TRACKED_QUESTIONS
    # More than the four the old per-turn slice allowed.
    assert brain.loop.stats()["totals"]["goals_added"] > 4


def test_a_negated_instruction_never_becomes_a_goal():
    """Recording something she was told *not* to do as a thing to do is the worst reading."""
    brain = NJPBrain()
    before = brain.goals.stats()["nodes"]
    brain.think("mat karo")
    names = [str(getattr(n, "name", "")) for n in brain.goals.nodes.values()]
    assert not any("mat karo" in n for n in names), names
    assert brain.goals.stats()["nodes"] >= before
