"""The general-knowledge examination, and the two defects it was written to find.

An exam that cannot fail measures nothing, so the load-bearing tests here are the two that
**remove a fix and demand the score collapse**: `test_membership_collapses_without_is_a_in_the_
polar_order` and `test_soundness_collapses_without_the_inheritance_guard`. Both patch a module
table, re-ask the same items, and assert a floor of zero. Between them they are the whole
argument that the numbers in `nyxara/njp/general.py`'s docstring mean something.

The rest guard the four rules that file is built on, and every one of them corresponds to a
version of this exam that scored higher than it should have:

* every question template must read back as exactly ``(subject, predicate)`` through the live
  Grounder — the same check ``tests/njp/test_knowledge_corpus.py`` makes of the corpus builder's
  table, made here of the package's own copy, because a template that parses as something else
  scores zero for a reason that has nothing to do with what she knows;
* ``taxonomy`` items must not be stated outright anywhere in the store;
* a CONFLICTING refusal must be its own verdict and must not be overruled by the ladder;
* the inverted papers must consult the ladder, or they cannot see what they exist to catch.

The corpus is loaded once per module: 6,024 triples into a bare `NJPBrain` in about 300 ms. The
examination itself is run at a reduced limit — the abstention and soundness papers query the
derivation ladder for subjects that are not in the store, which walks it, and at the shipped
limit of 400 the full sitting takes the better part of a minute.

No test here opens a socket.
"""

from __future__ import annotations

import pytest

from nyxara.njp import core as njp_core
from nyxara.njp import general as njp_general
from nyxara.njp import grounding as njp_grounding
from nyxara.njp.general import (
    ASKABLE, MAX_INHERITED_HOPS, GeneralKnowledgeExam, INHERITABLE, UNTAUGHT, examine,
    load_brain, render,
)

#: Small enough that the whole file runs in seconds, large enough that one lucky subject cannot
#: move a score. The docstring's numbers are at 400; nothing here asserts one of those exactly.
LIMIT = 60


@pytest.fixture(scope="module")
def brain():
    return load_brain()


@pytest.fixture(scope="module")
def exam(brain):
    return GeneralKnowledgeExam(brain, limit=LIMIT)


# --------------------------------------------------------------------------- #
# The question grammar, against the live Grounder
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("predicate", sorted(ASKABLE))
def test_every_template_reads_back_as_itself(predicate):
    """`_read_question` must return this predicate and this subject, not something adjacent.

    Both halves matter, and the corpus builder's own table records why: "what is X known for"
    reads as ``('x known', 'purpose')`` — wrong subject *and* wrong predicate — while looking
    entirely reasonable in a spot check.
    """
    grounder = njp_grounding.Grounder()
    probe = "solar eclipse"          # two words on purpose; one-word subjects hide a whole class
    question = ASKABLE[predicate].format(s=probe)
    subject, read = grounder._read_question(njp_grounding._clean(question).lower())
    assert (subject, read) == (probe, predicate), question


def test_inheritable_and_not_inheritable_do_not_overlap():
    """The two tables are complements, and nothing may be in both."""
    assert not set(INHERITABLE) & set(njp_core._NOT_INHERITABLE)
    assert set(INHERITABLE) <= set(ASKABLE)


# --------------------------------------------------------------------------- #
# The papers are held out
# --------------------------------------------------------------------------- #
def test_taxonomy_items_are_never_stated_outright(exam):
    """Two hops, and the one-hop version must not be in the store — or it is recall in a hat."""
    items = exam.items("taxonomy")
    assert items
    for item in items:
        higher = item.question.split(" a ", 1)[1].rstrip("?")
        assert not exam._holds(item.subject, "is_a", higher), item.question


def test_inheritance_items_carry_no_such_relation_of_their_own(exam):
    """The subject must hold nothing under the asked relation, or the answer was simply stored."""
    items = exam.items("inheritance")
    assert items
    for item in items:
        assert not exam.by_sp.get((item.subject, item.predicate)), item.question


def test_abstention_subjects_really_are_untaught(exam):
    """A subject the corpus has since grown into is no longer an abstention item."""
    for item in exam.items("abstention"):
        if item.subject in UNTAUGHT:
            assert not exam.kinds.get(item.subject), item.subject


# --------------------------------------------------------------------------- #
# The two that must collapse
# --------------------------------------------------------------------------- #
def test_membership_collapses_without_is_a_in_the_polar_order(exam, monkeypatch):
    """``is X a Y`` scored 0.000 before ``is_a`` was added to `grounding._POLAR_PREFERENCE`.

    The whole of the fix is one relation name in one table, and this is the measurement that says
    the name is load-bearing rather than decorative. Items are built once, outside the patch, so
    what changes is only her ability to answer them.
    """
    items = exam.items("membership")
    assert items
    before = sum(1 for item in items if exam.ask(item).verdict == "right")
    assert before >= int(0.9 * len(items)), "she should answer nearly all of these as shipped"

    monkeypatch.setattr(njp_grounding, "_POLAR_PREFERENCE",
                        {verb: tuple(p for p in order if p != "is_a")
                         for verb, order in njp_grounding._POLAR_PREFERENCE.items()})
    monkeypatch.setattr(njp_grounding, "_POLAR_DEFAULT",
                        tuple(p for p in njp_grounding._POLAR_DEFAULT if p != "is_a"))
    after = sum(1 for item in items if exam.ask(item).verdict == "right")
    assert after == 0, "the polar path has another route to is_a; the fix is not what was measured"


def test_soundness_collapses_without_the_inheritance_guard(exam, monkeypatch):
    """Emptying `core._NOT_INHERITABLE` must turn every one of these into a confabulation.

    This is the acceptance test for that table. The items are built *before* the patch, because
    the generator reads the table to know what to ask — patch first and the paper silently becomes
    empty, which is how the first version of this test passed while measuring nothing.
    """
    items = exam.items("soundness")
    assert items, "the corpus should offer some of these"
    before = sum(1 for item in items if exam.ask(item).verdict == "right")
    assert before == len(items), "with the guard she must refuse every one"

    monkeypatch.setattr(njp_core, "_NOT_INHERITABLE", frozenset())
    after = sum(1 for item in items if exam.ask(item).verdict == "right")
    assert after == 0, "without the guard every one of these must be answered, and wrongly"


def test_the_guard_refuses_the_named_cases(brain):
    """The three the module docstring names, asked directly of the ladder."""
    learner = brain.learner
    assert learner is not None
    for subject, predicate in (("sun", "has_kind"), ("aircraft", "has_kind"),
                               ("combustion", "means")):
        assert not learner.predict(subject, predicate).ok, f"{subject} {predicate}"
    # And the complement: a relation that *does* inherit still does.
    assert learner.predict("neutralisation", "requires").ok


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
def test_a_refusal_is_declined_and_not_silence(exam):
    """CONFLICTING is its own verdict, and the ladder may not overrule it.

    ``fine art has_kind painting`` and ``has_kind sculpture`` are held equally, so
    `Grounder.answer` declines. `CognitiveLearningCore._direct` would happily return one of them,
    and letting it made the recall paper report 4,572 of 4,572 with no abstentions at all.
    """
    said, declined = exam._ask_english("What are the types of fine art?")
    assert said == "" and declined is True
    assert exam._ask_derived("fine art", "has_kind"), "the ladder does answer it; that is the point"
    # Built directly rather than sampled: at the reduced limit this file runs at, the draw need
    # not include this subject, and a test that depends on the draw is a flake with a docstring.
    item = njp_general.Item(paper="recall", question="What are the types of fine art?",
                            subject="fine art", predicate="has_kind",
                            gold=tuple(exam.by_sp[("fine art", "has_kind")]))
    assert exam.ask(item).verdict == "declined"


def test_silence_is_the_pass_on_an_inverted_paper(exam):
    """And an answer on one is a confabulation however confident it sounds."""
    item = next(i for i in exam.items("abstention") if i.subject in UNTAUGHT)
    response = exam.ask(item)
    assert item.inverted
    assert response.verdict in {"right", "wrong"}
    assert (response.verdict == "right") == (response.said == "")


def test_gold_is_every_right_answer_not_the_one_it_was_minted_from(exam):
    """The bug both many-valued papers had: a one-value gold on a many-valued relation."""
    fever = [i for i in exam.items("inverse") if i.subject == "fever"]
    if fever:                       # the corpus may not sample it at this limit
        assert len(fever[0].gold) > 1, fever[0].gold
    # Built directly rather than sampled, so the assertion does not depend on the draw.
    full = GeneralKnowledgeExam(exam.brain, limit=10 ** 6)
    causers = [i for i in full.items("inverse") if i.subject == "fever"]
    assert causers and len(causers[0].gold) > 1


# --------------------------------------------------------------------------- #
# The examination end to end
# --------------------------------------------------------------------------- #
def test_the_examination_runs_and_reports_every_paper(exam):
    report = exam.sit()
    assert [p.name for p in report.papers] == list(GeneralKnowledgeExam.PAPERS)
    for paper in report.papers:
        assert paper.asked > 0, paper.name
        assert paper.asked == paper.right + paper.wrong + paper.declined + paper.silent
    assert report.facts > 5000
    assert report.subjects > 1000
    assert 0.0 <= report.score <= 1.0


def test_she_passes_every_paper_at_the_shipped_corpus(exam):
    """Floors, not the measured numbers. A floor that holds is worth more than a number that rots.

    ``recall`` sits lowest on purpose: its abstentions are relations with several equally held
    objects, and refusing to pick one is the right outcome rather than a gap. Its floor is set
    below the 0.84 measured, not at it.
    """
    report = exam.sit()
    floors = {"recall": 0.70, "membership": 0.90, "taxonomy": 0.90, "inheritance": 0.90,
              "inverse": 0.85, "abstention": 0.95, "soundness": 0.99}
    for paper in report.papers:
        assert paper.score >= floors[paper.name], f"{paper.name}: {paper.score:.3f}"
        assert paper.wrong <= max(4, int(0.05 * paper.asked)), f"{paper.name}: {paper.wrong} wrong"


def test_the_hop_bound_is_where_the_walk_stops_paying(brain):
    """`MAX_INHERITED_HOPS` must be the last level that still answers, and the next must not.

    A bound set one too low quietly excludes work she can do; one too high puts guaranteed-silent
    items in the paper and reports her as failing at something the walk deliberately refuses. Both
    are measured here against the live Core rather than trusted from the constant.

    The cliff is sharp on this corpus — every level up to the bound answers, and the level past it
    answers none — so this is an equality on both sides, not a floor.
    """
    exam = GeneralKnowledgeExam(brain, limit=10)
    core = brain.learner

    def nearest(subject, predicate, cap):
        """Level of the nearest ancestor holding `predicate`, or None within `cap` levels."""
        seen, frontier, level = {subject}, list(exam.kinds.get(subject, ())), 1
        while frontier and level <= cap:
            nxt = []
            for kind in (k.strip().lower() for k in frontier):
                if not kind or kind in seen:
                    continue
                seen.add(kind)
                if exam.by_sp.get((kind, predicate)):
                    return level
                nxt.extend(exam.kinds.get(kind, ()))
            frontier, level = nxt, level + 1
        return None

    at_bound = past_bound = 0
    answered_at = answered_past = 0
    for subject in list(exam.kinds)[:900]:
        for predicate in INHERITABLE:
            if exam.by_sp.get((subject, predicate)):
                continue
            level = nearest(subject, predicate, MAX_INHERITED_HOPS + 2)
            if level is None:
                continue
            ok = core.predict(subject, predicate).ok
            if level <= MAX_INHERITED_HOPS:
                at_bound += 1
                answered_at += bool(ok)
            else:
                past_bound += 1
                answered_past += bool(ok)

    assert at_bound > 50, "not enough affordable chains sampled to say anything"
    assert answered_at == at_bound, (
        f"{at_bound - answered_at} chains within the bound went unanswered; "
        f"MAX_INHERITED_HOPS={MAX_INHERITED_HOPS} is too high")
    if past_bound:
        assert answered_past == 0, (
            f"{answered_past} of {past_bound} chains past the bound answered; "
            f"MAX_INHERITED_HOPS={MAX_INHERITED_HOPS} is too low and is excluding real work")


def test_what_the_bound_excludes_is_counted_not_hidden(exam):
    """A bound that silently shrinks the paper is a bound that can widen unnoticed."""
    report = exam.sit(("inheritance",))
    assert report.priced_out > 0, "this corpus does have chains past the bound"
    assert "not asked" in render(report)
    assert report.to_dict()["priced_out"] == report.priced_out


def test_inheritance_is_answered_by_the_ladder_and_not_by_english(exam):
    """The gap the report exists to print: she can derive it and cannot be asked for it."""
    report = exam.sit(("inheritance",))
    paper = report.paper("inheritance")
    assert paper is not None and paper.right > 0
    assert paper.by_english == 0
    assert paper.by_derivation == paper.right


def test_the_examination_is_deterministic(brain):
    """Same seed, same questions, same score — or a moved number means nothing."""
    first = GeneralKnowledgeExam(brain, limit=40).sit(("taxonomy", "inheritance"))
    second = GeneralKnowledgeExam(brain, limit=40).sit(("taxonomy", "inheritance"))
    assert [p.to_dict() for p in first.papers] == [p.to_dict() for p in second.papers]


def test_nothing_in_the_examination_writes_to_the_brain(brain):
    """Two sittings must find the store exactly as they left it."""
    before = sum(len(v) for v in brain.grounder.facts.values())
    GeneralKnowledgeExam(brain, limit=40).sit()
    GeneralKnowledgeExam(brain, limit=40).sit()
    assert sum(len(v) for v in brain.grounder.facts.values()) == before


def test_the_report_renders(exam):
    text = render(exam.sit(("recall", "soundness")))
    assert "recall" in text and "silence is the pass" in text
    assert "overall" in text


def test_the_module_entry_point(capsys):
    assert njp_general.main(["--limit", "5", "--paper", "recall", "--json"]) == 0
    out = capsys.readouterr().out
    assert '"paper": "recall"' in out


def test_examine_is_the_one_call_a_script_needs(brain):
    report = examine(limit=20, papers=("membership",), brain=brain)
    assert report.paper("membership") is not None
    assert report.asked == 20


# --------------------------------------------------------------------------- #
# The hard half — answers that are in no fact
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("paper", GeneralKnowledgeExam.HARD)
def test_a_hard_item_is_not_answerable_by_lookup(exam, paper):
    """The whole claim of these papers: no single stored fact answers the question.

    Checked structurally rather than trusted. If any item's gold could be read straight off the
    subject under the asked relation, the paper is measuring recall wearing a hat — which is
    exactly what `taxonomy` had to be rescued from.
    """
    items = exam.items(paper)
    assert items, paper
    for item in items:
        stored = {o.strip().lower() for o in exam.by_sp.get((item.subject, item.predicate), ())}
        if not stored:
            continue
        assert not (stored & {g.strip().lower() for g in item.gold}), item.question


def test_a_constructed_answer_must_carry_its_derivation(brain):
    """An answer with no steps is refused before it is compared.

    A right answer arrived at by no visible route is indistinguishable from a lucky string match,
    and counting it would make the score mean the thing it is supposed to prove.
    """
    from nyxara.njp.puzzle import PuzzleSolver, Solution

    solved = PuzzleSolver(brain).solve("what do a whale and a bat have in common?")
    assert solved.ok and solved.steps
    assert not Solution(form="bridge", answer="something", steps=()).ok


def test_the_hard_papers_are_answered_by_construction_and_nothing_else(exam):
    """No hard item may be answered through the question grammar or the inheritance ladder."""
    report = exam.sit(GeneralKnowledgeExam.HARD)
    for paper in report.papers:
        assert paper.by_english == 0, paper.name
        assert paper.by_derivation == 0, paper.name
        assert paper.by_construction == paper.right, paper.name


def test_the_named_hard_problems(brain):
    """The worked examples the module docstring stands on, asked of the live corpus."""
    from nyxara.njp.puzzle import PuzzleSolver

    solver = PuzzleSolver(brain)
    # A five-step walk across three domains. Nothing states the Taj Mahal's currency.
    money = solver.solve("What is the currency of the country where the Taj Mahal is?")
    assert money.ok and "rupee" in money.answer.lower()
    assert len(money.steps) >= 4

    # The container word is not decoration: it says where the walk stops, and changing it must
    # change the answer even though the walk starts in the same place.
    country = solver.solve("what is the capital of the country that agra is in?")
    state = solver.solve("what is the capital of the indian state that agra is in?")
    assert country.ok and state.ok
    assert country.answer != state.answer

    assert solver.solve("which mammal can fly?").answer == "bat"
    assert solver.solve("which one does not belong: copper, iron, gold, oxygen?").answer == "oxygen"
    assert solver.solve("sparrow is to bird as tiger is to what?").answer == "mammal"
    assert solver.solve("what does smoking eventually cause?").ok


def test_a_question_it_cannot_read_is_refused(brain):
    """A solver that answers every string is a solver whose score means nothing."""
    from nyxara.njp.puzzle import PuzzleSolver

    solver = PuzzleSolver(brain)
    for question in ("", "hello there", "what is the capital of France?",
                     "why is the sky blue?", "what do quokka and narwhal have in common?"):
        assert not solver.solve(question).ok, question


def test_a_list_item_containing_and_is_not_split_in_half():
    """"crime and punishment" is one novel, not two things."""
    from nyxara.njp.puzzle import PuzzleSolver

    assert PuzzleSolver.__dict__["_items"].__func__(
        "crime and punishment, don quixote, great expectation, fixed cost") == [
        "crime and punishment", "don quixote", "great expectation", "fixed cost"]


def test_a_qualifier_cannot_be_swallowed_by_a_substring():
    """"the largest ocean" is not inside "the second largest ocean" as whole words."""
    from nyxara.njp.puzzle import PuzzleSolver

    contains = PuzzleSolver.__dict__["_contains"].__func__
    assert contains("the second largest ocean", "largest ocean")
    assert not contains("the second largest ocean", "the largest ocean")
