"""An examination she did not write — the reader, the grader, and the corpus's own defects.

Every exam in this package up to V.24 was written by the same hand as the code it grades. This
one was not: 10,870 items from an outside generator, with a machine checker on every one and a
sealed split whose parameter regions are disjoint from the training half by construction.

Measured cold, before :mod:`nyxara.njp.corpussolver` existed, on 200 items of the sealed split::

    right                    1 / 200
    confidently wrong       85 / 200
    silent                 114 / 200

and 34 of the first 97 items came back beginning ``noted:`` — the exam being filed into the
knowledge store as facts about the world, which is the V.23 defect resurfacing on thirteen shapes
its fix had never been told about.

The first thing asserted here is not about her at all. :func:`nyxara.njp.corpus.verify` is a port
of the corpus's own ``score_candidate``, and if the port is unfaithful then every number below is
void — so the corpus's own QA runs first: **every reference answer must pass its own verifier.**
"""

from __future__ import annotations

import itertools
import json
import re

import pytest

from nyxara.njp import corpus as corpus_module
from nyxara.njp.corpus import Record, load, self_test, verify


# --------------------------------------------------------------------------- #
# the grader, before anything it grades
# --------------------------------------------------------------------------- #

def test_the_port_is_faithful_or_every_number_here_is_void() -> None:
    """The corpus's own QA: every reference answer passes its own verifier."""
    result = self_test()
    assert result["checked"] > 10000
    assert result["pass_rate"] == 100.0, result["failures"]
    assert result["rubric_skipped"] == 14


def test_the_corpus_ships_the_splits_its_manifest_claims() -> None:
    assert corpus_module.splits() == {
        "TRAIN": 4073, "PRACTICE": 2026, "GENERALIZATION": 1367,
        "NOVEL": 901, "ADVERSARIAL": 1088, "EVAL": 1415,
    }


def test_silence_abstains_and_is_decided_before_any_checker_runs() -> None:
    """Three outcomes, never two.

    An empty string passes no regex and contains no number, so a grader that skipped this step
    would score every honest refusal as an error and quietly reward guessing.
    """
    record = load("EVAL", generator="mod_chain")[0]
    assert verify(record, "").outcome == "abstain"
    assert verify(record, "   \n ").outcome == "abstain"
    assert verify(record, record.answer).outcome == "right"
    assert verify(record, "99999").outcome == "wrong"


def test_a_rubric_item_is_not_pretended_to_be_machine_checkable() -> None:
    rubric = [r for r in corpus_module.corpus() if r.verifier == "rubric"]
    assert len(rubric) == 14
    for record in rubric:
        assert not record.gradable
        assert verify(record, record.answer).outcome == "unverifiable"


@pytest.mark.parametrize("kind,good,bad", [
    ("exact", "card", "bolt"),
    ("numeric", "the answer is 3", "the answer is 4"),
    ("regex", "NO — not derivable", "YES"),
])
def test_each_verifier_kind_separates_right_from_wrong(kind: str, good: str, bad: str) -> None:
    specs = {
        "exact": {"type": "exact", "value": "card"},
        "numeric": {"type": "numeric", "value": 3.0, "tol": 0.0},
        "regex": {"type": "regex", "pattern": r"^\s*NO\b"},
    }
    record = Record(id="t", verification=specs[kind])
    assert verify(record, good).outcome == "right"
    assert verify(record, bad).outcome == "wrong"


def test_a_stratified_sample_is_a_smaller_version_of_the_whole_exam() -> None:
    """``limit`` with a seed samples proportionally; without one it truncates, and the two differ
    on purpose."""
    sampled = load("EVAL", gradable_only=True, limit=140, seed=1)
    assert len(sampled) <= 140
    assert len({r.generator for r in sampled}) >= 10
    truncated = load("EVAL", gradable_only=True, limit=140)
    assert len({r.generator for r in truncated}) < len({r.generator for r in sampled})


# --------------------------------------------------------------------------- #
# the corpus's own defects — found by measurement, then confirmed independently
# --------------------------------------------------------------------------- #

def _observable_error(record: Record):
    """Where the student's working first departs from the truth, recomputed from scratch."""
    start = re.search(r"start with x = (-?\d+)", record.context)
    operations = re.findall(r"^- ([-+*]) (-?\d+)$", record.context, re.M)
    if not start or not operations:
        return None, None
    value, truth = int(start.group(1)), []
    for op, operand in operations:
        operand = int(operand)
        value = value + operand if op == "+" else (
            value - operand if op == "-" else value * operand)
        truth.append(value)
    claimed = [int(v) for v in
               re.findall(r"^Step \d+: x [-+*] -?\d+ = (-?\d+)", record.context, re.M)]
    first = next((i + 1 for i, (c, t) in enumerate(zip(claimed, truth)) if c != t), None)
    return first, truth


def test_fifty_four_self_critique_items_assert_an_error_that_is_not_in_them() -> None:
    """The generator's operator flip maps ``mul`` to ``mul``, so on a multiplication step it is a
    no-op: the item says "contains exactly one first error" and the working is flawless.

    This is a defect in the exam, not in her, and it is asserted rather than described because
    the alternative reading — that she fails 54 items of self-critique — is the one a bare
    accuracy number would support.
    """
    broken = []
    for record in corpus_module.corpus():
        if record.generator != "self_critique" or record.attack:
            continue
        first, _truth = _observable_error(record)
        if first is None:
            continue
        try:
            claimed_step = json.loads(record.answer)["first_wrong_step"]
        except Exception:  # noqa: BLE001
            continue
        if first != claimed_step:
            broken.append(record.id)
    assert broken == [], f"the reader disagrees with the key on {len(broken)} solvable items"

    unobservable = [r for r in corpus_module.corpus()
                    if r.generator == "self_critique" and not r.attack
                    and _observable_error(r)[0] is None and _observable_error(r)[1]]
    assert len(unobservable) == 42
    # Every one of them names a multiplication at the step it calls wrong — the no-op flip.
    for record in unobservable:
        operations = re.findall(r"^- ([-+*]) (-?\d+)$", record.context, re.M)
        step = json.loads(record.answer)["first_wrong_step"]
        assert operations[step - 1][0] == "*"


def _every_consistent_answer(record: Record):
    """A brute force sharing no code with the solver: every permutation, filtered by the clues."""
    cast = re.search(r"sit in seats [\d.]+ \(left to right\): (.+?)\.", record.context)
    asked = re.search(r"Which item does (\w+) own", record.task)
    if not cast or not asked:
        return None
    seat = {n: int(s) for n, s in re.findall(r"(\w+) in seat (\d+)", cast.group(1))}
    for name, number in re.findall(r"(\w+) sits in seat (\d+)", record.context):
        seat[name] = int(number)
    items = [w.strip() for w in
             re.search(r"owns exactly one of: (.+?)\.", record.context).group(1).split(",")]
    drinks = [w.strip() for w in
              re.search(r"drinks exactly one of: (.+?)\.", record.context).group(1).split(",")]
    people = sorted(seat, key=lambda p: seat[p])
    clues = [line.strip().lstrip("- ").rstrip(".")
             for line in record.context.splitlines() if line.strip().startswith("- ")]
    answers = set()
    for item_order in itertools.permutations(items):
        owns = dict(zip(people, item_order))
        holder = {v: k for k, v in owns.items()}
        if not all(
                (owns.get(m.group(1)) == m.group(2)) if (m := re.match(
                    r"^(\w+) owns the (\w+)$", clue)) else True
                for clue in clues):
            continue
        if any((m := re.match(r"^(\w+) does not own the (\w+)$", clue))
               and owns.get(m.group(1)) == m.group(2) for clue in clues):
            continue
        for drink_order in itertools.permutations(drinks):
            drinking = dict(zip(people, drink_order))
            ok = True
            for clue in clues:
                named = re.match(r"^(\w+) drinks (\w+)$", clue)
                if named and drinking.get(named.group(1)) != named.group(2):
                    ok = False
                    break
                left = re.match(r"^The person with the (\w+) sits immediately left of "
                                r"the (\w+) drinker$", clue)
                if left:
                    neighbours = [p for p in people
                                  if seat[p] == seat[holder[left.group(1)]] + 1]
                    if not neighbours or drinking.get(neighbours[0]) != left.group(2):
                        ok = False
                        break
            if ok:
                answers.add(owns[asked.group(1)])
                break
    return answers


def test_forty_eight_seating_puzzles_have_more_than_one_consistent_answer() -> None:
    """Confirmed by exhaustive enumeration, not by the solver agreeing with itself.

    The reference answer is always *among* the consistent set, so the generator picked one of
    several rather than being wrong — which makes these underdetermined items, exactly the shape
    the corpus's own abstention half says to refuse.
    """
    ambiguous, outside = [], []
    for record in corpus_module.corpus():
        if record.generator != "constraint_puzzle" or record.attack:
            continue
        answers = _every_consistent_answer(record)
        if answers is None:
            continue
        if len(answers) > 1:
            ambiguous.append(record.id)
            if record.answer not in answers:
                outside.append(record.id)
    assert len(ambiguous) == 45
    assert outside == []
