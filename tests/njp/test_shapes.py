"""V.57 — the shapes she induced, and the guards that stop a shape from being nothing.

Alignment is the one piece of this package that reads the dataset without anybody having written
down what its rows look like, so what these tests pin is not accuracy but *refusal*: a shape that
cannot read its own rows back, a shape that is one hole covering everything, a shape proposed from
a coincidence that a third row would have killed. Each of those has to be caught here, because on
the reconstruction exam every one of them scores perfectly.
"""

from __future__ import annotations

from nyxara.njp.shapes import MAX_SLOTS, Group, Shape, align, induce
from nyxara.njp.shapeschool import LEARN_ROWS, _rebuild, examine, grade

TEMPLATE = ("In this task, you are given a question and a context passage. "
            "You have to answer the question based on the given passage."
            "\nQuestion: {}\nContext passage: {}")

#: No two of these share an opening run, and that is not incidental. An anchor is proposed from the
#: first pair *whole*, so if rows one and two both began `wh` the proposal would be `"\nQuestion: wh"`
#: and row five would drop it entirely rather than trim it back — see
#: :func:`test_an_anchor_the_first_pair_over_proposed_is_trimmed_not_dropped`.
FILLERS = (
    ("who wrote the book", "It was written by Ada in the spring of that year."),
    ("did it happen at all", "The event took place on a Tuesday, after the rains stopped."),
    ("name the town she left", "She was born in a small town some way north of the river."),
    ("what colour was it", "The door had been painted green, then blue, then green again."),
    ("how many were there", "Eleven of them arrived, and two more came later in the evening."),
    ("give a reason for it", "They left because the lease was up and nobody renewed it."),
)


def a_group(fillers=FILLERS, targets=None) -> Group:
    return Group(task="task_made_up", template="0", source="test",
                 prompts=tuple(TEMPLATE.format(q, c) for q, c in fillers),
                 targets=tuple(targets or ["yes", "no", "yes", "no", "yes", "no"][:len(fillers)]))


# --------------------------------------------------------------------------------------------- #
#  what alignment finds
# --------------------------------------------------------------------------------------------- #
def test_it_recovers_the_template_nobody_told_it_about():
    """The instruction is what did not vary. It was never written down here as a pattern."""
    shape = induce(a_group())
    assert shape is not None
    assert "In this task, you are given a question" in shape.constant
    assert "\nContext passage:" in shape.constant


def test_it_finds_the_two_holes():
    shape = induce(a_group())
    assert len(shape.slots) == 2


def test_it_reads_a_row_it_was_not_induced_from():
    """A template induced from six rows either parses a seventh or it has found something else."""
    shape = induce(a_group(FILLERS[:4]))
    got = shape.read(TEMPLATE.format("which one was first", "The first was the one on the left."))
    assert got == ["which one was first", "The first was the one on the left."]


def test_a_row_of_another_shape_is_refused_rather_than_forced():
    shape = induce(a_group())
    assert shape.read("Translate to French: the cat sat down.") is None


# --------------------------------------------------------------------------------------------- #
#  what it refuses
# --------------------------------------------------------------------------------------------- #
def test_two_rows_propose_an_anchor_that_a_third_row_kills():
    """Any two English passages share ' and the ' somewhere. Three do not, and that is the point."""
    pair = ["the cat and the dog went out", "a bird and the fish came in"]
    assert align(pair, least=6)
    assert not [p for p in align(pair + ["nothing whatever in common here"], least=6)
                if isinstance(p, str)]


def test_unrelated_rows_yield_no_shape():
    junk = Group(task="junk", prompts=("alpha beta gamma", "delta epsilon zeta",
                                       "eta theta iota", "kappa lambda mu"))
    assert induce(junk) is None


def test_every_shape_returned_can_read_its_own_rows_back():
    """The invariant :func:`induce` refuses on, stated as the property rather than as a case.

    A template that parses a row but recovers fields that do not rebuild it has found a pattern
    that is not the template, and is worse than finding nothing — it scores well on everything
    downstream. So the guarantee is checked over a spread of groups, including awkward ones whose
    anchors repeat within a row.
    """
    groups = [a_group(), a_group(FILLERS[:4]),
              Group(task="repeats", prompts=tuple(f"X: {n} Y: {n} X: {n} Y: {n}" for n in "abcd")),
              Group(task="ragged", prompts=("A: 1 B: one", "A: 22 B: two",
                                            "A: 333 B: three", "A: 4444 B: four"))]
    for group in groups:
        shape = induce(group, least=3)
        if shape is None:
            continue
        back = [shape.read(p) for p in group.prompts]
        rebuilt = sum(1 for p, got in zip(group.prompts, back)
                      if got is not None and _rebuild(shape, got) == p)
        assert rebuilt >= max(2, len(group.prompts) // 2), (group.task, shape.render())


def test_too_many_slots_is_not_a_template():
    shape = induce(a_group())
    assert len(shape.slots) <= MAX_SLOTS


# --------------------------------------------------------------------------------------------- #
#  an answer space is a space because rows reuse it
# --------------------------------------------------------------------------------------------- #
def test_repeated_answers_are_a_space():
    shape = induce(a_group(targets=["yes", "no", "yes", "no", "yes", "no"]))
    assert shape.answer_space == ("no", "yes")


def test_six_different_answers_are_free_text_not_a_space():
    """`task1295` was reported as answering from four free-text spans because nothing repeated."""
    spans = ["Antarctica", "Barrio Sur", "mediums", "the existence of Ishvara",
             "a second Tuesday", "the older of the two"]
    assert induce(a_group(targets=spans)).answer_space == ()


# --------------------------------------------------------------------------------------------- #
#  the exam, and the floor under it
# --------------------------------------------------------------------------------------------- #
def test_held_out_rows_are_returned_character_for_character():
    report = grade([a_group()])
    assert report.rows == len(FILLERS) - LEARN_ROWS
    assert report.reconstructs == 1.0


def test_the_degenerate_shape_is_perfect_and_is_counted_apart():
    """One hole over the whole prompt reconstructs everything and knows nothing."""
    got = examine([a_group()])
    assert got["one_slot"].reconstructs == 1.0
    assert got["one_slot"].degenerate == 1.0
    assert got["induced"].degenerate == 0.0


def test_the_induced_shape_holds_template_text_and_the_degenerate_one_holds_none():
    got = examine([a_group()])
    assert got["induced"].held > 80
    assert got["one_slot"].held == 0.0


def test_a_group_that_yields_no_shape_is_counted_not_hidden():
    junk = Group(task="junk", prompts=tuple("abcdefghij"[i] * 9 + str(i) for i in range(6)))
    report = grade([a_group(), junk])
    assert report.groups == 2
    assert report.shaped == 1
    assert report.coverage == 0.5
