"""The coding faculty: what she may run, what she must refuse, and what a lesson leaves behind.

Three claims are under test here and they are not the same claim.

The first is **safety**, and it is the one that has to hold unconditionally: a program is a term
tree, the only machine that runs one is her own interpreter, and the only route from text to a
term tree is a whitelist. So no test here asserts that dangerous source is *handled* — they assert
it does not parse at all.

The second is **honesty**: a program that fits the shown examples and fails a held-out one is a
coincidence, and the split exists to call it one. `write` returns nothing rather than something
unverified, and that abstention is tested as a result rather than as a failure.

The third is **acquisition**, and it is the only one that needs a before and an after. A shape is
demonstrated on one task and the *same shape* is then required on a different task, with different
constants, a different inner function and different data — and the untaught coder, given the same
budget, is asked for the same thing and does not find it.
"""

from __future__ import annotations

import pytest

from nyxara.njp.coding import (
    Call,
    Coder,
    CodeError,
    Exhausted,
    Interpreter,
    Lambda,
    Lit,
    Program,
    Spec,
    Var,
    abstract,
    read_python,
    render,
)
from nyxara.njp.teacher import Verdict


def _spec(name, ref, rows, shown=4):
    rows = [tuple(r) for r in rows]
    return Spec.of(name, "", ["xs"],
                   [([r], ref(r)) for r in rows[:shown]],
                   [([r], ref(r)) for r in rows[shown:]])


DATA = [(1, 2, 3, 4), (5, 6, 7, 8), (2, 2, 9), (10, 1, 4, 6), (3, 3, 3), (12, 5, 0, 7)]


# --------------------------------------------------------------------------- #
# the language runs, and it stops
# --------------------------------------------------------------------------- #

def test_a_program_is_a_term_tree_and_runs_in_her_own_interpreter():
    coder = Coder()
    program = read_python("def f(xs):\n    return sum([x * 2 for x in xs if x % 2 == 0])")
    assert coder.run(program, ((1, 2, 3, 4),)) == 12


def test_every_program_halts():
    """There is no open recursion in the language, and the budget is a hard stop rather than a
    timer — so the same program costs the same on any machine."""
    interpreter = Interpreter(max_steps=20)
    with pytest.raises(Exhausted):
        interpreter.run(Call("map", (Lambda(("x",), Call("mul", (Var("x"), Lit(2)))),
                                     Lit(tuple(range(100))))))


def test_a_conditional_does_not_evaluate_the_branch_it_did_not_take():
    """Eager evaluation here turns a correct program into a CodeError, which is why `if`, `and`
    and `or` are handled by the interpreter rather than sitting in the operator table."""
    coder = Coder()
    safe = Program("f", ("n",), Call("if", (Call("gt", (Var("n"), Lit(0))),
                                            Call("div", (Lit(10), Var("n"))), Lit(0))))
    assert coder.run(safe, (0,)) == 0
    assert coder.run(safe, (5,)) == 2


def test_a_predicate_cannot_leak_into_arithmetic():
    """Python says ``True + True == 2``. A language that agreed would turn a type error into a
    plausible wrong answer, which is the one failure a verifier cannot catch."""
    coder = Coder()
    with pytest.raises(CodeError):
        coder.run(Program("f", ("x",), Call("add", (Lit(True), Lit(True)))), (1,))


@pytest.mark.parametrize("source", [
    "def f(x):\n    return __import__('os').system('ls')",
    "def f(x):\n    return open('/etc/passwd').read()",
    "def f(x):\n    return eval('2+2')",
    "def f(x):\n    return exec('x = 1')",
    "def f(x):\n    return compile('1', '', 'eval')",
    "def f(x):\n    return x.__class__.__bases__",
    "def f(x):\n    return x.__dict__",
    "def f(x):\n    return getattr(x, 'y')",
    "def f(x):\n    return globals()",
    "def f(x):\n    return vars()",
    "def f(x):\n    return dir(x)",
    "def f(x):\n    return type(x)",
    "def f(x):\n    return input()",
    "def f(x):\n    return print(x)",
    "import os\n",
    "from os import path\n",
    "def f(x):\n    import sys\n    return sys",
    "class A:\n    pass",
    "def f(x):\n    with open('a') as h:\n        return h.read()",
    "def f(x):\n    try:\n        return 1\n    except Exception:\n        return 2",
    "def f(x):\n    global y\n    return x",
    "def f(x):\n    del x\n    return 1",
    "def f(x):\n    assert x\n    return 1",
    "def f(x):\n    yield x",
    "async def f(x):\n    return x",
    "def f(x):\n    xs.pop()\n    return x",
])
def test_source_outside_the_whitelist_does_not_parse(source):
    """The refusal is at parse time. Nothing dangerous is ever *handled*, because nothing
    dangerous is ever built — and the accepted set is enumerated rather than the rejected one, so
    a new Python node type cannot quietly widen it.

    ``xs.pop()`` is in the list for a different reason from the rest. It is not dangerous; it is a
    statement this language cannot *give effect to*, and dropping it silently would mean reading a
    program as something it is not. A reader that quietly discards a line it does not understand
    reports a wrong program as a correct one, so an unrecognised statement is refused too.
    """
    with pytest.raises(CodeError):
        read_python(source)


@pytest.mark.parametrize("source,args,want", [
    ("def f(xs):\n    return sorted(xs)[::-1]", ((3, 1, 2),), (3, 2, 1)),
    ("def f(xs):\n    return xs[1:]", ((7, 8, 9),), (8, 9)),
    ("def f(a, b):\n    return max(a, b) - min(a, b)", (4, 11), 7),
    ("def f(n):\n    return n if n > 0 else -n", (-6,), 6),
    ("def f(s):\n    return len(s.split())", ("a b c",), 3),
    ("lambda xs: sum([x % 2 for x in xs])", ((1, 2, 3),), 2),
])
def test_the_reader_covers_the_python_a_person_actually_writes(source, args, want):
    assert Coder().run(read_python(source), args) == want


# --------------------------------------------------------------------------- #
# a shape is what survives a lesson
# --------------------------------------------------------------------------- #

def test_abstraction_keeps_the_skeleton_and_throws_the_specifics_away():
    program = read_python("def f(xs):\n    return sum([x * 3 for x in xs if x % 2 == 1])")
    shape = abstract(program)
    assert shape.key == "sum(map(?fn1#0, filter(?fn1#1, #0)))"
    assert shape.arity == 1 and shape.cost == 2


def test_two_different_iteration_shapes_are_two_different_keys():
    """Regression: `map`, `filter` and `fold` carried empty rendering templates, a schema's key
    *is* its rendering, and every higher-order shape with a blanked function collapsed onto the
    single key ``sum()``. One key is one shape as far as learning is concerned."""
    keys = {abstract(read_python(source)).key for source in (
        "def f(xs):\n    return sum([x * 2 for x in xs])",
        "def f(xs):\n    return sum([x for x in xs if x > 3])",
        "def f(xs):\n    return len([x for x in xs if x > 3])",
    )}
    assert len(keys) == 3


def test_a_demonstration_that_fails_its_own_examples_teaches_nothing():
    """A teacher is not trusted here any more than anywhere else. What a bad demonstration buys is
    a mark *against* the shape, never the shape."""
    coder = Coder()
    spec = _spec("total", sum, DATA)
    learned = coder.learn_python(spec, "def total(xs):\n    return len(xs)")
    assert learned.verdict == Verdict.REFUTED
    assert not coder.taught_shapes()


def test_an_unreadable_demonstration_is_undecided_rather_than_wrong():
    coder = Coder()
    learned = coder.learn_python(_spec("total", sum, DATA),
                                 "def total(xs):\n    return __import__('os').listdir('.')")
    assert learned.verdict == Verdict.UNDECIDED


def test_a_verified_demonstration_records_the_shape_and_not_the_answer():
    coder = Coder()
    spec = _spec("odds", lambda xs: sum(x * 3 for x in xs if x % 2 == 1), DATA)
    learned = coder.learn_python(spec, "def odds(xs):\n    return sum([x*3 for x in xs if x % 2 == 1])")
    assert learned.verdict == Verdict.SURVIVED
    assert learned.moved > 0
    shapes = coder.taught_shapes()
    assert len(shapes) == 1 and "?fn1" in shapes[0]["key"]
    # the demonstrated task is nowhere in what she kept
    assert "odds" not in shapes[0]["key"]


# --------------------------------------------------------------------------- #
# writing, and refusing to write
# --------------------------------------------------------------------------- #

def test_she_writes_a_primitive_from_examples_alone():
    written = Coder().write(_spec("total", sum, DATA), attempts=4000)
    assert written.ok
    assert Coder().check(written.program, _spec("total", sum, DATA).held_out).ok


def test_an_impossible_task_is_an_abstention_not_a_guess():
    """No best-effort return. A program that nearly works is a wrong one that cost more to find."""
    coder = Coder()
    spec = Spec.of("impossible", "", ["xs"],
                   [([(1, 2)], 7), ([(1, 2)], 9)])       # the same input, two answers
    written = coder.write(spec, attempts=800, graft=False)
    assert not written.ok and written.program is None
    assert coder.stats()["abstained"] == 1


def test_a_shape_taught_on_one_task_is_used_on_another():
    """The acquisition claim, with a control: the same spec, the same budget, an untaught coder."""
    taught = Coder()
    lesson = _spec("triples", lambda xs: sum(x * 3 for x in xs if x % 2 == 1), DATA)
    assert taught.learn_python(
        lesson, "def triples(xs):\n    return sum([x*3 for x in xs if x % 2 == 1])"
    ).verdict == Verdict.SURVIVED

    unseen = _spec("doubles", lambda xs: sum(x * 2 for x in xs if x % 2 == 0), DATA)
    after = taught.write(unseen, attempts=20000, graft=False)
    cold = Coder().write(unseen, attempts=20000, graft=False)

    assert after.ok, "a demonstrated shape should carry to a task nobody demonstrated"
    assert taught.check(after.program, unseen.held_out).ok, "and it must hold on held-out data"
    assert not cold.ok, "the same budget without the lesson should not find it"


def test_a_composite_nobody_taught_is_reached_by_invention_not_by_grafting():
    """Two routes to a shape she was never shown, and they cost very different amounts.

    This test used to assert that a small budget could *not* reach ``sum(filter(even, xs))``.
    That was true when grafting — composing two seed shapes and enumerating the result — was the
    only route, and it is no longer true: ``Coder.invent`` builds the composition out of the
    grammar directly and finds it in a fraction of the budget. So the assertion is inverted here
    rather than deleted, and the old claim is kept beside it with invention switched off, because
    both are still facts about the search and the difference between them is the point.
    """
    spec = _spec("evens", lambda xs: sum(x for x in xs if x % 2 == 0), DATA)

    inventor = Coder()
    quick = inventor.write(spec, attempts=1200, graft=True)
    assert quick.ok and inventor.invented == 1, quick.note
    assert inventor.check(quick.program, spec.held_out).ok

    # The old route, on its own: grafting reaches it, and pays for it.
    assert not Coder().write(spec, attempts=1200, graft=True, invent=False).ok
    patient = Coder().write(spec, attempts=200000, graft=True, invent=False)
    assert patient.ok and patient.grafted
    assert patient.attempts > 1200


def test_held_out_examples_catch_a_program_fitted_to_the_shown_ones():
    """The split does real work: a program can match four pairs by coincidence, and the fifth is
    how anyone finds out."""
    coder = Coder()
    program = read_python("def f(xs):\n    return sum([x for x in xs if x > 3])")
    spec = _spec("over_nine", lambda xs: sum(x for x in xs if x > 9), DATA)
    assert not coder.check(program, spec.held_out).ok


# --------------------------------------------------------------------------- #
# reading a program back, and repairing one
# --------------------------------------------------------------------------- #

def test_a_trace_is_a_derivation_rather_than_a_number():
    coder = Coder()
    program = read_python("def f(xs):\n    return sum([x * 2 for x in xs])")
    steps = coder.trace(program, ((1, 2, 3),))
    assert steps and steps[-1].value == 12
    assert any("for x in" in step.source for step in steps)


def test_one_edit_repairs_a_program_and_the_fix_has_to_run():
    coder = Coder()
    spec = _spec("over_nine", lambda xs: sum(x for x in xs if x > 9), DATA)
    broken = read_python("def f(xs):\n    return sum([x for x in xs if x > 3])", name="over_nine")
    fixed = coder.repair(broken, spec, attempts=20000)
    assert fixed.ok
    assert coder.check(fixed.program, spec.held_out).ok


def test_explaining_a_program_says_what_it_does():
    coder = Coder()
    said = coder.explain(read_python("def f(xs):\n    return sum([x for x in xs if x > 3])"))
    assert "total" in said and "holds" in said


def test_rendering_round_trips_through_the_reader():
    program = read_python("def f(xs):\n    return sum([x * 2 for x in xs if x % 2 == 0])")
    again = read_python(f"lambda xs: {render(program.body)}")
    assert Coder().run(again, ((1, 2, 3, 4),)) == 12


# --------------------------------------------------------------------------- #
# wired into the brain, not sitting beside it
# --------------------------------------------------------------------------- #

def test_the_brain_holds_the_faculty_and_reports_it():
    """An organ nothing can reach is the failure mode this package names by name. It is built, it
    is reported in `stats()`, and the four verbs are on the brain rather than one import away."""
    from nyxara.njp.brain import NJPBrain
    from nyxara.njp.school import ExamConditions

    brain = NJPBrain(ExamConditions())
    assert brain.coder is not None
    assert "coding" in brain.stats()
    assert brain.run_code("def f(xs):\n    return sum(xs) * 2", (1, 2, 3)) == 12
    assert brain.trace_code("def f(xs):\n    return sum(xs)", (1, 2))[-1].value == 3
    with pytest.raises(CodeError):
        brain.read_code("def f(x):\n    return __import__('os').getcwd()")


def test_the_brain_learns_a_shape_and_then_writes_with_it():
    from nyxara.njp.brain import NJPBrain
    from nyxara.njp.school import ExamConditions

    brain = NJPBrain(ExamConditions())
    lesson = _spec("triples", lambda xs: sum(x * 3 for x in xs if x % 2 == 1), DATA)
    assert brain.learn_code(
        lesson, "def triples(xs):\n    return sum([x*3 for x in xs if x % 2 == 1])"
    ).verdict == Verdict.SURVIVED

    unseen = _spec("doubles", lambda xs: sum(x * 2 for x in xs if x % 2 == 0), DATA)
    written = brain.write_code(unseen, attempts=20000)
    assert written.ok and brain.coder.check(written.program, unseen.held_out).ok
    assert brain.stats()["coding"]["taught"] == 1
