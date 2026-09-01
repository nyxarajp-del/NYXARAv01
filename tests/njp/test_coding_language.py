"""The wider language: statements, loops, recursion, mappings, text — and what still halts.

:mod:`tests.njp.test_coding` covers the expression core, the whitelist, and the discipline that a
lesson keeps a shape rather than an answer. This file covers what the language grew into: local
variables, ``for`` and ``while``, functions that call themselves, mappings and sets, and a reader
that takes the Python a person actually writes.

Two properties are load-bearing and are tested as properties rather than as examples.

**It halts.** ``while True`` is now something she can write, so "every program terminates" stopped
being true by construction and became true by budget. Both budgets are exercised: the step counter
for a loop that never ends, and the call-depth counter for a recursion that never bottoms out.

**It agrees with Python.** The oracle is CPython — the same program run in a throwaway namespace —
rather than a table of expected answers, because a hand-written table is a table with a mistake in
it. One was written during this module's development and it marked a correct lesson as refuted.
"""

from __future__ import annotations

import pytest

from nyxara.njp.coding import (
    Coder,
    CodeError,
    Exhausted,
    Spec,
    abstract,
    read_python,
)
from nyxara.njp.tasks import EVERY_TASK, instance, normalise, reference
from nyxara.njp.teacher import Verdict


def run(source, *args):
    """Her answer."""
    return Coder().run(read_python(source), args)


def truth(source, *args):
    """Python's answer, from CPython itself."""
    return normalise(reference(source)(*args))


def agrees(source, *args):
    return run(source, *args) == truth(source, *args)


# --------------------------------------------------------------------------- #
# statements
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("source,args", [
    ("def f(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t", ((1, 2, 3),)),
    ("def f(xs):\n    out = []\n    for x in xs:\n        if x % 2:\n            out.append(x)\n"
     "    return out", ((1, 2, 3, 4),)),
    ("def f(n):\n    i = 0\n    t = 0\n    while i < n:\n        t += i\n        i += 1\n"
     "    return t", (6,)),
    ("def f(xs):\n    for x in xs:\n        if x < 0:\n            return x\n    return 0",
     ((1, -2, 3),)),
    ("def f(xs):\n    t = 0\n    for x in xs:\n        if x % 2:\n            continue\n"
     "        t += x\n    return t", ((1, 2, 3, 4),)),
    ("def f(n):\n    i = 0\n    while True:\n        if i * i > n:\n            break\n"
     "        i += 1\n    return i", (30,)),
    ("def f(m):\n    t = 0\n    for row in m:\n        for v in row:\n            t += v\n"
     "    return t", (((1, 2), (3, 4)),)),
    ("def f(xs):\n    a, b = xs[0], xs[1]\n    a, b = b, a\n    return [a, b]", ((1, 2),)),
    ("def f(xs):\n    ys = list(xs)\n    ys[0], ys[1] = ys[1], ys[0]\n    return ys", ((1, 2, 3),)),
    ("def f(n):\n    'a docstring'\n    if n:\n        pass\n    return n", (4,)),
])
def test_statements_agree_with_python(source, args):
    assert agrees(source, *args)


def test_parallel_assignment_is_one_statement():
    """``a, b = b, a % b`` is Euclid's algorithm. Two assignments would be a different program —
    the second would read what the first had just overwritten — and it would be wrong quietly."""
    assert run("def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a", 48, 18) == 6


# --------------------------------------------------------------------------- #
# recursion, and the fact that it stops
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("source,args", [
    ("def f(n):\n    if n <= 1:\n        return 1\n    return n * f(n - 1)", (6,)),
    ("def f(n):\n    if n < 2:\n        return n\n    return f(n - 1) + f(n - 2)", (12,)),
    ("def f(xs):\n    if len(xs) == 0:\n        return 0\n    return xs[0] + f(xs[1:])",
     ((1, 2, 3),)),
    ("def q(xs):\n    if len(xs) <= 1:\n        return list(xs)\n    p = xs[0]\n"
     "    lo = [x for x in xs[1:] if x < p]\n    hi = [x for x in xs[1:] if x >= p]\n"
     "    return q(lo) + [p] + q(hi)", ((5, 3, 8, 1),)),
    ("def a(m, n):\n    if m == 0:\n        return n + 1\n    if n == 0:\n        return a(m - 1, 1)\n"
     "    return a(m - 1, a(m, n - 1))", (2, 3)),
])
def test_recursion_agrees_with_python(source, args):
    assert agrees(source, *args)


def test_a_helper_is_reachable_and_nothing_else_is():
    """A program travels with the functions it may call, and with no others. There is no global
    scope and no import, so the registry *is* the whitelist, one level below the reader's."""
    source = "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n\n" \
             "def f(a, b):\n    return a * b // gcd(a, b)"
    assert run(source, 4, 6) == 12
    with pytest.raises(CodeError):
        run("def f(x):\n    return helper(x)", 1)


def test_a_loop_that_never_ends_is_exhausted_rather_than_hung():
    coder = Coder()
    coder.interpreter.max_steps = 5000
    program = read_python("def f(n):\n    while True:\n        n += 1\n    return n")
    with pytest.raises(Exhausted):
        coder.run(program, (0,))


def test_recursion_that_never_bottoms_out_is_exhausted_rather_than_a_crash():
    """The depth budget sits *below* CPython's own limit on purpose: one level here costs about
    ten host frames, so a budget set past ~90 is never reached — the interpreter dies of
    RecursionError first, which is not a CodeError and escapes every guard a search has."""
    program = read_python("def f(n):\n    return f(n + 1)")
    with pytest.raises(Exhausted):
        Coder().run(program, (0,))


# --------------------------------------------------------------------------- #
# mappings, sets, text
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("source,args", [
    ("def f(ws):\n    d = {}\n    for w in ws:\n        d[w] = d.get(w, 0) + 1\n    return d",
     (("a", "b", "a"),)),
    ("def f(d):\n    return {v: k for k, v in d.items()}", ({"a": 1, "b": 2},)),
    ("def f(d):\n    return {k: v for k, v in d.items() if v > 1}", ({"a": 1, "b": 2},)),
    ("def f(d):\n    return sum(d.values())", ({"a": 1, "b": 2},)),
    ("def f(d):\n    return max(sorted(d.keys()), key=lambda k: d[k])", ({"a": 1, "b": 5},)),
    ("def f(a, b):\n    return sorted(set(a) & set(b))", ((1, 2, 3), (2, 3, 4))),
    ("def f(a, b):\n    return sorted((set(a) | set(b)) - (set(a) & set(b)))", ((1, 2), (2, 3))),
    ("def f(xs):\n    return {x for x in xs if x > 2}", ((1, 2, 3, 4),)),
    ("def f(s):\n    return ' '.join([w.title() for w in s.split()])", ("the quick fox",)),
    ("def f(s):\n    return s.replace('a', 'b').upper()", ("cat",)),
    ("def f(s, k):\n    out = ''\n    for ch in s:\n        out = out + chr((ord(ch) - 97 + k)"
     " % 26 + 97)\n    return out", ("abc", 2)),
    ("def f(s):\n    return f'{s} has {len(s)}'", ("abc",)),
    ("def f(s):\n    return s.count('ab')", ("abcab",)),
    ("def f(xs):\n    return [a + b for a, b in xs]", (((1, 2), (3, 4)),)),
    ("def f(a, b):\n    return [x * y for x, y in zip(a, b)]", ((1, 2), (3, 4))),
])
def test_containers_and_text_agree_with_python(source, args):
    assert agrees(source, *args)


def test_a_slice_of_a_string_is_a_string():
    """Returning a tuple of characters is the kind of nearly-right that shows up three operations
    later as a type error in unrelated code."""
    assert run("def f(s):\n    return s[1:]", "abc") == "bc"
    assert run("def r(s):\n    if len(s) <= 1:\n        return s\n    return r(s[1:]) + s[0]",
               "abc") == "cba"


def test_plus_is_decided_by_the_values_not_by_the_source():
    """The reader used to choose between addition and concatenation statically and guessed wrong
    on ``out + chr(c)``. Mixed kinds are still an error, exactly as in Python."""
    assert run("def f(a, b):\n    return a + b", 2, 3) == 5
    assert run("def f(a, b):\n    return a + b", "a", "b") == "ab"
    assert run("def f(a, b):\n    return a + b", (1,), (2,)) == (1, 2)
    with pytest.raises(CodeError):
        run("def f(a, b):\n    return a + b", "a", 1)


def test_a_set_comes_back_in_a_settled_order():
    """Python iterates a set in an order that depends on hashing, so ``list(set(xs))`` is a
    different answer on a different run. A language whose verification story is "run it on the
    examples" cannot have an operation whose answer moves."""
    once = run("def f(xs):\n    return [x for x in set(xs)]", (3, 1, 2, 1))
    again = run("def f(xs):\n    return [x for x in set(xs)]", (3, 1, 2, 1))
    assert once == again == (1, 2, 3)


# --------------------------------------------------------------------------- #
# the whole bank
# --------------------------------------------------------------------------- #

def test_she_reads_and_runs_every_worked_solution_in_the_syllabus():
    """Sixty-five families across loops, recursion, mappings, text, nested data and the classic
    algorithms, each on its own examples. Not a sample — the whole bank, because "she can read
    Python" is only worth saying if it is a number over everything she is examined on."""
    import random

    coder, rng = Coder(), random.Random(5)
    wrong = []
    for task in EVERY_TASK:
        spec, source = instance(task, rng, task.id)
        program = read_python(source, name=spec.name)
        for example in spec.shown + spec.held_out:
            got = coder.run(program, example.args)
            if got != example.out:
                wrong.append(f"{task.id}: {got!r} != {example.out!r}")
    assert not wrong, wrong[:5]


def test_a_block_program_abstracts_to_a_shape_and_a_recursive_one_names_itself():
    loop = abstract(read_python("def t(xs):\n    t = 0\n    for x in xs:\n        t += x * 3\n"
                                "    return t"))
    assert loop.block and loop.cost == 2

    rec = abstract(read_python("def fact(n):\n    if n <= 1:\n        return 1\n"
                               "    return n * fact(n - 1)"))
    # The callee's name is the identity of the call, not a constant — `#self` is what lets the
    # shape land on a function called something else.
    assert "#self" in rec.key and rec.block


def test_teaching_carries_a_loop_shape_to_a_task_nobody_demonstrated():
    import math
    import random

    rng = random.Random(4)
    rows = [tuple(rng.randrange(1, 9) for _ in range(4)) for _ in range(12)]
    lesson = Spec.of("triples", "", ["xs"],
                     [([r], sum(x * 3 for x in r)) for r in rows[:8]],
                     [([r], sum(x * 3 for x in r)) for r in rows[8:]])
    unseen = Spec.of("products", "", ["xs"],
                     [([r], math.prod(r)) for r in rows[:8]],
                     [([r], math.prod(r)) for r in rows[8:]])

    taught = Coder()
    assert taught.learn_python(
        lesson, "def triples(xs):\n    total = 0\n    for x in xs:\n        total += x * 3\n"
                "    return total").verdict == Verdict.SURVIVED
    written = taught.write(unseen, attempts=30000, graft=False)
    assert written.ok and taught.check(written.program, unseen.held_out).ok
    # Not asserted: that the answer is the *loop*. She reaches this one with `fold` — a cheaper,
    # correct program she already held — and a test that demanded the taught shape would be
    # demanding a worse answer. What the lesson has to buy is the answer, not its own reuse.
    assert taught.taught_shapes(), "the loop shape is held, whether or not this task needed it"


def test_what_she_writes_can_be_read_back():
    """``source()`` is what a person reviews, so it has to be Python — and the reader has to take
    its own dialect back, or the round trip is a picture of one."""
    import random

    coder, rng = Coder(), random.Random(6)
    for task in EVERY_TASK[:20]:
        spec, source = instance(task, rng, task.id)
        coder.learn_python(spec, source)
        written = coder.write(spec, attempts=20000, graft=False)
        assert written.ok, task.id
        again = read_python(written.program.source(), name=spec.name)
        for example in spec.shown:
            assert coder.run(again, example.args) == coder.run(written.program, example.args)
