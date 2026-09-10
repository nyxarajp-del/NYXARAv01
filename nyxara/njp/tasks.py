"""NYXARA · njp/tasks.py — the programming she is examined on, and where it comes from (📒).

:mod:`nyxara.njp.school` decides how to teach and how to grade. This is the *material*: every
program a subject reads, every task a subject asks her to write, and every piece of source she is
required to refuse. It is a separate module because a syllabus and its question bank change for
different reasons — a bank grows whenever a gap is found, and the grading discipline should not
have to be touched when it does.

**Every task carries its own worked solution, and the expected answers come from running it.**
A :class:`Task` produces Python source; the reference outputs are obtained by executing that
source in a throwaway namespace. This is the one place in the package that calls :func:`exec`, and
the reason is worth stating plainly rather than hiding: the alternative is a table of
hand-computed answers, and a hand-computed table is a table with a mistake in it — one was written
during this module's development and it marked a *correct* lesson as refuted. The source executed
here is a literal in this file, never anything she produced and never anything a caller passed;
:mod:`nyxara.njp.coding` still executes nothing, and her own interpreter is what her answers come
from. The oracle and the student do not share a machine, which is the whole point of an oracle.

**Instances, not questions.** A task is a *family*: ``count the ones over a threshold`` with the
threshold and the data drawn per instance. Teaching gets one instance and the exam gets another,
so a shape can be demonstrated on ``sum triple the odd ones`` and asked for on ``sum double the
even ones`` — the transfer claim the whole school is built to measure.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

__all__ = [
    "Task", "reference", "normalise", "instance",
    "BASIC", "COMPOSITE", "LOOPS", "RECURSION", "DICTS", "STRINGS", "STRUCTURES", "ALGORITHMS",
    "EVERY_TASK", "READING", "REFUSALS", "EDGE_CASES",
]


# --------------------------------------------------------------------------- #
# the oracle
# --------------------------------------------------------------------------- #

def reference(source: str) -> Callable[..., Any]:
    """The worked solution, as a real Python function, to generate expected answers from.

    The last function defined wins, so a task may define helpers above the one under test.
    """
    namespace: Dict[str, Any] = {}
    exec(compile(source, "<syllabus>", "exec"), namespace)   # noqa: S102 - the oracle; see above
    functions = [value for key, value in namespace.items()
                 if callable(value) and not key.startswith("__")]
    if not functions:
        raise ValueError("a task's source must define a function")
    return functions[-1]


def normalise(value: Any) -> Any:
    """Python's answer in the value domain :mod:`nyxara.njp.coding` uses.

    Lists become tuples and sets become frozensets — not to make her look right, but because the
    two languages spell the same value differently and a comparison that failed on the spelling
    would report every correct program as wrong.
    """
    if isinstance(value, (list, tuple)):
        return tuple(normalise(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(normalise(item) for item in value)
    if isinstance(value, dict):
        return {key: normalise(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class Task:
    """One family of programming task: how to write an instance, and what to run it on."""

    id: str = ""
    intent: str = ""
    params: Tuple[str, ...] = ("xs",)
    build: Any = None            # (rng) -> source
    data: Any = None             # (rng) -> rows of argument tuples
    kind: str = "expression"     # expression | loop | recursion | dict | string | structure

    def source(self, rng: random.Random, name: str) -> str:
        return str(self.build(rng)).replace("FNAME", name)


def instance(task: Task, rng: random.Random, name: str, *, shown: int = 8) -> Tuple[Any, str]:
    """One task instance: a spec with a shown/held-out split, and the solution that passes it.

    Eight shown and the rest held out. Six was not enough: two families were solved by a program
    that fitted all six and failed the seventh, which is the coincidence the split exists to catch
    — but a split that catches one *often* is a split whose training half is too small, and
    catching a coincidence costs more than not producing it.

    Rows that the reference cannot answer — ``max`` of an empty list, a division by zero the
    family did not guard — are dropped rather than kept as errors. A row a *correct* program
    cannot survive is not an example of anything.
    """
    from nyxara.njp.coding import Spec

    source = task.source(rng, name)
    fn = reference(source)
    rows: List[Tuple[Tuple[Any, ...], Any]] = []
    for args in task.data(rng):
        try:
            rows.append((tuple(normalise(a) for a in args), normalise(fn(*args))))
        except Exception:  # noqa: BLE001 - an unanswerable row is not an example
            continue
    spec = Spec.of(name, task.intent, task.params,
                   [(args, out) for args, out in rows[:shown]],
                   [(args, out) for args, out in rows[shown:]])
    return spec, source


# --------------------------------------------------------------------------- #
# data generators
# --------------------------------------------------------------------------- #

def _ints(lo: int = 1, hi: int = 10, rows: int = 14) -> Any:
    return lambda rng: [(rng.randint(lo, hi),) for _ in range(rows)]


def _lists(rows: int = 14, size: Tuple[int, int] = (3, 7), lo: int = 0, hi: int = 20,
           allow_empty: bool = False) -> Any:
    def make(rng: random.Random) -> List[Tuple[Any, ...]]:
        out = []
        for index in range(rows):
            length = 0 if (allow_empty and index == 0) else rng.randint(*size)
            out.append((tuple(rng.randrange(lo, hi) for _ in range(length)),))
        return out
    return make


def _two_lists(rows: int = 14) -> Any:
    def make(rng: random.Random) -> List[Tuple[Any, ...]]:
        return [(tuple(rng.randrange(0, 12) for _ in range(rng.randint(2, 5))),
                 tuple(rng.randrange(0, 12) for _ in range(rng.randint(2, 5))))
                for _ in range(rows)]
    return make


_WORDS = ("the", "cat", "sat", "on", "a", "mat", "dog", "ran", "far", "red", "big", "sky",
          "code", "loop", "list", "map", "sum", "data", "node", "tree")


def _sentences(rows: int = 14) -> Any:
    def make(rng: random.Random) -> List[Tuple[Any, ...]]:
        return [(" ".join(rng.choice(_WORDS) for _ in range(rng.randint(3, 7))),)
                for _ in range(rows)]
    return make


def _words(rows: int = 14) -> Any:
    def make(rng: random.Random) -> List[Tuple[Any, ...]]:
        return [("".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 8))),)
                for _ in range(rows)]
    return make


def _dicts(rows: int = 14) -> Any:
    def make(rng: random.Random) -> List[Tuple[Any, ...]]:
        out = []
        for _ in range(rows):
            keys = rng.sample(_WORDS, rng.randint(2, 5))
            out.append(({key: rng.randint(1, 9) for key in keys},))
        return out
    return make


def _matrices(rows: int = 14) -> Any:
    def make(rng: random.Random) -> List[Tuple[Any, ...]]:
        out = []
        for _ in range(rows):
            height, width = rng.randint(2, 3), rng.randint(2, 3)
            out.append((tuple(tuple(rng.randrange(0, 9) for _ in range(width))
                              for _ in range(height)),))
        return out
    return make


def _pair_lists(rows: int = 14) -> Any:
    def make(rng: random.Random) -> List[Tuple[Any, ...]]:
        return [(tuple((rng.choice(_WORDS[:5]), rng.randint(1, 9))
                       for _ in range(rng.randint(3, 6))),) for _ in range(rows)]
    return make


# --------------------------------------------------------------------------- #
# the banks
# --------------------------------------------------------------------------- #

def _pick(rng: random.Random, *options: Any) -> Any:
    return rng.choice(options)


#: One operator deep. Reachable from the seed shapes alone, so the subject that uses these is a
#: floor rather than a lesson — and a low score here is a broken search, not a missing shape.
BASIC: Tuple[Task, ...] = (
    Task("total", "add them up", ("xs",),
         lambda rng: "def FNAME(xs):\n    return sum(xs)", _lists()),
    Task("count", "how many there are", ("xs",),
         lambda rng: "def FNAME(xs):\n    return len(xs)", _lists()),
    Task("largest", "the biggest one", ("xs",),
         lambda rng: "def FNAME(xs):\n    return max(xs)", _lists()),
    Task("ordered", "put them in order", ("xs",),
         lambda rng: "def FNAME(xs):\n    return sorted(xs)", _lists()),
    Task("backwards", "reverse them", ("xs",),
         lambda rng: "def FNAME(xs):\n    return xs[::-1]", _lists()),
    Task("evens", "keep the even ones", ("xs",),
         lambda rng: "def FNAME(xs):\n    return [x for x in xs if x % 2 == 0]", _lists()),
    Task("scaled", "multiply each by a constant", ("xs",),
         lambda rng: f"def FNAME(xs):\n    return [x * {rng.choice([2, 3, 5])} for x in xs]",
         _lists()),
    Task("distinct", "drop the repeats", ("xs",),
         lambda rng: "def FNAME(xs):\n    return len(set(xs))", _lists()),
)

#: Two and three operators deep. Not reachable from a seed without grafting, which is expensive —
#: so this is where a lesson shows up as a number.
COMPOSITE: Tuple[Task, ...] = (
    Task("sum_mapped_filtered", "add up a multiple of the ones that pass a test", ("xs",),
         lambda rng: (f"def FNAME(xs):\n    return sum([x * {rng.choice([2, 3, 4])} "
                      f"for x in xs if x % 2 == {rng.choice([0, 1])}])"), _lists()),
    Task("count_over", "how many are over a threshold", ("xs",),
         lambda rng: (f"def FNAME(xs):\n    return len([x for x in xs "
                      f"if x > {rng.choice([5, 8, 10, 12])}])"), _lists()),
    Task("sum_over", "add up only the ones over a threshold", ("xs",),
         lambda rng: (f"def FNAME(xs):\n    return sum([x for x in xs "
                      f"if x > {rng.choice([4, 7, 9, 11])}])"), _lists()),
    Task("max_mapped", "the largest after a transform", ("xs",),
         lambda rng: f"def FNAME(xs):\n    return max([x * {rng.choice([2, 3, 5])} for x in xs])",
         _lists()),
    Task("sum_shifted", "add up a shifted copy of each", ("xs",),
         lambda rng: f"def FNAME(xs):\n    return sum([x + {rng.choice([2, 3, 4])} for x in xs])",
         _lists()),
    Task("nth_smallest", "the n-th smallest", ("xs",),
         lambda rng: f"def FNAME(xs):\n    return sorted(xs)[{rng.choice([0, 1, 2])}]", _lists()),
    Task("descending", "in order, largest first", ("xs",),
         lambda rng: "def FNAME(xs):\n    return sorted(xs)[::-1]", _lists()),
    Task("mapped_filtered", "transform the ones that pass a test", ("xs",),
         lambda rng: (f"def FNAME(xs):\n    return [x * {rng.choice([2, 3])} for x in xs "
                      f"if x > {rng.choice([4, 6, 9])}]"), _lists()),
)

#: A variable, a loop, and a body — the second half of a first course, and the half she could
#: read and not write until the procedural seed shapes existed.
LOOPS: Tuple[Task, ...] = (
    Task("accumulate", "add them up with a running total", ("xs",),
         lambda rng: ("def FNAME(xs):\n    total = 0\n    for x in xs:\n"
                      "        total += x\n    return total"), _lists(), kind="loop"),
    Task("product_loop", "multiply them together", ("xs",),
         lambda rng: ("def FNAME(xs):\n    total = 1\n    for x in xs:\n"
                      "        total *= x\n    return total"), _lists(), kind="loop"),
    Task("running_max", "the biggest, found by walking the list", ("xs",),
         lambda rng: ("def FNAME(xs):\n    best = xs[0]\n    for x in xs:\n"
                      "        if x > best:\n            best = x\n    return best"),
         _lists(), kind="loop"),
    Task("build_filtered", "collect the ones that pass a test", ("xs",),
         lambda rng: (f"def FNAME(xs):\n    out = []\n    for x in xs:\n"
                      f"        if x % {rng.choice([2, 3])} == 0:\n"
                      f"            out.append(x)\n    return out"), _lists(), kind="loop"),
    Task("count_matching", "count the ones that pass a test", ("xs",),
         lambda rng: (f"def FNAME(xs):\n    n = 0\n    for x in xs:\n"
                      f"        if x > {rng.choice([5, 8, 11])}:\n"
                      f"            n += 1\n    return n"), _lists(), kind="loop"),
    Task("weighted_sum", "add up each item after a transform", ("xs",),
         lambda rng: (f"def FNAME(xs):\n    total = 0\n    for x in xs:\n"
                      f"        total += x * {rng.choice([2, 3, 4])}\n    return total"),
         _lists(), kind="loop"),
    Task("countdown", "add up every number down to one", ("n",),
         lambda rng: ("def FNAME(n):\n    total = 0\n    while n > 0:\n"
                      "        total += n\n        n -= 1\n    return total"),
         _ints(1, 12), kind="loop"),
    Task("digit_sum", "add up the digits", ("n",),
         lambda rng: ("def FNAME(n):\n    total = 0\n    while n > 0:\n"
                      "        total += n % 10\n        n = n // 10\n    return total"),
         _ints(1, 9999), kind="loop"),
)

#: A base case and a step. Nothing else in the language can express "of itself".
RECURSION: Tuple[Task, ...] = (
    Task("factorial", "multiply every number down to one", ("n",),
         lambda rng: ("def FNAME(n):\n    if n <= 1:\n        return 1\n"
                      "    return n * FNAME(n - 1)"), _ints(1, 9), kind="recursion"),
    Task("sum_to_n", "add every number down to one", ("n",),
         lambda rng: ("def FNAME(n):\n    if n <= 1:\n        return n\n"
                      "    return n + FNAME(n - 1)"), _ints(1, 15), kind="recursion"),
    Task("sum_list_rec", "add up a list without a loop", ("xs",),
         lambda rng: ("def FNAME(xs):\n    if len(xs) == 0:\n        return 0\n"
                      "    return xs[0] + FNAME(xs[1:])"), _lists(size=(2, 5), allow_empty=True),
         kind="recursion"),
    Task("max_rec", "the biggest, without a loop", ("xs",),
         lambda rng: ("def FNAME(xs):\n    if len(xs) == 1:\n        return xs[0]\n"
                      "    return max(xs[0], FNAME(xs[1:]))"), _lists(size=(1, 5)),
         kind="recursion"),
    Task("fib", "each number is the sum of the two before it", ("n",),
         lambda rng: ("def FNAME(n):\n    if n < 2:\n        return n\n"
                      "    return FNAME(n - 1) + FNAME(n - 2)"), _ints(0, 14), kind="recursion"),
    Task("power_rec", "raise a constant to a power, by recursion", ("n",),
         lambda rng: (f"def FNAME(n):\n    if n == 0:\n        return 1\n"
                      f"    return {rng.choice([2, 3])} * FNAME(n - 1)"), _ints(0, 9),
         kind="recursion"),
    Task("count_down_rec", "how many steps down to zero", ("n",),
         lambda rng: ("def FNAME(n):\n    if n <= 0:\n        return 0\n"
                      "    return 1 + FNAME(n - 1)"), _ints(0, 20), kind="recursion"),
    Task("reverse_rec", "reverse a list without a loop", ("xs",),
         lambda rng: ("def FNAME(xs):\n    if len(xs) == 0:\n        return []\n"
                      "    return FNAME(xs[1:]) + [xs[0]]"), _lists(size=(1, 5), allow_empty=True),
         kind="recursion"),
)

#: Counting, grouping and lookup — the data structure most real programs are mostly made of.
DICTS: Tuple[Task, ...] = (
    Task("tally", "count how often each word appears", ("ws",),
         lambda rng: ("def FNAME(ws):\n    d = {}\n    for w in ws:\n"
                      "        d[w] = d.get(w, 0) + 1\n    return d"),
         lambda rng: [(tuple(rng.choice(_WORDS[:6]) for _ in range(rng.randint(4, 9))),)
                      for _ in range(14)], kind="dict"),
    Task("sum_values", "add up everything in the mapping", ("d",),
         lambda rng: "def FNAME(d):\n    return sum(d.values())", _dicts(), kind="dict"),
    Task("biggest_key", "the key with the largest value", ("d",),
         lambda rng: "def FNAME(d):\n    return max(sorted(d.keys()), key=lambda k: d[k])",
         _dicts(), kind="dict"),
    Task("filter_map", "keep the entries over a threshold", ("d",),
         lambda rng: (f"def FNAME(d):\n    return {{k: v for k, v in d.items() "
                      f"if v > {rng.choice([2, 4, 6])}}}"), _dicts(), kind="dict"),
    Task("invert", "swap the keys and the values", ("d",),
         lambda rng: "def FNAME(d):\n    return {v: k for k, v in d.items()}", _dicts(),
         kind="dict"),
    Task("group_by", "collect the values under each key", ("rows",),
         lambda rng: ("def FNAME(rows):\n    out = {}\n    for k, v in rows:\n"
                      "        out[k] = out.get(k, []) + [v]\n    return out"),
         _pair_lists(), kind="dict"),
    Task("keys_sorted", "the keys, in order", ("d",),
         lambda rng: "def FNAME(d):\n    return sorted(d.keys())", _dicts(), kind="dict"),
    Task("shared", "what both lists have in common", ("a", "b"),
         lambda rng: "def FNAME(a, b):\n    return sorted(set(a) & set(b))", _two_lists(),
         kind="dict"),
)

#: Text. Half of what anybody actually writes.
STRINGS: Tuple[Task, ...] = (
    Task("shout", "in capitals", ("s",),
         lambda rng: "def FNAME(s):\n    return s.upper()", _sentences(), kind="string"),
    Task("word_count", "how many words", ("s",),
         lambda rng: "def FNAME(s):\n    return len(s.split())", _sentences(), kind="string"),
    Task("longest_word", "the longest word in it", ("s",),
         lambda rng: ("def FNAME(s):\n    best = ''\n    for w in s.split():\n"
                      "        if len(w) > len(best):\n            best = w\n    return best"),
         _sentences(), kind="string"),
    Task("is_palindrome", "does it read the same backwards", ("s",),
         lambda rng: "def FNAME(s):\n    return s == s[::-1]", _words(), kind="string"),
    Task("count_letter", "how many times a letter appears", ("s",),
         lambda rng: f"def FNAME(s):\n    return s.count('{rng.choice('aeiou')}')",
         _sentences(), kind="string"),
    Task("initials", "the first letter of each word", ("s",),
         lambda rng: ("def FNAME(s):\n    out = ''\n    for w in s.split():\n"
                      "        out = out + w[0]\n    return out"), _sentences(), kind="string"),
    Task("reverse_words", "the words, in reverse order", ("s",),
         lambda rng: "def FNAME(s):\n    return ' '.join(s.split()[::-1])", _sentences(),
         kind="string"),
    Task("no_vowels", "with the vowels taken out", ("s",),
         lambda rng: ("def FNAME(s):\n    out = ''\n    for ch in s:\n"
                      "        if ch not in 'aeiou':\n            out = out + ch\n    return out"),
         _words(), kind="string"),
)

#: Lists of lists. Nesting is where a language either composes or does not.
STRUCTURES: Tuple[Task, ...] = (
    Task("matrix_total", "add up every cell", ("m",),
         lambda rng: ("def FNAME(m):\n    total = 0\n    for row in m:\n        for v in row:\n"
                      "            total += v\n    return total"), _matrices(), kind="structure"),
    Task("row_totals", "the total of each row", ("m",),
         lambda rng: "def FNAME(m):\n    return [sum(row) for row in m]", _matrices(),
         kind="structure"),
    Task("flatten", "one flat list out of the rows", ("m",),
         lambda rng: ("def FNAME(m):\n    out = []\n    for row in m:\n"
                      "        out.extend(row)\n    return out"), _matrices(), kind="structure"),
    Task("biggest_cell", "the largest value anywhere in it", ("m",),
         lambda rng: "def FNAME(m):\n    return max([max(row) for row in m])", _matrices(),
         kind="structure"),
    Task("first_column", "the first value of every row", ("m",),
         lambda rng: "def FNAME(m):\n    return [row[0] for row in m]", _matrices(),
         kind="structure"),
    Task("pairwise", "multiply two lists together, item by item", ("a", "b"),
         lambda rng: "def FNAME(a, b):\n    return [x * y for x, y in zip(a, b)]",
         _two_lists(), kind="structure"),
    Task("longest_row", "how long the longest row is", ("m",),
         lambda rng: "def FNAME(m):\n    return max([len(row) for row in m])", _matrices(),
         kind="structure"),
)

#: The problems a course sets when it wants to know whether you can actually program.
ALGORITHMS: Tuple[Task, ...] = (
    Task("gcd", "the largest number that divides both", ("a", "b"),
         lambda rng: ("def FNAME(a, b):\n    while b:\n        a, b = b, a % b\n    return a"),
         lambda rng: [(rng.randint(2, 90), rng.randint(2, 90)) for _ in range(14)],
         kind="loop"),
    Task("is_prime", "has it any divisor but one and itself", ("n",),
         lambda rng: ("def FNAME(n):\n    if n < 2:\n        return False\n    d = 2\n"
                      "    while d * d <= n:\n        if n % d == 0:\n            return False\n"
                      "        d += 1\n    return True"), _ints(1, 60), kind="loop"),
    Task("primes_below", "every prime under a limit", ("n",),
         lambda rng: ("def FNAME(n):\n    out = []\n    for k in range(2, n):\n        ok = True\n"
                      "        for d in range(2, k):\n            if k % d == 0:\n"
                      "                ok = False\n        if ok:\n            out.append(k)\n"
                      "    return out"), _ints(5, 40), kind="loop"),
    Task("binary_search", "find it by halving the range", ("xs", "t"),
         lambda rng: ("def FNAME(xs, t):\n    lo = 0\n    hi = len(xs) - 1\n"
                      "    while lo <= hi:\n        mid = (lo + hi) // 2\n"
                      "        if xs[mid] == t:\n            return mid\n"
                      "        if xs[mid] < t:\n            lo = mid + 1\n"
                      "        else:\n            hi = mid - 1\n    return -1"),
         lambda rng: [(tuple(sorted(rng.sample(range(30), rng.randint(4, 9)))),
                       rng.randrange(30)) for _ in range(14)], kind="loop"),
    Task("bubble_sort", "sort it by swapping neighbours", ("xs",),
         lambda rng: ("def FNAME(xs):\n    ys = list(xs)\n    for i in range(len(ys)):\n"
                      "        for j in range(len(ys) - 1):\n"
                      "            if ys[j] > ys[j + 1]:\n"
                      "                ys[j], ys[j + 1] = ys[j + 1], ys[j]\n    return ys"),
         _lists(size=(2, 6)), kind="loop"),
    Task("quicksort", "sort it by splitting around a pivot", ("xs",),
         lambda rng: ("def FNAME(xs):\n    if len(xs) <= 1:\n        return list(xs)\n"
                      "    p = xs[0]\n    lo = [x for x in xs[1:] if x < p]\n"
                      "    hi = [x for x in xs[1:] if x >= p]\n    return FNAME(lo) + [p] + "
                      "FNAME(hi)"), _lists(size=(1, 6), allow_empty=True), kind="recursion"),
    Task("fizzbuzz", "the classic, up to a limit", ("n",),
         lambda rng: ("def FNAME(n):\n    out = []\n    for i in range(1, n + 1):\n"
                      "        if i % 15 == 0:\n            out.append('FizzBuzz')\n"
                      "        elif i % 3 == 0:\n            out.append('Fizz')\n"
                      "        elif i % 5 == 0:\n            out.append('Buzz')\n"
                      "        else:\n            out.append(str(i))\n    return out"),
         _ints(1, 20), kind="loop"),
    Task("dedupe_ordered", "drop repeats but keep the order", ("xs",),
         lambda rng: ("def FNAME(xs):\n    seen = set()\n    out = []\n    for x in xs:\n"
                      "        if x not in seen:\n            out.append(x)\n"
                      "            seen.add(x)\n    return out"), _lists(), kind="loop"),
    Task("to_binary", "written in base two", ("n",),
         lambda rng: ("def FNAME(n):\n    if n == 0:\n        return '0'\n    s = ''\n"
                      "    while n > 0:\n        s = str(n % 2) + s\n        n = n // 2\n"
                      "    return s"), _ints(0, 200), kind="loop"),
    Task("best_run", "the largest total of any run of neighbours", ("xs",),
         lambda rng: ("def FNAME(xs):\n    best = xs[0]\n    cur = xs[0]\n    for x in xs[1:]:\n"
                      "        cur = max(x, cur + x)\n        best = max(best, cur)\n"
                      "    return best"),
         lambda rng: [(tuple(rng.randrange(-9, 10) for _ in range(rng.randint(3, 8))),)
                      for _ in range(14)], kind="loop"),
)

#: Every family, for the subjects that sweep the whole syllabus.
EVERY_TASK: Tuple[Task, ...] = (BASIC + COMPOSITE + LOOPS + RECURSION + DICTS + STRINGS
                                + STRUCTURES + ALGORITHMS)


# --------------------------------------------------------------------------- #
# the awkward inputs
# --------------------------------------------------------------------------- #

#: ``(label, source, args)``. Every program here is *correct*; what is under test is whether her
#: interpreter answers what Python answers on the input nobody thinks about — including when the
#: right answer is an error. A language that quietly returns something for ``max([])`` is worse
#: than one that raises, because the something is wrong and looks fine.
#: An entry may carry a fourth field, ``"refuse"``, marking a case where this language
#: *deliberately* differs from Python and the right answer is an error. There is exactly one such
#: divergence and it is argued for at :func:`~nyxara.njp.coding._int`: Python evaluates
#: ``True + 1`` to ``2``, and a language that agreed would let a predicate leak into arithmetic
#: and return a plausible wrong number instead of failing — the one outcome a verifier that works
#: by running the program cannot catch. Listing it here rather than quietly scoring it wrong is
#: the difference between a documented divergence and an unnoticed one.
EDGE_CASES: Tuple[Tuple[Any, ...], ...] = (
    ("sum of nothing", "def f(xs):\n    return sum(xs)", ((),)),
    ("length of nothing", "def f(xs):\n    return len(xs)", ((),)),
    ("max of nothing", "def f(xs):\n    return max(xs)", ((),)),
    ("first of nothing", "def f(xs):\n    return xs[0]", ((),)),
    ("index past the end", "def f(xs):\n    return xs[5]", ((1, 2),)),
    ("negative index", "def f(xs):\n    return xs[-1]", ((1, 2, 3),)),
    ("divide by zero", "def f(a, b):\n    return a // b", (1, 0)),
    ("modulo by zero", "def f(a, b):\n    return a % b", (1, 0)),
    ("one element", "def f(xs):\n    return sorted(xs)", ((7,),)),
    ("all the same", "def f(xs):\n    return len(set(xs))", ((4, 4, 4),)),
    ("all negative", "def f(xs):\n    return max(xs)", ((-5, -2, -9),)),
    ("negative floor division", "def f(a, b):\n    return a // b", (-7, 2)),
    ("negative modulo", "def f(a, b):\n    return a % b", (-7, 2)),
    ("zero times", "def f(n):\n    return [i for i in range(n)]", (0,)),
    ("empty string", "def f(s):\n    return len(s.split())", ("",)),
    ("string of spaces", "def f(s):\n    return len(s.split())", ("   ",)),
    ("slice past the end", "def f(xs):\n    return xs[1:99]", ((1, 2),)),
    ("slice of nothing", "def f(xs):\n    return xs[1:]", ((),)),
    ("missing key", "def f(d):\n    return d['nope']", ({"a": 1},)),
    ("missing key with default", "def f(d):\n    return d.get('nope', 0)", ({"a": 1},)),
    ("empty mapping", "def f(d):\n    return sum(d.values())", ({},)),
    ("empty set", "def f(xs):\n    return len(set(xs))", ((),)),
    ("loop over nothing", "def f(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t",
     ((),)),
    ("while that never runs", "def f(n):\n    t = 0\n    while n > 0:\n        t += n\n"
     "        n -= 1\n    return t", (0,)),
    ("recursion at the base", "def f(n):\n    if n <= 0:\n        return 0\n"
     "    return n + f(n - 1)", (0,)),
    ("zero in the middle", "def f(xs):\n    return [x for x in xs if x]", ((1, 0, 2),)),
    ("boolean is not a number", "def f(a, b):\n    return a + b", (True, 1), "refuse"),
    ("string plus number", "def f(a, b):\n    return a + b", ("x", 1)),
)


# --------------------------------------------------------------------------- #
# reading: constructions, and refusals
# --------------------------------------------------------------------------- #

#: ``(source, reference, args)``. Constructions a person writes that the worked solutions in the
#: banks above do not all exercise — a ternary, an f-string, ``in``, a negative index, a string
#: method. The expected value is computed by the reference rather than typed in, because a
#: hand-computed table is a table with a mistake in it.
READING: Tuple[Tuple[str, Any, Tuple[Any, ...]], ...] = (
    ("def f(xs):\n    return sum(xs) + len(xs)", lambda xs: sum(xs) + len(xs), ((3, 1, 4),)),
    ("def f(xs):\n    return [x * x for x in xs]", lambda xs: [x * x for x in xs], ((2, 3),)),
    ("def f(xs):\n    return sorted(xs)[::-1]", lambda xs: sorted(xs)[::-1], ((5, 2, 9),)),
    ("def f(xs):\n    return xs[1:]", lambda xs: xs[1:], ((7, 8, 9),)),
    ("def f(xs):\n    return xs[:2]", lambda xs: xs[:2], ((7, 8, 9),)),
    ("def f(xs):\n    return xs[-1]", lambda xs: xs[-1], ((7, 8, 9),)),
    ("def f(xs):\n    return xs[1:-1]", lambda xs: xs[1:-1], ((1, 2, 3, 4),)),
    ("def f(n):\n    return n if n > 0 else -n", lambda n: n if n > 0 else -n, (-6,)),
    ("def f(n):\n    return 'neg' if n < 0 else ('zero' if n == 0 else 'pos')",
     lambda n: "neg" if n < 0 else ("zero" if n == 0 else "pos"), (0,)),
    ("def f(a, b):\n    return max(a, b) - min(a, b)", lambda a, b: max(a, b) - min(a, b),
     (4, 11)),
    ("def f(a):\n    return 0 <= a <= 10 and a != 5", lambda a: 0 <= a <= 10 and a != 5, (3,)),
    ("def f(a, b):\n    return (a and not b) or (b and not a)",
     lambda a, b: (a and not b) or (b and not a), (True, False)),
    ("def f(xs):\n    return 3 in xs", lambda xs: 3 in xs, ((1, 2, 4),)),
    ("def f(xs):\n    return 3 not in xs", lambda xs: 3 not in xs, ((1, 2, 4),)),
    ("def f(s):\n    return len(s.split())", lambda s: len(s.split()), ("do the thing",)),
    ("def f(s):\n    return s.upper()", lambda s: s.upper(), ("nyxara",)),
    ("def f(s):\n    return s.replace('a', 'b')", lambda s: s.replace("a", "b"), ("cat",)),
    ("def f(s):\n    return s.startswith('ny')", lambda s: s.startswith("ny"), ("nyxara",)),
    ("def f(s, n):\n    return f'{s} has {n}'", lambda s, n: f"{s} has {n}", ("bag", 3)),
    ("lambda xs: sum([x % 2 for x in xs])", lambda xs: sum(x % 2 for x in xs), ((1, 2, 3),)),
    ("def f(xs):\n    return sum(x * 2 for x in xs if x > 1)",
     lambda xs: sum(x * 2 for x in xs if x > 1), ((1, 2, 3),)),
    ("def f(d, k):\n    v = d.get(k)\n    if v is None:\n        return -1\n    return v",
     lambda d, k: (d.get(k) if d.get(k) is not None else -1), ({"a": 1}, "b")),
    ("def f(xs):\n    x = 0\n    for x in xs:\n        pass\n    return x",
     lambda xs: list(xs)[-1], ((1, 2, 3),)),
    ("def f(n):\n    'a docstring'\n    return n + 1", lambda n: n + 1, (4,)),
)

#: Source a reader must **refuse**. Half of reading code is knowing what you are not going to run.
#:
#: Maintained *against the language*: a ``for`` loop and a dict literal were on this list and both
#: came off it when the language grew to include them. A refusal list that is not revised as the
#: reader grows stops testing the boundary and starts testing a memory of where it used to be.
REFUSALS: Tuple[str, ...] = (
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
    "def f(x):\n    return object()",
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
    "def f(x):\n    return [i for i in x for j in i]",
)
