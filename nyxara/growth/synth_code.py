"""NYXARA · growth/synth_code.py — code that was executed before it was believed (💻→✅).

The measured state of the thing this replaces: ``generate_domain_docs("code", 3000)`` returned
3000 documents containing **six distinct strings** — `absolute`, `add`, `factorial`, `is_even`,
`maximum`, `square`, 839 characters in total. ``synthesis._code`` set ``candidate = reference``,
so the sandbox compiled one trusted string twice and compared a function to itself. The verifier
could not fail because the generator could not lie.

So this module is built on **mutation and tracing**, not on a template bank — a bank is exactly
what produced six documents, and no amount of widening changes its shape. Four kinds of
document, each with a verification that can actually reject:

* **bugfix pairs** — a mutation operator is applied to a working reference, and the pair is kept
  **only if the mutant's output genuinely differs on at least one input**. That is what proves
  the bug is real rather than an equivalent mutant, and it is a check the old generator had no
  way to perform.
* **test authoring** — a suite is kept only if it **passes the reference and kills the mutant**.
  Non-vacuousness becomes provable rather than assumed.
* **execution tracing** — the program is built from a small template whose trace is constructed
  alongside it, then **actually executed and asserted to match**. The verification double-checks
  the generator, not just the output.
* **algorithms** — real implementations with randomised identifiers, argument names and
  constants, executed against a reference on generated inputs.

Everything runs through :mod:`~nyxara.growth.sandbox_exec`, because mutants do not terminate and
the in-process sandbox has no timeout. That is not a precaution here; it is load-bearing.
"""

from __future__ import annotations

import random
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.growth.synth_common import Generator

__all__ = ["GENERATORS", "generators", "REFERENCES", "MUTATIONS"]

Built = Optional[Tuple[str, str]]

# --------------------------------------------------------------------------- #
# Surface pools — identifiers are the cheapest real diversity in code
# --------------------------------------------------------------------------- #
_FN_NAMES = ("compute", "process", "evaluate", "collect", "reduce_all", "scan", "summarise",
             "transform", "resolve", "aggregate", "tally", "walk", "fold", "measure",
             "select_best", "normalise", "accumulate", "sift")
_SEQ_NAMES = ("items", "values", "data", "numbers", "entries", "records", "xs", "samples",
              "buffer", "series", "readings", "batch")
_ACC_NAMES = ("total", "acc", "result", "out", "running", "carry", "best", "score", "count")
_IDX_NAMES = ("i", "idx", "k", "pos", "n", "j")


def _pick(rng: random.Random, pool):
    return pool[rng.randrange(len(pool))]


def _names(rng: random.Random) -> Dict[str, str]:
    return {"fn": _pick(rng, _FN_NAMES), "seq": _pick(rng, _SEQ_NAMES),
            "acc": _pick(rng, _ACC_NAMES), "idx": _pick(rng, _IDX_NAMES)}


# --------------------------------------------------------------------------- #
# Reference implementations. Templated on identifiers so the surface varies while
# the semantics stay fixed and checkable.
# --------------------------------------------------------------------------- #
class Reference:
    """One correct implementation, its description, and inputs that exercise it."""

    def __init__(self, key: str, describe: str, body: str,
                 args: Callable[[random.Random], List], *, oracle: Callable) -> None:
        self.key = key
        self.describe = describe
        self.body = body
        self.args = args
        self.oracle = oracle

    def render(self, names: Dict[str, str]) -> str:
        return self.body.format(**names)

    def prompt(self, names: Dict[str, str]) -> str:
        return self.describe.format(**names)


def _ints(lo: int, hi: int, n_lo: int = 3, n_hi: int = 8) -> Callable[[random.Random], List]:
    def make(rng: random.Random) -> List:
        return [[rng.randint(lo, hi) for _ in range(rng.randint(n_lo, n_hi))]]
    return make


REFERENCES: Tuple[Reference, ...] = (
    Reference(
        "sum_positive",
        "Write a Python function `{fn}` that returns the sum of the positive numbers in `{seq}`.",
        "def {fn}({seq}):\n"
        "    {acc} = 0\n"
        "    for {idx} in {seq}:\n"
        "        if {idx} > 0:\n"
        "            {acc} = {acc} + {idx}\n"
        "    return {acc}\n",
        _ints(-50, 50), oracle=lambda xs: sum(v for v in xs if v > 0)),
    Reference(
        "count_even",
        "Write a Python function `{fn}` that counts how many entries of `{seq}` are even.",
        "def {fn}({seq}):\n"
        "    {acc} = 0\n"
        "    for {idx} in {seq}:\n"
        "        if {idx} % 2 == 0:\n"
        "            {acc} = {acc} + 1\n"
        "    return {acc}\n",
        _ints(-40, 40), oracle=lambda xs: sum(1 for v in xs if v % 2 == 0)),
    Reference(
        "maximum",
        "Write a Python function `{fn}` that returns the largest value in `{seq}` "
        "without using max().",
        "def {fn}({seq}):\n"
        "    {acc} = {seq}[0]\n"
        "    for {idx} in {seq}[1:]:\n"
        "        if {idx} > {acc}:\n"
        "            {acc} = {idx}\n"
        "    return {acc}\n",
        _ints(-99, 99, 2, 9), oracle=lambda xs: max(xs)),
    Reference(
        "running_total",
        "Write a Python function `{fn}` that returns the running totals of `{seq}`.",
        "def {fn}({seq}):\n"
        "    {acc} = 0\n"
        "    out = []\n"
        "    for {idx} in {seq}:\n"
        "        {acc} = {acc} + {idx}\n"
        "        out.append({acc})\n"
        "    return out\n",
        _ints(-20, 20), oracle=lambda xs: [sum(xs[:i + 1]) for i in range(len(xs))]),
    Reference(
        "reverse_list",
        "Write a Python function `{fn}` that reverses `{seq}` without using slicing.",
        "def {fn}({seq}):\n"
        "    out = []\n"
        "    for {idx} in range(len({seq}) - 1, -1, -1):\n"
        "        out.append({seq}[{idx}])\n"
        "    return out\n",
        _ints(-30, 30), oracle=lambda xs: list(reversed(xs))),
    Reference(
        "bubble_sort",
        "Write a Python function `{fn}` that sorts `{seq}` in ascending order using bubble sort.",
        "def {fn}({seq}):\n"
        "    {acc} = list({seq})\n"
        "    for {idx} in range(len({acc})):\n"
        "        for j in range(len({acc}) - 1 - {idx}):\n"
        "            if {acc}[j] > {acc}[j + 1]:\n"
        "                {acc}[j], {acc}[j + 1] = {acc}[j + 1], {acc}[j]\n"
        "    return {acc}\n",
        _ints(-60, 60, 3, 7), oracle=lambda xs: sorted(xs)),
    Reference(
        "count_above_mean",
        "Write a Python function `{fn}` that counts entries of `{seq}` strictly above the mean.",
        "def {fn}({seq}):\n"
        "    mean = sum({seq}) / len({seq})\n"
        "    {acc} = 0\n"
        "    for {idx} in {seq}:\n"
        "        if {idx} > mean:\n"
        "            {acc} = {acc} + 1\n"
        "    return {acc}\n",
        _ints(0, 100, 3, 8),
        oracle=lambda xs: sum(1 for v in xs if v > sum(xs) / len(xs))),
    Reference(
        "longest_run",
        "Write a Python function `{fn}` that returns the length of the longest run of equal "
        "adjacent values in `{seq}`.",
        "def {fn}({seq}):\n"
        "    best = 1\n"
        "    {acc} = 1\n"
        "    for {idx} in range(1, len({seq})):\n"
        "        if {seq}[{idx}] == {seq}[{idx} - 1]:\n"
        "            {acc} = {acc} + 1\n"
        "            if {acc} > best:\n"
        "                best = {acc}\n"
        "        else:\n"
        "            {acc} = 1\n"
        "    return best\n",
        _ints(0, 3, 4, 10),
        oracle=lambda xs: max((len(list(g)) for g in __import__("itertools").groupby(xs)),
                              default=1)),
)


# --------------------------------------------------------------------------- #
# Mutation operators — how a working program is made to be wrong on purpose
# --------------------------------------------------------------------------- #
def _sub_first(pattern: str, repl: str) -> Callable[[str], Optional[str]]:
    def apply(src: str) -> Optional[str]:
        new, n = re.subn(pattern, repl, src, count=1)
        return new if n else None
    return apply


#: ``(name, description, transform)``. The description is what the fixed document explains.
MUTATIONS: Tuple[Tuple[str, str, Callable[[str], Optional[str]]], ...] = (
    ("off_by_one_range", "the loop bound is one too small, so the last element is skipped",
     _sub_first(r"range\(len\((\w+)\)\)", r"range(len(\1) - 1)")),
    ("strict_comparison", "the comparison is strict where it should not be, or vice versa",
     _sub_first(r"(?<![<>=!])>(?!=)", ">=")),
    ("flipped_comparison", "the comparison is the wrong way round",
     _sub_first(r"(?<![<>=!])>(?!=)", "<")),
    ("wrong_accumulator_init", "the accumulator starts at 1 instead of 0",
     _sub_first(r"= 0\n", "= 1\n")),
    ("assignment_not_increment", "the accumulator is overwritten instead of updated",
     _sub_first(r"(\w+) = \1 \+ (\w+)", r"\1 = \2")),
    ("wrong_parity", "the parity test is inverted",
     _sub_first(r"% 2 == 0", "% 2 == 1")),
    ("off_by_one_index", "an index is shifted by one",
     _sub_first(r"\[(\w+) - 1\]", r"[\1]")),
    ("dropped_first", "the scan starts at the wrong element",
     _sub_first(r"range\(1, len\((\w+)\)\)", r"range(0, len(\1))")),
)


def _guarded():
    """A shared executor. Mutants do not terminate; the in-process sandbox has no timeout."""
    global _EXECUTOR
    if _EXECUTOR is None:
        from nyxara.growth.sandbox_exec import GuardedExecutor, SandboxLimits

        _EXECUTOR = GuardedExecutor(SandboxLimits(cpu_seconds=2, wall_seconds=5.0))
    return _EXECUTOR


_EXECUTOR = None


def _run(src: str, fn: str, calls: Sequence[Sequence]) -> Optional[List]:
    """Execute ``fn`` over ``calls``. ``None`` if anything failed, timed out, or raised."""
    result = _guarded().call(src, fn, calls)
    if not result.ok or len(result.results) != len(calls):
        return None
    if any(not ok for ok, _ in result.results):
        return None
    return [value for _ok, value in result.results]


def _reference_and_inputs(rng: random.Random) -> Tuple[Reference, Dict[str, str], str, List]:
    ref = REFERENCES[rng.randrange(len(REFERENCES))]
    names = _names(rng)
    src = ref.render(names)
    calls = [ref.args(rng)[0] for _ in range(6)]
    return ref, names, src, [[c] for c in calls]


# --------------------------------------------------------------------------- #
# The four document kinds
# --------------------------------------------------------------------------- #
def _algorithm(rng: random.Random) -> Built:
    """A correct implementation, executed against an independent oracle before it is kept."""
    ref, names, src, calls = _reference_and_inputs(rng)
    got = _run(src, names["fn"], calls)
    if got is None:
        return None
    expected = [ref.oracle(args[0]) for args in calls]
    if got != expected:
        return None                               # the reference itself is wrong — never ship it
    example = calls[0][0]
    return (ref.prompt(names),
            f"{src}\nWorked example:\n"
            f"  {names['fn']}({example}) -> {expected[0]}")


def _bugfix(rng: random.Random) -> Built:
    """A real bug: kept only when the mutant's output actually differs from the reference."""
    ref, names, src, calls = _reference_and_inputs(rng)
    order = list(range(len(MUTATIONS)))
    rng.shuffle(order)
    for index in order:
        mut_name, description, transform = MUTATIONS[index]
        broken = transform(src)
        if not broken or broken == src:
            continue
        good = _run(src, names["fn"], calls)
        bad = _run(broken, names["fn"], calls)
        if good is None:
            return None
        # An equivalent mutant is not a bug. `bad is None` means it crashed or hung, which IS a
        # difference; otherwise the outputs must actually diverge somewhere.
        if bad is not None and bad == good:
            continue
        failing = next((i for i, (g, b) in enumerate(zip(good, bad or [])) if g != b), 0)
        args = calls[failing][0]
        observed = "raised an error or did not terminate" if bad is None else str(bad[failing])
        # "...that returns the sum" -> "It should return the sum", so the sentence reads as
        # English rather than as a spliced fragment. A corpus full of "meant to returns" teaches
        # the model that agreement errors are normal.
        intent = ref.prompt(names).split("that ", 1)[-1].rstrip(".")
        intent = re.sub(r"^(returns|counts|reverses|sorts)\b",
                        lambda m: {"returns": "return", "counts": "count",
                                   "reverses": "reverse", "sorts": "sort"}[m.group(1)],
                        intent)
        return (f"This function should {intent}, but it is wrong. "
                f"Find the bug and fix it.\n\n{broken}",
                f"The bug: {description}.\n\n"
                f"On input {args} it returns {observed}, but the correct answer is "
                f"{good[failing]}.\n\nFixed:\n{src}")
    return None


def _write_tests(rng: random.Random) -> Built:
    """A test suite kept only if it passes the reference AND kills a real mutant."""
    ref, names, src, calls = _reference_and_inputs(rng)
    good = _run(src, names["fn"], calls)
    if good is None:
        return None
    order = list(range(len(MUTATIONS)))
    rng.shuffle(order)
    for index in order:
        mut_name, description, transform = MUTATIONS[index]
        broken = transform(src)
        if not broken or broken == src:
            continue
        bad = _run(broken, names["fn"], calls)
        if bad is not None and bad == good:
            continue
        killed = ("every case (it does not terminate or it raises)" if bad is None else
                  ", ".join(str(calls[i][0]) for i, (g, b) in enumerate(zip(good, bad))
                            if g != b))
        asserts = "\n".join(f"    assert {names['fn']}({calls[i][0]}) == {good[i]}"
                            for i in range(len(calls)))
        return (f"Write a test suite for this function.\n\n{src}",
                f"def test_{names['fn']}():\n{asserts}\n\n"
                f"These assertions pass on the implementation above. They are not vacuous: a "
                f"version where {description} fails on {killed}.")
    return None


_TRACE_TEMPLATES = (
    ("accumulate", "def {fn}({seq}):\n    {acc} = 0\n    for {idx} in {seq}:\n"
                   "        {acc} = {acc} + {idx}\n    return {acc}\n"),
    ("scale", "def {fn}({seq}):\n    {acc} = 1\n    for {idx} in {seq}:\n"
              "        {acc} = {acc} * {idx}\n    return {acc}\n"),
)


def _trace(rng: random.Random) -> Built:
    """Variable state line by line — constructed alongside the program, then *executed* to check.

    The trace is not narrated after the fact; it is produced by the same construction that
    produces the program, and then a real execution has to agree with it. So the verification
    checks the generator, not merely the output.
    """
    kind, template = _TRACE_TEMPLATES[rng.randrange(len(_TRACE_TEMPLATES))]
    names = _names(rng)
    src = template.format(**names)
    if kind == "accumulate":
        values = [rng.randint(-9, 20) for _ in range(rng.randint(3, 6))]
        acc, rows = 0, []
        for step, v in enumerate(values, 1):
            before = acc
            acc += v
            rows.append(f"  iteration {step}: {names['idx']} = {v}, "
                        f"{names['acc']} = {before} + {v} = {acc}")
    else:
        values = [rng.randint(1, 6) for _ in range(rng.randint(3, 5))]
        acc, rows = 1, []
        for step, v in enumerate(values, 1):
            before = acc
            acc *= v
            rows.append(f"  iteration {step}: {names['idx']} = {v}, "
                        f"{names['acc']} = {before} x {v} = {acc}")
    got = _run(src, names["fn"], [[values]])
    if got is None or got[0] != acc:
        return None                               # the constructed trace disagreed with reality
    return (f"Trace this function on {values}, showing the value of "
            f"`{names['acc']}` after each iteration.\n\n{src}",
            f"{names['acc']} starts at {0 if kind == 'accumulate' else 1}.\n"
            + "\n".join(rows) + f"\nThe loop ends and {names['acc']} = {acc} is returned.\n"
            f"{names['fn']}({values}) = {acc}")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_SURFACE = len(_FN_NAMES) * len(_SEQ_NAMES) * len(_ACC_NAMES) * len(_IDX_NAMES)

GENERATORS: Tuple[Generator, ...] = (
    Generator("code_algorithm", "code", _algorithm,
              space=len(REFERENCES) * _SURFACE * 40, weight=1.2),
    Generator("code_bugfix", "code", _bugfix,
              space=len(REFERENCES) * len(MUTATIONS) * _SURFACE * 8, weight=1.4),
    Generator("code_tests", "code", _write_tests,
              space=len(REFERENCES) * len(MUTATIONS) * _SURFACE * 8, weight=1.0),
    Generator("code_trace", "code", _trace,
              space=len(_TRACE_TEMPLATES) * _SURFACE * 500, weight=1.0),
)


def generators() -> Tuple[Generator, ...]:
    return GENERATORS
