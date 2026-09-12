"""NYXARA · njp/coding.py — she writes programs, and keeps only the ones that run (📟, NJP V.09).

**The gap, stated as a measurement rather than a feeling.** :mod:`nyxara.njp.calculate` closes an
arithmetic expression; :mod:`nyxara.njp.prove` certifies a proposition; :mod:`nyxara.njp.forge`
lowers *her own* already-written Python to C. Between them there is no organ that takes a
description of what a program should do and produces one — ``grep`` for a synthesiser across
``nyxara/njp/`` returns nothing. Asked to *write* code she had exactly one honest answer, and it
was "I can't". About 36% of the bundled corpus asks her to write something
(:mod:`nyxara.njp.study` says so and counts it); every one of those items was unreachable by
construction.

This is the organ that makes them reachable, and it is deliberately narrow in four ways that are
each a refusal rather than an omission:

**Nothing here is ever `eval`\\ ed or `exec`\\ ed.** A program is a :class:`Term` tree over a fixed
operator table (:data:`OPS`) and it runs in :class:`Interpreter`, which is forty lines of dispatch
and a step counter. Python source only ever enters through :func:`read_python`, which walks
:func:`ast.parse` output and raises :class:`CodeError` on any node outside its whitelist — a
"solution" like ``__import__('os').system('rm -rf /')`` is a parse-time refusal, not a shell
command. The whitelist is the security boundary and it is a whitelist precisely so that a new
Python node type cannot silently widen it.

**Every program halts, and it is a budget that makes that true rather than a missing feature.**
The language has ``while``, and it has recursion — a function may call itself, and
:mod:`nyxara.njp.school` examines her on ``gcd``, ``quicksort`` and Ackermann-shaped descent — so
``while True`` is something she can write. What she cannot do is run it forever:
:attr:`Interpreter.max_steps` counts evaluated nodes and :attr:`Interpreter.max_calls` bounds the
nesting, so a program that does not terminate is an :class:`Exhausted` rather than a hung process.
Steps rather than seconds, because a budget that varied by hardware would make a synthesis result
irreproducible, and reproducibility is the only reason to prefer a counter to a timer.

**The language, stated plainly.** Integers, booleans, strings, lists, mappings, sets and ``None``;
arithmetic, comparison and the boolean connectives; indexing, slicing and the container
operations; ``map``/``filter``/``fold`` and the rest of the higher-order vocabulary; local
variables, ``if``/``else``, ``for`` with unpacking, ``while``, ``break``, ``continue``, parallel
assignment, and functions that call themselves or each other. What it does not have: classes,
exceptions, generators, imports, closures over mutable state, and shared mutation — a container
here is a value, so ``xs.append(v)`` reads as ``xs = xs + [v]``. That last one is the only place
the subset *diverges* from Python rather than merely omitting something, and :func:`_dict_set`
says why.

**A program is never believed, only executed.** Everything else in NJP produces claims that carry a
confidence and go to :class:`~nyxara.njp.truth.TruthGauntlet`. Code is the one thing in this brain
with a decision procedure attached: it runs on the examples or it does not. So :meth:`Coder.write`
returns nothing at all unless the candidate passes **every** shown example, and
:meth:`Coder.learn` refuses a teacher's demonstration that does not pass its own — an
unverified demonstration teaches nothing, which is :mod:`nyxara.njp.teacher`'s rule applied to the
one domain that can enforce it mechanically.

**What a lesson leaves behind is the shape, never the answer.** §17's line — *"Weights ko symbolic
brain mein copy nahi karna. Behavior ko structured knowledge/programs mein convert karna."* —
is literal here. :meth:`Coder.learn` takes a verified demonstration, throws the program away, and
keeps a :class:`Schema`: the same control-flow skeleton with its constants and its inner functions
blanked out, and its parameters renamed to positions so it can land on a task whose variables have
different names. ``sum(map(double, filter(even, xs)))`` is retained as
``sum(map(?f, filter(?p, #0)))`` — which is worth something on a task nobody demonstrated, and
the demonstrated answer is worth nothing twice.

**And what teaching actually buys is search.** Synthesis here is enumeration under a hard attempt
budget. :data:`SEED_SHAPES` is what she has without a teacher: the primitive expressions, and the
procedural skeletons — accumulate, build, count, recur — with their operators blanked. Those are
enough for the simple half of the syllabus and not for the rest, because a composite shape is
reachable cold only by grafting, which is combinatorial.

A shape she has been *shown* is reached three ways before enumeration begins, and the order is the
design: the exact fillings that stood in it before; the same operators with different constants;
then one hole swept while the others hold. That is what a family *is* — the same skeleton with the
constants moved — so transfer between two instances of one costs tens of attempts rather than tens
of thousands.

**And a shape nobody ever showed her is now buildable rather than unreachable.**
:meth:`Coder.invent` runs when nothing she holds fits: bottom-up enumeration over the operator
grammar, pruned by observational equivalence. It composes *expressions* — a fold over a filtered
map, an index into a sort — and it does not compose an algorithm with a loop and a carried state.

Measured over the sixty-five families in :mod:`nyxara.njp.tasks`, on instances the teacher never
showed her and with the same attempt budget on both sides: **37 of 65 untaught, 65 of 65 taught**,
median 26 attempts, ten of the untaught solves built from the grammar outright. Measured over
thirty classic hard algorithms she was never taught — edit distance, Dijkstra, KMP, N-Queens —
invention changed nothing at all: **1 of 28 before it and 1 of 28 after**, because every one of
those needs the thing it cannot build. It moved the shape of the ceiling; it did not remove it.
:mod:`nyxara.njp.school` and ``tests/njp/test_unseen_hard.py`` re-measure both rather than
quoting them.

Pure standard library. No LLM anywhere in the path.
"""

from __future__ import annotations

import ast
import itertools
import time
from dataclasses import dataclass, field as dc_field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

__all__ = [
    "CodeError", "Exhausted",
    "Var", "Lit", "Call", "Lambda", "Hole", "Term",
    "Assign", "Unpack", "SetItem", "If", "For", "While", "Return", "Break", "Continue",
    "Nothing", "Stmt",
    "HIGHER",
    "OPS", "render", "read_python", "arity_of",
    "TraceStep", "Interpreter",
    "Example", "Spec", "Program", "Schema", "Demonstration", "Learned", "Check", "Written",
    "Coder", "SEED_SHAPES", "Sketch", "SKETCHES",
]


class CodeError(Exception):
    """A program did something the language does not allow. Never raised past :class:`Coder`.

    The distinction that matters: this is *the candidate* failing, not the organ. A synthesiser
    that crashed on a bad candidate would be unable to search at all, so every evaluation is
    wrapped and a raised :class:`CodeError` is simply a candidate that does not pass.
    """


class Exhausted(CodeError):
    """The step budget ran out. A separate class because it is not the program being *wrong*."""


# --------------------------------------------------------------------------- #
# the term language
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Var:
    """A parameter reference. ``#0``-style names are positional slots inside a :class:`Schema`."""

    name: str = ""


@dataclass(frozen=True)
class Lit:
    """A constant. Ints, bools, strings and tuples of those — the values the language holds."""

    value: Any = 0


@dataclass(frozen=True)
class Call:
    """An operator applied to arguments. ``op`` must be a key of :data:`OPS`."""

    op: str = ""
    args: Tuple["Term", ...] = ()


@dataclass(frozen=True)
class Lambda:
    """An inner function, only ever an argument to ``map``/``filter``/``fold``."""

    params: Tuple[str, ...] = ()
    body: Any = None


@dataclass(frozen=True)
class Hole:
    """A blank in a :class:`Schema`. ``kind`` restricts what may be dropped into it.

    Kinds: ``int``, ``str``, ``list``, ``bool``, ``any`` for values; ``fn1``/``fn2`` for the
    inner function of a ``map``/``filter`` (one parameter) or a ``fold`` (two).
    """

    index: int = 0
    kind: str = "any"


Term = Union[Var, Lit, Call, Lambda, Hole]


# --------------------------------------------------------------------------- #
# statements — the half of programming that is not an expression
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Assign:
    """``name = expr``. Local to the call, and the only way a name comes to hold anything."""

    name: str = ""
    value: Any = None


@dataclass(frozen=True)
class SetItem:
    """``name[index] = value``, and ``name[index]`` for a mapping too.

    Rebinding rather than mutating: it compiles to ``name = setat(name, index, value)``. See
    :func:`_dict_set` for why, and for the one place that differs from Python.
    """

    name: str = ""
    index: Any = None
    value: Any = None


@dataclass(frozen=True)
class Unpack:
    r"""``a, b = b, a % b``. One statement, because the whole right side is evaluated first.

    Two ``Assign``\ s would be a different program: the second would read the value the first had
    just overwritten, which is exactly the bug parallel assignment exists to prevent, and the
    idiom it exists for — Euclid's algorithm, a Fibonacci step — would silently give wrong
    answers rather than fail.
    """

    names: Tuple[str, ...] = ()
    value: Any = None


@dataclass(frozen=True)
class If:
    """``if test: … else: …``. ``orelse`` is a block and may be empty."""

    test: Any = None
    then: Tuple[Any, ...] = ()
    orelse: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class For:
    """``for name in iterable: …``. ``names`` holds more than one for tuple unpacking."""

    names: Tuple[str, ...] = ()
    iterable: Any = None
    body: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class While:
    """``while test: …``, under the loop budget — which is what keeps "every program halts" true
    once a language has a ``while`` in it at all."""

    test: Any = None
    body: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class Return:
    """``return expr``, or a bare ``return`` when ``value`` is ``None``."""

    value: Any = None


@dataclass(frozen=True)
class Break:
    """``break``."""


@dataclass(frozen=True)
class Continue:
    """``continue``."""


@dataclass(frozen=True)
class Nothing:
    """``pass``, and what a docstring becomes."""


Stmt = Union[Assign, Unpack, SetItem, If, For, While, Return, Break, Continue, Nothing]

#: Sentinels the executor uses to carry control flow out of a nested block. Exceptions would cost
#: a traceback per loop iteration; these are compared by identity and cost nothing.
_BREAK = object()
_CONTINUE = object()


class _Returned:
    """A value on its way out of a function. Distinct from ``None``, which is a real value here."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


# --------------------------------------------------------------------------- #
# the operator table — the entire vocabulary, and the security boundary
# --------------------------------------------------------------------------- #

def _int(value: Any) -> int:
    """Ints only, and ``bool`` is not an int here.

    Python says ``True + True == 2``; a language that agrees would let a predicate leak into
    arithmetic and produce a *plausible wrong answer* rather than an error, which is the one
    outcome a verifier cannot catch.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise CodeError(f"expected an integer, got {type(value).__name__}")
    return value


def _seq(value: Any) -> Tuple[Any, ...]:
    """Anything iterable, as a tuple.

    A set comes back **sorted** where its members can be ordered. Python iterates a set in an
    order that depends on hashing, so ``list(set(xs))`` is a different answer on a different run;
    a language whose whole verification story is "run it on the examples" cannot have an operation
    whose answer moves. Sorting is the smallest change that makes it a function.
    """
    if isinstance(value, tuple):
        return value
    if isinstance(value, str):
        return tuple(value)
    if isinstance(value, frozenset):
        try:
            return tuple(sorted(value))
        except TypeError:
            return tuple(value)
    if isinstance(value, dict):
        return tuple(value.keys())
    raise CodeError(f"expected a sequence, got {type(value).__name__}")


def _size(value: Any) -> int:
    """``len`` over everything that has one."""
    if isinstance(value, (tuple, str, frozenset, dict)):
        return len(value)
    raise CodeError(f"{type(value).__name__} has no length")


def _keep_text(original: Any, items: Tuple[Any, ...]) -> Any:
    """A slice of a string is a string. Returning a tuple of characters is the kind of nearly-right
    that shows up three operations later as a type error in unrelated code."""
    return "".join(items) if isinstance(original, str) else items


#: The largest container or string any operator may produce. "Every program halts" was true of
#: *time* and quietly false of *space*: a loop whose body is ``s = s + s`` doubles its value every
#: iteration and exhausts the machine's memory in about thirty of them, long before a step budget
#: counted in tens of thousands notices. A synthesiser that offers exactly that program by the
#: hundred does not get a failed candidate back — it gets the process killed, which is what
#: happened, twice, silently. A budget that bounds one dimension and not the other is not a budget.
_MAX_SIZE = 100000


def _big(value: int) -> int:
    """An integer that has run away is refused too. Repeated squaring in a loop reaches numbers
    with millions of digits, and arithmetic on those is slow rather than impossible — which is
    worse, because it does not fail, it just stops finishing."""
    if value.bit_length() > 4096:
        raise CodeError("a number grew past what these tasks can need")
    return value


def _bounded(value: Any) -> Any:
    """Refuse a value that has grown past what any answer to these tasks could need."""
    if isinstance(value, (tuple, str, frozenset, dict)) and len(value) > _MAX_SIZE:
        raise CodeError(f"a value grew past {_MAX_SIZE} items")
    return value


def _plus(a: Any, b: Any) -> Any:
    """Python's ``+``, dispatched on what the values actually are.

    The reader used to choose between integer addition and concatenation *statically*, by guessing
    from the shape of the operands, and it guessed wrong on ``out + chr(c)`` and on
    ``recurse(s[1:]) + s[0]`` — both ordinary lines. A static guess about a dynamically typed
    language is a wrong answer waiting for the right input, so the choice is made here, where the
    values are known. Mixed kinds are still an error, exactly as in Python: ``1 + "a"`` raises.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        raise CodeError("a truth value is not a number here")
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    if isinstance(a, str) and isinstance(b, str):
        return _bounded(a + b)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return _bounded(a + b)
    if isinstance(a, frozenset) and isinstance(b, frozenset):
        return _bounded(a | b)
    raise CodeError(f"cannot add {type(a).__name__} and {type(b).__name__}")


def _minus(a: Any, b: Any) -> Any:
    """``-``: integers, or set difference."""
    if isinstance(a, frozenset) and isinstance(b, frozenset):
        return a - b
    return _int(a) - _int(b)


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise CodeError(f"expected a string, got {type(value).__name__}")
    return value


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str, tuple)):
        return bool(value)
    raise CodeError("not a truth value")


def _div(a: Any, b: Any) -> int:
    a, b = _int(a), _int(b)
    if b == 0:
        raise CodeError("division by zero")
    return a // b


def _mod(a: Any, b: Any) -> int:
    a, b = _int(a), _int(b)
    if b == 0:
        raise CodeError("modulo by zero")
    return a % b


def _at(xs: Any, i: Any) -> Any:
    """``xs[i]`` — a position in a sequence, a **key** in a mapping.

    One operator for both because that is what the bracket means in Python and what a reader
    writing ``d[k]`` inside a ``key=`` lambda expects. A missing key raises rather than returning
    ``None``: ``d[k]`` and ``d.get(k)`` are different operations and collapsing them hides the
    bug that the first one exists to surface.
    """
    if isinstance(xs, dict):
        key = _hashable(i)
        if key not in xs:
            raise CodeError(f"no such key: {key!r}")
        return xs[key]
    items, index = _seq(xs), _int(i)
    if not -len(items) <= index < len(items):
        raise CodeError("index out of range")
    return items[index]


def _head(xs: Any) -> Any:
    items = _seq(xs)
    if not items:
        raise CodeError("head of an empty sequence")
    return items[0]


def _last(xs: Any) -> Any:
    items = _seq(xs)
    if not items:
        raise CodeError("last of an empty sequence")
    return items[-1]


def _sorted(xs: Any) -> Tuple[Any, ...]:
    items = _seq(xs)
    try:
        return tuple(sorted(items))
    except TypeError as exc:  # mixed types — a real bug in the candidate
        raise CodeError("cannot order these values") from exc


def _reduce_num(xs: Any, fn: Callable[[Iterable[int]], int], what: str) -> int:
    items = [_int(x) for x in _seq(xs)]
    if not items:
        raise CodeError(f"{what} of an empty sequence")
    return fn(items)


def _cat(a: Any, b: Any) -> Any:
    if isinstance(a, str) and isinstance(b, str):
        return _bounded(a + b)
    return _bounded(_seq(a) + _seq(b))


def _range1(n: Any) -> Tuple[int, ...]:
    count = _int(n)
    if not 0 <= count <= 4096:
        raise CodeError("range out of bounds")
    return tuple(range(count))


def _range2(a: Any, b: Any) -> Tuple[int, ...]:
    lo, hi = _int(a), _int(b)
    if hi - lo > 4096:
        raise CodeError("range out of bounds")
    return tuple(range(lo, hi))


def _mapping(value: Any) -> Dict[Any, Any]:
    if isinstance(value, dict):
        return value
    raise CodeError(f"expected a mapping, got {type(value).__name__}")


def _bag(value: Any) -> frozenset:
    if isinstance(value, frozenset):
        return value
    if isinstance(value, (tuple, str)):
        return frozenset(value)
    raise CodeError(f"expected a set, got {type(value).__name__}")


def _hashable(value: Any) -> Any:
    """Keys and set members have to be hashable, and a dict is not.

    Raised rather than coerced: a program that uses a mutable thing as a key has a bug, and
    silently freezing it would produce a lookup that works here and fails everywhere else.
    """
    try:
        hash(value)
    except TypeError as exc:
        raise CodeError(f"{type(value).__name__} cannot be a key") from exc
    return value


def _dict_set(d: Any, key: Any, value: Any) -> Dict[Any, Any]:
    """Copy-on-write. Every container operation here returns a **new** container.

    Real Python mutates in place and shares the result through every alias. This language has no
    aliasing: ``xs.append(v)`` reads as ``xs = xs + [v]`` and ``d[k] = v`` as ``d = {**d, k: v}``,
    which is the same answer for the accumulator idiom that is 95% of what a first course writes,
    and a *different* answer wherever two names point at one object. That is the one place the
    subset diverges from Python's semantics rather than merely omitting them, and it is stated
    here rather than discovered later: a program that depends on shared mutation is outside what
    she runs, and :func:`read_python` accepts it, so it is a wrong answer rather than a refusal.
    """
    out = dict(_mapping(d))
    out[_hashable(key)] = value
    return out


def _dict_get(d: Any, key: Any, default: Any = None) -> Any:
    return _mapping(d).get(_hashable(key), default)


def _dict_del(d: Any, key: Any) -> Dict[Any, Any]:
    out = dict(_mapping(d))
    out.pop(_hashable(key), None)
    return out


def _index_of(xs: Any, value: Any) -> int:
    items = _seq(xs)
    for index, item in enumerate(items):
        if item == value:
            return index
    return -1


def _replace_in(xs: Any, index: Any, value: Any) -> Any:
    """``xs[i] = v`` as an expression, because there is nothing to mutate."""
    if isinstance(xs, dict):
        return _dict_set(xs, index, value)
    items = list(_seq(xs))
    position = _int(index)
    if not -len(items) <= position < len(items):
        raise CodeError("index out of range")
    items[position] = value
    return tuple(items)


def _slice3(xs: Any, lo: Any, hi: Any) -> Any:
    items = _seq(xs)
    low = 0 if lo is None else _int(lo)
    high = len(items) if hi is None else _int(hi)
    out = items[low:high]
    return "".join(out) if isinstance(xs, str) else out


def _to_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise CodeError(f"{value!r} is not a number") from exc
    raise CodeError(f"cannot read {type(value).__name__} as an integer")


def _to_str(value: Any) -> str:
    if isinstance(value, tuple):
        return "[" + ", ".join(_to_str(v) for v in value) + "]"
    return str(value)


def _split_on(text: Any, sep: Any) -> Tuple[str, ...]:
    body, marker = _text(text), _text(sep)
    return tuple(body.split()) if not marker else tuple(body.split(marker))


def _fold_plus(items: Tuple[Any, ...]) -> Any:
    """``sum`` over things that are not integers — the concatenating kind."""
    if not items:
        return 0
    total = items[0]
    for item in items[1:]:
        total = _plus(total, item)
    return total


def _range3(a: Any, b: Any, step: Any) -> Tuple[int, ...]:
    lo, hi, by = _int(a), _int(b), _int(step)
    if by == 0:
        raise CodeError("range step of zero")
    span = abs(hi - lo)
    if span // abs(by) > 4096:
        raise CodeError("range out of bounds")
    return tuple(range(lo, hi, by))


def _pow(base: Any, exponent: Any) -> int:
    a, b = _int(base), _int(exponent)
    if b < 0:
        raise CodeError("negative exponent is not an integer power")
    if b > 64 or abs(a) > 10 ** 6:
        raise CodeError("power out of bounds")
    return a ** b


def _sort_by(fn: Any, xs: Any, caller: Any) -> Tuple[Any, ...]:
    items = list(_seq(xs))
    return tuple(sorted(items, key=lambda item: caller(fn, item)))


#: name → (arity, implementation, rendering template). The rendering template is a Python
#: expression with ``{0}``/``{1}`` placeholders, used by :func:`render` so a synthesised program
#: can be read by a person. ``if``/``and``/``or`` are absent: they are lazy and the interpreter
#: handles them itself, because evaluating the untaken branch of a conditional turns a correct
#: program into a :class:`CodeError` (``if_(gt(n,0), div(x,n), 0)`` is the standard example).
OPS: Dict[str, Tuple[int, Callable[..., Any], str]] = {
    # arithmetic
    "add":      (2, lambda a, b: _int(a) + _int(b), "({0} + {1})"),
    # `plus` and `minus` are what the reader emits for `+` and `-`; `add` and `sub` stay strict
    # and integer-only because the synthesiser enumerates them, and a search arm that silently
    # concatenates when it meant to add finds programs that pass by coincidence.
    "plus":     (2, _plus, "({0} + {1})"),
    "minus":    (2, _minus, "({0} - {1})"),
    "sub":      (2, lambda a, b: _int(a) - _int(b), "({0} - {1})"),
    "mul":      (2, lambda a, b: _big(_int(a) * _int(b)), "({0} * {1})"),
    "div":      (2, _div, "({0} // {1})"),
    "mod":      (2, _mod, "({0} % {1})"),
    "neg":      (1, lambda a: -_int(a), "(-{0})"),
    "abs":      (1, lambda a: abs(_int(a)), "abs({0})"),
    "min2":     (2, lambda a, b: min(_int(a), _int(b)), "min({0}, {1})"),
    "max2":     (2, lambda a, b: max(_int(a), _int(b)), "max({0}, {1})"),
    # comparison — generic, because equality on strings and tuples is meaningful
    "eq":       (2, lambda a, b: a == b, "({0} == {1})"),
    "ne":       (2, lambda a, b: a != b, "({0} != {1})"),
    "lt":       (2, lambda a, b: _int(a) < _int(b), "({0} < {1})"),
    "le":       (2, lambda a, b: _int(a) <= _int(b), "({0} <= {1})"),
    "gt":       (2, lambda a, b: _int(a) > _int(b), "({0} > {1})"),
    "ge":       (2, lambda a, b: _int(a) >= _int(b), "({0} >= {1})"),
    "not":      (1, lambda a: not _truth(a), "(not {0})"),
    # sequences
    "len":      (1, _size, "len({0})"),
    "sum":      (1, lambda xs: sum(_int(x) for x in _seq(xs)), "sum({0})"),
    "sumany":   (1, lambda xs: _fold_plus(_seq(xs)), "sum({0})"),
    "maxof":    (1, lambda xs: _reduce_num(xs, max, "max"), "max({0})"),
    "minof":    (1, lambda xs: _reduce_num(xs, min, "min"), "min({0})"),
    "rev":      (1, lambda xs: xs[::-1] if isinstance(xs, str) else _seq(xs)[::-1],
                 "{0}[::-1]"),
    "sort":     (1, _sorted, "sorted({0})"),
    "head":     (1, _head, "{0}[0]"),
    "last":     (1, _last, "{0}[-1]"),
    "tail":     (1, lambda xs: _keep_text(xs, _seq(xs)[1:]), "{0}[1:]"),
    "at":       (2, _at, "{0}[{1}]"),
    "take":     (2, lambda xs, n: _keep_text(xs, _seq(xs)[:_int(n)]), "{0}[:{1}]"),
    "drop":     (2, lambda xs, n: _keep_text(xs, _seq(xs)[_int(n):]), "{0}[{1}:]"),
    "cat":      (2, _cat, "({0} + {1})"),
    "count":    (2, lambda xs, x: (xs.count(x) if isinstance(xs, str) and isinstance(x, str)
                                   else sum(1 for item in _seq(xs) if item == x)),
                 "{0}.count({1})"),
    "has":      (2, lambda xs, x: ((x in xs) if isinstance(xs, str) and isinstance(x, str)
                                   else (_hashable(x) in xs if isinstance(xs, (dict, frozenset))
                                         else any(item == x for item in _seq(xs)))),
                 "({1} in {0})"),
    "uniq":     (1, lambda xs: tuple(dict.fromkeys(_seq(xs))), "list(dict.fromkeys({0}))"),
    "range":    (1, _range1, "list(range({0}))"),
    "range2":   (2, _range2, "list(range({0}, {1}))"),
    "range3":   (3, _range3, "list(range({0}, {1}, {2}))"),
    # strings
    "upper":    (1, lambda s: _text(s).upper(), "{0}.upper()"),
    "lower":    (1, lambda s: _text(s).lower(), "{0}.lower()"),
    "words":    (1, lambda s: tuple(_text(s).split()), "{0}.split()"),
    "join":     (2, lambda sep, xs: _text(sep).join(_text(x) for x in _seq(xs)),
                 "{0}.join({1})"),
    # numbers, continued
    "pow":      (2, _pow, "({0} ** {1})"),
    "toint":    (1, _to_int, "int({0})"),
    "tostr":    (1, _to_str, "str({0})"),
    "chr":      (1, lambda n: chr(_int(n) % 0x110000), "chr({0})"),
    "ord":      (1, lambda c: ord(_text(c)[:1] or "\0"), "ord({0})"),
    # sequences, continued — every one of these RETURNS A NEW CONTAINER; see `_dict_set`
    "append":   (2, lambda xs, v: _bounded(_seq(xs) + (v,)), "({0} + [{1}])"),
    "extend":   (2, lambda xs, ys: _bounded(_seq(xs) + _seq(ys)), "({0} + {1})"),
    "setat":    (3, _replace_in, "setat({0}, {1}, {2})"),
    "slice":    (3, _slice3, "{0}[{1}:{2}]"),
    "find":     (2, _index_of, "find({0}, {1})"),
    "tolist":   (1, _seq, "list({0})"),
    "flat":     (1, lambda xs: tuple(v for row in _seq(xs) for v in _seq(row)),
                 "[v for row in {0} for v in row]"),
    "zip":      (2, lambda a, b: tuple(zip(_seq(a), _seq(b))), "list(zip({0}, {1}))"),
    "pairs":    (1, lambda xs: tuple(enumerate(_seq(xs))), "list(enumerate({0}))"),
    "first":    (1, lambda p: _seq(p)[0] if _seq(p) else None, "{0}[0]"),
    "second":   (1, lambda p: _seq(p)[1] if len(_seq(p)) > 1 else None, "{0}[1]"),
    "pair":     (2, lambda a, b: (a, b), "({0}, {1})"),
    "anyof":    (1, lambda xs: any(_truth(v) for v in _seq(xs)), "any({0})"),
    "allof":    (1, lambda xs: all(_truth(v) for v in _seq(xs)), "all({0})"),
    # mappings
    "emptydict": (0, dict, "{{}}"),
    "get":      (3, _dict_get, "{0}.get({1}, {2})"),
    "put":      (3, _dict_set, "put({0}, {1}, {2})"),
    "dropkey":  (2, _dict_del, "dropkey({0}, {1})"),
    "haskey":   (2, lambda d, k: _hashable(k) in _mapping(d), "({1} in {0})"),
    "keys":     (1, lambda d: tuple(_mapping(d).keys()), "list({0}.keys())"),
    "vals":     (1, lambda d: tuple(_mapping(d).values()), "list({0}.values())"),
    "items":    (1, lambda d: tuple(_mapping(d).items()), "list({0}.items())"),
    "frompairs": (1, lambda xs: {_hashable(_seq(p)[0]): _seq(p)[1] for p in _seq(xs)},
                  "dict({0})"),
    # sets
    "setof":    (1, lambda xs: frozenset(_hashable(v) for v in _seq(xs)), "set({0})"),
    "inset":    (2, lambda s, v: _hashable(v) in _bag(s), "({1} in {0})"),
    "union":    (2, lambda a, b: _bag(a) | _bag(b), "({0} | {1})"),
    "inter":    (2, lambda a, b: _bag(a) & _bag(b), "({0} & {1})"),
    "without":  (2, lambda a, b: _bag(a) - _bag(b), "({0} - {1})"),
    "symdiff":  (2, lambda a, b: _bag(a) ^ _bag(b), "({0} ^ {1})"),
    # strings, continued
    "strip":    (1, lambda s: _text(s).strip(), "{0}.strip()"),
    "replace":  (3, lambda s, a, b: _text(s).replace(_text(a), _text(b)),
                 "{0}.replace({1}, {2})"),
    "starts":   (2, lambda s, p: _text(s).startswith(_text(p)), "{0}.startswith({1})"),
    "ends":     (2, lambda s, p: _text(s).endswith(_text(p)), "{0}.endswith({1})"),
    "spliton":  (2, _split_on, "{0}.split({1})"),
    "isdigit":  (1, lambda s: _text(s).isdigit(), "{0}.isdigit()"),
    "isalpha":  (1, lambda s: _text(s).isalpha(), "{0}.isalpha()"),
    "title":    (1, lambda s: _text(s).title(), "{0}.title()"),
    # nothing
    "isnone":   (1, lambda v: v is None, "({0} is None)"),
    # higher order — the first argument is a Lambda, handled in the interpreter
    # The rendering templates matter more here than anywhere else in the table: a schema's key
    # *is* its rendering, so two shapes that render alike are one shape as far as learning is
    # concerned. These three carried empty templates once, and every map/filter/fold schema
    # collapsed onto the same key the moment its function was a hole.
    "map":      (2, None, "map({0}, {1})"),
    "filter":   (2, None, "filter({0}, {1})"),
    "fold":     (3, None, "fold({0}, {1}, {2})"),
    "sortby":   (2, None, "sorted({1}, key={0})"),
    "maxby":    (2, None, "max({1}, key={0})"),
    "minby":    (2, None, "min({1}, key={0})"),
    "takewhile": (2, None, "takewhile({0}, {1})"),
    "dropwhile": (2, None, "dropwhile({0}, {1})"),
    "countif":  (2, None, "sum(1 for x in {1} if {0}(x))"),
    "findfirst": (2, None, "next((x for x in {1} if {0}(x)), None)"),
    # Applying a function to values, rather than folding it over a sequence. This is what lets a
    # *shape* leave its operator blank: `apply2(?f, n, recurse(n - 1))` is the recursive skeleton
    # that factorial and sum-to-n share, and the thing that differs between them is a filling.
    "apply1":   (2, None, "{0}({1})"),
    "apply2":   (3, None, "{0}({1}, {2})"),
}

#: Higher-order operators: the first argument is a :class:`Lambda` and the interpreter applies it
#: rather than evaluating it. Kept as a set so a new one cannot be added to :data:`OPS` and be
#: silently evaluated as an ordinary call.
HIGHER: frozenset = frozenset({
    "map", "filter", "fold", "sortby", "maxby", "minby",
    "takewhile", "dropwhile", "countif", "findfirst", "apply1", "apply2",
})

#: Lazy forms. Not in :data:`OPS` because their arguments must *not* all be evaluated.
LAZY: Dict[str, int] = {"if": 3, "and": 2, "or": 2}


def arity_of(op: str) -> int:
    """How many arguments ``op`` takes, lazy forms included. ``-1`` for an unknown operator."""
    if op in LAZY:
        return LAZY[op]
    entry = OPS.get(op)
    return entry[0] if entry else -1


# --------------------------------------------------------------------------- #
# the interpreter
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TraceStep:
    """One evaluated sub-expression, for reading a program back to whoever asked.

    Depth is kept so the trace can be printed as a tree; ``source`` is the rendered Python of
    the sub-expression, so a step reads as ``(x * 2) = 8`` rather than as a term dump.
    """

    depth: int = 0
    source: str = ""
    value: Any = None

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{'  ' * self.depth}{self.source} = {_show(self.value)}"


class Interpreter:
    """Evaluates a :class:`Term`. Halts, always, and says why when it does not finish.

    ``max_steps`` counts *evaluated nodes*, not wall time, so the same program costs the same on
    any machine — a budget that varied by hardware would make a synthesis result irreproducible,
    and reproducibility is the only reason to prefer a step counter to a timer here.
    """

    def __init__(self, *, max_steps: int = 200000, max_calls: int = 60) -> None:
        self.max_steps = int(max_steps)
        self.max_calls = int(max_calls)
        self.steps = 0
        #: Evaluated nodes since this interpreter was made, across every run, never reset.
        #: :attr:`steps` is per-run and is what :class:`Exhausted` is measured against; this one
        #: is what a *search* is measured against, because counting candidates says nothing about
        #: what they cost.
        self.spent = 0
        self.depth = 0
        #: How deep ``invoke`` may nest. Sixty, not four hundred, and the number is a fact about
        #: the host rather than a taste: one level here costs about ten CPython frames, and
        #: CPython's own limit is a thousand, so a budget set past ~90 is never reached — the
        #: interpreter dies of :class:`RecursionError` first, which is not a :class:`CodeError`
        #: and therefore escapes every guard the searcher has. Sixty leaves room for the whole
        #: recursive curriculum and keeps the failure inside this module where it can be reported.
        #:
        #: name → :class:`Program`, the only thing ``invoke`` can reach. A program is registered
        #: under its own name before it runs, which is what makes self-recursion possible and is
        #: also the whole of the mechanism: there is no global scope, no import, and no way to
        #: name a function that was not handed to the interpreter.
        self.functions: Dict[str, Any] = {}

    # -- evaluation ------------------------------------------------------- #
    def run(self, term: Term, env: Optional[Dict[str, Any]] = None,
            *, trace: Optional[List[TraceStep]] = None) -> Any:
        """Evaluate ``term`` under ``env``. Raises :class:`CodeError` on anything malformed."""
        self.steps = 0
        self.depth = 0
        return self._eval(term, dict(env or {}), 0, trace)

    # -- execution --------------------------------------------------------- #
    def execute(self, block: Sequence[Any], env: Optional[Dict[str, Any]] = None,
                *, trace: Optional[List[TraceStep]] = None) -> Any:
        """Run a block of statements and return what it returned.

        A function that falls off the end returns ``None``, exactly as Python's does — which is a
        real value here rather than an error, because ``None`` is what a lookup that missed gives
        back and a program has to be able to hand that on.
        """
        self.steps = 0
        self.depth = 0
        scope = dict(env or {})
        outcome = self._block(tuple(block), scope, 0, trace)
        if isinstance(outcome, _Returned):
            return outcome.value
        if outcome is _BREAK or outcome is _CONTINUE:
            raise CodeError("break or continue outside a loop")
        return None

    def _block(self, block: Sequence[Any], env: Dict[str, Any], depth: int,
               trace: Optional[List[TraceStep]]) -> Any:
        for statement in block:
            outcome = self._step(statement, env, depth, trace)
            if outcome is not None:
                return outcome
        return None

    def _step(self, node: Any, env: Dict[str, Any], depth: int,
              trace: Optional[List[TraceStep]]) -> Any:  # noqa: C901 - one arm per statement
        self._tick()
        if isinstance(node, Nothing):
            return None
        if isinstance(node, Assign):
            env[node.name] = self._eval(node.value, env, depth + 1, trace)
            if trace is not None:
                trace.append(TraceStep(depth, f"{node.name} = …", env[node.name]))
            return None
        if isinstance(node, Unpack):
            self._bind(node.names, self._eval(node.value, env, depth + 1, trace), env)
            return None
        if isinstance(node, SetItem):
            if node.name not in env:
                raise CodeError(f"unbound name {node.name!r}")
            env[node.name] = _replace_in(env[node.name],
                                         self._eval(node.index, env, depth + 1, trace),
                                         self._eval(node.value, env, depth + 1, trace))
            return None
        if isinstance(node, Return):
            value = (None if node.value is None
                     else self._eval(node.value, env, depth + 1, trace))
            if trace is not None:
                trace.append(TraceStep(depth, "return", value))
            return _Returned(value)
        if isinstance(node, Break):
            return _BREAK
        if isinstance(node, Continue):
            return _CONTINUE
        if isinstance(node, If):
            branch = node.then if _truth(self._eval(node.test, env, depth + 1, trace)) \
                else node.orelse
            return self._block(branch, env, depth + 1, trace)
        if isinstance(node, For):
            return self._loop_for(node, env, depth, trace)
        if isinstance(node, While):
            return self._loop_while(node, env, depth, trace)
        raise CodeError(f"not a statement: {type(node).__name__}")

    def _bind(self, names: Tuple[str, ...], value: Any, env: Dict[str, Any]) -> None:
        if len(names) == 1:
            env[names[0]] = value
            return
        items = _seq(value)
        if len(items) != len(names):
            raise CodeError(f"cannot unpack {len(items)} values into {len(names)} names")
        for name, item in zip(names, items):
            env[name] = item

    def _loop_for(self, node: For, env: Dict[str, Any], depth: int,
                  trace: Optional[List[TraceStep]]) -> Any:
        for item in _seq(self._eval(node.iterable, env, depth + 1, trace)):
            self._tick()
            self._bind(node.names, item, env)
            outcome = self._block(node.body, env, depth + 1, trace)
            if outcome is _BREAK:
                break
            if outcome is _CONTINUE or outcome is None:
                continue
            return outcome
        return None

    def _loop_while(self, node: While, env: Dict[str, Any], depth: int,
                    trace: Optional[List[TraceStep]]) -> Any:
        """The one construct that can genuinely fail to terminate, and the budget is why it does
        not: every iteration costs a step, and the step budget is finite, so a ``while True`` is
        an :class:`Exhausted` rather than a hung process."""
        while _truth(self._eval(node.test, env, depth + 1, trace)):
            self._tick()
            outcome = self._block(node.body, env, depth + 1, trace)
            if outcome is _BREAK:
                break
            if outcome is _CONTINUE or outcome is None:
                continue
            return outcome
        return None

    def call(self, program: Any, args: Sequence[Any], depth: int = 0,
             trace: Optional[List[TraceStep]] = None) -> Any:
        """Apply a :class:`Program`, block-bodied or expression-bodied. Used by ``invoke``."""
        if len(args) != len(program.params):
            raise CodeError(f"{program.name} takes {len(program.params)} argument(s), "
                            f"given {len(args)}")
        scope = dict(zip(program.params, args))
        if program.is_block:
            outcome = self._block(program.body, scope, depth, trace)
            return outcome.value if isinstance(outcome, _Returned) else None
        return self._eval(program.body, scope, depth, trace)

    def _tick(self) -> None:
        self.steps += 1
        self.spent += 1
        if self.steps > self.max_steps:
            raise Exhausted(f"step budget of {self.max_steps} exhausted")

    def _eval(self, term: Term, env: Dict[str, Any], depth: int,
              trace: Optional[List[TraceStep]]) -> Any:
        self._tick()
        value = self._dispatch(term, env, depth, trace)
        if trace is not None and not isinstance(term, (Lit, Lambda)):
            trace.append(TraceStep(depth=depth, source=render(term), value=value))
        return value

    def _dispatch(self, term: Term, env: Dict[str, Any], depth: int,
                  trace: Optional[List[TraceStep]]) -> Any:
        if isinstance(term, Lit):
            return term.value
        if isinstance(term, Var):
            if term.name not in env:
                raise CodeError(f"unbound name {term.name!r}")
            return env[term.name]
        if isinstance(term, Hole):
            raise CodeError(f"unfilled hole #{term.index}")
        if isinstance(term, Lambda):
            return term
        if not isinstance(term, Call):
            raise CodeError(f"not a term: {type(term).__name__}")
        return self._call(term, env, depth, trace)

    def _call(self, term: Call, env: Dict[str, Any], depth: int,
              trace: Optional[List[TraceStep]]) -> Any:
        op, args = term.op, term.args
        if op in LAZY:
            return self._lazy(op, args, env, depth, trace)
        if op == "invoke":
            return self._invoke(args, env, depth, trace)
        entry = OPS.get(op)
        if entry is None:
            raise CodeError(f"unknown operator {op!r}")
        want, fn, _ = entry
        if len(args) != want:
            raise CodeError(f"{op} takes {want} argument(s), given {len(args)}")
        if op in HIGHER:
            return self._higher(op, args, env, depth, trace)
        values = [self._eval(a, env, depth + 1, trace) for a in args]
        try:
            return fn(*values)
        except CodeError:
            raise
        except Exception as exc:  # noqa: BLE001 — any operator failure is a failed candidate
            raise CodeError(f"{op} failed: {exc}") from exc

    def _lazy(self, op: str, args: Tuple[Term, ...], env: Dict[str, Any], depth: int,
              trace: Optional[List[TraceStep]]) -> Any:
        if len(args) != LAZY[op]:
            raise CodeError(f"{op} takes {LAZY[op]} argument(s), given {len(args)}")
        if op == "if":
            test = _truth(self._eval(args[0], env, depth + 1, trace))
            return self._eval(args[1] if test else args[2], env, depth + 1, trace)
        left = _truth(self._eval(args[0], env, depth + 1, trace))
        if op == "and":
            return left and _truth(self._eval(args[1], env, depth + 1, trace))
        return left or _truth(self._eval(args[1], env, depth + 1, trace))

    def _invoke(self, args: Tuple[Term, ...], env: Dict[str, Any], depth: int,
                trace: Optional[List[TraceStep]]) -> Any:
        """``invoke(name, …)`` — the only call there is, and the only route to recursion.

        The callee is looked up in :attr:`functions`, which holds exactly what the caller
        registered. There is no global scope and no import, so a program can reach its own name,
        the helpers it was handed, and nothing else — which is the same whitelist stance as the
        reader, one level down.

        ``max_calls`` bounds the nesting. Unbounded recursion is the one way a program with no
        ``while`` in it can still fail to stop, and a depth limit reports it as
        :class:`Exhausted` — the honest answer, and the same one CPython gives.
        """
        if not args or not isinstance(args[0], Lit) or not isinstance(args[0].value, str):
            raise CodeError("invoke needs a literal function name")
        name = args[0].value
        program = self.functions.get(name)
        if program is None:
            raise CodeError(f"no function named {name!r}")
        self._tick()
        if self.depth >= self.max_calls:
            raise Exhausted(f"call depth of {self.max_calls} exceeded — is it recursing forever?")
        values = [self._eval(a, env, depth + 1, trace) for a in args[1:]]
        self.depth += 1
        try:
            return self.call(program, values, depth + 1, None)
        finally:
            self.depth -= 1

    def _higher(self, op: str, args: Tuple[Term, ...], env: Dict[str, Any], depth: int,
                trace: Optional[List[TraceStep]]) -> Any:
        """Every higher-order operator, with the host held at arm's length.

        ``sorted(items, key=f)`` raises a bare :class:`TypeError` when the key returns values the
        host cannot order against each other, and that escaped as a *host* exception — past every
        guard the searcher has, out of the interpreter, and into the caller. An enumerative search
        offers exactly those combinations by the thousand, so the leak turned "a candidate does
        not work" into "the search crashed". Nothing may leave here except a
        :class:`CodeError`.
        """
        try:
            return self._apply(op, args, env, depth, trace)
        except (CodeError, Exhausted):
            raise
        except RecursionError as exc:
            raise Exhausted("the host stack gave out inside a higher-order call") from exc
        except Exception as exc:  # noqa: BLE001 — any host failure is a failed candidate
            raise CodeError(f"{op} failed: {exc}") from exc

    def _apply(self, op: str, args: Tuple[Term, ...], env: Dict[str, Any], depth: int,
               trace: Optional[List[TraceStep]]) -> Any:
        fn = args[0]
        if not isinstance(fn, Lambda):
            fn = self._eval(fn, env, depth + 1, trace)
        if not isinstance(fn, Lambda):
            raise CodeError(f"{op} needs a function as its first argument")
        if op in ("apply1", "apply2"):
            wants = 1 if op == "apply1" else 2
            if len(fn.params) != wants:
                raise CodeError(f"{op} needs a {wants}-parameter function")
            values = [self._eval(a, env, depth + 1, trace) for a in args[1:]]
            self._tick()
            return self._eval(fn.body, {**env, **dict(zip(fn.params, values))}, depth + 1, None)
        if op == "fold":
            init = self._eval(args[1], env, depth + 1, trace)
            items = _seq(self._eval(args[2], env, depth + 1, trace))
            if len(fn.params) != 2:
                raise CodeError("fold needs a two-parameter function")
            acc = init
            for item in items:
                self._tick()
                acc = self._eval(fn.body, {**env, fn.params[0]: acc, fn.params[1]: item},
                                 depth + 1, None)
            return acc
        items = _seq(self._eval(args[1], env, depth + 1, trace))
        if len(fn.params) != 1:
            raise CodeError(f"{op} needs a one-parameter function")

        def apply(item: Any) -> Any:
            self._tick()
            return self._eval(fn.body, {**env, fn.params[0]: item}, depth + 1, None)

        if op == "map":
            return tuple(apply(item) for item in items)
        if op == "filter":
            return tuple(item for item in items if _truth(apply(item)))
        if op == "countif":
            return sum(1 for item in items if _truth(apply(item)))
        if op == "findfirst":
            for item in items:
                if _truth(apply(item)):
                    return item
            return None
        if op == "takewhile":
            out: List[Any] = []
            for item in items:
                if not _truth(apply(item)):
                    break
                out.append(item)
            return tuple(out)
        if op == "dropwhile":
            index = 0
            while index < len(items) and _truth(apply(items[index])):
                index += 1
            return items[index:]
        if op == "sortby":
            return tuple(sorted(items, key=apply))
        if not items:
            raise CodeError(f"{op} of an empty sequence")
        return (max if op == "maxby" else min)(items, key=apply)


# --------------------------------------------------------------------------- #
# rendering — a term, read back as Python
# --------------------------------------------------------------------------- #

def _show(value: Any) -> str:
    """Values print as Python literals, with tuples shown as lists — they *are* her lists."""
    if isinstance(value, tuple):
        return "[" + ", ".join(_show(v) for v in value) + "]"
    if isinstance(value, Lambda):
        return f"lambda {', '.join(value.params)}: {render(value.body)}"
    return repr(value)


#: Operators with no Python surface to render into. ``fold`` is not a builtin and
#: ``functools.reduce`` needs an import the language does not have, so these print as themselves
#: — and :data:`_DIALECT` teaches the reader to take them back, which is what keeps
#: :meth:`Program.source` a round trip rather than a picture of one.
_DIALECT_CALLS = frozenset({"fold", "takewhile", "dropwhile", "findfirst", "countif",
                            "put", "setat", "dropkey", "find", "pair", "frompairs", "flat"})


def _inline(fn: Lambda, args: Sequence[Term]) -> Term:
    """``apply2(lambda a, x: a * x, total, n)`` read back as ``(total * n)``.

    Rendering the application literally gives ``lambda a, x: (a * x)(total, n)``, which is not
    the program — it is not even valid Python, because the lambda swallows the call. Since
    :meth:`Program.source` is what a person reviews and what the reader has to be able to take
    back, the application is inlined, which is exactly what it means.
    """
    bound = dict(zip(fn.params, args))

    def walk(node: Term) -> Term:
        if isinstance(node, Var):
            return bound.get(node.name, node)
        if isinstance(node, Call):
            return Call(node.op, tuple(walk(a) for a in node.args))
        if isinstance(node, Lambda):
            inner = {k: v for k, v in bound.items() if k not in node.params}
            return Lambda(node.params, _inline(Lambda(tuple(inner), node.body),
                                               tuple(inner.values()))) if inner else node
        return node

    return walk(fn.body)


def _as_list_literal(term: Term) -> Optional[str]:
    """``append(append([], a), b)`` reads back as ``[a, b]``.

    A list literal whose items are not constants is *built* rather than stored — there is no
    node for it — and rendering the construction gives ``(([] + [a]) + [b])``, which is correct
    and unreadable. Since :meth:`Program.source` is the thing a person reviews, the chain is
    recognised and printed as what it was written as.
    """
    items: List[str] = []
    node = term
    while isinstance(node, Call) and node.op == "append" and len(node.args) == 2:
        items.append(render(node.args[1]))
        node = node.args[0]
    if not items or not (isinstance(node, Lit) and node.value == ()):
        return None
    return "[" + ", ".join(reversed(items)) + "]"


def render(term: Term) -> str:
    """A :class:`Term` as readable Python. Round-trips through :func:`read_python` for the
    subset that reader accepts, which is most of what :func:`render` emits."""
    if isinstance(term, Lit):
        return _show(term.value)
    if isinstance(term, Var):
        return term.name
    if isinstance(term, Hole):
        return f"?{term.kind}#{term.index}"
    if isinstance(term, Lambda):
        return f"lambda {', '.join(term.params)}: {render(term.body)}"
    if not isinstance(term, Call):
        return "<?>"
    literal = _as_list_literal(term)
    if literal is not None:
        return literal
    if term.op in ("apply1", "apply2") and term.args and isinstance(term.args[0], Lambda):
        return render(_inline(term.args[0], term.args[1:]))
    if term.op in _DIALECT_CALLS:
        return f"{term.op}({', '.join(render(a) for a in term.args)})"
    if term.op == "invoke" and term.args and isinstance(term.args[0], Lit):
        return f"{term.args[0].value}({', '.join(render(a) for a in term.args[1:])})"
    op, args = term.op, tuple(render(a) for a in term.args)
    if op == "if":
        return f"({args[1]} if {args[0]} else {args[2]})"
    if op in ("and", "or"):
        return f"({args[0]} {op} {args[1]})"
    if op == "map" and isinstance(term.args[0], Lambda):
        inner = term.args[0]
        return f"[{render(inner.body)} for {inner.params[0]} in {args[1]}]"
    if op == "filter" and isinstance(term.args[0], Lambda):
        inner = term.args[0]
        name = inner.params[0]
        return f"[{name} for {name} in {args[1]} if {render(inner.body)}]"
    entry = OPS.get(op)
    if entry is None:
        return f"{op}({', '.join(args)})"
    return entry[2].format(*args)


def render_block(block: Sequence[Any], level: int = 0) -> List[str]:
    """A block of statements as indented Python lines. The inverse of what the reader accepts."""
    pad = "    " * level
    lines: List[str] = []
    for node in block:
        if isinstance(node, Nothing):
            lines.append(f"{pad}pass")
        elif isinstance(node, Assign):
            lines.append(f"{pad}{node.name} = {render(node.value)}")
        elif isinstance(node, Unpack):
            lines.append(f"{pad}{', '.join(node.names)} = {render(node.value)}")
        elif isinstance(node, SetItem):
            lines.append(f"{pad}{node.name}[{render(node.index)}] = {render(node.value)}")
        elif isinstance(node, Return):
            lines.append(f"{pad}return" if node.value is None
                         else f"{pad}return {render(node.value)}")
        elif isinstance(node, Break):
            lines.append(f"{pad}break")
        elif isinstance(node, Continue):
            lines.append(f"{pad}continue")
        elif isinstance(node, If):
            lines.append(f"{pad}if {render(node.test)}:")
            lines.extend(render_block(node.then, level + 1) or [f"{pad}    pass"])
            if node.orelse:
                lines.append(f"{pad}else:")
                lines.extend(render_block(node.orelse, level + 1))
        elif isinstance(node, For):
            target = ", ".join(node.names)
            lines.append(f"{pad}for {target} in {render(node.iterable)}:")
            lines.extend(render_block(node.body, level + 1) or [f"{pad}    pass"])
        elif isinstance(node, While):
            lines.append(f"{pad}while {render(node.test)}:")
            lines.extend(render_block(node.body, level + 1) or [f"{pad}    pass"])
        else:  # pragma: no cover - a node type with no rendering is a bug, not a fallback
            lines.append(f"{pad}<?{type(node).__name__}>")
    return lines


def _walk_term(node: Any, fn: Callable[[Any], Any]) -> Any:
    """Rebuild any node — expression or statement — with ``fn`` applied to every expression."""
    if isinstance(node, Assign):
        return Assign(node.name, fn(node.value))
    if isinstance(node, Unpack):
        return Unpack(node.names, fn(node.value))
    if isinstance(node, SetItem):
        return SetItem(node.name, fn(node.index), fn(node.value))
    if isinstance(node, Return):
        return Return(None if node.value is None else fn(node.value))
    if isinstance(node, If):
        return If(fn(node.test), tuple(_walk_term(b, fn) for b in node.then),
                  tuple(_walk_term(b, fn) for b in node.orelse))
    if isinstance(node, For):
        return For(node.names, fn(node.iterable), tuple(_walk_term(b, fn) for b in node.body))
    if isinstance(node, While):
        return While(fn(node.test), tuple(_walk_term(b, fn) for b in node.body))
    if isinstance(node, (Break, Continue, Nothing)):
        return node
    return fn(node)


def _is_stmt(node: Any) -> bool:
    return isinstance(node, (Assign, Unpack, SetItem, If, For, While, Return, Break, Continue,
                             Nothing))


# --------------------------------------------------------------------------- #
# tasks, programs and the shapes distilled out of them
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Example:
    """One input/output pair. ``args`` is positional and lists are tuples — values are hashable
    here so a candidate's behaviour can be memoised and duplicates dropped."""

    args: Tuple[Any, ...] = ()
    out: Any = None


def _example_cost(example: "Example") -> Tuple[int, int]:
    """How expensive this example is likely to be to run. Magnitudes for numbers, lengths for
    everything else — a proxy, and it only has to get the *order* roughly right."""
    total = 0
    for arg in example.args:
        if isinstance(arg, bool):
            total += 1
        elif isinstance(arg, int):
            total += abs(arg)
        elif isinstance(arg, (tuple, str, frozenset, dict)):
            total += len(arg) * 2
    return total, len(example.args)


@dataclass
class Spec:
    """A task: what it is called, what it is for, and the pairs that decide the question.

    The split is the whole discipline. ``shown`` is what synthesis may fit — she is allowed to
    search until she matches every one of them. ``held_out`` she never sees during search, and it
    is what :class:`~nyxara.njp.school.School` grades on. A program that passes ``shown`` and
    fails ``held_out`` is the interesting failure and is reported as one rather than smoothed
    away: it means the search found a coincidence, which is what an unseen case is for.
    """

    name: str = ""
    intent: str = ""
    params: Tuple[str, ...] = ()
    shown: Tuple[Example, ...] = ()
    held_out: Tuple[Example, ...] = ()

    @staticmethod
    def of(name: str, intent: str, params: Sequence[str],
           shown: Sequence[Tuple[Sequence[Any], Any]],
           held_out: Sequence[Tuple[Sequence[Any], Any]] = ()) -> "Spec":
        """Build a spec from plain pairs: ``([args], out)``. Lists are frozen to tuples."""
        def freeze(value: Any) -> Any:
            return tuple(freeze(v) for v in value) if isinstance(value, (list, tuple)) else value

        # Shown examples are ordered **cheapest first**, and it is a performance decision with a
        # correctness flavour. `matches` refutes on the first example that disagrees, and almost
        # every candidate a search looks at is wrong — so the first example is the one that gets
        # run millions of times. Putting `fib(14)` there makes refuting a wrong recursion cost
        # what computing the right one does; putting `fib(1)` there makes it free. The set is the
        # same set, and `check` reports over all of it regardless.
        made = tuple(Example(tuple(freeze(a) for a in args), freeze(out))
                     for args, out in shown)
        return Spec(name=name, intent=intent, params=tuple(params),
                    shown=tuple(sorted(made, key=_example_cost)),
                    held_out=tuple(Example(tuple(freeze(a) for a in args), freeze(out))
                                   for args, out in held_out))

    @property
    def arity(self) -> int:
        return len(self.params)

    def constants(self) -> Tuple[List[int], List[str]]:
        """Ints and strings that appear anywhere in the shown pairs.

        Search needs *some* source of literals and inventing them from nothing is a much bigger
        space than the task gives any reason to explore. Taking them from the examples is the
        standard move and it is also the honest one: these are constants the task itself put on
        the table, not knowledge from somewhere else.
        """
        ints: List[int] = []
        texts: List[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, bool):
                return
            if isinstance(value, int):
                ints.append(value)
            elif isinstance(value, str):
                texts.append(value)
                # The characters too. A predicate like `ch == 'e'` or `s.count('a')` needs the
                # single character, and a task whose only strings are whole sentences would
                # otherwise never put one in the pool — so every string task that turns on one
                # letter was unreachable however long the search ran.
                texts.extend(dict.fromkeys(value[:40]))
            elif isinstance(value, (tuple, frozenset)):
                for item in value:
                    walk(item)
            elif isinstance(value, dict):
                # Mappings were skipped entirely, so a threshold that only ever appears as a
                # dict *value* was not a candidate constant.
                for key, item in value.items():
                    walk(key)
                    walk(item)

        for example in self.shown:
            for arg in example.args:
                walk(arg)
            walk(example.out)
        seen_i = sorted({i for i in ints if -64 <= i <= 64})
        seen_s = sorted({s for s in texts if len(s) <= 32}, key=lambda t: (len(t), t))
        return seen_i, seen_s


@dataclass(frozen=True)
class Program:
    """A finished function: its parameters, and either one expression or a block of statements.

    Both shapes are first class. An expression body is what synthesis produces and what a schema
    abstracts most cleanly; a block body is what a person writes and what recursion, loops and
    local variables need. ``helpers`` carries any other functions this one may ``invoke`` — a
    program that calls a helper travels with it, because a program that needs something the
    interpreter was not handed is a program that does not run.
    """

    name: str = "f"
    params: Tuple[str, ...] = ()
    body: Any = None
    helpers: Tuple["Program", ...] = ()

    @property
    def is_block(self) -> bool:
        return isinstance(self.body, tuple)

    @property
    def recursive(self) -> bool:
        """Does it call itself? Reported because it is the one property that changes what a
        reader has to hold in their head, and the one that a step budget exists for."""
        return any(isinstance(node, Call) and node.op == "invoke" and node.args
                   and isinstance(node.args[0], Lit) and node.args[0].value == self.name
                   for _path, node in _positions(self.body))

    def source(self) -> str:
        """The program as Python source. Nothing executes this — it is for people."""
        heads = [helper.source() for helper in self.helpers]
        signature = f"def {self.name}({', '.join(self.params)}):"
        if self.is_block:
            lines = render_block(self.body, 1) or ["    pass"]
            heads.append(signature + "\n" + "\n".join(lines))
        else:
            heads.append(f"{signature}\n    return {render(self.body)}")
        return "\n\n".join(heads)

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.source()


@dataclass
class Schema:
    """A control-flow shape with its constants and inner functions blanked out.

    This is what survives a lesson. The parameters are positional (``#0``, ``#1``) so the shape
    can land on a task whose variables are named something else, and the posterior is Laplace over
    the times it was confirmed and the times a demonstration carrying it was refuted — one
    demonstration is one observation, never certainty.
    """

    key: str = ""
    arity: int = 1
    skeleton: Any = None
    holes: Tuple[Hole, ...] = ()
    confirmed: int = 0
    refuted: int = 0
    seed: bool = False
    block: bool = False
    origin: str = ""
    last_used: float = 0.0
    solved: int = 0
    #: hole index → the concrete terms that have actually stood there, most recent first.
    #: A shape generalises; the constants that have been seen to go in it make it *cheap*. A
    #: six-hole skeleton has millions of fillings and one of them is the one the demonstration
    #: used, so trying the remembered values first turns a combinatorial search back into a
    #: lookup — while the rest of the pool stays behind them, so a task with a new constant is
    #: still reachable rather than merely slower.
    seen: Dict[int, List[Any]] = dc_field(default_factory=dict)

    def remember(self, fillings: Sequence[Any]) -> None:
        for index, filling in enumerate(fillings):
            slot = self.seen.setdefault(index, [])
            if filling in slot:
                slot.remove(filling)
            slot.insert(0, filling)
            del slot[6:]

    @property
    def posterior(self) -> float:
        """``(confirmed + 1) / (confirmed + refuted + 2)`` — a shape nobody has demonstrated sits
        at 0.5 rather than at zero, because an unseen shape is unknown, not wrong."""
        return (self.confirmed + 1.0) / (self.confirmed + self.refuted + 2.0)

    @property
    def cost(self) -> int:
        """How many blanks have to be guessed. Search tries the cheap shapes first."""
        return len(self.holes)

    def instantiate(self, params: Sequence[str], fillings: Sequence[Term],
                    *, name: str = "f") -> Any:
        """Bind ``#i`` to ``params[i]``, ``#self`` to ``name``, and fill the holes."""
        return _substitute(self.skeleton, tuple(params), tuple(fillings), name=name)

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "arity": self.arity, "holes": self.cost, "block": self.block,
                "confirmed": self.confirmed, "refuted": self.refuted,
                "posterior": round(self.posterior, 4), "seed": self.seed,
                "origin": self.origin, "solved": self.solved}


def _fillings_of(skeleton: Any, body: Any) -> List[Any]:
    """Read back what the demonstration actually had where the skeleton has holes.

    Walked in parallel rather than recomputed: :func:`abstract` numbers the holes in traversal
    order, so the same traversal over the original body lands on the term each hole replaced.
    """
    found: Dict[int, Any] = {}

    def walk(shape: Any, real: Any) -> None:
        if isinstance(shape, Hole):
            found[shape.index] = real
            return
        if isinstance(shape, tuple) and isinstance(real, tuple):
            for a, b in zip(shape, real):
                walk(a, b)
            return
        if isinstance(shape, Call) and isinstance(real, Call):
            for a, b in zip(shape.args, real.args):
                walk(a, b)
            return
        if isinstance(shape, Lambda) and isinstance(real, Lambda):
            walk(shape.body, real.body)
            return
        if _is_stmt(shape) and _is_stmt(real) and type(shape) is type(real):
            for a, b in zip(_children(shape), _children(real)):
                walk(a, b)

    walk(skeleton, body)
    return [found.get(index) for index in range(len(found) and max(found) + 1)]


def _substitute(term: Any, params: Tuple[str, ...], fillings: Tuple[Term, ...],
                *, name: str = "") -> Any:
    if isinstance(term, tuple) and (not term or _is_stmt(term[0])):
        return tuple(_substitute(node, params, fillings, name=name) for node in term)
    if _is_stmt(term):
        return _walk_term(term, lambda node: _substitute(node, params, fillings, name=name))
    if isinstance(term, Hole):
        if term.index >= len(fillings):
            raise CodeError(f"no filling for hole #{term.index}")
        return fillings[term.index]
    if isinstance(term, Var):
        if term.name.startswith("#"):
            slot = int(term.name[1:])
            if slot >= len(params):
                raise CodeError(f"schema wants {slot + 1} parameters")
            return Var(params[slot])
        return term
    if isinstance(term, Lambda):
        return Lambda(term.params, _substitute(term.body, params, fillings, name=name))
    if isinstance(term, Call):
        if (term.op == "invoke" and term.args and isinstance(term.args[0], Lit)
                and term.args[0].value == "#self"):
            # A recursive schema names itself positionally, the same way it names its parameters,
            # so the shape can land on a task whose function is called something else.
            head: Tuple[Any, ...] = (Lit(name or "f"),)
            rest = tuple(_substitute(a, params, fillings, name=name) for a in term.args[1:])
            return Call("invoke", head + rest)
        return Call(term.op, tuple(_substitute(a, params, fillings, name=name)
                                   for a in term.args))
    return term


def _holes_of(term: Term) -> Tuple[Hole, ...]:
    found: List[Hole] = []

    def walk(node: Any) -> None:
        if isinstance(node, Hole):
            found.append(node)
        elif isinstance(node, Call):
            for arg in node.args:
                walk(arg)
        elif isinstance(node, Lambda):
            walk(node.body)
        elif isinstance(node, tuple):
            for item in node:
                walk(item)
        elif _is_stmt(node):
            _walk_term(node, lambda inner: (walk(inner), inner)[1])

    walk(term)
    return tuple(sorted(found, key=lambda h: h.index))


def _renumber(term: Any, offset: int) -> Any:
    """Shift every hole index by ``offset`` — needed when one schema is grafted into another."""
    if isinstance(term, tuple) and (not term or _is_stmt(term[0])):
        return tuple(_renumber(node, offset) for node in term)
    if _is_stmt(term):
        return _walk_term(term, lambda node: _renumber(node, offset))
    if isinstance(term, Hole):
        return Hole(term.index + offset, term.kind)
    if isinstance(term, Call):
        return Call(term.op, tuple(_renumber(a, offset) for a in term.args))
    if isinstance(term, Lambda):
        return Lambda(term.params, _renumber(term.body, offset))
    return term


def _kind_of(value: Any) -> str:
    """What kind of value this is. Every kind the language has, and no catch-all for the ones it
    grew: mappings and sets returned ``"any"`` long after they were real values, which made
    :func:`_element_kinds` report a dictionary task as having no kinds at all — so the function
    pool was pruned down to the identity and every mapping task was unreachable."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, str):
        return "str"
    if isinstance(value, tuple):
        return "list"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, frozenset):
        return "set"
    if value is None:
        return "none"
    return "any"


def abstract(program: Program) -> Schema:
    """Throw the program away and keep its shape.

    Every literal becomes a hole of its own kind and every inner function becomes a hole of the
    right arity; parameters become positions. What is left is the thing that transfers — the
    order of the operations — and *only* that. This is the mechanical version of
    :mod:`nyxara.njp.teacher`'s rule that a demonstration's conclusion is discarded and its
    structure retained, and it is the reason a lesson about doubling even numbers is worth
    something on a task about tripling odd ones.
    """
    slots = {name: index for index, name in enumerate(program.params)}
    counter = itertools.count()

    def walk(node: Any) -> Any:
        if _is_stmt(node) or (isinstance(node, tuple) and node and _is_stmt(node[0])):
            return _walk_term(node, walk) if _is_stmt(node) else tuple(walk(n) for n in node)
        if isinstance(node, Call):
            if node.op == "invoke" and node.args and isinstance(node.args[0], Lit):
                # The callee's *name* is not a constant to be blanked — it is the identity of the
                # call. A self-call becomes `#self` so the shape can land on a function called
                # something else, which is what makes a recursive shape transferable at all.
                callee = node.args[0].value
                head = Lit("#self") if callee == program.name else node.args[0]
                return Call("invoke", (head,) + tuple(walk(a) for a in node.args[1:]))
            return Call(node.op, tuple(walk(a) for a in node.args))
        if isinstance(node, Lit):
            if _is_structural(node.value):
                # An empty container is *structure*, not a constant worth varying. Blanking it
                # turned `[] + [b] + [a % b]` — which is how a tuple literal is built, since there
                # is no node for one — into `?list + [b] + [a % b]`, a shape with a hole no pool
                # can fill. Seventeen of sixty-five families failed to be writable that way, and
                # every one of them had been taught: `out = []`, `s = ''`, `[lo, hi]`. The rule
                # is the same one that makes `#self` not a constant — the identity of a
                # construction is not one of its parameters.
                return node
            return Hole(next(counter), _kind_of(node.value))
        if isinstance(node, Lambda):
            return Hole(next(counter), f"fn{len(node.params)}")
        if isinstance(node, Var):
            return Var(f"#{slots[node.name]}") if node.name in slots else node
        return node

    skeleton = walk(program.body)
    key = _skeleton_key(skeleton)
    return Schema(key=key, arity=len(program.params), skeleton=skeleton,
                  holes=_holes_of(skeleton), origin="taught",
                  block=isinstance(skeleton, tuple))


def _is_structural(value: Any) -> bool:
    """Is this literal part of how the program is *built* rather than a number it happens to use?

    Empty containers only. ``0`` in ``total = 0`` is a real accumulator seed and varying it is how
    one loop shape covers summing and multiplying; ``[]`` in ``out = []`` is not a choice anybody
    made.
    """
    return value in ((), "", frozenset()) and isinstance(value, (tuple, str, frozenset))


def _skeleton_key(skeleton: Any) -> str:
    """A schema's identity. Two shapes that print alike are one shape as far as learning goes."""
    if isinstance(skeleton, tuple):
        return " ; ".join(line.strip() for line in render_block(skeleton, 0))
    return render(skeleton)


# --------------------------------------------------------------------------- #
# lessons, checks, results
# --------------------------------------------------------------------------- #

@dataclass
class Demonstration:
    """A teacher showing a worked solution: the task, and a program that is claimed to solve it."""

    spec: Optional[Spec] = None
    program: Optional[Program] = None
    source: str = "teacher"


@dataclass
class Check:
    """What happened when a program met a set of examples. Errors are counted apart from
    wrong answers, because "it crashed" and "it returned 7" are different bugs."""

    passed: int = 0
    failed: int = 0
    errored: int = 0
    first_failure: Optional[Example] = None
    got: Any = None
    error: str = ""

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errored

    @property
    def ok(self) -> bool:
        """Every example, or there were none — an empty check is not a pass and says so."""
        return self.total > 0 and self.passed == self.total

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "failed": self.failed, "errored": self.errored,
                "rate": round(self.rate, 4), "error": self.error}


@dataclass
class Learned:
    """What one demonstration changed. ``verdict`` is :class:`~nyxara.njp.teacher.Verdict`."""

    verdict: str = "undecided"
    key: str = ""
    before: float = 0.0
    after: float = 0.0
    check: Optional[Check] = None
    fresh: bool = False
    why: str = ""

    @property
    def moved(self) -> float:
        return self.after - self.before

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "key": self.key, "moved": round(self.moved, 4),
                "fresh": self.fresh, "why": self.why,
                "check": self.check.to_dict() if self.check else None}


@dataclass
class Written:
    """A program she wrote, with the search that produced it kept beside it.

    ``attempts`` is the honest cost and is reported everywhere, because the claim this module
    supports is about *search*: the same task solved in 12 attempts instead of 9000 is the whole
    of what a learned shape buys, and hiding the number would hide the claim.
    """

    program: Optional[Program] = None
    schema: str = ""
    attempts: int = 0
    seconds: float = 0.0
    shown: Optional[Check] = None
    grafted: bool = False
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.program is not None

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "source": self.program.source() if self.program else "",
                "schema": self.schema, "attempts": self.attempts,
                "seconds": round(self.seconds, 4), "grafted": self.grafted, "note": self.note}


# --------------------------------------------------------------------------- #
# the shapes she starts with, before anybody teaches her anything
# --------------------------------------------------------------------------- #

#: Unary operators worth a seed shape of their own. ``sum``, ``len`` and friends are the whole of
#: what a first program looks like, and each costs exactly one attempt to rule out.
_UNARY_SEEDS = ("len", "sum", "maxof", "minof", "rev", "sort", "head", "last", "tail",
                "uniq", "abs", "neg", "not", "upper", "lower", "words", "range")

#: Binary operators, seeded twice: once against a constant (``xs[k]``) and once against a second
#: parameter (``a + b``), because those are different tasks and one shape cannot serve both.
_BINARY_SEEDS = ("add", "sub", "mul", "div", "mod", "min2", "max2", "at", "take", "drop",
                 "count", "has", "cat", "eq", "ne", "lt", "le", "gt", "ge", "join")


def _seed_shapes() -> Tuple[Tuple[int, Any], ...]:
    """The primitive shapes: one operator deep, plus the three iteration forms.

    Nothing composite is here, and that is the design rather than an oversight. A composite shape
    is reachable from these only by grafting (:meth:`Coder._grafts`), which is combinatorial —
    so *the difference a lesson makes is measurable in attempts*, and :mod:`nyxara.njp.school`
    measures it. Seeding the composites would make her look better on the exam and would make the
    exam meaningless.
    """
    shapes: List[Tuple[int, Any]] = [(1, Var("#0"))]
    for op in _UNARY_SEEDS:
        shapes.append((1, Call(op, (Var("#0"),))))
    for op in _BINARY_SEEDS:
        shapes.append((1, Call(op, (Var("#0"), Hole(0, "any")))))
        shapes.append((2, Call(op, (Var("#0"), Var("#1")))))
        shapes.append((2, Call(op, (Var("#1"), Var("#0")))))
    shapes.append((1, Call("map", (Hole(0, "fn1"), Var("#0")))))
    shapes.append((1, Call("filter", (Hole(0, "fn1"), Var("#0")))))
    shapes.append((1, Call("fold", (Hole(0, "fn2"), Hole(1, "any"), Var("#0")))))
    shapes.append((1, Call("if", (Hole(0, "test"), Hole(1, "any"), Hole(2, "any")))))
    shapes.extend(_procedural_seeds())
    return tuple(shapes)


def _procedural_seeds() -> List[Tuple[int, Any]]:
    """The shapes a first course spends its whole second half on: accumulate, build, count, recur.

    Without these she could *read* a loop and never write one, which is not knowing how to loop.
    Each is the skeleton with its operator blanked — ``apply2(?f, total, x)`` rather than
    ``total + x`` — so one seed covers summing, multiplying, taking a maximum and concatenating,
    and the search picks between them by running the examples.
    """
    xs, out, total, n, x = Var("#0"), Var("out"), Var("total"), Var("k"), Var("x")

    def h(index: int, kind: str) -> Hole:
        return Hole(index, kind)

    return [
        # total = ?; for x in xs: total = f(total, x); return total
        (1, (Assign("total", h(0, "any")),
             For(("x",), xs, (Assign("total", Call("apply2", (h(1, "fn2"), total, x))),)),
             Return(total))),
        # …with each item transformed on the way in
        (1, (Assign("total", h(0, "any")),
             For(("x",), xs, (Assign("total", Call("apply2",
                 (h(1, "fn2"), total, Call("apply1", (h(2, "fn1"), x))))),)),
             Return(total))),
        # …only the items that pass a test
        (1, (Assign("total", h(0, "any")),
             For(("x",), xs, (If(Call("apply1", (h(1, "fn1"), x)),
                                 (Assign("total", Call("apply2", (h(2, "fn2"), total, x))),), ()),)),
             Return(total))),
        # out = []; for x in xs: if p(x): out.append(g(x)); return out
        (1, (Assign("out", Lit(())),
             For(("x",), xs, (If(Call("apply1", (h(0, "fn1"), x)),
                                 (Assign("out", Call("append",
                                     (out, Call("apply1", (h(1, "fn1"), x))))),), ()),)),
             Return(out))),
        # count the ones that pass
        (1, (Assign("total", Lit(0)),
             For(("x",), xs, (If(Call("apply1", (h(0, "fn1"), x)),
                                 (Assign("total", Call("add", (total, Lit(1)))),), ()),)),
             Return(total))),
        # recursion down an integer: if n <= k: return b; return f(n, recurse(n - 1))
        (1, (If(Call("le", (xs, h(0, "int"))), (Return(h(1, "any")),), ()),
             Return(Call("apply2", (h(2, "fn2"), xs,
                                    Call("invoke", (Lit("#self"), Call("sub", (xs, Lit(1)))))))))),
        # recursion down a list: if not xs: return b; return f(xs[0], recurse(xs[1:]))
        (1, (If(Call("eq", (Call("len", (xs,)), Lit(0))), (Return(h(0, "any")),), ()),
             Return(Call("apply2", (h(1, "fn2"), Call("head", (xs,)),
                                    Call("invoke", (Lit("#self"), Call("tail", (xs,))))))))),
        # a while loop counting down
        (1, (Assign("total", h(0, "any")), Assign("k", xs),
             While(Call("gt", (n, Lit(0))),
                   (Assign("total", Call("apply2", (h(1, "fn2"), total, n))),
                    Assign("k", Call("sub", (n, Lit(1)))))),
             Return(total))),
    ]


SEED_SHAPES: Tuple[Tuple[int, Any], ...] = _seed_shapes()


def _lambda1(body_of: Callable[[Term], Term]) -> Lambda:
    return Lambda(("x",), body_of(Var("x")))


def _element_kinds(spec: "Spec") -> set:
    """What the task's arguments are made of, one level in.

    One level is the right depth. The arguments themselves say whether this is a list task or a
    text task; their *elements* say whether the inner function will be working on numbers, on
    strings or on rows. Going deeper would start admitting the vocabulary of a nesting the task
    only has by accident.
    """
    found: set = set()
    for example in spec.shown[:3]:
        for arg in example.args:
            kind = _kind_of(arg)
            found.add(kind)
            if kind == "dict":
                # A mapping is iterated as pairs — `d.items()` is how half of what anyone writes
                # about one begins — so its element kind is "pair", and without that the pool has
                # no way to express `lambda kv: kv[1] > 2` and every dict comprehension is out of
                # reach however long the search runs.
                found.add("pair")
                for key, item in list(arg.items())[:6]:
                    found.add(_kind_of(key))
                    found.add(_kind_of(item))
            elif kind in ("list", "str"):
                for item in _seq(arg)[:6]:
                    found.add(_kind_of(item))
                    if isinstance(item, tuple) and len(item) == 2:
                        found.add("pair")
        found.add(_kind_of(example.out))
    return found or {"int", "str", "list", "bool"}


def _pick_constants(ints: Sequence[int], count: int = 10) -> List[int]:
    """Choose the constants worth building functions out of.

    Taking the *smallest* few — the obvious implementation, and the one this had — quietly makes
    a whole class of task unsolvable: a predicate like ``x > 10`` over data in 0..19 needs the
    ``10``, and the three smallest distinct values are ``0, 1, 2``. So the sorted distinct values
    are sampled **evenly across their range**, which keeps both ends and the middle, and the small
    integers every program uses are unioned in regardless of whether the task mentioned them.
    """
    distinct = sorted(dict.fromkeys(int(i) for i in ints))
    if len(distinct) > count:
        step = (len(distinct) - 1) / (count - 1)
        distinct = [distinct[round(i * step)] for i in range(count)]
    picked: List[int] = []
    for value in list(distinct) + [1, 2, 3, 4, 5]:
        if value not in picked:
            picked.append(value)
    return picked[:count + 2]


def _fn1_library(ints: Sequence[int], texts: Sequence[str] = (),
                 kinds: Optional[set] = None) -> Tuple[Lambda, ...]:
    """Unary functions worth trying inside a ``map`` or a ``filter``.

    The fixed part is the arithmetic and predicate vocabulary a first course covers; the
    parameterised part is built over :func:`_pick_constants`, so the pool stays a few dozen wide
    instead of unbounded and still reaches the constant the task actually needs.

    ``kinds`` is what the task's data is actually *made of*, and it prunes rather than decorates.
    A predicate over strings is not a candidate for a list of integers — trying it is guaranteed
    waste, and guaranteed waste multiplies: a two-function shape enumerates the pool squared, so
    twenty irrelevant entries cost four hundred attempts on every schema of that shape. Passing
    ``None`` keeps everything, which is what a caller with no data to look at should get.
    """
    x = Var("x")
    want = kinds if kinds is not None else {"int", "str", "list", "bool"}
    out: List[Lambda] = [Lambda(("x",), x)]
    # One operator applied to the item. `maxof`/`sort`/`head` are here because a row-wise reduce
    # over a list of lists is ordinary work and was unreachable without them: `max(max(row) …)`
    # needs `lambda row: max(row)` in the pool or it is not a candidate at all.
    groups: Dict[str, Tuple[str, ...]] = {
        "int": ("neg", "abs", "tostr"),
        "str": ("upper", "lower", "title", "strip", "words", "isdigit", "isalpha", "len",
                "rev", "head", "last", "tail", "toint", "sort"),
        "list": ("len", "sum", "maxof", "minof", "sort", "rev", "uniq", "head", "last", "tail"),
    }
    for kind, ops in groups.items():
        if kind in want:
            out.extend(Lambda(("x",), Call(op, (x,))) for op in ops)
    if "pair" in want:
        # Projections and tests over a two-element item: what `for k, v in …` compiles to here.
        pair = Var("p")
        left, right = Call("at", (pair, Lit(0))), Call("at", (pair, Lit(1)))
        out.extend([
            Lambda(("p",), left),
            Lambda(("p",), right),
            Lambda(("p",), Call("pair", (left, right))),
            Lambda(("p",), Call("pair", (right, left))),
            Lambda(("p",), Call("plus", (left, right))),
            Lambda(("p",), Call("mul", (left, right))),
        ])
        for k in _pick_constants(ints, 12):
            out.extend([
                Lambda(("p",), Call("gt", (right, Lit(k)))),
                Lambda(("p",), Call("lt", (right, Lit(k)))),
                Lambda(("p",), Call("eq", (right, Lit(k)))),
            ])
    if "int" in want:
        out.extend([
            Lambda(("x",), Call("mul", (x, x))),
            Lambda(("x",), Call("eq", (Call("mod", (x, Lit(2))), Lit(0)))),
            Lambda(("x",), Call("eq", (Call("mod", (x, Lit(2))), Lit(1)))),
            Lambda(("x",), Call("gt", (x, Lit(0)))),
            Lambda(("x",), Call("lt", (x, Lit(0)))),
            Lambda(("x",), Call("ge", (x, Lit(0)))),
        ])
    for k in (_pick_constants(ints, 24) if "int" in want or "list" in want else ()):
        out.extend([
            Lambda(("x",), Call("mul", (x, Lit(k)))),
            Lambda(("x",), Call("add", (x, Lit(k)))),
            Lambda(("x",), Call("gt", (x, Lit(k)))),
            Lambda(("x",), Call("lt", (x, Lit(k)))),
            Lambda(("x",), Call("sub", (x, Lit(k)))),
            Lambda(("x",), Call("eq", (x, Lit(k)))),
            Lambda(("x",), Call("gt", (Call("len", (x,)), Lit(k)))),
        ])
    for text in (list(dict.fromkeys(texts))[:4] if "str" in want else ()):
        # Text the task itself put on the table. Without these a predicate like
        # `ch not in "aeiou"` is not in the space at all, so every string task that needs one is
        # an abstention no matter how long the search runs.
        out.extend([
            Lambda(("x",), Call("eq", (x, Lit(text)))),
            Lambda(("x",), Call("has", (Lit(text), x))),
            Lambda(("x",), Call("not", (Call("has", (Lit(text), x)),))),
            Lambda(("x",), Call("starts", (x, Lit(text)))),
        ])
    return tuple(dict.fromkeys(out))


def _fn2_library() -> Tuple[Lambda, ...]:
    a, x = Var("a"), Var("x")
    return tuple(Lambda(("a", "x"), Call(op, (a, x)))
                 for op in ("plus", "mul", "max2", "min2", "sub", "cat", "add"))


# --------------------------------------------------------------------------- #
# sketches — imperative skeletons with holes that know their own scope
# --------------------------------------------------------------------------- #

def _v(name: str) -> Var:
    return Var(name)


def _c(op: str, *args: Any) -> Call:
    return Call(op, tuple(args))


def _num_ops(a: Term, b: Term) -> List[Term]:
    """The arithmetic a running variable is updated with."""
    return [_c("plus", a, b), _c("mul", a, b), _c("max2", a, b), _c("min2", a, b),
            _c("plus", a, Lit(1)), _c("minus", a, b), a, b]


def _tests(a: Term, b: Term) -> List[Term]:
    """The comparisons a loop body branches on."""
    return [_c("gt", a, b), _c("lt", a, b), _c("ge", a, b), _c("le", a, b),
            _c("eq", a, b), _c("ne", a, b)]


@dataclass(frozen=True)
class Sketch:
    """An imperative skeleton, its holes, and what may go in each of them.

    ``holes`` names each hole's *kind* — ``value``, ``test``, ``seed`` — and ``fill`` is asked for
    that hole's candidates given the task, so a hole inside a two-index loop is offered
    comparisons between the two indexed elements while a hole outside it is not. Attaching the
    scope to the hole rather than to the sketch is the whole reason the product stays countable.
    """

    id: str = ""
    arity: int = 1
    holes: Tuple[str, ...] = ()
    build: Any = None                 # (params, fillings) -> block
    fill: Any = None                  # (index, spec, ints) -> [Term]
    #: How many fillings this one skeleton may be given, when the shared cap is the wrong size
    #: for it. A relaxation has eight holes and half of them range over *which parameter is
    #: which*, so its product is legitimately larger than a two-hole fold's; capping them the
    #: same would rule it out for being general rather than for being wrong.
    cap: Optional[int] = None
    #: ``(params, fillings) -> (Program, ...)``. A skeleton whose answer is a **recursive helper**
    #: cannot be a block alone: the block is one line calling something that has to travel with
    #: it. Everything else leaves this ``None`` and builds a block as before.
    helpers: Any = None

    def fillings(self, index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
        return list(self.fill(index, spec, ints))


def _seeds(ints: Sequence[int]) -> List[Term]:
    """What an accumulator starts at."""
    return [Lit(0), Lit(1), Lit(-1), Lit(())] + [Lit(v) for v in _pick_constants(ints, 4)]


# --- one pass over the items, carrying an accumulator ---------------------- #

def _accumulate(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    xs = params[0]
    return (Assign("acc", f[0]),
            For(("x",), _v(xs), (Assign("acc", f[1]),)),
            Return(_v("acc")))


def _accumulate_if(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    xs = params[0]
    return (Assign("acc", f[0]),
            For(("x",), _v(xs), (If(f[1], (Assign("acc", f[2]),), ()),)),
            Return(_v("acc")))


# --- one pass by index, so the position is available ----------------------- #

def _by_index(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    xs = params[0]
    return (Assign("acc", f[0]),
            For(("i",), _c("range", _c("len", _v(xs))),
                (Assign("acc", f[1]),)),
            Return(_v("acc")))


# --- two nested indices, the shape of every quadratic scan ----------------- #

def _pairs(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    xs = params[0]
    inner = If(f[1], (Assign("acc", f[2]),), ())
    return (Assign("acc", f[0]),
            For(("i",), _c("range", _c("len", _v(xs))),
                (For(("j",), _c("range2", _c("plus", _v("i"), Lit(1)),
                                _c("len", _v(xs))), (inner,)),)),
            Return(_v("acc")))


def _windows(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """Every contiguous slice, and the best of them by some measure."""
    xs = params[0]
    part = _c("slice", _v(xs), _v("i"), _c("plus", _v("j"), Lit(1)))
    inner = (Assign("part", part), If(f[1], (Assign("best", f[2]),), ()))
    return (Assign("best", f[0]),
            For(("i",), _c("range", _c("len", _v(xs))),
                (For(("j",), _c("range2", _v("i"), _c("len", _v(xs))), inner),)),
            Return(_v("best")))


# --- two variables turning over each other -------------------------------- #

def _two_var_while(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    a, b = params[0], params[1]
    return (Assign("p", _v(a)), Assign("q", _v(b)),
            While(f[0], (Unpack(("p", "q"), _c("append", _c("append", Lit(()), f[1]), f[2])),)),
            Return(_v("p")))


def _countdown_while(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    n = params[0]
    return (Assign("acc", f[0]), Assign("k", _v(n)),
            While(_c("gt", _v("k"), Lit(0)),
                  (Assign("acc", f[1]), Assign("k", _c("minus", _v("k"), Lit(1))))),
            Return(_v("acc")))


# --- one-dimensional dynamic programming ---------------------------------- #

def _dp_row(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """``dp = [seed]; for i in 1..n: dp = dp + [f(dp, i)]; return dp[n]`` — the table that
    Catalan numbers, a Fibonacci by table and a coin count are all written on."""
    n = params[0]
    return (Assign("dp", _c("append", Lit(()), f[0])),
            For(("i",), _c("range2", Lit(1), _c("plus", _v(n), Lit(1))),
                (Assign("dp", _c("append", _v("dp"), f[1])),)),
            Return(_c("at", _v("dp"), _v(n))))


def _build_list(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    xs = params[0]
    return (Assign("out", Lit(())),
            For(("x",), _v(xs), (If(f[0], (Assign("out", _c("append", _v("out"), f[1])),), ()),)),
            Return(_v("out")))


def _running_best(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """A best-so-far and a current-so-far, which is the shape of every one-pass optimum."""
    xs = params[0]
    body = (Assign("cur", f[1]), Assign("best", f[2]))
    return (Assign("best", _c("head", _v(xs))), Assign("cur", f[0]),
            For(("x",), _c("tail", _v(xs)), body),
            Return(_v("best")))


def _fill_accumulate(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    if index == 0:
        return _seeds(ints)
    return _num_ops(_v("acc"), _v("x")) + [_c("plus", _v("acc"), _c("len", _v("x"))),
                                           _c("cat", _v("acc"), _v("x"))]


def _fill_accumulate_if(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    if index == 0:
        return _seeds(ints)
    if index == 1:
        out = [_c("eq", _c("mod", _v("x"), Lit(2)), Lit(0)),
               _c("eq", _c("mod", _v("x"), Lit(2)), Lit(1))]
        for k in _pick_constants(ints, 8):
            out += [_c("gt", _v("x"), Lit(k)), _c("lt", _v("x"), Lit(k)),
                    _c("eq", _v("x"), Lit(k))]
        return out
    return _num_ops(_v("acc"), _v("x"))


def _fill_by_index(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    xs = _v(spec.params[0])
    at_i = _c("at", xs, _v("i"))
    if index == 0:
        return _seeds(ints)
    return (_num_ops(_v("acc"), at_i) + _num_ops(_v("acc"), _v("i"))
            + [_c("plus", _v("acc"), _c("mul", at_i, _v("i")))])


def _fill_pairs(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    xs = _v(spec.params[0])
    a, b = _c("at", xs, _v("i")), _c("at", xs, _v("j"))
    if index == 0:
        return _seeds(ints)
    if index == 1:
        return _tests(a, b) + _tests(_c("plus", a, b), Lit(0))
    return [_c("plus", _v("acc"), Lit(1)), _c("plus", _v("acc"), a),
            _c("plus", _v("acc"), b), _c("max2", _v("acc"), _c("plus", a, b)),
            _c("max2", _v("acc"), _c("mul", a, b))]


def _fill_windows(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    part = _v("part")
    if index == 0:
        return [Lit(""), Lit(0), Lit(())]
    if index == 1:
        return [_c("and", _c("eq", part, _c("rev", part)),
                  _c("gt", _c("len", part), _c("len", _v("best")))),
                _c("gt", _c("len", part), _c("len", _v("best"))),
                _c("gt", _c("sum", part), _v("best")),
                _c("gt", _c("mul", _c("minof", part), _c("len", part)), _v("best"))]
    return [part, _c("len", part), _c("sum", part),
            _c("mul", _c("minof", part), _c("len", part))]


def _fill_two_var(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    p, q = _v("p"), _v("q")
    if index == 0:
        return [q, _c("gt", q, Lit(0)), _c("ne", q, Lit(0)), _c("gt", p, q)]
    if index == 1:
        return [q, _c("mod", p, q), _c("minus", p, q), _c("plus", p, q)]
    return [_c("mod", p, q), _c("minus", p, q), _c("minus", q, p), _c("plus", p, q), p]


def _fill_countdown(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    if index == 0:
        return _seeds(ints)
    return _num_ops(_v("acc"), _v("k"))


def _fill_dp_row(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    dp, i = _v("dp"), _v("i")
    prev = _c("at", dp, _c("minus", i, Lit(1)))
    if index == 0:
        return [Lit(0), Lit(1)]
    return [_c("plus", prev, i), _c("mul", prev, i), _c("plus", prev, Lit(1)),
            _c("mul", prev, Lit(2)), _c("plus", prev, _c("at", dp, _c("minus", i, Lit(2)))),
            _c("plus", _c("sum", dp), i), _c("sum", dp)]


def _fill_build_list(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    if index == 0:
        out: List[Term] = [Lit(True), _c("eq", _c("mod", _v("x"), Lit(2)), Lit(0)),
                           _c("eq", _c("mod", _v("x"), Lit(2)), Lit(1))]
        for k in _pick_constants(ints, 6):
            out += [_c("gt", _v("x"), Lit(k)), _c("lt", _v("x"), Lit(k))]
        return out
    out = [_v("x"), _c("mul", _v("x"), _v("x"))]
    for k in _pick_constants(ints, 6):
        out += [_c("mul", _v("x"), Lit(k)), _c("plus", _v("x"), Lit(k))]
    return out


def _fill_running_best(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    best, cur, x = _v("best"), _v("cur"), _v("x")
    if index == 0:
        return [_c("head", _v(spec.params[0])), Lit(0)]
    if index == 1:
        return [_c("max2", x, _c("plus", cur, x)), _c("plus", cur, x),
                _c("min2", x, _c("plus", cur, x)), x]
    return [_c("max2", best, cur), _c("min2", best, cur), best, cur]


# --------------------------------------------------------------------------- #
# the dynamic-programming skeletons
#
# Everything above carries a *scalar* across an iteration — a running sum, a best so far, a
# pair of variables turning over each other. That is one rung, and it is not the rung the
# classic hard problems sit on. Every one of edit distance, longest common subsequence,
# longest increasing subsequence, coin change and Catalan carries a **table**: a row of
# answers to smaller instances, read back by index while the next row is being built. No
# depth of expression search reaches it and no scalar accumulator expresses it, so it is its
# own family of skeletons rather than a wider filling of the ones above.
#
# The vocabulary offered for each hole is deliberately the vocabulary a first course uses in
# that position, not an enumeration to depth: a table lookup one step back, a comparison
# between the two indexed elements, a minimum of the three neighbours. That is a real
# restriction and it is why these reach some of these problems and not others.
# --------------------------------------------------------------------------- #

def _rename(term: Term, was: str, now: str) -> Term:
    """``term`` with every occurrence of the variable ``was`` renamed to ``now``."""
    if isinstance(term, Var):
        return Var(now) if term.name == was else term
    if isinstance(term, Call):
        return Call(term.op, tuple(_rename(a, was, now) for a in term.args))
    if isinstance(term, Lambda):
        if was in term.params:
            return term
        return Lambda(term.params, _rename(term.body, was, now))
    return term


def _dp_scan(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """``tab[i] = best over j < i``, then one aggregate over the table.

    The shape of longest-increasing-subsequence and of every other "answer at each position,
    computed from the answers before it" problem.
    """
    xs = params[0]
    inner = If(f[1], (Assign("cur", f[2]),), ())
    return (Assign("tab", Lit(())),
            For(("i",), _c("range", _c("len", _v(xs))),
                (Assign("cur", f[0]),
                 For(("j",), _c("range", _v("i")), (inner,)),
                 Assign("tab", _c("append", _v("tab"), _v("cur"))))),
            If(_c("eq", _c("len", _v("tab")), Lit(0)), (Return(Lit(0)),), ()),
            Return(f[3]))


def _dp_row_inner(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """``tab[i] = fold over j < i of the earlier entries``, answer at ``tab[n]``.

    Catalan numbers are the canonical member: each entry is a sum of products of two entries
    that both sit behind it in the same table.
    """
    n = params[0]
    return (Assign("tab", _c("append", Lit(()), f[0])),
            For(("i",), _c("range2", Lit(1), _c("plus", _v(n), Lit(1))),
                (Assign("cur", f[1]),
                 For(("j",), _c("range", _v("i")), (Assign("cur", f[2]),)),
                 Assign("tab", _c("append", _v("tab"), _v("cur"))))),
            Return(_c("at", _v("tab"), _v(n))))


def _dp_two_seq(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """The two-sequence table, kept one row at a time.

    ``prev`` is the finished row above, ``cur`` the row being built; a cell branches on whether
    the two characters under it agree. Edit distance and longest common subsequence are the
    same skeleton with different fillings, which is exactly the claim a sketch makes.
    """
    a, b = params[0], params[1]
    inner = If(f[2], (Assign("cur", _c("append", _v("cur"), f[3])),),
               (Assign("cur", _c("append", _v("cur"), f[4])),))
    return (Assign("prev", Lit(())),
            For(("j",), _c("range2", Lit(0), _c("plus", _c("len", _v(b)), Lit(1))),
                (Assign("prev", _c("append", _v("prev"), f[0])),)),
            For(("i",), _c("range2", Lit(1), _c("plus", _c("len", _v(a)), Lit(1))),
                (Assign("cur", _c("append", Lit(()), f[1])),
                 For(("j",), _c("range2", Lit(1), _c("plus", _c("len", _v(b)), Lit(1))),
                     (inner,)),
                 Assign("prev", _v("cur")))),
            Return(_c("at", _v("prev"), _c("len", _v(b)))))


def _dp_amounts(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """``tab[amount]`` built from ``tab[amount - item]`` over a bag of items.

    Coin change is the canonical member. The final hole exists because this family answers
    with a sentinel — "no way to make it" — as often as with the table entry itself.
    """
    items, limit = params[0], params[1]
    inner = If(f[2], (Assign("cur", f[3]),), ())
    return (Assign("tab", _c("append", Lit(()), f[0])),
            For(("i",), _c("range2", Lit(1), _c("plus", _v(limit), Lit(1))),
                (Assign("cur", f[1]),
                 For(("w",), _v(items), (inner,)),
                 Assign("tab", _c("append", _v("tab"), _v("cur"))))),
            Return(f[4]))


def _row_rebuild(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """A whole row rebuilt from the row above it, ``n`` times, and returned.

    Pascal's triangle: the answer is the row itself rather than one cell of it, which is a
    different ending from every other table skeleton here.
    """
    n = params[0]
    return (Assign("row", _c("append", Lit(()), f[0])),
            For(("i",), _c("range", _v(n)),
                (Assign("nxt", _c("append", Lit(()), f[1])),
                 For(("j",), _c("range", _c("minus", _c("len", _v("row")), Lit(1))),
                     (Assign("nxt", _c("append", _v("nxt"), f[2])),)),
                 Assign("nxt", _c("append", _v("nxt"), f[3])),
                 Assign("row", _v("nxt")))),
            Return(_v("row")))


def _prefix_suffix(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """For each position, a scan of everything to its left and everything to its right.

    Trapping rain water is written on this: what a column holds depends on the tallest wall
    on each side of it, and neither side is known from a single pass.
    """
    xs = params[0]
    left = For(("j",), _c("range2", Lit(0), _c("plus", _v("i"), Lit(1))),
               (Assign("left", f[1]),))
    # The two scans are the same update run over different ranges, so the right-hand one is the
    # left-hand one with its carried name renamed. Offering the hole twice would square the
    # search for a pair that is never usefully different.
    right = For(("j",), _c("range2", _v("i"), _c("len", _v(xs))),
                (Assign("right", _rename(f[1], "left", "right")),))
    return (Assign("acc", f[0]),
            For(("i",), _c("range", _c("len", _v(xs))),
                (Assign("left", Lit(0)), left,
                 Assign("right", Lit(0)), right,
                 Assign("acc", f[2]))),
            Return(_v("acc")))


def _pairwise(terms: Sequence[Term], ops: Sequence[str]) -> List[Term]:
    """Every ordered pair of ``terms`` under every operator in ``ops``."""
    out: List[Term] = []
    for a, b in itertools.permutations(terms, 2):
        for op in ops:
            out.append(_c(op, a, b))
    return out


_CMP = ("lt", "le", "gt", "ge", "eq")


def _fill_dp_scan(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    xs = _v(spec.params[0])
    xi, xj = _c("at", xs, _v("i")), _c("at", xs, _v("j"))
    tab, cur = _v("tab"), _v("cur")
    tj = _c("at", tab, _v("j"))
    tj1 = _c("plus", tj, Lit(1))
    if index == 0:
        return [Lit(1), Lit(0), xi]
    if index == 1:
        # A test between the two elements, a test between the two running numbers, or both.
        elems = _pairwise((xi, xj), _CMP)
        accs = _pairwise((cur, tj, tj1), _CMP)
        both = [_c("and", e, a) for e in elems for a in accs]
        return elems + accs + both
    if index == 2:
        return [tj1, tj, _c("plus", cur, Lit(1)), _c("max2", cur, tj1),
                xj, _c("plus", cur, xj), _c("max2", cur, tj)]
    return [_c("maxof", tab), _c("sum", tab), _c("minof", tab), _c("len", tab)]


def _fill_dp_row_inner(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    tab, cur, i, j = _v("tab"), _v("cur"), _v("i"), _v("j")
    tj = _c("at", tab, j)
    mirror = _c("at", tab, _c("minus", _c("minus", i, Lit(1)), j))
    prev = _c("at", tab, _c("minus", i, Lit(1)))
    if index == 0:
        return [Lit(1), Lit(0)]
    if index == 1:
        return [Lit(0), Lit(1)]
    return [_c("plus", cur, _c("mul", tj, mirror)), _c("plus", cur, tj),
            _c("plus", cur, Lit(1)), _c("plus", cur, mirror),
            _c("max2", cur, tj), _c("plus", cur, _c("mul", tj, i)),
            _c("mul", cur, _c("plus", j, Lit(1))), _c("plus", cur, prev)]


def _fill_dp_two_seq(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    a, b = _v(spec.params[0]), _v(spec.params[1])
    i, j = _v("i"), _v("j")
    ai = _c("at", a, _c("minus", i, Lit(1)))
    bj = _c("at", b, _c("minus", j, Lit(1)))
    p_j = _c("at", _v("prev"), j)
    p_j1 = _c("at", _v("prev"), _c("minus", j, Lit(1)))
    c_j1 = _c("at", _v("cur"), _c("minus", j, Lit(1)))
    if index == 0:
        return [Lit(0), j]
    if index == 1:
        return [Lit(0), i]
    if index == 2:
        return [_c("eq", ai, bj), _c("ne", ai, bj)]
    three_min = _c("min2", _c("plus", p_j, Lit(1)), _c("plus", c_j1, Lit(1)))
    return [_c("plus", p_j1, Lit(1)), p_j1,
            _c("max2", p_j, c_j1), _c("plus", _c("max2", p_j, c_j1), Lit(1)),
            _c("min2", p_j, c_j1), _c("plus", _c("min2", p_j, c_j1), Lit(1)),
            _c("min2", p_j1, three_min),
            _c("min2", _c("plus", p_j1, Lit(1)), three_min)]


def _fill_dp_amounts(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    tab, cur, i, w = _v("tab"), _v("cur"), _v("i"), _v("w")
    limit = _v(spec.params[1])
    back = _c("at", tab, _c("minus", i, w))
    back1 = _c("plus", back, Lit(1))
    big = _c("plus", limit, Lit(1))
    if index == 0:
        return [Lit(0), Lit(1), Lit(True), Lit(False)]
    if index == 1:
        return [big, Lit(0), Lit(False), Lit(True)]
    if index == 2:
        guard = _c("le", w, i)
        return [_c("and", guard, _c("lt", back1, cur)),
                _c("and", guard, _c("gt", back1, cur)),
                _c("and", guard, back),
                _c("and", guard, _c("not", back)),
                guard]
    if index == 3:
        return [back1, back, _c("plus", cur, back), Lit(True), Lit(False),
                _c("min2", cur, back1), _c("max2", cur, back1)]
    at_limit = _c("at", tab, limit)
    return [at_limit,
            _c("if", _c("eq", at_limit, _c("plus", limit, Lit(1))), Lit(-1), at_limit),
            _c("sum", tab), _c("maxof", tab)]


def _fill_row_rebuild(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    row, j = _v("row"), _v("j")
    rj = _c("at", row, j)
    rj1 = _c("at", row, _c("plus", j, Lit(1)))
    if index in (0, 1, 3):
        return [Lit(1), Lit(0)]
    return [_c("plus", rj, rj1), _c("mul", rj, rj1), _c("max2", rj, rj1),
            _c("minus", rj1, rj), _c("plus", _c("plus", rj, rj1), Lit(1))]


def _fill_prefix_suffix(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    xs = _v(spec.params[0])
    xi, xj = _c("at", xs, _v("i")), _c("at", xs, _v("j"))
    acc, left, right = _v("acc"), _v("left"), _v("right")
    side = _v("left")  # the scan body is shared, so it is written over whichever side is live
    if index == 0:
        return [Lit(0)]
    if index == 1:
        # The same update runs on both sides; ``left`` is the name bound by the block builder,
        # so the two scans differ only in what they range over.
        return [_c("max2", side, xj), _c("min2", side, xj), _c("plus", side, xj)]
    return [_c("plus", acc, _c("minus", _c("min2", left, right), xi)),
            _c("plus", acc, _c("min2", left, right)),
            _c("max2", acc, _c("minus", _c("min2", left, right), xi)),
            _c("plus", acc, _c("minus", _c("max2", left, right), xi))]


# --------------------------------------------------------------------------- #
# recurrences, searches and closures
# --------------------------------------------------------------------------- #

def _conjunctions(atoms: Sequence[Term], *, upto: int = 3) -> List[Term]:
    """Every conjunction of up to ``upto`` of ``atoms``, and each atom on its own.

    A guarded table step — "the piece fits **and** the table says the rest is reachable **and**
    the piece is actually there" — is three atoms deep, and writing that conjunction out as one
    candidate would be writing the answer down. Offering the *atoms* and letting every subset of
    them be tried is the honest version of the same reach: the sketch supplies a vocabulary of
    conditions, not a condition.
    """
    out: List[Term] = []
    for size in range(1, min(upto, len(atoms)) + 1):
        for combo in itertools.combinations(atoms, size):
            term = combo[0]
            for extra in combo[1:]:
                term = _c("and", term, extra)
            out.append(term)
    return out


def _recur_two(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """``g(1) = seed; g(i) = step(g(i-1), i, k)`` for a recurrence in two parameters.

    The Josephus recurrence is the classic: where the survivor stands in a circle of ``i`` people
    is where they stood in a circle of ``i - 1``, shifted by ``k`` and wrapped. Nothing about the
    skeleton is specific to it — it is the general "one number carried up from the base case,
    with a second parameter along for the ride".
    """
    n = params[0]
    return (Assign("tab", _c("append", Lit(()), f[0])),
            For(("i",), _c("range2", Lit(1), _v(n)),
                (Assign("tab", _c("append", _v("tab"), f[1])),)),
            Return(f[2]))


def _fill_recur_two(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    tab, i = _v("tab"), _v("i")
    k, n = _v(spec.params[1]), _v(spec.params[0])
    back = _c("at", tab, _c("minus", i, Lit(1)))
    if index == 0:
        return [Lit(0), Lit(1)]
    if index == 1:
        return [_c("mod", _c("plus", back, k), _c("plus", i, Lit(1))),
                _c("mod", _c("plus", back, k), i),
                _c("mod", _c("plus", back, _c("minus", k, Lit(1))), _c("plus", i, Lit(1))),
                _c("plus", back, k), _c("mul", back, k), _c("plus", back, i),
                _c("mul", back, i), _c("plus", back, Lit(1))]
    last = _c("last", tab)
    return [last, _c("plus", last, Lit(1)), _c("at", tab, _c("minus", n, Lit(1))), tab]


def _scan_find(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """The first position that satisfies something, or a miss value.

    Every "where does this occur" answer is written on it, including substring search: the fast
    algorithms differ from this one in cost, not in what they return, and cost is not what an
    example set can distinguish.
    """
    a = params[0]
    return (For(("i",), _c("range", _c("len", _v(a))),
                (If(f[0], (Return(f[1]),), ()),)),
            Return(f[2]))


def _fill_scan_find(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    a = _v(spec.params[0])
    b = _v(spec.params[1]) if spec.arity > 1 else None
    i = _v("i")
    at_i = _c("at", a, i)
    if index == 0:
        out = [_c("eq", at_i, Lit(0)), _c("gt", at_i, Lit(0)), _c("lt", at_i, Lit(0))]
        if b is not None:
            piece = _c("slice", a, i, _c("plus", i, _c("len", b)))
            out = [_c("eq", piece, b), _c("eq", at_i, b), _c("inset", b, at_i),
                   _c("ne", at_i, b)] + out
        return out
    if index == 1:
        return [i, _c("plus", i, Lit(1)), at_i]
    return [Lit(-1), Lit(0), Lit(None), _c("len", a)]


def _merge_pick(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """Put two sequences together, sort, and read one place off the result.

    The median of two sorted lists is this and nothing more, once the linear-time trick is set
    aside; what has to be found is *which* place, and that the even case reads two.
    """
    a, b = params[0], params[1]
    return (Assign("m", _c("sort", _c("cat", _c("tolist", _v(a)), _c("tolist", _v(b))))),
            Assign("n", _c("len", _v("m"))),
            If(f[0], (Return(f[1]),), ()),
            Return(f[2]))


def _fill_merge_pick(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    m, n = _v("m"), _v("n")
    half = _c("div", n, Lit(2))
    below = _c("at", m, _c("minus", half, Lit(1)))
    at_half = _c("at", m, half)
    if index == 0:
        return [_c("eq", _c("mod", n, Lit(2)), Lit(1)), _c("eq", _c("mod", n, Lit(2)), Lit(0))]
    return [at_half, below, _c("div", _c("plus", below, at_half), Lit(2)),
            _c("head", m), _c("last", m), _c("div", _c("sum", m), n)]


def _set_closure(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """Everything reachable by using each item at most once, then a question about that set.

    Subset sum is exactly this and so is every other "can these pieces make that", which is why
    the skeleton asks a *question* of the closure at the end rather than returning it.
    """
    xs, = params[0],
    grow = For(("r",), _v("acc"), (Assign("add", _c("append", _v("add"), f[1])),))
    return (Assign("acc", _c("append", Lit(()), f[0])),
            For(("x",), _v(xs),
                (Assign("add", Lit(())), grow,
                 Assign("acc", _c("uniq", _c("cat", _v("acc"), _v("add")))))),
            Return(f[2]))


def _fill_set_closure(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    acc, r, x = _v("acc"), _v("r"), _v("x")
    if index == 0:
        return [Lit(0), Lit(1), Lit("")]
    if index == 1:
        return [_c("plus", r, x), _c("mul", r, x), _c("max2", r, x), _c("cat", r, x)]
    target = _v(spec.params[1]) if spec.arity > 1 else Lit(0)
    return [_c("inset", acc, target), _c("not", _c("inset", acc, target)),
            _c("len", acc), _c("maxof", acc), _c("sum", acc)]


def _dp_prefix(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """``tab[i]`` is the answer for the first ``i`` characters, built from a bag of pieces.

    Word break is the member of this family everyone meets first: whether a prefix can be spelled
    depends on whether some shorter prefix can be, plus one piece.
    """
    text, items = params[0], params[1]
    inner = If(f[2], (Assign("cur", f[3]),), ())
    return (Assign("tab", _c("append", Lit(()), f[0])),
            For(("i",), _c("range2", Lit(1), _c("plus", _c("len", _v(text)), Lit(1))),
                (Assign("cur", f[1]),
                 For(("w",), _v(items), (inner,)),
                 Assign("tab", _c("append", _v("tab"), _v("cur"))))),
            Return(_c("at", _v("tab"), _c("len", _v(text)))))


def _fill_dp_prefix(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    text = _v(spec.params[0])
    tab, cur, i, w = _v("tab"), _v("cur"), _v("i"), _v("w")
    back = _c("minus", i, _c("len", w))
    if index == 0:
        return [Lit(True), Lit(0), Lit(1)]
    if index == 1:
        return [Lit(False), Lit(0), Lit(-1)]
    if index == 2:
        atoms = [_c("le", _c("len", w), i),
                 _c("at", tab, back),
                 _c("eq", _c("slice", text, back, i), w),
                 _c("gt", _c("len", w), Lit(0))]
        return _conjunctions(atoms)
    return [Lit(True), _c("plus", cur, Lit(1)),
            _c("max2", cur, _c("plus", _c("at", tab, back), Lit(1))),
            _c("plus", _c("at", tab, back), Lit(1))]


def _dp_grid_items(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """A whole row of amounts rebuilt once per item — the shape of 0/1 knapsack.

    The row has to be rebuilt rather than updated in place, because an item used once must not
    be seen again by the same row; that is the entire difference between this and
    :func:`_dp_amounts`, and it is why they are separate skeletons.
    """
    cap = params[2]
    span = _c("range2", Lit(0), _c("plus", _v(cap), Lit(1)))
    inner = (Assign("cur", f[1]), If(f[2], (Assign("cur", f[3]),), ()),
             Assign("nxt", _c("append", _v("nxt"), f[4])))
    return (Assign("tab", Lit(())),
            For(("c",), span, (Assign("tab", _c("append", _v("tab"), f[0])),)),
            For(("i",), _c("range", _c("len", _v(params[0]))),
                (Assign("nxt", Lit(())), For(("c",), span, inner),
                 Assign("tab", _v("nxt")))),
            Return(_c("at", _v("tab"), _v(cap))))


def _fill_dp_grid_items(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    weights, values = _v(spec.params[0]), _v(spec.params[1])
    tab, cur, i, c = _v("tab"), _v("cur"), _v("i"), _v("c")
    wi, vi = _c("at", weights, i), _c("at", values, i)
    room = _c("at", tab, _c("minus", c, wi))
    if index == 0:
        return [Lit(0), Lit(False)]
    if index == 1:
        return [Lit(0), Lit(False)]
    if index == 2:
        return [_c("le", wi, c), _c("lt", wi, c), _c("ge", c, wi)]
    if index == 3:
        return [_c("plus", vi, room), _c("plus", wi, room), room, _c("plus", room, Lit(1))]
    return [_c("max2", _c("at", tab, c), cur), _c("min2", _c("at", tab, c), cur),
            _c("plus", _c("at", tab, c), cur), cur]


def _coalesce(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """Walk a sorted sequence, either **extending the last thing produced** or starting a new one.

    Merging overlapping intervals is the member everyone meets, but the pattern is the general
    one for collapsing runs: whether the item joins what is already there is a question about the
    output built so far, which is why this cannot be a filter or a map.
    """
    xs = params[0]
    joins = _c("and", _c("gt", _c("len", _v("out")), Lit(0)), f[0])
    return (Assign("out", Lit(())),
            For(("it",), _c("sort", _v(xs)),
                (If(joins,
                    (Assign("out", _c("setat", _v("out"),
                                      _c("minus", _c("len", _v("out")), Lit(1)), f[1])),),
                    (Assign("out", _c("append", _v("out"), f[2])),)),)),
            Return(_v("out")))


def _fill_coalesce(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    it, out = _v("it"), _v("out")
    last = _c("at", out, _c("minus", _c("len", out), Lit(1)))
    lo, hi = _c("at", it, Lit(0)), _c("at", it, Lit(1))
    lo_last, hi_last = _c("at", last, Lit(0)), _c("at", last, Lit(1))
    if index == 0:
        return [_c("le", lo, hi_last), _c("lt", lo, hi_last), _c("eq", lo, hi_last),
                _c("le", lo, _c("plus", hi_last, Lit(1)))]
    if index == 1:
        return [_c("pair", lo_last, _c("max2", hi_last, hi)),
                _c("pair", lo_last, hi), _c("pair", _c("min2", lo_last, lo), hi),
                _c("pair", lo_last, _c("plus", hi_last, hi))]
    return [_c("pair", lo, hi), it]


# --------------------------------------------------------------------------- #
# the control structures a table still cannot express
#
# A table carried across a loop reaches dynamic programming and stops there. Three shapes sit
# outside it and each rules out a whole class of algorithm rather than one problem:
#
# * a **relaxation fixpoint** — a state that is improved by every fact, over and over, until
#   improving stops. Shortest paths and connected components are the same loop with different
#   improvements, which is exactly what makes it one skeleton and not two;
# * a **window judged by a scan of its own** — the test on a slice is itself a loop, so it
#   cannot be an expression in a hole no matter how deep the hole goes;
# * a **monotonic stack** — a structure that is *popped* before it is pushed to, which is the
#   only shape here that shrinks something it built.
# --------------------------------------------------------------------------- #

def _fixpoint(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """A state improved by every fact, ``size`` times over: relaxation to a fixed point.

    Bellman and Ford's shortest paths and label propagation over a graph's edges are the same
    loop — start every node at something, sweep the facts improving what they touch, and repeat
    enough times that nothing can improve again. Repeating ``size`` times rather than "until it
    stops changing" is the standard bound and it makes the loop deterministic.
    """
    facts, size = f[5], f[6]
    return (Assign("st", Lit(())),
            For(("i",), _c("range", size), (Assign("st", _c("append", _v("st"), f[0])),)),
            Assign("st", f[1]),
            For(("_r",), _c("range", size),
                (For(("e",), facts, (If(f[2], (Assign("st", _c("setat", _v("st"), f[3], f[4])),),
                                        ()),)),)),
            Return(f[7]))


def _fill_fixpoint(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    """The vocabulary of a relaxation, offered over whichever parameters have the right shapes.

    ``size`` is an integer parameter or the length of a sequence one; ``facts`` is a sequence
    parameter. Both are offered as *candidates* rather than picked, because which parameter is
    the node count and which is the edge list is part of what the search has to work out.
    """
    st, e, i = _v("st"), _v("e"), _v("i")
    names = [_v(name) for name in spec.params]
    src, dst = _c("at", e, Lit(0)), _c("at", e, Lit(1))
    weight = _c("at", e, Lit(2))
    if index == 0:                                  # what every slot starts at
        return [Lit(1000000), i, Lit(0), Lit(-1)]
    if index == 1:                                  # one optional seeding of a start slot
        return [_v("st")] + [_c("setat", _v("st"), name, Lit(0)) for name in names]
    if index == 2:                                  # is this fact an improvement?
        return [_c("lt", _c("plus", _c("at", st, src), weight), _c("at", st, dst)),
                _c("lt", _c("at", st, src), _c("at", st, dst)),
                _c("gt", _c("at", st, src), _c("at", st, dst)),
                _c("ne", _c("at", st, src), _c("at", st, dst)),
                Lit(True)]
    if index == 3:                                  # which slot it improves
        return [dst, src]
    if index == 4:                                  # what it becomes
        return [_c("plus", _c("at", st, src), weight),
                _c("min2", _c("at", st, src), _c("at", st, dst)),
                _c("at", st, src), _c("min2", _c("at", st, dst), src)]
    if index == 5:                                  # the facts swept over
        return names
    if index == 6:                                  # how many slots, and how many sweeps
        return names + [_c("len", name) for name in names]
    return [st, _c("len", _c("uniq", st)), _c("sum", st), _c("maxof", st), _c("minof", st)]


def _windows_scan(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """The best slice, where whether a slice counts needs **a loop of its own** to decide.

    :func:`_windows` can only ask an expression of a slice — is it a palindrome, how long is it.
    Whether a bracket string is balanced is not an expression: it is a running depth that must
    never go negative and must end at nought, and that is a scan. Nesting the judgement inside
    the window is the whole point of this skeleton.

    **What gets scanned is itself a hole.** Balanced brackets scan the slice; a smallest window
    covering another string scans *that other string*, asking of each of its characters whether
    the slice holds enough of them. Same skeleton, different sequence in the inner loop, and
    fixing it to the slice would have made the second problem a different skeleton for no reason.
    """
    xs = params[0]
    part = _c("slice", _v(xs), _v("i"), _c("plus", _v("j"), Lit(1)))
    scan = For(("ch",), f[2],
               (Assign("d", f[3]), If(f[4], (Assign("ok", Lit(False)),), ())))
    body = (Assign("part", part), Assign("d", f[1]), Assign("ok", Lit(True)), scan,
            If(_c("and", _v("ok"), f[5]), (Assign("best", f[6]),), ()))
    return (Assign("best", f[0]),
            For(("i",), _c("range", _c("len", _v(xs))),
                (For(("j",), _c("range2", _v("i"), _c("len", _v(xs))), body),)),
            Return(f[7]))


def _fill_windows_scan(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    d, ch, part, best = _v("d"), _v("ch"), _v("part"), _v("best")
    xs = _v(spec.params[0])
    other = _v(spec.params[1]) if spec.arity > 1 else None
    over = _c("plus", _c("len", xs), Lit(1))
    texts = spec.constants()[1]
    marks = sorted({piece[0] for piece in texts if piece})[:3] or ["("]
    if index == 0:
        return [Lit(0), over, Lit("")]
    if index == 1:
        return [Lit(0)]
    if index == 2:
        return [part] + ([other] if other is not None else [])
    if index == 3:
        return [_c("plus", d, _c("if", _c("eq", ch, Lit(mark)), Lit(1), Lit(-1)))
                for mark in marks] + [_c("plus", d, Lit(1))]
    if index == 4:
        out = [_c("lt", d, Lit(0)), Lit(False)]
        if other is not None:
            out.append(_c("lt", _c("count", part, ch), _c("count", other, ch)))
        return out
    if index == 5:
        longer = _c("gt", _c("len", part), best)
        shorter = _c("lt", _c("len", part), best)
        return [_c("and", _c("eq", d, Lit(0)), longer), longer, shorter,
                _c("and", _c("eq", d, Lit(0)), _c("gt", _c("len", part), _c("len", best)))]
    if index == 6:
        return [_c("len", part), part, d]
    return [best, _c("if", _c("eq", best, over), Lit(0), best)]


def _stack_scan(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """Push each item, **popping first** while the top of the stack fails a test.

    The monotonic stack, and the only skeleton here that takes back something it already built.
    A convex hull is its geometric member: a point that turns the wrong way undoes the point
    before it, and possibly the one before that.
    """
    xs = params[0]
    pop = While(_c("and", _c("ge", _c("len", _v("st")), f[0]), f[1]),
                (Assign("st", _c("take", _v("st"), _c("minus", _c("len", _v("st")), Lit(1)))),))
    return (Assign("st", Lit(())),
            For(("p",), _c("sort", _v(xs)), (pop, Assign("st", _c("append", _v("st"), _v("p"))))),
            Return(f[2]))


def _fill_stack_scan(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    st, p = _v("st"), _v("p")
    n = _c("len", st)
    top = _c("at", st, _c("minus", n, Lit(1)))
    under = _c("at", st, _c("minus", n, Lit(2)))
    if index == 0:
        return [Lit(2), Lit(1)]
    if index == 1:
        # The turn a triple of points makes, and the plain comparisons a numeric stack pops on.
        cross = _c("minus",
                   _c("mul", _c("minus", _c("at", top, Lit(0)), _c("at", under, Lit(0))),
                      _c("minus", _c("at", p, Lit(1)), _c("at", under, Lit(1)))),
                   _c("mul", _c("minus", _c("at", top, Lit(1)), _c("at", under, Lit(1))),
                      _c("minus", _c("at", p, Lit(0)), _c("at", under, Lit(0)))))
        return [_c("le", cross, Lit(0)), _c("ge", cross, Lit(0)),
                _c("lt", cross, Lit(0)), _c("gt", cross, Lit(0)),
                _c("ge", top, p), _c("le", top, p)]
    return [_c("len", st), st, _c("maxof", st)]


# --------------------------------------------------------------------------- #
# search, peeling, and both axes of a grid
# --------------------------------------------------------------------------- #

def _disjunctions(atoms: Sequence[Term], *, upto: int = 3) -> List[Term]:
    """Every disjunction of up to ``upto`` of ``atoms``, and each atom on its own.

    The mirror of :func:`_conjunctions`, and it is what a *disqualifying* test wants: a partial
    solution is rejected because of this reason **or** that one **or** the third. Offering the
    reasons and letting the search pick which of them matter is the difference between supplying
    a vocabulary and supplying the answer.
    """
    out: List[Term] = []
    for size in range(1, min(upto, len(atoms)) + 1):
        for combo in itertools.combinations(atoms, size):
            term = combo[0]
            for extra in combo[1:]:
                term = _c("or", term, extra)
            out.append(term)
    return out


def _recursive_search(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """``f(n)`` is one line: hand an empty partial solution to a helper that searches."""
    return (Return(_c("invoke", Lit("go"), Lit(()), _v(params[0]))),)


def _helpers_recursive_search(params: Sequence[str], f: Sequence[Term]) -> Tuple[Program, ...]:
    """The backtracker: extend a partial solution every legal way, and sum what comes back.

    This is the one skeleton here whose state goes **down** the call stack rather than across a
    loop, and it is why counting arrangements was out of reach at any table size: a table
    remembers what smaller instances answered, and a search remembers what it has already chosen.
    Whether a choice is legal is itself a scan over the choices already made — the inner loop —
    so the disqualifying test is offered as *reasons* and every disjunction of them is tried.
    """
    reject = If(f[2], (Assign("ok", Lit(False)),), ())
    inner = For(("r",), _c("range", _c("len", _v("chosen"))), (reject,))
    deeper = _c("invoke", Lit("go"), _c("append", _v("chosen"), _v("k")), _v("n"))
    step = If(_v("ok"), (Assign("total", _c("plus", _v("total"), deeper)),), ())
    body = (If(_c("eq", _c("len", _v("chosen")), _v("n")), (Return(f[0]),), ()),
            Assign("total", f[1]),
            For(("k",), _c("range", _v("n")),
                (Assign("ok", Lit(True)), inner, step)),
            Return(_v("total")))
    return (Program("go", ("chosen", "n"), body),)


def _fill_recursive_search(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    chosen, r, k = _v("chosen"), _v("r"), _v("k")
    placed = _c("at", chosen, r)
    row = _c("len", chosen)
    if index == 0:
        return [Lit(1), Lit(0)]
    if index == 1:
        return [Lit(0)]
    # Reasons a choice is illegal: it repeats one already made, or it lies on either diagonal
    # from one already made. Nothing says which of these a given problem cares about.
    atoms = [_c("eq", placed, k),
             _c("eq", _c("minus", row, r), _c("minus", k, placed)),
             _c("eq", _c("minus", row, r), _c("minus", placed, k)),
             _c("gt", placed, k)]
    return _disjunctions(atoms)


def _peel(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """Repeatedly take an item nothing is waiting on, and release what was waiting on it.

    A **worklist**, and the shape of every "in what order can these be done" answer. It is not a
    table and it is not a relaxation: the loop shrinks a set of outstanding work, and what
    becomes available next depends on what was just taken.
    """
    size, facts = f[3], f[4]
    count = For(("e",), facts,
                (Assign("deg", _c("setat", _v("deg"), f[0],
                                  _c("plus", _c("at", _v("deg"), f[0]), Lit(1)))),))
    ready = For(("v",), _c("range", size),
                (If(_c("and", _c("eq", _c("at", _v("deg"), _v("v")), Lit(0)),
                       _c("not", _c("inset", _v("out"), _v("v")))),
                    (Assign("pick", _v("v")),), ()),))
    release = For(("e",), facts,
                  (If(_c("eq", _c("at", _v("e"), Lit(0)), _v("pick")),
                      (Assign("deg", _c("setat", _v("deg"), _c("at", _v("e"), Lit(1)),
                                        _c("minus", _c("at", _v("deg"),
                                                       _c("at", _v("e"), Lit(1))), Lit(1)))),),
                      ()),))
    return (Assign("deg", Lit(())),
            For(("i",), _c("range", size),
                (Assign("deg", _c("append", _v("deg"), Lit(0))),)),
            count,
            Assign("out", Lit(())),
            For(("_r",), _c("range", size),
                (Assign("pick", Lit(-1)), ready,
                 If(_c("eq", _v("pick"), Lit(-1)), (Return(f[1]),), ()),
                 Assign("out", _c("append", _v("out"), _v("pick"))),
                 Assign("deg", _c("setat", _v("deg"), _v("pick"), Lit(-1))),
                 release)),
            Return(f[2]))


def _fill_peel(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    e, out = _v("e"), _v("out")
    names = [_v(name) for name in spec.params]
    if index == 0:
        return [_c("at", e, Lit(1)), _c("at", e, Lit(0))]
    if index == 1:
        return [Lit(()), Lit(-1)]
    if index == 2:
        return [out, _c("len", out)]
    if index == 3:
        return names + [_c("len", name) for name in names]
    return names


def _grid_axes(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """The same property checked **along the rows and then down the columns** of a grid.

    Two passes that differ only in which index moves fastest. There is no transpose in the
    language, so a skeleton that reads a grid both ways is the only route to a question about
    both — and that question, "does anything repeat", is what validity means for a puzzle grid.
    """
    g = _v(params[0])
    wide = _c("len", _c("at", g, Lit(0)))
    tall = _c("len", g)

    def sweep(outer, inner, span_out, span_in):
        cell = Assign("v", _c("at", _c("at", g, _v("r")), _v("c")))
        seen = _v("seen")
        guard = If(_c("and", f[0], _c("inset", seen, _v("v"))), (Return(f[1]),), ())
        keep = If(f[0], (Assign("seen", _c("append", seen, _v("v"))),), ())
        return For((outer,), span_out,
                   (Assign("seen", Lit(())),
                    For((inner,), span_in, (cell, guard, keep))))

    return (sweep("r", "c", _c("range", tall), _c("range", wide)),
            sweep("c", "r", _c("range", wide), _c("range", tall)),
            Return(f[2]))


def _fill_grid_axes(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    v = _v("v")
    if index == 0:
        return [_c("ne", v, Lit(0)), Lit(True), _c("gt", v, Lit(0))]
    if index == 1:
        return [Lit(False), Lit(0)]
    return [Lit(True), Lit(1)]


# --------------------------------------------------------------------------- #
# recursion that takes the problem apart
# --------------------------------------------------------------------------- #

def _cofactor(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    return (Return(_c("invoke", Lit("det"), _v(params[0]))),)


def _helpers_cofactor(params: Sequence[str], f: Sequence[Term]) -> Tuple[Program, ...]:
    """Expansion along the first row: for each column, the value times the determinant of what
    is left when that column and the first row are struck out, with the sign flipping each time.

    Divide-and-conquer where the smaller instance has to be *built* — the minor is a grid with a
    row and a column removed — rather than being a slice of the original. That is what puts it
    outside every table skeleton: there is no index into an existing structure that names it.
    """
    m, c, r, k = _v("m"), _v("c"), _v("r"), _v("k")
    minor = For(("r",), _c("range2", Lit(1), _c("len", m)),
                (Assign("row", Lit(())),
                 For(("k",), _c("range", _c("len", m)),
                     (If(_c("ne", k, c),
                         (Assign("row", _c("append", _v("row"), _c("at", _c("at", m, r), k))),),
                         ()),)),
                 Assign("sub", _c("append", _v("sub"), _v("row")))))
    term = _c("mul", _c("mul", _v("sign"), _c("at", _c("at", m, Lit(0)), c)),
              _c("invoke", Lit("det"), _v("sub")))
    body = (If(_c("eq", _c("len", m), Lit(1)), (Return(f[0]),), ()),
            Assign("total", Lit(0)), Assign("sign", f[1]),
            For(("c",), _c("range", _c("len", m)),
                (Assign("sub", Lit(())), minor,
                 Assign("total", _c("plus", _v("total"), term)),
                 Assign("sign", _c("neg", _v("sign"))))),
            Return(_v("total")))
    return (Program("det", ("m",), body),)


def _fill_cofactor(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    m = _v("m")
    if index == 0:
        return [_c("at", _c("at", m, Lit(0)), Lit(0)), _c("sum", _c("at", m, Lit(0)))]
    return [Lit(1), Lit(-1)]


def _two_cursor(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    return (Return(_c("invoke", Lit("go"), _v(params[0]), _v(params[1]))),)


def _helpers_two_cursor(params: Sequence[str], f: Sequence[Term]) -> Tuple[Program, ...]:
    """Eat the front of two sequences together, where one of them may say "any number of these".

    **This is the narrowest skeleton in the set and it should be read as such.** The others name
    a control structure that many algorithms share — a table, a relaxation, a stack, a search.
    This one names a control structure that *pattern matching* has: try the repeat zero times,
    else consume one symbol and stay on the same pattern. Glob matching and regular-expression
    matching are both written on it and nothing else here is, so its claim to generality is
    thinner than the rest, and the fresh benchmark rather than this file is where that claim
    should be judged.

    What is **not** written in: which character means "any" and which means "repeat". Those come
    out of the task's own examples like every other constant in this module.
    """
    sub, pat = _v("s"), _v("p")
    head_s, head_p = _c("at", sub, Lit(0)), _c("at", pat, Lit(0))
    first = _c("and", _c("gt", _c("len", sub), Lit(0)),
               _c("or", _c("eq", head_p, head_s), _c("eq", head_p, f[0])))
    starred = _c("and", _c("gt", _c("len", pat), Lit(1)), _c("eq", _c("at", pat, Lit(1)), f[1]))
    body = (If(_c("eq", _c("len", pat), Lit(0)),
               (Return(_c("eq", _c("len", sub), Lit(0))),), ()),
            Assign("first", first),
            If(starred,
               (If(_c("invoke", Lit("go"), sub, _c("drop", pat, Lit(2))),
                   (Return(Lit(True)),), ()),
                If(_v("first"),
                   (Return(_c("invoke", Lit("go"), _c("drop", sub, Lit(1)), pat)),), ()),
                Return(Lit(False))),
               ()),
            If(_v("first"),
               (Return(_c("invoke", Lit("go"), _c("drop", sub, Lit(1)),
                          _c("drop", pat, Lit(1)))),), ()),
            Return(Lit(False)))
    return (Program("go", ("s", "p"), body),)


def _fill_two_cursor(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    marks: List[str] = []
    for piece in spec.constants()[1]:
        for ch in piece:
            if not ch.isalnum() and ch not in marks:
                marks.append(ch)
    if not marks:
        marks = [".", "*"]
    return [Lit(mark) for mark in marks[:6]]


# --------------------------------------------------------------------------- #
# the axes, rather than the patterns
#
# Twenty-five of twenty-eight on the bank these skeletons were written against, and **nought of
# twenty-one** on a bank chosen afterwards. That result is what this section exists because of.
# A hand-written skeleton answers the problem it was written for and its near neighbours; a fixed
# list of them is a list of answers, however carefully each one is phrased.
#
# What the fresh bank asked for was not twenty-one more patterns. Reading what actually blocked
# each one, the same two gaps came up over and over: a loop may carry **more than one** variable
# updated together — the two-number recurrences, the take-or-skip choices, the best-so-far that
# needs a worst-so-far beside it — and a table may be laid over a **grid** rather than over a
# sequence. Those are *axes* of the skeletons already here, not new members, and the fillings for
# them are enumerated over the live scope and pruned by behaviour rather than written down.
# --------------------------------------------------------------------------- #

def _behaviours(terms: Sequence[Term], samples: Sequence[Dict[str, Any]],
                width: int) -> List[Term]:
    """Keep one term per distinct behaviour, up to ``width``.

    The same observational-equivalence pruning :meth:`Coder.invent` uses, applied to a *hole*
    instead of to a whole program. ``v0 + v1`` and ``v1 + v0`` are two expressions and one
    behaviour, and a product over three holes pays for that difference three times over. A term
    that errors on the sample is kept unpruned: the sample is numeric and the real data may not
    be, so an error there is ignorance rather than evidence.
    """
    scratch = Interpreter(max_steps=4000)
    kept: List[Term] = []
    seen = set()
    for term in terms:
        try:
            signature = tuple(repr(scratch.run(term, dict(env))) for env in samples)
        except (CodeError, RecursionError):
            kept.append(term)
            if len(kept) >= width:
                break
            continue
        if signature in seen:
            continue
        seen.add(signature)
        kept.append(term)
        if len(kept) >= width:
            break
    return kept


_UPDATE_OPS = ("plus", "minus", "mul", "max2", "min2")


def _carried(params: Sequence[str], f: Sequence[Term], *, k: int) -> Tuple[Any, ...]:
    """``k`` running variables, seeded, updated **together** each pass, and answered from.

    Simultaneous update is the whole point and it is why this is not ``k`` copies of
    :func:`_accumulate`: ``take, skip = skip + x, max(skip, take)`` reads the *old* value of both
    on the right-hand side, and writing it as two statements gets a different answer. One
    :class:`Unpack` over a tuple built from the old values is the only faithful spelling.
    """
    names = [f"v{i}" for i in range(k)]
    packed: Term = Lit(())
    for index in range(k):
        packed = _c("append", packed, f[k + index])
    return (tuple(Assign(names[index], f[index]) for index in range(k))
            + (For(("x",), f[2 * k], (Unpack(tuple(names), packed),)),
               Return(f[2 * k + 1])))


def _fill_carried(index: int, spec: Spec, ints: Sequence[int], *, k: int) -> List[Term]:
    names = [_v(f"v{i}") for i in range(k)]
    x = _v("x")
    first = _v(spec.params[0])
    if index < k:                                    # what each one starts at
        return [Lit(0), Lit(1), Lit(-1), _c("head", first)]
    if index < 2 * k:                                # what each one becomes
        atoms: List[Term] = list(names) + [x, Lit(0), Lit(1)]
        pool: List[Term] = list(names) + [x]
        # Pairs outer, operators inner. The other way round — all the additions, then all the
        # subtractions — puts every ``max`` past any width worth having, and a running maximum
        # beside a running total is exactly the shape this axis exists for.
        for a, b in itertools.permutations(atoms, 2):
            for op in _UPDATE_OPS:
                pool.append(_c(op, a, b))
        # One more level, but only over the products a running variable actually forms with the
        # element -- which is where a running maximum of a signed product lives.
        scaled = [_c("mul", name, x) for name in names]
        for a, b in itertools.permutations(scaled + [x], 2):
            pool.append(_c("max2", a, b))
            pool.append(_c("min2", a, b))
        samples = [dict(zip([n.name for n in names] + ["x"], row))
                   for row in ((3, 5, 2, 4), (-1, 2, 7, 0), (7, 7, 1, -3), (0, 1, 4, 6))]
        return _behaviours(pool, samples, _CARRIED_WIDTH if k == 2 else _CARRIED_WIDTH_3)
    if index == 2 * k:                               # what the loop runs over
        out: List[Term] = []
        for name in spec.params:
            out.append(_v(name))
            out.append(_c("range", _v(name)))
            out.append(_c("range", _c("len", _v(name))))
        return out
    out = list(names)                                # what the answer is
    for a, b in itertools.permutations(names, 2):
        out += [_c("max2", a, b), _c("min2", a, b), _c("plus", a, b)]
    out += [_c("len", names[0]), _c("sum", names[0])]
    return out


def _carried_of(k: int) -> Tuple[Any, Any]:
    """The build and fill for ``k`` carried variables, as a pair of closures."""
    return (lambda params, f, k=k: _carried(params, f, k=k),
            lambda index, spec, ints, k=k: _fill_carried(index, spec, ints, k=k))


def _grid_table(params: Sequence[str], f: Sequence[Term]) -> Tuple[Any, ...]:
    """A table laid over a **grid**, one row at a time, with a running best beside it.

    :func:`_dp_two_seq` builds a table indexed by two *sequences*; this one is indexed by the
    grid's own rows and columns, and every cell may read the cell above it, the cell to its left,
    and the cell diagonally back. The running best exists because half of these problems answer
    with the largest cell rather than with the last one.
    """
    g = _v(params[0])
    wide = _c("len", _c("at", g, Lit(0)))
    tall = _c("len", g)
    top = For(("c",), _c("range", wide),
              (Assign("run", f[1]),
               Assign("best", _c("max2", _v("best"), _v("run"))),
               Assign("prev", _c("append", _v("prev"), _v("run")))))
    inner = For(("c",), _c("range2", Lit(1), wide),
                (Assign("cur", _c("append", _v("cur"), f[3])),
                 Assign("best", _c("max2", _v("best"),
                                   _c("at", _v("cur"), _c("minus", _c("len", _v("cur")),
                                                          Lit(1)))))))
    return (Assign("prev", Lit(())), Assign("run", f[0]), Assign("best", Lit(0)), top,
            For(("r",), _c("range2", Lit(1), tall),
                (Assign("cur", _c("append", Lit(()), f[2])),
                 Assign("best", _c("max2", _v("best"), _c("at", _v("cur"), Lit(0)))),
                 inner,
                 Assign("prev", _v("cur")))),
            Return(f[4]))


def _fill_grid_table(index: int, spec: Spec, ints: Sequence[int]) -> List[Term]:
    g = _v(spec.params[0])
    r, c, run, best = _v("r"), _v("c"), _v("run"), _v("best")
    prev, cur = _v("prev"), _v("cur")
    here = _c("at", _c("at", g, r), c)
    first_row = _c("at", _c("at", g, Lit(0)), c)
    first_col = _c("at", _c("at", g, r), Lit(0))
    up = _c("at", prev, c)
    left = _c("at", cur, _c("minus", c, Lit(1)))
    back = _c("at", prev, _c("minus", c, Lit(1)))
    blocked = _c("eq", first_row, Lit(1))
    if index == 0:
        return [Lit(0), Lit(1)]
    if index == 1:                                   # the top row, carrying `run`
        return [_c("plus", run, first_row), first_row,
                _c("if", blocked, Lit(0), run), _c("if", blocked, Lit(0), Lit(1)),
                _c("if", _c("eq", first_row, Lit(0)), Lit(0), Lit(1))]
    if index == 2:                                   # the left column
        return [_c("plus", _c("at", prev, Lit(0)), first_col), first_col,
                _c("if", _c("eq", first_col, Lit(1)), Lit(0), _c("at", prev, Lit(0))),
                _c("if", _c("eq", first_col, Lit(0)), Lit(0), Lit(1)),
                _c("at", prev, Lit(0))]
    if index == 3:                                   # every interior cell
        return [_c("plus", here, _c("min2", up, left)),
                _c("plus", here, _c("max2", up, left)),
                _c("if", _c("eq", here, Lit(1)), Lit(0), _c("plus", up, left)),
                _c("if", _c("eq", here, Lit(0)), Lit(0),
                   _c("plus", Lit(1), _c("min2", up, _c("min2", back, left)))),
                _c("plus", _c("min2", up, left), Lit(1)),
                _c("plus", here, back)]
    last = _c("at", prev, _c("minus", _c("len", prev), Lit(1)))
    return [last, best, _c("mul", best, best), _c("sum", prev), _c("maxof", prev)]


#: The imperative patterns, in the order a course teaches them. Every one of them carries a
#: variable across an iteration, which is the thing :meth:`Coder.invent` cannot build at any depth.
SKETCHES: Tuple[Sketch, ...] = (
    Sketch("accumulate", 1, ("seed", "value"), _accumulate, _fill_accumulate),
    Sketch("build_list", 1, ("test", "value"), _build_list, _fill_build_list),
    Sketch("accumulate_if", 1, ("seed", "test", "value"), _accumulate_if, _fill_accumulate_if),
    Sketch("by_index", 1, ("seed", "value"), _by_index, _fill_by_index),
    Sketch("running_best", 1, ("seed", "value", "value"), _running_best, _fill_running_best),
    Sketch("countdown", 1, ("seed", "value"), _countdown_while, _fill_countdown),
    Sketch("dp_row", 1, ("seed", "value"), _dp_row, _fill_dp_row),
    Sketch("pairs", 1, ("seed", "test", "value"), _pairs, _fill_pairs),
    Sketch("windows", 1, ("seed", "test", "value"), _windows, _fill_windows),
    Sketch("two_var_while", 2, ("test", "value", "value"), _two_var_while, _fill_two_var),
    Sketch("dp_row_inner", 1, ("seed", "seed", "value"), _dp_row_inner, _fill_dp_row_inner),
    Sketch("row_rebuild", 1, ("seed", "seed", "value", "seed"), _row_rebuild, _fill_row_rebuild),
    Sketch("prefix_suffix", 1, ("seed", "value", "value"), _prefix_suffix, _fill_prefix_suffix),
    Sketch("dp_scan", 1, ("seed", "test", "value", "value"), _dp_scan, _fill_dp_scan),
    Sketch("dp_two_seq", 2, ("seed", "seed", "test", "value", "value"),
           _dp_two_seq, _fill_dp_two_seq),
    Sketch("dp_amounts", 2, ("seed", "seed", "test", "value", "value"),
           _dp_amounts, _fill_dp_amounts),
    Sketch("merge_pick", 2, ("test", "value", "value"), _merge_pick, _fill_merge_pick),
    Sketch("scan_find", 2, ("test", "value", "seed"), _scan_find, _fill_scan_find),
    Sketch("recur_two", 2, ("seed", "value", "value"), _recur_two, _fill_recur_two),
    Sketch("set_closure", 2, ("seed", "value", "value"), _set_closure, _fill_set_closure),
    Sketch("dp_prefix", 2, ("seed", "seed", "test", "value"), _dp_prefix, _fill_dp_prefix),
    Sketch("dp_grid_items", 3, ("seed", "seed", "test", "value", "value"),
           _dp_grid_items, _fill_dp_grid_items),
    Sketch("coalesce", 1, ("test", "value", "value"), _coalesce, _fill_coalesce),
    Sketch("stack_scan", 1, ("seed", "test", "value"), _stack_scan, _fill_stack_scan),
    Sketch("windows_scan", 1,
           ("seed", "seed", "value", "value", "test", "test", "value", "value"),
           _windows_scan, _fill_windows_scan),
    Sketch("windows_scan2", 2,
           ("seed", "seed", "value", "value", "test", "test", "value", "value"),
           _windows_scan, _fill_windows_scan),
    Sketch("fixpoint2", 2, ("seed", "value", "test", "value", "value", "value", "value", "value"),
           _fixpoint, _fill_fixpoint, cap=200000),
    Sketch("fixpoint3", 3, ("seed", "value", "test", "value", "value", "value", "value", "value"),
           _fixpoint, _fill_fixpoint, cap=200000),
    Sketch("grid_axes", 1, ("test", "value", "value"), _grid_axes, _fill_grid_axes),
    Sketch("peel", 2, ("value", "value", "value", "value", "value"), _peel, _fill_peel),
    Sketch("recursive_search", 1, ("value", "seed", "test"),
           _recursive_search, _fill_recursive_search, helpers=_helpers_recursive_search),
    Sketch("cofactor", 1, ("value", "seed"), _cofactor, _fill_cofactor,
           helpers=_helpers_cofactor),
    Sketch("two_cursor", 2, ("value", "value"), _two_cursor, _fill_two_cursor,
           helpers=_helpers_two_cursor),
    Sketch("grid_table", 1, ("seed", "value", "value", "value", "value"),
           _grid_table, _fill_grid_table),
) + tuple(
    # The carried-variable axis, one member per width and per arity. Two variables updated
    # together is a different *shape* from one, not a different problem, and it is the shape a
    # fifth of the fresh bank turned out to want.
    Sketch(f"carried{k}", arity, ("seed",) * k + ("value",) * k + ("value", "value"),
           _carried_of(k)[0], _carried_of(k)[1], cap=cap)
    for k, cap in ((2, 1_600_000), (3, 1_600_000))
    for arity in (1, 2)
)


# --------------------------------------------------------------------------- #
# the coder
# --------------------------------------------------------------------------- #

#: How many filled skeletons :meth:`Coder.compose` may run in total, across every skeleton, on
#: one call. Per-skeleton caps say what is *reachable*; this says what one question is allowed to
#: cost. Without it the carried-variable axis alone — whose product is a pool raised to the number
#: of variables — turned a failed question from a second into twenty, and a syllabus of a few
#: hundred questions into an afternoon. A caller who is willing to search harder says so.
#:
#: **Why this number and not a larger one.** It is not a guess: every one of the six problems the
#: axes reach on the second bank is found inside a hundred thousand candidates, five of the six
#: inside sixty thousand. Above that the budget buys nothing measured, and :meth:`write` reaches
#: :meth:`compose` on almost every question it does not answer by recall — so the price of a
#: failure is the price of the syllabus.
_COMPOSE_BUDGET = 120_000

#: How many *evaluated nodes* one :meth:`Coder.compose` call may spend. The candidate count above
#: bounds how much is looked at; this bounds what looking **costs**, and it is the one that
#: actually bounds the clock. A candidate that folds a three-element list and one that runs a
#: nested loop over thirty are both one candidate and are not both one cost, so a syllabus of
#: small questions gets to try far more skeletons than a benchmark of large ones — which is the
#: right way round. Counting candidates alone is what took the syllabus from four minutes to
#: over twenty when the carried-variable axis went in.
_COMPOSE_EFFORT = 12_000_000

#: How wide a hole's filling pool may be after behaviours are collapsed. ``k`` carried variables
#: give ``k`` holes over this pool, so the product is its ``k``-th power — which is why the
#: collapse happens before the cap rather than after it, and why three variables get a much
#: narrower pool than two.
_CARRIED_WIDTH = 44
_CARRIED_WIDTH_3 = 15

#: How many candidates that ran out of budget — rather than being shown wrong — get a second
#: hearing under :attr:`Coder.deep`. Small, because a candidate that loops forever also times out
#: and there is no way to tell the two apart except by running them.
_REPRIEVE_WIDTH = 24

#: How many taught shapes get the *wide* one-changed sweep. It costs holes × pool each, so the whole
#: pass is bounded at a few thousand attempts — cheap enough to run before the enumeration and
#: expensive enough that it cannot run over every shape she has ever been shown.
_RECALL_WIDTH = 12

#: How many pairings one binary operator may contribute per level of :meth:`Coder.invent`. The
#: binary layer is the bank squared; without a cap it consumes the budget that the level above it
#: — where the compositions actually live — still needs.
_BINARY_WIDTH = 200

#: How many terms of one kind a level may build on. The bank keeps every distinct behaviour, but
#: a level that ranged over all of them would spend its budget on breadth at the depth it already
#: has rather than on reaching the next one.
_BANK_WIDTH = 220


class Coder:
    """The organ that reads programs, writes them, checks them, and learns their shapes.

    Nothing it produces is asserted. :meth:`write` returns a program only when that program
    reproduces every shown example under :class:`Interpreter`, and returns ``None`` — an
    abstention, the same one the rest of NJP is built to make — when the budget runs out. There
    is no "best effort" return, because a program that nearly works is not a partial answer, it is
    a wrong one that costs more to find out about.
    """

    def __init__(self, *, max_attempts: int = 6000, max_steps: int = 200000,
                 graft: bool = True, seeded: bool = True,
                 search_steps: int = 20000) -> None:
        self.max_attempts = int(max_attempts)
        self.interpreter = Interpreter(max_steps=max_steps)
        #: A second, tighter interpreter used only while searching. A candidate that needs two
        #: hundred thousand steps on a six-element example is not the answer — it is a wrong
        #: program looping — and letting it run to the full budget makes every wrong candidate
        #: cost what the worst right one would. The generous budget stays on :meth:`run` and
        #: :meth:`trace`, which is where a person's own program is executed.
        self.searcher = Interpreter(max_steps=int(search_steps), max_calls=32)
        #: The interpreter a *reprieved* candidate is re-run under. Big enough for a backtracking
        #: search over eight items, and used on at most :data:`_REPRIEVE_WIDTH` candidates per
        #: call, so a search that finds nothing still costs a bounded amount.
        self.deep = Interpreter(max_steps=4_000_000, max_calls=48)
        #: Set by :meth:`matches` when a candidate ran out of budget rather than being refuted.
        self.timed_out = False
        self.graft = bool(graft)
        self.schemas: Dict[str, Schema] = {}
        self.written = 0
        self.abstained = 0
        #: How many of the programs she wrote came from *recall* — a shape she was taught, with
        #: fillings she had seen in it — rather than from enumerating the space. Reported because
        #: it is the number that says what teaching is doing: a high share means the lessons are
        #: answering the questions, a low one means the seeds are and the lessons are decoration.
        self.recalled = 0
        #: How many programs came from :meth:`invent` — built out of the grammar rather than
        #: instantiated from any shape she held. This is the counter that says whether she can
        #: produce something genuinely new, and it is reported separately from `written` for
        #: exactly that reason.
        self.invented = 0
        #: How many programs came from :meth:`compose` — an imperative skeleton with a carried
        #: variable, filled and run. Counted apart from `invented` because they are different
        #: abilities: one composes expressions, the other builds a loop.
        self.composed = 0
        #: How many programs were found only on the second, generous hearing — candidates the
        #: search budget had timed out rather than ruled out.
        self.reprieved = 0
        self.attempts_spent = 0
        self.lessons = 0
        self.refusals = 0
        if seeded:
            for arity, skeleton in SEED_SHAPES:
                key = _skeleton_key(skeleton)
                self.schemas[key] = Schema(key=key, arity=arity, skeleton=skeleton,
                                           holes=_holes_of(skeleton), seed=True, origin="seed",
                                           block=isinstance(skeleton, tuple))

    # -- running and checking --------------------------------------------- #
    def run(self, program: Program, args: Sequence[Any]) -> Any:
        """Call ``program``. Raises :class:`CodeError`; callers that search use :meth:`check`."""
        if len(args) != len(program.params):
            raise CodeError(f"{program.name} takes {len(program.params)} argument(s)")
        self._register(program)
        env = dict(zip(program.params, args))
        try:
            if program.is_block:
                return self.interpreter.execute(program.body, env)
            return self.interpreter.run(program.body, env)
        except RecursionError as exc:
            # Belt as well as braces. `max_calls` is set below CPython's limit so this should be
            # unreachable, and "should be unreachable" is exactly the condition under which a
            # search that runs millions of candidates finds the way. A blown host stack is a
            # failed candidate like any other.
            raise Exhausted("the host stack gave out before the call budget did") from exc

    def _register(self, program: Program) -> None:
        """Put the program and its helpers where ``invoke`` can find them, and nothing else.

        Rebuilt on every call rather than accumulated. A registry that grew would let a program
        reach a function some *earlier* task happened to define, which is exactly the kind of
        invisible dependency that makes a passing test mean nothing.
        """
        table = self.interpreter.functions
        table.clear()
        table[program.name] = program
        for helper in program.helpers:
            table[helper.name] = helper

    def check(self, program: Program, examples: Sequence[Example]) -> Check:
        """Run every example and count what happened. The first failure is kept for the report.

        **A budget that runs out is retried, not scored.** Counting the arrangements of eight
        queens costs a few million evaluated nodes however the program is written, and the
        ordinary budget is a fifth of that. Scoring that as a failure marked a *correct*
        backtracker wrong on held-out data — the program was right and the clock was short, and
        those are not the same finding. The generous interpreter settles it; only a real
        mismatch, or a budget that runs out twice, counts against the program.
        """
        result = Check()
        for example in examples:
            try:
                try:
                    got = self.run(program, example.args)
                except Exhausted:
                    table = self.deep.functions
                    table.clear()
                    table[program.name] = program
                    for helper in program.helpers:
                        table[helper.name] = helper
                    env = dict(zip(program.params, example.args))
                    got = (self.deep.execute(program.body, env) if program.is_block
                           else self.deep.run(program.body, env))
            except CodeError as exc:
                result.errored += 1
                if result.first_failure is None:
                    result.first_failure, result.error = example, str(exc)
                continue
            if got == example.out and _kind_of(got) == _kind_of(example.out):
                result.passed += 1
            else:
                result.failed += 1
                if result.first_failure is None:
                    result.first_failure, result.got = example, got
        return result

    def matches(self, program: Program, examples: Sequence[Example]) -> bool:
        """Does it reproduce every example? Stops at the first that it does not.

        :meth:`check` counts, and counting means running all of them. Search does not need the
        count — it needs the yes/no — and the first mismatch settles that. Almost every candidate
        a search looks at is wrong on its first example, so this is the difference between
        running one example per candidate and running all of them.
        """
        if not examples:
            return False
        if len(program.params) != len(examples[0].args):
            return False
        table = self.searcher.functions
        table.clear()
        table[program.name] = program
        for helper in program.helpers:
            table[helper.name] = helper
        for example in examples:
            try:
                env = dict(zip(program.params, example.args))
                got = (self.searcher.execute(program.body, env) if program.is_block
                       else self.searcher.run(program.body, env))
            except Exhausted:
                # Not a refutation. The tight search interpreter is sized for the candidates that
                # are wrong, which is nearly all of them; a candidate that is still *running* when
                # it runs out has told us nothing except that it is expensive. Recorded so the
                # caller can give a few of these a second, generous hearing.
                self.timed_out = True
                return False
            except (CodeError, RecursionError):
                return False
            if got != example.out or _kind_of(got) != _kind_of(example.out):
                return False
        return True

    def matches_slowly(self, program: Program, examples: Sequence[Example]) -> bool:
        """:meth:`matches` again, with a budget sized for a program that is *doing* something.

        Counting the arrangements of eight queens is a few million evaluated nodes however it is
        written. Under the search budget every filling of the backtracking skeleton — including
        the exactly right one — came back :class:`Exhausted`, and the problem read as out of reach
        when what had actually happened is that nobody let it finish.
        """
        table = self.deep.functions
        table.clear()
        table[program.name] = program
        for helper in program.helpers:
            table[helper.name] = helper
        for example in examples:
            try:
                env = dict(zip(program.params, example.args))
                got = (self.deep.execute(program.body, env) if program.is_block
                       else self.deep.run(program.body, env))
            except (CodeError, RecursionError):
                return False
            if got != example.out or _kind_of(got) != _kind_of(example.out):
                return False
        return True

    def trace(self, program: Program, args: Sequence[Any]) -> List[TraceStep]:
        """Every sub-expression with the value it took — reading code, not just running it.

        This is the faculty behind *"what does this print, and why"*: the steps come back in
        evaluation order with their depth, so the answer is a derivation rather than a number.
        """
        steps: List[TraceStep] = []
        env = dict(zip(program.params, args))
        self._register(program)
        try:
            if program.is_block:
                self.interpreter.execute(program.body, env, trace=steps)
            else:
                self.interpreter.run(program.body, env, trace=steps)
        except CodeError as exc:
            steps.append(TraceStep(depth=0, source=f"<{exc}>", value=None))
        return steps

    # -- learning ---------------------------------------------------------- #
    def learn(self, demo: Demonstration) -> Learned:
        """Take a demonstration, verify it by execution, and keep only its shape.

        Three outcomes and they are kept apart on purpose:

        ``survived``   it reproduced every example the spec carries — the shape is confirmed;
        ``refuted``    it ran and got answers wrong — the shape's posterior moves *down*;
        ``undecided``  it could not run at all, so it is evidence about nothing.

        A teacher is not trusted here any more than anywhere else in NJP. What a good
        demonstration buys is one observation on one shape, and what a bad one buys is a mark
        against it.
        """
        from nyxara.njp.teacher import Verdict  # local: teacher.py imports nothing from here

        self.lessons += 1
        if demo.spec is None or demo.program is None:
            return Learned(verdict=Verdict.UNDECIDED, why="nothing was demonstrated")
        examples = tuple(demo.spec.shown) + tuple(demo.spec.held_out)
        if not examples:
            return Learned(verdict=Verdict.UNDECIDED, why="the task carries no examples")
        check = self.check(demo.program, examples)
        try:
            shape = abstract(demo.program)
        except Exception as exc:  # noqa: BLE001 — an unabstractable program teaches nothing
            return Learned(verdict=Verdict.UNDECIDED, check=check,
                           why=f"the demonstration has no shape to keep: {exc}")

        held = self.schemas.get(shape.key)
        before = held.posterior if held else Schema(key=shape.key).posterior
        if check.errored == check.total:
            self.refusals += 1
            return Learned(verdict=Verdict.UNDECIDED, key=shape.key, before=before, after=before,
                           check=check, why=f"it does not run: {check.error}")
        if not check.ok:
            self.refusals += 1
            if held is None:
                return Learned(verdict=Verdict.REFUTED, key=shape.key, before=before,
                               after=before, check=check,
                               why="the demonstration fails its own examples; nothing recorded")
            held.refuted += 1
            return Learned(verdict=Verdict.REFUTED, key=shape.key, before=before,
                           after=held.posterior, check=check,
                           why="the demonstration fails its own examples")
        fresh = held is None
        if held is None:
            held = shape
            self.schemas[shape.key] = held
        held.remember(_fillings_of(held.skeleton, demo.program.body))
        held.confirmed += 1
        held.origin = held.origin or "taught"
        if not held.seed:
            held.origin = "taught"
        return Learned(verdict=Verdict.SURVIVED, key=shape.key, before=before,
                       after=held.posterior, check=check, fresh=fresh,
                       why="verified by execution; shape kept, answer discarded")

    def learn_python(self, spec: Spec, source: str, *, name: str = "") -> Learned:
        """Same lesson, demonstrated as Python source rather than as a built term."""
        try:
            program = read_python(source, name=name or spec.name, params=spec.params)
        except CodeError as exc:
            from nyxara.njp.teacher import Verdict
            self.lessons += 1
            self.refusals += 1
            return Learned(verdict=Verdict.UNDECIDED, why=f"unreadable demonstration: {exc}")
        return self.learn(Demonstration(spec=spec, program=program, source="python"))

    # -- writing ------------------------------------------------------------ #
    def write(self, spec: Spec, *, attempts: Optional[int] = None,
              graft: Optional[bool] = None, invent: bool = True) -> Written:
        """Search for a program that reproduces every **shown** example, or abstain.

        Shapes are tried in order of posterior and then of cost, so what she has been taught is
        tried before what she was seeded with, and a cheap shape before an expensive one. The
        budget is in attempts — one instantiated candidate, run against the examples — so the
        cost of an answer is comparable across tasks and across a taught and an untaught brain.
        """
        budget = int(self.max_attempts if attempts is None else attempts)
        allow_graft = self.graft if graft is None else bool(graft)
        started = time.time()
        pools = self._pools(spec)
        tried = 0
        #: Running out of attempts stops the *enumeration*; it does not end the search. Returning
        #: on it did, and that quietly made invention unreachable on every task that needed it:
        #: a task no held shape fits is exactly a task whose enumeration runs to the end of its
        #: budget, so `invent` was only ever called for tasks the enumeration had already given
        #: up on cheaply — which is almost none of them.
        exhausted = False
        # ---- pass one: recall ------------------------------------------- #
        # Every taught shape, with only the fillings that have actually stood in it. Bounded and
        # tiny — a couple of dozen candidates per shape — and it goes first because it is the
        # cheapest thing in the search: a shape she was shown, with the constants she was shown in
        # it. Without it the enumeration order decides, and the enumeration order is dominated by
        # one-hole seed shapes whose function pool is a hundred wide: `is_prime` and `fizzbuzz`
        # were both *taught* and both timed out behind them.
        recallable = self._recallable(spec.arity)
        for schema in recallable:
            if exhausted:
                break
            for candidate, fillings in self._remembered(schema, spec, pools):
                if tried >= budget:
                    exhausted = True
                    break
                tried += 1
                if self.matches(candidate, spec.shown):
                    self.recalled += 1
                    return self._won(spec, candidate, schema, tried, started,
                                     self.check(candidate, spec.shown), False)
        # ---- pass one and a half: one thing changed ---------------------- #
        # The shape she was shown, with every hole at a value that has stood there before except
        # one, which sweeps its pool. This is what transfer between two instances of a family
        # actually looks like — ``sum triple the odd ones`` and ``sum double the even ones`` are
        # the same skeleton with one filling different — and it costs holes × pool rather than
        # pool to the power of holes, which is the difference between finding it and not.
        # `_same_operators` is signature-narrowed and therefore cheap, so it runs over every shape
        # she has been taught; `_one_changed` sweeps a whole pool per hole and runs over the most
        # confirmed few. Capping the narrow pass as well is what hid three families for two
        # measurements running: their shape was taught, its fillings were remembered, and it sat
        # at position twenty-one in a list of fifty-three sorted by a tie-break.
        for pass_over, width in ((self._same_operators, len(recallable)),
                                 (self._one_changed, _RECALL_WIDTH)):
            for schema in recallable[:width]:
                if exhausted:
                    break
                for candidate, fillings in pass_over(schema, spec, pools):
                    if tried >= budget:
                        exhausted = True
                        break
                    tried += 1
                    if self.matches(candidate, spec.shown):
                        self.recalled += 1
                        return self._won(spec, candidate, schema, tried, started,
                                         self.check(candidate, spec.shown), False)
        # ---- pass two: enumerate ---------------------------------------- #
        for schema in self._ranked(spec.arity):
            if exhausted:
                break
            for candidate, fillings in self._candidates(schema, spec, pools):
                if tried >= budget:
                    exhausted = True
                    break
                tried += 1
                if self.matches(candidate, spec.shown):
                    return self._won(spec, candidate, schema, tried, started,
                                     self.check(candidate, spec.shown), False)
        if allow_graft and not exhausted:
            for outer, inner in self._grafts(spec.arity):
                if exhausted:
                    break
                composite = _graft(outer, inner)
                if composite is None:
                    continue
                for candidate, fillings in self._candidates(composite, spec, pools):
                    if tried >= budget:
                        exhausted = True
                        break
                    tried += 1
                    if self.matches(candidate, spec.shown):
                        return self._won(spec, candidate, composite, tried, started,
                                         self.check(candidate, spec.shown), True)
        # ---- pass three: compose a loop --------------------------------- #
        # Before the expression search, because the thing it builds is strictly outside what the
        # expression search can reach: a variable that survives an iteration.
        if invent:
            looped = self.compose(spec)
            if looped is not None:
                self.composed += 1
                # Kept, like anything else that ran. A composed loop that worked is evidence of
                # the same kind a demonstration is, and not keeping it meant the next task of the
                # same shape paid for the whole search again instead of recalling it.
                shape = abstract(looped)
                shape.origin = "composed"
                shape.confirmed += 1
                shape.remember(_fillings_of(shape.skeleton, looped.body))
                self.schemas.setdefault(shape.key, shape)
                self.written += 1
                return Written(program=looped, schema=shape.key, attempts=tried,
                               seconds=time.time() - started,
                               shown=self.check(looped, spec.shown), grafted=True,
                               note="composed from a loop sketch, not from a shape she held")
        # ---- pass four: invent ------------------------------------------ #
        # Nothing she holds fits. Rather than abstain on that, build one out of the grammar —
        # the only pass here that can produce a shape nobody ever showed her.
        if invent:
            found = self.invent(spec)
            if found is not None and self.matches(found, spec.shown):
                self.invented += 1
                shape = abstract(found)
                shape.origin = "invented"
                shape.confirmed += 1
                shape.remember(_fillings_of(shape.skeleton, found.body))
                self.schemas.setdefault(shape.key, shape)
                self.written += 1
                return Written(program=found, schema=shape.key, attempts=tried,
                               seconds=time.time() - started,
                               shown=self.check(found, spec.shown), grafted=True,
                               note="invented from the grammar, not from a shape she held")
        return self._abstain(spec, tried, started,
                             "attempt budget exhausted and nothing invented" if exhausted
                             else "no shape she holds fits this task")

    def _won(self, spec: Spec, program: Program, schema: Schema, tried: int, started: float,
             check: Check, grafted: bool) -> Written:
        self.written += 1
        self.attempts_spent += tried
        schema.solved += 1
        schema.last_used = time.time()
        if grafted and schema.key not in self.schemas:
            # A shape she *invented* by composition and that worked is worth keeping, at one
            # observation — the same weight a demonstration gets, because it is the same kind of
            # evidence: it ran.
            schema.origin = "grafted"
            schema.confirmed += 1
            self.schemas[schema.key] = schema
        return Written(program=program, schema=schema.key, attempts=tried,
                       seconds=time.time() - started, shown=check, grafted=grafted)

    def _abstain(self, spec: Spec, tried: int, started: float, why: str) -> Written:
        self.abstained += 1
        self.attempts_spent += tried
        return Written(program=None, attempts=tried, seconds=time.time() - started, note=why)

    # -- search machinery --------------------------------------------------- #
    def _ranked(self, arity: int) -> List[Schema]:
        """Cheap before expensive, then taught before seeded, then confident before doubtful.

        Cost leads, and it did not always. Ordering by ``(seed, posterior, cost)`` — taught
        shapes first, whatever they cost — is the intuitive rule and it is wrong in a way
        :mod:`nyxara.njp.school` caught immediately: after six composites were taught,
        ``code-basics`` fell from 1.00 to 0.50 on the *same* tasks, because every two-hole taught
        shape was enumerated in full before the zero-hole seed that answers ``sum(xs)`` in one
        attempt, and the budget was gone before the search got there. Learning a hard thing had
        made an easy thing time out.

        A zero-hole shape costs one attempt. Trying it first is nearly free no matter what she
        believes about it, and no belief is strong enough to be worth four thousand attempts
        ahead of it. Within a cost tier the taught shape still leads the seeded one, which is
        where a lesson is supposed to pay — and it still does: a composite is only ever reached
        after the cheap tiers are exhausted, and those are a few hundred attempts in total.
        """
        usable = [s for s in self.schemas.values() if s.arity == arity]
        return sorted(usable, key=self._rank_key)

    @staticmethod
    def _rank_key(schema: Schema) -> Tuple[Any, ...]:
        """Cheap first, then **taught**, then the expensive seeds.

        Two failures, one each way, produced this order and both are worth keeping in view.

        Ordering by belief alone (``taught before seeded, whatever it costs``) made learning a hard
        thing break an easy one: every two-hole composite was enumerated in full before the
        zero-hole seed that answers ``sum(xs)`` in one attempt, and ``code-basics`` fell from 1.00
        to 0.50 on tasks it had just aced.

        Ordering by cost alone — the fix for that — broke the thing the lessons were for: a taught
        four-hole shape sits in the last tier, so a task whose answer she had been *shown* spent
        sixty thousand attempts on seeds first and timed out. Nineteen of sixty-five families
        failed that way, every one of them already learned.

        So: anything with at most one hole is tried first, because it is nearly free however little
        she believes in it; then everything she was taught, in order of belief, however expensive,
        because a shape she has seen work on this kind of task is the best guess available; then
        the expensive seeds, which are the fallback and are priced like one.
        """
        tier = 0 if schema.cost <= 1 else (1 if not schema.seed else 2)
        return (tier, -round(schema.posterior, 6), schema.cost, schema.key)

    def _grafts(self, arity: int) -> Iterator[Tuple[Schema, Schema]]:
        outers = [s for s in self._ranked(arity) if s.cost <= 2 and not s.block]
        inners = [s for s in self.schemas.values()
                  if s.arity == 1 and s.cost <= 1 and not s.block
                  and not isinstance(s.skeleton, Var)]
        inners.sort(key=lambda s: (s.seed, -round(s.posterior, 6), s.cost, s.key))
        for outer in outers:
            for inner in inners:
                yield outer, inner

    # -- composition: inventing a loop --------------------------------------- #
    def compose(self, spec: Spec, *, cap: int = 60000,
                budget: int = _COMPOSE_BUDGET,
                effort: int = _COMPOSE_EFFORT) -> Optional[Program]:
        """Build a program with a **loop and a carried variable**, having no shape for it.

        :meth:`invent` composes expressions, and that is a real ceiling rather than a soft one:
        measured over thirty classic hard algorithms it changed nothing at all, 1 of 28 before and
        1 of 28 after, because every one of them needs a running variable that survives an
        iteration. ``sum(map(f, xs))`` is an expression; ``for i: for j: if xs[i] > xs[j]: n += 1``
        is not, and no depth of expression search reaches it.

        This is *sketch-based synthesis*, and the pieces are worth naming plainly:

        * a :class:`Sketch` is an imperative skeleton with typed holes — a loop, an accumulator, a
          test, an update — and a **scope** attached to each hole saying which names are live
          there. ``xs[i]`` is only a candidate inside a loop that has an ``i``;
        * each hole is filled from a small set of expressions built over *its own scope*, not from
          a global pool, which is what keeps the product of holes countable;
        * the filled sketch is run against the shown examples like anything else, and nothing is
          returned that does not reproduce all of them.

        **Why the filling sets are small and curated rather than enumerated to depth.** A hole in
        scope of four names over twenty operators has thousands of depth-two fillings, and three
        such holes is a product nothing finishes. What is offered instead is the vocabulary a
        first course actually uses in that position: comparisons between the two indexed elements,
        an accumulator plus one, an accumulator plus the element. That is a real restriction and
        it is the reason this reaches some of these problems and not others. Where a real answer
        needs a *deep* condition — "the piece fits and the table says the rest is reachable and the
        piece is actually there" — the pool holds the three **atoms** and :func:`_conjunctions`
        offers every subset of them, so the sketch supplies a vocabulary of conditions rather than
        a condition.

        **Measured, on twenty-eight classic algorithms chosen before any of this existed.** With
        expression search alone: 1 of 28. With the ten scalar skeletons — a running sum, a best so
        far, a nested pair scan: 4 of 28, which reached the quadratic scans and nothing else. With
        the thirteen that carry a **table** — a row of answers to smaller instances, read back by
        index while the next row is built: **16 of 28**, including edit distance, longest common
        subsequence, knapsack, coin change, subset sum, word break and trapping rain water, each
        checked on inputs it was not fitted on.

        **What is still out of reach, named rather than glossed.** A backtracker that passes a
        partial solution down and undoes it; a worklist that is chosen from, removed from and
        grown until it empties; a window whose validity needs a scan of its own to decide; a
        monotonic stack; a grid read along both axes. Twelve of the twenty-eight are one of those,
        and adding skeletons until that list reads 28 of 28 would be fitting the benchmark rather
        than the ability. :meth:`write` abstains, which is the property that makes the limit safe.
        """
        rng_free = [sk for sk in SKETCHES if sk.arity == spec.arity]
        if not rng_free:
            return None
        ints, texts = spec.constants()
        tried = 0
        started = self.searcher.spent
        reprieved: List[Program] = []
        for sketch in rng_free:
            choices = [sketch.fillings(index, spec, ints) for index in range(len(sketch.holes))]
            if any(not group for group in choices):
                continue
            room = 1
            for group in choices:
                room *= len(group)
            if room > (sketch.cap or cap):
                continue
            for fillings in itertools.product(*choices):
                if tried >= budget or self.searcher.spent - started >= effort:
                    break
                tried += 1
                try:
                    block = sketch.build(spec.params, fillings)
                except CodeError:
                    break
                extra = (tuple(sketch.helpers(spec.params, fillings)) if sketch.helpers
                         else ())
                candidate = Program(spec.name or "f", spec.params, block, extra)
                self.timed_out = False
                if self.matches(candidate, spec.shown):
                    return candidate
                if self.timed_out and len(reprieved) < _REPRIEVE_WIDTH:
                    reprieved.append(candidate)
        for candidate in reprieved:
            if self.matches_slowly(candidate, spec.shown):
                self.reprieved += 1
                return candidate
        return None

    # -- invention ---------------------------------------------------------- #
    def invent(self, spec: Spec, *, max_terms: int = 4000, max_level: int = 3,
               witnesses: int = 4) -> Optional[Program]:
        """Build a program out of the grammar itself, having never been shown its shape.

        Everything else in :meth:`write` *instantiates* — it takes a skeleton she was taught or
        seeded with and fills the blanks. That is why she could be shown ``sum(map(f, filter(p,
        xs)))`` once and then write it for a task nobody demonstrated, and equally why a shape
        nobody ever showed her was simply out of reach: there was no skeleton to fill. Measured,
        that ceiling was 29 of 65 families cold and 1 of 28 unseen hard problems.

        This is the other half. It is **bottom-up enumerative synthesis with observational
        equivalence**, which is the standard method and is worth naming rather than dressing up:

        * start with the terminals — the parameters, and the constants the task itself put on the
          table;
        * a term's *signature* is what it evaluates to on each shown input. Two terms with the
          same signature are indistinguishable on this task, so only the first is kept. That is
          the pruning, and it is what makes the search finite in practice rather than in principle:
          the bank holds behaviours, not expressions;
        * build level ``k`` by applying every operator to terms from the levels below, keeping
          only those whose behaviour is new;
        * a term whose signature *is* the answer column is the program.

        **Type-directed, because otherwise it does not finish.** Terms are grouped by the kind of
        value they produce — the signature already says it — and ``add`` is only ever offered two
        integer-valued terms. Without that the binary layer alone is the bank squared.

        **The budget is small on purpose.** A success is cheap — the answer is usually two levels
        down and found in under a second — while a *failure* costs the whole budget, and a failure
        is what most calls are. A generous budget made a sweep of sixty-five tasks take longer
        than every other pass combined, for finds it had already made in the first few hundred
        terms. A caller who wants to search harder passes a larger ``max_terms``.

        **What this can and cannot invent.** It composes *expressions*: a fold over a filtered map,
        an index into a sort, a length of a deduplication. It does not invent an algorithm with a
        loop and a carried state — edit distance is not in this space at any depth, and no budget
        reaches it. So it moves the shape of the ceiling rather than removing it, and
        :meth:`write` still abstains when nothing in the space matches.
        """
        if not spec.shown:
            return None
        # Signatures are computed on a few *witness* examples rather than on all of them. The bank
        # is a pruning structure, not a proof: two terms that agree on four inputs are worth
        # collapsing into one, and a term that matches the answer on four is worth *checking*
        # properly. Running every candidate on every example spends the budget proving things
        # about candidates that were going to be discarded.
        sample = spec.shown[:max(2, witnesses)]
        envs = [dict(zip(spec.params, ex.args)) for ex in sample]
        target = tuple(ex.out for ex in sample)
        target_kind = _kind_of(target[0])

        def signature(term: Term) -> Optional[Tuple[Any, ...]]:
            out = []
            for env in envs:
                try:
                    value = self.searcher.run(term, env)
                except (CodeError, RecursionError):
                    return None
                out.append(value)
            return tuple(out)

        seen: Dict[Any, Term] = {}
        by_kind: Dict[str, List[Term]] = {}
        minted = 0

        def offer(term: Term) -> Optional[Program]:
            """Record a term if its behaviour is new; return a program if it is the answer."""
            nonlocal minted
            sig = signature(term)
            if sig is None:
                return None
            try:
                key = (tuple(map(_hashable, sig)))
            except CodeError:
                key = repr(sig)
            if key in seen:
                return None
            seen[key] = term
            minted += 1
            kinds = {_kind_of(v) for v in sig}
            by_kind.setdefault(kinds.pop() if len(kinds) == 1 else "mixed", []).append(term)
            if sig == target:
                # It agrees on the witnesses. Now make it earn the rest.
                candidate = Program(spec.name or "f", spec.params, term)
                if self.matches(candidate, spec.shown):
                    return candidate
            return None

        ints, texts = spec.constants()
        terminals: List[Term] = [Var(p) for p in spec.params]
        terminals += [Lit(v) for v in _pick_constants(ints, 8)]
        terminals += [Lit(v) for v in list(dict.fromkeys(texts))[:4]]
        terminals += [Lit(0), Lit(1), Lit(True), Lit(False), Lit(())]
        for term in terminals:
            hit = offer(term)
            if hit is not None:
                return hit

        functions = _fn1_library(ints, texts, _element_kinds(spec))
        folders = _fn2_library()
        # Unary operators are offered only to terms of a kind they can accept. Untyped, `sum` was
        # offered to every term in the bank including the booleans and the strings, and at level
        # two that layer alone is the bank times twenty-five — it consumed the budget before the
        # composition it was there to build could be reached.
        unary_by_kind: Dict[str, Tuple[str, ...]] = {
            "list": ("len", "sum", "maxof", "minof", "sort", "rev", "uniq", "head", "last",
                     "tail", "flat", "setof", "tolist", "anyof", "allof"),
            "str": ("len", "upper", "lower", "title", "strip", "words", "rev", "head", "last",
                    "tail", "isdigit", "toint"),
            "int": ("abs", "neg", "tostr", "range"),
            "bool": ("not",),
            "dict": ("len", "keys", "vals", "items"),
        }

        # The order inside a level is the whole of whether this finishes. Unary and higher-order
        # constructions are few and productive — `sum(...)`, `filter(f, ...)` — while the binary
        # layer is the bank squared and mostly noise. Offering binary first spent the entire
        # budget at level one, so level two never ran and `sum(filter(p, xs))` — a composition two
        # steps deep, and the commonest shape there is — was never reached at all.
        for _level in range(max_level):
            if minted >= max_terms:
                break
            pool = [t for group in by_kind.values() for t in group]
            numbers = list(by_kind.get("int", []))[:_BANK_WIDTH]
            lists = (list(by_kind.get("list", [])) + list(by_kind.get("str", [])))[:_BANK_WIDTH]

            for kind, ops in unary_by_kind.items():
                for term in by_kind.get(kind, [])[:_BANK_WIDTH]:
                    if minted >= max_terms:
                        break
                    for op in ops:
                        hit = offer(Call(op, (term,)))
                        if hit is not None:
                            return hit
            for fn in functions:
                for term in lists[:_BANK_WIDTH]:
                    if minted >= max_terms:
                        break
                    for op in ("map", "filter", "countif", "sortby"):
                        hit = offer(Call(op, (fn, term)))
                        if hit is not None:
                            return hit
            for fn in folders:
                for term in lists:
                    for start_at in (numbers[:4] + [Lit(0), Lit(1)]):
                        if minted >= max_terms:
                            break
                        hit = offer(Call("fold", (fn, start_at, term)))
                        if hit is not None:
                            return hit

            # Binary, with both sides typed as narrowly as the operator allows and each pairing
            # capped, so one operator cannot eat the budget the levels above it still need.
            pairs: List[Tuple[str, List[Term], List[Term]]] = [
                ("add", numbers, numbers), ("sub", numbers, numbers),
                ("mul", numbers, numbers), ("div", numbers, numbers),
                ("mod", numbers, numbers), ("min2", numbers, numbers),
                ("max2", numbers, numbers),
                ("at", lists, numbers), ("take", lists, numbers), ("drop", lists, numbers),
                ("count", lists, pool[:12]), ("has", lists, pool[:12]),
                ("cat", lists, lists), ("append", lists, pool[:12]),
                ("eq", pool[:14], pool[:14]), ("lt", numbers, numbers),
                ("gt", numbers, numbers), ("ge", numbers, numbers),
            ]
            for op, left, right in pairs:
                budgeted = 0
                for a in left:
                    for b in right:
                        if minted >= max_terms or budgeted >= _BINARY_WIDTH:
                            break
                        budgeted += 1
                        hit = offer(Call(op, (a, b)))
                        if hit is not None:
                            return hit
            if target_kind == "bool":
                flags = by_kind.get("bool", [])[:16]
                for a in flags:
                    for b in flags:
                        for op in ("and", "or"):
                            hit = offer(Call(op, (a, b)))
                            if hit is not None:
                                return hit
        return None

    def _recallable(self, arity: int) -> List[Schema]:
        """Taught shapes that have remembered fillings, most-confirmed first."""
        usable = [s for s in self.schemas.values()
                  if s.arity == arity and not s.seed and (s.seen or not s.holes)]
        return sorted(usable, key=lambda s: (-s.confirmed, -round(s.posterior, 6), s.key))

    def _remembered(self, schema: Schema, spec: Spec, pools: Dict[str, List[Term]],
                    *, cap: int = 24) -> Iterator[Tuple[Program, Tuple[Term, ...]]]:
        """Combinations built only from what has stood in this shape's holes before."""
        if schema.arity != spec.arity:
            return
        choices = [list(schema.seen.get(hole.index, ())) for hole in schema.holes]
        if any(not choice for choice in choices):
            return          # a hole nothing has stood in is not a recollection
        name = spec.name or "f"
        for count, fillings in enumerate(itertools.product(*choices)):
            if count >= cap:
                return
            try:
                body = schema.instantiate(spec.params, fillings, name=name)
            except CodeError:
                return
            yield Program(name=name, params=spec.params, body=body), fillings

    def _one_changed(self, schema: Schema, spec: Spec,
                     pools: Dict[str, List[Term]]) -> Iterator[Tuple[Program, Tuple[Term, ...]]]:
        """Every filling that differs from a remembered one in exactly one hole."""
        if schema.arity != spec.arity or len(schema.holes) < 2:
            return
        base = [schema.seen.get(hole.index, [None])[0] for hole in schema.holes]
        if any(value is None for value in base):
            return
        name = spec.name or "f"
        for position, hole in enumerate(schema.holes):
            for filling in (pools.get(hole.kind) or pools["any"]):
                if filling == base[position]:
                    continue
                fillings = tuple(base[:position] + [filling] + base[position + 1:])
                try:
                    body = schema.instantiate(spec.params, fillings, name=name)
                except CodeError:
                    break
                yield Program(name=name, params=spec.params, body=body), fillings

    def _same_operators(self, schema: Schema, spec: Spec,
                        pools: Dict[str, List[Term]],
                        *, cap: int = 2000) -> Iterator[Tuple[Program, Tuple[Term, ...]]]:
        """The shape she was shown, with the same *operations* in its holes and any constants.

        ``_one_changed`` covers a family whose instances differ in one place. Two of them differ in
        two — ``sum(map(x * 2, filter(x % 2 == 0, xs)))`` against ``x * 3`` and ``x % 2 == 1`` —
        and no amount of changing one thing reaches that. What does not change between instances
        of a family is the *shape of the fillings*: a multiplication is still a multiplication and
        a parity test is still a parity test. Matching on that signature narrows a two-hole
        product from forty thousand to a few hundred, and it narrows it by exactly the thing a
        family holds constant.
        """
        if schema.arity != spec.arity or not schema.holes:
            return
        choices: List[List[Term]] = []
        for hole in schema.holes:
            remembered = schema.seen.get(hole.index, ())
            if not remembered:
                return
            wanted = {_signature(value) for value in remembered}
            pool = [f for f in (pools.get(hole.kind) or pools["any"])
                    if _signature(f) in wanted]
            if not pool:
                return
            choices.append(pool)
        total = 1
        for pool in choices:
            total *= len(pool)
        if total > cap:
            return
        name = spec.name or "f"
        for fillings in itertools.product(*choices):
            try:
                body = schema.instantiate(spec.params, fillings, name=name)
            except CodeError:
                return
            yield Program(name=name, params=spec.params, body=body), fillings

    def _candidates(self, schema: Schema, spec: Spec,
                    pools: Dict[str, List[Term]]) -> Iterator[Tuple[Program, Tuple[Term, ...]]]:
        """Every filling of ``schema``'s holes, as a runnable program."""
        if schema.arity != spec.arity:
            return
        try:
            # `or pools["any"]` and not `pools.get(kind, any)`: the key is present and *empty*
            # for a kind this task has no values of, and an empty choice list silently makes the
            # whole schema unreachable rather than falling back.
            choices = []
            for hole in schema.holes:
                pool = pools.get(hole.kind) or pools["any"]
                known = [f for f in schema.seen.get(hole.index, ()) if f is not None]
                choices.append(list(dict.fromkeys(known + list(pool))) if known else list(pool))
        except KeyError:  # pragma: no cover - pools always carry "any"
            return
        if any(not choice for choice in choices):
            return
        for fillings in itertools.product(*choices):
            name = spec.name or "f"
            try:
                body = schema.instantiate(spec.params, fillings, name=name)
            except CodeError:
                return
            yield Program(name=name, params=spec.params, body=body), fillings

    def _pools(self, spec: Spec) -> Dict[str, List[Term]]:
        """What may be dropped into a hole, drawn from the task rather than from thin air."""
        ints, texts = spec.constants()
        sample = spec.shown[0].args if spec.shown else ()
        kinds = [_kind_of(value) for value in sample]
        by_kind: Dict[str, List[Term]] = {"int": [], "str": [], "list": [], "bool": [], "any": []}
        for index, kind in enumerate(kinds):
            if index < len(spec.params):
                by_kind.setdefault(kind, []).append(Var(spec.params[index]))

        kinds = _element_kinds(spec)
        int_lits = [Lit(v) for v in dict.fromkeys(_pick_constants(ints, 8) + [0, -1])][:14]
        str_lits = [Lit(v) for v in dict.fromkeys(list(texts) + [" ", ""])][:24]
        pools: Dict[str, List[Term]] = {
            "int": by_kind.get("int", []) + int_lits,
            "str": by_kind.get("str", []) + str_lits,
            "list": by_kind.get("list", []) + by_kind.get("str", []),
            "bool": [Lit(True), Lit(False)],
            "fn1": list(_fn1_library(ints, texts, kinds)),
            "fn2": list(_fn2_library()),
        }
        pools["any"] = (by_kind.get("int", []) + by_kind.get("list", []) +
                        by_kind.get("str", []) + int_lits[:8] + str_lits[:2])
        pools["test"] = self._tests(spec, ints)
        return pools

    def _tests(self, spec: Spec, ints: Sequence[int]) -> List[Term]:
        """Boolean terms over the task's own parameters, for the conditional shape."""
        out: List[Term] = []
        sample = spec.shown[0].args if spec.shown else ()
        constants = [Lit(v) for v in dict.fromkeys(_pick_constants(ints, 5) + [0])][:6]
        for index, value in enumerate(sample):
            if index >= len(spec.params):
                break
            var = Var(spec.params[index])
            if _kind_of(value) == "int":
                for k in constants:
                    out.extend([Call("gt", (var, k)), Call("lt", (var, k)),
                                Call("eq", (var, k))])
                out.append(Call("eq", (Call("mod", (var, Lit(2))), Lit(0))))
            elif _kind_of(value) in ("list", "str"):
                for k in constants[:3]:
                    out.extend([Call("gt", (Call("len", (var,)), k)),
                                Call("eq", (Call("len", (var,)), k))])
        return out[:36]

    # -- repair -------------------------------------------------------------- #
    def repair(self, program: Program, spec: Spec, *,
               attempts: Optional[int] = None) -> Written:
        """Debugging: one edit at a time, and the fix has to *run* before it is a fix.

        Every single-node edit is tried — a constant swapped, an operator swapped for one of the
        same arity, an inner function swapped — and the first edit that passes every shown example
        is returned. Nothing here reasons about *why* the program was wrong; it localises the fault
        by execution, which is the one method that cannot talk itself into a wrong answer.
        """
        budget = int(self.max_attempts if attempts is None else attempts)
        started = time.time()
        pools = self._pools(spec)
        tried = 0
        for path, node in _positions(program.body):
            for replacement in self._edits(node, pools):
                if tried >= budget:
                    return self._abstain(spec, tried, started, "attempt budget exhausted")
                tried += 1
                body = _replace_at(program.body, path, replacement)
                candidate = Program(program.name, program.params, body)
                if self.matches(candidate, spec.shown):
                    check = self.check(candidate, spec.shown)
                    self.written += 1
                    self.attempts_spent += tried
                    return Written(program=candidate, schema="repair", attempts=tried,
                                   seconds=time.time() - started, shown=check,
                                   note=f"replaced {render(node)} with {render(replacement)}")
        return self._abstain(spec, tried, started, "no single edit makes it pass")

    def _edits(self, node: Term, pools: Dict[str, List[Term]]) -> Iterator[Term]:
        if isinstance(node, Lit):
            for candidate in pools["any"]:
                if candidate != node:
                    yield candidate
        elif isinstance(node, Lambda):
            library = pools["fn1"] if len(node.params) == 1 else pools["fn2"]
            for candidate in library:
                if candidate != node:
                    yield candidate
        elif isinstance(node, Call) and node.op not in ("map", "filter", "fold"):
            want = arity_of(node.op)
            for op in OPS:
                if op != node.op and arity_of(op) == want and op not in ("map", "filter", "fold"):
                    yield Call(op, node.args)

    # -- explanation and reporting ------------------------------------------- #
    def explain(self, program: Program) -> str:
        """The program in a sentence. Reading code back is half of understanding it."""
        return f"{program.name}({', '.join(program.params)}) → {_describe(program.body)}"

    def stats(self) -> Dict[str, Any]:
        """What the brain reports about its own programming, and what the school grades."""
        taught = [s for s in self.schemas.values() if not s.seed]
        confirmed = sum(s.confirmed for s in self.schemas.values())
        asked = self.written + self.abstained
        return {
            "schemas": len(self.schemas),
            "taught": len(taught),
            "grafted": sum(1 for s in taught if s.origin == "grafted"),
            "confirmed": confirmed,
            "refuted": sum(s.refuted for s in self.schemas.values()),
            "lessons": self.lessons,
            "refusals": self.refusals,
            "written": self.written,
            "recalled": self.recalled,
            "invented": self.invented,
            "composed": self.composed,
            "recall_share": (round(self.recalled / self.written, 4) if self.written else 0.0),
            "abstained": self.abstained,
            "write_rate": round(self.written / asked, 4) if asked else 0.0,
            "attempts_spent": self.attempts_spent,
            "mean_attempts": round(self.attempts_spent / asked, 2) if asked else 0.0,
        }

    def taught_shapes(self) -> List[Dict[str, Any]]:
        """Every shape she was not born with, most-confirmed first."""
        taught = [s for s in self.schemas.values() if not s.seed]
        taught.sort(key=lambda s: (-s.confirmed, s.key))
        return [s.to_dict() for s in taught]


def _signature(term: Any) -> str:
    """The shape of a filling with its constants blanked. ``x * 2`` and ``x * 7`` share one."""
    if isinstance(term, Lit):
        return "?"
    if isinstance(term, Var):
        return term.name
    if isinstance(term, Lambda):
        return f"λ{len(term.params)}.{_signature(term.body)}"
    if isinstance(term, Call):
        return f"{term.op}({','.join(_signature(a) for a in term.args)})"
    return "?"


def _graft(outer: Schema, inner: Schema) -> Optional[Schema]:
    """Plug ``inner`` into the first ``#0`` of ``outer`` — how a composite shape gets invented.

    This is the only route from the primitives to a shape like ``sum(filter(?p, xs))`` without a
    teacher, and it is deliberately shallow: one graft, one level. Two levels is the same idea and
    a space large enough that the attempt budget always wins, which would make "she found it" a
    statement about the budget rather than about her.
    """
    if isinstance(outer.skeleton, Var) or isinstance(inner.skeleton, Var):
        return None
    shifted = _renumber(inner.skeleton, len(outer.holes))
    done = False

    def walk(node: Term) -> Term:
        nonlocal done
        if isinstance(node, Var) and node.name == "#0" and not done:
            done = True
            return shifted
        if isinstance(node, Call):
            return Call(node.op, tuple(walk(a) for a in node.args))
        if isinstance(node, Lambda):
            return Lambda(node.params, walk(node.body))
        return node

    skeleton = walk(outer.skeleton)
    if not done:
        return None
    key = render(skeleton)
    return Schema(key=key, arity=outer.arity, skeleton=skeleton, holes=_holes_of(skeleton),
                  origin="grafted")


def _children(node: Any) -> List[Any]:
    """Every sub-node, expressions and statements alike, in a fixed order.

    The order *is* the address: :func:`_positions` numbers by it and :func:`_replace_at` walks by
    it, so the two have to agree, and they agree by both going through here.
    """
    if isinstance(node, Call):
        return list(node.args)
    if isinstance(node, Lambda):
        return [node.body]
    if isinstance(node, tuple):
        return list(node)
    if isinstance(node, (Assign, Unpack)):
        return [node.value]
    if isinstance(node, SetItem):
        return [node.index, node.value]
    if isinstance(node, Return):
        return [] if node.value is None else [node.value]
    if isinstance(node, If):
        return [node.test, node.then, node.orelse]
    if isinstance(node, For):
        return [node.iterable, node.body]
    if isinstance(node, While):
        return [node.test, node.body]
    return []


def _rebuild(node: Any, children: List[Any]) -> Any:
    if isinstance(node, Call):
        return Call(node.op, tuple(children))
    if isinstance(node, Lambda):
        return Lambda(node.params, children[0])
    if isinstance(node, tuple):
        return tuple(children)
    if isinstance(node, Assign):
        return Assign(node.name, children[0])
    if isinstance(node, Unpack):
        return Unpack(node.names, children[0])
    if isinstance(node, SetItem):
        return SetItem(node.name, children[0], children[1])
    if isinstance(node, Return):
        return Return(children[0] if children else None)
    if isinstance(node, If):
        return If(children[0], children[1], children[2])
    if isinstance(node, For):
        return For(node.names, children[0], children[1])
    if isinstance(node, While):
        return While(children[0], children[1])
    return node


def _positions(term: Any, path: Tuple[int, ...] = ()) -> Iterator[Tuple[Tuple[int, ...], Any]]:
    yield path, term
    for index, child in enumerate(_children(term)):
        yield from _positions(child, path + (index,))


def _replace_at(term: Any, path: Tuple[int, ...], new: Any) -> Any:
    if not path:
        return new
    children = _children(term)
    head, rest = path[0], path[1:]
    if head >= len(children):
        return term
    children[head] = _replace_at(children[head], rest, new)
    return _rebuild(term, children)


_PHRASES = {
    "sum": "the total of {0}", "len": "how many things are in {0}",
    "maxof": "the largest of {0}", "minof": "the smallest of {0}",
    "sort": "{0} in order", "rev": "{0} backwards", "uniq": "{0} without repeats",
    "head": "the first of {0}", "last": "the last of {0}", "tail": "{0} after the first",
    "add": "{0} plus {1}", "sub": "{0} minus {1}", "mul": "{0} times {1}",
    "div": "{0} divided by {1}", "mod": "the remainder of {0} over {1}",
    "count": "how many times {1} appears in {0}", "has": "whether {0} contains {1}",
    "at": "item {1} of {0}", "range": "the numbers below {0}",
    "eq": "{0} equals {1}", "ne": "{0} differs from {1}", "lt": "{0} is under {1}",
    "gt": "{0} is over {1}", "le": "{0} is at most {1}", "ge": "{0} is at least {1}",
    "not": "{0} does not hold", "upper": "{0} in capitals", "lower": "{0} in lower case",
    "words": "the words of {0}", "join": "{1} joined by {0}",
}


def _describe(term: Term) -> str:
    if isinstance(term, Lit):
        return _show(term.value)
    if isinstance(term, Var):
        return term.name
    if isinstance(term, Hole):
        return "something"
    if isinstance(term, Lambda):
        return f"({', '.join(term.params)} ↦ {_describe(term.body)})"
    if not isinstance(term, Call):
        return "something"
    args = [_describe(a) for a in term.args]
    if term.op == "map":
        return f"each of {args[1]} put through {args[0]}"
    if term.op == "filter":
        return f"the parts of {args[1]} where {args[0]} holds"
    if term.op == "fold":
        return f"{args[2]} accumulated with {args[0]} starting from {args[1]}"
    if term.op == "if":
        return f"{args[1]} when {args[0]}, otherwise {args[2]}"
    phrase = _PHRASES.get(term.op)
    return phrase.format(*args) if phrase else f"{term.op} of {' and '.join(args)}"


# --------------------------------------------------------------------------- #
# reading Python — the whitelist, and the reason it is a whitelist
# --------------------------------------------------------------------------- #

_BINOPS = {ast.Add: "plus", ast.Sub: "minus", ast.Mult: "mul", ast.FloorDiv: "div",
           ast.Mod: "mod", ast.Pow: "pow", ast.BitAnd: "inter", ast.BitOr: "union",
           ast.BitXor: "symdiff"}
_CMPOPS = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt",
           ast.GtE: "ge"}
#: One-argument builtins that map straight onto an operator. Everything not named here and not
#: special-cased in :func:`_call` is unreachable — which is what makes ``open``, ``eval``,
#: ``exec``, ``__import__``, ``globals`` and every dunder a parse error rather than a policy.
_CALLS = {"len": "len", "sum": "sum", "abs": "abs", "reversed": "rev", "any": "anyof",
          "all": "allof", "chr": "chr", "ord": "ord", "bool": "not", "enumerate": "pairs"}

#: method name → (operator, how many arguments it takes *after* the receiver).
_METHODS: Dict[str, Tuple[str, int]] = {
    "upper": ("upper", 0), "lower": ("lower", 0), "title": ("title", 0),
    "strip": ("strip", 0), "isdigit": ("isdigit", 0), "isalpha": ("isalpha", 0),
    "split": ("spliton", 0), "join": ("join", 1), "count": ("count", 1),
    "replace": ("replace", 2), "startswith": ("starts", 1), "endswith": ("ends", 1),
    "index": ("find", 1), "keys": ("keys", 0), "values": ("vals", 0), "items": ("items", 0),
    "get": ("get", 1), "union": ("union", 1), "intersection": ("inter", 1),
    "difference": ("without", 1),
}

#: Her own dialect, readable back. Every one of these is already an operator in :data:`OPS`, so
#: accepting the name widens nothing: it is the same whitelist, spelled the way `render` spells it.
_DIALECT = frozenset({"fold", "takewhile", "dropwhile", "findfirst", "countif", "put", "setat",
                      "dropkey", "find", "pair", "frompairs", "flat", "invoke"})

#: Names the reader treats as calls to a sibling function rather than as unknown. Populated per
#: read by :func:`_function`; module level so :func:`_call` can see it, cleared between reads so
#: one source's helpers can never satisfy another's.
_READING_FUNCTIONS: set = set()


def read_python(source: str, *, name: str = "f",
                params: Optional[Sequence[str]] = None) -> Program:
    """Turn Python into a :class:`Program`, or refuse.

    **What is accepted.** One or more ``def``\\ s (the last is the program, the rest become its
    ``helpers``), a bare ``lambda``, or a bare expression when ``params`` is given. Inside a
    function: assignment and augmented assignment, subscript assignment, ``if``/``elif``/``else``,
    ``for`` with tuple unpacking, ``while``, ``break``, ``continue``, ``return``, ``pass``,
    docstrings, and recursion — a function may call itself and any function defined beside it.
    Inside an expression: arithmetic, comparison (chained included), ``and``/``or``/``not``,
    ``in``/``not in``, ``is None``, conditional expressions, indexing and slicing, list/dict/set
    literals, list/dict/set comprehensions with any number of ``if``\\ s, f-strings, and a fixed
    list of builtins and methods.

    **What is refused, at parse time and by enumeration.** Imports, ``with``, ``try``, ``class``,
    ``global``/``nonlocal``, ``del``, ``assert``, ``yield``, ``await``, ``lambda`` with defaults,
    starred and keyword arguments except ``key=``, attribute access outside :data:`_METHODS`, and
    every name outside :data:`_CALLS` — which is what makes ``open``, ``eval``, ``exec``,
    ``__import__`` and every dunder unreachable rather than merely discouraged. The accepted set
    is the one written down, so a future Python grammar cannot widen it by adding a node type.

    **One semantic divergence, and it is deliberate.** Containers here are values: ``xs.append(v)``
    reads as ``xs = xs + [v]`` and ``d[k] = v`` as ``d = {**d, k: v}``. For the accumulator idiom
    that is the same answer; for two names pointing at one object it is not. See
    :func:`_dict_set`.
    """
    try:
        tree = ast.parse(source.strip())
    except SyntaxError as exc:
        raise CodeError(f"not parseable: {exc}") from exc
    if not tree.body:
        raise CodeError("nothing to read")

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if functions:
        if len(functions) != len(tree.body):
            raise CodeError("only function definitions may sit beside a function definition")
        built = _read_all(functions)
        main = built[-1]
        return Program(main.name, main.params, main.body, tuple(built[:-1]))

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        raise CodeError("expected a definition or a single expression")
    value = tree.body[0].value
    if isinstance(value, ast.Lambda):
        return Program(name, _argnames(value.args), _from_ast(value.body))
    if params is None:
        raise CodeError("a bare expression needs its parameter names")
    return Program(name, tuple(params), _from_ast(value))


def read_module(source: str) -> Dict[str, Program]:
    """Every function in ``source``, by name. Each one carries the others as helpers."""
    tree = ast.parse(source.strip())
    built = _read_all([n for n in tree.body if isinstance(n, ast.FunctionDef)])
    by_name = {program.name: program for program in built}
    return {p.name: Program(p.name, p.params, p.body,
                            tuple(q for q in built if q.name != p.name))
            for p in by_name.values()}


def _read_all(nodes: Sequence[ast.FunctionDef]) -> List[Program]:
    """Read a group of functions with every name in the group already in scope.

    Two passes are not needed and one global is: a function may call a sibling defined *below*
    it, so the names have to be known before any body is read. The set is cleared afterwards so
    one source's helpers can never satisfy another's — the same reason
    :meth:`Coder._register` rebuilds its table on every call.
    """
    global _READING_FUNCTIONS
    previous = _READING_FUNCTIONS
    _READING_FUNCTIONS = {node.name for node in nodes}
    try:
        return [_function(node) for node in nodes]
    finally:
        _READING_FUNCTIONS = previous


def _function(node: ast.FunctionDef) -> Program:
    if node.decorator_list:
        raise CodeError("decorators are not read")
    body = _block(node.body)
    if len(body) == 1 and isinstance(body[0], Return) and body[0].value is not None:
        # A single `return` is an expression program, which is the shape schemas abstract most
        # cleanly. Keeping the block form here would make every one-liner unlearnable.
        return Program(node.name, _argnames(node.args), body[0].value)
    return Program(node.name, _argnames(node.args), body)


def _argnames(args: ast.arguments) -> Tuple[str, ...]:
    if args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg or args.defaults:
        raise CodeError("only plain positional parameters are read")
    return tuple(a.arg for a in args.args)


# --------------------------------------------------------------------------- #
# statements
# --------------------------------------------------------------------------- #

def _block(body: Sequence[ast.stmt]) -> Tuple[Any, ...]:
    out: List[Any] = []
    for node in body:
        statement = _statement(node)
        if statement is None:
            continue
        if isinstance(statement, tuple):
            out.extend(statement)          # one Python statement, several of ours
        else:
            out.append(statement)
    return tuple(out)


def _statement(node: ast.stmt) -> Any:  # noqa: C901 - a whitelist is a long if-chain by nature
    if isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None                      # a docstring
        return _bare_call(node.value)
    if isinstance(node, ast.Return):
        return Return(None if node.value is None else _from_ast(node.value))
    if isinstance(node, ast.Pass):
        return Nothing()
    if isinstance(node, ast.Break):
        return Break()
    if isinstance(node, ast.Continue):
        return Continue()
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            raise CodeError("only a single assignment target is read")
        return _assign(node.targets[0], _from_ast(node.value))
    if isinstance(node, ast.AugAssign):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise CodeError(f"augmented {type(node.op).__name__} is not in the language")
        target = node.target
        if isinstance(target, ast.Name):
            return Assign(target.id, Call(op, (Var(target.id), _from_ast(node.value))))
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            index = _from_ast(target.slice)
            current = Call("at", (Var(target.value.id), index))
            return SetItem(target.value.id, index, Call(op, (current, _from_ast(node.value))))
        raise CodeError("that augmented assignment target is not read")
    if isinstance(node, ast.AnnAssign):
        if node.value is None:
            raise CodeError("an annotation with no value assigns nothing")
        return _assign(node.target, _from_ast(node.value))
    if isinstance(node, ast.If):
        return If(_from_ast(node.test), _block(node.body), _block(node.orelse))
    if isinstance(node, ast.For):
        if node.orelse:
            raise CodeError("for/else is not read")
        return For(_targets(node.target), _from_ast(node.iter), _block(node.body))
    if isinstance(node, ast.While):
        if node.orelse:
            raise CodeError("while/else is not read")
        return While(_from_ast(node.test), _block(node.body))
    raise CodeError(f"{type(node).__name__} is not in the language")


#: Counter behind the temporaries a compound unpack needs. Module level so two reads in one
#: process cannot mint the same name and shadow each other.
_TEMPS = itertools.count()


def _assign(target: ast.expr, value: Term) -> Any:
    if isinstance(target, ast.Name):
        return Assign(target.id, value)
    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        return SetItem(target.value.id, _from_ast(target.slice), value)
    if isinstance(target, (ast.Tuple, ast.List)):
        if all(isinstance(e, ast.Name) for e in target.elts):
            return Unpack(_targets(target), value)
        return _compound_unpack(target, value)
    raise CodeError("only a name or one subscript may be assigned to")


def _compound_unpack(target: Any, value: Term) -> Tuple[Any, ...]:
    """``ys[j], ys[j+1] = ys[j+1], ys[j]`` — the line a swap is written on.

    The right side is evaluated once into a temporary and the targets are filled from it, which
    is what makes a swap a swap: filling them straight from the source would read a slot the
    previous target had already overwritten. Returns a *block*, and :func:`_block` splices it,
    because one Python statement is several here.
    """
    holder = f"_swap{next(_TEMPS)}"
    out: List[Any] = [Assign(holder, value)]
    for index, element in enumerate(target.elts):
        piece = Call("at", (Var(holder), Lit(index)))
        if isinstance(element, ast.Name):
            out.append(Assign(element.id, piece))
        elif isinstance(element, ast.Subscript) and isinstance(element.value, ast.Name):
            out.append(SetItem(element.value.id, _from_ast(element.slice), piece))
        else:
            raise CodeError("only names and subscripts may be unpacked into")
    return tuple(out)


def _targets(target: ast.expr) -> Tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)):
        if not all(isinstance(e, ast.Name) for e in target.elts):
            raise CodeError("only plain names may be unpacked")
        return tuple(e.id for e in target.elts)          # type: ignore[union-attr]
    raise CodeError("only a plain loop variable is read")


def _bare_call(node: ast.expr) -> Any:
    """A statement that is just a call. In this language only the mutators mean anything.

    ``xs.append(v)`` is the single most common line in a first course and refusing it would make
    the reader useless on real code, so it is compiled to the rebinding it is equivalent to here.
    Any other bare expression is dead code — it cannot have an effect, because nothing in the
    language has one — and it is dropped rather than run.
    """
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        # Refused rather than dropped. A statement whose value is discarded has no effect in this
        # language, so dropping it is *semantically* right and a reading hazard all the same:
        # `xs.pop()`, `d.update(…)`, a bare call to a helper — a reader that silently discards
        # them reports a wrong program as a correct one, which is the failure mode this whole
        # module is built to avoid. Saying "I do not read that line" is the honest answer.
        raise CodeError("a statement with no effect is not read; did you mean to return it?")
    receiver, method = node.func.value, node.func.attr
    if not isinstance(receiver, ast.Name):
        raise CodeError(f"{method}() on something that is not a name is not read")
    name = receiver.id
    args = [_from_ast(a) for a in node.args]
    if method == "append" and len(args) == 1:
        return Assign(name, Call("append", (Var(name), args[0])))
    if method == "extend" and len(args) == 1:
        return Assign(name, Call("extend", (Var(name), args[0])))
    if method == "add" and len(args) == 1:
        return Assign(name, Call("union", (Var(name), Call("setof", (Call("append",
                      (Lit(()), args[0])),)))))
    if method == "sort" and not args:
        return Assign(name, Call("sort", (Var(name),)))
    if method == "reverse" and not args:
        return Assign(name, Call("rev", (Var(name),)))
    raise CodeError(f"{method}() as a statement is not in the language")


# --------------------------------------------------------------------------- #
# expressions
# --------------------------------------------------------------------------- #

def _from_ast(node: ast.AST) -> Term:  # noqa: C901 - a whitelist is a long if-chain by nature
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, (bool, int, str)):
            return Lit(node.value)
        raise CodeError(f"literal of type {type(node.value).__name__} is not in the language")
    if isinstance(node, ast.Name):
        return Var(node.id)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [_from_ast(e) for e in node.elts]
        if all(isinstance(i, Lit) for i in items):
            values = tuple(i.value for i in items)       # type: ignore[union-attr]
            return Lit(frozenset(values)) if isinstance(node, ast.Set) else Lit(values)
        built: Term = Lit(())
        for item in items:
            built = Call("append", (built, item))
        return Call("setof", (built,)) if isinstance(node, ast.Set) else built
    if isinstance(node, ast.Dict):
        built = Call("emptydict", ())
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise CodeError("dict unpacking is not read")
            built = Call("put", (built, _from_ast(key), _from_ast(value)))
        return built
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise CodeError(f"operator {type(node.op).__name__} is not in the language")
        return Call(op, (_from_ast(node.left), _from_ast(node.right)))
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return Call("neg", (_from_ast(node.operand),))
        if isinstance(node.op, ast.Not):
            return Call("not", (_from_ast(node.operand),))
        if isinstance(node.op, ast.UAdd):
            return _from_ast(node.operand)
        raise CodeError(f"unary {type(node.op).__name__} is not in the language")
    if isinstance(node, ast.BoolOp):
        op = "and" if isinstance(node.op, ast.And) else "or"
        term = _from_ast(node.values[0])
        for extra in node.values[1:]:
            term = Call(op, (term, _from_ast(extra)))
        return term
    if isinstance(node, ast.Compare):
        return _compare(node)
    if isinstance(node, ast.IfExp):
        return Call("if", (_from_ast(node.test), _from_ast(node.body), _from_ast(node.orelse)))
    if isinstance(node, ast.Subscript):
        return _subscript(node)
    if isinstance(node, ast.Call):
        return _call(node)
    if isinstance(node, (ast.ListComp, ast.GeneratorExp, ast.SetComp)):
        return _comprehension(node)
    if isinstance(node, ast.DictComp):
        return _dict_comprehension(node)
    if isinstance(node, ast.JoinedStr):
        return _fstring(node)
    if isinstance(node, ast.Lambda):
        return Lambda(_argnames(node.args), _from_ast(node.body))
    raise CodeError(f"{type(node).__name__} is not in the language")


def _compare(node: ast.Compare) -> Term:
    """``a < b <= c`` becomes ``(a < b) and (b <= c)``, which is what Python means by it."""
    parts: List[Term] = []
    left = _from_ast(node.left)
    for operator, comparator in zip(node.ops, node.comparators):
        right = _from_ast(comparator)
        if isinstance(operator, ast.In):
            parts.append(Call("has", (right, left)))
        elif isinstance(operator, ast.NotIn):
            parts.append(Call("not", (Call("has", (right, left)),)))
        elif isinstance(operator, (ast.Is, ast.IsNot)):
            if not (isinstance(right, Lit) and right.value is None):
                raise CodeError("`is` is only read against None")
            check: Term = Call("isnone", (left,))
            parts.append(check if isinstance(operator, ast.Is) else Call("not", (check,)))
        else:
            op = _CMPOPS.get(type(operator))
            if op is None:
                raise CodeError(f"comparison {type(operator).__name__} is not in the language")
            parts.append(Call(op, (left, right)))
        left = right
    term = parts[0]
    for extra in parts[1:]:
        term = Call("and", (term, extra))
    return term


def _subscript(node: ast.Subscript) -> Term:
    value = _from_ast(node.value)
    index = node.slice
    if isinstance(index, ast.Slice):
        if index.step is not None:
            step = _from_ast(index.step)
            if (isinstance(step, Call) and step.op == "neg" and step.args == (Lit(1),)
                    and index.lower is None and index.upper is None):
                return Call("rev", (value,))
            raise CodeError("only [::-1] is read as a step")
        if index.lower is not None and index.upper is not None:
            return Call("slice", (value, _from_ast(index.lower), _from_ast(index.upper)))
        if index.lower is not None:
            low = _from_ast(index.lower)
            return Call("tail", (value,)) if low == Lit(1) else Call("drop", (value, low))
        if index.upper is not None:
            return Call("take", (value, _from_ast(index.upper)))
        return value
    return Call("at", (value, _from_ast(index)))


def _keyword_fn(node: ast.Call) -> Optional[Term]:
    """``key=`` is the one keyword argument that is read, because ``sorted(xs, key=f)`` is
    ordinary Python rather than an advanced form."""
    for keyword in node.keywords:
        if keyword.arg != "key":
            raise CodeError(f"keyword argument {keyword.arg!r} is not read")
        return _from_ast(keyword.value)
    return None


def _call(node: ast.Call) -> Term:  # noqa: C901 - one arm per accepted builtin
    key = _keyword_fn(node)
    args = [_from_ast(a) for a in node.args]

    if isinstance(node.func, ast.Attribute):
        return _method(node.func.attr, _from_ast(node.func.value), args)
    if not isinstance(node.func, ast.Name):
        raise CodeError("only named calls are read")
    fname = node.func.id

    if fname in ("list", "tuple"):
        if len(args) != 1:
            raise CodeError(f"{fname}() takes one argument here")
        return args[0]
    if fname == "int":
        if len(args) != 1:
            raise CodeError("int() takes one argument here")
        return Call("toint", (args[0],))
    if fname == "str":
        if len(args) != 1:
            raise CodeError("str() takes one argument here")
        return Call("tostr", (args[0],))
    if fname == "set":
        return Call("setof", (args[0] if args else Lit(()),))
    if fname == "dict":
        return Call("frompairs", (args[0],)) if args else Call("emptydict", ())
    if fname in ("max", "min"):
        stem = "max" if fname == "max" else "min"
        if key is not None:
            if len(args) != 1:
                raise CodeError(f"{fname}(…, key=…) takes one sequence")
            return Call(f"{stem}by", (key, args[0]))
        if len(args) == 1:
            return Call(f"{stem}of", (args[0],))
        if len(args) >= 2:
            # `min(a, b, c)` folds into nested two-argument minimums, which is what it means.
            # Two was the limit, and the line it made unreadable is the one at the centre of edit
            # distance: `min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)`. A reader that stops at
            # two arguments cannot read dynamic programming.
            folded = args[0]
            for extra in args[1:]:
                folded = Call(f"{stem}2", (folded, extra))
            return folded
        raise CodeError(f"{fname}() needs at least one argument")
    if fname == "sorted":
        if len(args) != 1:
            raise CodeError("sorted() takes one sequence")
        return Call("sortby", (key, args[0])) if key is not None else Call("sort", (args[0],))
    if fname == "range":
        if len(args) == 1:
            return Call("range", (args[0],))
        if len(args) == 2:
            return Call("range2", tuple(args))
        if len(args) == 3:
            # `range(len(xs) - 1, -1, -1)` is how a descending loop is written, and a language
            # that stops at two arguments cannot express one — which ruled out the backwards pass
            # that half of dynamic programming is made of.
            return Call("range3", tuple(args))
        raise CodeError("range() takes one to three arguments here")
    if fname == "sum":
        if len(args) == 1:
            return Call("sum", (args[0],))
        if len(args) == 2:
            return Call("add", (Call("sum", (args[0],)), args[1]))
        raise CodeError("sum() takes one or two arguments here")
    if fname == "zip":
        if len(args) != 2:
            raise CodeError("zip() takes two sequences here")
        return Call("zip", tuple(args))
    if fname == "pow":
        if len(args) != 2:
            raise CodeError("pow() takes two arguments here")
        return Call("pow", tuple(args))

    if fname in _DIALECT:
        want = arity_of(fname)
        if len(args) != want:
            raise CodeError(f"{fname}() takes {want} argument(s)")
        return Call(fname, tuple(args))
    op = _CALLS.get(fname)
    if op is not None:
        if len(args) != 1:
            raise CodeError(f"{fname}() takes one argument here")
        return Call(op, (args[0],))
    if fname in _READING_FUNCTIONS:
        return Call("invoke", (Lit(fname),) + tuple(args))
    raise CodeError(f"function {fname!r} is not in the language")


def _method(attr: str, receiver: Term, args: List[Term]) -> Term:
    entry = _METHODS.get(attr)
    if entry is None:
        raise CodeError(f"method {attr!r} is not in the language")
    op, wants = entry
    # `split` and `get` are the two with an optional argument, and both are ordinary Python:
    # `s.split()` and `s.split(",")`, `d.get(k)` and `d.get(k, default)`.
    optional = {"spliton": (0, 1), "get": (1, 2)}.get(op)
    allowed = range(optional[0], optional[1] + 1) if optional else range(wants, wants + 1)
    if len(args) not in allowed:
        raise CodeError(f"{attr}() takes {wants} argument(s) here")
    if op == "join":
        return Call("join", (receiver, args[0]))
    if op == "spliton":
        return Call("words", (receiver,)) if not args else Call("spliton", (receiver, args[0]))
    if op == "get":
        return Call("get", (receiver, args[0], args[1] if len(args) > 1 else Lit(None)))
    return Call(op, (receiver,) + tuple(args))


def _comprehension(node: Any) -> Term:
    stream, name, names = _generator(node)
    element = _project(node.elt, names, name)
    built = stream if element == Var(name) else Call("map", (Lambda((name,), element), stream))
    if isinstance(node, ast.SetComp):
        return Call("setof", (built,))
    # `[x for x in xs]` is a *list*, whatever ``xs`` was. Returning the stream untouched gave back
    # a set when the source was a set — the comprehension is the thing that makes it a list, and
    # skipping it because the element is the loop variable skipped the conversion too.
    return Call("tolist", (built,)) if built is stream else built


def _dict_comprehension(node: ast.DictComp) -> Term:
    stream, name, names = _generator(node)
    pair = Call("pair", (_project(node.key, names, name), _project(node.value, names, name)))
    return Call("frompairs", (Call("map", (Lambda((name,), pair), stream)),))


def _project(node: ast.AST, names: Tuple[str, ...], holder: str) -> Term:
    """The body of a comprehension, with any unpacked loop names read out of the held pair."""
    return _unpack(node, names, holder) if len(names) > 1 else _from_ast(node)


def _generator(node: Any) -> Tuple[Term, str, Tuple[str, ...]]:
    """The ``for … in … if …`` part, shared by every comprehension form.

    Returns the filtered stream, the single name the lambda will bind, and the *original* target
    names. When more than one name was unpacked the lambda binds the pair under one holder and
    every use of ``a`` or ``b`` in the body is rewritten to read out of it — which is the only
    form a one-parameter lambda can express, and it has to be applied to the element as well as
    to the conditions. It was applied only to the conditions, so ``[a + b for a, b in ps]``
    parsed and then failed at run time with an unbound name.
    """
    if len(node.generators) != 1:
        raise CodeError("only a single-generator comprehension is read")
    gen = node.generators[0]
    if gen.is_async:
        raise CodeError("async comprehensions are not read")
    names = _targets(gen.target)
    stream = _from_ast(gen.iter)
    holder = "_pair" if len(names) > 1 else names[0]
    for condition in gen.ifs:
        stream = Call("filter", (Lambda((holder,), _project(condition, names, holder)), stream))
    return stream, holder, names


def _unpack(node: ast.AST, names: Tuple[str, ...], holder: str) -> Term:
    """Rewrite ``a`` and ``b`` to ``holder[0]`` and ``holder[1]`` inside an unpacked generator."""
    term = _from_ast(node)
    slots = {name: index for index, name in enumerate(names)}

    def walk(inner: Term) -> Term:
        if isinstance(inner, Var) and inner.name in slots:
            return Call("at", (Var(holder), Lit(slots[inner.name])))
        if isinstance(inner, Call):
            return Call(inner.op, tuple(walk(a) for a in inner.args))
        if isinstance(inner, Lambda):
            return Lambda(inner.params, walk(inner.body))
        return inner

    return walk(term)


def _fstring(node: ast.JoinedStr) -> Term:
    """``f"{a} and {b}"`` becomes string concatenation, which is all it is."""
    built: Term = Lit("")
    for part in node.values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            piece: Term = Lit(part.value)
        elif isinstance(part, ast.FormattedValue):
            if part.format_spec is not None or part.conversion not in (-1, 115):
                raise CodeError("format specifiers are not read")
            piece = Call("tostr", (_from_ast(part.value),))
        else:
            raise CodeError("that f-string is not read")
        built = Call("cat", (built, piece))
    return built
