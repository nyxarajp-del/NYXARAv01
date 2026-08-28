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

**Every program halts.** :attr:`Interpreter.max_steps` is a hard budget and there is no open
recursion in the language — iteration exists as ``map``/``filter``/``fold`` over a finite
sequence. So she cannot write ``while True`` and she cannot write Ackermann, and the price of that
is paid honestly here rather than discovered as a hung process. What the fold-based subset *does*
cover is the whole of the beginner-to-intermediate curriculum this module is built to teach.

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
budget. The seed schemas are the primitive shapes (:data:`SEED_SHAPES`); a composite shape is
reachable cold only by grafting, which is combinatorial and usually runs out of budget before it
lands. A schema she has been shown collapses that search to a handful of fillings. So the
acquisition claim this module supports is precise and small: *taught, she solves specs within a
budget that untaught she does not*, on tasks the teacher never showed her — and
:mod:`nyxara.njp.school` measures exactly that difference rather than asserting it.

Pure standard library. No LLM anywhere in the path.
"""

from __future__ import annotations

import ast
import itertools
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

__all__ = [
    "CodeError", "Exhausted",
    "Var", "Lit", "Call", "Lambda", "Hole", "Term",
    "OPS", "render", "read_python", "arity_of",
    "TraceStep", "Interpreter",
    "Example", "Spec", "Program", "Schema", "Demonstration", "Learned", "Check", "Written",
    "Coder", "SEED_SHAPES",
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
    if isinstance(value, tuple):
        return value
    if isinstance(value, str):
        return tuple(value)
    raise CodeError(f"expected a sequence, got {type(value).__name__}")


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
        return a + b
    return _seq(a) + _seq(b)


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


#: name → (arity, implementation, rendering template). The rendering template is a Python
#: expression with ``{0}``/``{1}`` placeholders, used by :func:`render` so a synthesised program
#: can be read by a person. ``if``/``and``/``or`` are absent: they are lazy and the interpreter
#: handles them itself, because evaluating the untaken branch of a conditional turns a correct
#: program into a :class:`CodeError` (``if_(gt(n,0), div(x,n), 0)`` is the standard example).
OPS: Dict[str, Tuple[int, Callable[..., Any], str]] = {
    # arithmetic
    "add":      (2, lambda a, b: _int(a) + _int(b), "({0} + {1})"),
    "sub":      (2, lambda a, b: _int(a) - _int(b), "({0} - {1})"),
    "mul":      (2, lambda a, b: _int(a) * _int(b), "({0} * {1})"),
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
    "len":      (1, lambda xs: len(_seq(xs)), "len({0})"),
    "sum":      (1, lambda xs: sum(_int(x) for x in _seq(xs)), "sum({0})"),
    "maxof":    (1, lambda xs: _reduce_num(xs, max, "max"), "max({0})"),
    "minof":    (1, lambda xs: _reduce_num(xs, min, "min"), "min({0})"),
    "rev":      (1, lambda xs: xs[::-1] if isinstance(xs, str) else _seq(xs)[::-1],
                 "{0}[::-1]"),
    "sort":     (1, _sorted, "sorted({0})"),
    "head":     (1, _head, "{0}[0]"),
    "last":     (1, _last, "{0}[-1]"),
    "tail":     (1, lambda xs: _seq(xs)[1:], "{0}[1:]"),
    "at":       (2, _at, "{0}[{1}]"),
    "take":     (2, lambda xs, n: _seq(xs)[:_int(n)], "{0}[:{1}]"),
    "drop":     (2, lambda xs, n: _seq(xs)[_int(n):], "{0}[{1}:]"),
    "cat":      (2, _cat, "({0} + {1})"),
    "count":    (2, lambda xs, x: sum(1 for item in _seq(xs) if item == x),
                 "{0}.count({1})"),
    "has":      (2, lambda xs, x: any(item == x for item in _seq(xs)), "({1} in {0})"),
    "uniq":     (1, lambda xs: tuple(dict.fromkeys(_seq(xs))), "list(dict.fromkeys({0}))"),
    "range":    (1, _range1, "list(range({0}))"),
    "range2":   (2, _range2, "list(range({0}, {1}))"),
    # strings
    "upper":    (1, lambda s: _text(s).upper(), "{0}.upper()"),
    "lower":    (1, lambda s: _text(s).lower(), "{0}.lower()"),
    "words":    (1, lambda s: tuple(_text(s).split()), "{0}.split()"),
    "join":     (2, lambda sep, xs: _text(sep).join(_text(x) for x in _seq(xs)),
                 "{0}.join({1})"),
    # higher order — the first argument is a Lambda, handled in the interpreter
    # The rendering templates matter more here than anywhere else in the table: a schema's key
    # *is* its rendering, so two shapes that render alike are one shape as far as learning is
    # concerned. These three carried empty templates once, and every map/filter/fold schema
    # collapsed onto the same key the moment its function was a hole.
    "map":      (2, None, "map({0}, {1})"),
    "filter":   (2, None, "filter({0}, {1})"),
    "fold":     (3, None, "fold({0}, {1}, {2})"),
}

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

    def __init__(self, *, max_steps: int = 20000) -> None:
        self.max_steps = int(max_steps)
        self.steps = 0

    # -- evaluation ------------------------------------------------------- #
    def run(self, term: Term, env: Optional[Dict[str, Any]] = None,
            *, trace: Optional[List[TraceStep]] = None) -> Any:
        """Evaluate ``term`` under ``env``. Raises :class:`CodeError` on anything malformed."""
        self.steps = 0
        return self._eval(term, dict(env or {}), 0, trace)

    def _tick(self) -> None:
        self.steps += 1
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
        entry = OPS.get(op)
        if entry is None:
            raise CodeError(f"unknown operator {op!r}")
        want, fn, _ = entry
        if len(args) != want:
            raise CodeError(f"{op} takes {want} argument(s), given {len(args)}")
        if op in ("map", "filter", "fold"):
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

    def _higher(self, op: str, args: Tuple[Term, ...], env: Dict[str, Any], depth: int,
                trace: Optional[List[TraceStep]]) -> Any:
        fn = args[0]
        if not isinstance(fn, Lambda):
            fn = self._eval(fn, env, depth + 1, trace)
        if not isinstance(fn, Lambda):
            raise CodeError(f"{op} needs a function as its first argument")
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
        out: List[Any] = []
        for item in items:
            self._tick()
            got = self._eval(fn.body, {**env, fn.params[0]: item}, depth + 1, None)
            if op == "map":
                out.append(got)
            elif _truth(got):
                out.append(item)
        return tuple(out)


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


# --------------------------------------------------------------------------- #
# tasks, programs and the shapes distilled out of them
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Example:
    """One input/output pair. ``args`` is positional and lists are tuples — values are hashable
    here so a candidate's behaviour can be memoised and duplicates dropped."""

    args: Tuple[Any, ...] = ()
    out: Any = None


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

        return Spec(name=name, intent=intent, params=tuple(params),
                    shown=tuple(Example(tuple(freeze(a) for a in args), freeze(out))
                                for args, out in shown),
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
            elif isinstance(value, tuple):
                for item in value:
                    walk(item)

        for example in self.shown:
            for arg in example.args:
                walk(arg)
            walk(example.out)
        seen_i = sorted({i for i in ints if -64 <= i <= 64})
        seen_s = sorted({s for s in texts if len(s) <= 32})
        return seen_i, seen_s


@dataclass(frozen=True)
class Program:
    """A finished function: its parameters and the term it returns."""

    name: str = "f"
    params: Tuple[str, ...] = ()
    body: Any = None

    def source(self) -> str:
        """The program as Python source. Nothing executes this — it is for people."""
        return (f"def {self.name}({', '.join(self.params)}):\n"
                f"    return {render(self.body)}")

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
    origin: str = ""
    last_used: float = 0.0
    solved: int = 0

    @property
    def posterior(self) -> float:
        """``(confirmed + 1) / (confirmed + refuted + 2)`` — a shape nobody has demonstrated sits
        at 0.5 rather than at zero, because an unseen shape is unknown, not wrong."""
        return (self.confirmed + 1.0) / (self.confirmed + self.refuted + 2.0)

    @property
    def cost(self) -> int:
        """How many blanks have to be guessed. Search tries the cheap shapes first."""
        return len(self.holes)

    def instantiate(self, params: Sequence[str], fillings: Sequence[Term]) -> Term:
        """Bind ``#i`` to ``params[i]`` and drop ``fillings`` into the holes."""
        return _substitute(self.skeleton, tuple(params), tuple(fillings))

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "arity": self.arity, "holes": self.cost,
                "confirmed": self.confirmed, "refuted": self.refuted,
                "posterior": round(self.posterior, 4), "seed": self.seed,
                "origin": self.origin, "solved": self.solved}


def _substitute(term: Term, params: Tuple[str, ...], fillings: Tuple[Term, ...]) -> Term:
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
        return Lambda(term.params, _substitute(term.body, params, fillings))
    if isinstance(term, Call):
        return Call(term.op, tuple(_substitute(a, params, fillings) for a in term.args))
    return term


def _holes_of(term: Term) -> Tuple[Hole, ...]:
    found: List[Hole] = []

    def walk(node: Term) -> None:
        if isinstance(node, Hole):
            found.append(node)
        elif isinstance(node, Call):
            for arg in node.args:
                walk(arg)
        elif isinstance(node, Lambda):
            walk(node.body)

    walk(term)
    return tuple(sorted(found, key=lambda h: h.index))


def _renumber(term: Term, offset: int) -> Term:
    """Shift every hole index by ``offset`` — needed when one schema is grafted into another."""
    if isinstance(term, Hole):
        return Hole(term.index + offset, term.kind)
    if isinstance(term, Call):
        return Call(term.op, tuple(_renumber(a, offset) for a in term.args))
    if isinstance(term, Lambda):
        return Lambda(term.params, _renumber(term.body, offset))
    return term


def _kind_of(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, str):
        return "str"
    if isinstance(value, tuple):
        return "list"
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

    def walk(node: Term) -> Term:
        if isinstance(node, Lit):
            return Hole(next(counter), _kind_of(node.value))
        if isinstance(node, Lambda):
            return Hole(next(counter), f"fn{len(node.params)}")
        if isinstance(node, Var):
            return Var(f"#{slots[node.name]}") if node.name in slots else node
        if isinstance(node, Call):
            return Call(node.op, tuple(walk(a) for a in node.args))
        return node

    skeleton = walk(program.body)
    return Schema(key=render(skeleton), arity=len(program.params), skeleton=skeleton,
                  holes=_holes_of(skeleton), origin="taught")


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
    return tuple(shapes)


SEED_SHAPES: Tuple[Tuple[int, Any], ...] = _seed_shapes()


def _lambda1(body_of: Callable[[Term], Term]) -> Lambda:
    return Lambda(("x",), body_of(Var("x")))


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


def _fn1_library(ints: Sequence[int]) -> Tuple[Lambda, ...]:
    """Unary functions worth trying inside a ``map`` or a ``filter``.

    The fixed part is the arithmetic and predicate vocabulary a first course covers; the
    parameterised part is built over :func:`_pick_constants`, so the pool stays a few dozen wide
    instead of unbounded and still reaches the constant the task actually needs.
    """
    x = Var("x")
    out: List[Lambda] = [
        Lambda(("x",), x),
        Lambda(("x",), Call("neg", (x,))),
        Lambda(("x",), Call("abs", (x,))),
        Lambda(("x",), Call("mul", (x, x))),
        Lambda(("x",), Call("len", (x,))),
        Lambda(("x",), Call("sum", (x,))),
        Lambda(("x",), Call("upper", (x,))),
        Lambda(("x",), Call("lower", (x,))),
        Lambda(("x",), Call("rev", (x,))),
        Lambda(("x",), Call("eq", (Call("mod", (x, Lit(2))), Lit(0)))),
        Lambda(("x",), Call("eq", (Call("mod", (x, Lit(2))), Lit(1)))),
        Lambda(("x",), Call("gt", (x, Lit(0)))),
        Lambda(("x",), Call("lt", (x, Lit(0)))),
        Lambda(("x",), Call("ge", (x, Lit(0)))),
    ]
    for k in _pick_constants(ints, 14):
        out.extend([
            Lambda(("x",), Call("mul", (x, Lit(k)))),
            Lambda(("x",), Call("add", (x, Lit(k)))),
            Lambda(("x",), Call("gt", (x, Lit(k)))),
            Lambda(("x",), Call("lt", (x, Lit(k)))),
            Lambda(("x",), Call("sub", (x, Lit(k)))),
            Lambda(("x",), Call("eq", (x, Lit(k)))),
        ])
    return tuple(dict.fromkeys(out))


def _fn2_library() -> Tuple[Lambda, ...]:
    a, x = Var("a"), Var("x")
    return tuple(Lambda(("a", "x"), Call(op, (a, x)))
                 for op in ("add", "mul", "max2", "min2", "sub", "cat"))


# --------------------------------------------------------------------------- #
# the coder
# --------------------------------------------------------------------------- #

class Coder:
    """The organ that reads programs, writes them, checks them, and learns their shapes.

    Nothing it produces is asserted. :meth:`write` returns a program only when that program
    reproduces every shown example under :class:`Interpreter`, and returns ``None`` — an
    abstention, the same one the rest of NJP is built to make — when the budget runs out. There
    is no "best effort" return, because a program that nearly works is not a partial answer, it is
    a wrong one that costs more to find out about.
    """

    def __init__(self, *, max_attempts: int = 6000, max_steps: int = 20000,
                 graft: bool = True, seeded: bool = True) -> None:
        self.max_attempts = int(max_attempts)
        self.interpreter = Interpreter(max_steps=max_steps)
        self.graft = bool(graft)
        self.schemas: Dict[str, Schema] = {}
        self.written = 0
        self.abstained = 0
        self.attempts_spent = 0
        self.lessons = 0
        self.refusals = 0
        if seeded:
            for arity, skeleton in SEED_SHAPES:
                key = render(skeleton)
                self.schemas[key] = Schema(key=key, arity=arity, skeleton=skeleton,
                                           holes=_holes_of(skeleton), seed=True, origin="seed")

    # -- running and checking --------------------------------------------- #
    def run(self, program: Program, args: Sequence[Any]) -> Any:
        """Call ``program``. Raises :class:`CodeError`; callers that search use :meth:`check`."""
        if len(args) != len(program.params):
            raise CodeError(f"{program.name} takes {len(program.params)} argument(s)")
        env = dict(zip(program.params, args))
        return self.interpreter.run(program.body, env)

    def check(self, program: Program, examples: Sequence[Example]) -> Check:
        """Run every example and count what happened. The first failure is kept for the report."""
        result = Check()
        for example in examples:
            try:
                got = self.run(program, example.args)
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
        for example in examples:
            try:
                got = self.run(program, example.args)
            except CodeError:
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
        try:
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
              graft: Optional[bool] = None) -> Written:
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
        for schema in self._ranked(spec.arity):
            for candidate, fillings in self._candidates(schema, spec, pools):
                if tried >= budget:
                    return self._abstain(spec, tried, started, "attempt budget exhausted")
                tried += 1
                if self.matches(candidate, spec.shown):
                    return self._won(spec, candidate, schema, tried, started,
                                     self.check(candidate, spec.shown), False)
        if allow_graft:
            for outer, inner in self._grafts(spec.arity):
                composite = _graft(outer, inner)
                if composite is None:
                    continue
                for candidate, fillings in self._candidates(composite, spec, pools):
                    if tried >= budget:
                        return self._abstain(spec, tried, started, "attempt budget exhausted")
                    tried += 1
                    if self.matches(candidate, spec.shown):
                        return self._won(spec, candidate, composite, tried, started,
                                         self.check(candidate, spec.shown), True)
        return self._abstain(spec, tried, started, "no shape she holds fits this task")

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
        return sorted(usable, key=lambda s: (s.cost, s.seed, -round(s.posterior, 6), s.key))

    def _grafts(self, arity: int) -> Iterator[Tuple[Schema, Schema]]:
        outers = [s for s in self._ranked(arity) if s.cost <= 2]
        inners = [s for s in self.schemas.values()
                  if s.arity == 1 and s.cost <= 1 and not isinstance(s.skeleton, Var)]
        inners.sort(key=lambda s: (s.seed, -round(s.posterior, 6), s.cost, s.key))
        for outer in outers:
            for inner in inners:
                yield outer, inner

    def _candidates(self, schema: Schema, spec: Spec,
                    pools: Dict[str, List[Term]]) -> Iterator[Tuple[Program, Tuple[Term, ...]]]:
        """Every filling of ``schema``'s holes, as a runnable program."""
        if schema.arity != spec.arity:
            return
        try:
            choices = [pools.get(hole.kind, pools["any"]) for hole in schema.holes]
        except KeyError:  # pragma: no cover - pools always carry "any"
            return
        if any(not choice for choice in choices):
            return
        for fillings in itertools.product(*choices):
            try:
                body = schema.instantiate(spec.params, fillings)
            except CodeError:
                return
            yield Program(name=spec.name or "f", params=spec.params, body=body), fillings

    def _pools(self, spec: Spec) -> Dict[str, List[Term]]:
        """What may be dropped into a hole, drawn from the task rather than from thin air."""
        ints, texts = spec.constants()
        sample = spec.shown[0].args if spec.shown else ()
        kinds = [_kind_of(value) for value in sample]
        by_kind: Dict[str, List[Term]] = {"int": [], "str": [], "list": [], "bool": [], "any": []}
        for index, kind in enumerate(kinds):
            if index < len(spec.params):
                by_kind.setdefault(kind, []).append(Var(spec.params[index]))

        int_lits = [Lit(v) for v in dict.fromkeys(_pick_constants(ints, 8) + [0, -1])][:14]
        str_lits = [Lit(v) for v in dict.fromkeys(list(texts) + [" ", ""])][:8]
        pools: Dict[str, List[Term]] = {
            "int": by_kind.get("int", []) + int_lits,
            "str": by_kind.get("str", []) + str_lits,
            "list": by_kind.get("list", []) + by_kind.get("str", []),
            "bool": [Lit(True), Lit(False)],
            "fn1": list(_fn1_library(ints)),
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


def _positions(term: Term, path: Tuple[int, ...] = ()) -> Iterator[Tuple[Tuple[int, ...], Term]]:
    yield path, term
    if isinstance(term, Call):
        for index, arg in enumerate(term.args):
            yield from _positions(arg, path + (index,))
    elif isinstance(term, Lambda):
        yield from _positions(term.body, path + (-1,))


def _replace_at(term: Term, path: Tuple[int, ...], new: Term) -> Term:
    if not path:
        return new
    head, rest = path[0], path[1:]
    if head == -1 and isinstance(term, Lambda):
        return Lambda(term.params, _replace_at(term.body, rest, new))
    if isinstance(term, Call):
        args = list(term.args)
        args[head] = _replace_at(args[head], rest, new)
        return Call(term.op, tuple(args))
    return term


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

_BINOPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.FloorDiv: "div",
           ast.Mod: "mod"}
_CMPOPS = {ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt",
           ast.GtE: "ge"}
_CALLS = {"len": "len", "sum": "sum", "abs": "abs", "sorted": "sort", "reversed": "rev"}
_METHODS = {"upper": "upper", "lower": "lower", "split": "words", "join": "join",
            "count": "count"}


def read_python(source: str, *, name: str = "f",
                params: Optional[Sequence[str]] = None) -> Program:
    """Turn a small piece of Python into a :class:`Program`, or refuse.

    Accepted: one ``def`` whose body is a ``return`` (a docstring above it is fine), a bare
    ``lambda``, or a bare expression when ``params`` is given. Inside it: arithmetic, comparison,
    ``and``/``or``/``not``, a conditional expression, slicing and indexing, list comprehensions
    with at most one ``if``, and a fixed list of builtins and string methods.

    **Everything else raises** :class:`CodeError` — imports, attribute access outside
    :data:`_METHODS`, assignment, loops, ``lambda`` with defaults, f-strings, starred arguments,
    keyword arguments. This is the same stance as :mod:`nyxara.njp.calculate`: the accepted set is
    enumerated rather than the rejected one, so a future Python grammar cannot quietly widen what
    she will read, and nothing that gets through can do anything but arithmetic on values, because
    the thing it becomes is a :class:`Term` and the only machine that runs a Term is
    :class:`Interpreter`.
    """
    try:
        tree = ast.parse(source.strip())
    except SyntaxError as exc:
        raise CodeError(f"not parseable: {exc}") from exc
    if len(tree.body) != 1:
        raise CodeError("expected exactly one definition or expression")
    node = tree.body[0]
    if isinstance(node, ast.FunctionDef):
        body = [stmt for stmt in node.body
                if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str))]
        if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
            raise CodeError("the function must be a single return statement")
        names = _argnames(node.args)
        return Program(name=node.name, params=names, body=_from_ast(body[0].value))
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Lambda):
        lam = node.value
        names = _argnames(lam.args)
        return Program(name=name, params=names, body=_from_ast(lam.body))
    if isinstance(node, ast.Expr):
        if params is None:
            raise CodeError("a bare expression needs its parameter names")
        return Program(name=name, params=tuple(params), body=_from_ast(node.value))
    raise CodeError(f"cannot read a {type(node).__name__}")


def _argnames(args: ast.arguments) -> Tuple[str, ...]:
    if args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg or args.defaults:
        raise CodeError("only plain positional parameters are read")
    return tuple(a.arg for a in args.args)


def _from_ast(node: ast.AST) -> Term:  # noqa: C901 - a whitelist is a long if-chain by nature
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (bool, int, str)):
            return Lit(node.value)
        raise CodeError(f"literal of type {type(node.value).__name__} is not in the language")
    if isinstance(node, ast.Name):
        return Var(node.id)
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_from_ast(e) for e in node.elts]
        if not all(isinstance(i, Lit) for i in items):
            raise CodeError("only literal lists are read")
        return Lit(tuple(i.value for i in items))  # type: ignore[union-attr]
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
        if len(node.ops) != 1:
            raise CodeError("chained comparisons are not read")
        left, right = _from_ast(node.left), _from_ast(node.comparators[0])
        if isinstance(node.ops[0], ast.In):
            return Call("has", (right, left))
        if isinstance(node.ops[0], ast.NotIn):
            return Call("not", (Call("has", (right, left)),))
        op = _CMPOPS.get(type(node.ops[0]))
        if op is None:
            raise CodeError(f"comparison {type(node.ops[0]).__name__} is not in the language")
        return Call(op, (left, right))
    if isinstance(node, ast.IfExp):
        return Call("if", (_from_ast(node.test), _from_ast(node.body),
                           _from_ast(node.orelse)))
    if isinstance(node, ast.Subscript):
        return _subscript(node)
    if isinstance(node, ast.Call):
        return _call(node)
    if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
        return _comprehension(node)
    raise CodeError(f"{type(node).__name__} is not in the language")


def _subscript(node: ast.Subscript) -> Term:
    value = _from_ast(node.value)
    index = node.slice
    if isinstance(index, ast.Slice):
        if index.step is not None:
            step = _from_ast(index.step)
            if isinstance(step, Call) and step.op == "neg" and step.args == (Lit(1),):
                if index.lower is None and index.upper is None:
                    return Call("rev", (value,))
            raise CodeError("only [::-1] is read as a step")
        if index.lower is not None and index.upper is None:
            low = _from_ast(index.lower)
            if low == Lit(1):
                return Call("tail", (value,))
            return Call("drop", (value, low))
        if index.lower is None and index.upper is not None:
            return Call("take", (value, _from_ast(index.upper)))
        raise CodeError("that slice is not read")
    return Call("at", (value, _from_ast(index)))


def _call(node: ast.Call) -> Term:
    if node.keywords:
        raise CodeError("keyword arguments are not read")
    args = [_from_ast(a) for a in node.args]
    if isinstance(node.func, ast.Attribute):
        op = _METHODS.get(node.func.attr)
        if op is None:
            raise CodeError(f"method {node.func.attr!r} is not in the language")
        receiver = _from_ast(node.func.value)
        if op in ("upper", "lower", "words"):
            if args:
                raise CodeError(f"{node.func.attr}() takes no arguments here")
            return Call(op, (receiver,))
        if len(args) != 1:
            raise CodeError(f"{node.func.attr}() takes one argument here")
        return Call(op, (receiver, args[0]))
    if not isinstance(node.func, ast.Name):
        raise CodeError("only named calls are read")
    fname = node.func.id
    if fname in ("list", "tuple", "int"):
        if len(args) != 1:
            raise CodeError(f"{fname}() takes one argument here")
        return args[0]
    if fname in ("max", "min"):
        if len(args) == 1:
            return Call("maxof" if fname == "max" else "minof", (args[0],))
        if len(args) == 2:
            return Call("max2" if fname == "max" else "min2", tuple(args))
        raise CodeError(f"{fname}() takes one or two arguments here")
    if fname == "range":
        if len(args) == 1:
            return Call("range", (args[0],))
        if len(args) == 2:
            return Call("range2", tuple(args))
        raise CodeError("range() takes one or two arguments here")
    op = _CALLS.get(fname)
    if op is None:
        raise CodeError(f"function {fname!r} is not in the language")
    if len(args) != 1:
        raise CodeError(f"{fname}() takes one argument here")
    return Call(op, (args[0],))


def _comprehension(node: Any) -> Term:
    if len(node.generators) != 1:
        raise CodeError("only a single-generator comprehension is read")
    gen = node.generators[0]
    if gen.is_async or len(gen.ifs) > 1:
        raise CodeError("that comprehension is not read")
    if not isinstance(gen.target, ast.Name):
        raise CodeError("only a plain loop variable is read")
    name = gen.target.id
    stream = _from_ast(gen.iter)
    if gen.ifs:
        stream = Call("filter", (Lambda((name,), _from_ast(gen.ifs[0])), stream))
    element = _from_ast(node.elt)
    if element == Var(name):
        return stream
    return Call("map", (Lambda((name,), element), stream))
