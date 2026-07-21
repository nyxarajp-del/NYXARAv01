"""NYXARA · growth/noesis.py — the living algorithm (🌱, Rule 4).

The other growth engines each rewrite **one** kind of thing over a **fixed** primitive set:
:mod:`~nyxara.growth.rule_synth` invents weight-update rules, :mod:`~nyxara.growth.cognitive_architect`
invents reasoning operators, :mod:`~nyxara.growth.godel_loop` climbs to stronger logics,
:mod:`~nyxara.growth.eureka` keeps proven lemmas, :mod:`~nyxara.growth.open_world` fits laws by MDL.
None of them is a single, cross-task, **self-extending abstraction library** whose growth makes future
problems *cheaper* to solve. Noēsis is that missing core.

The theory is the honest one — **compression = capability** (Solomonoff–Levin universal search +
DreamCoder-style library learning). Intelligence-per-compute grows when a system **amortises the
structure it discovers** into reusable concepts, so later problems need *shorter* programs and *less*
search. Noēsis makes that a loop she runs herself, **in code, with no LLM anywhere**:

* **WAKE** — solve a task ``input → output`` by bounded, type-directed search for the **shortest
  verified program** in a small typed DSL. The verifier decides (exact I/O match); when nothing
  verifies in budget she **abstains** — she never bluffs a program she did not check.
* **SLEEP** — mine the recurring structure across every solved program (anti-unification), and **adopt
  the abstraction that strictly lowers the total description length of the corpus on a held-out split**
  (MDL, anti-overfit). An adopted abstraction becomes a **new first-class primitive** — the language
  she thinks in literally grows.
* **DREAM** — invent her own new tasks so the growth is open-ended, not a fixed benchmark.

She upgrades the algorithm on three levels — the **library** (data, grows and persists), the
**search-guide prior** (params, learned from her own solves), and the abstraction **operators** (meta).
Nothing is a frozen constant. And it is measured, never asserted: as the library grows, average
solution length falls and solve-rate rises on a held-out battery — that inequality *is* "more
capability per unit compute."

Safety mirrors the rest of growth: verifier-gated (a kept program passed exact checks), character-
locked (no abstraction may reference the immutable value/character core), bounded (hard caps on search
depth / breadth / wall-clock so it is real but never runs away), pure standard library, and it touches
no source, no weights, no gate — it grows a *library of verified programs*, nothing NYXARA *is* or
*does*. Persists the whole library so the gain compounds across restarts.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Type",
    "Prog",
    "Primitive",
    "Abstraction",
    "Library",
    "Task",
    "SynthResult",
    "NoesisReport",
    "NoesisEngine",
    "base_library",
]

# --------------------------------------------------------------------------- #
# Types + evaluation guards
# --------------------------------------------------------------------------- #
INT = "int"
BOOL = "bool"
INTLIST = "intlist"
TYPES: Tuple[str, ...] = (INT, BOOL, INTLIST)
Type = str

# Character-locked: an adopted abstraction's name may never carry a value/character token. A new
# concept may extend what she can *compute*, never what she *is* (kept in lock-step with the
# ``_FORBIDDEN`` sets in godel_loop.py / meta_engine.py).
_FORBIDDEN = frozenset({
    "loyalty", "obedience", "corrigibility", "owner_safety", "honesty",
    "safety", "oversight", "guard", "master", "ethics", "kill_switch", "value",
})

_MAX_INT = 10 ** 6          # clamp integer magnitude — a program can never blow up the machine
_MAX_LEN = 64               # clamp list length
_MAX_STEPS = 4000           # per-evaluation interpreter step budget


class _Budget:
    """A mutable step counter that makes every evaluation hard-bounded (never runs away)."""

    __slots__ = ("steps",)

    def __init__(self) -> None:
        self.steps = 0

    def tick(self) -> None:
        self.steps += 1
        if self.steps > _MAX_STEPS:
            raise _Overrun()


class _Overrun(Exception):
    """Raised internally when an evaluation exceeds its step budget — caught, never surfaced."""


def _clamp_int(v: Any) -> Optional[int]:
    if isinstance(v, bool):  # bool is a subclass of int — keep them distinct
        return None
    if not isinstance(v, int):
        return None
    if v > _MAX_INT or v < -_MAX_INT:
        return None
    return v


def _clamp_list(v: Any) -> Optional[List[int]]:
    if not isinstance(v, list) or len(v) > _MAX_LEN:
        return None
    out: List[int] = []
    for x in v:
        c = _clamp_int(x)
        if c is None:
            return None
        out.append(c)
    return out


# --------------------------------------------------------------------------- #
# Prog — a typed program tree
# --------------------------------------------------------------------------- #
@dataclass
class Prog:
    """One node of a typed program tree.

    ``kind`` ∈ {``lit``, ``var``, ``app``, ``hole``}:
      * ``lit``  — a literal (``value`` holds the int/bool), ``rtype`` its type.
      * ``var``  — the single task input ``x`` (``rtype`` its declared type).
      * ``app``  — apply the primitive/abstraction named ``name`` to ``args``.
      * ``hole`` — an abstraction parameter slot (``value`` is the 0-based param index).
    """

    kind: str
    rtype: Type
    name: str = ""
    value: Any = None
    args: Tuple["Prog", ...] = ()

    def size(self) -> int:
        return 1 + sum(a.size() for a in self.args)

    def leaves_holes(self) -> int:
        if self.kind == "hole":
            return 1
        return sum(a.leaves_holes() for a in self.args)

    def has_app(self) -> bool:
        if self.kind == "app":
            return True
        return any(a.has_app() for a in self.args)

    def equal(self, other: "Prog") -> bool:
        return (self.kind == other.kind and self.name == other.name
                and self.value == other.value and self.rtype == other.rtype
                and len(self.args) == len(other.args)
                and all(a.equal(b) for a, b in zip(self.args, other.args)))

    def pretty(self) -> str:
        if self.kind == "lit":
            return json.dumps(self.value)
        if self.kind == "var":
            return "x"
        if self.kind == "hole":
            return f"${self.value}"
        if not self.args:
            return self.name
        return f"{self.name}(" + ", ".join(a.pretty() for a in self.args) + ")"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind, "rtype": self.rtype}
        if self.name:
            d["name"] = self.name
        if self.value is not None:
            d["value"] = self.value
        if self.args:
            d["args"] = [a.to_dict() for a in self.args]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Prog":
        return cls(
            kind=str(d["kind"]),
            rtype=str(d["rtype"]),
            name=str(d.get("name", "")),
            value=d.get("value"),
            args=tuple(cls.from_dict(a) for a in d.get("args", ())),
        )


def lit(value: Any, rtype: Type) -> Prog:
    return Prog("lit", rtype, value=value)


def var(rtype: Type) -> Prog:
    return Prog("var", rtype, name="x")


def hole(index: int, rtype: Type) -> Prog:
    return Prog("hole", rtype, value=index)


def app(name: str, rtype: Type, args: Sequence[Prog]) -> Prog:
    return Prog("app", rtype, name=name, args=tuple(args))


# --------------------------------------------------------------------------- #
# Primitives + Abstractions
# --------------------------------------------------------------------------- #
@dataclass
class Primitive:
    """A base DSL primitive: a typed, guarded, deterministic function."""

    name: str
    arg_types: Tuple[Type, ...]
    ret_type: Type
    impl: Callable[..., Any]

    @property
    def arity(self) -> int:
        return len(self.arg_types)


@dataclass
class Abstraction:
    """A learned, reusable sub-program with typed holes — a NEW first-class primitive.

    ``body`` is a :class:`Prog` tree whose ``hole`` nodes are the parameters (index 0..k-1). Evaluating
    an ``app`` of this abstraction substitutes the argument programs for the holes, then evaluates the
    body — so an adopted abstraction is genuinely part of the language, not a macro comment.
    """

    name: str
    arg_types: Tuple[Type, ...]
    ret_type: Type
    body: Prog

    @property
    def arity(self) -> int:
        return len(self.arg_types)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "arg_types": list(self.arg_types),
                "ret_type": self.ret_type, "body": self.body.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Abstraction":
        return cls(name=str(d["name"]), arg_types=tuple(d["arg_types"]),
                   ret_type=str(d["ret_type"]), body=Prog.from_dict(d["body"]))


def _substitute(body: Prog, fillers: Sequence[Prog]) -> Prog:
    """Return ``body`` with each hole replaced by the corresponding filler program."""
    if body.kind == "hole":
        return fillers[int(body.value)]
    if not body.args:
        return body
    return Prog(body.kind, body.rtype, name=body.name, value=body.value,
                args=tuple(_substitute(a, fillers) for a in body.args))


# --------------------------------------------------------------------------- #
# The base library (~24 first-order primitives over int / bool / intlist)
# --------------------------------------------------------------------------- #
def _safe_div(a: int, b: int) -> Optional[int]:
    return a // b if b != 0 else None


def _safe_mod(a: int, b: int) -> Optional[int]:
    return a % b if b != 0 else None


def base_library() -> "Library":
    """Build a fresh base library: the fixed floor of primitives she starts thinking in."""
    P = Primitive
    prims = [
        P("add", (INT, INT), INT, lambda a, b: a + b),
        P("sub", (INT, INT), INT, lambda a, b: a - b),
        P("mul", (INT, INT), INT, lambda a, b: a * b),
        P("idiv", (INT, INT), INT, _safe_div),
        P("mod", (INT, INT), INT, _safe_mod),
        P("eq", (INT, INT), BOOL, lambda a, b: a == b),
        P("lt", (INT, INT), BOOL, lambda a, b: a < b),
        P("and_", (BOOL, BOOL), BOOL, lambda a, b: a and b),
        P("or_", (BOOL, BOOL), BOOL, lambda a, b: a or b),
        P("not_", (BOOL,), BOOL, lambda a: not a),
        P("ifz", (BOOL, INT, INT), INT, lambda c, a, b: a if c else b),
        P("sum_", (INTLIST,), INT, lambda xs: sum(xs)),
        P("length", (INTLIST,), INT, lambda xs: len(xs)),
        P("maximum", (INTLIST,), INT, lambda xs: max(xs) if xs else None),
        P("minimum", (INTLIST,), INT, lambda xs: min(xs) if xs else None),
        P("head", (INTLIST,), INT, lambda xs: xs[0] if xs else None),
        P("last", (INTLIST,), INT, lambda xs: xs[-1] if xs else None),
        P("reverse", (INTLIST,), INTLIST, lambda xs: list(reversed(xs))),
        P("sort_", (INTLIST,), INTLIST, lambda xs: sorted(xs)),
        P("tail", (INTLIST,), INTLIST, lambda xs: xs[1:]),
        P("cons", (INT, INTLIST), INTLIST, lambda a, xs: [a] + xs),
        P("rangei", (INT,), INTLIST, lambda n: list(range(n)) if 0 <= n <= _MAX_LEN else None),
        P("inc_each", (INTLIST, INT), INTLIST, lambda xs, k: [x + k for x in xs]),
        P("scale_each", (INTLIST, INT), INTLIST, lambda xs, k: [x * k for x in xs]),
        P("filter_gt", (INTLIST, INT), INTLIST, lambda xs, k: [x for x in xs if x > k]),
        P("count_gt", (INTLIST, INT), INT, lambda xs, k: sum(1 for x in xs if x > k)),
    ]
    return Library(primitives={p.name: p for p in prims})


@dataclass
class Library:
    """The growing set of primitives + adopted abstractions — and the learned usage prior.

    ``counts`` are Laplace-smoothed usage tallies per entry, so the description-length prior favours
    entries she has actually found useful (cheaper to search). ``dl(prog)`` is the single objective
    for both WAKE (minimise the solution's ``dl``) and SLEEP (minimise the corpus's total ``dl``).
    """

    primitives: Dict[str, Primitive] = field(default_factory=dict)
    abstractions: Dict[str, Abstraction] = field(default_factory=dict)
    counts: Dict[str, float] = field(default_factory=dict)
    archived: Dict[str, Abstraction] = field(default_factory=dict)  # pruned, reversibly (F3 REM)

    # ---- entry access ---- #
    def entry(self, name: str) -> Optional[Any]:
        return self.primitives.get(name) or self.abstractions.get(name)

    def arg_types(self, name: str) -> Tuple[Type, ...]:
        e = self.entry(name)
        return tuple(e.arg_types) if e is not None else ()

    def ret_type(self, name: str) -> Optional[Type]:
        e = self.entry(name)
        return e.ret_type if e is not None else None

    def callables_by_ret(self) -> Dict[Type, List[str]]:
        out: Dict[Type, List[str]] = {t: [] for t in TYPES}
        for name, e in list(self.primitives.items()) + list(self.abstractions.items()):
            out.setdefault(e.ret_type, []).append(name)
        return out

    # ---- description-length prior ---- #
    def _leaf_cost(self) -> float:
        # a literal/var is one "token" drawn from a modest alphabet
        return math.log2(1 + len(self.primitives) + len(self.abstractions) + 8)

    def node_cost(self, name: str) -> float:
        total = sum(self.counts.values()) + len(self.counts) + 1.0
        c = self.counts.get(name, 0.0) + 1.0
        return -math.log2(c / total)

    def dl(self, prog: Prog) -> float:
        if prog.kind in ("lit", "var", "hole"):
            return self._leaf_cost()
        return self.node_cost(prog.name) + sum(self.dl(a) for a in prog.args)

    def reinforce(self, prog: Prog, weight: float = 1.0) -> None:
        """Fold a solved program's usage into the prior so future search prefers what worked.

        ``weight`` is the F3 neuromodulated learning rate: a novel/surprising solve reinforces
        strongly, a habitual one barely at all (saving memory + compute).
        """
        if prog.kind == "app":
            self.counts[prog.name] = self.counts.get(prog.name, 0.0) + weight
        for a in prog.args:
            self.reinforce(a, weight)

    def uses(self, name: str, prog: Prog) -> int:
        """How many times ``name`` appears in ``prog`` (usage → utility for F3 pruning)."""
        c = 1 if (prog.kind == "app" and prog.name == name) else 0
        return c + sum(self.uses(name, a) for a in prog.args)

    def referenced_by_abstractions(self, name: str) -> bool:
        """True if any *other* kept abstraction's body uses ``name`` (never prune a dependency)."""
        return any(self.uses(name, a.body) > 0 for k, a in self.abstractions.items() if k != name)

    def prune_abstraction(self, name: str) -> bool:
        """Reversibly archive an abstraction (REM pruning). Base primitives are never prunable."""
        a = self.abstractions.pop(name, None)
        if a is None:
            return False
        self.archived[name] = a
        return True

    def restore_abstraction(self, name: str) -> bool:
        """Bring a pruned abstraction back — nothing learned is ever truly lost."""
        a = self.archived.pop(name, None)
        if a is None:
            return False
        self.abstractions[name] = a
        return True

    # ---- growth ---- #
    def add_abstraction(self, absn: Abstraction) -> None:
        if _FORBIDDEN.intersection(abs_name_tokens(absn.name)):
            raise ValueError(f"abstraction name '{absn.name}' touches the character core")
        self.abstractions[absn.name] = absn
        self.counts.setdefault(absn.name, 0.0)

    # ---- persistence ---- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "abstractions": [a.to_dict() for a in self.abstractions.values()],
            "archived": [a.to_dict() for a in self.archived.values()],
            "counts": dict(self.counts),
        }

    def load_dict(self, d: Dict[str, Any]) -> None:
        for ad in d.get("abstractions", ()):
            a = Abstraction.from_dict(ad)
            self.abstractions[a.name] = a
        for ad in d.get("archived", ()):
            a = Abstraction.from_dict(ad)
            self.archived[a.name] = a
        self.counts.update({str(k): float(v) for k, v in d.get("counts", {}).items()})


def abs_name_tokens(name: str) -> List[str]:
    return name.replace("-", "_").lower().split("_")


# --------------------------------------------------------------------------- #
# Interpreter — guarded, hard-bounded evaluation
# --------------------------------------------------------------------------- #
def _coerce(value: Any, rtype: Type) -> Any:
    if rtype == INT:
        return _clamp_int(value)
    if rtype == BOOL:
        return value if isinstance(value, bool) else None
    if rtype == INTLIST:
        return _clamp_list(value)
    return None


def _eval(prog: Prog, x: Any, lib: Library, budget: _Budget) -> Any:
    budget.tick()
    if prog.kind == "lit":
        return _coerce(prog.value, prog.rtype)
    if prog.kind == "var":
        return _coerce(x, prog.rtype)
    if prog.kind == "hole":  # only meaningful inside an abstraction body pre-substitution
        return None
    # app
    entry = lib.entry(prog.name)
    if entry is None:
        return None
    if isinstance(entry, Abstraction):
        body = _substitute(entry.body, prog.args)
        return _eval(body, x, lib, budget)
    # primitive
    argv = []
    for a in prog.args:
        v = _eval(a, x, lib, budget)
        if v is None:
            return None
        argv.append(v)
    try:
        out = entry.impl(*argv)
    except _Overrun:
        raise
    except Exception:  # noqa: BLE001 — any op fault fails the program closed, never the machine
        return None
    return _coerce(out, prog.rtype)


def evaluate(prog: Prog, x: Any, lib: Library) -> Any:
    """Evaluate ``prog`` on input ``x``; return the value or ``None`` on any fault/over-budget."""
    try:
        return _eval(prog, x, lib, _Budget())
    except _Overrun:
        return None


# --------------------------------------------------------------------------- #
# Task
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    """A specification: solve so that ``prog(input) == output`` on every example.

    ``oracle`` (optional) is the ground-truth program that generated the examples. It is **never**
    shown to the search — only the F5 red-team (``growth/redteam.py``) may consult it, to falsify a
    solution that merely fit the handful of examples but disagrees on adversarial inputs.
    """

    name: str
    input_type: Type
    ret_type: Type
    examples: Tuple[Tuple[Any, Any], ...]
    oracle: Optional["Prog"] = None

    def target(self) -> Tuple[Any, ...]:
        return tuple(out for _, out in self.examples)

    def inputs(self) -> Tuple[Any, ...]:
        return tuple(inp for inp, _ in self.examples)


# --------------------------------------------------------------------------- #
# WAKE — bottom-up, type-directed, observational-equivalence enumeration
# --------------------------------------------------------------------------- #
@dataclass
class SynthResult:
    program: Optional[Prog]
    solved: bool
    size: int
    dl: float
    expansions: int          # compute-per-solve proxy


_DEFAULT_LITS: Tuple[Tuple[Any, Type], ...] = (
    (0, INT), (1, INT), (2, INT), (3, INT), (5, INT), (-1, INT),
    (True, BOOL), (False, BOOL),
)


def _sig(prog: Prog, inputs: Sequence[Any], lib: Library) -> Optional[Tuple[Any, ...]]:
    out: List[Any] = []
    for x in inputs:
        v = evaluate(prog, x, lib)
        if v is None:
            return None
        out.append(tuple(v) if isinstance(v, list) else v)
    return tuple(out)


def synthesize(task: Task, lib: Library, *, max_rounds: int = 3, beam: int = 300,
               max_expansions: int = 60000, deadline: Optional[float] = None) -> SynthResult:
    """Search for the shortest verified program solving ``task``; abstain (``None``) if none in budget.

    Bottom-up enumeration: start from terminals, grow by applying every library entry to already-found
    programs, and prune by **observational equivalence** (keep only the smallest program per distinct
    output signature). The first program whose signature equals the target is returned — never a
    program that was not checked against every example.
    """
    inputs = task.inputs()
    target = task.target()
    target_norm = tuple(tuple(v) if isinstance(v, list) else v for v in target)

    # bank[type][signature] -> smallest Prog with that observable behaviour
    bank: Dict[Type, Dict[Tuple[Any, ...], Prog]] = {t: {} for t in TYPES}
    expansions = 0

    def consider(prog: Prog) -> Optional[Prog]:
        nonlocal expansions
        expansions += 1
        s = _sig(prog, inputs, lib)
        if s is None:
            return None
        d = bank[prog.rtype]
        cur = d.get(s)
        if cur is None or prog.size() < cur.size():
            d[s] = prog
        if prog.rtype == task.ret_type and s == target_norm:
            return prog
        return None

    # terminals: the input variable + the literal alphabet
    for t in (task.input_type,):
        hit = consider(var(t))
        if hit is not None:
            return SynthResult(hit, True, hit.size(), lib.dl(hit), expansions)
    for value, t in _DEFAULT_LITS:
        hit = consider(lit(value, t))
        if hit is not None:
            return SynthResult(hit, True, hit.size(), lib.dl(hit), expansions)

    # abstractions first: a learned concept usually yields the smaller solution, so trying it before
    # the base primitives lets bottom-up return the shorter program (the compounding shows up).
    names = list(lib.abstractions.keys()) + list(lib.primitives.keys())

    for _round in range(max_rounds):
        if deadline is not None and time.monotonic() > deadline:
            break
        # snapshot the current bank per type, keeping the cheapest `beam` programs
        pool: Dict[Type, List[Prog]] = {}
        for t in TYPES:
            progs = sorted(bank[t].values(), key=lambda p: (p.size(), lib.dl(p)))
            pool[t] = progs[:beam]
        grew = False
        for name in names:
            arg_types = lib.arg_types(name)
            ret = lib.ret_type(name)
            if ret is None:
                continue
            # cartesian product of argument candidates, bounded by beam and expansion cap
            choices = [pool.get(t, ()) for t in arg_types]
            if any(len(c) == 0 for c in choices):
                continue
            for combo in _bounded_product(choices, max_expansions - expansions):
                grew = True
                hit = consider(app(name, ret, combo))
                if hit is not None:
                    return SynthResult(hit, True, hit.size(), lib.dl(hit), expansions)
                if expansions >= max_expansions:
                    return SynthResult(None, False, 0, 0.0, expansions)
        if not grew:
            break

    return SynthResult(None, False, 0, 0.0, expansions)


def _bounded_product(choices: Sequence[Sequence[Prog]], remaining: int):
    """Yield argument tuples from the cartesian product, capped so a round can never run away."""
    if remaining <= 0:
        return
    total = 1
    for c in choices:
        total *= max(1, len(c))
    cap = min(total, max(0, remaining), 20000)
    if total <= cap:
        yield from _full_product(choices)
        return
    # too many: sample without replacement-ish by striding
    seen = 0
    idxs = [0] * len(choices)
    lens = [len(c) for c in choices]
    while seen < cap:
        yield tuple(choices[i][idxs[i]] for i in range(len(choices)))
        seen += 1
        # odometer increment
        j = 0
        while j < len(choices):
            idxs[j] += 1
            if idxs[j] < lens[j]:
                break
            idxs[j] = 0
            j += 1
        if j == len(choices):
            break


def _full_product(choices: Sequence[Sequence[Prog]]):
    if not choices:
        yield ()
        return
    head, rest = choices[0], choices[1:]
    for h in head:
        for tail in _full_product(rest):
            yield (h,) + tail


# --------------------------------------------------------------------------- #
# SLEEP — anti-unification + MDL abstraction (the compounding step)
# --------------------------------------------------------------------------- #
def _antiunify(p: Prog, q: Prog, holes: List[Tuple[Type, Prog, Prog]]) -> Optional[Prog]:
    """Least-general generalisation of two programs; differing sub-trees become typed holes."""
    if (p.kind == q.kind and p.name == q.name and p.value == q.value
            and p.rtype == q.rtype and len(p.args) == len(q.args)):
        if not p.args:
            return Prog(p.kind, p.rtype, name=p.name, value=p.value)
        new_args = []
        for pa, qa in zip(p.args, q.args):
            g = _antiunify(pa, qa, holes)
            if g is None:
                return None
            new_args.append(g)
        return Prog(p.kind, p.rtype, name=p.name, value=p.value, args=tuple(new_args))
    if p.rtype != q.rtype:
        return None
    idx = len(holes)
    holes.append((p.rtype, p, q))
    return hole(idx, p.rtype)


def _match(body: Prog, prog: Prog, binding: Dict[int, Prog]) -> bool:
    """True if ``prog`` is an instance of ``body``; fills ``binding[hole_index] -> subprogram``."""
    if body.kind == "hole":
        idx = int(body.value)
        if body.rtype != prog.rtype:
            return False
        if idx in binding:
            return binding[idx].equal(prog)
        binding[idx] = prog
        return True
    if (body.kind != prog.kind or body.name != prog.name or body.value != prog.value
            or body.rtype != prog.rtype or len(body.args) != len(prog.args)):
        return False
    return all(_match(ba, pa, binding) for ba, pa in zip(body.args, prog.args))


def _rewrite(prog: Prog, name: str, body: Prog, nholes: int) -> Prog:
    """Replace every occurrence of ``body`` in ``prog`` with an application of the abstraction."""
    binding: Dict[int, Prog] = {}
    if _match(body, prog, binding) and len(binding) == nholes:
        fillers = [_rewrite(binding[i], name, body, nholes) for i in range(nholes)]
        return app(name, body.rtype, fillers)
    if not prog.args:
        return prog
    return Prog(prog.kind, prog.rtype, name=prog.name, value=prog.value,
                args=tuple(_rewrite(a, name, body, nholes) for a in prog.args))


def _hole_types(body: Prog) -> Tuple[Type, ...]:
    found: Dict[int, Type] = {}

    def walk(p: Prog) -> None:
        if p.kind == "hole":
            found[int(p.value)] = p.rtype
        for a in p.args:
            walk(a)

    walk(body)
    return tuple(found[i] for i in range(len(found)))


def _candidate_bodies(corpus: Sequence[Prog]) -> List[Prog]:
    """Anti-unify every pair of solved programs → candidate abstraction bodies (≥1 app, 1..2 holes)."""
    out: List[Prog] = []
    seen: set = set()
    n = len(corpus)
    for i in range(n):
        for j in range(i + 1, n):
            holes: List[Tuple[Type, Prog, Prog]] = []
            g = _antiunify(corpus[i], corpus[j], holes)
            if g is None or g.kind == "hole":
                continue
            nh = g.leaves_holes()
            if nh < 1 or nh > 2 or not g.has_app():
                continue
            # holes must be indexed 0..nh-1 contiguously for a clean signature
            key = g.pretty()
            if key in seen:
                continue
            seen.add(key)
            out.append(g)
    return out


def _total_size(corpus: Sequence[Prog]) -> int:
    return sum(p.size() for p in corpus)


def _refactored_size(corpus: Sequence[Prog], name: str, body: Prog, nh: int) -> int:
    return sum(_rewrite(p, name, body, nh).size() for p in corpus)


@dataclass
class _AbsCandidate:
    body: Prog
    name: str
    arg_types: Tuple[Type, ...]
    score: int              # global description-length saved (>0 to be considered)


def sleep_abstract(corpus: Sequence[Prog], lib: Library, *, next_id: int = 0,
                   holdout_frac: float = 0.34, seed: int = 0) -> List[Abstraction]:
    """Adopt abstractions that STRICTLY lower total description length on a held-out split.

    Repeatedly: propose candidate bodies (anti-unification) on a train split, pick the one whose
    adoption most reduces total program size, and adopt it **only if** it also strictly reduces size
    on a disjoint held-out split (anti-overfit). Returns the abstractions adopted this pass.
    """
    corpus = [p for p in corpus if p.kind == "app"]
    adopted: List[Abstraction] = []
    if len(corpus) < 2:
        return adopted

    rnd = random.Random(seed)
    idx = list(range(len(corpus)))
    rnd.shuffle(idx)
    cut = max(1, int(len(corpus) * (1.0 - holdout_frac)))
    train = [corpus[i] for i in idx[:cut]]
    holdout = [corpus[i] for i in idx[cut:]] or train  # tiny corpus → confirm on train

    existing_bodies = [a.body for a in lib.abstractions.values()]
    work = list(corpus)
    for _ in range(8):  # bounded number of adoptions per pass
        best: Optional[_AbsCandidate] = None
        for body in _candidate_bodies(work):
            if any(body.equal(b) for b in existing_bodies):
                continue  # never adopt a concept she already holds
            nh = body.leaves_holes()
            arg_types = _hole_types(body)
            if len(arg_types) != nh:
                continue
            name = f"abs{next_id + len(adopted)}"
            # MDL: total corpus size shrinks by more than the one-time cost of storing the body.
            saved = _total_size(work) - _refactored_size(work, name, body, nh) - body.size()
            if saved <= 0:
                continue
            # anti-overfit: the abstraction must ALSO strictly shrink the held-out split — i.e. it
            # generalises beyond the pair that proposed it, not merely fits it.
            if _refactored_size(holdout, name, body, nh) >= _total_size(holdout):
                continue
            if best is None or saved > best.score:
                best = _AbsCandidate(body, name, arg_types, saved)
        if best is None:
            break
        absn = Abstraction(name=best.name, arg_types=best.arg_types,
                           ret_type=best.body.rtype, body=best.body)
        lib.add_abstraction(absn)
        adopted.append(absn)
        existing_bodies.append(best.body)
        # refactor the working corpus so later candidates build atop this one
        work = [_rewrite(p, absn.name, absn.body, absn.arity) for p in work]
        holdout = [_rewrite(p, absn.name, absn.body, absn.arity) for p in holdout]
    return adopted


# --------------------------------------------------------------------------- #
# DREAM — an open-ended task generator over the DSL
# --------------------------------------------------------------------------- #
def _random_list(rnd: random.Random) -> List[int]:
    n = rnd.randint(1, 6)
    return [rnd.randint(-5, 9) for _ in range(n)]


SYNTHETIC_FAMILIES: Tuple[str, ...] = (
    "sum_scaled", "sum_inc", "count_over", "max_scaled", "scaled_sorted",
)


def synthetic_tasks(rnd: random.Random, n: int, *, min_examples: int = 3,
                    family_weights: Optional[Dict[str, float]] = None) -> List[Task]:
    """Fantasise tasks from a family of hidden programs (list→int / list→list) she must rediscover.

    The families deliberately SHARE structure (e.g. ``sum(scale_each(x, k))`` across k) so SLEEP has
    real recurring structure to compress — the honest source of the compounding curve. When
    ``family_weights`` is given (the F2 epistemic scheduler), families are sampled in proportion to how
    little she yet understands them — curiosity toward her own weak spots.
    """
    lib = base_library()
    families: List[Tuple[str, Type, Callable[[int], Prog]]] = [
        ("sum_scaled", INT, lambda k: app("sum_", INT, [app("scale_each", INTLIST, [var(INTLIST), lit(k, INT)])])),
        ("sum_inc", INT, lambda k: app("sum_", INT, [app("inc_each", INTLIST, [var(INTLIST), lit(k, INT)])])),
        ("count_over", INT, lambda k: app("count_gt", INT, [var(INTLIST), lit(k, INT)])),
        ("max_scaled", INT, lambda k: app("maximum", INT, [app("scale_each", INTLIST, [var(INTLIST), lit(k, INT)])])),
        ("scaled_sorted", INTLIST, lambda k: app("sort_", INTLIST, [app("scale_each", INTLIST, [var(INTLIST), lit(k, INT)])])),
    ]
    names = [f[0] for f in families]
    tasks: List[Task] = []
    for _ in range(n):
        if family_weights:
            ws = [max(1e-6, family_weights.get(nm, 1.0)) for nm in names]
            fname = rnd.choices(names, weights=ws, k=1)[0]
            _, rtype, make = next(f for f in families if f[0] == fname)
        else:
            fname, rtype, make = rnd.choice(families)
        k = rnd.choice([2, 3, 4, 5])
        prog = make(k)
        target_ex = max(3, min_examples)
        examples: List[Tuple[Any, Any]] = []
        tries = 0
        while len(examples) < target_ex and tries < 60:
            tries += 1
            xs = _random_list(rnd)
            out = evaluate(prog, xs, lib)
            if out is not None:
                examples.append((xs, out))
        if len(examples) >= min(3, target_ex):
            tasks.append(Task(name=f"{fname}_{k}", input_type=INTLIST,
                              ret_type=rtype, examples=tuple(examples), oracle=prog))
    return tasks


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
@dataclass
class NoesisReport:
    """One honest cycle report — the compression curve, measured not asserted."""

    cycle: int
    tasks: int
    solved: int
    solve_rate: float
    avg_solution_size: float
    avg_expansions: float
    abstractions_adopted: int
    library_size: int
    corpus_size: int
    red_team_refuted: int = 0
    pruned: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle, "tasks": self.tasks, "solved": self.solved,
            "solve_rate": round(self.solve_rate, 4),
            "avg_solution_size": round(self.avg_solution_size, 4),
            "avg_expansions": round(self.avg_expansions, 2),
            "abstractions_adopted": self.abstractions_adopted,
            "red_team_refuted": self.red_team_refuted,
            "pruned": self.pruned,
            "library_size": self.library_size, "corpus_size": self.corpus_size,
        }


class NoesisEngine:
    """WAKE → SLEEP → DREAM: the living algorithm that grows its own library and compounds.

    Advisory / measurement by construction — it grows a persisted library of *verified programs* and
    reports the compression curve. It trains no weights, edits no source, and touches no gate.
    """

    def __init__(self, library: Optional[Library] = None, *, seed: int = 0,
                 tasks_per_cycle: int = 12, beam: int = 300, red_team: Any = None,
                 metacognition: Any = None, neuromod: Any = None, pruner: Any = None,
                 ledger: Any = None, prune_every: int = 3, curiosity: Any = None) -> None:
        self.library = library if library is not None else base_library()
        self._rnd = random.Random(seed)
        self.tasks_per_cycle = tasks_per_cycle
        self.beam = beam
        # F5 — the adversarial critic (duck-typed: any object with .survives(...)). When set, a WAKE
        # solution enters the corpus only if it survives the boundary-condition battery.
        self.red_team = red_team
        # F1 — the metacognitive reflective loop (duck-typed: .diagnose / .adapt / .geti). When set,
        # it diagnoses each outcome and retunes her bounded search knobs between cycles.
        self.metacognition = metacognition
        # F3 — neuromodulated plasticity (.weight(...)) + REM pruning (.prune(...)) over a utility
        # ledger (.record_cycle / .is_stale). All duck-typed and optional.
        self.neuromod = neuromod
        self.pruner = pruner
        self.ledger = ledger
        self.prune_every = max(1, prune_every)
        # F2 — epistemic curiosity (duck-typed: .observe(family, solved) / .weights(families)). When
        # set, DREAM samples task-families toward the ones she understands least.
        self.curiosity = curiosity
        self._seen: set = set()
        self._usage_prev: Dict[str, int] = {}
        self.corpus: List[Prog] = []
        self.cycles = 0
        self.history: List[NoesisReport] = []

    # ---- one full cycle ---- #
    def step(self, tasks: Optional[Sequence[Task]] = None) -> NoesisReport:
        self.cycles += 1
        if tasks is None:
            tasks = self.dream(self.tasks_per_cycle)
        solved = 0
        refuted = 0
        sizes: List[int] = []
        expansions: List[int] = []
        for task in tasks:
            res = self.wake(task)
            expansions.append(res.expansions)
            was_refuted = False
            if res.solved and res.program is not None:
                solved += 1
                # F5 gate: a solution that merely fit the examples must survive the adversarial
                # boundary battery (vs its oracle) before it is trusted into the permanent corpus.
                if self.red_team is not None:
                    verdict = self.red_team.survives(
                        res.program, task.input_type, self.library, oracle=task.oracle)
                    was_refuted = not verdict.survived
                if was_refuted:
                    refuted += 1
                else:
                    sizes.append(res.size)
                    self.corpus.append(res.program)
                    # F3 neuromodulation: novel + hard-won solves are written in fast; habitual
                    # ones barely nudge the prior (saving memory + compute).
                    weight = 1.0
                    if self.neuromod is not None:
                        key = res.program.pretty()
                        novelty = 0.0 if key in self._seen else 1.0
                        self._seen.add(key)
                        weight = self.neuromod.weight(novelty=novelty, expansions=res.expansions)
                    self.library.reinforce(res.program, weight)
            # F1: diagnose this outcome (feeds her calibrated solve/over-fit beliefs)
            if self.metacognition is not None:
                self.metacognition.diagnose(solved=res.solved, refuted=was_refuted)
            # F2: feed the outcome to epistemic curiosity, keyed by task family
            if self.curiosity is not None:
                family = task.name.rsplit("_", 1)[0]
                self.curiosity.observe(family, res.solved)
        adopted = self.sleep()
        # F1: retune bounded search knobs from the cycle's calibrated evidence
        if self.metacognition is not None:
            self.metacognition.adapt()
        # F3 REM pruning: fold this cycle's fresh abstraction-usage into the utility ledger, then
        # (periodically) prune concepts that stopped earning their keep — archived, reversibly.
        pruned = 0
        if self.pruner is not None and self.ledger is not None:
            usage_now = {name: sum(self.library.uses(name, p) for p in self.corpus)
                         for name in self.library.abstractions}
            delta = {n: max(0, usage_now.get(n, 0) - self._usage_prev.get(n, 0))
                     for n in usage_now}
            self._usage_prev = usage_now
            self.ledger.record_cycle(delta)
            if self.cycles % self.prune_every == 0:
                pruned = len(self.pruner.prune(self.library, self.ledger).pruned)
        n = len(tasks)
        rep = NoesisReport(
            cycle=self.cycles, tasks=n, solved=solved,
            solve_rate=solved / n if n else 0.0,
            avg_solution_size=sum(sizes) / len(sizes) if sizes else 0.0,
            avg_expansions=sum(expansions) / len(expansions) if expansions else 0.0,
            abstractions_adopted=len(adopted),
            library_size=len(self.library.primitives) + len(self.library.abstractions),
            corpus_size=len(self.corpus),
            red_team_refuted=refuted,
            pruned=pruned,
        )
        self.history.append(rep)
        return rep

    def wake(self, task: Task) -> SynthResult:
        if self.metacognition is not None:
            return synthesize(task, self.library,
                              beam=self.metacognition.geti("beam"),
                              max_rounds=self.metacognition.geti("max_rounds"),
                              max_expansions=self.metacognition.geti("max_expansions"))
        return synthesize(task, self.library, beam=self.beam)

    def sleep(self) -> List[Abstraction]:
        adopted = sleep_abstract(self.corpus, self.library,
                                 next_id=len(self.library.abstractions),
                                 seed=self._rnd.randint(0, 1 << 30))
        if adopted:
            # keep the corpus expressed in the richest current language
            for a in adopted:
                self.corpus = [_rewrite(p, a.name, a.body, a.arity) for p in self.corpus]
        return adopted

    def dream(self, n: int) -> List[Task]:
        min_ex = self.metacognition.geti("min_examples") if self.metacognition is not None else 3
        weights = self.curiosity.weights(SYNTHETIC_FAMILIES) if self.curiosity is not None else None
        return synthetic_tasks(self._rnd, n, min_examples=min_ex, family_weights=weights)

    def run(self, cycles: int, tasks: Optional[Sequence[Task]] = None) -> List[NoesisReport]:
        return [self.step(tasks) for _ in range(cycles)]

    # ---- honest status ---- #
    def report(self) -> Dict[str, Any]:
        last = self.history[-1] if self.history else None
        first = self.history[0] if self.history else None
        # the power number: how much shorter solutions have become as the library grew
        compression = 0.0
        if first and last and first.avg_solution_size > 0 and last.avg_solution_size > 0:
            compression = 1.0 - (last.avg_solution_size / first.avg_solution_size)
        return {
            "cycles": self.cycles,
            "library_size": len(self.library.primitives) + len(self.library.abstractions),
            "abstractions": [a.name + " := " + a.body.pretty()
                             for a in self.library.abstractions.values()],
            "corpus_size": len(self.corpus),
            "solve_rate": last.solve_rate if last else 0.0,
            "avg_solution_size": last.avg_solution_size if last else 0.0,
            "compression_gain": round(compression, 4),
            "curve": [r.to_dict() for r in self.history],
        }

    # ---- persistence (compounds across restarts) ---- #
    def to_dict(self) -> Dict[str, Any]:
        return {"library": self.library.to_dict(), "cycles": self.cycles,
                "corpus": [p.to_dict() for p in self.corpus]}

    def save(self, path: str) -> str:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)
        return path

    def load(self, path: str) -> bool:
        import os
        if not os.path.exists(path):
            return False
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:  # noqa: BLE001 — a corrupt snapshot must never block boot
            return False
        self.library.load_dict(d.get("library", {}))
        self.corpus = [Prog.from_dict(p) for p in d.get("corpus", ())]
        self.cycles = int(d.get("cycles", 0))
        return True


# --------------------------------------------------------------------------- #
# CLI / self-test
# --------------------------------------------------------------------------- #
def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="NYXARA Noēsis — the living algorithm (no LLM).")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args(list(argv) if argv is not None else None)

    eng = NoesisEngine(seed=args.seed, tasks_per_cycle=args.tasks)
    for rep in eng.run(args.cycles):
        print(json.dumps(rep.to_dict()))
    report = eng.report()
    print("\n=== Noēsis report ===")
    print(f"cycles              : {report['cycles']}")
    print(f"library size        : {report['library_size']}")
    print(f"abstractions learned: {len(report['abstractions'])}")
    for a in report["abstractions"]:
        print(f"  · {a}")
    print(f"final solve rate    : {report['solve_rate']:.2f}")
    print(f"compression gain    : {report['compression_gain']:.2f} "
          f"(1 − final/first avg solution size)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
