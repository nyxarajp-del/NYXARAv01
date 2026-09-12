"""NYXARA · njp/programming.py — what breaks a program, learned by breaking one (🐞).

:mod:`nyxara.njp.coding` lets her *write* a program. Asked anything **about** programming she had
nothing at all — measured, twelve questions, twelve empty answers:

    what is a for loop?            ->  ""
    what causes an IndexError?     ->  ""
    how do you fix an IndexError?  ->  ""

The first of those is a fact somebody could tell her and :mod:`nyxara.njp.encyclopedia` now does.
The other two are not. *"IndexError happens when the index is at least the length"* is not a
sentence she should be handed, because handing it over is the difference between a lookup table
and knowing something: a table has no idea why, cannot say what would fix it, and is silent the
moment the situation is one nobody wrote a row for.

So this organ does not read about errors. **It causes them.**

A :class:`Situation` is a real Python operation and real arguments — ``[1, 2, 3][5]``,
``{"a": 1}["b"]``, ``7 / 0``, ``int("x")``. :meth:`Programmer.run` performs it and records what
actually happened, which is an exception class Python raised or a value it returned. Nothing in
this file states which operations raise what; the interpreter does, at run time, every time.

**The probes are senses, not answers.** Before running, a situation is measured by a fixed set of
generic relational questions — what type is each argument, how long is it, is it zero, is one
argument less than another's length, is one a member of another's names. Not one of them mentions
an error. What connects ``arg1 < len(arg0)`` to ``IndexError`` is induced from trials where it held
and trials where it did not, and if the trials do not support a connection none is claimed.

**A law is a minimal exact conjunction.** For each outcome, the smallest set of probe readings true
of every trial that produced it and of no trial that did not. Smallest, so a law that mentions the
operation is only kept when dropping it breaks it — which is how ``IndexError`` comes out as a fact
about indices and lengths rather than a fact about the ``index`` operation. A conjunction with a
counterexample is not a law and is reported as a near miss rather than rounded up.

**A repair is found, not recalled.** Given a situation that failed, :meth:`Programmer.repair`
changes one argument at a time, **runs it again**, and keeps the change that made it succeed. What
it returns names the probe that flipped — *"make the index less than the length"* — because a fix
that cannot say which property it restored is a coincidence.

What she can then be asked, and answer from what she saw: what causes this error, what would fix
it, and whether *this* situation is about to fail. On a situation no law covers she says so.

Nothing is ``eval``\\ ed and nothing is ``exec``\\ ed. Every operation here is a call this module
makes itself with arguments this module built; no text becomes code at any point, which is the same
boundary :mod:`nyxara.njp.coding` draws and for the same reason.

Pure standard library.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Operation", "Situation", "Trial", "Law", "Repair", "Programmer",
    "OPERATIONS", "VALUES", "POOLS", "TRAITS", "RAN",
]

#: The outcome of a situation that did not raise. Not an error name, so it never collides with one.
RAN = "ran"

#: How many trials must support a law. Below this a conjunction that happens to be exact on two
#: trials is a coincidence rather than a regularity.
MIN_SUPPORT = 4

#: The widest conjunction searched. Three probes is already a law nobody would call simple, and
#: the search is exponential in this number.
MAX_TERMS = 3

#: The most laws one outcome may have, and the smallest share of it one law may explain. A cause
#: with thirty shapes is not a cause, and a "law" covering four of nine hundred failures is a
#: coincidence the greedy cover found rather than a regularity. Both were measured: without them
#: ``TypeError`` came back as thirty exact laws, two of which asserted a reading and its negation.
MAX_LAWS = 4
MIN_SHARE = 0.08


# --------------------------------------------------------------------------------------------- #
#  the world she can act in
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Operation:
    """One real Python operation, how many arguments it takes, and what it is *written for*.

    ``kinds`` names the pool each argument is normally drawn from. It is a statement of **intent**,
    which is a thing a programmer has and a thing this module is allowed to know: code is written
    to run. It says nothing whatever about what happens — ``index`` is written for a sequence and
    an integer, and whether ``xs[i]`` works is still entirely up to Python.

    Without it the world was one in which nearly every action was a type error: drawing both
    arguments of ``divide`` from a pool containing lists and dicts made ``TypeError`` 60% of
    everything, the greedy cover carved that mass into **thirty** exact-but-meaningless laws
    ("argument 0 is not negative and the operation is sort"), and the errors worth learning never
    surfaced. A learner in a world where every act fails learns nothing about failing.
    """

    name: str
    arity: int
    perform: Callable[..., Any]
    kinds: Tuple[str, ...] = ()
    #: What kind of thing this operation *is*, in terms an operation she has never performed could
    #: also be described in. Intent again, not consequence: ``lookup`` and ``attribute`` both read
    #: a name out of something, and saying so is what lets a law learned from one of them fire on
    #: the other. Without traits every law she induced named its own verb, and the transfer paper
    #: — trained having never performed an operation, then asked about nothing else — scored
    #: **0.095** on ``lookup``. A law about the ``index`` operation is not a law about indices.
    traits: Tuple[str, ...] = ()

    def __call__(self, *args: Any) -> Any:
        return self.perform(*args)


def _unpack_two(xs: Any) -> Any:
    a, b = xs                    # raises ValueError on the wrong length, TypeError on a non-iterable
    return (a, b)


def _next(xs: Any) -> Any:
    return next(iter(xs))        # StopIteration on empty


#: Every operation she can perform. Deliberately ordinary: these are the things a program does on
#: its data all day, and between them they raise most of what a beginner's traceback ever says.
#: The point is not the list's length — it is that **this module never says what any of them
#: does**. Every outcome in every trial is what Python did when it was asked.
OPERATIONS: Tuple[Operation, ...] = (
    Operation("index", 2, lambda xs, i: xs[i], ("sequence", "integer"),
              ("takes_position", "needs_container")),
    Operation("slice", 3, lambda xs, i, j: xs[i:j], ("sequence", "integer", "integer"),
              ("takes_range", "needs_container")),
    Operation("member", 2, lambda xs, v: v in xs, ("container", "atom"),
              ("tests_name", "needs_container")),
    Operation("lookup", 2, lambda m, k: m[k], ("mapping", "atom"),
              ("takes_name", "needs_container")),
    Operation("attribute", 2, lambda obj, name: getattr(obj, name), ("object", "text"),
              ("takes_name",)),
    Operation("divide", 2, lambda a, b: a / b, ("integer", "integer"),
              ("arithmetic", "splits_by")),
    Operation("modulo", 2, lambda a, b: a % b, ("integer", "integer"),
              ("arithmetic", "splits_by")),
    Operation("to_int", 1, int, ("text",), ("converts",)),
    Operation("length", 1, len, ("container",), ("measures", "needs_container")),
    Operation("add", 2, lambda a, b: a + b, ("integer", "integer"),
              ("arithmetic", "combines")),
    Operation("multiply", 2, lambda a, b: a * b, ("integer", "integer"),
              ("arithmetic", "combines")),
    Operation("sort", 1, sorted, ("sequence",), ("orders", "needs_container")),
    Operation("unpack_two", 1, _unpack_two, ("sequence",), ("spreads", "needs_container")),
    Operation("first", 1, _next, ("container",), ("iterates", "needs_container")),
    Operation("call", 1, lambda f: f(), ("callable",), ("invokes",)),
)

#: Every trait any operation declares. Probed on every situation, so a law can be about a kind of
#: operation rather than about one.
TRAITS: Tuple[str, ...] = tuple(sorted({t for o in OPERATIONS for t in o.traits}))


class _Thing:
    """An ordinary object, so ``attribute`` has something with a real namespace to look in."""

    def __init__(self) -> None:
        self.value = 1
        self.name = "thing"

    def __repr__(self) -> str:  # pragma: no cover - readability in reports
        return "<thing>"


#: The pool situations are built from. Chosen to *span* the probes — empty and non-empty, zero and
#: non-zero, negative and positive, digits and letters — and not to hit any particular error.
VALUES: Tuple[Any, ...] = (
    0, 1, 2, 5, -1, -3, 10,
    [], [1, 2, 3], [4, 5], ["a", "b", "c", "d"],
    (), (1, 2), (7, 8, 9),
    "", "abc", "12", "7", "hello",
    {}, {"a": 1}, {"a": 1, "b": 2},
    set(), {1, 2},
    None, True, _Thing(), len,
)

#: What each argument kind can be. The attribute names include ones a ``_Thing`` **has** and ones
#: it has not, because a learner shown only failures concludes that the operation always fails —
#: which is precisely what she concluded about ``getattr`` before this existed.
POOLS: Dict[str, Tuple[Any, ...]] = {
    "integer": (0, 1, 2, 5, -1, -3, 10),
    "sequence": ([], [1, 2, 3], [4, 5], ["a", "b", "c", "d"], (), (1, 2), (7, 8, 9),
                 "", "abc", "hello"),
    "mapping": ({}, {"a": 1}, {"a": 1, "b": 2}),
    "container": ([], [1, 2, 3], (), (1, 2), "", "abc", {}, {"a": 1}, set(), {1, 2}),
    "text": ("", "abc", "12", "7", "hello", "value", "name", "upper"),
    "atom": (0, 1, "a", "b", "abc", 2, "value"),
    "object": (_Thing(), "abc", [1, 2], 5, {"a": 1}),
    "callable": (len, _Thing, int),
    "anything": VALUES,
}


# --------------------------------------------------------------------------------------------- #
#  measuring a situation before running it
# --------------------------------------------------------------------------------------------- #
def _names_of(value: Any) -> Optional[Set[Any]]:
    """What counts as a *name in* this value: its keys, its elements, or its attributes.

    One probe covering three things that are the same question asked of three kinds of value, which
    is why ``KeyError`` and ``AttributeError`` can come out of the same reading rather than needing
    a rule each.
    """
    try:
        if isinstance(value, dict):
            return set(value.keys())
        if isinstance(value, (list, tuple, set, frozenset, str)):
            return set(value)
        return set(dir(value))
    except Exception:  # noqa: BLE001
        return None


def _size(value: Any) -> Optional[int]:
    try:
        return len(value)
    except Exception:  # noqa: BLE001
        return None


def probe(operation: Operation, args: Sequence[Any]) -> Dict[str, Any]:
    """Everything measurable about a situation, before anything is run.

    Generic by construction: each reading is a question that can be put to any situation, and none
    of them names an outcome. That separation is the whole claim of this module — the readings are
    what she can see, the laws are what she works out.
    """
    out: Dict[str, Any] = {"op": operation.name, "arity": operation.arity}
    for trait in TRAITS:
        out[f"op.{trait}"] = trait in operation.traits
    for i, value in enumerate(args):
        out[f"a{i}.type"] = type(value).__name__
        out[f"a{i}.none"] = value is None
        size = _size(value)
        out[f"a{i}.sized"] = size is not None
        if size is not None:
            out[f"a{i}.empty"] = size == 0
            out[f"a{i}.len"] = size
        if isinstance(value, int) and not isinstance(value, bool):
            out[f"a{i}.zero"] = value == 0
            out[f"a{i}.negative"] = value < 0
        if isinstance(value, str):
            out[f"a{i}.digits"] = value.isdigit()
        out[f"a{i}.callable"] = callable(value)
    for i, j in itertools.permutations(range(len(args)), 2):
        size = _size(args[j])
        if isinstance(args[i], int) and not isinstance(args[i], bool) and size is not None:
            out[f"a{i}<len(a{j})"] = args[i] < size
            out[f"a{i}>=-len(a{j})"] = args[i] >= -size
        names = _names_of(args[j])
        if names is not None:
            try:
                out[f"a{i} in names(a{j})"] = args[i] in names
            except Exception:  # noqa: BLE001 — an unhashable argument simply cannot be looked up
                pass
        out[f"type(a{i})==type(a{j})"] = type(args[i]) is type(args[j])
    return out


# --------------------------------------------------------------------------------------------- #
#  what happened
# --------------------------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Situation:
    """An operation and its arguments. Nothing about it says whether it works."""

    operation: Operation
    args: Tuple[Any, ...]

    def render(self) -> str:
        shown = ", ".join(repr(a) for a in self.args)
        return f"{self.operation.name}({shown})"


@dataclass
class Trial:
    """A situation, what she measured of it, and what Python actually did."""

    situation: Situation
    reading: Dict[str, Any] = field(default_factory=dict)
    outcome: str = RAN
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.outcome != RAN

    def to_dict(self) -> Dict[str, Any]:
        return {"situation": self.situation.render(), "outcome": self.outcome,
                "detail": self.detail}


@dataclass
class Law:
    """A minimal conjunction of readings that every trial with this outcome had, and no other."""

    outcome: str = ""
    terms: Tuple[Tuple[str, Any], ...] = ()
    support: int = 0
    #: Trials the conjunction covers that did **not** have this outcome. Zero for a law; a law with
    #: any is not one, and is reported as a near miss rather than rounded up.
    counterexamples: int = 0

    @property
    def exact(self) -> bool:
        return self.counterexamples == 0 and self.support >= MIN_SUPPORT

    @property
    def mentions_operation(self) -> bool:
        return any(name == "op" for name, _ in self.terms)

    @property
    def about_a_kind(self) -> bool:
        """It names what the operation *is* rather than which one it is — so it can transfer."""
        return (not self.mentions_operation
                and any(name.startswith("op.") for name, _ in self.terms))

    def holds(self, reading: Dict[str, Any]) -> bool:
        return all(reading.get(name, _MISSING) == value for name, value in self.terms)

    def render(self) -> str:
        return " and ".join(f"{name} is {value}" for name, value in self.terms) or "always"

    def english(self) -> str:
        return f"{self.outcome} happens when {_english(self.terms)}"

    def to_dict(self) -> Dict[str, Any]:
        return {"outcome": self.outcome, "when": self.render(), "support": self.support,
                "counterexamples": self.counterexamples, "exact": self.exact}


@dataclass
class Repair:
    """A change that was tried and worked, and the reading it restored."""

    argument: int = -1
    was: str = ""
    became: str = ""
    restored: str = ""
    outcome: str = ""

    def render(self) -> str:
        """Said as a requirement, because that is what a fix is.

        ``_english`` returns a reading as a statement — "argument 0 is not empty" — and gluing
        "make" to the front of it produced "make argument 0 is not empty". A fix names the state
        the situation has to be in.
        """
        change = f"argument {self.argument}: {self.was} -> {self.became}"
        if not self.restored:
            return f"give argument {self.argument} a different value ({change})"
        return f"see that {self.restored} ({change})"

    def to_dict(self) -> Dict[str, Any]:
        return {"argument": self.argument, "was": self.was, "became": self.became,
                "restored": self.restored, "fix": self.render()}


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"


_MISSING = _Missing()


#: How to say a reading, and how to say its negation. Two phrasings rather than a "does not" glued
#: to the front, which produced "argument 0 does not is empty".
_PLAIN: Dict[str, Tuple[str, str]] = {
    "zero": ("is zero", "is not zero"),
    "negative": ("is negative", "is not negative"),
    "empty": ("is empty", "is not empty"),
    "none": ("is None", "is not None"),
    "sized": ("has a length", "has no length"),
    "digits": ("is all digits", "is not all digits"),
    "callable": ("can be called", "cannot be called"),
}


def _english(terms: Sequence[Tuple[str, Any]]) -> str:
    """Say a conjunction of readings the way a person would. Presentation only — never a rule."""
    said: List[str] = []
    for name, value in terms:
        if name == "op":
            said.append(f"the operation is {value}")
            continue
        if name.startswith("op."):
            trait = name[3:].replace("_", " ")
            said.append(f"the operation {'' if value else 'does not '}{trait}")
            continue
        if "<len(" in name:
            left, right = name.split("<len(")
            right = right.rstrip(")")
            said.append(f"argument {left[1:]} is {'' if value else 'not '}"
                        f"less than the length of argument {right[1:]}")
            continue
        if " in names(" in name:
            left, right = name.split(" in names(")
            right = right.rstrip(")")
            said.append(f"argument {left[1:]} is {'' if value else 'not '}"
                        f"a name in argument {right[1:]}")
            continue
        if name.endswith(">=-len(a0)") or ">=-len(" in name:
            left = name.split(">=-len(")[0]
            said.append(f"argument {left[1:]} is {'' if value else 'not '}"
                        f"within reach counting backwards")
            continue
        if "==type(" in name:
            said.append(f"the two arguments {'do' if value else 'do not'} have the same type")
            continue
        if "." in name:
            slot, what = name.split(".", 1)
            if what == "type":
                said.append(f"argument {slot[1:]} is a {value}")
                continue
            if what == "len":
                said.append(f"argument {slot[1:]} has {value} item"
                            f"{'' if value == 1 else 's'}")
                continue
            yes, no = _PLAIN.get(what, (f"is {what}", f"is not {what}"))
            said.append(f"argument {slot[1:]} {yes if value else no}")
            continue
        said.append(f"{name} is {value}")
    return " and ".join(said)


# --------------------------------------------------------------------------------------------- #
#  the organ
# --------------------------------------------------------------------------------------------- #
class Programmer:
    """Performs operations, watches what happens, and works out what predicts what."""

    def __init__(self, *, seed: int = 0, min_support: int = MIN_SUPPORT,
                 max_terms: int = MAX_TERMS, learn: bool = True, slips: float = 0.25,
                 max_laws: int = MAX_LAWS, min_share: float = MIN_SHARE) -> None:
        self.random = random.Random(seed)
        self.min_support = int(min_support)
        self.max_terms = int(max_terms)
        self.slips = float(slips)
        self.max_laws = int(max_laws)
        self.min_share = float(min_share)
        #: Off, she runs and records but induces nothing — the control that says what the laws
        #: were worth, rather than what running the operations was worth.
        self.learning = bool(learn)
        self.trials: List[Trial] = []
        self.laws: List[Law] = []
        self.failure_laws: List[Law] = []
        self.near_misses: List[Law] = []

    # -- acting -------------------------------------------------------------- #
    def run(self, situation: Situation) -> Trial:
        """Perform it and record what happened. The only source of an outcome in this module."""
        trial = Trial(situation=situation)
        try:
            trial.reading = probe(situation.operation, situation.args)
        except Exception:  # noqa: BLE001
            trial.reading = {"op": situation.operation.name}
        try:
            situation.operation(*situation.args)
            trial.outcome = RAN
        except Exception as error:  # noqa: BLE001 — the exception *is* the observation
            trial.outcome = type(error).__name__
            trial.detail = str(error)[:120]
        return trial

    def situations(self, n: int, *, operations: Sequence[Operation] = OPERATIONS,
                   values: Optional[Sequence[Any]] = None) -> List[Situation]:
        """Mostly things written to work, and sometimes not.

        ``slips`` is the share of arguments drawn from the whole pool instead of the one the
        operation is written for — a programmer's mistakes, at a programmer's rate. Set it to 1.0
        and the world becomes the useless one described on :class:`Operation`; set it to 0.0 and
        she never sees a type error at all and cannot learn what one is.
        """
        out: List[Situation] = []
        for _ in range(int(n)):
            operation = self.random.choice(list(operations))
            args = []
            for position in range(operation.arity):
                if values is not None:
                    pool: Sequence[Any] = values
                elif self.random.random() < self.slips or not operation.kinds:
                    pool = POOLS["anything"]
                else:
                    pool = POOLS.get(operation.kinds[position], POOLS["anything"])
                args.append(self.random.choice(list(pool)))
            out.append(Situation(operation=operation, args=tuple(args)))
        return out

    def experiment(self, n: int = 400, **kwargs: Any) -> List[Trial]:
        """Do n things and watch. Adds to whatever she has already seen."""
        made = [self.run(s) for s in self.situations(n, **kwargs)]
        self.trials.extend(made)
        return made

    # -- learning ------------------------------------------------------------ #
    def learn(self) -> List[Law]:
        """Induce the smallest exact conjunctions that name each outcome. **Several per outcome.**

        Smallest matters and is not tidiness: a conjunction including ``op is index`` is a fact
        about an operation, and one that does not is a fact about indices and lengths. Searching
        one term before two before three finds the second whenever it exists, so a law generalises
        to operations she has not tried without anybody deciding that it should.

        *Several* matters more, and the first version of this got it wrong. One outcome can have
        genuinely different causes: ``xs[i]`` raises ``IndexError`` when the index is at or past
        the length **or** when it is below minus the length, and ``ValueError`` comes both from
        ``int("x")`` and from unpacking the wrong number of things. A search for one conjunction
        covering every case finds a reading they all share — and the only readings they all share
        are irrelevant ones, which is exactly what it found: ``ValueError`` came out as *"argument
        0 has a length"*, with 254 counterexamples.

        So the positives are covered **greedily**: the best exact conjunction is taken, the trials
        it explains are set aside, and the rest are induced over again. What comes out is a
        disjunction of conjunctions, which is what a cause with two shapes actually is.
        """
        self.laws, self.near_misses = [], []
        if not self.learning or not self.trials:
            return self.laws
        for outcome in sorted({t.outcome for t in self.trials}):
            if outcome == RAN:
                continue
            self._cover(outcome)
        self.laws.sort(key=lambda law: (-law.support, law.outcome))
        return self.laws

    def _cover(self, outcome: str) -> None:
        """Explain this outcome with as few laws as the evidence needs, and no fewer."""
        positives = [t for t in self.trials if t.outcome == outcome]
        negatives = [t for t in self.trials if t.outcome != outcome]
        remaining, found = list(positives), 0
        floor = max(self.min_support, int(len(positives) * self.min_share))
        while len(remaining) >= floor and found < self.max_laws:
            law = self._induce(remaining, negatives, floor=floor)
            if law is None or not law.exact:
                if law is not None:
                    self.near_misses.append(law)
                return
            self.laws.append(law)
            found += 1
            remaining = [t for t in remaining if not law.holds(t.reading)]

    def _induce(self, positives: Sequence[Trial], negatives: Sequence[Trial],
                *, floor: int = 0) -> Optional[Law]:
        """The smallest conjunction true of **some** of these positives and none of the negatives.

        Not of *all* of them: a cause with two shapes has no conjunction covering both, and
        demanding one is what produced a law about lengths for a failure about digits. The
        candidates are the readings of the largest agreeing group, and the search widens only when
        a narrower one leaves a counterexample.
        """
        if not positives:
            return None
        outcome = positives[0].outcome
        # Candidates come from **one** positive, not from what all of them agree on. That was the
        # second thing this got wrong, and it was invisible until the world stopped being all type
        # errors: ``ZeroDivisionError`` arises from ``divide`` and from ``modulo``, so ``op`` is
        # not a reading they share, so no conjunction mentioning it could ever be built and the
        # best available was "argument 1 is zero" — true of ``xs[0]`` and ``a * 0`` as well, and
        # wrong 35 times. Seeding from one trial lets a law name its own operation; the greedy
        # cover then comes back for the cases that law does not reach, which is where the second
        # operation's law comes from. Readings the whole group shares are tried first, because a
        # law that holds without naming an operation is the more general one and should win when
        # it exists.
        shared = self._shared(positives)
        seed = dict(positives[0].reading)
        for law in self._search(shared, positives, negatives, outcome, floor):
            return law
        return self._search_best(seed, positives, negatives, outcome, floor)
    def _search(self, candidates: Dict[str, Any], positives: Sequence[Trial],
                negatives: Sequence[Trial], outcome: str, floor: int) -> List[Law]:
        """The **widest-covering** exact conjunction of these readings, narrowest first.

        Not the first one found. At a given width several conjunctions can be exact and they are
        not equally good: ``argument 1 is zero and the operation is divide`` and ``argument 0 is
        not negative and argument 1 is zero and the operation is divide`` are both true of every
        division by zero she saw, and the second carries a term that has nothing to do with it —
        it survived only because the alphabet reached it first, and it cost a second law for the
        negative case. Taking the exact law that covers the most cases drops the passenger term,
        because a law without it covers strictly more.
        """
        for width in range(1, self.max_terms + 1):
            best: Optional[Law] = None
            for terms in itertools.combinations(sorted(candidates), width):
                picked = tuple((name, candidates[name]) for name in terms)
                law = Law(outcome=outcome, terms=picked)
                law.support = sum(1 for t in positives if law.holds(t.reading))
                if law.support < max(self.min_support, floor):
                    continue
                law.counterexamples = sum(1 for t in negatives if law.holds(t.reading))
                if law.counterexamples != 0:
                    continue
                # Ties are broken **towards the law that does not name its verb**. Both are exact
                # and only one of them can ever fire on an operation she has not performed.
                if best is None or (law.support, not law.mentions_operation) > (
                        best.support, not best.mentions_operation):
                    best = law
            if best is not None:
                return [best]
        return []

    def _search_best(self, candidates: Dict[str, Any], positives: Sequence[Trial],
                     negatives: Sequence[Trial], outcome: str, floor: int) -> Optional[Law]:
        """As :meth:`_search`, but returns the least-wrong conjunction when none is exact."""
        found = self._search(candidates, positives, negatives, outcome, floor)
        if found:
            return found[0]
        best: Optional[Law] = None
        for width in range(1, self.max_terms + 1):
            for terms in itertools.combinations(sorted(candidates), width):
                picked = tuple((name, candidates[name]) for name in terms)
                law = Law(outcome=outcome, terms=picked)
                law.support = sum(1 for t in positives if law.holds(t.reading))
                if law.support < max(self.min_support, floor):
                    continue
                law.counterexamples = sum(1 for t in negatives if law.holds(t.reading))
                if best is None or (law.counterexamples, -law.support) < (best.counterexamples,
                                                                          -best.support):
                    best = law
        return best

    @staticmethod
    def _shared(trials: Sequence[Trial]) -> Dict[str, Any]:
        """Readings every one of these trials agreed on. The only candidates a law can be built of."""
        if not trials:
            return {}
        shared = dict(trials[0].reading)
        for trial in trials[1:]:
            for name in list(shared):
                if trial.reading.get(name, _MISSING) != shared[name]:
                    del shared[name]
        return shared

    def learn_failure(self) -> List[Law]:
        """The coarser question: **will this fail?**, without saying what it will be called.

        Worth asking separately because the two generalise differently. *"Reading a name that is
        not there fails"* holds for a dict and for an object alike; what it is **called** does not
        — the same act raises ``KeyError`` on one and ``AttributeError`` on the other. So a learner
        that only ever predicts the name of the error has no way to be right about an operation it
        has never performed, and one that can also answer *whether* it breaks has.
        """
        self.failure_laws = []
        if not self.learning or not self.trials:
            return self.failure_laws
        positives = [t for t in self.trials if t.failed]
        negatives = [t for t in self.trials if not t.failed]
        remaining = list(positives)
        # A far smaller floor than the per-outcome cover uses, and for a plain reason: "fails"
        # has as many positives as every error put together, so a share of them is a large number
        # and no single conjunction covers that many failures without catching a success. At 8%
        # this induced **nothing at all**. Failure has many shapes; the cover is allowed to take
        # them a few at a time.
        floor = self.min_support
        while len(remaining) >= floor and len(self.failure_laws) < self.max_laws * 6:
            law = self._induce(remaining, negatives, floor=floor)
            if law is None or not law.exact:
                return self.failure_laws
            law.outcome = "fails"
            self.failure_laws.append(law)
            remaining = [t for t in remaining if not law.holds(t.reading)]
        return self.failure_laws

    def will_fail(self, situation: Situation) -> Tuple[Optional[bool], str]:
        """``(True, why)``, ``(False, why)`` or ``(None, why)`` — the coarse prediction."""
        try:
            reading = probe(situation.operation, situation.args)
        except Exception:  # noqa: BLE001
            return None, "the situation could not be measured"
        for law in self.failure_laws:
            if law.holds(reading):
                return True, law.english()
        if not self.failure_laws:
            return None, "she has induced nothing about failing"
        return False, "no law she has says this fails"

    # -- trying to kill what she learned ------------------------------------- #
    def challenge(self, *, tries: int = 400) -> List[Law]:
        """Hunt for a situation that satisfies a law and refutes it. Kill the ones that die.

        A law exact on every trial she has run can still be wrong about the world, and one of hers
        was: *"AttributeError happens when argument 1 is a str and the operation is attribute"* was
        exact — because in a thousand trials she had never once asked for an attribute that exists.
        No amount of the same sampling would have found that out. Being right on the data is not
        the same as being right, and the difference is only visible to something that goes looking.

        So each law is put to the test with situations built **to satisfy it**, drawn from the
        whole pool rather than from what happened to come up. A law that survives has been shot at;
        one that does not is moved to the near misses with its counterexample counted.
        """
        if not self.laws:
            return self.laws
        survivors: List[Law] = []
        for law in self.laws:
            killed = 0
            for situation in self._matching(law, tries=tries):
                trial = self.run(situation)
                self.trials.append(trial)
                if trial.outcome != law.outcome:
                    killed += 1
            if killed:
                law.counterexamples += killed
                self.near_misses.append(law)
                continue
            survivors.append(law)
        self.laws = survivors
        return self.laws

    def _matching(self, law: Law, *, tries: int) -> List[Situation]:
        """Situations whose readings satisfy the law — found by trying, not by construction."""
        wanted = dict(law.terms)
        operations = [o for o in OPERATIONS
                      if "op" not in wanted or o.name == wanted["op"]]
        out: List[Situation] = []
        for _ in range(int(tries)):
            operation = self.random.choice(operations)
            args = tuple(self.random.choice(list(VALUES)) for _ in range(operation.arity))
            situation = Situation(operation=operation, args=args)
            try:
                if law.holds(probe(operation, args)):
                    out.append(situation)
            except Exception:  # noqa: BLE001
                continue
        return out

    # -- using what she learned ---------------------------------------------- #
    def predict(self, situation: Situation) -> Tuple[str, str]:
        """What she expects to happen, and why. ``("unknown", reason)`` where nothing settles it."""
        try:
            reading = probe(situation.operation, situation.args)
        except Exception:  # noqa: BLE001
            return "unknown", "the situation could not be measured"
        firing = [law for law in self.laws if law.holds(reading)]
        if not firing:
            return RAN, "no law she has says this fails"
        outcomes = {law.outcome for law in firing}
        if len(outcomes) > 1:
            return "unknown", "two laws disagree: " + ", ".join(sorted(outcomes))
        law = max(firing, key=lambda one: one.support)
        return law.outcome, law.english()

    def repair(self, situation: Situation, *, values: Sequence[Any] = VALUES) -> Optional[Repair]:
        """Change one argument, run it again, and keep what worked.

        The search is over the same pool the situations were built from, and the result names the
        reading that flipped — so the answer is *"make the index less than the length"* rather than
        *"pass 1 instead of 5"*, which would be a coincidence dressed as an explanation.
        """
        before = self.run(situation)
        if not before.failed:
            return None
        for index in range(len(situation.args)):
            for candidate in values:
                if candidate is situation.args[index]:
                    continue
                args = list(situation.args)
                args[index] = candidate
                after = self.run(Situation(situation.operation, tuple(args)))
                if after.failed:
                    continue
                flipped = [name for name, value in after.reading.items()
                           if before.reading.get(name, _MISSING) != value
                           and name not in ("op", "arity")]
                restored = _english([(flipped[0], after.reading[flipped[0]])]) if flipped else ""
                return Repair(argument=index, was=repr(situation.args[index])[:40],
                              became=repr(candidate)[:40], restored=restored,
                              outcome=before.outcome)
        return None

    def causes(self, outcome: str) -> Optional[Law]:
        for law in self.laws:
            if law.outcome == outcome:
                return law
        return None

    def fix_for(self, outcome: str, *, tries: int = 60) -> Optional[Repair]:
        """A repair for the first situation she can produce that fails this way."""
        for situation in self.situations(int(tries)):
            trial = self.run(situation)
            if trial.outcome != outcome:
                continue
            repair = self.repair(situation)
            if repair is not None:
                return repair
        return None

    # -- reporting ----------------------------------------------------------- #
    def seen(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for trial in self.trials:
            counts[trial.outcome] = counts.get(trial.outcome, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def stats(self) -> Dict[str, Any]:
        return {"trials": len(self.trials), "outcomes": len(self.seen()),
                "laws": len(self.laws), "failure_laws": len(self.failure_laws),
                "near_misses": len(self.near_misses),
                "about_a_kind": sum(1 for law in self.laws if law.about_a_kind),
                "operation_free": sum(1 for law in self.laws if not law.mentions_operation),
                "seen": self.seen()}

    def render(self) -> str:
        rows = [f"{len(self.trials)} trials, {len(self.seen())} outcomes, "
                f"{len(self.laws)} laws"]
        for law in self.laws:
            rows.append(f"  {law.outcome:<18} {law.english()}   (support {law.support})")
        for law in self.near_misses:
            rows.append(f"  {law.outcome:<18} NEAR MISS: {law.render()} "
                        f"({law.counterexamples} counterexamples)")
        return "\n".join(rows)
