"""NYXARA · njp/probing.py — finding out what an operation *is*, by trying to break it (🧪, NJP V.89).

V.88 proved that the linear primitives contain nothing to discover, and pointed at where novelty
could still live: operations that sort, compare, rank, threshold. The obvious next move is to write
those down and search them — and that is the same mistake one level out. A hand-written list of
nonlinear operations is the supplied vocabulary again, just longer.

So this asks a different question. Given an operation as a **black box** — inputs in, outputs out,
no name, no source, no declared type — what can be established about it?

**Only refutations.** That is the whole shape of this module and it is not a limitation to be
apologised for. No number of probes shows that an operation is linear; each one only fails to show
that it is not. So a :class:`Finding` reads ``refuted`` with the counterexample that did it, or
``not refuted in 200 tries``, and there is no third value. An operation that survives every probe
has survived every probe, which is a different sentence from *it is linear* and stays different
however many probes are run.

**And the families are counted, not named.** Ten unlabelled operations go in; what comes out is a
partition — ``family 1``, ``family 2`` — built from which laws each one breaks. Two operations land
together because nothing distinguishes them behaviourally, not because somebody called them both
filters. :meth:`Family.christen` exists, nothing here calls it, and it refuses a family that has
not been separated from anything.

**What is supplied, said plainly.** The laws in :data:`LAWS` are supplied — additivity, homogeneity,
shift-equivariance, and six more. They are *properties*, not operations, and the partition they
induce is not handed over; but they are a vocabulary and pretending otherwise would be the promotion
this package keeps refusing. Discovering the laws themselves is the next debt, and it is not paid
here.

**The failure mode this is built against.** V.88 named it: *we could not express it in our algebra,
therefore it cannot exist*. Over-eager closure. So the accounting is three-way and permanent —
``invented`` for a real distinction found, ``flattered`` for two operations called different that
behave identically, ``buried`` for two called the same that genuinely differ. The third is the one
that grows as the proofs get stronger.

Pure standard library.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ["Law", "Finding", "Reading", "Family", "LAWS", "probe", "partition", "TRIES",
           "WIDTH", "TOLERANCE", "SEED"]

#: How many counterexamples are attempted per law. Reported with every finding, because *not
#: refuted* means nothing without it: an operation that survived three tries and one that survived
#: three hundred are not in the same state of knowledge.
TRIES = 200

#: How long the rows fed to an operation are.
WIDTH = 9

#: How far two outputs may differ before they are different. Loose enough to survive floating point
#: and tight enough that a real break is not absorbed.
TOLERANCE = 1e-9

SEED = 89


def _close(one: Sequence[float], two: Sequence[float], tol: float = TOLERANCE) -> bool:
    return len(one) == len(two) and all(abs(a - b) <= tol for a, b in zip(one, two))


@dataclass(frozen=True)
class Law:
    """A structural property, written as *a way of trying to break it*.

    ``attempt`` builds one trial and says whether the operation failed it. It never says the
    operation passed — only that this particular attempt did not succeed in breaking it, which is
    the only thing a probe can ever report.
    """

    name: str = ""
    #: ``(operation, rng) -> (broken, what was tried)``.
    attempt: Optional[Callable[[Callable[[Sequence[float]], Sequence[float]], random.Random],
                               Tuple[bool, str]]] = None
    says: str = ""


@dataclass(frozen=True)
class Finding:
    """One law, and whether anything managed to break it."""

    law: str = ""
    #: ``refuted`` | ``not refuted`` | ``could not run``
    stands: str = "could not run"
    tries: int = 0
    witness: str = ""

    @property
    def broken(self) -> bool:
        return self.stands == "refuted"

    @property
    def informative(self) -> bool:
        return self.stands != "could not run"

    def render(self) -> str:
        mark = {"refuted": " x ", "not refuted": " ~ ", "could not run": " ? "}
        tail = (f"broken by {self.witness}" if self.broken
                else f"not broken in {self.tries} tries — which is not the same as true")
        return f"  {mark.get(self.stands, '   ')} {self.law:<22} {tail}"


@dataclass
class Reading:
    """Everything a battery of probes established about one operation, which is a set of breaks."""

    name: str = ""
    findings: List[Finding] = field(default_factory=list)

    @property
    def signature(self) -> Tuple[str, ...]:
        """The laws it **broke**, in order. Two operations with the same signature are, as far as
        anything here can tell, the same kind of thing — and *as far as anything here can tell* is
        carried in the phrase rather than dropped from it."""
        return tuple(f.law for f in self.findings if f.broken)

    @property
    def unrunnable(self) -> List[str]:
        return [f.law for f in self.findings if not f.informative]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "signature": list(self.signature),
                "unrunnable": self.unrunnable,
                "findings": [{"law": f.law, "stands": f.stands, "tries": f.tries,
                              "witness": f.witness} for f in self.findings]}

    def render(self) -> str:
        return "\n".join([f"{self.name}:"] + [f.render() for f in self.findings])


@dataclass
class Family:
    """A group of operations nothing here could tell apart. Numbered, never named."""

    number: int = 0
    signature: Tuple[str, ...] = ()
    members: List[str] = field(default_factory=list)
    called: str = ""

    @property
    def name(self) -> str:
        return self.called or f"family {self.number}"

    def christen(self, called: str, *, among: int = 1) -> "Family":
        """Name it — afterwards, and only once it has been separated from something else.

        A partition of one family is not a discovery about operations, it is a statement that the
        probes are too blunt to separate anything, and naming it would dress that up.
        """
        if among < 2:
            raise ValueError("a family separated from nothing has not been shown to be a family")
        self.called = called
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "signature": list(self.signature), "members": self.members}

    def render(self) -> str:
        broke = ", ".join(self.signature) or "nothing anybody tried"
        return f"  {self.name:<10} breaks {broke}\n      {', '.join(self.members)}"


# --------------------------------------------------------------------------------------------- #
#  the probes — each one a way of trying to break a law, never a way of confirming it
# --------------------------------------------------------------------------------------------- #
def _row(rng: random.Random, width: int = WIDTH) -> List[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(width)]


def _additivity(fn, rng):
    a, b = _row(rng), _row(rng)
    both = list(fn([x + y for x, y in zip(a, b)]))
    apart = [x + y for x, y in zip(fn(a), fn(b))]
    return (not _close(both, apart)), "f(a+b) against f(a)+f(b)"


def _homogeneity(fn, rng):
    a, k = _row(rng), rng.uniform(-3.0, 3.0)
    scaled = list(fn([k * x for x in a]))
    times = [k * x for x in fn(a)]
    return (not _close(scaled, times)), f"f({k:.2f}·a) against {k:.2f}·f(a)"


def _offset(fn, rng):
    """Does adding a constant to every input add it to every output?"""
    a, c = _row(rng), rng.uniform(-2.0, 2.0)
    lifted = list(fn([x + c for x in a]))
    added = [x + c for x in fn(a)]
    return (not _close(lifted, added)), f"a constant {c:.2f} added to the input"


def _shift(fn, rng):
    """Does sliding the input along slide the output along? Broken by anything that sorts."""
    a = _row(rng)
    before = list(fn(a))
    after = list(fn(a[1:] + a[:1]))
    if len(before) != len(a):        # a shortening operation cannot be probed this way
        k = len(before)
        return (not _close(list(fn(a[1:] + a[:1]))[:k - 1], before[1:k])), "the input slid along"
    return (not _close(after, before[1:] + before[:1])), "the input slid along"


def _monotone(fn, rng):
    """Raising every input, does no output fall?"""
    a = _row(rng)
    b = [x + abs(rng.uniform(0.0, 1.0)) for x in a]
    return (any(y < x - TOLERANCE for x, y in zip(fn(a), fn(b)))), "every input raised"


def _idempotent(fn, rng):
    a = _row(rng)
    once = list(fn(a))
    twice = list(fn(once)) if len(once) == len(a) else list(fn(once + once))[:len(once)]
    return (not _close(twice, once)), "applied twice"


def _order_blind(fn, rng):
    """Does shuffling the input leave the output alone? True of anything that only counts."""
    a = _row(rng)
    b = a[:]
    rng.shuffle(b)
    return (not _close(list(fn(b)), list(fn(a)))), "the input shuffled"


def _pointwise(fn, rng):
    """Does changing one input change only the output at that place?"""
    a = _row(rng)
    at = rng.randrange(len(a))
    b = a[:]
    b[at] += 1.0
    before, after = list(fn(a)), list(fn(b))
    if len(before) != len(after) or len(before) != len(a):
        return True, "one input moved (and the operation is not length-preserving)"
    moved = [i for i, (x, y) in enumerate(zip(before, after)) if abs(x - y) > TOLERANCE]
    return (moved != [] and moved != [at]), f"one input at {at} moved"


def _repeatable(fn, rng):
    """Same input, same output? An operation that wanders cannot be characterised at all."""
    a = _row(rng)
    return (not _close(list(fn(a)), list(fn(a)))), "the same input twice"


#: The battery. Supplied, and said to be supplied — these are **properties**, and the partition they
#: induce over a set of operations is what is not supplied. Each is phrased as an attempt to break,
#: because that is the only direction a probe can establish anything in.
LAWS: Tuple[Law, ...] = (
    Law("adds up", _additivity, "f(a+b) = f(a)+f(b)"),
    Law("scales", _homogeneity, "f(ka) = k·f(a)"),
    Law("follows the level", _offset, "a constant added to the input adds to the output"),
    Law("slides along", _shift, "sliding the input slides the output"),
    Law("never falls", _monotone, "raising every input lowers no output"),
    Law("settles", _idempotent, "applying it twice is applying it once"),
    Law("ignores order", _order_blind, "shuffling the input changes nothing"),
    Law("stays local", _pointwise, "moving one input moves only that output"),
    Law("repeats", _repeatable, "the same input gives the same output"),
)


def probe(operation: Callable[[Sequence[float]], Sequence[float]], *, name: str = "",
          laws: Sequence[Law] = LAWS, tries: int = TRIES,
          rng: Optional[random.Random] = None) -> Reading:
    """Try to break every law, and report what broke — never what held.

    A law that survives is reported with the number of attempts it survived, because *not refuted*
    is a claim whose whole content is that number. A probe that raises is ``could not run``, which
    is neither a break nor a survival: the third state this package has needed at every level.
    """
    rng = rng or random.Random(SEED)
    out = Reading(name=name or "an operation")
    for law in laws:
        if law.attempt is None:
            out.findings.append(Finding(law=law.name, witness="no way to try it was supplied"))
            continue
        broken, witness, ran = False, "", 0
        for _ in range(max(1, tries)):
            try:
                broken, witness = law.attempt(operation, rng)
            except Exception as error:  # noqa: BLE001
                out.findings.append(Finding(law=law.name, tries=ran,
                                            witness=f"the probe raised {type(error).__name__}"))
                break
            ran += 1
            if broken:
                out.findings.append(Finding(law=law.name, stands="refuted", tries=ran,
                                            witness=witness))
                break
        else:
            out.findings.append(Finding(law=law.name, stands="not refuted", tries=ran))
    return out


def partition(operations: Dict[str, Callable[[Sequence[float]], Sequence[float]]], *,
              laws: Sequence[Law] = LAWS, tries: int = TRIES,
              rng: Optional[random.Random] = None) -> Tuple[List[Family], Dict[str, Reading]]:
    """Group operations by what they break. The families are numbered in the order they appear.

    Nothing here knows what any of these operations is. Two of them land together because no probe
    separated them, and that is reported as exactly that — a failure to separate, which may mean
    they are the same kind of thing or may mean the battery is too blunt. Those are different, and
    only more probes can tell them apart.
    """
    rng = rng or random.Random(SEED)
    readings = {name: probe(fn, name=name, laws=laws, tries=tries, rng=rng)
                for name, fn in operations.items()}
    seen: Dict[Tuple[str, ...], Family] = {}
    for name, reading in readings.items():
        key = reading.signature
        if key not in seen:
            seen[key] = Family(number=len(seen) + 1, signature=key)
        seen[key].members.append(name)
    return list(seen.values()), readings
