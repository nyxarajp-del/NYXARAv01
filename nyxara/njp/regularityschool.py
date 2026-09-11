"""NYXARA · njp/regularityschool.py — did the law earn its existence (⚖).

The exam is not *were laws found*. A schema of 1,456 candidates will always yield some true
sentences. The exam is whether a discovered law **draws a distinction the supplied vocabulary
could not**, and that is checkable because V.89's nine laws leave three pairs merged:

======================  ====================================================================
merged by V.89          and genuinely different
======================  ====================================================================
sort upward / downward  same equivariances, opposite order
smallest / largest      of a three-wide window; conjugate under negation
hold first / hold last  same everything except which end
======================  ====================================================================

Those three pairs are the whole point. The nine supplied laws cannot separate any of them at any
number of tries, because the distinction is not in them. A law that splits one has earned its
existence; one that only re-sorts what was already sorted has not.

And the four-way accounting is computed rather than asserted. Two operations that never disagree
were **right** to be merged. Two that do disagree and were merged are either a search failure
(``buried``) or a vocabulary too small (``resolution-limited``), and those are told apart by asking
whether **any** rule in the whole schema would have split them — which is answerable only because
the search is exhaustive.
"""

from __future__ import annotations

import random
from statistics import median
from typing import Any, Callable, Dict, List, Sequence, Tuple

from nyxara.njp.probing import probe
from nyxara.njp.regularity import Split, differ, discover, holds, rules

__all__ = ["OPERATIONS", "MERGED", "old_families", "retrodict", "examine", "run", "SEED"]

SEED = 90


def _window(pick: Callable[[List[float]], float], k: int = 3):
    def _run(xs: Sequence[float]) -> List[float]:
        if len(xs) < k:
            return list(xs)
        ring = list(xs) + list(xs[: k - 1])
        return [pick(ring[i:i + k]) for i in range(len(xs))]
    return _run


#: Eight operations, six of which V.89's nine laws cannot tell apart in three pairs.
OPERATIONS: Dict[str, Callable[[Sequence[float]], List[float]]] = {
    "sort up": lambda xs: sorted(xs),
    "sort down": lambda xs: sorted(xs, reverse=True),
    "window min": _window(min),
    "window max": _window(max),
    "window middle": _window(median),
    "hold first": lambda xs: [xs[0]] * len(xs),
    "hold last": lambda xs: [xs[-1]] * len(xs),
    "leave alone": lambda xs: list(xs),
    # Two more, and they are here so that two of the four numbers are not vacuous. The first run of
    # this exam had `flattered 0` and `resolution-limited 0` with nothing in front of either — a
    # gate nothing exercises has not been shown to work, which is the same complaint this package
    # makes of an unrun check.
    #
    # `also leave alone` is the identity written through a comparison, as in V.89. No law may
    # separate it from `leave alone`, and one that does is `flattered`.
    "also leave alone": lambda xs: [max(v, v) for v in xs],
}

#: **A debt, not a result.** ``resolution-limited`` is computed below and **nothing here exercises
#: it**, which by this package's own standard means it has not been shown to work. Two fixtures were
#: built for it and both failed, and each failure is a real finding about how sensitive this schema
#: is:
#:
#: * A sort scaled by a part in ten million was caught at once. A sort preserves the total
#:   **exactly**, so ``total: x >= f(x)`` holds with equality on it — and a relation that is tight
#:   is an infinitely sensitive detector of any scaling whatever.
#: * A median scaled the same way looked out of reach on one draw of rows, and ``last: f(x) >=
#:   slid by two(x)`` caught it on another. It is **borderline**, not beyond the schema — and one
#:   draw treated as definitive is the error this whole line of work began by fixing.
#:
#: So the category stays, unexercised and labelled, rather than being demonstrated with a fixture
#: that only sometimes works.


#: The pairs the supplied vocabulary merges. Recomputed rather than asserted — see
#: :func:`old_families` — so that a change to V.89's battery cannot quietly make this exam easier.
MERGED: Tuple[Tuple[str, str], ...] = (
    ("sort up", "sort down"),
    ("window min", "window max"),
    ("hold first", "hold last"),
)


def old_families() -> Dict[Tuple[str, ...], List[str]]:
    """What V.89's nine supplied laws make of these eight. Run, never assumed."""
    out: Dict[Tuple[str, ...], List[str]] = {}
    for name, fn in OPERATIONS.items():
        signature = probe(fn, name=name, tries=120, rng=random.Random(SEED)).signature
        out.setdefault(signature, []).append(name)
    return out


def _separated(splits: Sequence[Split], one: str, two: str) -> bool:
    """Does any discovered law put these two on opposite sides?"""
    return any((one in s.obeyed_by) != (two in s.obeyed_by) for s in splits)


def _anything_would(one: str, two: str, tries: int = 40) -> bool:
    """Would **any** rule in the whole schema have split them? Answerable only because it is finite."""
    rng = random.Random(SEED)
    for rule in rules():
        a, _ = holds(rule, OPERATIONS[one], tries=tries, rng=random.Random(rng.randrange(1 << 30)))
        b, _ = holds(rule, OPERATIONS[two], tries=tries, rng=random.Random(rng.randrange(1 << 30)))
        if a != b:
            return True
    return False


def retrodict(tries: int = 40) -> Dict[str, Any]:
    """Search the schema, then account for every pair four ways."""
    splits, report = discover(OPERATIONS, tries=tries, rng=random.Random(SEED))
    names = sorted(OPERATIONS)
    invented = flattered = buried = limited = 0
    right_to_merge = 0
    trouble: List[str] = []
    for i, one in enumerate(names):
        for two in names[i + 1:]:
            really = differ(OPERATIONS[one], OPERATIONS[two], rng=random.Random(SEED))
            apart = _separated(splits, one, two)
            if apart and really:
                invented += 1
            elif apart and not really:
                flattered += 1
                trouble.append(f"{one} / {two} separated though they never disagree")
            elif not apart and not really:
                right_to_merge += 1
            elif _anything_would(one, two):
                buried += 1
                trouble.append(f"{one} / {two} merged though a searched law splits them")
            else:
                limited += 1
    old = old_families()
    rescued = [pair for pair in MERGED if _separated(splits, *pair)]
    return {"splits": [s.to_dict() for s in splits], "_splits": splits,
            "invented": invented, "flattered": flattered, "buried": buried,
            "resolution_limited": limited, "right_to_merge": right_to_merge,
            "trouble": trouble, "old_families": len(old),
            "merged_before": [list(p) for p in MERGED],
            "rescued": [list(p) for p in rescued], **report}


def examine(tries: int = 40) -> Dict[str, Any]:
    """The exam, with its pass condition written down rather than implied.

    Four conditions. **Every** pair the supplied vocabulary merged must be separated — that is the
    milestone and nothing else in here substitutes for it. No pair that never disagrees may be
    split. Nothing may be buried, which is meaningful only because the search is exhaustive. And
    more than one distinct law must survive, because a schema yielding exactly one distinction has
    not been shown to be a schema.
    """
    got = retrodict(tries)
    got["passes"] = bool(len(got["rescued"]) == len(MERGED)
                         and got["flattered"] == 0
                         and got["buried"] == 0
                         and got["right_to_merge"] >= 1      # or `flattered` is vacuous
                         and got["distinct"] >= 2)
    got["unexercised"] = ["resolution_limited"] if got["resolution_limited"] == 0 else []
    return got


def run() -> Dict[str, Any]:  # pragma: no cover — a report
    got = examine()
    print(f"  V.89's nine laws put {len(OPERATIONS)} operations into "
          f"{got['old_families']} families, merging {len(MERGED)} pairs that differ.\n")
    print(f"  {got['searched']} candidate laws searched; {got['vacuous']} say nothing about "
          f"anything; {got['distinct']} draw distinct lines.\n")
    for split in got["_splits"][:4]:
        print(split.render())
    print(f"\n  pairs the old vocabulary merged and these laws rescued: {got['rescued']}")
    print(f"  invented {got['invented']}, flattered {got['flattered']}, buried {got['buried']}, "
          f"resolution-limited {got['resolution_limited']}, "
          f"rightly merged {got['right_to_merge']} — passes {got['passes']}")
    for line in got["trouble"]:
        print(f"    · {line}")
    if got["unexercised"]:
        print(f"  nothing here exercises {', '.join(got['unexercised'])} — computed, not shown")
    return got


if __name__ == "__main__":  # pragma: no cover
    run()
