"""NYXARA · njp/probingschool.py — fourteen operations with their names taken off (🧪).

Fourteen black boxes go in as ``op 1`` … ``op 14``. What is known by construction — and never
shown to the prober — is which of them are behaviourally the same kind of thing. The exam scores
the **partition**, three ways, and the third number is the one that matters most now:

======================  ====================================================================
invented                a real distinction found: two operations that differ, separated
flattered               two that behave identically, called different — false novelty
buried                  two that genuinely differ, called the same — novelty refused
======================  ====================================================================

``buried`` grows as the proofs get stronger, and it is the failure V.88 warned about: *we could not
express the difference, therefore there is none*. A battery too blunt to separate two operations
reports one family and looks decisive.

Three of the fourteen are there to be nasty. ``op 6`` looks nonlinear and is the identity written
through ``max(x, x)``; a prober that reads source or names is fooled and one that probes is not.
``op 13`` is linear plus a thousandth of a square — catchable. ``op 14`` is linear plus a
quadrillionth of one, which no probe at this tolerance will ever catch, and it is here so that the
limitation is a measured number instead of a worry.
"""

from __future__ import annotations

import random
from statistics import median
from typing import Any, Callable, Dict, List, Sequence, Tuple

from nyxara.njp.probing import partition

__all__ = ["OPERATIONS", "TRUTH", "retrodict", "examine", "run", "SEED"]

SEED = 89


def _window(pick: Callable[[List[float]], float], k: int = 3):
    def _run(xs: Sequence[float]) -> List[float]:
        if len(xs) < k:
            return list(xs)
        ring = list(xs) + list(xs[: k - 1])          # wrapped, so length is preserved
        return [pick(ring[i:i + k]) for i in range(len(xs))]
    return _run


def _difference(xs: Sequence[float]) -> List[float]:
    ring = list(xs) + [xs[0]] if xs else []
    return [ring[i + 1] - ring[i] for i in range(len(xs))]


def _scale(k: float):
    return lambda xs: [k * x for x in xs]


def _sorted(xs: Sequence[float]) -> List[float]:
    return sorted(xs)


def _ranks(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    for place, i in enumerate(order):
        out[i] = float(place)
    return out


def _threshold(xs: Sequence[float]) -> List[float]:
    return [1.0 if x > 0 else 0.0 for x in xs]


def _square(xs: Sequence[float]) -> List[float]:
    return [x * x for x in xs]


def _reverse(xs: Sequence[float]) -> List[float]:
    return list(reversed(xs))


def _wobbly(xs: Sequence[float]) -> List[float]:
    return [x + random.random() * 1e-6 for x in xs]


def _sneaky_identity(xs: Sequence[float]) -> List[float]:
    """Written with a comparison, and is the identity. Nothing but probing can tell."""
    return [max(x, x) for x in xs]


def _nearly_linear(strength: float):
    """Linear plus a pinch of square. ``strength`` decides whether any probe can ever see it."""
    return lambda xs: [x + strength * x * x for x in xs]


#: The fourteen, with their names taken off. The prober is handed this dictionary's **values**
#: through keys that say nothing.
OPERATIONS: Dict[str, Callable[[Sequence[float]], List[float]]] = {
    "op 1": lambda xs: list(xs),                       # identity
    "op 2": _scale(2.5),                               # a scaling
    "op 3": _window(lambda w: sum(w) / len(w)),        # a smoothing
    "op 4": _difference,                               # a difference
    "op 5": _window(median),                           # a median
    "op 6": _sneaky_identity,                          # the identity in disguise
    "op 7": _window(max),                              # a largest
    "op 8": _sorted,                                   # a sort
    "op 9": _ranks,                                    # a rank
    "op 10": _threshold,                               # a threshold
    "op 11": _square,                                  # a square
    "op 12": _reverse,                                 # a reversal
    "op 13": _nearly_linear(1e-3),                     # linear plus a thousandth of a square
    "op 14": _nearly_linear(1e-15),                    # and plus a quadrillionth of one
}

#: Which of them are the same kind of thing, by construction and never shown to the prober.
#:
#: The first draft of this table put ``op 2`` — a scaling by 2.5 — in with the identity, on the
#: reasoning that a scaling is "the identity up to a constant". The prober separated them and was
#: **right**: add a constant to the input of a scaling and you get 2.5 times it back, and applying
#: it twice gives 6.25×. It breaks *follows the level* and *settles*, and the identity breaks
#: neither. The table was wrong and was corrected; the probes were not weakened to agree with it.
#:
#: ``op 14`` does belong with the identity — it is linear plus a quadrillionth of a square, and no
#: probe at this tolerance will ever see the difference. That is not the battery failing, it is the
#: battery's resolution, stated as a number instead of as a worry.
TRUTH: Tuple[Tuple[str, ...], ...] = (
    ("op 1", "op 6", "op 14"),
    ("op 2",),
    ("op 3",),
    ("op 4",),
    ("op 5",),
    ("op 7",),
    ("op 8",),
    ("op 9",),
    ("op 10",),
    ("op 11",),
    ("op 12",),
    ("op 13",),
)


def _together(groups: Sequence[Sequence[str]]) -> Dict[Tuple[str, str], bool]:
    """Every pair, and whether the grouping puts them together."""
    names = sorted({n for g in groups for n in g})
    where = {n: i for i, g in enumerate(groups) for n in g}
    return {(a, b): where.get(a, -1) == where.get(b, -2)
            for i, a in enumerate(names) for b in names[i + 1:]}


def retrodict(tries: int = 200) -> Dict[str, Any]:
    """Probe all fourteen, partition them, and compare the partition to what is true."""
    families, readings = partition(OPERATIONS, tries=tries, rng=random.Random(SEED))
    got = _together([f.members for f in families])
    want = _together(TRUTH)
    invented = sum(1 for k, v in want.items() if not v and not got.get(k, True))
    flattered = sum(1 for k, v in want.items() if v and not got.get(k, True))
    buried = sum(1 for k, v in want.items() if not v and got.get(k, False))
    return {"families": [f.to_dict() for f in families],
            "found": len(families), "of": len(TRUTH),
            "invented": invented, "flattered": flattered, "buried": buried,
            "pairs": len(want),
            # Pairs that genuinely differ, which is what `invented` can be measured against. The
            # first pass condition compared it to *every* pair, which counts the pairs that are
            # correctly together as failures to separate — a bar nothing can clear.
            "separable": sum(1 for v in want.values() if not v),
            "together": sum(1 for v in want.values() if v),
            "readings": {n: r.to_dict() for n, r in readings.items()},
            "signatures": {f.name: list(f.signature) for f in families},
            "_families": families, "_readings": readings}


def examine(tries: int = 200) -> Dict[str, Any]:
    """The exam, with its pass condition written down rather than implied.

    ``flattered`` must be zero: two operations that behave identically must not be reported as
    different, which is the false-novelty failure. ``buried`` is allowed to be non-zero **and is
    reported either way**, because the only operations that can be buried here are ones separated
    by a quadrillionth — and a battery that claimed to catch those would be lying about its
    tolerance rather than being good.
    """
    got = retrodict(tries)
    got["passes"] = bool(got["flattered"] == 0
                         and got["invented"] == got["separable"] - got["buried"]
                         and got["found"] == got["of"])
    return got


def run() -> Dict[str, Any]:  # pragma: no cover — a report
    got = examine()
    for family in got["_families"]:
        print(family.render())
    print(f"\n  {got['found']} families from {len(OPERATIONS)} operations "
          f"({got['of']} by construction)")
    print(f"  of {got['pairs']} pairs, {got['separable']} genuinely differ and "
          f"{got['together']} genuinely match")
    print(f"  invented {got['invented']}, flattered {got['flattered']}, buried {got['buried']} "
          f"— passes {got['passes']}\n")
    print(got["_readings"]["op 5"].render())
    return got


if __name__ == "__main__":  # pragma: no cover
    run()
