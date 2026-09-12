"""NYXARA · njp/substrateschool.py — did the move earn its place, or was it always there (🧱).

The exam is not *were moves generated*. Four numbers generate 252 of them and most are furniture.
It is whether a generated move **buys a separation the supplied vocabulary cannot make** — and
that is checkable, because a ``first`` law reads one position of ``f(x)`` against one position of
``x``, and the supplied seven reach only 16 of the 49 pairs.

So the fixture lives in the other 33. ``swap 3 and 4`` against ``swap 3 and 5`` are the identity
everywhere the supplied offsets ``{0, 1, 2, 6}`` can look, are both permutations so every total is
preserved, and differ only at positions no supplied move can name.

**And the case where generation buys nothing is here too**, because a version that only showed its
own successes would not be an exam. Two stride permutations are always separated by the supplied
seven, and not by luck: blocking all four supplied offsets at once admits a single stride, so two
distinct ones can never both be hidden. That is worked out by enumeration in
:func:`strides_are_already_covered` rather than asserted.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Sequence, Tuple

from nyxara.njp.substrate import (SUPPLIED, WIDTH, Shape, earns, every_move, reach, separations)

__all__ = ["PAIRS", "swap", "stride", "strides_are_already_covered", "retrodict", "examine",
           "run", "SEED"]

SEED = 91


def swap(i: int, j: int) -> Callable[[Sequence[float]], List[float]]:
    """The identity, except two positions change places."""
    def _run(row: Sequence[float]) -> List[float]:
        out = list(row)
        out[i], out[j] = out[j], out[i]
        return out
    return _run


def stride(k: int) -> Callable[[Sequence[float]], List[float]]:
    """Regroup the row by taking every ``k``-th position, wrapping."""
    return lambda row: [row[(k * i) % WIDTH] for i in range(WIDTH)]


#: Four pairs whose answer is known by where they live on the map of readable positions.
PAIRS: Tuple[Tuple[str, Any, str, Any, bool, str], ...] = (
    ("swap 3 and 4", swap(3, 4), "swap 3 and 5", swap(3, 5), True,
     "both the identity everywhere the supplied offsets can look"),
    ("stride two", stride(2), "stride three", stride(3), False,
     "the supplied seven separate these already — generation buys nothing"),
    ("swap 0 and 1", swap(0, 1), "swap 0 and 2", swap(0, 2), False,
     "inside the readable region, so the supplied vocabulary is enough"),
    ("leave alone", lambda r: list(r), "leave alone again", lambda r: [max(v, v) for v in r],
     False, "the same operation twice; no move may buy a separation here"),
)


def strides_are_already_covered(width: int = WIDTH) -> Dict[str, Any]:
    """Why generation buys nothing on stride permutations, worked out rather than asserted.

    A law ``first: slid-by-t₁(f(x)) == slid-by-t₂(x)`` holds for the stride-``s`` regrouping exactly
    when ``s·t₁ ≡ t₂``. For two distinct strides to be indistinguishable, **every** supplied offset
    would have to send both of them outside the supplied set at once — and enumerating which
    strides that admits gives at most one, so two distinct strides can never both hide.
    """
    offsets = sorted({m.offset % width for m in SUPPLIED})
    hidden = [s for s in range(1, width)
              if all((s * t) % width not in offsets for t in offsets if t)]
    return {"supplied_offsets": offsets, "strides_that_could_hide": hidden,
            "enough": len(hidden) < 2}


def retrodict(tries: int = 30) -> Dict[str, Any]:
    """Run every pair against the supplied vocabulary, then against each generated move."""
    rows: List[Dict[str, Any]] = []
    right = flattered = missed = 0
    generated = every_move()
    extras = [m for m in generated if not any(
        all(abs(p - q) <= 1e-9 for row in ((1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),)
            for p, q in zip(m.of(row), s.of(row))) for s in SUPPLIED)]
    for name_a, one, name_b, two, needs_new, note in PAIRS:
        with_supplied = separations(one, two, list(SUPPLIED), tries=tries,
                                    rng=random.Random(SEED))
        bought_by = ""
        if not with_supplied:
            for move in extras:
                if separations(one, two, list(SUPPLIED) + [move], tries=tries,
                               rng=random.Random(SEED)):
                    bought_by = move.name
                    break
        got_new = bool(bought_by)
        ok = (not with_supplied and got_new) if needs_new else bool(with_supplied) or (
            not with_supplied and not got_new)
        right += int(ok)
        if needs_new and not got_new:
            missed += 1
        if not needs_new and not with_supplied and got_new:
            flattered += 1
        rows.append({"pair": f"{name_a} / {name_b}", "needs_new": needs_new,
                     "supplied_laws": len(with_supplied), "bought_by": bought_by,
                     "note": note})
    return {"rows": rows, "right": right, "of": len(PAIRS),
            "flattered": flattered, "missed": missed,
            "moves": len(generated), "supplied": len(SUPPLIED), "extras": len(extras),
            "supplied_reach": len(reach(SUPPLIED)), "whole_reach": len(reach(generated)),
            "strides": strides_are_already_covered()}


def examine(tries: int = 30) -> Dict[str, Any]:
    """The exam, with its pass condition written down rather than implied.

    Four conditions. The pair that needs a new move must get one. No pair the supplied vocabulary
    already handles may be credited to a generated move — that is `flattered`, and three of the
    four pairs are there to have a chance at it. The supplied vocabulary must genuinely fall short
    of the whole, or the exam is about nothing. And the stride result must hold, because a version
    that only showed where generation wins would be advertising rather than measuring.
    """
    got = retrodict(tries)
    got["passes"] = bool(got["right"] == got["of"]
                         and got["flattered"] == 0 and got["missed"] == 0
                         and got["supplied_reach"] < got["whole_reach"]
                         and got["strides"]["enough"])
    return got


def run() -> Dict[str, Any]:  # pragma: no cover — a report
    got = examine()
    print(f"  the substrate yields {got['moves']} behaviourally distinct moves from four numbers; "
          f"{got['supplied']} were supplied and {got['extras']} were not.")
    print(f"  a `first` law reads one position of f(x) against one of x. The supplied moves reach "
          f"{got['supplied_reach']} of {got['whole_reach']} pairs.\n")
    for row in got["rows"]:
        want = "a new move" if row["needs_new"] else "no new move"
        print(f"  {row['pair']:<34} want {want:<12} supplied laws {row['supplied_laws']:>4}  "
              f"bought by {row['bought_by'] or '—'}")
    strides = got["strides"]
    print(f"\n  supplied offsets {strides['supplied_offsets']}; strides that could hide from all "
          f"of them: {strides['strides_that_could_hide']} — so two distinct strides never both can")
    print(f"  right {got['right']}/{got['of']}, flattered {got['flattered']}, "
          f"missed {got['missed']} — passes {got['passes']}")
    return got


if __name__ == "__main__":  # pragma: no cover
    run()
