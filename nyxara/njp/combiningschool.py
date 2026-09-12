"""NYXARA · njp/combiningschool.py — did it find a way of combining, or pay for one (🧩).

Five families and one property. The families check that a way of combining is found where one
exists and refused where none does. The property is the version's actual claim, and it is a
monotone rather than a verdict:

    **widening the search must make the same answer more expensive.**

Measured on a family of turns, which every width of search solves the same way:

======================  ======  ==========  ========  =======
words up to             ways    charged     winner    total
======================  ======  ==========  ========  =======
one                     2       1.0         none      193.0
two                     6       2.6         ``ab``    55.6
three                   14      3.8         ``ab``    56.8
four                    30      4.9         ``ab``    57.9
======================  ======  ==========  ========  =======

A search told to invent a way of combining will consider an enormous number of them and report the
one that fits. Charging for the size of the list is what stops that being free, and the row for
*words up to one* is the other half: with only the degenerate ways available — the ones that ignore
an argument — **nothing is found**, which is correct and is what a candidate list stops being a
hint by containing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

from nyxara.njp.combining import Found, search, ways
from nyxara.njp.generators import WIDTH, watch_behaviour

__all__ = ["FAMILIES", "KNOWN", "Case", "widening", "retrodict", "examine", "run", "SEED"]

SEED = 95


def _turn(k: int) -> Callable[[Sequence[float]], List[float]]:
    return lambda row: [row[(i + k) % WIDTH] for i in range(WIDTH)]


def _turns() -> List[Any]:
    return [watch_behaviour(_turn(k)) for k in range(WIDTH)]


def _turns_and_a_flip() -> List[Any]:
    seen = _turns()
    seen += [watch_behaviour(lambda r, k=k: list(reversed(_turn(k)(r)))) for k in range(WIDTH)]
    return sorted({s for s in seen if s is not None})


def _pairs() -> List[Any]:
    """Swapping the two halves, and turning within them. Structure, but not the same structure."""
    out = []
    for k in range(WIDTH // 2):
        out.append(watch_behaviour(
            lambda r, k=k: [r[(i + k) % (WIDTH // 2) + (WIDTH // 2) * (i >= WIDTH // 2)]
                            if i < WIDTH // 2 else
                            r[WIDTH // 2 + (i - WIDTH // 2 + k) % (WIDTH // 2)]
                            for i in range(WIDTH)]))
    return sorted({o for o in out if o is not None})


def _unrelated() -> List[Any]:
    rng = random.Random(SEED)
    out = []
    for _ in range(WIDTH):
        order = list(range(WIDTH))
        rng.shuffle(order)
        out.append(tuple(order))
    return out


FAMILIES: Dict[str, Callable[[], List[Any]]] = {
    "turns": _turns,
    "turns and a flip": _turns_and_a_flip,
    "turns inside halves": _pairs,
    "unrelated": _unrelated,
}


@dataclass
class Case:
    name: str = ""
    found: bool = True
    note: str = ""


KNOWN: Tuple[Case, ...] = (
    Case("turns", True, "one seed under one way reaches all of them"),
    Case("turns and a flip", True, "two seeds, and the way still has to be paid for"),
    Case("turns inside halves", True, "different structure, and it must not be forced into the same"),
    Case("unrelated", False, "nothing in common; the correct answer is that longhand is cheapest"),
)


def widening(family: str = "turns", widths: Sequence[int] = (1, 2, 3, 4)) -> List[Dict[str, Any]]:
    """The same family searched at several widths. The version's claim lives here.

    A wider search finds the same answer and pays more for having looked in more places. If this
    ever came back flat or falling, *invent a way of combining* would be an instruction a search
    could satisfy by widening itself, which is not an experiment.
    """
    observed = FAMILIES[family]()
    out = []
    for longest in widths:
        got = search(observed, longest=longest)
        out.append({"longest": longest, "ways": len(ways(longest)),
                    **got.to_dict()})
    return out


def retrodict() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    right = flattered = buried = 0
    degenerate = 0
    for case in KNOWN:
        got: Found = search(FAMILIES[case.name]())
        ok = got.worth_it == case.found
        right += int(ok)
        if got.worth_it and not case.found:
            flattered += 1
        if case.found and not got.worth_it:
            buried += 1
        if got.way is not None and got.worth_it and set(got.way.word) < {"a", "b"}:
            degenerate += 1
        rows.append({"family": case.name, "want": case.found, **got.to_dict(),
                     "note": case.note})
    steps = widening()
    winners = [s for s in steps if s["worth_it"]]
    climbs = all(b["cost"] >= a["cost"] for a, b in zip(winners, winners[1:]))
    return {"rows": rows, "right": right, "of": len(KNOWN), "flattered": flattered,
            "buried": buried, "degenerate": degenerate, "widening": steps,
            "cost_climbs_with_the_search": climbs,
            "nothing_from_degenerate_ways_alone": not steps[0]["worth_it"] if steps else False}


def examine() -> Dict[str, Any]:
    """The exam, with its pass condition written down rather than implied.

    Five conditions. Every family gets its answer; nothing is found where there is nothing;
    nothing with structure comes back empty; **no degenerate way ever wins**, because a way that
    ignores an argument reaching anything would mean the reaching was not the way's doing; and the
    cost climbs as the search widens, which is the claim the whole version rests on.
    """
    got = retrodict()
    got["passes"] = bool(got["right"] == got["of"] and got["flattered"] == 0
                         and got["buried"] == 0 and got["degenerate"] == 0
                         and got["cost_climbs_with_the_search"]
                         and got["nothing_from_degenerate_ways_alone"])
    return got


def run() -> Dict[str, Any]:  # pragma: no cover — a report
    got = examine()
    for row in got["rows"]:
        want = "something" if row["want"] else "nothing"
        print(f"  {row['family']:<22} want {want:<10} way `{row['way'] or '—':<4}` "
              f"seeds {row['seeds']}  covered {row['covered']}/{row['of']:<4} "
              f"{row['cost']:>7.0f} bits against {row['longhand']:>6.0f}")
    print(f"\n  does a wider search buy a cheaper answer?")
    print(f"    {'words up to':<13}{'ways':>6}{'charged':>10}{'winner':>9}{'total':>9}")
    for step in got["widening"]:
        print(f"    {step['longest']:<13}{step['ways']:>6}{step['charged']:>10.1f}"
              f"{(step['way'] or '—'):>9}{step['cost']:>9.1f}")
    print(f"\n  right {got['right']}/{got['of']}, flattered {got['flattered']}, "
          f"buried {got['buried']}, degenerate winners {got['degenerate']}, "
          f"cost climbs {got['cost_climbs_with_the_search']} — passes {got['passes']}")
    return got


if __name__ == "__main__":  # pragma: no cover
    run()
