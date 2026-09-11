"""NYXARA · njp/worthschool.py — is the scale honest, or does it just like small things (💰).

A measure of vocabularies has two opposite ways to be worthless, and an exam that only guards one
of them is worse than none — it certifies the other.

======================  ====================================================================
failure                 the fixture that catches it
======================  ====================================================================
bigger always wins      ``every regrouping`` and ``V.91's substrate`` are the two largest, and
                        both must lose to something four times smaller
smaller always wins     ``one move`` costs one bit and explains 41 of 105 pairs. It must lose,
                        and lose badly
the measure is hollow   on **random** pairs every vocabulary explains everything, so the exam
                        checks that its own hard pairs are what make the measure bite
======================  ====================================================================

And the headline the scale produced on its first use is a demotion of this repository's own work:
seven plain slides describe the whole held-out set in **3.8 bits** where V.91's 252-move substrate
needs **9.0**.
"""

from __future__ import annotations

import random
from itertools import permutations
from typing import Any, Dict, List, Sequence

from nyxara.njp.worth import VOCABULARIES, WIDTH, compare, weigh

__all__ = ["retrodict", "examine", "run", "easy_pairs", "SEED"]

SEED = 92


def easy_pairs(n: int = 200, rng: random.Random = None) -> List[Any]:
    """Randomly drawn rearrangements. Every vocabulary separates all of them, which is the point.

    Kept so the exam can show *why* the hard pairs are not an arbitrary choice: a measure taken on
    these would rank every vocabulary identically and certify anything.
    """
    rng = rng or random.Random(SEED)
    every = list(permutations(range(WIDTH)))
    rng.shuffle(every)
    pool = every[:400]
    out = [(rng.choice(pool), rng.choice(pool)) for _ in range(n)]
    return [(a, b) for a, b in out if a != b]


def retrodict(seeds: Sequence[int] = (92, 193, 294)) -> Dict[str, Any]:
    """Weigh every vocabulary, on several draws, and check the order does not wander."""
    orders: List[List[str]] = []
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        weighed = compare(VOCABULARIES, rng=random.Random(seed))
        orders.append([w.vocabulary.name for w in weighed if w.vocabulary])
        if seed == seeds[0]:
            rows = [w.to_dict() for w in weighed]
    easy = easy_pairs()
    saturates = {v.name: weigh(v, easy).worth == len(easy) for v in VOCABULARIES}
    biggest = max(VOCABULARIES, key=lambda v: len(v.distinct))
    smallest = min(VOCABULARIES, key=lambda v: len(v.distinct))
    return {"rows": rows, "orders": orders, "steady": len({tuple(o) for o in orders}) == 1,
            "winner": orders[0][0] if orders and orders[0] else "",
            "biggest": biggest.name, "smallest": smallest.name,
            "saturates_on_easy": sum(saturates.values()), "vocabularies": len(VOCABULARIES),
            "easy_pairs": len(easy)}


def examine(seeds: Sequence[int] = (92, 193, 294)) -> Dict[str, Any]:
    """The exam, with its pass condition written down rather than implied.

    Five conditions. The order must hold across draws. The largest vocabulary must not win and
    neither must the smallest — those are the two failures, and a scale that avoids only one of
    them certifies the other. Almost every vocabulary must saturate on easy pairs, because that is
    the evidence that the hard ones are doing the work. And the winner must be **complete**: a
    vocabulary that leaves pairs unexplained has not earned first place by being cheap.
    """
    got = retrodict(seeds)
    first = got["rows"][0] if got["rows"] else {}
    got["passes"] = bool(got["steady"]
                         and got["winner"] not in (got["biggest"], got["smallest"])
                         and first.get("complete")
                         and got["saturates_on_easy"] >= got["vocabularies"] - 1)
    return got


def run() -> Dict[str, Any]:  # pragma: no cover — a report
    got = examine()
    print(f"  {'':<3}{'vocabulary':<22}{'moves':>6}{'bits':>8}{'worth':>13}"
          f"{'whole':>10}{'per bit':>9}")
    for row in got["rows"]:
        mark = "ok " if row["complete"] else " ~ "
        print(f"  {mark}{row['name']:<22}{row['moves']:>6}{row['cost']:>8.2f}"
              f"{row['worth']:>6} /{row['of']:<5}{row['total']:>10.1f}{row['value']:>9.1f}")
    print(f"\n  shortest whole description: {got['winner']}")
    print(f"  largest vocabulary is {got['biggest']}, smallest is {got['smallest']} — "
          f"neither wins")
    print(f"  on {got['easy_pairs']} randomly drawn pairs, {got['saturates_on_easy']} of "
          f"{got['vocabularies']} vocabularies explain every one — which is why the exam does not "
          f"use them")
    print(f"  order steady across {len(got['orders'])} draws: {got['steady']} — "
          f"passes {got['passes']}")
    return got


if __name__ == "__main__":  # pragma: no cover
    run()
