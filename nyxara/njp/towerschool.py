"""NYXARA · njp/towerschool.py — does the tower stop, or is it told to (🗼).

A module that always answers *stop at storey one* would be a preference wearing a measurement's
clothes. So the exam contains a world where climbing **does** pay, built by making the default
language absurd: a search that already owes nine bits for naming a word among 510 is one that can
be improved by being told to look in a smaller language instead.

======================  ====================================================================
world                   where the arithmetic stops, and why
======================  ====================================================================
a sensible default      **one**. Naming a language costs more than naming well inside one saves.
an absurd default       **two**. The default is so wide that a smaller language pays for itself.
nothing to explain      **zero**. No storey explains anything, so the cheapest is writing it out.
======================  ====================================================================

The point is not the numbers in any one row. It is that the same arithmetic produces all three, and
nothing anywhere prefers a height.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

from nyxara.njp.generators import WIDTH, watch_behaviour
from nyxara.njp.tower import LANGUAGES, Climb, Language, climb

__all__ = ["WORLDS", "KNOWN", "Case", "retrodict", "examine", "run", "SEED"]

SEED = 96


def _turn(k: int) -> Callable[[Sequence[float]], List[float]]:
    return lambda row: [row[(i + k) % WIDTH] for i in range(WIDTH)]


def _turns() -> List[Any]:
    return [watch_behaviour(_turn(k)) for k in range(WIDTH)]


def _unrelated() -> List[Any]:
    rng = random.Random(SEED)
    out = []
    for _ in range(WIDTH):
        order = list(range(WIDTH))
        rng.shuffle(order)
        out.append(tuple(order))
    return out


#: Each world is the observations plus the language storey one is made to use.
WORLDS: Dict[str, Tuple[Callable[[], List[Any]], Language]] = {
    "a sensible default": (_turns, Language(3)),
    "an absurd default": (_turns, Language(8)),
    "nothing to explain": (_unrelated, Language(3)),
}


@dataclass
class Case:
    name: str = ""
    stops_at: int = 1
    note: str = ""


KNOWN: Tuple[Case, ...] = (
    Case("a sensible default", 1, "naming a language costs more than it saves"),
    Case("an absurd default", 2, "the default is wide enough that a smaller one pays"),
    Case("nothing to explain", 0, "no storey explains anything; writing it out is cheapest"),
)


def retrodict() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    right = 0
    for case in KNOWN:
        make, default = WORLDS[case.name]
        got: Climb = climb(make(), default=default)
        ok = got.stops_at == case.stops_at
        right += int(ok)
        rows.append({"world": case.name, "want": case.stops_at, **got.to_dict(),
                     "note": case.note})
    tolls = [lang.toll for lang in sorted(LANGUAGES, key=lambda l: l.longest)]
    return {"rows": rows, "right": right, "of": len(KNOWN),
            "toll_never_falls": all(b >= a for a, b in zip(tolls, tolls[1:])),
            "heights_seen": sorted({r["stops_at"] for r in rows}),
            "languages": [(l.name, l.size, round(l.toll, 2)) for l in LANGUAGES]}


def examine() -> Dict[str, Any]:
    """The exam, with its pass condition written down rather than implied.

    Three conditions. Every world stops where it should. **More than one height** is seen across
    the worlds, because a module that always answers the same number is not measuring. And a
    language's toll never falls as it widens, which is the property the whole tower rests on: the
    toll does not shrink as you climb, the savings do, and that is why there is a top.
    """
    got = retrodict()
    got["passes"] = bool(got["right"] == got["of"] and len(got["heights_seen"]) >= 3
                         and got["toll_never_falls"])
    return got


def run() -> Dict[str, Any]:  # pragma: no cover — a report
    got = examine()
    print("  languages on offer: "
          + ", ".join(f"{n} ({s} ways, {t} bits)" for n, s, t in got["languages"]) + "\n")
    for row in got["rows"]:
        print(f"  {row['world']}  — wanted storey {row['want']}, stopped at {row['stops_at']}")
        for storey in sorted(row["storeys"], key=lambda s: s["height"]):
            mark = "ok " if storey["explained"] else " ~ "
            print(f"    {mark}{storey['height']}· {storey['name']:<26}"
                  f"{storey['toll']:>7.1f} + {storey['description']:>8.1f} = "
                  f"{storey['total']:>8.1f}")
        print(f"    climbing pays at {row['worth_climbing']}\n")
    print(f"  right {got['right']}/{got['of']}, heights seen {got['heights_seen']}, "
          f"toll never falls {got['toll_never_falls']} — passes {got['passes']}")
    return got


if __name__ == "__main__":  # pragma: no cover
    run()
