"""NYXARA · njp/reachschool.py — does it find the edge, or does it always find one (📏➡).

Reporting a boundary is easy and reporting it honestly is not. The failure mode worth testing is
not *misses the cliff* — it is **finds a cliff in every ladder**, because a number that always
arrives looks like a result and a roadmap built on it sends work at noise.

So five ladders, and only one of them has an edge. The other four are the ways a capability fails
to have one, and the right answer for each is to say so rather than to name the highest passing
rung:

* **sharp** — holds to four, collapses at five. An edge, and a cliff.
* **graceful** — fades across the dial with no single step falling away. No one place to attack.
* **exhausted** — holds everywhere tried. No boundary *within what was tried*, which is not the
  same claim as no boundary.
* **barren** — holds nowhere. Nothing to put a boundary on.
* **patchy** — holds at one and three and not at two. Whatever that is, it is not a boundary, and
  the measurement or the dial is wrong before anything else can be said.

Scored on all five, and the pass condition requires the four non-edges to come back **without** an
edge. Getting the sharp one right while inventing edges for the rest is a failure here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.reach import Ladder, climb
from nyxara.njp.measurement import Benchmark

__all__ = ["Case", "KNOWN", "examine", "retrodict", "run", "SEED", "DIAL"]

SEED = 78

#: The settings every ladder is measured at, easiest first.
DIAL: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)

#: How many items each rung is scored on. Enough that `CLEAR` is distinguishable from noise on it:
#: at 400 items a floor of 0.5 has a standard error near 0.025, so the 0.08 bar is about three of
#: those. Fewer, and `holds` would be a coin toss dressed as a boundary.
ITEMS = 400


def _bench(at: int, right: Callable[[int], float], *, skew: float = 0.5,
           leaks: bool = False, wobbles: bool = False) -> Benchmark:
    """A rung: answers at a given skew, and a system right at whatever rate the shape says.

    The answers are drawn from **one** seed for every rung, so the floor is the same all the way up
    and only the system's accuracy moves. A real dial may well change the floor too — which is why
    :mod:`nyxara.njp.reach` measures it per rung — but a fixture that lets it drift is testing the
    drift as well as the shape. It cost a run: floors wandering by three points turned a step of a
    gentle fade into 0.163 and the fade was reported as a cliff.
    """
    rng = random.Random(SEED)
    gold = ["A" if rng.random() < skew else "B" for _ in range(ITEMS)]
    share = max(0.0, min(1.0, right(at)))
    hit = random.Random(SEED + at + 1000)
    said = [g if hit.random() < share else ("B" if g == "A" else "A") for g in gold]
    noise = random.Random(SEED + at + 2000)
    predict = ((lambda i: said[i] if noise.random() < 0.75 else "B") if wobbles
               else (lambda i: said[i]))
    return Benchmark(name=f"{at}", items=list(range(ITEMS)), gold=gold, predict=predict,
                     train=list(range(ITEMS)) if leaks else (),
                     key=(lambda i: i) if leaks else None)


# --------------------------------------------------------------------------------------------- #
#  five ladders, one of which has an edge
# --------------------------------------------------------------------------------------------- #
def _sharp(at: int) -> Benchmark:
    """Comfortably above its floor to four, then at it."""
    return _bench(at, lambda n: 0.92 if n <= 4 else 0.51)


def _graceful(at: int) -> Benchmark:
    """Fading by nine points a rung: past its floor by five, and no single step falling away.

    The first version faded too gently and still held at six, so it was `exhausted` — which was
    the right answer for what was built and the wrong fixture for what was being tested. A fade
    only differs from a cliff once it has actually crossed the floor.
    """
    return _bench(at, lambda n: 0.82 - 0.05 * (n - 1))


def _exhausted(at: int) -> Benchmark:
    """Holds everywhere the dial was turned. The dial was not turned far enough."""
    return _bench(at, lambda n: 0.90 - 0.005 * (n - 1))


def _barren(at: int) -> Benchmark:
    """At its floor everywhere. There is no capability here to bound."""
    return _bench(at, lambda _n: 0.50)


def _patchy(at: int) -> Benchmark:
    """Holds at one, two, four and six; at its floor at three and five."""
    return _bench(at, lambda n: 0.51 if n in (3, 5) else 0.90)


@dataclass
class Case:
    name: str = ""
    #: The edge, when there is one. ``None`` means there is none to report, and saying so is the
    #: right answer rather than a failure to find one.
    edge: Optional[int] = None
    #: Which of the four no-edge shapes this is, for cases where `edge` is None.
    shape: str = ""
    make: Optional[Callable[[int], Benchmark]] = None
    note: str = ""


KNOWN: Tuple[Case, ...] = (
    Case("collapses at five", 4, "sharp", _sharp, "an edge, and a cliff"),
    Case("fades across the dial", 5, "graceful", _graceful,
         "an edge like any other, and no cliff — so nowhere in particular to attack"),
    Case("holds everywhere tried", None, "exhausted", _exhausted,
         "no boundary within what was tried"),
    Case("holds nowhere", None, "barren", _barren, "nothing to put a boundary on"),
    Case("holds above where it fails", None, "patchy", _patchy, "not a boundary at all"),
)


def retrodict(cases: Sequence[Case] = KNOWN) -> Dict[str, Any]:
    """Climb all five and see what it claims."""
    rows: List[Dict[str, Any]] = []
    right = invented = missed = 0
    for case in cases:
        ladder: Ladder = climb(case.make, DIAL, name=case.name, called="difficulty")
        got = ladder.edge
        shape = ("" if got is not None else
                 "barren" if ladder.barren else
                 "exhausted" if ladder.exhausted else
                 "patchy" if ladder.patchy else
                 "graceful" if ladder.graceful else "unnamed")
        # An edge is only right if the *shape* is too: `sharp` and `graceful` reach the same rung
        # and mean different things, and a report that confuses them sends work to a setting where
        # nothing in particular breaks.
        fades = ladder.graceful and ladder.cliff is None
        shape = shape or ("graceful" if fades else "sharp")
        ok = got == case.edge and shape == (case.shape or "sharp")
        right += int(ok)
        if not ok:
            if case.edge is None and got is not None:
                invented += 1
            else:
                missed += 1
        rows.append({"case": case.name, "want": case.edge, "got": got,
                     "want_shape": case.shape, "got_shape": shape, "ok": ok,
                     "cliff": ladder.cliff, "next": ladder.next_to_build,
                     "note": case.note, "ladder": ladder})
    return {"rows": rows, "right": right, "of": len(cases),
            "invented": invented, "missed": missed}


def examine(cases: Sequence[Case] = KNOWN) -> Dict[str, Any]:
    """The exam, with its pass condition written down.

    **No invented edges** is a separate condition from getting them right, and the stricter one: a
    boundary reported where there is none becomes the next thing somebody builds.
    """
    got = retrodict(cases)
    got["passes"] = bool(got["right"] == got["of"] and got["invented"] == 0)
    return got


def run() -> Dict[str, Any]:  # pragma: no cover — a report
    got = examine()
    print("five ladders. one has an edge; the other four are the ways a capability fails to")
    print("have one, and naming the highest passing rung is wrong for every one of them.\n")
    for row in got["rows"]:
        mark = "ok" if row["ok"] else "!!"
        want = row["want"] if row["want"] is not None else f"({row['want_shape']})"
        gotv = row["got"] if row["got"] is not None else f"({row['got_shape']})"
        print(f" [{mark}] {row['case']:<28} want {str(want):<12} got {str(gotv):<12}"
              f"{'  next: ' + str(row['next']) if row['next'] is not None else ''}")
    print(f"\n  right {got['right']} of {got['of']}   invented edges {got['invented']}   "
          f"missed {got['missed']}")
    print(f"  passes {got['passes']}\n")
    for row in got["rows"]:
        print(row["ladder"].render())
        print()
    return {k: v for k, v in got.items() if k != "rows"}
