"""NYXARA · njp/genesisschool.py — three worlds, no vocabulary, and one quantity nobody named (📐).

The exam is built so the obvious answer is useless.

In all three worlds the **centre reading is matched** between the items that failed and the items
that did not. Every quantity anybody would compute about an item on its own — its score, its mean,
its best reading — is the same on both sides. What differs is the *shape of the neighbourhood
around it*: the failures come to a point and the successes sit on a plateau.

There is no word for that in anything this package supplies, and there is no word for it here
either. Nothing in this module defines a trait, a property, or a semantic name. The worlds emit
probes at numbered coordinates and the readings taken there, and that is all.

Three worlds, deliberately unrelated on the surface:

======================  ====================================================================
world                   what a probe is, and what a reading is
======================  ====================================================================
``spans``               a boundary moved a token; how well the answer scored
``gains``               a control gain turned up or down; how steady the loop ran
``doses``               a dose raised or lowered; how well the organism did
======================  ====================================================================

If one ``(operator, axis)`` is mined in the first, survives its gates in the second, and still
separates the groups in the third, then what was found is not a fact about spans. It is a
regularity that three unrelated arithmetics share — and it arrives called ``measurement 7``,
because naming it is a separate act performed afterwards by somebody who can defend the name.

**And a fourth world where the right answer is nothing.** ``fog`` has the same shape of trace, the
same probes, the same spread of readings, and no relationship between any of it and which items
failed. A register that is not empty there has learned to find structure rather than to find
*this* structure, which is the failure that matters and the one that looks like success.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

from nyxara.njp.genesis import ENDS, STEPS, Measurement, Register, Trace, admit

__all__ = ["WORLDS", "KNOWN", "Case", "ITEMS", "REACH", "NOISE", "SEED", "spans", "gains",
           "doses", "fog", "seeded", "retrodict", "examine", "run"]

SEED = 87

#: Items per world. Enough for a permutation null over the failures to mean something.
ITEMS = 240

#: How far either side of centre a probe is taken. Five probes: −2, −1, 0, +1, +2.
REACH = 2

#: Observation noise. Without it the steadiness gate is untestable, and a measurement that has
#: never been jiggled has not been shown to hold still.
NOISE = 0.015


def _trace(centre: float, width: float, rng: random.Random, failed: bool) -> Trace:
    """One neighbourhood: a reading at centre, falling away at a rate ``width`` decides.

    ``centre`` is drawn from the **same** distribution whichever side the item is on, and that is
    the fixture's whole construction. The two groups are indistinguishable by any quantity computed
    from the middle reading alone; they differ only in how fast the neighbourhood falls away.
    """
    probes: List[Tuple[float, ...]] = []
    readings: List[float] = []
    for step in range(-REACH, REACH + 1):
        probes.append((float(step),))
        fall = min(1.0, (abs(step) / max(1e-9, width)) ** 2)
        readings.append(max(0.0, centre * (1.0 - fall) + rng.uniform(-NOISE, NOISE)))
    return Trace(probes=tuple(probes), readings=tuple(readings), failed=failed)


def _world(n: int, seed: int, *, narrow: float, broad: float,
           low: float, high: float) -> List[Trace]:
    """Half the items sharp and half of them flat, with the centre reading matched across both."""
    rng = random.Random(seed)
    out: List[Trace] = []
    for i in range(n):
        failed = i % 2 == 0
        centre = rng.uniform(low, high)            # the same draw on both sides
        width = rng.uniform(*(narrow if failed else broad))
        out.append(_trace(centre, width, rng, failed))
    return out


def spans(n: int = ITEMS, seed: int = SEED) -> List[Trace]:
    """A boundary moved a token either way; how well the answer scored there."""
    return _world(n, seed, narrow=(0.7, 1.1), broad=(2.4, 3.6), low=0.30, high=0.95)


def gains(n: int = ITEMS, seed: int = SEED + 1) -> List[Trace]:
    """A control gain turned up or down; how steady the loop ran. A different arithmetic entirely."""
    return _world(n, seed, narrow=(0.6, 1.0), broad=(2.6, 4.0), low=0.20, high=0.80)


def doses(n: int = ITEMS, seed: int = SEED + 2) -> List[Trace]:
    """A dose raised or lowered; how well the organism did. Nothing to do with either of the above."""
    return _world(n, seed, narrow=(0.8, 1.2), broad=(2.2, 3.2), low=0.40, high=0.99)


def fog(n: int = ITEMS, seed: int = SEED + 3) -> List[Trace]:
    """The same kind of trace, and nothing in it has anything to do with which items failed.

    Widths are drawn from **one** distribution for both groups, so every shape is as likely on
    either side. A register that is not empty here has learned to find structure rather than to
    find this structure.
    """
    rng = random.Random(seed)
    out: List[Trace] = []
    for i in range(n):
        out.append(_trace(rng.uniform(0.30, 0.95), rng.uniform(0.7, 3.6), rng, i % 2 == 0))
    return out


WORLDS: Dict[str, Callable[..., List[Trace]]] = {
    "spans": spans, "gains": gains, "doses": doses, "fog": fog,
}


def seeded() -> List[Measurement]:
    """The register as it already stands: the quantity anybody would have computed anyway.

    ``level`` is the average reading — the score, essentially — and it is put here so that
    rediscovering it is refused as the duplicate it is. Without this the module "discovers" the
    thing it was already looking at and reports it as an invention.
    """
    one = Measurement(steps=(), end="mean", axis=0, number=0, called="the average reading")
    one.held.append(__import__("nyxara.njp.latent", fromlist=["Standing"]).Standing(
        kind="observational", says="supplied; what anybody would compute without being asked"))
    return [one]


# --------------------------------------------------------------------------------------------- #
#  the exam
# --------------------------------------------------------------------------------------------- #
@dataclass
class Case:
    name: str = ""
    #: Worlds in order: mine on the first, criticise on the second, transfer to the third.
    order: Tuple[str, str, str] = ()
    #: Must anything survive?
    expect: bool = True
    note: str = ""


KNOWN: Tuple[Case, ...] = (
    Case("spans → gains → doses", ("spans", "gains", "doses"), True,
         "mined in one arithmetic, checked in a second, transferred to a third"),
    Case("doses → spans → gains", ("doses", "spans", "gains"), True,
         "the same regularity found from a different starting world"),
    Case("gains → doses → spans", ("gains", "doses", "spans"), True,
         "and from the third; a regularity does not depend on where you start"),
    Case("fog → fog → fog", ("fog", "fog", "fog"), False,
         "nothing here relates to the failures; the register must come back empty"),
    Case("spans → fog → doses", ("spans", "fog", "doses"), False,
         "checked in a world where it means nothing — the gate is the second world"),
    Case("spans → gains → fog", ("spans", "gains", "fog"), False,
         "survives two worlds and not the third; transfer is not optional"),
)


def _look(case: Case) -> Register:
    a, b, c = (WORLDS[name]() for name in case.order)
    return admit(a, b, c, seeded=seeded(), rng=random.Random(SEED))


def retrodict(cases: Sequence[Case] = KNOWN) -> Dict[str, Any]:
    """Six orderings whose right answer is known by construction."""
    rows: List[Dict[str, Any]] = []
    right = 0
    invented = 0        # kept something where nothing should survive
    missed = 0          # kept nothing where something should
    for case in cases:
        got = _look(case)
        found = bool(got.kept)
        ok = found == case.expect
        right += int(ok)
        invented += int(found and not case.expect)
        missed += int(case.expect and not found)
        rows.append({"case": case.name, "want": case.expect, "kept": got.names,
                     "best": got.kept[0].to_dict() if got.kept else None,
                     "note": case.note, "register": got})
    every = [set(r["kept"]) for r in rows if r["want"] and r["kept"]]
    shared = sorted(set.intersection(*every)) if every else []
    return {"rows": rows, "right": right, "of": len(cases), "invented": invented,
            "missed": missed, "shared": shared,
            "steps": sorted(STEPS), "ends": sorted(ENDS)}


def examine(cases: Sequence[Case] = KNOWN) -> Dict[str, Any]:
    """The exam, with its pass condition written down rather than implied.

    Four conditions. Every ordering gets its answer; **nothing** survives the three orderings where
    nothing should; the real ones are not missed; and at least one measurement is common to all
    three that worked — because three unrelated worlds each yielding a *different* quantity is three
    facts about three arithmetics, not a regularity.
    """
    got = retrodict(cases)
    got["passes"] = bool(got["right"] == got["of"]
                         and got["invented"] == 0 and got["missed"] == 0
                         and len(got["shared"]) >= 1)
    return got


def run() -> Dict[str, Any]:  # pragma: no cover — a report
    got = examine()
    for row in got["rows"]:
        want = "something" if row["want"] else "nothing"
        print(f"  {row['case']:<26} want {want:<10} kept {row['kept'] or '[]'}")
    print(f"\n  right {got['right']}/{got['of']}, invented {got['invented']}, "
          f"missed {got['missed']}, passes {got['passes']}")
    print(f"  survived every ordering that worked: {len(got['shared'])} of them\n")
    first = _look(KNOWN[0])
    print(f"  {len(first.kept)} kept, {len(first.refused)} refused. The strongest:")
    for one in sorted(first.kept, key=lambda m: -m.separates)[:5]:
        print(one.render())
    print("\n  and why things were turned away:")
    seen = set()
    for one in first.refused:
        why = one.refused.split("(")[0].split("—")[0][:46]
        if why not in seen:
            seen.add(why)
            print(f"    {one.recipe:<24} {one.refused[:84]}")
    return got


if __name__ == "__main__":  # pragma: no cover
    run()
