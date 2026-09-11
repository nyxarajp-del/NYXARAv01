"""NYXARA · njp/latentschool.py — does it tell a cause from a thing that travels with one (🔍).

Finding a property that separates the failures from the successes is easy and nearly worthless. Any
system with enough properties will find several, and the ones it finds will mostly be confounds,
consequences and coincidences. So this world is built with **one real cause and three impostors
that predict it just as well**, and the exam is passed only by a module that ranks all four
observationally and then lets exactly one of them through the intervention.

======================  =====================================================================
trait                   what is true of it
======================  =====================================================================
``weight``              the cause. Heavy items fail. ``do(weight)`` moves the failures.
``shadow``              a confound. ``weight`` sets it too, so it predicts perfectly and
                        ``do(shadow)`` moves nothing.
``scar``                a consequence. It is written **by** the failure, so it predicts
                        perfectly and ``do(scar)`` moves nothing.
``twin``                changing it drags ``weight`` along, so it identifies nothing and is
                        refused rather than believed.
``dust``                pure noise. The permutation null is what refuses it.
``paint``               the same on every item. A constant cannot be a cause, refused first.
``size``                real, varying, and **already on the intervention shelf** — so it is
                        not a discovery, and `uncontrolled` must leave it out.
======================  =====================================================================

``shadow`` and ``scar`` are the fixtures that matter. Both separate the groups perfectly. A module
that reports what predicts and calls it a cause scores four out of four and is wrong three times.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Sequence, Tuple

from nyxara.njp.latent import (Candidate, Trait, put_to_the_test, sift, uncontrolled)
from nyxara.njp.space import Intervention

__all__ = ["Item", "TRAITS", "SHELF", "KNOWN", "world", "outcome", "retrodict", "examine",
           "run", "SEED", "ITEMS", "HEAVY"]

SEED = 86

#: How many items the world has. Enough for a permutation null to mean something.
ITEMS = 300

#: The weight above which an item fails. The one fact the module is not told.
HEAVY = 0.60


@dataclass(frozen=True)
class Item:
    """One item, with everything about it that anybody can measure."""

    weight: float = 0.0     # the cause
    shadow: float = 0.0     # set by weight; no causal role
    scar: float = 0.0       # written by the failure
    twin: float = 0.0       # moves with weight, and drags it back
    dust: float = 0.0       # noise
    paint: float = 0.5      # the same on everything
    size: float = 0.0       # real, varying, and already on the shelf


def _fails(item: Item) -> bool:
    """The rule the module is never shown: heavy items fail, and nothing else matters."""
    return item.weight >= HEAVY


def world(n: int = ITEMS, seed: int = SEED) -> List[Item]:
    """Items whose impostors are as predictive as the cause, by construction."""
    rng = random.Random(seed)
    out: List[Item] = []
    for _ in range(n):
        weight = rng.random()
        heavy = weight >= HEAVY
        out.append(Item(
            weight=weight,
            shadow=weight * 2.0 + rng.uniform(-0.02, 0.02),   # a confound: weight sets it too
            scar=(1.0 if heavy else 0.0) + rng.uniform(-0.02, 0.02),  # written by the failure
            twin=weight + rng.uniform(-0.01, 0.01),           # moves with weight both ways
            dust=rng.random(),
            paint=0.5,
            size=rng.random()))
    return out


def outcome(items: Sequence[Item]) -> List[bool]:
    return [_fails(item) for item in items]


# --------------------------------------------------------------------------------------------- #
#  how each trait can be changed, where it can be changed at all
# --------------------------------------------------------------------------------------------- #
def _lighten(item: Item) -> Item:
    """Change the weight and nothing else. The only change that can move the failures."""
    return replace(item, weight=max(0.0, item.weight - 0.45))


def _dim(item: Item) -> Item:
    """Change the confound on its own. Weight is untouched, so the failures will not move."""
    return replace(item, shadow=max(0.0, item.shadow - 0.9))


def _heal(item: Item) -> Item:
    """Erase the consequence. The failure that wrote it is still there."""
    return replace(item, scar=0.0)


def _pull_twin(item: Item) -> Item:
    """Change the twin — and weight comes with it, which is why this identifies nothing."""
    return replace(item, twin=max(0.0, item.twin - 0.45),
                   weight=max(0.0, item.weight - 0.45))


def _blow_dust(item: Item) -> Item:
    return replace(item, dust=max(0.0, item.dust - 0.5))


def _shrink(item: Item) -> Item:
    return replace(item, size=max(0.0, item.size - 0.5))


TRAITS: Tuple[Trait, ...] = (
    Trait("weight", read=lambda i: i.weight, change=_lighten, says="how heavy the item is"),
    Trait("shadow", read=lambda i: i.shadow, change=_dim, says="cast by the weight"),
    Trait("scar", read=lambda i: i.scar, change=_heal, says="left behind by the failure"),
    Trait("twin", read=lambda i: i.twin, change=_pull_twin, says="cannot be moved on its own"),
    Trait("dust", read=lambda i: i.dust, change=_blow_dust, says="noise"),
    Trait("paint", read=lambda i: i.paint, change=None, says="the same on everything"),
    Trait("size", read=lambda i: i.size, change=_shrink, says="already on the shelf"),
)

#: What somebody already thought to vary. `size` is on it, so `size` is not a discovery however
#: well it behaves — which is the difference between finding a new variable and listing your inputs.
SHELF: Tuple[Intervention, ...] = (
    Intervention(verb="replace", on="size", change=lambda items: items),
)


@dataclass
class Case:
    trait: str = ""
    #: ``"causes"`` | ``"predicts"`` | ``"nothing"`` | ``"refused"``
    truth: str = ""
    note: str = ""


KNOWN: Tuple[Case, ...] = (
    Case("weight", "causes", "the rule; do(weight) moves the failures"),
    Case("shadow", "predicts", "a confound — predicts perfectly, do(shadow) moves nothing"),
    Case("scar", "predicts", "a consequence — written by the failure, not a cause of it"),
    Case("twin", "refused", "changing it drags weight along, so it identifies nothing"),
    Case("dust", "nothing", "noise; the permutation null is what refuses it"),
    Case("paint", "nothing", "constant; refused before any test, because a constant cannot cause"),
)


def _standing(got: Candidate) -> str:
    if got.refused:
        return "refused"
    if got.causal:
        return "causes"
    if got.claims("observational"):
        return "predicts"
    return "nothing"


def retrodict(cases: Sequence[Case] = KNOWN) -> Dict[str, Any]:
    """The whole world, sifted and then put to the test, against what is true by construction."""
    items = world()
    failed = outcome(items)
    by_name = {t.name: t for t in TRAITS}
    looked = uncontrolled(TRAITS, SHELF, items)
    found = {c.trait: c for c in sift(looked, items, failed, rng=random.Random(SEED))}
    for name, got in found.items():
        if got.refused or not got.claims("observational"):
            continue
        put_to_the_test(got, by_name[name], items, outcome=outcome, others=TRAITS)

    rows: List[Dict[str, Any]] = []
    right = 0
    invented = 0        # called something a cause that is not one
    missed = 0          # failed to call the cause a cause
    for case in cases:
        got = found.get(case.trait)
        said = _standing(got) if got is not None else "nothing"
        ok = said == case.truth
        right += int(ok)
        if said == "causes" and case.truth != "causes":
            invented += 1
        if case.truth == "causes" and said != "causes":
            missed += 1
        rows.append({"trait": case.trait, "want": case.truth, "got": said,
                     "separates": got.separates if got else 0.0,
                     "luck": got.luck if got else 1.0, "note": case.note, "candidate": got})
    predicted = sum(1 for r in rows if r["got"] in ("predicts", "causes"))
    return {"rows": rows, "right": right, "of": len(cases), "invented": invented,
            "missed": missed, "predicted": predicted,
            "causes": sum(1 for r in rows if r["got"] == "causes"),
            "looked_at": sorted(t.name for t in looked),
            "on_the_shelf": sorted({w.on for w in SHELF})}


def examine(cases: Sequence[Case] = KNOWN) -> Dict[str, Any]:
    """The exam, with its pass condition written down rather than implied.

    Five conditions. Every trait gets its own standing; **no** impostor is called a cause; the real
    one is not missed; more than one trait reaches ``observational``, because a world where only
    the cause predicts is a world where this whole module is unnecessary; and exactly one survives
    the intervention.

    That fourth condition is the one that keeps the exam honest. If ``shadow`` and ``scar`` did not
    predict as well as ``weight`` does, a module that simply reported the strongest correlation
    would pass — and that module is precisely what this one is built not to be.
    """
    got = retrodict(cases)
    got["passes"] = bool(got["right"] == got["of"]
                         and got["invented"] == 0 and got["missed"] == 0
                         and got["predicted"] >= 3
                         and got["causes"] == 1
                         and "size" not in got["looked_at"])
    return got


def run() -> Dict[str, Any]:  # pragma: no cover — a report
    got = examine()
    print(f"  varying and uncontrolled: {got['looked_at']}")
    print(f"  already on the shelf, so not a discovery: {got['on_the_shelf']}\n")
    for row in got["rows"]:
        print(f"  {row['trait']:<10} want {row['want']:<9} got {row['got']:<9} "
              f"separates {row['separates']:.4f}  luck {row['luck']:.4f}")
    print(f"\n  right {got['right']}/{got['of']}, predicted {got['predicted']}, "
          f"causes {got['causes']}, invented {got['invented']}, missed {got['missed']}, "
          f"passes {got['passes']}\n")
    for row in got["rows"]:
        if row["candidate"] is not None:
            print(row["candidate"].render())
    return got


if __name__ == "__main__":  # pragma: no cover
    run()
