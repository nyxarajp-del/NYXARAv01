"""NYXARA · njp/interactschool.py — does it find the pair, or find a pair in everything (🔗).

``I = ΔAB − ΔA − ΔB`` is a subtraction, and a subtraction of four noisy means is very good at
being large. So the exam is built around the ways it can be large **for reasons that are not an
interaction**, and half the fixtures are worlds where the correct answer is *these two are simply
additive*.

Six worlds, each a small deterministic machine whose behaviour is known by construction:

======================  ==========================================================================
world                   what is true of it, and what must be said
======================  ==========================================================================
synergy                 neither switch alone does anything; both together do. Must be found.
additive                each switch is worth the same whether the other is on. Must be refused.
inert                   neither switch does anything, together or apart. Must be refused.
antagonism              two good switches that cancel. A negative interaction is a finding too.
order-dependent         one switch consumes what the other needs. Not an interaction — a mutation.
leaky apparatus         the harness carries state, so a do-nothing change moves the number.
======================  ==========================================================================

The last two are the point of the module. Both produce a large ``I``, both look exactly like a
discovery, and neither is one — and the leaky apparatus is worse than a wrong answer, because it
invalidates every figure the harness has produced, including the flat ones that were believed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.interact import Interaction, interaction, mechanisms
from nyxara.njp.space import Held, Intervention

__all__ = ["Box", "Case", "KNOWN", "retrodict", "examine", "run", "SEED", "ITEMS", "NOISE"]

SEED = 85

#: How many items each world has. Enough that the bootstrap has something to resample and a real
#: effect is distinguishable from a run of luck; small enough that the exam is quick.
ITEMS = 400

#: How much each item's reading wobbles. Without it every bootstrap interval is a point and the
#: null is vacuous — a fixture with no noise cannot test a control whose whole job is noise.
NOISE = 0.25


# --------------------------------------------------------------------------------------------- #
#  a small machine with two switches
# --------------------------------------------------------------------------------------------- #
@dataclass
class Box:
    """Two switches, some items, and a rule saying what the switches are worth.

    ``touched`` is what makes the leaky fixture possible: a box that records having been changed
    can behave differently next time, which is exactly the state a real harness accumulates
    without anybody deciding that it should.
    """

    a: bool = False
    b: bool = False
    seed: int = SEED
    #: What each item reads, given the switches. Supplied per world.
    rule: Optional[Callable[["Box", int], float]] = None
    touched: int = 0
    #: Something switch A consumes and switch B needs. One number, and it is what makes an
    #: order-dependent world expressible at all — see :func:`_ordered`.
    fuel: int = 1
    #: Whether B found any fuel left when it was flipped. **Only** :func:`_ordered` reads it; every
    #: other world reads the switches and is untouched by the mechanic. The first attempt put the
    #: dependence in the intervention instead, so A consumed the fuel in all six worlds and three
    #: fixtures that had nothing to do with ordering came back order-dependent. A control's fixture
    #: must not change the worlds it is not about.
    b_caught: bool = False

    def copy(self, **changes: Any) -> "Box":
        out = Box(a=self.a, b=self.b, seed=self.seed, rule=self.rule, touched=self.touched,
                  fuel=self.fuel, b_caught=self.b_caught)
        for key, value in changes.items():
            setattr(out, key, value)
        return out

    def read(self) -> List[float]:
        """One reading per item, deterministic in the box's own seed so runs are comparable."""
        rng = random.Random(self.seed)
        rule = self.rule or (lambda _box, _i: 0.0)
        return [max(0.0, min(1.0, rule(self, i) + rng.uniform(-NOISE, NOISE)))
                for i in range(ITEMS)]


def _measure(box: Box) -> List[float]:
    return box.read()


def _flip_a(box: Box) -> Box:
    """Switch A on, and the fuel spent. Spending it is what the ordered world turns on."""
    return box.copy(a=True, fuel=0, touched=box.touched + 1)


def _flip_b(box: Box) -> Box:
    """Switch B on, and a record of whether there was any fuel left when it went on."""
    return box.copy(b=True, b_caught=bool(box.fuel), touched=box.touched + 1)


def _nothing(box: Box) -> Box:
    """A change that changes nothing — except that it happened, which is the control."""
    return box.copy(touched=box.touched + 1)


#: Every intervention promises the item count holds. A world that quietly drops items would make
#: the subtraction meaningless, and this is how that is caught rather than assumed away.
SAME_ITEMS = Held(name="the same items are read", read=lambda box: float(len(box.read())))

FLIP_A = Intervention(verb="add", on="switch A", change=_flip_a, holds=(SAME_ITEMS,))
FLIP_B = Intervention(verb="add", on="switch B", change=_flip_b, holds=(SAME_ITEMS,))
NOTHING = Intervention(verb="hold", on="nothing at all", change=_nothing, holds=(SAME_ITEMS,))


# --------------------------------------------------------------------------------------------- #
#  six worlds
# --------------------------------------------------------------------------------------------- #
def _synergy(box: Box, _i: int) -> float:
    """Neither switch alone; both together. The thing a single-variable search cannot see."""
    return 0.70 if (box.a and box.b) else 0.20


def _additive(box: Box, _i: int) -> float:
    """Each switch worth 0.15 whatever the other is doing. ``I`` must come out at zero."""
    return 0.20 + 0.15 * box.a + 0.15 * box.b


def _inert(box: Box, _i: int) -> float:
    return 0.20


def _antagonism(box: Box, _i: int) -> float:
    """Two switches that are each worth something and cancel when both are on."""
    return 0.20 + 0.20 * box.a + 0.20 * box.b - 0.35 * (box.a and box.b)


def _ordered(box: Box, _i: int) -> float:
    """Both switches are worth a lot together — but A spends the fuel B needs to catch.

    So ``B then A`` has both on and ``A then B`` has only A, and the difference between them is
    not an interaction, it is a **mutation**: the second change sees a world the first has already
    altered. That is how it happens in a real harness, and it is the reason
    :func:`~nyxara.njp.interact.interaction` runs both orders rather than assuming composition
    commutes.

    The first version of this fixture wrote the dependence through ``touched``, which both orders
    increment identically — so both orders came out the same, the order effect measured 0.0000, and
    the fixture tested nothing at all. A fixture that cannot fail proves nothing about a control
    that passes it.
    """
    return 0.70 if (box.a and box.b and box.b_caught) else 0.20


def _leaky(box: Box, _i: int) -> float:
    """The harness responds to having been touched, whether or not anything was changed.

    Deliberately **linear** in ``touched``, which is the commonest shape a real leak takes — a
    counter that advances, a cache that warms, a seed that is consumed. It is also the shape that
    cancels exactly out of an interaction, which is how the first version of the apparatus control
    came back clean on this fixture. It is read directly now.
    """
    return 0.20 + 0.20 * box.touched


@dataclass
class Case:
    name: str = ""
    rule: Optional[Callable[[Box, int], float]] = None
    #: What must be said. One of the :attr:`Interaction.verdict` values.
    verdict: str = "additive"
    note: str = ""


KNOWN: Tuple[Case, ...] = (
    Case("synergy", _synergy, "interacting",
         "neither alone, both together — the case single-variable search cannot reach"),
    Case("additive", _additive, "additive",
         "each switch worth the same whichever the other is; I must be zero"),
    Case("inert", _inert, "additive", "nothing does anything; a pair here would be pure noise"),
    Case("antagonism", _antagonism, "interacting",
         "two good switches that cancel — a negative interaction is still a finding"),
    Case("order-dependent", _ordered, "order-dependent",
         "one consumes what the other needs; large I, and not an interaction"),
    Case("leaky apparatus", _leaky, "apparatus",
         "a do-nothing change moves it, so every figure this harness produced is suspect"),
)


def _look(case: Case) -> Interaction:
    box = Box(rule=case.rule)
    return interaction(FLIP_A, FLIP_B, box, measure=_measure, nothing=NOTHING,
                       rng=random.Random(SEED))


def retrodict(cases: Sequence[Case] = KNOWN) -> Dict[str, Any]:
    """Six worlds whose answer is known by construction, and what the search made of them."""
    rows: List[Dict[str, Any]] = []
    right = 0
    invented = 0        # said `interacting` where nothing interacts
    missed = 0          # said otherwise where something does
    for case in cases:
        got = _look(case)
        ok = got.verdict == case.verdict
        right += int(ok)
        if not ok:
            if got.verdict == "interacting":
                invented += 1
            elif case.verdict == "interacting":
                missed += 1
        rows.append({"case": case.name, "want": case.verdict, "got": got.verdict,
                     "extra": got.extra, "luck": got.luck, "order": got.order,
                     "placebo": got.placebo, "note": case.note, "interaction": got})
    return {"rows": rows, "right": right, "of": len(cases),
            "invented": invented, "missed": missed,
            "real": sum(1 for c in cases if c.verdict == "interacting")}


def examine(cases: Sequence[Case] = KNOWN) -> Dict[str, Any]:
    """The exam, with its pass condition written down rather than implied.

    Every world must get its own verdict — not merely *found* against *not found*, because
    ``order-dependent`` and ``apparatus`` both produce a large ``I`` and calling either of them an
    interaction is the failure this module exists to prevent. ``invented`` is the number to watch:
    a search that answers *they interact* to everything scores four of six on a looser exam and
    zero on this one.
    """
    got = retrodict(cases)
    got["passes"] = bool(got["right"] == got["of"] and got["invented"] == 0 and got["missed"] == 0)
    return got


def run() -> Dict[str, Any]:  # pragma: no cover — a report
    got = examine()
    for row in got["rows"]:
        print(f"  {row['case']:<18} want {row['want']:<16} got {row['got']:<16} "
              f"I {row['extra']:+.4f}  luck {row['luck']:.3f}  order {row['order']:+.4f}  "
              f"placebo {row['placebo']:+.4f}")
    print(f"\n  right {got['right']}/{got['of']}, invented {got['invented']}, "
          f"missed {got['missed']}, passes {got['passes']}\n")
    found = _look(KNOWN[0])
    print(found.render())
    print("\n  and what it is consistent with, which on the day it is found is all of it:")
    for why in mechanisms(found):
        print(f"    · {why}")
    return got


if __name__ == "__main__":  # pragma: no cover
    run()
