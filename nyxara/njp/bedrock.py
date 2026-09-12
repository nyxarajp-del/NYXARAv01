"""NYXARA · njp/bedrock.py — the family of languages, taken off the shelf (🪨, NJP V.97).

V.96 priced a tower: a way costs bits, the language it was picked from costs bits, and the height
worth climbing to comes out of the arithmetic rather than out of anybody's preference. It left one
line unpaid, and named it: **the family of languages was five entries long because I wrote five.**

The obvious next version writes a family of families and moves the same debt one storey up. So this
one does not. Instead the family is **derived**, by a rule with no free numbers in it, and the rules
compete:

======================  ====================================================================
rule                     the family it gives
======================  ====================================================================
``as long as it takes``  word lengths one up to the shortest that could reach every
                         observation — a bound **computed from the data**
``a handful``            lengths one to five, which is what V.96 used
``one for each``         one length per observation
``just the shortest``    length one only
======================  ====================================================================

**Two things make this different from adding a storey.** First, a rule takes no parameters, so it
cannot be tuned toward an answer — where a five-entry list has five knobs, ``as long as it takes``
has none, and the family it produces changes with the observations rather than with me. Second,
**V.96's own hand-written choice is in the list as a competitor**, charged identically. If a rule I
made up beats the rules derived from the data, that is the finding and it gets written down.

**And the rules are charged too**, because the point of V.95 and V.96 was that a choice costs
whatever it costs to say which choice was made. Four rules is two bits. The regress does not end
here either — somebody chose four rules — but the bill now says so on its own line, and the thing
being bought with those two bits is a family that **varies with the data instead of with the
author**, which is a different kind of purchase from anything below it.

**V.98 — and then the layer turned out not to pay, which was provable without running anything.**
A shelf exists for exactly one purpose: to let you name one language. So it can be compared against
naming that language straight out of the bound the data already gives. The language's own toll and
the description are identical either way, so they cancel, and what is left is:

    via a shelf   log₂(rules) + log₂(family)
    directly      log₂(L_max)

    **the shelf pays if and only if  rules × family < L_max**

``L_max`` is :func:`reach_needed`, a function of the observations with no free numbers in it, so it
costs nothing and both sides have it. On the eight observations V.97 was built around, ``L_max`` is
three and the shelf is four rules over a family of three: twelve against three. Measured, the two
routes are **57.17 directly** and **59.17 through the best rule** — a loss of exactly 2.00 bits,
which is ``log₂(4)``, the cost of choosing among four rules, buying nothing at all.

So V.97's headline was right about which rule is best and wrong about whether having rules was worth
it. The layer stays in the module because :func:`pays` says when it would be worth it — a shelf
earns its place on data needing long words and a small family — and deleting the machinery would
delete the ability to say that. But :func:`direct` is the baseline now, and on this data it wins.

Pure standard library, and every price is V.92's, V.95's or V.96's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from nyxara.njp.combining import charged
from nyxara.njp.generators import Observed, WIDTH, longhand
from nyxara.njp.tower import Language, price

__all__ = ["Rule", "RULES", "Footing", "reach_needed", "found_on", "stand", "direct", "pays",
           "SEED"]

SEED = 97


def reach_needed(observed: Sequence[Observed]) -> int:
    """The shortest word length that could possibly reach every observation.

    Each extra letter at most doubles what a set of seeds can build, so reaching ``n`` things needs
    at least ``log₂ n`` letters. **This is read off the observations**, which is the whole point:
    nothing about the number comes from the author, and a different set of transformations gives a
    different family without anybody editing anything.
    """
    count = max(1, len(set(observed)))
    length = 1
    while 2 ** length < count:
        length += 1
    return length


@dataclass(frozen=True)
class Rule:
    """A way of producing a family of languages. **No free numbers**, which is the point.

    A rule with a knob in it can be turned toward an answer. These have none: given the same
    observations each produces the same family, and given different observations the family moves
    on its own.
    """

    name: str = ""
    of: Optional[Callable[[Sequence[Observed]], List[int]]] = None
    says: str = ""

    def family(self, observed: Sequence[Observed]) -> List[Language]:
        lengths = self.of(observed) if self.of else [3]
        return [Language(n) for n in lengths if n >= 1]


#: The rules, including the one V.96 used. Putting a hand-written choice in as a competitor is the
#: only way to find out whether it was a good one, and it is charged exactly like the rest.
RULES: Tuple[Rule, ...] = (
    Rule("as long as it takes", lambda obs: list(range(1, reach_needed(obs) + 1)),
         "up to the shortest length that could reach everything, read off the data"),
    Rule("a handful", lambda obs: [1, 2, 3, 4, 5],
         "what V.96 used, here as a competitor rather than as the floor"),
    Rule("one for each", lambda obs: list(range(1, max(1, len(set(obs))) + 1)),
         "one length per observation, which is generous and pays for being so"),
    Rule("just the shortest", lambda obs: [1],
         "the smallest family there is; it should lose, and losing is informative"),
)


@dataclass
class Footing:
    """One rule, the family it gave, and what the whole tower cost standing on it."""

    rule: Optional[Rule] = None
    family: List[Language] = field(default_factory=list)
    #: Bits to say which rule, which language, and which way — then the description itself.
    choosing_a_rule: float = 0.0
    choosing_a_language: float = 0.0
    choosing_a_way: float = 0.0
    description: float = 0.0
    explained: bool = False
    chose: str = ""

    @property
    def toll(self) -> float:
        return self.choosing_a_rule + self.choosing_a_language + self.choosing_a_way

    @property
    def total(self) -> float:
        return round(self.toll + self.description, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {"rule": self.rule.name if self.rule else "", "family": len(self.family),
                "lengths": [lang.longest for lang in self.family],
                "rule_bits": round(self.choosing_a_rule, 2),
                "language_bits": round(self.choosing_a_language, 2),
                "way_bits": round(self.choosing_a_way, 2),
                "toll": round(self.toll, 2), "description": round(self.description, 2),
                "total": self.total, "explained": self.explained, "chose": self.chose}

    def render(self) -> str:
        mark = "ok " if self.explained else " ~ "
        name = self.rule.name if self.rule else "?"
        return (f"  {mark} {name:<22}{len(self.family):>4} langs  "
                f"{self.choosing_a_rule:>4.1f}+{self.choosing_a_language:>4.1f}"
                f"+{self.choosing_a_way:>4.1f} + {self.description:>7.1f} = {self.total:>8.1f}"
                f"   {self.chose}")


def found_on(observed: Sequence[Observed], rule: Rule, *, rules: int = len(RULES),
             width: int = WIDTH) -> Footing:
    """Price the whole tower standing on one rule, charging for every choice along the way."""
    unique = sorted(set(observed))
    family = rule.family(unique)
    out = Footing(rule=rule, family=family, choosing_a_rule=charged(max(1, rules)))
    if not family:
        out.description = longhand(len(unique), width)
        out.chose = "the rule gave no languages at all"
        return out
    out.choosing_a_language = charged(len(family))
    best: Optional[Tuple[float, Language, float, bool, str]] = None
    for language in family:
        bare, ok, note = price(unique, language, width=width)
        here = language.toll + bare
        if best is None or here < best[0]:
            best = (here, language, bare, ok, note)
    _, chosen, bare, ok, note = best
    out.choosing_a_way = chosen.toll
    out.description = bare
    out.explained = ok
    out.chose = f"{chosen.name}; {note}" if ok else "nothing in this family paid"
    return out


def pays(rules: int, family: int, bound: int) -> bool:
    """Would a shelf of this shape be worth having? Derived, not searched.

    Both routes name one language and then pay the same toll and the same description, so those
    cancel. What is left is the naming: ``log₂(rules) + log₂(family)`` against ``log₂(bound)``,
    which is ``rules × family < bound``. No observations are needed to answer it — only the three
    numbers — and that is why V.98 did not run a search to find out.
    """
    return max(1, rules) * max(1, family) < max(1, bound)


def direct(observed: Sequence[Observed], *, width: int = WIDTH) -> Footing:
    """Name the language straight out of the bound the data gives, with no shelf at all.

    The baseline every rule has to beat. The bound is :func:`reach_needed`, which has no free
    numbers, so nobody pays to choose it — and a shelf that cannot beat this is a layer of
    indirection charging for itself.
    """
    unique = sorted(set(observed))
    bound = reach_needed(unique)
    out = Footing(rule=Rule("no shelf at all", None, "name the language from the data's own bound"),
                  family=[Language(n) for n in range(1, bound + 1)],
                  choosing_a_rule=0.0, choosing_a_language=charged(bound))
    best: Optional[Tuple[float, Language, float, bool, str]] = None
    for length in range(1, bound + 1):
        language = Language(length)
        bare, ok, note = price(unique, language, width=width)
        here = language.toll + bare
        if best is None or here < best[0]:
            best = (here, language, bare, ok, note)
    _, chosen, bare, ok, note = best
    out.choosing_a_way = chosen.toll
    out.description = bare
    out.explained = ok
    out.chose = f"{chosen.name}; {note}" if ok else "nothing at any length paid"
    return out


def stand(observed: Sequence[Observed], *, rules: Sequence[Rule] = RULES,
          width: int = WIDTH) -> List[Footing]:
    """Every rule priced on the same observations — **and no shelf at all** — cheapest first.

    The baseline is in the list rather than beside it, because a route that is never made to
    compete is a route nobody has checked.
    """
    out = [found_on(observed, rule, rules=len(rules), width=width) for rule in rules]
    out.append(direct(observed, width=width))
    return sorted(out, key=lambda f: f.total)
