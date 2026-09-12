"""NYXARA · njp/fusionschool.py — how often does it see a shape that is not there (🧩📏).

:mod:`nyxara.njp.fusion` finds the same structure in two subjects that never met. Its own docstring
names the danger exactly — *"matching on four of five edges is exactly the false analogy that makes
this kind of system"* worthless — and it guards against it with exact isomorphism, a minimum edge
count and a bounded radius.

**Every one of those guards is an argument.** `MIN_EDGES = 3` is justified by "below this an
isomorphism is arithmetic rather than a finding: every one-edge subgraph matches every other",
which is true and is not a measurement. Nothing here has ever been shown a pair of domains with no
relationship and asked what it says about them.

That is the number an analogy finder lives or dies by, because the failure is not missing a real
analogy — it is producing one from any two graphs put in front of it. A finder that reports an
analogy between every pair of domains has told nobody anything, and reads as insight.

So two questions, scored together:

* **planted** — two domains built around the same structure in disjoint vocabularies. Does it find
  the analogy that is there? A guard tight enough to never fire is not a guard, it is a refusal.
* **unrelated** — two domains generated independently, with no structure in common beyond what
  chance provides. Does it claim one anyway?

And the bar is swept rather than argued: `MIN_EDGES` from one upward, so that what it buys and what
it costs are a table instead of a sentence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from nyxara.njp.fusion import LUCK, MAX_RADIUS, MIN_EDGES, Fusion

__all__ = ["Store", "Explainer", "Report", "planted", "unrelated", "examine", "sweep",
           "density", "run", "SEED", "SHAPES"]

SEED = 80

#: How many domain pairs each figure is averaged over. Enough that a false-alarm rate of a few
#: percent is distinguishable from zero, which one of a dozen pairs would not be.
PAIRS = 120

#: How many nodes an unrelated domain gets, and how many edges. Chosen to be the **same size** as
#: the planted domains: a false-alarm rate measured on smaller graphs than the real ones would
#: flatter the finder, since a small graph has fewer chances to match.
NODES = 8
EDGES = 10


# --------------------------------------------------------------------------------------------- #
#  the smallest store the finder will read from
# --------------------------------------------------------------------------------------------- #
@dataclass
class Store:
    """A fact store shaped the way :class:`~nyxara.njp.fusion.Fusion` reads one."""

    rows: Sequence[Tuple[str, str, str]] = ()
    facts: Dict[Tuple[str, str], List[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for subject, relation, obj in self.rows:
            self.facts.setdefault((subject.lower(), relation), []).append(_Fact(obj))


@dataclass(frozen=True)
class _Fact:
    object: str = ""
    confidence: float = 1.0


class Explainer:
    """The one method :class:`~nyxara.njp.fusion.Fusion` calls, and nothing else."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def _out(self, subject: str, relation: str) -> List[Tuple[str, float]]:
        return [(f.object, f.confidence)
                for f in self.store.facts.get((self._key(subject), relation), [])]

    @staticmethod
    def _key(text: Any) -> str:
        return " ".join(str(text or "").split()).lower()


def _fuser(rows: Sequence[Tuple[str, str, str]], **kw: Any) -> Fusion:
    return Fusion(Explainer(Store(rows)), **kw)


# --------------------------------------------------------------------------------------------- #
#  the two kinds of pair
# --------------------------------------------------------------------------------------------- #
def _ring(names: Sequence[str], relation: str = "causes") -> List[Tuple[str, str, str]]:
    return [(names[i], relation, names[(i + 1) % len(names)]) for i in range(len(names))]


def _words(rng: random.Random, tag: str, n: int) -> List[str]:
    return [f"{tag}{i}_{rng.randrange(10_000)}" for i in range(n)]


def planted(rng: random.Random, *, size: int = 4) -> Tuple[List[Tuple[str, str, str]],
                                                           Dict[str, List[str]]]:
    """Two domains around the same cycle, sharing no word. The analogy is there by construction."""
    left, right = _words(rng, "a", size), _words(rng, "b", size)
    rows = _ring(left) + _ring(right)
    return rows, {"left": [left[0]], "right": [right[0]]}


def unrelated(rng: random.Random, *, nodes: int = NODES, edges: int = EDGES,
              relation: str = "causes"
              ) -> Tuple[List[Tuple[str, str, str]], Dict[str, List[str]]]:
    """Two domains drawn independently. Any analogy between them is the finder's own invention.

    **One relation, like the planted pairs**, and this is the whole design of the control. The
    first version drew each edge from all nine structural relations at random, which made a
    labelled isomorphism between two random graphs nearly impossible — and duly reported a
    false-alarm rate of **0.000 at every bar**, a flawless-looking number produced by a negative
    control strictly easier to reject than the positive was to accept. A floor measured on a softer
    problem than the finding is not a floor.

    So these are the same size, the same density and the same single relation as the shapes that
    are meant to be found. The only thing they lack is a shared structure, which is the one thing
    being tested for.
    """
    left, right = _words(rng, "x", nodes), _words(rng, "y", nodes)
    rows: List[Tuple[str, str, str]] = []
    for names in (left, right):
        seen = set()
        while len(seen) < edges:
            a, b = rng.choice(names), rng.choice(names)
            if a != b and (a, b) not in seen:
                seen.add((a, b))
                rows.append((a, relation, b))
    return rows, {"left": [left[0]], "right": [right[0]]}


# --------------------------------------------------------------------------------------------- #
#  the exam
# --------------------------------------------------------------------------------------------- #
@dataclass
class Report:
    """What the finder said about pairs that had a shape, and pairs that did not."""

    min_edges: int = MIN_EDGES
    radius: int = MAX_RADIUS
    luck: float = LUCK
    nodes: int = NODES
    edges: int = EDGES
    found: int = 0
    planted_pairs: int = 0
    claimed: int = 0
    unrelated_pairs: int = 0

    @property
    def recall(self) -> float:
        """Of the analogies that were there, how many it found."""
        return round(self.found / self.planted_pairs, 4) if self.planted_pairs else 0.0

    @property
    def false_alarms(self) -> float:
        """Of the pairs with nothing in common, how many it claimed an analogy between."""
        return round(self.claimed / self.unrelated_pairs, 4) if self.unrelated_pairs else 0.0

    @property
    def worth(self) -> float:
        """Recall less the false-alarm rate. A finder that fires on everything scores zero."""
        return round(self.recall - self.false_alarms, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {"min_edges": self.min_edges, "radius": self.radius, "luck": self.luck,
                "nodes": self.nodes, "edges": self.edges, "recall": self.recall,
                "false_alarms": self.false_alarms, "worth": self.worth,
                "planted": self.planted_pairs, "unrelated": self.unrelated_pairs}

    def render(self) -> str:
        return (f"  min_edges {self.min_edges:<3} finds {self.recall:.3f} of what is there, "
                f"claims {self.false_alarms:.3f} of what is not   worth {self.worth:+.3f}")


def examine(pairs: int = PAIRS, *, min_edges: int = MIN_EDGES, radius: int = MAX_RADIUS,
            seed: int = SEED, size: int = 4, luck: float = LUCK,
            nodes: int = NODES, edges: int = EDGES) -> Report:
    """Show it `pairs` domains that share a shape and `pairs` that do not.

    ``luck`` at zero turns the surprise guard off, which is how the ablation is run: the guard has
    to be shown to matter by being taken away, and on small dense domains it takes the invented
    rate from 0.213 to 0.000 while leaving recall at 1.000.
    """
    out = Report(min_edges=min_edges, radius=radius, luck=luck, nodes=nodes, edges=edges)
    rng = random.Random(seed)
    for _ in range(pairs):
        rows, seeds = planted(rng, size=size)
        out.planted_pairs += 1
        out.found += int(bool(_fuser(rows, min_edges=min_edges,
                                     radius=radius).analogies(seeds, luck=luck)))
    for _ in range(pairs):
        rows, seeds = unrelated(rng, nodes=nodes, edges=edges)
        out.unrelated_pairs += 1
        out.claimed += int(bool(_fuser(rows, min_edges=min_edges,
                                       radius=radius).analogies(seeds, luck=luck)))
    return out


#: The domain shapes the density sweep walks. The last two are where the finder used to invent.
SHAPES: Tuple[Tuple[int, int], ...] = ((8, 10), (6, 8), (5, 8), (4, 6), (4, 10), (3, 4))


def density(pairs: int = PAIRS, shapes: Sequence[Tuple[int, int]] = SHAPES,
            *, luck: float = LUCK) -> List[Report]:
    """Where the finder starts inventing: small dense domains, and how far the guard reaches."""
    return [examine(pairs, nodes=n, edges=e, luck=luck) for n, e in shapes]


def sweep(pairs: int = PAIRS, bars: Sequence[int] = (1, 2, 3, 4, 5, 6)) -> List[Report]:
    """What the minimum edge count buys and what it costs, as a table rather than a sentence."""
    return [examine(pairs, min_edges=bar) for bar in bars]


def run(pairs: int = PAIRS) -> Dict[str, Any]:  # pragma: no cover — a report
    print(f"{pairs} domain pairs that share a shape, and {pairs} that do not.\n")
    rows = sweep(pairs)
    for report in rows:
        print(report.render())
    best = max(rows, key=lambda r: r.worth)
    here = next((r for r in rows if r.min_edges == MIN_EDGES), None)
    print(f"\n  the bar is set at {MIN_EDGES}, which scores {here.worth:+.3f}"
          if here else "")
    print(f"  the best bar swept is {best.min_edges}, at {best.worth:+.3f}")
    print("\nand where it starts inventing — small dense domains, with the guard off and on:\n")
    print("  nodes edges |  guard off   guard on   recall")
    off = density(pairs, luck=0.0)
    on = density(pairs, luck=LUCK)
    for a, b in zip(off, on):
        print(f"  {a.nodes:>5} {a.edges:>5} |     {a.false_alarms:.3f}      {b.false_alarms:.3f}"
              f"    {b.recall:.3f}")
    return {"sweep": [r.to_dict() for r in rows],
            "at_the_set_bar": here.to_dict() if here else None,
            "best": best.to_dict(),
            "density_guard_off": [r.to_dict() for r in off],
            "density_guard_on": [r.to_dict() for r in on]}
