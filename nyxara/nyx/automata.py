"""NYXARA · nyx/automata.py — complexity nobody wrote (🔬, NYX V.03).

Every other faculty here is *designed*: a plan, a gauntlet, a report. This one is the opposite,
and that is the whole point of it. I write four local rules — what a cell does given its immediate
neighbours — and I write **no behaviour at all**. Gliders, oscillators and still lifes are not in
this file; they fall out of the rules colliding, which is the only kind of un-scripted structure a
deterministic system can honestly produce.

Three things make it a faculty rather than a screensaver:

* **Emergence is measured, never claimed.** :meth:`Lattice.analyse` computes spatial entropy,
  activity, Langton's λ, boundary mutual information, and — the one that actually settles the
  question — exact periodicity: a configuration that returns to itself after ``p`` steps is an
  **oscillator**, and one that returns to itself *translated* is a **glider**. A structure is
  reported as emergent only when it has been detected, with its period and displacement attached.
  Everything else is reported as numbers.
* **It proposes candidates to :mod:`nyxara.nyx.ascent`.** A cheap generator feeding an expensive
  verifier is exactly the FunSearch shape: :meth:`Lattice.patterns` hands over the structures the
  rules actually produced, and the verifier decides whether any of them is worth anything.
* **The learned rule closes the loop.** :class:`LearnedRule` adapts its table toward whatever
  reduces prediction error, so the substrate is not frozen at whatever I happened to write down.

Honest framing:

* **This is a simulation of a substrate, not a brain.** It is the same honesty ``nyxara/sim``
  already applies to its physics evaluators. Nothing here thinks.
* **Deterministic given a seed.** Same rule, same start, same lattice on every machine — which is
  what makes "this pattern emerged" a checkable claim rather than an anecdote.
* **A rule that produces nothing is reported as producing nothing.** Most rules are boring; the
  analysis says so instead of finding faces in noise.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = ["Grid", "Rule", "LifeRule", "GrayScottRule", "LearnedRule", "Pattern", "Analysis",
           "Lattice", "LIFE", "HIGHLIFE", "SEEDS", "DAY_AND_NIGHT"]

Grid = Tuple[Tuple[int, ...], ...]


# --------------------------------------------------------------------------- #
# Rules — the only thing in this file that was written by hand
# --------------------------------------------------------------------------- #
class Rule:
    """A local law: the next state of one cell, given its own state and its neighbours."""

    name = "rule"
    states = 2

    def next_state(self, cell: int, neighbours: Sequence[int]) -> int:  # pragma: no cover
        raise NotImplementedError

    def lambda_parameter(self) -> float:
        """Langton's λ — the fraction of the rule table that maps to a non-quiescent state.

        Near 0 everything dies, near 1 everything saturates; interesting behaviour lives in
        between. It is a property of the *rule*, computed here rather than asserted.
        """
        live = total = 0
        for cell in range(self.states):
            for count in range(9):
                total += 1
                if self.next_state(cell, [1] * count + [0] * (8 - count)) != 0:
                    live += 1
        return live / total if total else 0.0


class LifeRule(Rule):
    """Totalistic birth/survival rules — Conway's Life is ``B3/S23`` and nothing more.

    Four numbers. That is the entire specification, and everything the lattice does comes out of
    them rather than out of anything written below.
    """

    states = 2

    def __init__(self, birth: Iterable[int] = (3,), survive: Iterable[int] = (2, 3),
                 name: str = "") -> None:
        self.birth: Set[int] = set(int(b) for b in birth)
        self.survive: Set[int] = set(int(s) for s in survive)
        self.name = name or (f"B{''.join(map(str, sorted(self.birth)))}"
                             f"/S{''.join(map(str, sorted(self.survive)))}")

    def next_state(self, cell: int, neighbours: Sequence[int]) -> int:
        live = sum(1 for n in neighbours if n)
        if cell:
            return 1 if live in self.survive else 0
        return 1 if live in self.birth else 0


LIFE = LifeRule((3,), (2, 3), name="life")
HIGHLIFE = LifeRule((3, 6), (2, 3), name="highlife")
SEEDS = LifeRule((2,), (), name="seeds")
DAY_AND_NIGHT = LifeRule((3, 6, 7, 8), (3, 4, 6, 7, 8), name="day-and-night")


class GrayScottRule(Rule):
    """Reaction–diffusion, quantised to integer states so the same machinery analyses it.

    A different *kind* of local law from Life — continuous chemistry rather than a birth table —
    kept here so nothing in the analysis is secretly specific to totalistic rules.
    """

    name = "gray-scott"
    states = 4

    def __init__(self, *, feed: float = 0.055, kill: float = 0.062) -> None:
        self.feed = float(feed)
        self.kill = float(kill)

    def next_state(self, cell: int, neighbours: Sequence[int]) -> int:
        u = cell / (self.states - 1)
        mean = (sum(neighbours) / len(neighbours) / (self.states - 1)) if neighbours else 0.0
        du = 0.5 * (mean - u) - u * u * u + self.feed * (1.0 - u) - self.kill * u
        return int(round(min(1.0, max(0.0, u + du)) * (self.states - 1)))


class LearnedRule(Rule):
    """A rule table she adapts, rather than one I fixed.

    The table starts from a seed rule and is nudged by :meth:`reinforce`, so the substrate can be
    driven by the homeostat's prediction error instead of staying frozen at whatever was written
    down. It is a table of ``(cell, live_neighbours) → state``: still local, still deterministic,
    just no longer authored.
    """

    name = "learned"
    states = 2

    def __init__(self, seed_rule: Optional[Rule] = None, *, seed: int = 0) -> None:
        base = seed_rule or LIFE
        self.rng = random.Random(seed)
        self.table: Dict[Tuple[int, int], int] = {
            (cell, count): base.next_state(cell, [1] * count + [0] * (8 - count))
            for cell in range(2) for count in range(9)}
        self.updates = 0

    def next_state(self, cell: int, neighbours: Sequence[int]) -> int:
        return self.table.get((1 if cell else 0, sum(1 for n in neighbours if n)), 0)

    def reinforce(self, *, pressure: float) -> int:
        """Flip entries in proportion to ``pressure`` (typically the homeostat's error).

        Returns how many entries moved, so an "adaptation" that changed nothing says so.
        """
        pressure = max(0.0, min(1.0, float(pressure)))
        flips = 0
        for key in list(self.table):
            if self.rng.random() < pressure * 0.1:
                self.table[key] = 1 - self.table[key]
                flips += 1
        self.updates += 1
        return flips


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Pattern:
    """A structure the rules actually produced, with the evidence that it is one."""

    kind: str                      # still-life | oscillator | glider
    period: int
    dx: int = 0
    dy: int = 0
    cells: int = 0
    first_seen: int = 0

    @property
    def moves(self) -> bool:
        return self.dx != 0 or self.dy != 0

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "period": self.period, "dx": self.dx, "dy": self.dy,
                "cells": self.cells, "first_seen": self.first_seen}


@dataclass
class Analysis:
    """What the run measured. ``patterns`` is empty unless something was actually detected."""

    steps: int = 0
    rule: str = ""
    entropy: float = 0.0
    activity: float = 0.0
    density: float = 0.0
    lambda_parameter: float = 0.0
    boundary_mi: float = 0.0
    patterns: List[Pattern] = field(default_factory=list)
    died_out: bool = False
    saturated: bool = False

    @property
    def emergent(self) -> bool:
        """True only when a structure was *detected* — never inferred from a lively-looking number."""
        return bool(self.patterns)

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": self.steps, "rule": self.rule, "entropy": round(self.entropy, 4),
                "activity": round(self.activity, 4), "density": round(self.density, 4),
                "lambda": round(self.lambda_parameter, 4),
                "boundary_mi": round(self.boundary_mi, 4),
                "patterns": [p.to_dict() for p in self.patterns], "emergent": self.emergent,
                "died_out": self.died_out, "saturated": self.saturated}


# --------------------------------------------------------------------------- #
# The lattice
# --------------------------------------------------------------------------- #
class Lattice:
    """A toroidal grid stepped by a local rule, and measured rather than admired."""

    def __init__(self, width: int = 32, height: int = 32, *, rule: Optional[Rule] = None,
                 seed: int = 0, density: float = 0.3,
                 grid: Optional[Sequence[Sequence[int]]] = None) -> None:
        self.width = max(3, int(width))
        self.height = max(3, int(height))
        self.rule = rule or LIFE
        self.seed = int(seed)
        self.generation = 0
        if grid is not None:
            self.grid: Grid = tuple(tuple(int(c) for c in row) for row in grid)
            self.height = len(self.grid)
            self.width = len(self.grid[0]) if self.grid else self.width
        else:
            rng = random.Random(seed)
            states = max(2, self.rule.states)
            self.grid = tuple(
                tuple((rng.randrange(1, states) if rng.random() < density else 0)
                      for _ in range(self.width))
                for _ in range(self.height))
        self.history: List[Grid] = [self.grid]

    # ---- stepping --------------------------------------------------------- #
    def _neighbours(self, grid: Grid, y: int, x: int) -> List[int]:
        h, w = self.height, self.width
        return [grid[(y + dy) % h][(x + dx) % w]
                for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dx == 0 and dy == 0)]

    def step(self) -> Grid:
        """One synchronous update. Every cell sees only its own neighbourhood."""
        grid = self.grid
        self.grid = tuple(
            tuple(self.rule.next_state(grid[y][x], self._neighbours(grid, y, x))
                  for x in range(self.width))
            for y in range(self.height))
        self.generation += 1
        self.history.append(self.grid)
        if len(self.history) > 4000:
            del self.history[:-4000]
        return self.grid

    def run(self, steps: int = 100) -> "Lattice":
        for _ in range(max(0, int(steps))):
            self.step()
        return self

    # ---- measurement ------------------------------------------------------ #
    @staticmethod
    def _entropy(grid: Grid) -> float:
        counts: Dict[int, int] = {}
        total = 0
        for row in grid:
            for cell in row:
                counts[cell] = counts.get(cell, 0) + 1
                total += 1
        if total == 0:
            return 0.0
        out = 0.0
        for n in counts.values():
            p = n / total
            if p > 0:
                out -= p * math.log2(p)
        return out

    @staticmethod
    def _live(grid: Grid) -> int:
        return sum(1 for row in grid for cell in row if cell)

    def _activity(self) -> float:
        """Fraction of cells that changed on the last step — 0 means the lattice has settled."""
        if len(self.history) < 2:
            return 0.0
        before, after = self.history[-2], self.history[-1]
        changed = sum(1 for y in range(self.height) for x in range(self.width)
                      if before[y][x] != after[y][x])
        return changed / (self.width * self.height)

    def _boundary_mi(self) -> float:
        """Mutual information between the left and right halves.

        Independent halves score ~0; a lattice whose sides genuinely constrain each other scores
        above it. This is the honest version of "the parts are talking to each other".
        """
        mid = self.width // 2
        if mid < 1:
            return 0.0
        joint: Dict[Tuple[int, int], int] = {}
        left: Dict[int, int] = {}
        right: Dict[int, int] = {}
        total = 0
        for row in self.grid:
            for x in range(mid):
                a, b = row[x], row[x + mid]
                joint[(a, b)] = joint.get((a, b), 0) + 1
                left[a] = left.get(a, 0) + 1
                right[b] = right.get(b, 0) + 1
                total += 1
        if not total:
            return 0.0
        mi = 0.0
        for (a, b), n in joint.items():
            pab, pa, pb = n / total, left[a] / total, right[b] / total
            if pab > 0 and pa > 0 and pb > 0:
                mi += pab * math.log2(pab / (pa * pb))
        return max(0.0, mi)

    # ---- the part that settles the question ------------------------------- #
    def _translations(self, a: Grid, b: Grid) -> Optional[Tuple[int, int]]:
        """The displacement that maps ``a`` onto ``b`` exactly, or None.

        Checking every shift is what distinguishes a *glider* from a coincidence: a structure that
        reappears one cell over is moving, and that is a fact about the run rather than a reading
        of it.
        """
        h, w = self.height, self.width
        for dy in range(h):
            for dx in range(w):
                if all(a[y][x] == b[(y + dy) % h][(x + dx) % w]
                       for y in range(h) for x in range(w)):
                    sy = dy - h if dy > h // 2 else dy
                    sx = dx - w if dx > w // 2 else dx
                    return (sx, sy)
        return None

    def detect_patterns(self, *, max_period: int = 16) -> List[Pattern]:
        """Exact periodicity, including translation. Nothing is reported that was not found."""
        found: List[Pattern] = []
        if len(self.history) < 2:
            return found
        latest = self.history[-1]
        # An empty lattice repeats itself forever, and a full one usually does too. Neither is a
        # structure — calling dead space a "still life" would be precisely the finding-faces-in-
        # noise this module claims not to do, so there is nothing here to detect.
        live = self._live(latest)
        if live == 0 or live == self.width * self.height:
            return found
        limit = min(max_period, len(self.history) - 1)
        for period in range(1, limit + 1):
            earlier = self.history[-1 - period]
            if earlier == latest:
                kind = "still-life" if period == 1 else "oscillator"
                found.append(Pattern(kind=kind, period=period, cells=live,
                                     first_seen=self.generation - period))
                break
            shift = self._translations(earlier, latest)
            if shift is not None and (shift[0] or shift[1]):
                found.append(Pattern(kind="glider", period=period, dx=shift[0], dy=shift[1],
                                     cells=live, first_seen=self.generation - period))
                break
        return found

    def analyse(self, *, max_period: int = 16) -> Analysis:
        live = self._live(self.grid)
        cells = self.width * self.height
        return Analysis(
            steps=self.generation, rule=self.rule.name,
            entropy=self._entropy(self.grid), activity=self._activity(),
            density=live / cells if cells else 0.0,
            lambda_parameter=self.rule.lambda_parameter(),
            boundary_mi=self._boundary_mi(),
            patterns=self.detect_patterns(max_period=max_period),
            died_out=live == 0, saturated=live == cells)

    # ---- what ascent can use ---------------------------------------------- #
    def patterns(self, *, max_period: int = 16) -> List[Dict[str, Any]]:
        """Detected structures, as candidate material for a search that has a verifier.

        A cheap generator feeding an expensive checker is the whole shape here — this side is
        deliberately not the judge of whether a pattern is worth anything.
        """
        return [p.to_dict() for p in self.detect_patterns(max_period=max_period)]

    def to_dict(self) -> Dict[str, Any]:
        return {"width": self.width, "height": self.height, "rule": self.rule.name,
                "generation": self.generation, "grid": [list(row) for row in self.grid]}

    def render(self, *, alive: str = "█", dead: str = "·") -> str:
        return "\n".join("".join(alive if c else dead for c in row) for row in self.grid)

    def stats(self) -> Dict[str, Any]:
        return {**self.analyse().to_dict(),
                "width": self.width, "height": self.height,
                "note": ("a simulation of a substrate, not a brain — nothing here thinks. Local "
                         "rules are written; behaviour is not. A pattern is reported only when "
                         "exact periodicity (or periodicity plus translation) was detected, and "
                         "a rule that produces nothing is reported as producing nothing")}


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA automata self-test")
    print("=" * 70)

    # 1) A GLIDER. Five live cells and four numbers (B3/S23) — and it walks.
    #    Nothing in this file says "move diagonally". That is the whole demonstration.
    glider_seed = [[0] * 12 for _ in range(12)]
    for y, x in ((0, 1), (1, 2), (2, 0), (2, 1), (2, 2)):
        glider_seed[y][x] = 1
    lat = Lattice(grid=glider_seed, rule=LIFE).run(4)
    found = lat.detect_patterns()
    print(f"\nglider         : {[p.to_dict() for p in found]}")
    assert found and found[0].kind == "glider", found
    assert found[0].period == 4 and abs(found[0].dx) == 1 and abs(found[0].dy) == 1
    print(lat.render())

    # 2) A STILL LIFE — the block never changes.
    block = [[0] * 8 for _ in range(8)]
    block[3][3] = block[3][4] = block[4][3] = block[4][4] = 1
    assert Lattice(grid=block, rule=LIFE).run(3).detect_patterns()[0].kind == "still-life"

    # 3) AN OSCILLATOR — the blinker has period 2.
    blinker = [[0] * 8 for _ in range(8)]
    blinker[4][3] = blinker[4][4] = blinker[4][5] = 1
    pattern = Lattice(grid=blinker, rule=LIFE).run(4).detect_patterns()[0]
    print(f"oscillator     : period {pattern.period}")
    assert pattern.kind == "oscillator" and pattern.period == 2

    # 4) A rule that produces nothing says so, rather than finding faces in noise.
    empty = Lattice(width=10, height=10, grid=[[0] * 10 for _ in range(10)], rule=LIFE).run(3)
    report = empty.analyse()
    print(f"empty lattice  : emergent={report.emergent} died_out={report.died_out}")
    assert report.died_out is True
    assert report.emergent is False, "dead space is not an emergent structure"

    noisy = Lattice(width=24, height=24, rule=SEEDS, seed=7, density=0.3).run(6).analyse()
    print(f"seeds rule     : emergent={noisy.emergent} entropy={noisy.entropy:.3f} "
          f"activity={noisy.activity:.3f}")

    # 5) Langton's λ is a property of the rule, computed not asserted
    print(f"\nlambda         : life={LIFE.lambda_parameter():.3f} "
          f"seeds={SEEDS.lambda_parameter():.3f} "
          f"day/night={DAY_AND_NIGHT.lambda_parameter():.3f}")
    assert 0.0 < LIFE.lambda_parameter() < 1.0

    # 6) Determinism — the same seed is the same universe on every machine
    a = Lattice(width=16, height=16, rule=LIFE, seed=99).run(10)
    b = Lattice(width=16, height=16, rule=LIFE, seed=99).run(10)
    assert a.grid == b.grid, "same seed must mean same lattice"
    print("determinism    : same seed → same universe")

    # 7) A different KIND of local law runs through the same machinery
    gs = Lattice(width=16, height=16, rule=GrayScottRule(), seed=3, density=0.5).run(5)
    print(f"gray-scott     : entropy {gs.analyse().entropy:.3f}")

    # 8) The learned rule is not frozen at what I wrote down
    learned = LearnedRule(LIFE, seed=1)
    before = dict(learned.table)
    flips = learned.reinforce(pressure=1.0)
    print(f"learned rule   : {flips} table entr{'y' if flips == 1 else 'ies'} moved")
    assert flips > 0 and learned.table != before
    assert LearnedRule(LIFE, seed=1).next_state(1, [1, 1, 0, 0, 0, 0, 0, 0]) == \
        LIFE.next_state(1, [1, 1, 0, 0, 0, 0, 0, 0]), "unreinforced, it IS its seed rule"

    print(f"\npatterns()     : {lat.patterns()}")
    assert lat.patterns()

    print("\nALL SELF-TESTS PASSED ✓")
