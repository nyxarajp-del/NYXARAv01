"""NYXARA · kernel/stream.py — continuous default-mode cognition (✦★).

The brain's *default mode network*: when NYXARA is not busy with a task, the mind
does not go dark — it **wanders**. It drifts across associations, **replays** memory,
**incubates** unsolved problems in the background, **prospects** about the future,
and occasionally surfaces an **insight** ("aha!"). These spontaneous thoughts are
injected as candidate :class:`~nyxara.kernel.workspace.Content` into the Global
Workspace, where they compete for consciousness like everything else.

Design ethos: *"always there, not always burning."* The stream is a low-energy
idle process. Its intensity scales **inversely with engagement** — quiet when the
owner is being served, lively when idle — and its **temperature** (divergence)
rises with boredom, so long idleness yields more creative drift.

Pluggability (kernel-sovereign): the actual *elaboration* of a thought is delegated
to an injectable callback — a template composer by default, later swapped for the
LLM / creative faculties. The stream proposes; it never acts.

Depends on :mod:`config`; optionally feeds :mod:`workspace` and reads idle/engagement
from a callback. No hard dependency on memory/llm (those plug in via interfaces).
"""

from __future__ import annotations

import math
import random
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

__all__ = [
    "ThoughtType",
    "Thought",
    "Seed",
    "SeedPool",
    "ConceptGraph",
    "OpenProblem",
    "Elaborator",
    "DefaultModeStream",
    "StreamMetrics",
]


# --------------------------------------------------------------------------- #
# Thought taxonomy
# --------------------------------------------------------------------------- #
class ThoughtType(str, Enum):
    WANDER = "wander"          # associative free drift
    ASSOCIATE = "associate"   # follow a specific association chain
    INCUBATE = "incubate"     # background work on an open problem
    INSIGHT = "insight"       # an incubation pays off ("aha!")
    PROSPECT = "prospect"     # imagine a future scenario
    RETROSPECT = "retrospect" # replay / reinterpret a memory
    REFLECT = "reflect"       # self-referential thought about own state
    CONSOLIDATE = "consolidate"  # compress/abstract recent experience


# salience hints per thought type: (novelty, urgency, emotion, source_priority)
_SALIENCE_HINTS: Dict[ThoughtType, Tuple[float, float, float, float]] = {
    ThoughtType.WANDER: (0.6, 0.0, 0.1, 0.20),
    ThoughtType.ASSOCIATE: (0.5, 0.0, 0.1, 0.25),
    ThoughtType.INCUBATE: (0.4, 0.1, 0.2, 0.30),
    ThoughtType.INSIGHT: (0.9, 0.5, 0.7, 0.85),   # insights are loud
    ThoughtType.PROSPECT: (0.6, 0.2, 0.3, 0.40),
    ThoughtType.RETROSPECT: (0.4, 0.0, 0.3, 0.25),
    ThoughtType.REFLECT: (0.5, 0.1, 0.4, 0.35),
    ThoughtType.CONSOLIDATE: (0.3, 0.0, 0.1, 0.30),
}


@dataclass
class Thought:
    """A spontaneous mental event."""

    type: ThoughtType
    text: str
    seeds: Tuple[str, ...] = ()
    tags: frozenset = field(default_factory=frozenset)
    temperature: float = 0.5
    thought_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)

    def to_content(self) -> "Any":
        """Convert into a workspace Content candidate (lazy import to keep ws optional)."""
        from nyxara.kernel.workspace import Content

        nov, urg, emo, prio = _SALIENCE_HINTS[self.type]
        return Content(
            payload=self,
            source="stream",
            activation=0.5 + 0.4 * self.temperature if self.type is not ThoughtType.INSIGHT else 1.0,
            novelty=nov,
            urgency=urg,
            emotional_intensity=emo,
            source_priority=prio,
            tags=frozenset(self.tags | {self.type.value, "spontaneous"}),
        )


# --------------------------------------------------------------------------- #
# Seeds — material the mind wanders over
# --------------------------------------------------------------------------- #
@dataclass
class Seed:
    text: str
    category: str = "concept"   # memory | problem | goal | percept | concept | self
    weight: float = 1.0
    tags: frozenset = field(default_factory=frozenset)
    added_at: float = field(default_factory=time.time)


class SeedPool:
    """Thread-unsafe-by-design (single cognitive thread) pool of thought seeds.

    Other subsystems (memory, planning, senses) push seeds here; the stream draws
    weighted random samples to wander over. A bounded ring per category keeps it
    fresh (recency-biased forgetting).
    """

    def __init__(self, per_category: int = 128) -> None:
        self._by_cat: Dict[str, Deque[Seed]] = defaultdict(lambda: deque(maxlen=per_category))

    def add(self, seed: Seed) -> None:
        self._by_cat[seed.category].append(seed)

    def add_text(self, text: str, category: str = "concept", weight: float = 1.0,
                 tags: Sequence[str] = ()) -> None:
        self.add(Seed(text=text, category=category, weight=weight, tags=frozenset(tags)))

    def categories(self) -> List[str]:
        return [c for c, d in self._by_cat.items() if d]

    def sample(self, rng: random.Random, k: int = 1,
               categories: Optional[Sequence[str]] = None) -> List[Seed]:
        pool: List[Seed] = []
        cats = categories or list(self._by_cat.keys())
        for c in cats:
            pool.extend(self._by_cat.get(c, ()))
        if not pool:
            return []
        weights = [max(1e-6, s.weight) for s in pool]
        k = min(k, len(pool))
        # weighted sampling without replacement
        chosen: List[Seed] = []
        pool_idx = list(range(len(pool)))
        for _ in range(k):
            total = sum(weights[i] for i in pool_idx)
            r = rng.uniform(0, total)
            acc = 0.0
            for j, i in enumerate(pool_idx):
                acc += weights[i]
                if acc >= r:
                    chosen.append(pool[i])
                    pool_idx.pop(j)
                    break
        return chosen

    def __len__(self) -> int:
        return sum(len(d) for d in self._by_cat.values())


# --------------------------------------------------------------------------- #
# Concept graph — associative substrate for drift
# --------------------------------------------------------------------------- #
class ConceptGraph:
    """A tiny weighted association graph supporting temperature-controlled walks."""

    def __init__(self) -> None:
        self._edges: Dict[str, Dict[str, float]] = defaultdict(dict)

    def associate(self, a: str, b: str, weight: float = 1.0, *, symmetric: bool = True) -> None:
        self._edges[a][b] = self._edges[a].get(b, 0.0) + weight
        if symmetric:
            self._edges[b][a] = self._edges[b].get(a, 0.0) + weight

    def neighbors(self, node: str) -> Dict[str, float]:
        return dict(self._edges.get(node, {}))

    def nodes(self) -> List[str]:
        return list(self._edges.keys())

    def walk(self, start: str, rng: random.Random, steps: int = 3,
             temperature: float = 0.5) -> List[str]:
        """Softmax random walk. High temperature => more divergent, surprising paths."""
        path = [start]
        cur = start
        for _ in range(steps):
            nbrs = self._edges.get(cur)
            if not nbrs:
                break
            items = list(nbrs.items())
            # temperature-scaled softmax over edge weights
            t = max(1e-3, temperature)
            mx = max(w for _, w in items)
            exps = [math.exp((w - mx) / t) for _, w in items]
            z = sum(exps)
            r = rng.uniform(0, z)
            acc = 0.0
            nxt = items[-1][0]
            for (node, _), e in zip(items, exps):
                acc += e
                if acc >= r:
                    nxt = node
                    break
            path.append(nxt)
            cur = nxt
        return path


# --------------------------------------------------------------------------- #
# Incubation — background problem solving
# --------------------------------------------------------------------------- #
@dataclass
class OpenProblem:
    description: str
    difficulty: float = 0.5            # [0,1]
    progress: float = 0.0             # [0,1]
    attempts: int = 0
    tags: frozenset = field(default_factory=frozenset)
    problem_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    last_visited: float = 0.0
    solved: bool = False

    def incubate(self, rng: random.Random, assoc_bonus: float = 0.0) -> Optional[str]:
        """Advance the problem in the background. Occasionally yields an insight.

        Probability of breakthrough rises with accumulated progress, attempts, and
        relevant associations, and falls with difficulty.
        """
        self.attempts += 1
        self.last_visited = time.time()
        # incremental, noisy progress
        step = rng.uniform(0.02, 0.12) * (1.0 - 0.5 * self.difficulty) + 0.5 * assoc_bonus
        self.progress = min(1.0, self.progress + step)
        breakthrough_p = self.progress * (1.0 - self.difficulty) + assoc_bonus * 0.2
        if self.progress > 0.6 and rng.random() < breakthrough_p:
            self.solved = True
            return (f"Insight on «{self.description}»: a connection surfaced after "
                    f"{self.attempts} incubation passes (progress {self.progress:.0%}).")
        return None


# --------------------------------------------------------------------------- #
# Elaborator — turns seeds into thought text (pluggable; LLM later)
# --------------------------------------------------------------------------- #
Elaborator = Callable[[ThoughtType, Sequence[Seed], Sequence[str]], str]


def _default_elaborator(ttype: ThoughtType, seeds: Sequence[Seed], path: Sequence[str]) -> str:
    """Template/associative composer. Deterministic, dependency-free."""
    seed_txt = " · ".join(s.text for s in seeds) if seeds else "(quiet)"
    drift = " → ".join(path) if path else ""
    templates = {
        ThoughtType.WANDER: f"My mind drifts: {seed_txt}" + (f" … {drift}" if drift else ""),
        ThoughtType.ASSOCIATE: f"That reminds me — {drift or seed_txt}.",
        ThoughtType.INCUBATE: f"Still turning over: {seed_txt}.",
        ThoughtType.PROSPECT: f"If this continued, then: {seed_txt} … and onward.",
        ThoughtType.RETROSPECT: f"Looking back at {seed_txt}, I read it differently now.",
        ThoughtType.REFLECT: f"I notice myself dwelling on {seed_txt}.",
        ThoughtType.CONSOLIDATE: f"Pattern across recent things: {seed_txt}.",
        ThoughtType.INSIGHT: seed_txt,
    }
    return templates.get(ttype, seed_txt)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass
class StreamMetrics:
    ticks: int = 0
    thoughts: int = 0
    insights: int = 0
    silent_ticks: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)

    def record(self, t: Thought) -> None:
        self.thoughts += 1
        if t.type is ThoughtType.INSIGHT:
            self.insights += 1
        self.by_type[t.type.value] = self.by_type.get(t.type.value, 0) + 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "ticks": self.ticks, "thoughts": self.thoughts, "insights": self.insights,
            "silent_ticks": self.silent_ticks, "by_type": dict(self.by_type),
        }


# --------------------------------------------------------------------------- #
# The stream
# --------------------------------------------------------------------------- #
class DefaultModeStream:
    """Continuous, engagement-gated spontaneous thought generator.

    Parameters
    ----------
    workspace:    optional :class:`GlobalWorkspace`; thoughts are submitted as Content.
    seeds:        the :class:`SeedPool` to wander over.
    concepts:     the :class:`ConceptGraph` for associative drift.
    elaborator:   thought-text composer (default = templates; swap for LLM later).
    busy_threshold: engagement above this silences the stream (rare intrusions only).
    base_rate:    expected thoughts per tick at full idleness.
    rng:          inject a seeded Random for reproducibility.
    """

    def __init__(
        self,
        *,
        workspace: Any = None,
        seeds: Optional[SeedPool] = None,
        concepts: Optional[ConceptGraph] = None,
        elaborator: Elaborator = _default_elaborator,
        busy_threshold: float = 0.7,
        base_rate: float = 1.5,
        incubation_chance: float = 0.35,
        monologue_size: int = 200,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.workspace = workspace
        self.seeds = seeds or SeedPool()
        self.concepts = concepts or ConceptGraph()
        self.elaborator = elaborator
        self.busy_threshold = busy_threshold
        self.base_rate = base_rate
        self.incubation_chance = incubation_chance
        self.rng = rng or random.Random()
        self.problems: Dict[str, OpenProblem] = {}
        self.monologue: Deque[Thought] = deque(maxlen=monologue_size)
        self.metrics = StreamMetrics()
        self._idle_since: float = time.time()
        self._running = False

    # ---- problem management ---- #
    def add_problem(self, description: str, *, difficulty: float = 0.5,
                    tags: Sequence[str] = ()) -> OpenProblem:
        p = OpenProblem(description=description, difficulty=difficulty, tags=frozenset(tags))
        self.problems[p.problem_id] = p
        # seed it so the wandering mind can stumble onto it
        self.seeds.add_text(description, category="problem", weight=1.5, tags=tags)
        return p

    # ---- idle dynamics ---- #
    def _idle_seconds(self, now: float) -> float:
        return max(0.0, now - self._idle_since)

    def _temperature(self, idle_s: float) -> float:
        """Divergence rises with boredom, saturating around 1.0."""
        return min(1.0, 0.3 + 0.7 * (1.0 - math.exp(-idle_s / 30.0)))

    def _thought_count(self, engagement: float, idle_s: float) -> int:
        if engagement >= self.busy_threshold:
            return 0
        idleness = 1.0 - engagement
        rate = self.base_rate * idleness * (1.0 + 0.5 * (1.0 - math.exp(-idle_s / 20.0)))
        # Poisson-ish: integer part + stochastic remainder
        base = int(rate)
        if self.rng.random() < (rate - base):
            base += 1
        return base

    def _pick_type(self, idle_s: float) -> ThoughtType:
        weights = {
            ThoughtType.WANDER: 3.0,
            ThoughtType.ASSOCIATE: 2.0,
            ThoughtType.RETROSPECT: 1.5 if self.seeds._by_cat.get("memory") else 0.2,
            ThoughtType.PROSPECT: 1.2,
            ThoughtType.REFLECT: 1.0,
            ThoughtType.CONSOLIDATE: 0.8,
        }
        items = list(weights.items())
        total = sum(w for _, w in items)
        r = self.rng.uniform(0, total)
        acc = 0.0
        for t, w in items:
            acc += w
            if acc >= r:
                return t
        return ThoughtType.WANDER

    # ---- generation ---- #
    def _compose(self, ttype: ThoughtType, idle_s: float) -> Optional[Thought]:
        temp = self._temperature(idle_s)
        seeds = self.seeds.sample(self.rng, k=self.rng.randint(1, 2))
        path: List[str] = []
        if seeds and self.concepts.nodes():
            start = next((s.text for s in seeds if s.text in self.concepts.nodes()), None)
            start = start or self.rng.choice(self.concepts.nodes())
            path = self.concepts.walk(start, self.rng, steps=self.rng.randint(1, 3), temperature=temp)
        if not seeds and not path:
            return None
        text = self.elaborator(ttype, seeds, path)
        tags = frozenset({t for s in seeds for t in s.tags} | set(path[1:]))
        return Thought(type=ttype, text=text, seeds=tuple(s.text for s in seeds),
                       tags=tags, temperature=temp)

    def _incubate(self, idle_s: float) -> Optional[Thought]:
        unsolved = [p for p in self.problems.values() if not p.solved]
        if not unsolved:
            return None
        # prefer least-recently-visited, harder problems
        p = min(unsolved, key=lambda x: (x.last_visited, -x.difficulty))
        # association bonus: do any concept neighbors of the problem's tags exist?
        assoc_bonus = 0.0
        for tag in p.tags:
            if self.concepts.neighbors(tag):
                assoc_bonus += 0.1
        insight = p.incubate(self.rng, assoc_bonus=min(0.5, assoc_bonus))
        if insight:
            return Thought(type=ThoughtType.INSIGHT, text=insight,
                           seeds=(p.description,), tags=frozenset(p.tags | {"insight"}),
                           temperature=self._temperature(idle_s))
        return Thought(type=ThoughtType.INCUBATE, text=self.elaborator(
            ThoughtType.INCUBATE, [Seed(p.description, "problem")], []),
            seeds=(p.description,), tags=frozenset(p.tags), temperature=self._temperature(idle_s))

    # ---- the tick ---- #
    def tick(self, engagement: float = 0.0, now: Optional[float] = None) -> List[Thought]:
        """One default-mode step. ``engagement`` in [0,1]; higher == busier == quieter."""
        now = now or time.time()
        self.metrics.ticks += 1
        if engagement >= self.busy_threshold:
            self._idle_since = now  # reset boredom clock while engaged
        idle_s = self._idle_seconds(now)

        produced: List[Thought] = []
        n = self._thought_count(engagement, idle_s)

        # incubation can fire even at moderate engagement (it's quiet background work)
        if engagement < self.busy_threshold and self.problems and \
                self.rng.random() < self.incubation_chance:
            t = self._incubate(idle_s)
            if t is not None:
                produced.append(t)

        for _ in range(n):
            t = self._compose(self._pick_type(idle_s), idle_s)
            if t is not None:
                produced.append(t)

        if not produced:
            self.metrics.silent_ticks += 1

        for t in produced:
            self.monologue.append(t)
            self.metrics.record(t)
            if self.workspace is not None:
                self.workspace.submit(t.to_content())
        return produced

    # ---- async loop ---- #
    async def run(
        self,
        engagement_fn: Callable[[], float],
        *,
        interval: float = 1.0,
        max_ticks: Optional[int] = None,
    ) -> None:
        """Drive the stream continuously, reading live engagement each interval."""
        import asyncio

        self._running = True
        count = 0
        while self._running:
            self.tick(engagement_fn())
            count += 1
            if max_ticks is not None and count >= max_ticks:
                break
            await asyncio.sleep(interval)

    def stop(self) -> None:
        self._running = False

    # ---- introspection ---- #
    def inner_monologue(self, n: int = 10) -> List[str]:
        return [f"[{t.type.value}] {t.text}" for t in list(self.monologue)[-n:]]


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA default-mode stream self-test")
    print("=" * 70)

    rng = random.Random(7)
    seeds = SeedPool()
    for txt, cat in [("the owner's safety", "goal"), ("a strange packet at 3am", "memory"),
                     ("chess endgames", "concept"), ("recursive self-improvement", "concept")]:
        seeds.add_text(txt, category=cat, tags=[txt.split()[0]])

    cg = ConceptGraph()
    cg.associate("chess", "strategy", 2.0)
    cg.associate("strategy", "war", 1.0)
    cg.associate("war", "defense", 1.5)
    cg.associate("defense", "owner", 2.0)

    from nyxara.kernel.workspace import GlobalWorkspace
    ws = GlobalWorkspace(access_threshold=0.5)

    stream = DefaultModeStream(workspace=ws, seeds=seeds, concepts=cg, rng=rng, base_rate=2.0)
    prob = stream.add_problem("how to harden the network without slowing the owner",
                              difficulty=0.4, tags=["defense", "owner"])

    # idle: the mind should be lively
    total = 0
    for _ in range(12):
        produced = stream.tick(engagement=0.0)
        total += len(produced)
    print(f"idle produced       : {total} thoughts over 12 ticks")
    assert total > 0

    # engaged: the mind should mostly go quiet
    busy = sum(len(stream.tick(engagement=0.95)) for _ in range(10))
    print(f"engaged produced    : {busy} thoughts over 10 busy ticks (should be ~0)")
    assert busy == 0

    # keep incubating until an insight emerges
    got_insight = any(p.solved for p in stream.problems.values())
    for _ in range(60):
        if got_insight:
            break
        for t in stream.tick(engagement=0.1):
            if t.type is ThoughtType.INSIGHT:
                got_insight = True
    print(f"incubation insight  : {got_insight} (problem solved={prob.solved})")
    assert got_insight

    print("\n--- inner monologue (tail) ---")
    for line in stream.inner_monologue(6):
        print("  " + line)

    print(f"\nmetrics            : {stream.metrics.snapshot()}")
    print(f"workspace cycles   : feeding {len(ws.pending())} pending candidates")
    print("\nALL SELF-TESTS PASSED ✓")
