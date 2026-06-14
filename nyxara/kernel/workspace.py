"""NYXARA · kernel/workspace.py — the Global Workspace (★ core-conscious).

An implementation of Global Workspace Theory (Baars / Dehaene): many specialist
processes generate candidate contents that **compete** for a narrow **conscious
bottleneck**; the winner is **broadcast** to the entire system, becoming the
momentary content of awareness around which everything else coordinates.

Mechanics
---------
* **Competition** — every :class:`Content` is scored for *salience* from a weighted
  blend of bottom-up signals (activation, novelty, urgency, emotional intensity,
  source priority) and top-down attention (goal relevance + goal bias).
* **Coalitions** — contents that share tags reinforce one another; a coalition's
  strength is the synergistic sum of its members. Coalitions, not lone items, win.
* **Bottleneck** — only ``capacity`` coalition(s) above the access threshold reach
  consciousness per cycle. Everything else stays unconscious (and decays).
* **Broadcast** — the winner is published to all registered listeners and, if wired,
  onto the :class:`~nyxara.kernel.bus.EventBus` (topic ``workspace.broadcast``).
* **Habituation & inhibition-of-return** — recently-broadcast content is transiently
  suppressed so attention doesn't fixate; novelty fades with repetition.

This module is *sovereign-friendly*: it only **selects and broadcasts**; it never
acts. The kernel/orchestrator decides what to do with conscious content.

Depends on :mod:`config`; optionally integrates with :mod:`bus`.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
)

__all__ = [
    "Content",
    "SalienceWeights",
    "Coalition",
    "Broadcast",
    "WorkspaceMetrics",
    "GlobalWorkspace",
]


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #
@dataclass
class Content:
    """A candidate for consciousness, produced by some specialist process.

    All salience features are in ``[0, 1]`` except ``source_priority`` which is a
    soft weight (typically 0–1). ``activation`` decays over cycles unless refreshed.
    """

    payload: Any
    source: str
    activation: float = 1.0
    novelty: float = 0.5
    urgency: float = 0.0
    goal_relevance: float = 0.0
    emotional_intensity: float = 0.0
    source_priority: float = 0.5
    tags: FrozenSet[str] = field(default_factory=frozenset)
    item_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    # filled in during arbitration
    salience: float = 0.0

    def signature(self) -> str:
        """Identity for habituation/IOR — same source+tags == 'the same thing'."""
        return f"{self.source}:{','.join(sorted(self.tags))}"

    def refresh(self, activation: float = 1.0) -> "Content":
        self.activation = max(self.activation, activation)
        self.created_at = time.time()
        return self


# --------------------------------------------------------------------------- #
# Salience weighting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SalienceWeights:
    activation: float = 1.0
    novelty: float = 0.8
    urgency: float = 1.2
    goal_relevance: float = 1.5
    emotional_intensity: float = 0.9
    source_priority: float = 0.6
    # multiplier applied to top-down goal-bias contribution
    goal_bias: float = 1.0

    def score(self, c: Content, goal_bias: float) -> float:
        return (
            self.activation * c.activation
            + self.novelty * c.novelty
            + self.urgency * c.urgency
            + self.goal_relevance * c.goal_relevance
            + self.emotional_intensity * c.emotional_intensity
            + self.source_priority * c.source_priority
            + self.goal_bias * goal_bias
        )


# --------------------------------------------------------------------------- #
# Coalition & broadcast
# --------------------------------------------------------------------------- #
@dataclass
class Coalition:
    members: List[Content]
    strength: float = 0.0
    tags: FrozenSet[str] = field(default_factory=frozenset)

    @property
    def lead(self) -> Content:
        return max(self.members, key=lambda c: c.salience)


@dataclass
class Broadcast:
    """The content of consciousness for one cycle."""

    winner: Content
    coalition: Coalition
    salience: float
    cycle: int
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "source": self.winner.source,
            "salience": round(self.salience, 4),
            "tags": sorted(self.winner.tags),
            "coalition_size": len(self.coalition.members),
            "at": self.at,
        }


@dataclass
class WorkspaceMetrics:
    cycles: int = 0
    submitted: int = 0
    broadcasts: int = 0
    starved_cycles: int = 0  # nothing crossed the threshold
    dropped_decayed: int = 0

    def snapshot(self) -> Dict[str, int]:
        return dict(self.__dict__)


# --------------------------------------------------------------------------- #
# The workspace
# --------------------------------------------------------------------------- #
BroadcastListener = Callable[[Broadcast], None]


class GlobalWorkspace:
    """Attention arbitration + conscious bottleneck + broadcast.

    Parameters
    ----------
    capacity:           how many coalitions may reach consciousness per cycle (≈1).
    access_threshold:   minimum coalition strength to enter consciousness.
    decay:              per-cycle activation multiplier for losers.
    activation_floor:   losers below this are forgotten (dropped from candidates).
    habituation:        novelty multiplier applied when a signature recurs.
    ior_strength:       salience suppression for a just-broadcast signature.
    ior_window:         cycles over which inhibition-of-return relaxes.
    coalition_synergy:  per-extra-member multiplier when items share tags.
    history:            length of the retained stream-of-consciousness.
    """

    def __init__(
        self,
        *,
        capacity: int = 1,
        access_threshold: float = 1.0,
        decay: float = 0.85,
        activation_floor: float = 0.05,
        habituation: float = 0.6,
        ior_strength: float = 0.5,
        ior_window: int = 3,
        coalition_synergy: float = 0.25,
        weights: Optional[SalienceWeights] = None,
        history: int = 256,
        bus: Any = None,
        broadcast_topic: str = "workspace.broadcast",
    ) -> None:
        self.capacity = max(1, capacity)
        self.access_threshold = access_threshold
        self.decay = decay
        self.activation_floor = activation_floor
        self.habituation = habituation
        self.ior_strength = ior_strength
        self.ior_window = max(1, ior_window)
        self.coalition_synergy = coalition_synergy
        self.weights = weights or SalienceWeights()
        self._bus = bus
        self._broadcast_topic = broadcast_topic

        self._candidates: Dict[str, Content] = {}
        self._goals: Dict[str, float] = {}                 # tag -> weight (top-down)
        self._recent_signatures: Deque[str] = deque(maxlen=64)  # habituation memory
        self._ior: Dict[str, int] = {}                     # signature -> cycles remaining
        self._listeners: List[BroadcastListener] = []
        self.history: Deque[Broadcast] = deque(maxlen=history)
        self.metrics = WorkspaceMetrics()
        self._cycle = 0

    # ---- input ---- #
    def submit(self, content: Content) -> str:
        """Enter a candidate into the competition. Same item_id refreshes in place."""
        self.metrics.submitted += 1
        existing = self._candidates.get(content.item_id)
        if existing is not None:
            existing.activation = max(existing.activation, content.activation)
            return existing.item_id
        self._candidates[content.item_id] = content
        return content.item_id

    def submit_many(self, contents: Sequence[Content]) -> List[str]:
        return [self.submit(c) for c in contents]

    # ---- top-down attention ---- #
    def set_goals(self, goals: Dict[str, float]) -> None:
        """Set top-down attentional bias: tag -> importance weight."""
        self._goals = dict(goals)

    def bias_for(self, content: Content) -> float:
        if not self._goals:
            return 0.0
        return sum(self._goals.get(t, 0.0) for t in content.tags)

    # ---- listeners ---- #
    def add_listener(self, fn: BroadcastListener) -> BroadcastListener:
        self._listeners.append(fn)
        return fn

    def remove_listener(self, fn: BroadcastListener) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    # ---- salience & coalitions ---- #
    def _salience(self, c: Content) -> float:
        base = self.weights.score(c, self.bias_for(c))
        # novelty habituation: things seen recently are less novel
        if c.signature() in self._recent_signatures:
            base -= self.weights.novelty * c.novelty * (1.0 - self.habituation)
        # inhibition of return: suppress recently-broadcast signatures
        ior = self._ior.get(c.signature())
        if ior:
            relax = ior / self.ior_window  # 1.0 -> just broadcast, decays to ~0
            base *= max(0.0, 1.0 - self.ior_strength * relax)
        return max(0.0, base)

    def _form_coalitions(self, scored: List[Content]) -> List[Coalition]:
        """Greedy tag-overlap clustering into mutually-supporting coalitions."""
        coalitions: List[Coalition] = []
        for c in sorted(scored, key=lambda x: x.salience, reverse=True):
            placed = False
            for coal in coalitions:
                if c.tags & coal.tags:
                    coal.members.append(c)
                    coal.tags = coal.tags | c.tags
                    placed = True
                    break
            if not placed:
                coalitions.append(Coalition(members=[c], tags=frozenset(c.tags)))
        # strength = sum of member salience * synergy bonus for size
        for coal in coalitions:
            raw = sum(m.salience for m in coal.members)
            synergy = 1.0 + self.coalition_synergy * (len(coal.members) - 1)
            coal.strength = raw * synergy
        return coalitions

    # ---- the cognitive cycle ---- #
    def cycle(self) -> List[Broadcast]:
        """Run one arbitration cycle. Returns the broadcast(s) (possibly empty)."""
        self._cycle += 1
        self.metrics.cycles += 1

        # relax inhibition-of-return timers
        for sig in list(self._ior):
            self._ior[sig] -= 1
            if self._ior[sig] <= 0:
                del self._ior[sig]

        if not self._candidates:
            self.metrics.starved_cycles += 1
            return []

        # score
        candidates = list(self._candidates.values())
        for c in candidates:
            c.salience = self._salience(c)

        coalitions = self._form_coalitions(candidates)
        coalitions.sort(key=lambda k: k.strength, reverse=True)

        winners = [k for k in coalitions if k.strength >= self.access_threshold][: self.capacity]

        broadcasts: List[Broadcast] = []
        won_ids = set()
        if not winners:
            self.metrics.starved_cycles += 1
        for coal in winners:
            lead = coal.lead
            b = Broadcast(winner=lead, coalition=coal, salience=lead.salience, cycle=self._cycle)
            broadcasts.append(b)
            self.history.append(b)
            self.metrics.broadcasts += 1
            # habituation + IOR bookkeeping for everything that just won
            for m in coal.members:
                self._recent_signatures.append(m.signature())
                self._ior[m.signature()] = self.ior_window
                won_ids.add(m.item_id)
            self._emit(b)

        # winners leave the competition (they were 'used'); losers decay
        self._post_cycle_decay(won_ids)
        return broadcasts

    def _post_cycle_decay(self, won_ids: set) -> None:
        for item_id in list(self._candidates):
            if item_id in won_ids:
                del self._candidates[item_id]
                continue
            c = self._candidates[item_id]
            c.activation *= self.decay
            if c.activation < self.activation_floor:
                del self._candidates[item_id]
                self.metrics.dropped_decayed += 1

    def _emit(self, b: Broadcast) -> None:
        for fn in list(self._listeners):
            try:
                fn(b)
            except Exception:  # noqa: BLE001 - a broken listener must not stop the broadcast
                continue
        if self._bus is not None:
            self._publish_to_bus(b)

    def _publish_to_bus(self, b: Broadcast) -> None:
        """Best-effort publish onto the EventBus (sync or async contexts)."""
        try:
            from nyxara.kernel.bus import Event, Priority  # local import to keep bus optional
        except Exception:  # pragma: no cover
            return
        ev = Event(
            topic=self._broadcast_topic,
            payload=b,
            priority=Priority.HIGH,
            source="workspace",
            headers={"cycle": b.cycle, "salience": b.salience},
        )
        published = self._bus.publish_nowait(ev)
        if not published:
            # if running inside a loop and queue is full/blocking, fall back to schedule
            try:
                import asyncio

                loop = asyncio.get_running_loop()
                loop.create_task(self._bus.publish(ev))
            except RuntimeError:
                pass

    # ---- introspection ---- #
    @property
    def conscious(self) -> Optional[Broadcast]:
        """The most recent content of consciousness."""
        return self.history[-1] if self.history else None

    def pending(self) -> List[Content]:
        return sorted(self._candidates.values(), key=lambda c: c.activation, reverse=True)

    def stream(self, n: int = 10) -> List[Broadcast]:
        return list(self.history)[-n:]

    def clear(self) -> None:
        self._candidates.clear()


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA global-workspace self-test")
    print("=" * 70)

    ws = GlobalWorkspace(access_threshold=1.0)
    log: List[str] = []
    ws.add_listener(lambda b: log.append(f"cycle{b.cycle}:{b.winner.source}"))

    # competing contents — a salient threat should win over idle musing
    ws.submit(Content(payload="idle daydream", source="stream",
                      activation=0.6, novelty=0.4, source_priority=0.3,
                      tags=frozenset({"idle"})))
    ws.submit(Content(payload="INTRUDER on port 22", source="netsec",
                      activation=1.0, urgency=1.0, emotional_intensity=0.8,
                      source_priority=0.9, tags=frozenset({"threat", "security"})))
    ws.submit(Content(payload="owner asked a question", source="dialogue",
                      activation=0.9, goal_relevance=0.7, source_priority=0.8,
                      tags=frozenset({"owner", "task"})))

    ws.set_goals({"owner": 1.0, "task": 0.5})  # top-down: prioritise the owner

    b = ws.cycle()
    assert b, "expected a broadcast"
    winner = b[0].winner.source
    print(f"cycle 1 winner     : {winner} (salience={b[0].salience:.2f})")
    print(f"  pending after     : {[c.source for c in ws.pending()]}")

    # next cycle: winner is gone + under inhibition-of-return; another item surfaces
    b2 = ws.cycle()
    print(f"cycle 2 winner     : {b2[0].winner.source if b2 else 'STARVED'}")

    # habituation: resubmit the same threat repeatedly -> novelty fades
    for _ in range(3):
        ws.submit(Content(payload="same threat", source="netsec",
                          activation=1.0, novelty=1.0, urgency=0.2,
                          tags=frozenset({"threat"})))
        ws.cycle()

    print(f"\nstream of consciousness: {[x.winner.source for x in ws.stream()]}")
    print(f"metrics            : {ws.metrics.snapshot()}")

    # coalition: two weak-but-related items beat one stronger-but-lonely item
    ws2 = GlobalWorkspace(access_threshold=1.5, coalition_synergy=0.5)
    ws2.submit(Content(payload="A", source="a", activation=0.9, tags=frozenset({"plan"})))
    ws2.submit(Content(payload="B", source="b", activation=0.9, tags=frozenset({"plan"})))
    ws2.submit(Content(payload="C", source="c", activation=1.0, tags=frozenset({"lonely"})))
    win = ws2.cycle()
    assert win and "plan" in win[0].winner.tags
    print(f"\ncoalition winner   : tags={sorted(win[0].winner.tags)}, "
          f"size={len(win[0].coalition.members)}, strength={win[0].coalition.strength:.2f}")

    print("\nALL SELF-TESTS PASSED ✓")
