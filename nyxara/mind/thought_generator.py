"""NYXARA · mind/thought_generator.py — Cognitive Workspace thought pipeline (Level 1).

Implements the **prefrontal bottleneck**: many candidate thoughts compete for a narrow
conscious window; only the most salient winners enter the reason step.

Mechanics
---------
* Multiple *specialist sources* each emit :class:`~nyxara.kernel.workspace.Content`
  objects: the stimulus itself, recalled memories, active goals, the current affective
  tone, and percept-derived themes.
* All candidates are submitted to a :class:`~nyxara.kernel.workspace.GlobalWorkspace`
  which runs one arbitration cycle: salience scoring → coalition formation → broadcast.
* The top-N broadcast winners (default 3) are returned as enriched context strings that
  ground the LLM reasoning step.

This mirrors the human prefrontal cortex: 100 micro-thoughts compete; 3 make it through.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, FrozenSet, List, Optional

from nyxara.kernel.workspace import Content, GlobalWorkspace

__all__ = ["ThoughtGenerator", "ThoughtContext"]


class ThoughtContext:
    """Thin container for the inputs to a thought-generation pass."""

    def __init__(self, stimulus: str, memories: Optional[List[Any]] = None,
                 goals: Any = None, affect: Any = None,
                 percept: Any = None) -> None:
        self.stimulus = stimulus
        self.memories = memories or []
        self.goals = goals        # GoalSystem | None
        self.affect = affect      # AffectSystem | None
        self.percept = percept    # Percept | None


class ThoughtGenerator:
    """Generates candidate thoughts from multiple sources and arbitrates via GlobalWorkspace.

    Parameters
    ----------
    workspace:  the shared global workspace to use (default: a fresh one per call is not
                used — pass a long-lived one from NyxaraCore so history/IOR accumulate).
    top_n:      how many broadcast winners to return (the conscious bottleneck width).
    max_from_memories: cap on how many memory-derived Contents to generate.
    """

    def __init__(self, workspace: Optional[GlobalWorkspace] = None,
                 top_n: int = 3, max_from_memories: int = 20) -> None:
        self.workspace = workspace or GlobalWorkspace(capacity=top_n, access_threshold=0.5)
        self.top_n = max(1, top_n)
        self.max_from_memories = max(1, max_from_memories)

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #
    def generate(self, ctx: ThoughtContext) -> List[str]:
        """Run one thought-competition cycle.

        Returns a list of up to ``top_n`` text strings — the conscious broadcast
        winners — suitable for injection into the reasoner's context.  Always
        returns at least the stimulus itself, so the result is never empty.
        """
        self._sync_goals(ctx)
        contents = self._emit_contents(ctx)
        for c in contents:
            self.workspace.submit(c)
        broadcasts = self.workspace.cycle()
        winners = [b.winner.payload for b in broadcasts if isinstance(b.winner.payload, str)]
        if not winners:
            winners = [ctx.stimulus]
        return winners[: self.top_n]

    def workspace_metrics(self) -> Dict[str, int]:
        return self.workspace.metrics.snapshot()

    # ---------------------------------------------------------------------- #
    # Goal synchronisation — top-down attentional bias
    # ---------------------------------------------------------------------- #
    def _sync_goals(self, ctx: ThoughtContext) -> None:
        """Wire active goals into the workspace as top-down attentional weights."""
        if ctx.goals is None:
            return
        try:
            ranked = ctx.goals.prioritize()
            goal_tags: Dict[str, float] = {}
            for g in ranked[:5]:
                # use the first word of the goal name as a tag
                tag = g.name.replace(" ", "_").lower()[:20]
                goal_tags[tag] = float(g.priority)
            if goal_tags:
                self.workspace.set_goals(goal_tags)
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------------- #
    # Content emitters — one per specialist source
    # ---------------------------------------------------------------------- #
    def _emit_contents(self, ctx: ThoughtContext) -> List[Content]:
        contents: List[Content] = []
        contents.extend(self._from_stimulus(ctx))
        contents.extend(self._from_memories(ctx))
        contents.extend(self._from_goals(ctx))
        contents.extend(self._from_affect(ctx))
        contents.extend(self._from_percept(ctx))
        return contents

    def _from_stimulus(self, ctx: ThoughtContext) -> List[Content]:
        """The primary stimulus is always the most urgent thought."""
        stim = ctx.stimulus.strip()
        if not stim:
            return []
        tags: FrozenSet[str] = frozenset({"stimulus", "owner_input"})
        return [Content(
            payload=stim,
            source="stimulus",
            activation=1.0,
            novelty=0.8,
            urgency=0.9,
            goal_relevance=0.7,
            source_priority=1.0,
            tags=tags,
        )]

    def _from_memories(self, ctx: ThoughtContext) -> List[Content]:
        """Each recalled memory is a candidate thought with relevance-proportional salience."""
        out: List[Content] = []
        for i, mem in enumerate(ctx.memories[: self.max_from_memories]):
            text = self._memory_text(mem)
            if not text:
                continue
            # earlier (more relevant) memories get higher activation
            activation = max(0.3, 1.0 - i * 0.05)
            importance = float(getattr(getattr(mem, "record", mem), "importance", 0.5))
            tags: FrozenSet[str] = frozenset({"memory", "recall"})
            out.append(Content(
                payload=text[:200],
                source=f"memory[{i}]",
                activation=activation,
                novelty=max(0.1, 0.6 - i * 0.04),
                urgency=0.0,
                goal_relevance=importance,
                source_priority=0.6,
                tags=tags,
            ))
        return out

    def _from_goals(self, ctx: ThoughtContext) -> List[Content]:
        """Active goals surface as goal-shaped thoughts that bias attention."""
        if ctx.goals is None:
            return []
        out: List[Content] = []
        try:
            ranked = ctx.goals.prioritize()
            for g in ranked[:5]:
                tag = g.name.replace(" ", "_").lower()[:20]
                out.append(Content(
                    payload=f"[goal] {g.name}",
                    source="goals",
                    activation=float(g.priority),
                    novelty=0.3,
                    urgency=float(g.priority) * 0.5,
                    goal_relevance=1.0,
                    source_priority=0.7,
                    tags=frozenset({"goal", tag}),
                ))
        except Exception:  # noqa: BLE001
            pass
        return out

    def _from_affect(self, ctx: ThoughtContext) -> List[Content]:
        """The current emotional tone enters consciousness as a colouring thought."""
        if ctx.affect is None:
            return []
        try:
            mood = ctx.affect.mood
            valence = float(getattr(mood, "valence", 0.0))
            arousal = float(getattr(mood, "arousal", 0.25))
            label = getattr(mood, "label", "neutral")
            text = f"[affect] {label} (valence={valence:.2f}, arousal={arousal:.2f})"
            emotional_intensity = _clamp01(math.sqrt(valence ** 2 + arousal ** 2) / math.sqrt(2))
            return [Content(
                payload=text,
                source="affect",
                activation=emotional_intensity,
                novelty=0.2,
                urgency=arousal * 0.4,
                goal_relevance=0.2,
                emotional_intensity=emotional_intensity,
                source_priority=0.5,
                tags=frozenset({"affect", label}),
            )]
        except Exception:  # noqa: BLE001
            return []

    def _from_percept(self, ctx: ThoughtContext) -> List[Content]:
        """The attended percept's content, modality, and salience become a thought."""
        if ctx.percept is None:
            return []
        try:
            content_text = getattr(ctx.percept, "content", "") or ""
            modality = getattr(getattr(ctx.percept, "modality", None), "value", "text")
            salience = float(getattr(ctx.percept, "salience", 0.5))
            if not content_text:
                return []
            return [Content(
                payload=f"[percept/{modality}] {content_text[:120]}",
                source="percept",
                activation=salience,
                novelty=0.5,
                urgency=salience * 0.3,
                goal_relevance=salience * 0.6,
                source_priority=0.65,
                tags=frozenset({"percept", modality}),
            )]
        except Exception:  # noqa: BLE001
            return []

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _memory_text(mem: Any) -> str:
        rec = getattr(mem, "record", None)
        if rec is not None and hasattr(rec, "text"):
            try:
                return rec.text()
            except Exception:  # noqa: BLE001
                return ""
        if isinstance(mem, tuple) and mem and hasattr(mem[0], "text"):
            try:
                return mem[0].text()
            except Exception:  # noqa: BLE001
                return ""
        if hasattr(mem, "text"):
            try:
                return mem.text()
            except Exception:  # noqa: BLE001
                return ""
        return str(mem) if isinstance(mem, str) else ""


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.kernel.workspace import GlobalWorkspace

    print("=" * 70)
    print("NYXARA thought-generator self-test")
    print("=" * 70)

    ws = GlobalWorkspace(capacity=3, access_threshold=0.5)
    tg = ThoughtGenerator(workspace=ws, top_n=3)

    # simulate a context: stimulus + 5 fake memories + no goals/affect
    class _FakeMem:
        def __init__(self, t: str, imp: float = 0.5):
            self._t = t
            self.importance = imp
        def text(self) -> str:
            return self._t

    ctx = ThoughtContext(
        stimulus="What is the capital of France?",
        memories=[_FakeMem("Paris is the capital of France.", 0.9),
                  _FakeMem("France is in Europe.", 0.6),
                  _FakeMem("The Eiffel Tower is in Paris.", 0.7)],
    )

    winners = tg.generate(ctx)
    print(f"\nwinners (top-3)     : {winners}")
    assert len(winners) > 0
    assert any("France" in w or "Paris" in w or "capital" in w for w in winners)

    metrics = tg.workspace_metrics()
    print(f"workspace metrics   : {metrics}")
    assert metrics["broadcasts"] > 0

    print("\nALL SELF-TESTS PASSED ✓")
