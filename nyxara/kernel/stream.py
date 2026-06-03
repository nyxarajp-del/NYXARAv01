"""Continuous default-mode cognition — idle thinking, mind-wandering, incubation.

When Nyxara is not actively engaged, the mind does not go dark. The stream produces a
low-intensity flow of spontaneous thoughts: it samples *seeds* (recent memories, open
intentions, unresolved problems), recombines them, and offers them to the global
workspace at low salience. Occasionally an incubating problem gets a fresh angle —
this is where insight ("aha") can surface without deliberate effort.

The stream takes seed providers as callables so it does not hard-depend on the memory or
planning layers; the orchestrator wires real providers in once those layers exist.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

from nyxara.kernel.workspace import Candidate, GlobalWorkspace

SeedProvider = Callable[[], list[str]]  # returns text fragments to wander over


@dataclass
class Incubation:
    """An unresolved problem being chewed on in the background."""

    problem: str
    opened_at: float = field(default_factory=time.time)
    attempts: int = 0
    best_angle: str = ""

    def age(self) -> float:
        return time.time() - self.opened_at


class DefaultModeStream:
    """Spontaneous, low-salience background cognition."""

    def __init__(self, workspace: GlobalWorkspace, *,
                 wander_salience: float = 0.15, rng: random.Random | None = None) -> None:
        self._ws = workspace
        self._wander_salience = wander_salience
        self._rng = rng or random.Random()
        self._seed_providers: list[SeedProvider] = []
        self._incubating: list[Incubation] = []
        self._thoughts_emitted = 0

    # ── wiring ──────────────────────────────────────────────────────────────────
    def add_seed_provider(self, provider: SeedProvider) -> None:
        self._seed_providers.append(provider)

    def incubate(self, problem: str) -> Incubation:
        inc = Incubation(problem=problem)
        self._incubating.append(inc)
        return inc

    def resolve(self, problem: str) -> None:
        self._incubating = [i for i in self._incubating if i.problem != problem]

    # ── generation ──────────────────────────────────────────────────────────────
    def _gather_seeds(self) -> list[str]:
        seeds: list[str] = []
        for provider in self._seed_providers:
            try:
                seeds.extend(provider())
            except Exception:  # a flaky provider must not stop the stream
                continue
        return seeds

    def _wander(self, seeds: list[str]) -> str | None:
        if not seeds:
            return None
        if len(seeds) >= 2 and self._rng.random() < 0.6:
            a, b = self._rng.sample(seeds, 2)
            return f"What connects '{a}' and '{b}'?"          # associative recombination
        return f"Idle reflection on '{self._rng.choice(seeds)}'."

    def _incubation_angle(self) -> Candidate | None:
        if not self._incubating:
            return None
        inc = self._rng.choice(self._incubating)
        inc.attempts += 1
        angle = f"Fresh angle on: {inc.problem} (attempt {inc.attempts})"
        inc.best_angle = angle
        # Incubation insights get a slightly higher salience than idle wandering.
        return Candidate(source="stream.incubation", content=angle,
                         salience=min(0.5, self._wander_salience + 0.1 * inc.attempts),
                         kind="thought", meta={"problem": inc.problem})

    def tick(self) -> list[Candidate]:
        """One default-mode step: produce 0..2 spontaneous candidates."""
        produced: list[Candidate] = []

        thought = self._wander(self._gather_seeds())
        if thought is not None:
            produced.append(Candidate(source="stream.wander", content=thought,
                                      salience=self._wander_salience, kind="thought"))

        if self._rng.random() < 0.4:
            inc = self._incubation_angle()
            if inc is not None:
                produced.append(inc)

        for c in produced:
            self._ws.offer(c)
            self._thoughts_emitted += 1
        return produced

    @property
    def stats(self) -> dict[str, object]:
        return {
            "thoughts_emitted": self._thoughts_emitted,
            "incubating": [i.problem for i in self._incubating],
            "seed_providers": len(self._seed_providers),
        }


__all__ = ["DefaultModeStream", "Incubation", "SeedProvider"]
