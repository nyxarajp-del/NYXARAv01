"""NYXARA · growth/prompt_grammar.py — holographic prompt-grammar mutation (Part L).

The *way* NYXARA phrases a request to a cloud model (its structure, constraints, format) changes how
often the model answers correctly. So she treats that phrasing — the **prompt grammar** — as an evolvable
genome and keeps the grammars that maximise the **verified-correct** rate (scored by the Part-M grounded
verifier). When a task-type keeps producing reasoning errors, she runs a small genetic search over
grammars and adopts the one whose answers actually verify.

Honest boundary: this *maximises* verified-correct output on a task battery — it does **not** guarantee
"zero hallucination" (no prompt can), and it only optimises what a verifier can score. Bounded generations,
deterministic under a seed, LLM-free at this layer (any ``generate(system, user) -> str`` model plugs in).

Reuses the GP discipline of ``growth/evolve.py`` / ``growth/rule_synth.py`` and the Part-M verifier.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

__all__ = ["PromptGrammar", "PromptGrammarEvolver"]

# The evolvable directives — each is a structural constraint that may be present or absent.
_DIRECTIVES = ("show_steps", "double_check", "answer_only", "cite_rule", "be_terse", "no_hedging")
_FORMATS = ("zero_shot", "few_shot")

_DIRECTIVE_TEXT = {
    "show_steps": "Work through the problem step by step.",
    "double_check": "Double-check every calculation before you commit to an answer.",
    "answer_only": "Output ONLY the final answer — no explanation.",
    "cite_rule": "Name the rule or theorem you applied.",
    "be_terse": "Be maximally concise.",
    "no_hedging": "Do not hedge; never say 'maybe' or 'I think'.",
}


@dataclass(frozen=True)
class PromptGrammar:
    """One candidate phrasing strategy — a set of structural directives + a format."""

    directives: frozenset
    fmt: str = "zero_shot"

    def render(self, question: str) -> Tuple[str, str]:
        """Return (system, user) for this grammar."""
        parts = ["You are a precise reasoning engine."]
        for d in _DIRECTIVES:
            if d in self.directives:
                parts.append(_DIRECTIVE_TEXT[d])
        if self.fmt == "few_shot":
            parts.append("Example: Q: 2+2 -> A: 4.")
        return ("\n".join(parts), question)

    # ---- genetic operators ---- #
    @classmethod
    def random(cls, rng: random.Random) -> "PromptGrammar":
        chosen = frozenset(d for d in _DIRECTIVES if rng.random() < 0.5)
        return cls(directives=chosen, fmt=rng.choice(_FORMATS))

    def mutate(self, rng: random.Random) -> "PromptGrammar":
        directives = set(self.directives)
        d = rng.choice(_DIRECTIVES)
        directives.symmetric_difference_update({d})        # flip one directive
        fmt = self.fmt if rng.random() < 0.7 else rng.choice(_FORMATS)
        return PromptGrammar(directives=frozenset(directives), fmt=fmt)

    def crossover(self, other: "PromptGrammar", rng: random.Random) -> "PromptGrammar":
        directives = {d for d in _DIRECTIVES
                      if (d in self.directives if rng.random() < 0.5 else d in other.directives)}
        fmt = rng.choice((self.fmt, other.fmt))
        return PromptGrammar(directives=frozenset(directives), fmt=fmt)

    def to_dict(self) -> dict:
        return {"directives": sorted(self.directives), "fmt": self.fmt}


class PromptGrammarEvolver:
    """Evolve the prompt grammar toward the highest verified-correct rate (Part L)."""

    def __init__(self, *, verifier: Any = None, population: int = 6, generations: int = 6,
                 seed: int = 0) -> None:
        if verifier is None:
            from nyxara.growth.verified_distill import GroundedVerifier
            verifier = GroundedVerifier()
        self.verifier = verifier
        self.population = int(population)
        self.generations = int(generations)
        self.seed = int(seed)

    def fitness(self, grammar: PromptGrammar, tasks: Sequence[str],
                generate: Callable[[str, str], str]) -> float:
        """Fraction of the task battery whose answer *verifies* under this grammar."""
        if not tasks:
            return 0.0
        total = 0.0
        for q in tasks:
            system, user = grammar.render(q)
            try:
                answer = generate(system, user) or ""
            except Exception:  # noqa: BLE001 — a failed generation scores zero, never crashes the search
                answer = ""
            vr = self.verifier.verify(q, answer)
            total += 1.0 if vr.proven else (vr.confidence if vr.durable else 0.0)
        return total / len(tasks)

    def evolve(self, tasks: Sequence[str], generate: Callable[[str, str], str]
               ) -> Tuple[PromptGrammar, float]:
        """Return the (best_grammar, best_fitness) after a bounded genetic search."""
        rng = random.Random(self.seed)
        pop: List[PromptGrammar] = [PromptGrammar.random(rng) for _ in range(self.population)]
        best, best_fit = pop[0], self.fitness(pop[0], tasks, generate)
        for _ in range(self.generations):
            scored = sorted(((self.fitness(g, tasks, generate), g) for g in pop),
                            key=lambda t: t[0], reverse=True)
            if scored[0][0] > best_fit:
                best_fit, best = scored[0]
            keep = max(1, self.population // 3)
            elite = [g for _, g in scored[:keep]]
            children: List[PromptGrammar] = []
            while len(children) < self.population - keep:
                a, b = rng.choice(elite), rng.choice(elite)
                child = a.crossover(b, rng).mutate(rng)
                children.append(child)
            pop = elite + children
        return best, best_fit
