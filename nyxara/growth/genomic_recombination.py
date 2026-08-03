"""NYXARA · growth/genomic_recombination.py — code as DNA (Part D).

A living species reproduces and evolves. NYXARA treats her own deterministic algorithms as **Digital
Chromosomes**: she takes two high-fitness pure functions, performs **crossover** (splice a subtree of one
into the other) and **mutation**, and breeds a new candidate "child algorithm" — then adopts it **only** if
it is *verified* correct against a specification (a decidable certificate), never merging an unproven child.
This is how a genuinely novel hybrid can emerge — but always behind the mathematical validation of the
Immutable Core, never before it.

Bounded and honest: the chromosome here is a small arithmetic **expression tree** over integer inputs (the
same narrow, pure domain as Part C's native forge). Crossover is subtree-splice, adoption is exact match on
the spec (fail-closed — no match, no adoption). It reuses the GP discipline of ``growth/rule_synth`` and the
"prove before you keep" discipline of ``growth/improvement_proof`` / ``growth/prover``.

Pure standard library; deterministic under a seed.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

__all__ = ["Expr", "RecombinationResult", "GenomicRecombiner"]

# An expression chromosome is a nested tuple:
#   ("var",) | ("const", int) | ("add"|"sub"|"mul", left, right)
Expr = tuple
_BIN = ("add", "sub", "mul")


def evaluate(node: Expr, x: int) -> int:
    op = node[0]
    if op == "var":
        return x
    if op == "const":
        return int(node[1])
    a, b = evaluate(node[1], x), evaluate(node[2], x)
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    return a * b            # mul


def _positions(node: Expr, prefix: Tuple[int, ...] = ()) -> Iterable[Tuple[int, ...]]:
    yield prefix
    if node[0] in _BIN:
        yield from _positions(node[1], prefix + (1,))
        yield from _positions(node[2], prefix + (2,))


def _get(node: Expr, path: Tuple[int, ...]) -> Expr:
    for i in path:
        node = node[i]
    return node


def _set(node: Expr, path: Tuple[int, ...], sub: Expr) -> Expr:
    if not path:
        return sub
    children = list(node)
    children[path[0]] = _set(node[path[0]], path[1:], sub)
    return tuple(children)


def _crossovers(a: Expr, b: Expr) -> Iterable[Expr]:
    """Every child formed by splicing a subtree of ``b`` into a position of ``a``."""
    for pa in _positions(a):
        for pb in _positions(b):
            yield _set(a, pa, _get(b, pb))


def _mutate(node: Expr, rng: random.Random) -> Expr:
    positions = list(_positions(node))
    path = rng.choice(positions)
    sub = _get(node, path)
    roll = rng.random()
    if sub[0] == "const":
        new = ("const", int(sub[1]) + rng.choice((-2, -1, 1, 2)))
    elif roll < 0.4:
        new = ("const", rng.randint(0, 5))
    elif roll < 0.7:
        new = ("var",)
    else:
        new = (rng.choice(_BIN), sub, ("const", rng.randint(1, 3)))
    return _set(node, path, new)


@dataclass
class RecombinationResult:
    verified: bool
    child: Optional[Expr]
    certificate: str = ""
    reason: str = ""


class GenomicRecombiner:
    """Breed two pure algorithms and adopt a child only if it verifies against the spec (Part D)."""

    def __init__(self, *, seed: int = 0, mutation_rounds: int = 200, enabled: Optional[bool] = None) -> None:
        self.seed = int(seed)
        self.mutation_rounds = int(mutation_rounds)
        if enabled is None:
            try:
                from nyxara.kernel.config import get_settings
                enabled = bool(getattr(get_settings().genome, "recombination", True))
            except Exception:  # noqa: BLE001
                enabled = True
        self.enabled = bool(enabled)

    @staticmethod
    def _matches(child: Expr, spec: Sequence[Tuple[int, int]]) -> bool:
        try:
            return all(evaluate(child, x) == y for x, y in spec)
        except Exception:  # noqa: BLE001 — a malformed child simply fails verification
            return False

    def breed(self, parent_a: Expr, parent_b: Expr,
              spec: Sequence[Tuple[int, int]]) -> RecombinationResult:
        """Crossover + mutation of the two parents; return a child that EXACTLY satisfies ``spec``,
        else an unverified result (fail-closed — nothing is adopted without a certificate)."""
        if not self.enabled:
            return RecombinationResult(False, None, reason="recombination disabled")
        if not spec:
            return RecombinationResult(False, None, reason="no specification to verify against")

        seen: set = set()
        # 1) recombination first — every crossover of the two parents (both directions)
        for child in itertools.chain(_crossovers(parent_a, parent_b),
                                     _crossovers(parent_b, parent_a)):
            key = repr(child)
            if key in seen:
                continue
            seen.add(key)
            if self._matches(child, spec):
                return RecombinationResult(True, child, certificate=self._cert(child, spec),
                                           reason="verified crossover child")

        # 2) then bounded mutation of the recombination pool
        rng = random.Random(self.seed)
        bases = [parent_a, parent_b] + [self._parse(k) for k in list(seen)[:32]]
        for _ in range(self.mutation_rounds):
            child = _mutate(rng.choice(bases), rng)
            key = repr(child)
            if key in seen:
                continue
            seen.add(key)
            if self._matches(child, spec):
                return RecombinationResult(True, child, certificate=self._cert(child, spec),
                                           reason="verified mutated child")
        return RecombinationResult(False, None, reason="no child verified against the spec")

    @staticmethod
    def _cert(child: Expr, spec: Sequence[Tuple[int, int]]) -> str:
        return f"matches {len(spec)}/{len(spec)} specification cases exactly"

    @staticmethod
    def _parse(key: str) -> Expr:
        return eval(key, {"__builtins__": {}})   # keys are repr() of our own tuples — safe, no builtins
