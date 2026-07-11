"""NYXARA · growth/eureka.py — truly novel problem solving (💡, Pillar F · Edge 1+2, Rule 4).

Every other faculty in :mod:`nyxara.growth` answers a question someone (a user, a gap-miner, or an
LLM) *handed* to NYXARA. Even the :class:`~nyxara.growth.autonomous_scientist.AutonomousScientist`
poses questions from templates or LLM hints, and the :class:`~nyxara.growth.frontier.FrontierEngine`
only *measures and directs* novelty — it never **generates** the candidate itself. So the honest
limit was: *"ultimately whatever the LLM proposes is what gets verified."* Truly novel — not yet.

This module closes that gap. The **Eureka Engine** *creates* its own candidates — by combinatorial
and evolutionary search over a small formal grammar, and by **generalising** a lucky numeric
observation into a symbolic theorem — with **no LLM in the loop at all**. It then lives by the same
asymmetry NYXARA already trusts: *generation is cheap and unbounded; verification is exact.* Every
self-made conjecture is handed to the :class:`~nyxara.growth.prover.Prover` (the existing
machine-checkable theorem-checker over the decidable domains), and **only what is certified
``PROVEN`` survives** — never a guess, never a bluff. A refuted conjecture is discarded with its
counter-example; an undecidable one is abstained on.

Of what survives, the engine keeps only what is *genuinely new*: each proof is scored for **novelty**
against the :class:`~nyxara.growth.frontier.NoveltyArchive` (distance from everything she has ever
discovered) and for **interestingness** (a non-triviality / surprise / compression score that throws
away ``2 + 2 = 4`` and rewards a compact, general, certified identity). A discovery is kept iff it is
**proven ∧ novel ∧ interesting** — and the kept discovery is fed back into the frontier so the next
generation pushes into still-unexplored niches. That feedback is what makes the search *open-ended*:
she does not converge, she keeps walking off the map.

Everything she certifies is folded back through the machinery that already exists — long-term memory
(one record per breakthrough), the grounded knowledge base, and the verified-data flywheel that feeds
the foundry's training corpus — so a self-invented, machine-checked theorem becomes part of what she
*knows* and what she *trains on*. This is "truly novel" in the only honest form: novelty that is
**certified**, not asserted, and therefore confined to the domains where truth is decidable.

Design rules (the growth/ contract): pure standard library at the core (``random`` / ``ast`` /
``fractions`` / ``math``); ``sympy`` / ``z3`` are reached *only* through the optional paths inside the
:class:`Prover` and only ever *strengthen* a verdict. It is gather-only — it never acts on the world,
never reaches the network, touches no source, no weights, and no gate. Fully deterministic under a
fixed seed and CI-safe. Character-locked like all of growth/.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from fractions import Fraction
from random import Random
from typing import Any, Dict, List, Optional, Tuple

from nyxara.growth.prover import ProofClaim, ProofResult, ProofVerdict, Prover, _eval_rational

__all__ = ["Conjecture", "Breakthrough", "BreakthroughReport", "EurekaEngine", "LemmaLibrary"]

# The verifiable domains over which the engine invents and certifies. Each maps to a Prover ``kind``.
DOMAINS: Tuple[str, ...] = ("algebra", "arithmetic", "logic", "number_theory", "inequality")

_MEM_TAG = "eureka-breakthrough"


# --------------------------------------------------------------------------- #
# A self-generated candidate — the engine's own proposal, before verification
# --------------------------------------------------------------------------- #
@dataclass
class Conjecture:
    """One candidate the engine *invented* (not prompted). Carries what the Prover needs."""

    domain: str                                  # Prover kind: algebra | arithmetic | logic | ...
    statement: str
    candidate_answer: Optional[str] = None       # for substitution/inequality checks
    origin: str = "seed"                         # seed | mutate | crossover | generalize
    lineage: Tuple[str, ...] = ()                # parent statements / observed instance
    ops: int = 0                                 # operator count — a cheap complexity proxy
    # the expression genome behind an algebraic conjecture (when one exists) — carried so the next
    # generation can *genuinely* mutate/recombine its subtrees, not merely re-seed a template.
    tree: Any = field(default=None, repr=False, compare=False)

    def key(self) -> str:
        """A canonical de-duplication key (domain + whitespace-stripped statement)."""
        return f"{self.domain}::{''.join(self.statement.split())}"

    def to_dict(self) -> Dict[str, Any]:
        return {"domain": self.domain, "statement": self.statement,
                "candidate_answer": self.candidate_answer, "origin": self.origin,
                "lineage": list(self.lineage), "ops": self.ops}


# --------------------------------------------------------------------------- #
# A kept discovery — proven, novel, interesting
# --------------------------------------------------------------------------- #
@dataclass
class Breakthrough:
    """A conjecture that survived: certified by the Prover, far from the known, and non-trivial."""

    conjecture: Conjecture
    proof: ProofResult
    novelty: float = 0.0
    interestingness: float = 0.0
    at: float = field(default_factory=time.time)

    @property
    def statement(self) -> str:
        return self.conjecture.statement

    @property
    def certificate(self) -> str:
        return self.proof.certificate

    def score(self) -> float:
        """The selection score that ranks survivors — novelty × interestingness."""
        return round(self.novelty * self.interestingness, 6)

    def to_dict(self) -> Dict[str, Any]:
        return {"statement": self.statement, "domain": self.conjecture.domain,
                "origin": self.conjecture.origin, "lineage": list(self.conjecture.lineage),
                "verdict": self.proof.verdict.value, "certificate": self.certificate,
                "method": self.proof.method,
                "novelty": round(float(self.novelty), 6),
                "interestingness": round(float(self.interestingness), 6),
                "score": self.score(), "at": self.at}


# --------------------------------------------------------------------------- #
# The auditable summary of one open-ended discovery run
# --------------------------------------------------------------------------- #
@dataclass
class BreakthroughReport:
    """Everything one ``discover`` run produced — nothing kept silently."""

    generations: int = 0
    population: int = 0
    generated: int = 0
    proven: int = 0
    refuted: int = 0
    abstained: int = 0
    novel_kept: int = 0
    # certified identities promoted into the self-extending lemma library this run + its total size
    lemmas_added: int = 0
    lemma_library_size: int = 0
    by_domain: Dict[str, int] = field(default_factory=dict)   # kept count per domain
    breakthroughs: List[Breakthrough] = field(default_factory=list)
    fed_knowledge: int = 0
    fed_flywheel: int = 0
    frontier: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    elapsed_ms: float = 0.0

    def best(self) -> Optional[Breakthrough]:
        return max(self.breakthroughs, key=Breakthrough.score) if self.breakthroughs else None

    def to_dict(self) -> Dict[str, Any]:
        return {"generations": self.generations, "population": self.population,
                "generated": self.generated, "proven": self.proven, "refuted": self.refuted,
                "abstained": self.abstained, "novel_kept": self.novel_kept,
                "lemmas_added": self.lemmas_added, "lemma_library_size": self.lemma_library_size,
                "by_domain": dict(self.by_domain),
                "breakthroughs": [b.to_dict() for b in self.breakthroughs[:12]],
                "best": self.best().to_dict() if self.best() else None,
                "fed_knowledge": self.fed_knowledge, "fed_flywheel": self.fed_flywheel,
                "frontier": self.frontier, "reason": self.reason,
                "elapsed_ms": round(self.elapsed_ms, 3)}


# --------------------------------------------------------------------------- #
# A tiny exact integer-polynomial in one variable — for *constructing* identities
# (factored ↔ expanded), so the engine can manufacture genuinely-true theorems to certify.
# --------------------------------------------------------------------------- #
class _Poly:
    """``{power: coeff}`` over the integers, in the single variable ``n``. Pure stdlib."""

    __slots__ = ("c",)

    def __init__(self, coeffs: Optional[Dict[int, int]] = None) -> None:
        self.c: Dict[int, int] = {p: v for p, v in (coeffs or {}).items() if v != 0}

    def __mul__(self, other: "_Poly") -> "_Poly":
        out: Dict[int, int] = {}
        for p1, v1 in self.c.items():
            for p2, v2 in other.c.items():
                out[p1 + p2] = out.get(p1 + p2, 0) + v1 * v2
        return _Poly(out)

    def __add__(self, other: "_Poly") -> "_Poly":
        out = dict(self.c)
        for p, v in other.c.items():
            out[p] = out.get(p, 0) + v
        return _Poly(out)

    def __sub__(self, other: "_Poly") -> "_Poly":
        out = dict(self.c)
        for p, v in other.c.items():
            out[p] = out.get(p, 0) - v
        return _Poly(out)

    def __pow__(self, exp: int) -> "_Poly":
        exp = int(exp)
        if exp < 0:
            raise ValueError("negative polynomial power")
        result = _Poly({0: 1})
        base = _Poly(dict(self.c))
        for _ in range(exp):
            result = result * base
        return result

    def degree(self) -> int:
        return max(self.c) if self.c else 0

    def key(self) -> Tuple[Tuple[int, int], ...]:
        """A canonical, hashable identity of this polynomial (for de-dup / lemma equality)."""
        return tuple(sorted(self.c.items()))

    @staticmethod
    def var() -> "_Poly":
        """The identity polynomial ``n`` -> ``{1: 1}``."""
        return _Poly({1: 1})

    @staticmethod
    def const(value: int) -> "_Poly":
        return _Poly({0: int(value)})

    def render(self) -> str:
        """A canonical, prover-parseable string like ``n^2 + -1*n + 6`` (descending powers)."""
        if not self.c:
            return "0"
        terms: List[str] = []
        for p in sorted(self.c, reverse=True):
            v = self.c[p]
            if p == 0:
                terms.append(f"{v}")
            elif p == 1:
                terms.append(f"{v}*n")
            else:
                terms.append(f"{v}*n^{p}")
        return " + ".join(terms)

    @staticmethod
    def linear(root: int) -> "_Poly":
        """The factor ``(n - root)`` -> ``{1: 1, 0: -root}``."""
        return _Poly({1: 1, 0: -root})


# --------------------------------------------------------------------------- #
# The expression GENOME — a real syntax tree over one variable ``n`` that the engine
# recombines. This is what makes mutate/crossover *genuine* structural recombination instead
# of re-seeding a fixed template: a child shares real subtrees with its parents, and because a
# tree is expanded to its exact canonical polynomial for the right-hand side, every recombination
# still yields a *provable* identity — the Prover certifies or refutes it, never the engine.
# --------------------------------------------------------------------------- #
_BINOPS: Tuple[str, ...] = ("add", "sub", "mul")
_MAX_NODES = 22            # hard cap on tree size (mutation/crossover reject past this)
_MAX_DEGREE = 10           # reject an expansion whose polynomial degree explodes past this
_MAX_COEFF = 1_000_000     # reject an expansion whose coefficients blow up (keeps lemmas tidy + fast)


@dataclass
class _Node:
    """One node of the expression tree. ``kind`` ∈ {var, const, lemma, add, sub, mul, pow}."""

    kind: str
    value: Any = None                                    # const int / pow exponent / (lemma_name, _Poly)
    children: Tuple["_Node", ...] = ()

    def copy(self) -> "_Node":
        return _Node(self.kind, self.value, tuple(c.copy() for c in self.children))

    def nodes(self) -> List["_Node"]:
        """This node and every descendant, pre-order — the pool mutation/crossover picks from."""
        out = [self]
        for c in self.children:
            out.extend(c.nodes())
        return out

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def lemmas_used(self) -> List[str]:
        found: List[str] = []
        for nd in self.nodes():
            if nd.kind == "lemma" and nd.value is not None:
                found.append(nd.value[0])
        return found

    def render(self) -> str:
        """Render to a prover-parseable string — always explicit ``*`` and parenthesised."""
        if self.kind == "var":
            return "n"
        if self.kind == "const":
            return str(int(self.value))
        if self.kind == "lemma":
            return f"({self.value[1].render()})"
        if self.kind == "pow":
            return f"({self.children[0].render()})^{int(self.value)}"
        op = {"add": " + ", "sub": " - ", "mul": " * "}[self.kind]
        return f"({self.children[0].render()}{op}{self.children[1].render()})"

    def evaluate(self) -> _Poly:
        """Expand the tree to its exact canonical polynomial (this becomes the identity's RHS)."""
        if self.kind == "var":
            return _Poly.var()
        if self.kind == "const":
            return _Poly.const(int(self.value))
        if self.kind == "lemma":
            return _Poly(dict(self.value[1].c))
        if self.kind == "pow":
            return self.children[0].evaluate() ** int(self.value)
        a, b = self.children[0].evaluate(), self.children[1].evaluate()
        if self.kind == "add":
            return a + b
        if self.kind == "sub":
            return a - b
        return a * b


class _TreeFactory:
    """Builds and recombines expression trees deterministically from a shared RNG, drawing terminals
    from ``{n, small integers}`` **plus** the lemmas NYXARA has herself proven (a self-extending
    alphabet). All growth bounded so the expanded polynomial can never explode."""

    def __init__(self, rng: Random, lemmas: "LemmaLibrary") -> None:
        self.rng = rng
        self.lemmas = lemmas

    def _terminal(self, allow_var: bool) -> _Node:
        pool = self.lemmas.terminals()
        # ~30% of the time, if she has proven lemmas, compose over one of her own theorems.
        if allow_var and pool and self.rng.random() < 0.30:
            name, poly = self.rng.choice(pool)
            return _Node("lemma", (name, poly))
        if allow_var and self.rng.random() < 0.5:
            return _Node("var")
        return _Node("const", self.rng.randint(-6, 6))

    def random(self, depth: int, allow_var: bool = True) -> _Node:
        if depth <= 0 or self.rng.random() < 0.32:
            return self._terminal(allow_var)
        roll = self.rng.random()
        if roll < 0.22:                                  # a power of a smaller subtree
            base = self.random(depth - 1, allow_var)
            return _Node("pow", self.rng.randint(2, 3), (base,))
        kind = self.rng.choice(_BINOPS)
        left = self.random(depth - 1, allow_var)
        right = self.random(depth - 1, allow_var)
        return _Node(kind, None, (left, right))

    def _acceptable(self, tree: _Node) -> bool:
        if tree.size() > _MAX_NODES:
            return False
        try:
            poly = tree.evaluate()
        except Exception:  # noqa: BLE001 — a pathological expansion is simply rejected
            return False
        if poly.degree() > _MAX_DEGREE:
            return False
        return all(abs(v) <= _MAX_COEFF for v in poly.c.values())

    def fresh(self, allow_var: bool = True, depth: int = 3) -> _Node:
        for _ in range(8):
            t = self.random(depth, allow_var)
            if self._acceptable(t):
                return t
        return _Node("var") if allow_var else _Node("const", self.rng.randint(1, 9))

    def mutate(self, parent: _Node, allow_var: bool = True) -> Optional[_Node]:
        """Genuine subtree mutation: copy the parent, replace one randomly chosen node with a fresh
        subtree (or point-mutate a constant), keeping shared structure with the parent."""
        for _ in range(8):
            child = parent.copy()
            pool = child.nodes()
            target = self.rng.choice(pool)
            if target.kind == "const" and self.rng.random() < 0.5:
                target.value = int(target.value) + self.rng.choice((-3, -2, -1, 1, 2, 3))
            else:
                repl = self.random(self.rng.randint(0, 2), allow_var)
                target.kind, target.value, target.children = repl.kind, repl.value, repl.children
            if self._acceptable(child):
                return child
        return None

    def crossover(self, a: _Node, b: _Node, allow_var: bool = True) -> Optional[_Node]:
        """Genuine subtree crossover: splice a randomly chosen subtree of ``b`` into a copy of ``a``."""
        for _ in range(8):
            child = a.copy()
            donor = self.rng.choice(b.nodes()).copy()
            target = self.rng.choice(child.nodes())
            target.kind, target.value, target.children = donor.kind, donor.value, donor.children
            if self._acceptable(child):
                return child
        return None


# --------------------------------------------------------------------------- #
# The LEMMA LIBRARY — every proven ∧ novel ∧ interesting algebraic identity becomes a named,
# reusable definition the grammar can compose over in later generations. This is the honest form
# of "inventing beyond the templates a human wrote down": the *alphabet itself grows* from what
# NYXARA has certified. Persisted through memory (best-effort) so invention compounds across runs.
# --------------------------------------------------------------------------- #
_LEMMA_TAG = "eureka-lemma-library"


@dataclass
class _Lemma:
    name: str
    poly: _Poly
    statement: str
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "coeffs": {str(p): v for p, v in self.poly.c.items()},
                "statement": self.statement, "at": self.at}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_Lemma":
        coeffs = {int(p): int(v) for p, v in (d.get("coeffs") or {}).items()}
        return cls(name=str(d.get("name", "L?")), poly=_Poly(coeffs),
                   statement=str(d.get("statement", "")), at=float(d.get("at", 0.0)))


class LemmaLibrary:
    """A growing, deduplicated set of NYXARA's own certified algebraic identities, usable as terminals."""

    def __init__(self, *, memory: Any = None, max_lemmas: int = 64) -> None:
        self.memory = memory
        self.max_lemmas = int(max_lemmas)
        self._lemmas: List[_Lemma] = []
        self._keys: set = set()
        self._counter = 0
        self.added_this_session = 0
        self._load()

    def __len__(self) -> int:
        return len(self._lemmas)

    def terminals(self) -> List[Tuple[str, _Poly]]:
        """The reusable terminals the grammar may compose over — her proven, non-trivial lemmas."""
        return [(lm.name, lm.poly) for lm in self._lemmas]

    def add(self, poly: _Poly, statement: str) -> Optional[str]:
        """Record a certified identity as a named lemma; returns the name, or None if trivial/dup."""
        if poly.degree() < 1 or not poly.c:              # skip constants / the empty polynomial
            return None
        k = poly.key()
        if k in self._keys:
            return None
        name = f"L{self._counter}"
        self._counter += 1
        self._keys.add(k)
        self._lemmas.append(_Lemma(name=name, poly=poly, statement=statement))
        self.added_this_session += 1
        if len(self._lemmas) > self.max_lemmas:          # keep the most recent (freshest niches)
            drop = self._lemmas.pop(0)
            self._keys.discard(drop.poly.key())
        return name

    def to_dict(self) -> Dict[str, Any]:
        return {"counter": self._counter, "lemmas": [lm.to_dict() for lm in self._lemmas]}

    def from_dict(self, d: Dict[str, Any]) -> None:
        self._counter = int(d.get("counter", 0))
        for ld in d.get("lemmas", []):
            lm = _Lemma.from_dict(ld)
            k = lm.poly.key()
            if k in self._keys:
                continue
            self._keys.add(k)
            self._lemmas.append(lm)

    # ---- persistence (best-effort, mirrors FrontierEngine.save) ---- #
    def _load(self) -> None:
        if self.memory is None:
            return
        try:
            for rec in self.memory._kv.values():
                if _LEMMA_TAG in rec.tags and "library" in rec.metadata:
                    self.from_dict(rec.metadata.get("library", {}))
                    return
        except Exception:  # noqa: BLE001 — a fresh library is a valid state
            return

    def save(self) -> bool:
        if self.memory is None or not self._lemmas:
            return False
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
        except Exception:  # noqa: BLE001
            return False
        try:
            for rec in list(getattr(self.memory, "_kv", {}).values()):
                if _LEMMA_TAG in rec.tags and "library" in rec.metadata:
                    self.memory.forget(rec.mem_id)
            self.memory.remember(
                f"[eureka-lemmas] {len(self._lemmas)} self-certified identities",
                mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.9),
                importance=1.0, tags=[_LEMMA_TAG, "discovery"],
                metadata={"library": self.to_dict()})
            return True
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            return False


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class EurekaEngine:
    """Invent candidates by search, certify them with the Prover, keep only the novel + interesting.

    Parameters
    ----------
    prover:    the machine-checkable verifier (defaults to a fresh :class:`Prover`).
    frontier:  a :class:`~nyxara.growth.frontier.FrontierEngine` for novelty + open-ended direction
               (defaults to one bound to ``memory`` — falls back to in-process state if absent).
    memory / knowledge / flywheel:
               optional sinks; each kept breakthrough is remembered, grounded, and fed to the
               verified-data flywheel when present. All best-effort — a missing sink never raises.
    settings:  :class:`~nyxara.kernel.config.NyxaraSettings` (only its memory tag is read).
    seed:      makes the whole search deterministic and CI-reproducible.
    """

    def __init__(self, *, prover: Any = None, frontier: Any = None, memory: Any = None,
                 knowledge: Any = None, flywheel: Any = None, settings: Any = None,
                 seed: int = 1337, novelty_floor: float = 0.06,
                 interest_floor: float = 0.30) -> None:
        self.prover = prover or Prover(seed=seed)
        self.memory = memory
        self.knowledge = knowledge
        self.flywheel = flywheel
        self.rng = Random(seed)
        self.novelty_floor = float(novelty_floor)
        self.interest_floor = float(interest_floor)
        self._frontier = frontier or self._default_frontier(memory, settings)
        # PROVEN elites per domain — the parents the next generation mutates / recombines.
        self._elites: Dict[str, List[Conjecture]] = {d: [] for d in DOMAINS}
        self._seen: set[str] = set()
        self.total_breakthroughs: int = 0   # cumulative kept discoveries across all runs
        # the self-extending grammar: her own certified identities become reusable terminals, and a
        # deterministic tree factory does the *genuine* subtree recombination (no template re-seeding).
        self.lemmas = LemmaLibrary(memory=memory)
        self._trees = _TreeFactory(self.rng, self.lemmas)

    @staticmethod
    def _default_frontier(memory: Any, settings: Any) -> Any:
        try:
            from nyxara.growth.frontier import FrontierEngine
            return FrontierEngine(memory=memory, settings=settings)
        except Exception:  # noqa: BLE001 — novelty scoring degrades to "always novel" without it
            return None

    # ================================================================== #
    # Public entry point — the open-ended evolutionary discovery loop.
    # ================================================================== #
    def discover(self, generations: int = 4, population: int = 24) -> BreakthroughReport:
        """Run ``generations`` rounds of invent → verify → score → select. Always returns a report.

        Each generation searches **all** verifiable domains in parallel (population split evenly),
        plus a few *generalisations* — lucky numeric instances lifted to symbolic laws — so the
        frontier is pushed on every axis at once, not one domain at a time."""
        t0 = time.perf_counter()
        generations = max(1, int(generations))
        population = max(len(DOMAINS), int(population))
        per_domain = max(1, population // len(DOMAINS))
        report = BreakthroughReport(generations=generations, population=population)
        for _ in range(generations):
            batch: List[Conjecture] = []
            for domain in DOMAINS:
                batch.extend(self._propose(domain, per_domain))
            batch.extend(self._generalizations(max(2, per_domain // 2)))
            for conj in batch:
                report.generated += 1
                if conj.key() in self._seen:
                    continue
                self._seen.add(conj.key())
                proof = self._verify(conj)
                if proof.verdict is ProofVerdict.REFUTED:
                    report.refuted += 1
                    continue
                if proof.verdict is not ProofVerdict.PROVEN:
                    report.abstained += 1
                    continue
                report.proven += 1
                self._elites.setdefault(conj.domain, []).append(conj)   # a proven parent
                bt = self._consider(conj, proof)
                if bt is not None:
                    report.breakthroughs.append(bt)
                    report.novel_kept += 1
                    report.by_domain[conj.domain] = report.by_domain.get(conj.domain, 0) + 1
                    report.fed_knowledge += self._feed_knowledge(bt)
                    report.fed_flywheel += self._feed_flywheel(bt)
                    self._remember(bt)
                    report.lemmas_added += self._promote_lemma(bt)
            self._prune_elites()
        report.breakthroughs.sort(key=Breakthrough.score, reverse=True)
        self.total_breakthroughs += report.novel_kept
        report.lemma_library_size = len(self.lemmas)
        if report.lemmas_added:
            self.lemmas.save()
        report.frontier = self._frontier_snapshot()
        report.reason = (f"invented {report.generated}, proven {report.proven}, "
                         f"kept {report.novel_kept} novel+interesting across {generations} gen(s)")
        report.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return report

    # convenience alias mirroring the rest of growth/ (``run`` like SyntheticCurator/MetaResearcher)
    def run(self, generations: int = 4, population: int = 24) -> BreakthroughReport:
        return self.discover(generations=generations, population=population)

    # ================================================================== #
    # Step 1 — INVENT a population of candidates (no LLM, ever).
    # ================================================================== #
    def _propose(self, domain: str, n: int) -> List[Conjecture]:
        out: List[Conjecture] = []
        elites = self._elites.get(domain, [])
        for _ in range(n):
            r = self.rng.random()
            conj: Optional[Conjecture] = None
            if elites and r < 0.30:
                conj = self._mutate(self.rng.choice(elites))
            elif len(elites) >= 2 and r < 0.45:
                conj = self._crossover(self.rng.choice(elites), self.rng.choice(elites))
            else:
                conj = self._seed(domain)
            if conj is not None and conj.statement:
                out.append(conj)
        return out

    def _generalizations(self, count: int) -> List[Conjecture]:
        """A few candidates per generation that lift a numeric observation to a symbolic law."""
        out: List[Conjecture] = []
        attempts = 0
        while len(out) < count and attempts < count * 40:
            attempts += 1
            gen = self._generalize()
            if gen is not None:
                out.append(gen)
        return out

    def _seed(self, domain: str) -> Optional[Conjecture]:
        return {
            "algebra": self._seed_algebra,
            "arithmetic": self._seed_arithmetic,
            "logic": self._seed_logic,
            "number_theory": self._seed_number_theory,
            "inequality": self._seed_inequality,
        }.get(domain, self._seed_algebra)()

    # ---- algebra: an expression TREE ↔ its exact expansion — genuine, recombinable identities ---- #
    def _conj_from_tree(self, tree: Any, *, domain: str, origin: str,
                        lineage: Tuple[str, ...] = ()) -> Optional[Conjecture]:
        """Turn an expression tree into a conjecture ``tree = expansion``. The right-hand side is the
        tree's *exact* canonical polynomial, so a true instance is genuinely provable; ~25% of the
        time the expansion is perturbed into a real, refutable false guess. Returns None if trivial."""
        try:
            poly = tree.evaluate()
        except Exception:  # noqa: BLE001 — a pathological tree is simply dropped
            return None
        lhs = tree.render()
        true_rhs = poly.render()
        rhs = true_rhs
        if self.rng.random() < 0.25:                     # a real, refutable perturbation
            rhs = self._perturb_poly(poly).render()
        if "".join(lhs.split()) == "".join(rhs.split()):  # already canonical ⇒ vacuous "x = x"
            return None
        ops = sum(1 for nd in tree.nodes() if nd.kind in _BINOPS or nd.kind == "pow")
        lin = lineage + tuple(f"uses {nm}" for nm in dict.fromkeys(tree.lemmas_used()))
        return Conjecture(domain=domain, statement=f"{lhs} = {rhs}", origin=origin,
                          lineage=lin, ops=ops, tree=tree)

    def _seed_algebra(self) -> Optional[Conjecture]:
        tree = self._trees.fresh(allow_var=True, depth=self.rng.choice((2, 3, 3)))
        return self._conj_from_tree(tree, domain="algebra", origin="seed")

    def _promote_lemma(self, bt: Breakthrough) -> int:
        """Record a proven algebraic identity as a reusable lemma so later generations compose over
        it — the self-extending alphabet. Only genuine (degree ≥ 1) identities from a real tree."""
        conj = bt.conjecture
        if conj.domain != "algebra" or conj.tree is None:
            return 0
        try:
            name = self.lemmas.add(conj.tree.evaluate(), conj.statement)
        except Exception:  # noqa: BLE001
            return 0
        return 1 if name else 0

    def _perturb_poly(self, p: _Poly) -> _Poly:
        c = dict(p.c) or {0: 0}
        power = self.rng.choice(list(c.keys()))
        c[power] = c.get(power, 0) + self.rng.choice((-2, -1, 1, 2))
        return _Poly(c)

    # ---- arithmetic: a constant expression TREE ↔ its exact value (the Prover decides) ---- #
    def _seed_arithmetic(self) -> Optional[Conjecture]:
        tree = self._trees.fresh(allow_var=False, depth=self.rng.choice((2, 2, 3)))
        return self._conj_from_tree(tree, domain="arithmetic", origin="seed")

    # ---- logic: equivalence between a formula and a transform of it (De Morgan etc.) ---- #
    def _seed_logic(self) -> Conjecture:
        g = self._rand_bool(depth=self.rng.choice((1, 2, 2)))
        if self.rng.random() < 0.30:
            g2 = self._rand_bool(depth=2)                 # likely *not* equivalent — refutable
        else:
            g2 = self._logic_rewrite(g)                   # an equivalence to certify
        ops = sum(g.count(t) for t in (" and ", " or ", "not "))
        return Conjecture(domain="logic", statement=f"({g}) <-> ({g2})", origin="seed", ops=ops)

    def _rand_bool(self, depth: int) -> str:
        var = self.rng.choice(("A", "B", "C"))
        if depth <= 0 or self.rng.random() < 0.34:
            return f"not {var}" if self.rng.random() < 0.4 else var
        op = self.rng.choice((" and ", " or "))
        return f"({self._rand_bool(depth - 1)}{op}{self._rand_bool(depth - 1)})"

    def _logic_rewrite(self, g: str) -> str:
        """A truth-preserving rewrite to certify: idempotence ``g <-> g or g`` / ``g and g``,
        or double negation ``g <-> not (not g)``. All are tautological equivalences the
        truth-table checker proves exactly."""
        choice = self.rng.random()
        if choice < 0.34:
            return f"({g}) or ({g})"      # idempotence of OR
        if choice < 0.67:
            return f"({g}) and ({g})"     # idempotence of AND
        return f"not (not ({g}))"         # double negation

    # ---- number theory: primality / divisibility conjectures the Prover can decide ---- #
    def _seed_number_theory(self) -> Conjecture:
        if self.rng.random() < 0.6:
            n = self.rng.randint(11, 999)
            return Conjecture(domain="number_theory", statement=f"is {n} prime",
                              origin="seed", ops=0)
        a = self.rng.randint(2, 30)
        b = a * self.rng.randint(2, 12) if self.rng.random() < 0.5 else self.rng.randint(2, 400)
        return Conjecture(domain="number_theory", statement=f"{a} divides {b}",
                          origin="seed", ops=0)

    # ---- inequality: a *universal* relation, decided by z3 (no point-check, never overstated) ---- #
    def _seed_inequality(self) -> Conjecture:
        a = self.rng.randint(1, 6)
        if self.rng.random() < 0.5:
            # perfect square (n + a)^2 >= 0 — always true, a genuine universal inequality.
            stmt, ops = f"n^2 + {2 * a}*n + {a * a} >= 0", 2
        else:
            # a*n^2 + b >= 0 — true iff b >= 0, so ~half are real, refutable conjectures.
            b = self.rng.randint(-9, 9)
            stmt, ops = f"{a}*n^2 + {b} >= 0", 2
        return Conjecture(domain="inequality", statement=stmt, origin="seed", ops=ops)

    # ---- mutate / crossover over proven elites — GENUINE structural recombination ---- #
    def _mutate(self, parent: Conjecture) -> Optional[Conjecture]:
        # Real subtree mutation when the parent carries an expression genome: the child keeps most of
        # the parent's structure and swaps one subtree (or point-mutates a constant) — not a re-seed.
        if parent.tree is not None and parent.domain in ("algebra", "arithmetic"):
            child_tree = self._trees.mutate(parent.tree, allow_var=(parent.domain == "algebra"))
            if child_tree is not None:
                conj = self._conj_from_tree(child_tree, domain=parent.domain, origin="mutate",
                                            lineage=(parent.statement[:60],))
                if conj is not None:
                    return conj
        # Domains without a tree genome (logic / number_theory / inequality) re-seed honestly.
        if parent.domain == "number_theory":
            return self._seed_number_theory()
        base = self._seed(parent.domain)
        if base is None:
            return None
        base.origin = "mutate"
        base.lineage = (parent.statement[:60],)
        return base

    def _crossover(self, a: Conjecture, b: Conjecture) -> Optional[Conjecture]:
        # Real subtree crossover when both parents share a tree genome: splice a subtree of ``b`` into
        # ``a`` so the child genuinely inherits structure from both — not a fresh template instance.
        if (a.tree is not None and b.tree is not None and a.domain == b.domain
                and a.domain in ("algebra", "arithmetic")):
            child_tree = self._trees.crossover(a.tree, b.tree, allow_var=(a.domain == "algebra"))
            if child_tree is not None:
                conj = self._conj_from_tree(child_tree, domain=a.domain, origin="crossover",
                                            lineage=(a.statement[:48], b.statement[:48]))
                if conj is not None:
                    return conj
        child = self._seed(a.domain if a.domain == b.domain else self.rng.choice((a.domain, b.domain)))
        if child is None:
            return None
        child.origin = "crossover"
        child.lineage = (a.statement[:48], b.statement[:48])
        return child

    # ---- the headline: observe a numeric instance, conjecture its general form ---- #
    def _generalize(self) -> Optional[Conjecture]:
        """Take a random polynomial in ``n``, propose a *re-expressed* form of it (mostly a true
        rearrangement, sometimes a perturbation), **observe** that the two agree at two distinct
        points, then conjecture they are *identically* equal — and let the Prover certify or refute
        the lifted law. The engine guesses generality from an instance; the Prover decides."""
        terms = self._rand_poly_terms()
        p = self._render_terms(terms)
        if self.rng.random() < 0.25:
            q = self._render_terms(self._rand_poly_terms())     # an independent guess — usually false
        else:
            q = self._render_terms(self._split_terms(terms))    # a true rearrangement to certify
        if "".join(p.split()) == "".join(q.split()):
            return None
        # Observe at two distinct points; only generalise from a *repeated*, *non-vacuous* instance.
        k1, k2 = self.rng.sample([x for x in range(-7, 8) if x != 0], 2)
        p1, q1 = _eval_rational(p, {"n": Fraction(k1)}), _eval_rational(q, {"n": Fraction(k1)})
        p2, q2 = _eval_rational(p, {"n": Fraction(k2)}), _eval_rational(q, {"n": Fraction(k2)})
        if None in (p1, q1, p2, q2) or p1 != q1 or p2 != q2:
            return None                                  # not observed to agree — nothing to lift
        if p1 == 0 and p2 == 0:
            return None                                  # both sides vanish — a vacuous "0 = 0"
        ops = p.count("*") + p.count("+") + q.count("*") + q.count("+")
        return Conjecture(domain="algebra", statement=f"{p} = {q}", origin="generalize",
                          lineage=(f"observed at n={k1},{k2}: {p} = {q}",), ops=ops)

    def _rand_poly_terms(self) -> List[Tuple[int, int]]:
        terms: List[Tuple[int, int]] = []
        for _ in range(self.rng.choice((2, 2, 3))):
            c = self.rng.choice([x for x in range(-4, 5) if x != 0])
            pw = self.rng.choice((0, 1, 1, 2, 3))
            terms.append((c, pw))
        return terms

    @staticmethod
    def _render_terms(terms: List[Tuple[int, int]]) -> str:
        parts = [f"{c}" if pw == 0 else (f"{c}*n" if pw == 1 else f"{c}*n^{pw}")
                 for c, pw in terms]
        return " + ".join(parts) if parts else "0"

    def _split_terms(self, terms: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Split some coefficients into two summands of the same power — a true rearrangement."""
        out: List[Tuple[int, int]] = []
        for c, pw in terms:
            if abs(c) >= 2 and self.rng.random() < 0.6:
                a = self.rng.randint(1, abs(c) - 1) * (1 if c > 0 else -1)
                out.append((a, pw))
                out.append((c - a, pw))
            else:
                out.append((c, pw))
        self.rng.shuffle(out)
        return out

    # ================================================================== #
    # Step 2 — VERIFY (the Prover decides; the engine never asserts truth).
    # ================================================================== #
    def _verify(self, conj: Conjecture) -> ProofResult:
        return self.prover.prove(ProofClaim(kind=conj.domain, statement=conj.statement,
                                            candidate_answer=conj.candidate_answer))

    # ================================================================== #
    # Step 3 — KEEP iff novel ∧ interesting.
    # ================================================================== #
    def _consider(self, conj: Conjecture, proof: ProofResult) -> Optional[Breakthrough]:
        interest = self._interestingness(conj, proof)
        if interest < self.interest_floor:
            return None
        novelty = self._novelty(conj)
        if novelty < self.novelty_floor:
            return None
        # Record it on the frontier so future generations push *past* it (open-endedness).
        self._ingest_frontier(conj, quality=interest)
        return Breakthrough(conjecture=conj, proof=proof, novelty=novelty, interestingness=interest)

    def _interestingness(self, conj: Conjecture, proof: ProofResult) -> float:
        """Throw away trivia; reward generality, structure, surprise and compression."""
        stmt = conj.statement
        if self._is_trivial(conj):
            return 0.0
        score = 0.0
        nvars = len(set(re.findall(r"\b[A-Za-z]\b", stmt)) & {"n", "A", "B", "C"})
        if nvars >= 1:                                     # holds for a *variable* -> general
            score += 0.45
        if conj.domain == "logic" and nvars <= 1:          # a one-variable tautology is shallow
            score -= 0.25
        score += min(0.30, 0.05 * conj.ops)                # structural richness
        if conj.origin == "generalize":                    # an instance lifted to a law — the prize
            score += 0.25
        if "Schwartz" in proof.certificate or "sympy" in proof.method:
            score += 0.10                                  # a real identity, not a single point
        compression = 1.0 - min(1.0, len(stmt) / 160.0)    # short + true == surprising
        score += 0.15 * compression
        # arithmetic single facts and point-checked inequalities are the least interesting
        if conj.domain == "arithmetic":
            score -= 0.20
        if conj.domain == "inequality" and conj.candidate_answer is not None:
            score -= 0.10
        return max(0.0, min(1.0, score))

    @staticmethod
    def _is_trivial(conj: Conjecture) -> bool:
        """Reject vacuous statements: both sides of ``=`` / ``<->`` identical after stripping."""
        for sep in ("<->", "="):
            if sep in conj.statement:
                lhs, _, rhs = conj.statement.partition(sep)
                # for "<->" the partition on "=" inside would misfire; guard by checking sep first
                if sep == "=" and ("<" in conj.statement or ">" in conj.statement):
                    return False
                return "".join(lhs.split()) == "".join(rhs.split())
        return False

    def _descriptor(self, conj: Conjecture) -> Any:
        """A 3-coordinate behaviour descriptor (archive-consistent) that *separates distinct
        theorems*: (domain × structural-class, complexity, content-fingerprint). The fingerprint is
        a deterministic hash of the canonical statement, so genuinely different theorems land in
        different niches while exact restatements collide (zero novelty)."""
        from nyxara.growth.frontier import BehaviorDescriptor
        di = DOMAINS.index(conj.domain) if conj.domain in DOMAINS else 0
        cls = self._structural_class(conj)
        d0 = _clamp01((di + min(cls, 5) / 6.0) / len(DOMAINS))
        d1 = _clamp01(conj.ops / 12.0)
        digest = hashlib.md5(conj.key().encode("utf-8")).hexdigest()[:8]
        d2 = (int(digest, 16) % 1000) / 1000.0
        return BehaviorDescriptor(dims=(d0, d1, d2),
                                  labels=(conj.domain, "complexity", "fingerprint"))

    @staticmethod
    def _structural_class(conj: Conjecture) -> int:
        """A coarse form-class: polynomial degree, logic var-count, or number-theory pattern."""
        s = conj.statement
        if conj.domain in ("algebra", "arithmetic", "inequality"):
            powers = [int(m) for m in re.findall(r"n\^(\d+)", s)]
            return max(powers) if powers else (1 if "n" in s else 0)
        if conj.domain == "logic":
            return len(set(re.findall(r"[ABC]", s)))
        if conj.domain == "number_theory":
            return 0 if "prime" in s else (1 if "divides" in s else 2)
        return 0

    def _novelty(self, conj: Conjecture) -> float:
        if self._frontier is None:
            return 1.0
        try:
            return float(self._frontier.archive().novelty(self._descriptor(conj)))
        except Exception:  # noqa: BLE001 — novelty is a filter, never a hard dependency
            return 1.0

    def _ingest_frontier(self, conj: Conjecture, *, quality: float) -> None:
        if self._frontier is None:
            return
        try:
            self._frontier.ingest(conj.statement, domain="mathematics", quality=quality,
                                  provenance="eureka", descriptor=self._descriptor(conj),
                                  save=False)
        except Exception:  # noqa: BLE001
            pass

    # ================================================================== #
    # Step 4 — FEED what survived back into memory / knowledge / training.
    # ================================================================== #
    def _feed_knowledge(self, bt: Breakthrough) -> int:
        if self.knowledge is None:
            return 0
        try:
            text = (f"Self-discovered theorem ({bt.conjecture.domain}): {bt.statement}\n"
                    f"Certificate: {bt.certificate} [{bt.proof.method}]")
            self.knowledge.ingest_text(text, source=f"eureka:{bt.conjecture.domain}")
            return 1
        except Exception:  # noqa: BLE001
            return 0

    def _feed_flywheel(self, bt: Breakthrough) -> int:
        if self.flywheel is None:
            return 0
        try:
            dec = self.flywheel.consider(
                prompt=f"Prove or refute ({bt.conjecture.domain}): {bt.statement}",
                answer=f"PROVEN. {bt.certificate} [{bt.proof.method}]",
                confidence=float(bt.proof.confidence), verified=True)
            return 1 if getattr(dec, "collected", False) else 0
        except Exception:  # noqa: BLE001
            return 0

    def _remember(self, bt: Breakthrough) -> bool:
        if self.memory is None:
            return False
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
        except Exception:  # noqa: BLE001
            return False
        try:
            content = f"[eureka] {bt.conjecture.domain}: {bt.statement}  ✓ {bt.certificate}"[:300]
            self.memory.remember(
                content, mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.9),
                importance=0.7, tags=[_MEM_TAG, bt.conjecture.domain, "discovery"],
                metadata={"breakthrough": bt.to_dict()})
            return True
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            return False

    # ================================================================== #
    # housekeeping
    # ================================================================== #
    def _prune_elites(self, keep: int = 16) -> None:
        for dom, lst in self._elites.items():
            if len(lst) > keep:
                self._elites[dom] = lst[-keep:]

    def _frontier_snapshot(self) -> Dict[str, Any]:
        if self._frontier is None:
            return {}
        try:
            self._frontier.save()
            return self._frontier.report()
        except Exception:  # noqa: BLE001
            return {}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _clamp01(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:  # noqa: BLE001
        return 0.0


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 74)
    print("NYXARA eureka (truly novel problem solving) self-test")
    print("=" * 74)
    eng = EurekaEngine(seed=7)
    rep = eng.discover(generations=4, population=28)
    print(f"\ninvented={rep.generated}  proven={rep.proven}  refuted={rep.refuted}  "
          f"abstained={rep.abstained}  kept(novel+interesting)={rep.novel_kept}")
    print(f"by domain: {rep.by_domain}")
    print(f"lemmas added this run: {rep.lemmas_added}  (library size: {rep.lemma_library_size})")
    if eng.lemmas.terminals():
        print("self-extending alphabet (her own certified lemmas, now reusable terminals):")
        for nm, poly in eng.lemmas.terminals()[:5]:
            print(f"  {nm}(n) := {poly.render()}")
    print("\ntop self-discovered, machine-certified theorems:")
    for b in rep.breakthroughs[:8]:
        print(f"  [{b.conjecture.domain:13s}|{b.conjecture.origin:10s}] {b.statement}")
        print(f"        ✓ {b.certificate}  (novelty={b.novelty:.3f} interest={b.interestingness:.3f})")
    print(f"\nfrontier: {rep.frontier}")
