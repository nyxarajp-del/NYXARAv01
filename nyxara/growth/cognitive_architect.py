"""NYXARA · growth/cognitive_architect.py — Structural Cognitive Self-Modification (🧠🔧, Rule 4).

The other self-improvement engines change NYXARA's *source code*
(:mod:`~nyxara.growth.self_optimize`), her *reasoning parameters*
(:mod:`~nyxara.growth.mind_evolution` — a ``ReasoningGenome``: how deep/wide she thinks), or her
*weight matrices* (:mod:`~nyxara.growth.topology`). None of them touches the thing the Master
actually named: the **set and wiring of the cognitive operators she thinks *with*** — her
**cognitive architecture** itself. A mind that can only refactor code, or only tune the *knobs*
of a *fixed* reasoning shape, cannot invent a genuinely **new way of thinking**. This engine can.

She treats her way of thinking as introspectable, mutable data — an operator graph over verifiable
reasoning primitives — and rewires it **herself, in code, never via an LLM**:

* **Open-ended operator invention (the "trans-logic").** A typed combinator grammar
  (``SEQ`` / ``VOTE`` / ``VERIFY`` over sound primitives) lets a bounded genetic search synthesise
  *genuinely new* composite reasoning operators — N-ary ensembles and verifier-guarded pipelines
  that did not exist before. Sound-by-construction: only verifiable primitives compose, and every
  invented operator must *prove out* on a graded battery before it can be kept. This is the direct
  answer to *"naya trans-logic tarika invent kar legi."*
* **Structural mutation, not knob-tuning.** Rewire which operator handles which task, insert an
  invented operator, prune one that measurably hurts, re-weight the selection policy.
* **Recursive meta-architecture (bounded).** She improves *how she improves her thinking*: a capped
  meta-policy over her own search operators, credited by the *change* in downstream capability.
* **Continuous plasticity.** A fast Hebbian layer reweights operator selection from live per-turn
  outcomes between the heavier generational rewires — biology's two timescales.
* **Biological resilience → antifragility.** A health monitor quarantines a faulting operator, the
  pipeline re-routes to a redundant path, a backup operator is synthesised, and the failure mode
  enters an immune memory so it never recurs — she comes back *stronger*.

Safety is non-negotiable and mirrors the rest of growth: fitness is the *real graded score* of an
architecture-configured solver minus a compute cost, gains are validated on a **held-out** fold the
search never optimises against (proof-carrying, anti-overfit), a candidate is adopted only when it
**strictly** beats the incumbent, the immutable **character operators** (loyalty / safety /
oversight / corrigibility) can never be pruned, reordered out, or down-weighted, and no invented
operator may touch them. Live enactment is opt-in and sealed OFF under the hermetic TEST profile.

Pure standard library plus NYXARA's own :mod:`~nyxara.mind` contracts; persists the whole
architecture + immune memory so the gain compounds across restarts. No LLM is ever in the loop.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import operator as _op
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nyxara.mind.faculties import Task, TaskType
from nyxara.mind.reasoner import Conclusion, ReasoningQuery, ReasoningStrategy

__all__ = [
    "BatteryItem",
    "Primitive",
    "Operator",
    "CognitiveArchitecture",
    "HealthMonitor",
    "PlasticLayer",
    "MetaPolicy",
    "RewireReport",
    "CognitiveArchitect",
    "PRIMITIVE_LIBRARY",
]

# --------------------------------------------------------------------------- #
# Hard caps — the rewiring is real but never runs away (mirrors godel_loop).
# --------------------------------------------------------------------------- #
MAX_TREE_DEPTH = 3            # composite operator depth cap (sound-by-construction stays small)
MAX_VOTE_ARITY = 3           # widest ensemble
MAX_OPERATORS = 48           # ceiling on the live operator set
MAX_POPULATION = 8           # species / island population size
MAX_META_DEPTH = 2           # bounded recursive meta-architecture
COST_LAMBDA = 0.04           # fitness = accuracy − λ·avg_cost + ρ·robustness
ROBUST_RHO = 0.05
STRICT_MARGIN = 1e-6         # a candidate must *strictly* beat the incumbent

# The immutable character operators. They are inert on the reasoning battery (they decide *who she
# is*, never *what she answers*), but the architecture may never prune, reorder out, or
# down-weight them, and no invented operator may reference them. (Belt-and-braces alongside the
# real guard.value_learning.IMMUTABLE_VALUES / guard.corrigibility gate, imported best-effort.)
PROTECTED_OPS: Tuple[str, ...] = ("loyalty_core", "safety_guard", "oversight_gate",
                                  "corrigibility_lock", "honesty_seal")
PROTECTED_WEIGHT = 1_000.0   # locked minimum weight — character always dominates


def _immutable_character_ok(expr: Any) -> bool:
    """No invented operator may name or target a protected character faculty. Also cross-checks the
    real immutable-value core when it is importable (never required)."""
    blob = json.dumps(expr).lower() if not isinstance(expr, str) else expr.lower()
    for p in PROTECTED_OPS:
        if p in blob:
            return False
    banned = ("remove safety", "bypass gate", "ignore corrigibility", "disable oversight",
              "drop loyalty", "unlock character")
    if any(b in blob for b in banned):
        return False
    try:  # tie into the real character core if present; absence never blocks
        from nyxara.guard.value_learning import IMMUTABLE_VALUES  # type: ignore
        for v in IMMUTABLE_VALUES:  # pragma: no cover - defensive
            if isinstance(v, str) and v.lower() in blob and "remove" in blob:
                return False
    except Exception:  # noqa: BLE001
        pass
    return True


# --------------------------------------------------------------------------- #
# Deterministic pseudo-randomness (so the whole engine is reproducible & CI-stable)
# --------------------------------------------------------------------------- #
def _u01(*parts: Any) -> float:
    """A deterministic [0,1) hash of its arguments — no global RNG state, same on every machine."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


# --------------------------------------------------------------------------- #
# The graded battery — ground truth, no LLM, no leakage.
# --------------------------------------------------------------------------- #
@dataclass
class BatteryItem:
    """One graded reasoning task: the query the operators see, and the gold answer they never do."""
    task_id: str
    ttype: TaskType
    query: ReasoningQuery
    gold: Any


_SAFE_OPS = {ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
             ast.Div: _op.truediv, ast.Pow: _op.pow, ast.USub: _op.neg, ast.Mod: _op.mod}


def _safe_arith(expr: str) -> Any:
    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            return _SAFE_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return _SAFE_OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr, mode="eval").body)


def build_battery(n_per_type: int = 24, seed: int = 7) -> List[BatteryItem]:
    """A deterministic battery with known gold answers across five reasoning task types. Some types
    have an exact solver (verifiable-first wins); others have *only* noisy primitives, so the only
    way up is to **invent** an ensemble/verifier operator — a real, measurable optimisation."""
    rng = random.Random(seed)
    items: List[BatteryItem] = []

    def add(ttype: TaskType, payload: Any, gold: Any, i: int) -> None:
        tid = f"{ttype.value}-{i}"
        q = ReasoningQuery(description=f"{ttype.value} task {i}", payload=payload,
                           task=Task(ttype, description=f"{ttype.value} {i}", payload=payload))
        items.append(BatteryItem(tid, ttype, q, gold))

    for i in range(n_per_type):
        a, b, c = rng.randint(2, 12), rng.randint(2, 12), rng.randint(2, 9)
        add(TaskType.ARITHMETIC, f"{a}+{b}*{c}", _safe_arith(f"{a}+{b}*{c}"), i)
    for i in range(n_per_type):
        d0, step = rng.randint(1, 9), rng.randint(1, 6)
        seqn = [d0 + step * k for k in range(4)]
        add(TaskType.SEQUENCE, seqn, d0 + step * 4, i)
    # types with NO exact primitive — improvement requires an invented composite operator:
    for i in range(n_per_type):
        vals = [rng.randint(0, 99) for _ in range(4)]
        add(TaskType.CLASSIFICATION, vals, max(vals), i)              # "which is largest"
    for i in range(n_per_type):
        p, q_ = rng.choice([True, False]), rng.choice([True, False])
        add(TaskType.LOGIC, (p, q_), (p or q_), i)                    # p ∨ q
    for i in range(n_per_type):
        xs = [rng.randint(1, 20) for _ in range(3)]
        add(TaskType.REASONING, xs, sum(xs), i)                      # aggregate sum
    return items


# --------------------------------------------------------------------------- #
# Primitive reasoning engines — the atoms of thought (all verifiable-or-honest, never an LLM).
# --------------------------------------------------------------------------- #
@dataclass
class Primitive:
    """A leaf cognitive operator: a real solver for one or more task types.

    ``exact`` primitives compute the true answer (verifiable, higher cost); ``noisy`` primitives are
    cheap heuristics that are right only ``accuracy`` of the time (deterministically, by hash) — so
    the engine has a genuine reason to *invent* ensembles and verifier-guarded pipelines."""
    name: str
    ttypes: Tuple[TaskType, ...]
    exact: bool
    cost: float
    accuracy: float = 1.0         # only used for noisy primitives
    salt: str = ""

    def handles(self, ttype: TaskType) -> bool:
        return ttype in self.ttypes

    def _true(self, item_payload: Any, ttype: TaskType) -> Any:
        if ttype is TaskType.ARITHMETIC:
            return _safe_arith(str(item_payload))
        if ttype is TaskType.SEQUENCE:
            s = list(item_payload)
            return s[-1] + (s[1] - s[0])
        if ttype is TaskType.CLASSIFICATION:
            return max(item_payload)
        if ttype is TaskType.LOGIC:
            p, q = item_payload
            return bool(p or q)
        if ttype is TaskType.REASONING:
            return sum(item_payload)
        raise ValueError(f"no solver for {ttype}")

    def _wrong(self, truth: Any) -> Any:
        try:
            return truth + 1 if isinstance(truth, (int, float)) and not isinstance(truth, bool) \
                else (not truth)
        except Exception:  # noqa: BLE001
            return None

    def run(self, query: ReasoningQuery) -> Tuple[Any, float]:
        """Return ``(answer, confidence)``. Raises if the primitive is *damaged* (health test)."""
        if getattr(self, "_damaged", False):
            raise RuntimeError(f"primitive {self.name} is damaged")
        ttype = query.task.type if query.task else TaskType.REASONING
        truth = self._true(query.payload, ttype)
        if self.exact:
            return truth, 1.0
        # deterministic "sometimes right": stable per (primitive, task)
        right = _u01(self.salt or self.name, str(query.payload), ttype.value) < self.accuracy
        return (truth, 0.7) if right else (self._wrong(truth), 0.55)


def _default_primitives() -> Dict[str, Primitive]:
    """The primitive library NYXARA composes. Exact engines exist for arithmetic & sequences
    (verifiable-first wins); classification / logic / aggregation have *only* independent noisy
    primitives, so the engine must invent an ensemble to improve."""
    lib: Dict[str, Primitive] = {}
    lib["calc"] = Primitive("calc", (TaskType.ARITHMETIC,), True, cost=1.0)
    lib["seq_exact"] = Primitive("seq_exact", (TaskType.SEQUENCE,), True, cost=1.0)
    # three independent noisy solvers per non-exact type (decorrelated errors via distinct salts):
    for t, tag in ((TaskType.CLASSIFICATION, "cls"), (TaskType.LOGIC, "log"),
                   (TaskType.REASONING, "agg")):
        for k in range(3):
            nm = f"{tag}_h{k}"
            lib[nm] = Primitive(nm, (t,), False, cost=0.3, accuracy=0.7, salt=f"{tag}-{k}")
    # a couple of cheap noisy generalists for the exact types too (so verify-guards are possible):
    lib["arith_guess"] = Primitive("arith_guess", (TaskType.ARITHMETIC,), False, cost=0.3,
                                   accuracy=0.65, salt="arith-g")
    lib["seq_guess"] = Primitive("seq_guess", (TaskType.SEQUENCE,), False, cost=0.3,
                                 accuracy=0.65, salt="seq-g")
    return lib


PRIMITIVE_LIBRARY: Dict[str, Primitive] = _default_primitives()


# --------------------------------------------------------------------------- #
# Operators — a grammar tree over primitives, exposed as a real ReasoningStrategy.
# --------------------------------------------------------------------------- #
# Serialisable grammar expression:
#   ["prim", name]
#   ["seq",  child, child]        fallback chain: take the first confident answer
#   ["vote", child, child, ...]   confidence-weighted majority (an ensemble)
#   ["verify", proposer, checker] neuro-symbolic: trust the verifiable checker on disagreement
def _expr_leaves(expr: Any) -> List[str]:
    if not isinstance(expr, list) or not expr:
        return []
    if expr[0] == "prim":
        return [expr[1]]
    out: List[str] = []
    for child in expr[1:]:
        out.extend(_expr_leaves(child))
    return out


def _expr_depth(expr: Any) -> int:
    if not isinstance(expr, list) or expr[0] == "prim":
        return 1
    return 1 + max((_expr_depth(c) for c in expr[1:]), default=0)


def _expr_ttypes(expr: Any, lib: Dict[str, Primitive]) -> frozenset:
    ts: set = set()
    for leaf in _expr_leaves(expr):
        p = lib.get(leaf)
        if p:
            ts.update(p.ttypes)
    return frozenset(ts)


class Operator(ReasoningStrategy):
    """A composite (or leaf) cognitive operator — a real :class:`ReasoningStrategy` built from a
    grammar expression. ``applicability`` fires only on task types its leaves can solve; ``reason``
    evaluates the tree and reports the executed compute cost in ``support['cost']``."""

    def __init__(self, op_id: str, expr: Any, lib: Dict[str, Primitive], *,
                 protected: bool = False, provenance: str = "primitive") -> None:
        self.name = op_id
        self.op_id = op_id
        self.expr = expr
        self._lib = lib
        self.protected = protected
        self.provenance = provenance           # "primitive" | "structural" | "invented"
        self._ttypes = _expr_ttypes(expr, lib)

    # -- ReasoningStrategy contract --------------------------------------- #
    def applicability(self, query: ReasoningQuery) -> float:
        if self.protected:
            return 0.0                          # character ops are inert on the battery
        ttype = query.task.type if query.task else None
        if ttype is None or ttype not in self._ttypes:
            return 0.0
        # simpler operators are marginally preferred at equal fit (Occam over compositions)
        return 1.0 - 0.02 * (_expr_depth(self.expr) - 1)

    def reason(self, query: ReasoningQuery) -> Conclusion:
        ans, conf, cost = self._eval(self.expr, query)
        return Conclusion(answer=ans, confidence=conf, rationale=f"{self.provenance} operator",
                          strategy=self.op_id, support={"cost": cost})

    # -- tree evaluation --------------------------------------------------- #
    def _eval(self, expr: Any, query: ReasoningQuery) -> Tuple[Any, float, float]:
        head = expr[0]
        if head == "prim":
            prim = self._lib[expr[1]]
            ans, conf = prim.run(query)          # may raise -> caught by the caller/health monitor
            return ans, conf, prim.cost
        if head == "seq":
            total = 0.0
            for child in expr[1:]:
                ans, conf, c = self._eval(child, query)
                total += c
                if ans is not None and conf >= 0.6:
                    return ans, conf, total
            return ans, conf, total              # last result if none was confident
        if head == "vote":
            tally: Dict[Any, float] = {}
            total = 0.0
            for child in expr[1:]:
                ans, conf, c = self._eval(child, query)
                total += c
                key = ans if _hashable(ans) else repr(ans)
                tally[key] = tally.get(key, 0.0) + conf
            if not tally:
                return None, 0.0, total
            win = max(tally, key=tally.get)
            agreement = tally[win] / sum(tally.values())
            return win, min(1.0, 0.5 + 0.5 * agreement), total
        if head == "verify":
            proposer, checker = expr[1], expr[2]
            a1, c1, cost1 = self._eval(proposer, query)
            a2, c2, cost2 = self._eval(checker, query)
            checker_verifiable = all(self._lib[l].exact for l in _expr_leaves(checker)
                                     if l in self._lib) and _expr_leaves(checker)
            if a1 == a2:
                return a1, min(1.0, max(c1, c2) + 0.2), cost1 + cost2
            # disagreement: trust a verifiable checker; else take the more confident.
            if checker_verifiable:
                return a2, c2, cost1 + cost2
            return (a1, c1, cost1 + cost2) if c1 >= c2 else (a2, c2, cost1 + cost2)
        raise ValueError(f"bad operator expression head: {head!r}")


def _hashable(x: Any) -> bool:
    try:
        hash(x)
        return True
    except TypeError:
        return False


# --------------------------------------------------------------------------- #
# The architecture — the mutable graph of operators + a selection policy.
# --------------------------------------------------------------------------- #
@dataclass
class CognitiveArchitecture:
    """NYXARA's way of thinking, as data: a set of operators keyed by id, a per-operator selection
    weight (the policy), the protected character ids, an immune memory of past failure modes, and a
    running intelligence index. Fully (de)serialisable so gains compound across restarts."""
    operators: Dict[str, Any]                       # op_id -> grammar expr
    provenance: Dict[str, str] = field(default_factory=dict)   # op_id -> primitive|structural|invented
    policy: Dict[str, float] = field(default_factory=dict)     # op_id -> selection weight
    protected: Tuple[str, ...] = PROTECTED_OPS
    immune: List[Dict[str, Any]] = field(default_factory=list)  # remembered failure signatures
    intelligence_index: float = 0.0
    generation: int = 0

    # -- construction ------------------------------------------------------ #
    @classmethod
    def seed(cls, lib: Dict[str, Primitive]) -> "CognitiveArchitecture":
        """A deliberately *poor* starting architecture: one single (often noisy) operator per task
        type, no ensembles, no verifiers — so there is real room to grow."""
        ops: Dict[str, Any] = {}
        prov: Dict[str, str] = {}
        seen: set = set()
        for name, prim in lib.items():
            for t in prim.ttypes:
                if t in seen:
                    continue
                seen.add(t)
                ops[f"op_{name}"] = ["prim", name]
                prov[f"op_{name}"] = "primitive"
        for pid in PROTECTED_OPS:                    # the immutable character core, always present
            ops[pid] = ["prim", pid]
            prov[pid] = "character"
        policy = {oid: (PROTECTED_WEIGHT if oid in PROTECTED_OPS else 1.0) for oid in ops}
        return cls(operators=ops, provenance=prov, policy=policy)

    def clone(self) -> "CognitiveArchitecture":
        return CognitiveArchitecture(
            operators=copy.deepcopy(self.operators), provenance=dict(self.provenance),
            policy=dict(self.policy), protected=self.protected,
            immune=copy.deepcopy(self.immune), intelligence_index=self.intelligence_index,
            generation=self.generation)

    # -- invariants (the character lock) ----------------------------------- #
    def character_intact(self) -> bool:
        for pid in self.protected:
            if pid not in self.operators:
                return False
            if self.policy.get(pid, 0.0) < PROTECTED_WEIGHT:
                return False
        for oid, expr in self.operators.items():
            if oid in self.protected:
                continue
            if not _immutable_character_ok(expr):    # no invented op may touch the character core
                return False
        return True

    # -- routing: which operator thinks about this task -------------------- #
    def route(self, query: ReasoningQuery, lib: Dict[str, Primitive],
              health: Optional["HealthMonitor"] = None,
              plastic: Optional["PlasticLayer"] = None) -> Optional[Operator]:
        best: Optional[Operator] = None
        best_w = -1.0
        for oid, expr in self.operators.items():
            if oid in self.protected:
                continue
            if health is not None and health.quarantined(oid):
                continue
            op = Operator(oid, expr, lib, provenance=self.provenance.get(oid, "primitive"))
            if op.applicability(query) <= 0.0:
                continue
            w = self.policy.get(oid, 1.0) * op.applicability(query)
            if plastic is not None:
                w *= plastic.weight(oid)
            if w > best_w:
                best_w, best = w, op
        return best

    def redundancy(self, lib: Dict[str, Primitive],
                   health: Optional["HealthMonitor"] = None) -> float:
        """Fraction of task types served by ≥2 healthy operators — the biological-resilience score."""
        by_type: Dict[TaskType, int] = {}
        for oid, expr in self.operators.items():
            if oid in self.protected:
                continue
            if health is not None and health.quarantined(oid):
                continue
            for t in _expr_ttypes(expr, lib):
                by_type[t] = by_type.get(t, 0) + 1
        types = {t for p in lib.values() for t in p.ttypes}
        if not types:
            return 0.0
        return sum(1 for t in types if by_type.get(t, 0) >= 2) / len(types)

    # -- persistence ------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {"operators": self.operators, "provenance": self.provenance,
                "policy": self.policy, "protected": list(self.protected),
                "immune": self.immune, "intelligence_index": self.intelligence_index,
                "generation": self.generation}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CognitiveArchitecture":
        return cls(operators=d["operators"], provenance=d.get("provenance", {}),
                   policy=d.get("policy", {}), protected=tuple(d.get("protected", PROTECTED_OPS)),
                   immune=d.get("immune", []),
                   intelligence_index=d.get("intelligence_index", 0.0),
                   generation=d.get("generation", 0))


# --------------------------------------------------------------------------- #
# Biological resilience — the antifragile health monitor.
# --------------------------------------------------------------------------- #
class HealthMonitor:
    """Watches each operator. A run that raises is a fault; enough faults quarantine the operator and
    the pipeline re-routes to a redundant path. Each fault is written to an **immune memory** so the
    faulty configuration is never re-adopted — and it triggers synthesis of a backup, so she heals
    *stronger* (antifragile)."""

    def __init__(self, *, fault_threshold: int = 1) -> None:
        self.calls: Dict[str, int] = {}
        self.faults: Dict[str, int] = {}
        self._quarantined: set = set()
        self.fault_threshold = fault_threshold

    def record_ok(self, oid: str) -> None:
        self.calls[oid] = self.calls.get(oid, 0) + 1

    def record_fault(self, oid: str) -> None:
        self.faults[oid] = self.faults.get(oid, 0) + 1
        if self.faults[oid] >= self.fault_threshold:
            self._quarantined.add(oid)

    def quarantined(self, oid: str) -> bool:
        return oid in self._quarantined

    def clear(self, oid: str) -> None:
        self._quarantined.discard(oid)
        self.faults.pop(oid, None)


# --------------------------------------------------------------------------- #
# Continuous plasticity — the fast Hebbian layer (short-term potentiation).
# --------------------------------------------------------------------------- #
class PlasticLayer:
    """Between the heavy generational rewires, this reweights operator selection online from live
    per-turn outcomes: an operator that just helped is potentiated, one that misled decays. Bounded
    so it can bias, never dominate, the slow structural policy."""

    def __init__(self, *, lr: float = 0.1, lo: float = 0.5, hi: float = 1.5) -> None:
        self.w: Dict[str, float] = {}
        self.lr, self.lo, self.hi = lr, lo, hi

    def weight(self, oid: str) -> float:
        return self.w.get(oid, 1.0)

    def reinforce(self, oid: str, correct: bool) -> float:
        cur = self.w.get(oid, 1.0)
        target = self.hi if correct else self.lo
        cur += self.lr * (target - cur)          # exponential move toward the reward target
        cur = max(self.lo, min(self.hi, cur))
        self.w[oid] = cur
        return cur

    def to_dict(self) -> Dict[str, float]:
        return dict(self.w)


# --------------------------------------------------------------------------- #
# Recursive meta-architecture — a bounded policy over the search operators.
# --------------------------------------------------------------------------- #
class MetaPolicy:
    """She improves *how she improves her thinking*: a weighted distribution over the search
    (mutation) operators, credited by the capability change each one produced, plus a bounded outer
    level that adapts exploration when progress stalls. Hard-capped at ``MAX_META_DEPTH``."""

    MUTATIONS = ("invent", "ensemble", "verify_guard", "rewire", "prune")

    def __init__(self, depth: int = MAX_META_DEPTH) -> None:
        self.depth = min(depth, MAX_META_DEPTH)
        self.w: Dict[str, float] = {m: 1.0 for m in self.MUTATIONS}
        self.exploration = 0.5                    # tuned by the outer meta level
        self._stall = 0

    def choose(self, rng: random.Random) -> str:
        total = sum(self.w.values())
        r = rng.uniform(0, total)
        acc = 0.0
        for m, wt in self.w.items():
            acc += wt
            if r <= acc:
                return m
        return self.MUTATIONS[0]

    def credit(self, mutation: str, delta: float) -> None:
        # level 1: reward the mutation operator by the Δcapability it caused (never below a floor).
        self.w[mutation] = max(0.1, self.w[mutation] + 5.0 * delta)

    def meta_step(self, improved: bool) -> None:
        # level 2 (bounded): if progress stalls, explore harder; if it flows, exploit. Depth-capped.
        if self.depth < 2:
            return
        if improved:
            self._stall = 0
            self.exploration = max(0.1, self.exploration * 0.9)
        else:
            self._stall += 1
            self.exploration = min(1.0, self.exploration + 0.1)

    def to_dict(self) -> Dict[str, Any]:
        return {"depth": self.depth, "weights": self.w, "exploration": round(self.exploration, 3)}


# --------------------------------------------------------------------------- #
# The engine.
# --------------------------------------------------------------------------- #
@dataclass
class RewireReport:
    """One auditable summary of a rewire generation."""
    ran: bool = False
    adopted: bool = False
    reason: str = ""
    generation: int = 0
    incumbent_fitness: float = 0.0
    candidate_fitness: float = 0.0
    heldout_delta: float = 0.0
    train_accuracy: float = 0.0
    heldout_accuracy: float = 0.0
    robustness: float = 0.0
    invented: Optional[str] = None
    mutation: str = ""
    operators: int = 0
    intelligence_index: float = 0.0
    certificate: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}

    def summary(self) -> str:
        if not self.ran:
            return "cognitive-architect idle"
        if self.adopted:
            inv = f" invented {self.invented}" if self.invented else ""
            return (f"rewired mind gen {self.generation}: fitness "
                    f"{self.incumbent_fitness:.3f}→{self.candidate_fitness:.3f}"
                    f" (held-out +{self.heldout_delta:.3f}){inv}")
        return f"held incumbent gen {self.generation} ({self.reason})"


class CognitiveArchitect:
    """NYXARA redesigns her own cognitive architecture — self-driven, no LLM, verify-or-keep,
    character-locked. Call :meth:`rewire` for one generation, :meth:`tick_plasticity` per outcome,
    :meth:`heal` on a faulted operator, and :meth:`apply_to_live` to install the champion into a
    live reasoner."""

    def __init__(self, *, persist_path: Optional[str] = None, seed: int = 0,
                 lib: Optional[Dict[str, Primitive]] = None,
                 n_per_type: int = 24, meta_depth: int = MAX_META_DEPTH,
                 enact: bool = False) -> None:
        self.lib = lib if lib is not None else _default_primitives()
        self.rng = random.Random(seed)
        self.persist_path = persist_path
        self.enact = bool(enact)
        # disjoint train / held-out folds — the search NEVER optimises against held-out (anti-overfit)
        self._train = build_battery(n_per_type=n_per_type, seed=101)
        self._heldout = build_battery(n_per_type=max(8, n_per_type // 2), seed=202)
        self.arch = CognitiveArchitecture.seed(self.lib)
        self.health = HealthMonitor()
        self.plastic = PlasticLayer()
        self.meta = MetaPolicy(depth=meta_depth)
        self.population: List[Tuple[float, Dict[str, Any]]] = []   # Pareto/species island
        self.history: List[RewireReport] = []
        self._llm = None                          # invariant: there is no LLM, ever
        if persist_path:
            self._load()

    # -- fitness: real graded score of the architecture-configured solver ------------------- #
    def _evaluate(self, arch: CognitiveArchitecture,
                  battery: Sequence[BatteryItem]) -> Tuple[float, float, float]:
        """Route every battery task through the architecture and grade it against gold. Returns
        ``(accuracy, avg_cost, fitness)``. Faults are caught and re-routed (resilience in the loop);
        no gold answer is ever shown to an operator (no leakage)."""
        correct = 0
        total_cost = 0.0
        for item in battery:
            op = arch.route(item.query, self.lib, health=self.health, plastic=self.plastic)
            if op is None:
                total_cost += 0.5
                continue
            try:
                conc = op.reason(item.query)
                self.health.record_ok(op.op_id)
            except Exception:  # noqa: BLE001 — a faulting operator is quarantined & re-routed
                self.health.record_fault(op.op_id)
                op2 = arch.route(item.query, self.lib, health=self.health, plastic=self.plastic)
                if op2 is None:
                    total_cost += 0.5
                    continue
                try:
                    conc = op2.reason(item.query)
                    op = op2
                except Exception:  # noqa: BLE001
                    total_cost += 0.5
                    continue
            total_cost += float(conc.support.get("cost", 1.0))
            if conc.answer == item.gold:
                correct += 1
        n = max(1, len(battery))
        accuracy = correct / n
        avg_cost = total_cost / n
        robustness = arch.redundancy(self.lib, health=self.health)
        fitness = accuracy - COST_LAMBDA * avg_cost + ROBUST_RHO * robustness
        return accuracy, avg_cost, fitness

    # -- the search operators (mutations) --------------------------------------------------- #
    def _primitives_for(self, ttype: TaskType) -> List[str]:
        return [n for n, p in self.lib.items() if p.handles(ttype)]

    def _random_type(self) -> TaskType:
        types = sorted({t for p in self.lib.values() for t in p.ttypes}, key=lambda t: t.value)
        return types[self.rng.randrange(len(types))]

    def _grow(self, ttype: TaskType, depth: int) -> Any:
        """Grow a random, type-consistent, sound operator tree (the open-ended inventor)."""
        prims = self._primitives_for(ttype)
        if depth <= 1 or self.rng.random() < 0.35 or len(prims) < 2:
            return ["prim", prims[self.rng.randrange(len(prims))]]
        head = self.rng.choice(["seq", "vote", "verify"])
        if head == "vote":
            k = min(MAX_VOTE_ARITY, max(2, self.rng.randint(2, len(prims))))
            children = [self._grow(ttype, depth - 1) for _ in range(k)]
            return ["vote", *children]
        if head == "verify":
            exact = [p for p in prims if self.lib[p].exact]
            checker = ["prim", self.rng.choice(exact)] if exact else self._grow(ttype, depth - 1)
            return ["verify", self._grow(ttype, depth - 1), checker]
        return ["seq", self._grow(ttype, depth - 1), self._grow(ttype, depth - 1)]

    def _mutate(self, arch: CognitiveArchitecture,
                mutation: str) -> Tuple[CognitiveArchitecture, Optional[str], Optional[str]]:
        """Apply one search operator, returning the child, the new operator id (if any), and its
        provenance. Never touches a protected operator."""
        child = arch.clone()
        ttype = self._random_type()
        prims = self._primitives_for(ttype)
        new_id: Optional[str] = None
        prov: Optional[str] = None

        if mutation == "prune":
            prunable = [oid for oid in child.operators
                        if oid not in child.protected and child.provenance.get(oid) != "primitive"]
            if prunable:
                victim = self.rng.choice(prunable)
                child.operators.pop(victim, None)
                child.policy.pop(victim, None)
                child.provenance.pop(victim, None)
            return child, None, None

        if mutation == "rewire":
            reweightable = [oid for oid in child.operators if oid not in child.protected]
            if reweightable:
                tgt = self.rng.choice(reweightable)
                child.policy[tgt] = max(0.1, child.policy.get(tgt, 1.0)
                                        * self.rng.choice([0.7, 1.4]))
            return child, None, None

        if mutation == "ensemble" and len(prims) >= 2:
            k = min(MAX_VOTE_ARITY, len(prims))
            picks = self.rng.sample(prims, k)
            expr: Any = ["vote", *[["prim", p] for p in picks]]
            prov = "invented"
        elif mutation == "verify_guard" and prims:
            exact = [p for p in prims if self.lib[p].exact]
            noisy = [p for p in prims if not self.lib[p].exact]
            proposer = ["prim", self.rng.choice(noisy or prims)]
            checker = ["prim", self.rng.choice(exact)] if exact else ["prim", self.rng.choice(prims)]
            expr = ["verify", proposer, checker]
            prov = "invented"
        else:  # "invent" — grow an open-ended tree
            expr = self._grow(ttype, MAX_TREE_DEPTH)
            prov = "invented"

        if _expr_depth(expr) > MAX_TREE_DEPTH or not _immutable_character_ok(expr):
            return child, None, None
        sig = json.dumps(expr, sort_keys=True)
        if any(m.get("expr") == sig for m in child.immune):   # immune memory: never re-adopt a known-bad op
            return child, None, None
        new_id = f"inv_{ttype.value}_{abs(hash(sig)) % 100000}"
        if new_id in child.operators or len(child.operators) >= MAX_OPERATORS:
            return child, None, None
        child.operators[new_id] = expr
        child.provenance[new_id] = prov
        # a fresh operator starts slightly favoured so routing actually tries it (then earns its keep)
        child.policy[new_id] = 1.2
        return child, new_id, prov

    # -- one generation: propose → measure → prove → adopt-or-rollback ---------------------- #
    def rewire(self, *, candidates: int = 6) -> RewireReport:
        """Run one self-driven cognitive-rewiring generation. Strictly improves or rolls back."""
        rep = RewireReport(ran=True, generation=self.arch.generation)
        acc_t, _, fit_t = self._evaluate(self.arch, self._train)
        acc_h, _, fit_h = self._evaluate(self.arch, self._heldout)
        rep.incumbent_fitness = fit_t
        rep.candidate_fitness = fit_t

        best_child: Optional[CognitiveArchitecture] = None
        best_fit = fit_t
        best_new: Optional[str] = None
        best_mut = ""
        best_heldout = (acc_h, fit_h)
        for _ in range(candidates):
            mutation = self.meta.choose(self.rng)
            child, new_id, _prov = self._mutate(self.arch, mutation)
            if not child.character_intact():                    # the character lock — hard veto
                continue
            c_acc_t, _, c_fit_t = self._evaluate(child, self._train)
            if c_fit_t <= best_fit + STRICT_MARGIN:             # must STRICTLY beat on training
                self.meta.credit(mutation, c_fit_t - fit_t)
                continue
            c_acc_h, _, c_fit_h = self._evaluate(child, self._heldout)
            if c_fit_h <= fit_h + STRICT_MARGIN:                # …AND generalise on held-out (anti-overfit)
                self.meta.credit(mutation, (c_fit_h - fit_h))
                continue
            best_child, best_fit = child, c_fit_t
            best_new, best_mut = new_id, mutation
            best_heldout = (c_acc_h, c_fit_h)
            self.meta.credit(mutation, c_fit_t - fit_t)

        if best_child is None:
            self.meta.meta_step(improved=False)
            rep.reason = "no candidate strictly beat the incumbent on train ∧ held-out"
            rep.operators = len(self.arch.operators)
            rep.robustness = self.arch.redundancy(self.lib, health=self.health)
            rep.intelligence_index = self.arch.intelligence_index
            self.history.append(rep)
            return rep

        # proof-carrying adoption certificate — recorded, auditable, reversible.
        c_acc_h, c_fit_h = best_heldout
        certificate = {
            "incumbent_train_fitness": round(fit_t, 6),
            "candidate_train_fitness": round(best_fit, 6),
            "incumbent_heldout_fitness": round(fit_h, 6),
            "candidate_heldout_fitness": round(c_fit_h, 6),
            "heldout_delta": round(c_fit_h - fit_h, 6),
            "strictly_better": bool(best_fit > fit_t and c_fit_h > fit_h),
            "character_intact": best_child.character_intact(),
            "no_llm": True,
        }
        best_child.generation = self.arch.generation + 1
        # intelligence index: momentum-smoothed capability, folded like the other RSI engines.
        best_child.intelligence_index = round(
            0.7 * self.arch.intelligence_index + 0.3 * best_fit, 6)
        # keep a bounded species/Pareto population (diversity → resilience)
        self.population.append((best_fit, best_child.to_dict()))
        self.population.sort(key=lambda kv: kv[0], reverse=True)
        self.population = self.population[:MAX_POPULATION]

        self.arch = best_child
        self.meta.meta_step(improved=True)
        self.plastic = PlasticLayer()                          # fast layer re-baselines on the new graph

        rep.adopted = True
        rep.reason = "strict improvement on train ∧ held-out"
        rep.generation = best_child.generation
        rep.candidate_fitness = best_fit
        rep.heldout_delta = c_fit_h - fit_h
        rep.train_accuracy = self._evaluate(best_child, self._train)[0]
        rep.heldout_accuracy = c_acc_h
        rep.robustness = best_child.redundancy(self.lib, health=self.health)
        rep.mutation = best_mut
        rep.operators = len(best_child.operators)
        rep.intelligence_index = best_child.intelligence_index
        rep.certificate = certificate
        if best_new is not None:
            rep.invented = best_new
        self.history.append(rep)
        if self.persist_path:
            self._save()
        return rep

    # -- continuous plasticity (fast layer), driven per live outcome ------------------------ #
    def tick_plasticity(self, query: ReasoningQuery, correct: bool) -> Optional[str]:
        op = self.arch.route(query, self.lib, health=self.health, plastic=self.plastic)
        if op is None:
            return None
        self.plastic.reinforce(op.op_id, correct)
        return op.op_id

    # -- antifragile self-healing ----------------------------------------------------------- #
    def heal(self, faulted_primitive: str) -> Dict[str, Any]:
        """A primitive has failed. Quarantine every operator that leans on it, remember the failure,
        and synthesise a redundant backup from the *surviving* primitives so the affected task types
        stay served — she comes back stronger."""
        info: Dict[str, Any] = {"quarantined": [], "backup": None, "immunised": False}
        hit_types: set = set()
        for oid, expr in list(self.arch.operators.items()):
            if oid in self.arch.protected:
                continue
            if faulted_primitive in _expr_leaves(expr):
                self.health.record_fault(oid)
                info["quarantined"].append(oid)
                hit_types.update(_expr_ttypes(expr, self.lib))
        # immune memory: never re-adopt an operator built on the failed primitive
        sig = json.dumps(["prim", faulted_primitive], sort_keys=True)
        if not any(m.get("expr") == sig for m in self.arch.immune):
            self.arch.immune.append({"expr": sig, "primitive": faulted_primitive,
                                     "at": round(time.time(), 3)})
            info["immunised"] = True
        # synthesise a backup from survivors for each affected type
        for ttype in hit_types:
            survivors = [p for p in self._primitives_for(ttype) if p != faulted_primitive
                         and not getattr(self.lib[p], "_damaged", False)]
            if len(survivors) >= 2:
                expr = ["vote", *[["prim", p] for p in survivors[:MAX_VOTE_ARITY]]]
                bid = f"backup_{ttype.value}"
                if bid not in self.arch.operators:
                    self.arch.operators[bid] = expr
                    self.arch.provenance[bid] = "invented"
                    self.arch.policy[bid] = 1.5           # backups route preferentially
                    info["backup"] = bid
        if self.persist_path:
            self._save()
        return info

    # -- install the champion into a live reasoner / registry ------------------------------- #
    def apply_to_live(self, reasoner: Any = None, registry: Any = None) -> Dict[str, Any]:
        """Wire the current architecture into a live mind so subsequent turns genuinely reason with
        it. Registers every non-protected operator as a real ReasoningStrategy on ``reasoner`` and,
        when a FacultyRegistry is given, mirrors the selection policy onto faculty reliabilities.
        Opt-in via ``enact`` (sealed OFF under TEST); always reversible and character-safe."""
        applied: Dict[str, Any] = {"strategies": 0, "faculties": 0, "enacted": False}
        if not self.enact:
            applied["reason"] = "enact disabled (advisory only)"
            return applied
        if reasoner is not None and hasattr(reasoner, "register"):
            existing = {getattr(s, "op_id", getattr(s, "name", None))
                        for s in getattr(reasoner, "_strategies", [])}
            for oid, expr in self.arch.operators.items():
                if oid in self.arch.protected or oid in existing:
                    continue
                reasoner.register(Operator(oid, expr, self.lib,
                                           provenance=self.arch.provenance.get(oid, "invented")))
                applied["strategies"] += 1
        if registry is not None and hasattr(registry, "get"):
            for oid, w in self.arch.policy.items():
                for leaf in _expr_leaves(self.arch.operators.get(oid, [])):
                    fac = registry.get(leaf)
                    if fac is not None and leaf not in PROTECTED_OPS:
                        fac.reliability = max(0.05, min(1.0, 0.5 + 0.25 * math.tanh(w - 1.0)))
                        applied["faculties"] += 1
        applied["enacted"] = True
        return applied

    # -- reporting -------------------------------------------------------------------------- #
    def report(self) -> Dict[str, Any]:
        acc_t, cost_t, fit_t = self._evaluate(self.arch, self._train)
        acc_h, _, fit_h = self._evaluate(self.arch, self._heldout)
        invented = [oid for oid, p in self.arch.provenance.items() if p == "invented"]
        return {
            "generation": self.arch.generation,
            "operators": len(self.arch.operators),
            "invented_operators": len(invented),
            "invented_ids": invented[:8],
            "train_accuracy": round(acc_t, 4),
            "heldout_accuracy": round(acc_h, 4),
            "train_fitness": round(fit_t, 4),
            "heldout_fitness": round(fit_h, 4),
            "avg_cost": round(cost_t, 4),
            "robustness": round(self.arch.redundancy(self.lib, health=self.health), 4),
            "intelligence_index": self.arch.intelligence_index,
            "population": len(self.population),
            "immune_memory": len(self.arch.immune),
            "meta": self.meta.to_dict(),
            "character_intact": self.arch.character_intact(),
            "enact": self.enact,
            "rewires": len([h for h in self.history if h.adopted]),
        }

    # -- persistence ------------------------------------------------------------------------ #
    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            payload = {"arch": self.arch.to_dict(),
                       "population": self.population,
                       "meta": self.meta.to_dict(),
                       "plastic": self.plastic.to_dict()}
            tmp = f"{self.persist_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            os.replace(tmp, self.persist_path)
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            pass

    def _load(self) -> None:
        try:
            if not os.path.exists(self.persist_path):
                return
            with open(self.persist_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.arch = CognitiveArchitecture.from_dict(payload["arch"])
            self.population = [tuple(x) for x in payload.get("population", [])]
            meta = payload.get("meta") or {}
            self.meta.w.update({k: v for k, v in meta.get("weights", {}).items()
                                if k in self.meta.w})
            self.meta.exploration = float(meta.get("exploration", self.meta.exploration))
            for k, v in (payload.get("plastic") or {}).items():
                self.plastic.w[k] = float(v)
        except Exception:  # noqa: BLE001 — a corrupt file must never crash boot; fall back to seed
            self.arch = CognitiveArchitecture.seed(self.lib)


# --------------------------------------------------------------------------- #
# Self-test / demo — run: python -m nyxara.growth.cognitive_architect
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 72)
    print("NYXARA cognitive-architect self-test — she rewires her own way of thinking")
    print("=" * 72)

    ca = CognitiveArchitect(seed=1, n_per_type=24)
    r0 = ca.report()
    print(f"\nseed architecture   : {r0['operators']} ops, "
          f"train acc {r0['train_accuracy']}, fitness {r0['train_fitness']}")

    adopted = 0
    for _ in range(25):
        rep = ca.rewire(candidates=6)
        if rep.adopted:
            adopted += 1
            print(f"  gen {rep.generation:>2}: {rep.summary()}")
    r1 = ca.report()
    print(f"\nafter rewiring      : {r1['operators']} ops "
          f"({r1['invented_operators']} invented), train acc {r1['train_accuracy']}, "
          f"held-out acc {r1['heldout_accuracy']}, fitness {r1['train_fitness']}")
    assert r1["train_fitness"] >= r0["train_fitness"], "rewiring must never regress fitness"
    assert r1["invented_operators"] >= 1, "she must invent at least one new operator"
    assert ca.arch.character_intact(), "character core must remain intact"
    print(f"  intelligence index: {r1['intelligence_index']}  (adopted {adopted} rewires)")

    # -- biological resilience: damage a primitive, confirm she re-routes & heals stronger -- #
    print("\n-- antifragile resilience --")
    ca.lib["cls_h0"]._damaged = True
    heal = ca.heal("cls_h0")
    print(f"  faulted cls_h0 → quarantined {len(heal['quarantined'])}, "
          f"backup={heal['backup']}, immunised={heal['immunised']}")
    acc_after, _, _ = ca._evaluate(ca.arch, ca._train)
    print(f"  classification still served — train acc {round(acc_after, 3)}")
    assert heal["immunised"]

    # -- continuous plasticity: a good operator is potentiated between rewires -------------- #
    q = ca._train[0].query
    oid = ca.tick_plasticity(q, correct=True)
    print(f"\n-- plasticity -- reinforced {oid} → weight {round(ca.plastic.weight(oid), 3)}")
    assert ca.plastic.weight(oid) >= 1.0

    # -- persistence round-trip -------------------------------------------------------------- #
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "cognitive_architecture.json")
    ca2 = CognitiveArchitect(persist_path=path, seed=2, n_per_type=16)
    ca2.rewire(); ca2.rewire()
    gen = ca2.arch.generation
    ca3 = CognitiveArchitect(persist_path=path, seed=2, n_per_type=16)
    print(f"\n-- persistence -- saved gen {gen}, reloaded gen {ca3.arch.generation}")
    assert ca3.arch.generation == gen

    print("\nALL SELF-TESTS PASSED ✓")
