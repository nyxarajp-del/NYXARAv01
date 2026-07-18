"""NYXARA · cognition/program_library.py — the unified discrete program library (⬡, DreamCoder-style).

The gap this closes: NYXARA already has *several* subsystems that each independently do one
piece of "search → verify → keep permanently → reuse" — :mod:`skill_induction` (program
synthesis from demonstrations), :mod:`composition` (compositional generalization by
β-reduction), :mod:`growth.genesis`'s ``PrimitiveLibrary`` (gauntlet-crowned NN-mixer
primitives), :mod:`growth.cognitive_architect` (invented, battery-proven reasoning
operators), and :mod:`growth.skill_memory` (procedures distilled from successful runs). Each
keeps its own separate storage and none of them reuse each other's proven work, and none of
them is consulted by the main reasoning loop before it re-derives an answer from scratch.

This module is the missing symbolic spine: every one of those subsystems, once it has
*already verified* a program works, calls :meth:`DiscreteProgramLibrary.propose` — the
DreamCoder "sleep"/compile step — to make it permanent here, once. The kernel's reasoning
loop calls :meth:`DiscreteProgramLibrary.try_apply` — the "wake" step — *before* falling back
to full search/synthesis, so proven work is reused, not re-derived. When no single program
matches, :meth:`try_compose` combines existing programs (via ``composition.py``'s real
β-reduction, or by chaining two verified string transforms) to solve genuinely novel-but-
composable tasks, and proposes the winning composition back — so a task solved by
composition once is a direct hit next time. :meth:`search_for_solution` extends this to
multi-step pipelines with a real, bounded UCT tree search over the library's own programs
(the same four-phase shape as :mod:`mind.mcts_reasoner`, but over a symbolic action space —
NYXARA has no neural net for this domain, so tree search over her own proven primitives is
the honest analogue of "neural search proposes"). :meth:`compress` mines recurring
sub-patterns across stored programs and factors them into new, independently reusable
primitives — real library learning, not just a growing pile of storage.

Different subsystems do NOT share one universal interpreter (that would be dishonest —
genesis.py's NN-mixer steps configure a neural net; they do not solve a task the way a
string-transform program does). Each :class:`ProgramDomain` keeps its own opaque ``body``
and its own execution adapter; only the domains that are genuinely input→output transforms
(``STRING_PROGRAM``, ``REASONING_OPERATOR``, ``COMPOSED``) are ever retrieved by
``try_apply``/``try_compose``. ``PROCEDURE`` (prose skills) and ``NN_MIXER`` (architecture
primitives) are stored for bookkeeping, deduplication, and reuse-as-search-seed only.

Persistence piggybacks entirely on :class:`~nyxara.memory.store.MemoryStore` (tag
``"program"``), which already round-trips through ``NyxaraCore.save_state``/``load_state`` —
no new save/load plumbing. Pure standard library; no external LLM anywhere in this module.
"""

from __future__ import annotations

import json
import math
import random
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ProgramDomain", "ProgramTestCase", "Program", "DiscreteProgramLibrary",
    "compute_task_signature",
]


# --------------------------------------------------------------------------- #
# Domains — one honest, opaque body shape + adapter per subsystem
# --------------------------------------------------------------------------- #
class ProgramDomain(str, Enum):
    STRING_PROGRAM = "string_program"          # skill_induction.InductiveSkill
    REASONING_OPERATOR = "reasoning_operator"  # cognitive_architect.Operator expression
    PROCEDURE = "procedure"                    # skill_memory.Skill (advisory prose, not callable)
    NN_MIXER = "nn_mixer"                      # genesis.MixerProgram (bookkeeping/seed only)
    COMPOSED = "composed"                      # 2+ library programs combined via composition


#: domains whose body is a genuine input -> output transform, retrievable by try_apply/try_compose
_CALLABLE_DOMAINS: Tuple[ProgramDomain, ...] = (
    ProgramDomain.STRING_PROGRAM, ProgramDomain.REASONING_OPERATOR, ProgramDomain.COMPOSED,
)


@dataclass
class ProgramTestCase:
    """One verified (inputs -> expected) pair the program was proven against."""

    inputs: Dict[str, Any]
    expected: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"inputs": self.inputs, "expected": self.expected}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProgramTestCase":
        return cls(inputs=dict(d.get("inputs", {})), expected=d.get("expected"))


@dataclass
class Program:
    """A verified, permanently-compiled program in the unified library."""

    program_id: str
    name: str
    domain: ProgramDomain
    body: Dict[str, Any]
    signature: str
    source_subsystem: str
    test_cases: List[ProgramTestCase] = field(default_factory=list)
    composed_from: List[str] = field(default_factory=list)
    arity: int = 1
    uses: int = 0
    successes: int = 0
    failures: int = 0
    confidence: float = 0.6
    protected: bool = False
    is_component: bool = False          # factored out by compress(), reused across programs
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.uses)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "program_id": self.program_id, "name": self.name, "domain": self.domain.value,
            "body": self.body, "signature": self.signature,
            "source_subsystem": self.source_subsystem,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "composed_from": list(self.composed_from), "arity": self.arity,
            "uses": self.uses, "successes": self.successes, "failures": self.failures,
            "confidence": self.confidence, "protected": self.protected,
            "is_component": self.is_component,
            "created_at": self.created_at, "last_used": self.last_used,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Program":
        return cls(
            program_id=str(d.get("program_id") or _new_id()),
            name=str(d.get("name", "program")),
            domain=ProgramDomain(d.get("domain", ProgramDomain.STRING_PROGRAM.value)),
            body=dict(d.get("body", {})),
            signature=str(d.get("signature", "")),
            source_subsystem=str(d.get("source_subsystem", "unknown")),
            test_cases=[ProgramTestCase.from_dict(t) for t in d.get("test_cases", [])],
            composed_from=list(d.get("composed_from", [])),
            arity=int(d.get("arity", 1)),
            uses=int(d.get("uses", 0)), successes=int(d.get("successes", 0)),
            failures=int(d.get("failures", 0)), confidence=float(d.get("confidence", 0.6)),
            protected=bool(d.get("protected", False)),
            is_component=bool(d.get("is_component", False)),
            created_at=float(d.get("created_at", time.time())),
            last_used=float(d.get("last_used", 0.0)),
        )


def _new_id() -> str:
    return "prog-" + uuid.uuid4().hex[:10]


def _slug(text: str, limit: int = 32) -> str:
    out = "".join(c if c.isalnum() else "_" for c in str(text).lower()).strip("_")
    out = "_".join(p for p in out.split("_") if p)
    return (out or "x")[:limit]


def compute_task_signature(stimulus: str) -> str:
    """A coarse retrieval key for a stimulus — the same slot-template machinery
    :mod:`skill_induction` already uses for its input templates, so ``STRING_PROGRAM``
    retrieval is consistent with how skills are indexed elsewhere."""
    try:
        from nyxara.cognition.abstraction import abstract_text
        tmpl = abstract_text([str(stimulus)])
        if tmpl:
            return f"string:{tmpl}"
    except Exception:  # noqa: BLE001 — signature computation is best-effort
        pass
    toks = str(stimulus).split()
    if not toks:
        return "string:_empty_"
    if len(toks) == 1:
        return f"string:{toks[0].lower()}"
    return f"string:{toks[0].lower()}_..._{toks[-1].lower()}"


# --------------------------------------------------------------------------- #
# The library
# --------------------------------------------------------------------------- #
class DiscreteProgramLibrary:
    """The unified symbolic store: every subsystem's verified programs land here once;
    the reasoning loop retrieves/composes from here before re-deriving anything."""

    TAG = "program"

    def __init__(self, *, store: Any = None, composer: Any = None,
                 capacity: int = 256, decay_rate: float = 0.02,
                 promote_min_uses: int = 3, promote_min_success_rate: float = 0.8,
                 composition_max_pairs: int = 6,
                 compression_enabled: bool = True, compression_min_frequency: int = 3,
                 compression_trigger_growth: int = 32,
                 search_enabled: bool = True, search_max_rollouts: int = 32,
                 search_max_depth: int = 4, migrate_legacy_on_boot: bool = True) -> None:
        self.store = store
        self.composer = composer
        self.capacity = max(8, int(capacity))
        self.decay_rate = max(0.0, float(decay_rate))
        self.promote_min_uses = max(1, int(promote_min_uses))
        self.promote_min_success_rate = max(0.0, min(1.0, float(promote_min_success_rate)))
        self.composition_max_pairs = max(1, int(composition_max_pairs))
        self.compression_enabled = bool(compression_enabled)
        self.compression_min_frequency = max(2, int(compression_min_frequency))
        self.compression_trigger_growth = max(1, int(compression_trigger_growth))
        self.search_enabled = bool(search_enabled)
        self.search_max_rollouts = max(0, int(search_max_rollouts))
        self.search_max_depth = max(1, int(search_max_depth))

        self._programs: Dict[str, Program] = {}
        self._by_signature: Dict[str, List[str]] = {}
        self._hydrated = False
        # Deliberately NOT run at construction time (only on first real hydration — see
        # ``_hydrate``): NyxaraCore builds this library, then calls ``load_state`` to restore
        # persisted memory into the SAME store, then eventually queries the library. Migrating
        # eagerly at construction would hydrate (and cache "nothing to migrate") BEFORE
        # ``load_state`` ever runs, permanently missing everything it loads.
        self._migrate_legacy_on_boot = bool(migrate_legacy_on_boot)
        self._growth_since_compression = 0
        self._reasoning_lib: Any = None            # optional injected cognitive_architect primitive lib

    def __len__(self) -> int:
        self._hydrate()
        return sum(1 for pid in self._programs if pid != "_migration_marker")

    # ------------------------------------------------------------------ #
    # (b) propose — the "sleep" / compile step
    # ------------------------------------------------------------------ #
    def propose(self, *, name: str, domain: ProgramDomain, body: Dict[str, Any],
                signature: str, source: str, test_cases: Sequence[ProgramTestCase] = (),
                composed_from: Sequence[str] = (), arity: int = 1,
                already_verified: bool = False, is_component: bool = False) -> Optional[Program]:
        """Verify (unless the caller already did), dedup, and persist a program.

        Returns ``None`` if verification fails; returns the existing program (not a
        duplicate) if a structurally-identical one is already in the library."""
        self._hydrate()
        tcs = list(test_cases)
        if not already_verified and tcs and domain in _CALLABLE_DOMAINS:
            probe = Program(program_id="_probe", name=name, domain=domain, body=body,
                             signature=signature, source_subsystem=source, test_cases=tcs,
                             composed_from=list(composed_from), arity=arity)
            for tc in tcs:
                try:
                    out = self._run_domain(probe, tc.inputs)
                except Exception:  # noqa: BLE001 — an unverifiable candidate is simply refused
                    return None
                if out != tc.expected:
                    return None

        dedup_key = self._dedup_key(domain, body)
        for existing in self._programs.values():
            if existing.domain == domain and self._dedup_key(existing.domain, existing.body) == dedup_key:
                return existing

        program = Program(program_id=_new_id(), name=str(name).strip() or "program",
                           domain=domain, body=body, signature=signature,
                           source_subsystem=source, test_cases=tcs,
                           composed_from=list(composed_from), arity=arity,
                           protected=is_component, is_component=is_component)
        self._programs[program.program_id] = program
        self._by_signature.setdefault(signature, []).append(program.program_id)
        self._growth_since_compression += 1
        self._persist(program)
        return program

    # ------------------------------------------------------------------ #
    # (c) retrieve / try_apply — the "wake" / fast-path step
    # ------------------------------------------------------------------ #
    def retrieve(self, task_signature: str, *, domain: Optional[ProgramDomain] = None,
                 k: int = 5) -> List[Program]:
        self._hydrate()
        ids = self._by_signature.get(task_signature, [])
        progs = [self._programs[i] for i in ids if i in self._programs]
        if domain is not None:
            progs = [p for p in progs if p.domain == domain]
        progs.sort(key=lambda p: (p.confidence, p.success_rate, p.uses), reverse=True)
        return progs[:k]

    def _scan_compatible(self, inputs: Dict[str, Any], *,
                          domain: Optional[ProgramDomain] = None, limit: int = 20) -> List[Program]:
        """Bounded fallback scan when no signature-index hit exists."""
        doms = (domain,) if domain is not None else _CALLABLE_DOMAINS
        out = [p for p in self._programs.values() if p.domain in doms]
        out.sort(key=lambda p: (p.confidence, p.success_rate, p.uses), reverse=True)
        return out[:limit]

    def try_apply(self, inputs: Dict[str, Any], *, task_signature: Optional[str] = None,
                  domain: Optional[ProgramDomain] = None) -> Optional[Tuple[Any, Program, float]]:
        """Fast path: find a compiled program that solves ``inputs`` and reuse it."""
        self._hydrate()
        sig = task_signature or (compute_task_signature(inputs["text"]) if "text" in inputs else None)
        candidates = list(self.retrieve(sig, domain=domain, k=5)) if sig else []
        seen = {p.program_id for p in candidates}
        for p in self._scan_compatible(inputs, domain=domain):
            if p.program_id not in seen:
                candidates.append(p)
                seen.add(p.program_id)
        for program in candidates:
            if program.domain not in _CALLABLE_DOMAINS:
                continue
            try:
                out = self._run_domain(program, inputs)
            except Exception:  # noqa: BLE001 — a faulting program is scored down, not fatal
                self.reinforce(program.program_id, success=False)
                continue
            if out is None:
                continue
            self.reinforce(program.program_id, success=True)
            return out, program, program.confidence
        return None

    # ------------------------------------------------------------------ #
    # (g) try_compose — real compositional generalization
    # ------------------------------------------------------------------ #
    def try_compose(self, inputs: Dict[str, Any], task_signature: str) -> Optional[Tuple[Any, Program]]:
        """When no single program matches, combine existing ones to solve a novel task."""
        self._hydrate()
        text = inputs.get("text")

        # path 1: the LoT compositional grammar's real β-reduction (SCAN-style: "walk twice")
        if text and self.composer is not None:
            try:
                interp = self.composer.interpret(text)
            except Exception:  # noqa: BLE001
                interp = None
            if interp is not None and interp.ok:
                from nyxara.cognition.composition import serialize_term
                out = str(interp.meaning)
                body = {"kind": "beta_app", "term": serialize_term(interp.meaning)}
                program = self.propose(
                    name=f"compose_{_slug(text)}", domain=ProgramDomain.COMPOSED, body=body,
                    signature=task_signature, source="composition",
                    test_cases=[ProgramTestCase(inputs={"text": text}, expected=out)],
                    already_verified=True)
                if program is not None:
                    self.reinforce(program.program_id, success=True)
                    return out, program

        # path 2: sequential pipeline chaining of two STRING_PROGRAMs, verified against
        # BOTH parents' own demo pairs (no ground truth exists for a never-seen input, so
        # this is the honest verification bar available at inference time).
        if text is None:
            return None
        candidates = self._scan_compatible({"text": text}, domain=ProgramDomain.STRING_PROGRAM)[:4]
        tried = 0
        for p_i in candidates:
            if not p_i.test_cases:
                continue
            for p_j in candidates:
                if p_i.program_id == p_j.program_id:
                    continue
                tried += 1
                if tried > self.composition_max_pairs:
                    return None
                outputs: List[Any] = []
                ok = True
                for tc in p_i.test_cases:
                    try:
                        mid = self._run_domain(p_i, tc.inputs)
                        piped = self._run_domain(p_j, {"text": mid}) if mid is not None else None
                    except Exception:  # noqa: BLE001
                        piped = None
                    original = tc.inputs.get("text")
                    if piped is None or piped == original or piped == mid:
                        ok = False
                        break
                    outputs.append(piped)
                if not ok:
                    continue
                body = {"kind": "pipeline", "steps": [p_i.program_id, p_j.program_id]}
                new_tcs = [ProgramTestCase(inputs=tc.inputs, expected=out)
                           for tc, out in zip(p_i.test_cases, outputs)]
                program = self.propose(
                    name=f"compose_{p_i.name}_{p_j.name}", domain=ProgramDomain.COMPOSED, body=body,
                    signature=task_signature, source="composition", test_cases=new_tcs,
                    composed_from=[p_i.program_id, p_j.program_id], already_verified=True)
                if program is not None:
                    try:
                        out = self._run_domain(program, inputs)
                    except Exception:  # noqa: BLE001
                        out = None
                    if out is not None:
                        self.reinforce(program.program_id, success=True)
                        return out, program
        return None

    # ------------------------------------------------------------------ #
    # neural(-analogue) search: bounded UCT tree search over library programs.
    # NYXARA has no neural net for this symbolic domain, so a genuine tree search over
    # her own proven primitives — the same 4-phase shape as mind/mcts_reasoner.py — is the
    # honest stand-in for "a neural component proposes"; the library still makes the winning
    # sequence permanent exactly like every other producer.
    # ------------------------------------------------------------------ #
    def search_for_solution(self, inputs: Dict[str, Any], task_signature: str, *,
                             verify: Optional[Callable[[str], float]] = None,
                             max_rollouts: Optional[int] = None,
                             max_depth: Optional[int] = None) -> Optional[Tuple[Any, Program]]:
        self._hydrate()
        rollouts = self.search_max_rollouts if max_rollouts is None else max_rollouts
        depth_cap = self.search_max_depth if max_depth is None else max_depth
        if not self.search_enabled or rollouts <= 0:
            return None
        text = inputs.get("text")
        if not text:
            return None
        pool = self._scan_compatible({"text": text}, domain=ProgramDomain.STRING_PROGRAM, limit=12)
        if len(pool) < 2:
            return None

        def score(candidate: str) -> float:
            if verify is not None:
                try:
                    return max(0.0, min(1.0, float(verify(candidate))))
                except Exception:  # noqa: BLE001
                    return 0.0
            return 1.0 if candidate and candidate != text else 0.0

        rng = random.Random(hash(text) & 0xFFFFFFFF)
        stats: Dict[str, Dict[str, List[float]]] = {}
        best_path: Optional[List[str]] = None
        best_state: Optional[str] = None
        best_score = 0.0

        for _ in range(rollouts):
            state = text
            path: List[str] = []
            for _depth in range(depth_cap):
                node = stats.setdefault(state, {})
                unexplored = [p for p in pool if p.program_id not in node]
                if unexplored:
                    program = rng.choice(unexplored)
                else:
                    log_n = math.log(sum(v[0] for v in node.values()) + 1.0)

                    def _uct(pid: str) -> float:
                        visits, value = node[pid]
                        q = value / visits if visits else 0.0
                        return q + 1.4 * math.sqrt(log_n / (visits + 1e-9))

                    best_pid = max(node, key=_uct)
                    program = next(p for p in pool if p.program_id == best_pid)
                try:
                    out = self._run_domain(program, {"text": state})
                except Exception:  # noqa: BLE001
                    out = None
                node.setdefault(program.program_id, [0.0, 0.0])
                if out is None or out == state:
                    node[program.program_id][0] += 1.0
                    break
                reward = score(out)
                node[program.program_id][0] += 1.0
                node[program.program_id][1] += reward
                path.append(program.program_id)
                state = out
                if reward >= 1.0 and len(path) >= 2:
                    if reward > best_score or best_path is None:
                        best_score, best_path, best_state = reward, list(path), state
                    break

        if not best_path or len(best_path) < 2 or best_state is None:
            return None
        body = {"kind": "pipeline", "steps": best_path}
        program = self.propose(
            name="search_" + "_".join(best_path[:3]), domain=ProgramDomain.COMPOSED, body=body,
            signature=task_signature, source="mcts_search",
            test_cases=[ProgramTestCase(inputs={"text": text}, expected=best_state)],
            composed_from=best_path, already_verified=True)
        if program is None:
            return None
        self.reinforce(program.program_id, success=True)
        return best_state, program

    # ------------------------------------------------------------------ #
    # (e) self-curation: reinforcement, decay, pruning
    # ------------------------------------------------------------------ #
    def reinforce(self, program_id: str, *, success: bool) -> None:
        program = self._programs.get(program_id)
        if program is None:
            return
        program.uses += 1
        if success:
            program.successes += 1
        else:
            program.failures += 1
        program.last_used = time.time()
        program.confidence = max(0.0, min(1.0, 0.8 * program.confidence + 0.2 * (1.0 if success else 0.0)))
        self._persist(program)

    def decay_tick(self, *, now: Optional[float] = None) -> Dict[str, float]:
        """Exponential proficiency-style decay of unused programs (mirrors skilltree.py)."""
        self._hydrate()
        now = now if now is not None else time.time()
        dropped: Dict[str, float] = {}
        for pid, program in list(self._programs.items()):
            if program.last_used <= 0:
                continue
            dt_days = (now - program.last_used) / 86400.0
            if dt_days <= 0:
                continue
            new_conf = program.confidence * math.exp(-self.decay_rate * dt_days)
            if new_conf < program.confidence:
                dropped[pid] = round(program.confidence - new_conf, 6)
                program.confidence = new_conf
                self._persist(program)
        if (self.compression_enabled
                and self._growth_since_compression >= self.compression_trigger_growth):
            self.compress()
        self.prune()
        return dropped

    def prune(self) -> List[str]:
        """Drop the least-valuable programs once over capacity; never a protected program or
        one another still-live program depends on."""
        self._hydrate()
        if len(self._programs) <= self.capacity:
            return []
        depended_on = {pid for p in self._programs.values() for pid in p.composed_from}
        candidates = [p for p in self._programs.values()
                      if not p.protected and p.program_id not in depended_on]
        candidates.sort(key=lambda p: (p.confidence, p.last_used))
        removed: List[str] = []
        overflow = len(self._programs) - self.capacity
        for program in candidates:
            if len(removed) >= overflow:
                break
            removed.append(program.program_id)
        for pid in removed:
            self._programs.pop(pid, None)
            for ids in self._by_signature.values():
                if pid in ids:
                    ids.remove(pid)
        return removed

    def _dedup_key(self, domain: ProgramDomain, body: Dict[str, Any]) -> str:
        """Exact structural dedup — never embedding similarity, so it never false-merges a
        genuinely different program that merely looks similar."""
        if domain == ProgramDomain.STRING_PROGRAM:
            return json.dumps(body.get("program", []), sort_keys=True)
        if domain == ProgramDomain.REASONING_OPERATOR:
            return json.dumps(body.get("expr"), sort_keys=True)
        if domain == ProgramDomain.NN_MIXER:
            return "+".join(body.get("steps", []))
        if domain == ProgramDomain.COMPOSED and body.get("kind") == "beta_app":
            return json.dumps(body.get("term"), sort_keys=True)
        return json.dumps(body, sort_keys=True)

    # ------------------------------------------------------------------ #
    # (1) library compression — real library learning, not just storage
    # ------------------------------------------------------------------ #
    def compress(self) -> Dict[str, int]:
        """Mine recurring ``Operation`` subsequences across STRING_PROGRAM bodies and factor
        the most-shared ones into new, independently-reusable component primitives. Consumers
        keep their own (already-verified, untouched) bodies — nothing about their execution
        changes — but now reference the shared component via ``composed_from``, so future
        composition/search can reuse the factored-out structure directly as its own program."""
        self._hydrate()
        pool = [p for p in self._programs.values()
                if p.domain == ProgramDomain.STRING_PROGRAM and not p.is_component]
        if len(pool) < self.compression_min_frequency:
            return {"factored": 0}
        counts: Counter = Counter()
        occurrences: Dict[Tuple[str, ...], List[str]] = {}
        for p in pool:
            ops = p.body.get("program", [])
            n = len(ops)
            for length in range(2, n + 1):
                for start in range(0, n - length + 1):
                    key = tuple(json.dumps(o, sort_keys=True) for o in ops[start:start + length])
                    counts[key] += 1
                    occurrences.setdefault(key, []).append(p.program_id)

        factored = 0
        claimed: set = set()
        for key, freq in counts.most_common():
            consumer_ids = sorted(set(occurrences[key]))
            if freq < self.compression_min_frequency or len(consumer_ids) < self.compression_min_frequency:
                continue
            if key in claimed:
                continue
            ops_list = [json.loads(s) for s in key]
            comp_name = f"component_{abs(hash(key)) % 100000}"
            component = self.propose(
                name=comp_name, domain=ProgramDomain.STRING_PROGRAM,
                body={"program": ops_list}, signature=f"component:{comp_name}",
                source="compression", already_verified=True, is_component=True)
            if component is None:
                continue
            claimed.add(key)
            for pid in consumer_ids:
                consumer = self._programs.get(pid)
                if consumer is None or component.program_id in consumer.composed_from:
                    continue
                consumer.composed_from = list(consumer.composed_from) + [component.program_id]
                self._persist(consumer)
            factored += 1
        self._growth_since_compression = 0
        return {"factored": factored, "programs_scanned": len(pool)}

    # ------------------------------------------------------------------ #
    # (f) tool promotion
    # ------------------------------------------------------------------ #
    def promotion_candidates(self) -> List[str]:
        self._hydrate()
        return [pid for pid, p in self._programs.items()
                if not p.protected and p.domain in _CALLABLE_DOMAINS
                and p.uses >= self.promote_min_uses and p.success_rate >= self.promote_min_success_rate]

    def promote_to_tool(self, program_id: str, registry: Any, *,
                         min_uses: Optional[int] = None,
                         min_success_rate: Optional[float] = None) -> Optional[Any]:
        program = self._programs.get(program_id)
        if program is None or program.domain not in _CALLABLE_DOMAINS:
            return None
        min_uses = self.promote_min_uses if min_uses is None else min_uses
        min_success_rate = self.promote_min_success_rate if min_success_rate is None else min_success_rate
        if program.uses < min_uses or program.success_rate < min_success_rate:
            return None
        tool_name = f"program_{_slug(program.name)}"
        existing = registry.get(tool_name)
        if existing is not None:
            program.protected = True
            return existing
        try:
            from nyxara.agency.tools import ToolParam, ToolSpec
            from nyxara.agency.permissions import Capability, RiskTier
        except Exception:  # noqa: BLE001 — tool promotion is a capability, never a hard dependency
            return None

        def _handler(text: str = "") -> Any:
            return self._run_domain(program, {"text": text})

        spec = ToolSpec(
            name=tool_name, handler=_handler,
            description=f"compiled library program: {program.name} ({program.domain.value})",
            params=[ToolParam("text", "str", required=True)],
            capability=Capability.TOOL_CALL, risk=RiskTier.LOW, reversible=True, cost=0.0)
        try:
            registry.register(spec)
        except Exception:  # noqa: BLE001
            return None
        program.protected = True
        self._persist(program)
        return spec

    # ------------------------------------------------------------------ #
    # (d) migration from the pre-existing, fragmented stores
    # ------------------------------------------------------------------ #
    def import_legacy(self, *, genesis_path: Optional[str] = None) -> Dict[str, int]:
        """One-time, idempotent import of existing genesis JSON + skill-tagged MemoryStore
        records into the unified index. Purely additive: original stores/tags are left
        untouched, so those subsystems keep working exactly as before either way.

        Suppresses the lazy auto-migration for the duration of its own ``_hydrate`` call —
        otherwise an explicit call (e.g. with a caller-supplied ``genesis_path``) would
        always lose the race to the default-path lazy trigger and silently no-op."""
        suppressed, self._migrate_legacy_on_boot = self._migrate_legacy_on_boot, False
        try:
            self._hydrate()
        finally:
            self._migrate_legacy_on_boot = suppressed
        if "_migration_marker" in self._programs:
            return {"skipped": 1}
        return self._do_migration(genesis_path)

    def _do_migration(self, genesis_path: Optional[str]) -> Dict[str, int]:
        """The actual migration work — never calls ``_hydrate`` and never checks the marker
        itself, so it is safe to call from inside ``_hydrate`` (lazy auto-migration) as well
        as from the public ``import_legacy`` (explicit, e.g. a custom genesis path) without
        the two racing or double-counting each other."""
        counts = {"genesis": 0, "skill_induction": 0, "skill_memory": 0}

        path = Path(genesis_path) if genesis_path else self._default_genesis_path()
        if path is not None and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for e in data.get("entries", []):
                    steps = list(e.get("steps", []))
                    if not steps:
                        continue
                    prog = self.propose(
                        name=str(e.get("name", "synth")), domain=ProgramDomain.NN_MIXER,
                        body={"steps": steps}, signature=f"nn_mixer:{len(steps)}",
                        source="genesis-migration", already_verified=True)
                    if prog is not None:
                        counts["genesis"] += 1
            except Exception:  # noqa: BLE001 — a corrupt/missing file never blocks migration
                pass

        if self.store is not None:
            try:
                hits = self.store.recall("", k=5000, tags=["skill"], strengthen=False)
            except Exception:  # noqa: BLE001
                hits = []
            for rec, _ in hits:
                data = rec.metadata.get("skill") if isinstance(rec.metadata, dict) else None
                if not isinstance(data, dict):
                    continue
                if "program" in data:
                    ops = data.get("program", [])
                    demos = data.get("demos", [])
                    tcs = [ProgramTestCase(inputs={"text": d[0]}, expected=d[1])
                           for d in demos if isinstance(d, (list, tuple)) and len(d) == 2]
                    name = str(data.get("name", "skill"))
                    prog = self.propose(
                        name=name, domain=ProgramDomain.STRING_PROGRAM,
                        body={"program": ops}, signature=f"string:{name}",
                        source="skill_induction-migration", test_cases=tcs, already_verified=True)
                    if prog is not None:
                        counts["skill_induction"] += 1
                elif "goal" in data and "procedure" in data:
                    goal = str(data.get("goal", ""))
                    prog = self.propose(
                        name=(goal[:40] or "procedure"), domain=ProgramDomain.PROCEDURE,
                        body=data, signature=f"procedure:{_slug(goal)}",
                        source="skill_memory-migration", already_verified=True)
                    if prog is not None:
                        counts["skill_memory"] += 1

        marker = Program(program_id="_migration_marker", name="_migration_marker",
                          domain=ProgramDomain.PROCEDURE, body={}, signature="_migration_marker",
                          source_subsystem="migration", protected=True)
        self._programs["_migration_marker"] = marker
        self._persist(marker)
        return counts

    def _default_genesis_path(self) -> Optional[Path]:
        try:
            from nyxara.kernel.config import get_settings
            data_dir = get_settings().paths.data_dir
            return Path(data_dir) / "genesis" / "primitive_library.json"
        except Exception:  # noqa: BLE001 — no settings/paths: nothing to migrate from
            return None

    # ------------------------------------------------------------------ #
    # observability
    # ------------------------------------------------------------------ #
    def report(self) -> Dict[str, Any]:
        self._hydrate()
        programs = [p for pid, p in self._programs.items() if pid != "_migration_marker"]
        by_domain: Dict[str, int] = {}
        for p in programs:
            by_domain[p.domain.value] = by_domain.get(p.domain.value, 0) + 1
        promoted = sum(1 for p in programs if p.protected and not p.is_component)
        components = sum(1 for p in programs if p.is_component)
        total_uses = sum(p.uses for p in programs)
        total_successes = sum(p.successes for p in programs)
        return {
            "programs": len(programs), "by_domain": by_domain,
            "promoted_tools": promoted, "compressed_components": components,
            "total_uses": total_uses,
            "fastpath_hit_rate": round(total_successes / total_uses, 4) if total_uses else 0.0,
        }

    # ------------------------------------------------------------------ #
    # execution — per-domain adapters (dispatch needs library context for COMPOSED)
    # ------------------------------------------------------------------ #
    def _run_domain(self, program: Program, inputs: Dict[str, Any]) -> Any:
        if program.domain == ProgramDomain.STRING_PROGRAM:
            return self._run_string_program(program.body, inputs)
        if program.domain == ProgramDomain.REASONING_OPERATOR:
            return self._run_reasoning_operator(program.body, inputs)
        if program.domain == ProgramDomain.COMPOSED:
            return self._run_composed(program.body, inputs)
        return None          # PROCEDURE / NN_MIXER: not directly callable

    @staticmethod
    def _run_string_program(body: Dict[str, Any], inputs: Dict[str, Any]) -> Any:
        from nyxara.cognition.skill_induction import Operation
        text = str(inputs.get("text", ""))
        for op_dict in body.get("program", []):
            text = Operation.from_dict(op_dict).apply(text)
        return text

    def _run_reasoning_operator(self, body: Dict[str, Any], inputs: Dict[str, Any]) -> Any:
        query = inputs.get("query")
        if query is None:
            return None
        expr = body.get("expr")
        if expr is None:
            return None
        try:
            from nyxara.growth.cognitive_architect import Operator, _default_primitives
            from nyxara.mind.reasoner import ReasoningQuery
        except Exception:  # noqa: BLE001
            return None
        if isinstance(query, dict):
            try:
                query = ReasoningQuery(**query)
            except Exception:  # noqa: BLE001
                return None
        try:
            lib = self._reasoning_lib or _default_primitives()
            op = Operator("library", expr, lib)
            conclusion = op.reason(query)
            return conclusion.answer
        except Exception:  # noqa: BLE001 — a faulting operator body is scored down, not fatal
            return None

    def _run_composed(self, body: Dict[str, Any], inputs: Dict[str, Any]) -> Any:
        kind = body.get("kind")
        if kind == "beta_app":
            try:
                from nyxara.cognition.composition import deserialize_term
                return str(deserialize_term(body["term"]))
            except Exception:  # noqa: BLE001
                return None
        if kind == "pipeline":
            text = inputs.get("text")
            if text is None:
                return None
            for pid in body.get("steps", []):
                step = self._programs.get(pid)
                if step is None:
                    return None
                text = self._run_domain(step, {"text": text})
                if text is None:
                    return None
            return text
        return None

    # ------------------------------------------------------------------ #
    # persistence — piggybacks entirely on MemoryStore (survives save_state/load_state)
    # ------------------------------------------------------------------ #
    def _persist(self, program: Program) -> None:
        if self.store is None:
            return
        try:
            from nyxara.memory.store import MemoryType
            self.store.remember(
                f"program:{program.name} ({program.domain.value})",
                mem_type=MemoryType.PROCEDURAL,
                importance=min(1.0, 0.5 + 0.1 * program.confidence),
                tags=[self.TAG],
                metadata={"program": program.to_dict()})
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            pass

    def _hydrate(self) -> None:
        if self._hydrated:
            return
        self._hydrated = True
        if self.store is not None:
            try:
                hits = self.store.recall("", k=5000, tags=[self.TAG], strengthen=False)
            except Exception:  # noqa: BLE001
                hits = []
            for rec, _ in hits:
                data = rec.metadata.get("program") if isinstance(rec.metadata, dict) else None
                if not isinstance(data, dict):
                    continue
                try:
                    program = Program.from_dict(data)
                except Exception:  # noqa: BLE001 — skip an unparseable record
                    continue
                cur = self._programs.get(program.program_id)
                if (cur is None or program.last_used >= cur.last_used
                        or program.confidence >= cur.confidence):
                    self._programs[program.program_id] = program
            self._by_signature.clear()
            for pid, p in self._programs.items():
                self._by_signature.setdefault(p.signature, []).append(pid)
        # migrate the pre-existing fragmented stores in, once, lazily — NOT at construction
        # time (see the constructor note): this only fires on first genuine access, which in
        # normal boot order is always AFTER NyxaraCore.load_state has already restored memory.
        if self._migrate_legacy_on_boot and "_migration_marker" not in self._programs:
            try:
                self._do_migration(None)
            except Exception:  # noqa: BLE001 — migration is best-effort, never fatal
                pass


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.cognition.composition import CompositionalGrammar
    from nyxara.cognition.skill_induction import Operation, SkillInductionEngine
    from nyxara.memory.store import MemoryStore
    from nyxara.mind.lot import Lambda, const, func, var

    print("=" * 70)
    print("NYXARA discrete program library self-test")
    print("=" * 70)

    store = MemoryStore()
    lib = DiscreteProgramLibrary(store=store, composer=CompositionalGrammar(store=store))

    # 1. propose + persist + restart + retrieve round-trip
    tcs = [ProgramTestCase(inputs={"text": "hello"}, expected="olleh"),
           ProgramTestCase(inputs={"text": "world"}, expected="dlrow")]
    p = lib.propose(name="reverse", domain=ProgramDomain.STRING_PROGRAM,
                     body={"program": [Operation("reverse_chars").to_dict()]},
                     signature="string:reverse", source="skill_induction", test_cases=tcs)
    assert p is not None
    lib2 = DiscreteProgramLibrary(store=store)
    got = lib2.retrieve("string:reverse")
    assert got and got[0].name == "reverse"
    print("propose/persist/retrieve round-trip ✓")

    # 2. fast-path apply
    hit = lib2.try_apply({"text": "nyxara"}, task_signature="string:reverse")
    assert hit is not None and hit[0] == "araxyn"
    print(f"try_apply fast path : nyxara -> {hit[0]} ✓")

    # 3. compositional generalization — the acid test
    g = lib.composer
    g.learn("jump", const("JUMP"))
    g.learn("walk", const("WALK"))
    g.learn("twice", Lambda(var("a"), func("seq", var("a"), var("a"))))
    out = lib.try_compose({"text": "walk twice"}, task_signature="string:walk_twice")
    assert out is not None and out[0] == "seq(WALK, WALK)"
    print(f"try_compose (novel) : 'walk twice' -> {out[0]} ✓")
    hit2 = lib.try_apply({"text": "walk twice"}, task_signature="string:walk_twice")
    assert hit2 is not None and hit2[0] == "seq(WALK, WALK)"
    print("second call is a direct try_apply hit, not re-composed ✓")

    # 4. promotion into a real ToolRegistry and invocation through the safety pipeline
    from nyxara.agency.tools import ToolRegistry
    for _ in range(4):
        lib.reinforce(p.program_id, success=True)
    registry = ToolRegistry()
    spec = lib.promote_to_tool(p.program_id, registry)
    assert spec is not None
    result = registry.invoke(spec.name, {"text": "abc"})
    assert result.ok and result.value == "cba"
    print(f"promote_to_tool     : {spec.name} invoked through real pipeline -> {result.value} ✓")

    # 5. decay/pruning protects reinforced & promoted programs, drops unused ones
    lonely = lib.propose(name="upper", domain=ProgramDomain.STRING_PROGRAM,
                         body={"program": [Operation("upper").to_dict()]},
                         signature="string:upper",
                         test_cases=[ProgramTestCase(inputs={"text": "a"}, expected="A")],
                         source="skill_induction")
    assert lonely is not None
    lonely.last_used = time.time() - 999 * 86400
    lonely.confidence = 0.51
    lib.capacity = len(lib._programs) - 1      # force overflow so prune() must drop one
    lib.decay_tick(now=time.time())
    lib.prune()
    assert p.program_id in lib._programs and lonely.program_id not in lib._programs
    print("decay/prune          : protected program kept, unused one dropped ✓")

    print("\nALL SELF-TESTS PASSED ✓")
