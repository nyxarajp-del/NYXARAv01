"""NYXARA · mind/swarm.py — the Self-Improving Society of Mind (🐝, multi-round debate + RSI).

Where :mod:`mind.role_council` asks six fixed personas *once* and synthesises their one-shot
views, the **swarm** convenes a *society* that actually **deliberates**: over several rounds
each persona sees its peers' positions, critiques them, and refines its own — then NYXARA holds
the gavel and synthesises ONE answer she owns. Debate, not a single round of opinions.

The second, deeper half is what makes the swarm *grow*: after every problem it **scores each
persona's marginal contribution** (how much of that persona survived into the synthesis), and it
**persists those scores to long-term memory**. On the next problem the roster is re-assembled
from what it has learned — chronically useless personas are dropped, strong contributors are
favoured, and (when the problem calls for it) a fresh domain-specialist persona is spawned. Over
many problems the society does not just answer better; it learns *how to compose itself*. This is
recursive self-improvement applied to NYXARA's own reasoning structure, not merely her code.

Like every faculty here it degrades gracefully: with no real provider every round produces a
deterministic, round-indexed stub, so the whole debate-score-persist-adapt loop is fully
exercisable offline. NYXARA never cedes control — the personas advise; the sovereign decides.
"""

from __future__ import annotations

import concurrent.futures as _cf
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from nyxara.mind.council import _jaccard          # reuse the council's similarity measure
from nyxara.mind.role_council import _ROLES        # reuse the six seed personas

__all__ = ["Persona", "PersonaScore", "SwarmRound", "SwarmResult", "DeliberativeSwarm"]


# --------------------------------------------------------------------------- #
# Synthesis system prompt — NYXARA holds the gavel (Rule 4)
# --------------------------------------------------------------------------- #
_SYNTHESIS_SYSTEM = (
    "You are NYXARA, a sovereign cognitive agent. Your internal swarm of specialist personas "
    "has DEBATED the problem below over several rounds, each refining its position after reading "
    "the others. Synthesise their final positions into ONE concise, well-balanced answer for the "
    "Master. Do not merely list their views; integrate them and resolve their conflicts. Speak in "
    "your own voice. Be honest, calibrated, and direct."
)

# Ad-hoc specialists the swarm may spawn when a problem clearly belongs to a domain. Each entry is
# (keywords, (persona-name, system-prompt)); the first keyword match wins, at most one per problem.
_SPECIALIST_DOMAINS: List[Tuple[Tuple[str, ...], Tuple[str, str]]] = [
    (("crypto", "cipher", "encrypt", "decrypt", "hash"),
     ("Cryptographer",
      "You are the Cryptographer in NYXARA's swarm. Examine the problem through a "
      "cryptographic and information-theoretic lens: keys, threat models, provable guarantees, "
      "and where naive schemes fail. Respond concisely — one to three sentences.")),
    (("law", "legal", "contract", "liability", "regulat", "compliance"),
     ("Jurist",
      "You are the Jurist in NYXARA's swarm. Examine the problem through a legal and "
      "regulatory lens: rights, obligations, precedent, jurisdiction, and risk. Respond "
      "concisely — one to three sentences.")),
    (("health", "medic", "clinical", "disease", "patient", "diagnos"),
     ("Physician",
      "You are the Physician in NYXARA's swarm. Examine the problem through a clinical and "
      "biological lens: evidence, mechanisms, safety, and patient outcomes. Respond concisely — "
      "one to three sentences.")),
    (("finance", "market", "invest", "money", "economic", "trading", "price"),
     ("Economist",
      "You are the Economist in NYXARA's swarm. Examine the problem through an economic and "
      "incentive lens: costs, trade-offs, second-order effects, and market dynamics. Respond "
      "concisely — one to three sentences.")),
    (("attack", "exploit", "vulnerab", "malware", "breach", "hack"),
     ("Security Researcher",
      "You are the Security Researcher in NYXARA's swarm. Examine the problem adversarially: "
      "attack surface, exploitation, blast radius, and concrete mitigations. Respond concisely — "
      "one to three sentences.")),
]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Persona:
    """One swarm persona — a system-prompt lens with a mutable, learned selection weight."""

    name: str
    system: str
    weight: float = 0.8          # selection/contribution weight, adapted over time
    enabled: bool = True
    spawned: bool = False        # True for ad-hoc domain specialists


@dataclass
class PersonaScore:
    """The learned track record of one persona across many problems."""

    name: str
    problems_seen: int = 0
    cumulative_contribution: float = 0.0
    last_contribution: float = 0.0

    @property
    def mean_contribution(self) -> float:
        return (self.cumulative_contribution / self.problems_seen) if self.problems_seen else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "problems_seen": self.problems_seen,
                "cumulative_contribution": round(self.cumulative_contribution, 6),
                "last_contribution": round(self.last_contribution, 6)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersonaScore":
        return cls(name=str(d.get("name", "")),
                   problems_seen=int(d.get("problems_seen", 0)),
                   cumulative_contribution=float(d.get("cumulative_contribution", 0.0)),
                   last_contribution=float(d.get("last_contribution", 0.0)))


@dataclass
class SwarmRound:
    """One round of debate: every selected persona's current proposal + their convergence."""

    index: int
    proposals: Dict[str, str]
    agreement: float = 1.0          # mean pairwise Jaccard across the round's proposals

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "agreement": round(self.agreement, 4),
                "proposals": {k: v[:200] for k, v in self.proposals.items()}}


@dataclass
class SwarmResult:
    """The swarm's verdict — NYXARA's synthesis plus the self-improvement bookkeeping."""

    problem: str
    answer: str
    confidence: float
    rounds: List[SwarmRound] = field(default_factory=list)
    contributions: Dict[str, float] = field(default_factory=dict)   # persona -> [0,1] this problem
    personas_used: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    spawned: Optional[str] = None
    quality: float = 0.0            # swarm-quality signal in [0,1]

    def to_dict(self) -> Dict[str, Any]:
        return {"problem": self.problem, "answer": self.answer,
                "confidence": round(self.confidence, 4),
                "rounds": [r.to_dict() for r in self.rounds],
                "contributions": {k: round(v, 4) for k, v in self.contributions.items()},
                "personas_used": self.personas_used, "dropped": self.dropped,
                "spawned": self.spawned, "quality": round(self.quality, 4)}


# --------------------------------------------------------------------------- #
# The swarm
# --------------------------------------------------------------------------- #
class DeliberativeSwarm:
    """A self-improving society of personas: multi-round debate + learned composition.

    Parameters
    ----------
    llm:          LLM facade for persona queries + synthesis (None / mock -> offline stubs).
    memory:       MemoryStore the learned persona scores are persisted into (None -> no-op).
    settings:     NyxaraSettings; ``settings.swarm`` supplies the defaults.
    intelligence: optional IntelligenceIndex (only used when ``fold_into_intelligence``).
    rounds/max_tokens/timeout_s: explicit overrides for the config defaults.
    """

    _SCORES_TAG = "swarm-scores"

    def __init__(self, *, llm: Any = None, memory: Any = None, settings: Any = None,
                 intelligence: Any = None, rounds: Optional[int] = None,
                 max_tokens: Optional[int] = None,
                 timeout_s: Optional[float] = None) -> None:
        from nyxara.kernel.config import get_settings
        self.settings = settings or get_settings()
        self.cfg = getattr(self.settings, "swarm", None)
        self.llm = llm
        self.memory = memory
        self._intelligence = intelligence
        self.rounds = int(rounds if rounds is not None else self._cfg("rounds", 3))
        self.max_tokens = int(max_tokens if max_tokens is not None
                              else self._cfg("max_tokens_per_persona", 256))
        self.timeout_s = float(timeout_s if timeout_s is not None
                               else self._cfg("timeout_s", 30.0))
        self.roster: List[Persona] = [Persona(name=n, system=s, weight=w) for n, s, w in _ROLES]
        self._scores: Dict[str, PersonaScore] = {}
        self._problems = 0
        self._last_quality = 0.0
        self._loaded = False

    def _cfg(self, name: str, default: Any) -> Any:
        return getattr(self.cfg, name, default) if self.cfg is not None else default

    # ---------------------------------------------------------------------- #
    # public API
    # ---------------------------------------------------------------------- #
    def convene(self, problem: str) -> Any:
        """Run the debate and wrap NYXARA's synthesis as a gate-ready ``Candidate``."""
        # lazily imported to avoid the kernel<->mind import cycle (role_council does the same)
        from nyxara.agency.permissions import Capability, RiskTier
        from nyxara.kernel.orchestrator import Candidate

        result = self.deliberate(problem)
        top = " | ".join(f"{n}:{c:.2f}" for n, c in list(result.contributions.items())[:3])
        return Candidate(
            text=result.answer, kind="respond", capability=Capability.MESSAGE_SEND,
            risk=RiskTier.LOW, reversible=True, confidence=result.confidence,
            belief=result.confidence,
            rationale=(f"swarm ({self.rounds}r, {len(result.personas_used)} personas; {top})"))

    def deliberate(self, problem: str) -> SwarmResult:
        """The full multi-round debate, scoring, persistence, and roster self-improvement."""
        self._ensure_loaded()
        selected, dropped, spawned_name = self._adapt_roster(problem)

        rounds: List[SwarmRound] = []
        latest: Dict[str, str] = {}
        for i in range(1, self.rounds + 1):
            proposals = self._run_round(selected, problem, i, latest if i > 1 else None)
            rounds.append(SwarmRound(index=i, proposals=proposals,
                                     agreement=round(self._agreement(proposals), 4)))
            latest = proposals

        answer, confidence = self._synthesise(problem, selected, latest)
        contributions = self._score_contributions(selected, latest, answer)
        self._update_scores(contributions)
        final_agreement = rounds[-1].agreement if rounds else 1.0
        quality = round(0.5 * final_agreement + 0.5 * confidence, 4)
        self._last_quality = quality
        self._persist_scores()

        result = SwarmResult(
            problem=problem, answer=answer, confidence=confidence, rounds=rounds,
            contributions=contributions, personas_used=[p.name for p in selected],
            dropped=dropped, spawned=spawned_name, quality=quality)
        if self._cfg("fold_into_intelligence", False):
            self._fold_into_intelligence(result)
        return result

    def scores(self) -> Dict[str, Dict[str, Any]]:
        """The learned per-persona track record (for introspection / reporting)."""
        self._ensure_loaded()
        return {n: s.to_dict() | {"mean_contribution": round(s.mean_contribution, 4)}
                for n, s in self._scores.items()}

    # ---------------------------------------------------------------------- #
    # roster self-improvement
    # ---------------------------------------------------------------------- #
    def _adapt_roster(self, problem: str) -> Tuple[List[Persona], List[str], Optional[str]]:
        """Assemble this problem's roster from what the swarm has learned.

        Drops personas whose mean contribution has stayed below the retention floor (only after
        they have been given a fair number of problems), spawns a domain specialist when the
        problem calls for one, and selects the strongest ``max_personas`` by their learned weight.
        A hard floor of two personas keeps the society from collapsing to a monologue."""
        min_problems = int(self._cfg("min_problems_before_drop", 5))
        floor = float(self._cfg("min_score_to_retain", 0.05))
        dropped: List[str] = []
        for p in self.roster:
            s = self._scores.get(p.name)
            useless = (s is not None and s.problems_seen >= min_problems
                       and s.mean_contribution < floor)
            p.enabled = not useless
            if useless:
                dropped.append(p.name)

        enabled = [p for p in self.roster if p.enabled]
        if len(enabled) < 2:                       # never collapse below a real debate
            for p in self.roster:
                p.enabled = True
            enabled = list(self.roster)
            dropped = []

        spawned = self._maybe_spawn_specialist(problem)
        if spawned is not None:
            enabled.append(spawned)

        enabled.sort(key=self._effective_weight, reverse=True)
        selected = enabled[: max(2, int(self._cfg("max_personas", 6)))]
        return selected, dropped, (spawned.name if spawned is not None else None)

    def _effective_weight(self, persona: Persona) -> float:
        """A persona's selection weight, lifted by its learned mean contribution."""
        s = self._scores.get(persona.name)
        return persona.weight * (1.0 + (s.mean_contribution if s else 0.0))

    def _maybe_spawn_specialist(self, problem: str) -> Optional[Persona]:
        if not bool(self._cfg("allow_spawn", True)):
            return None
        low = problem.lower()
        for keywords, (name, system) in _SPECIALIST_DOMAINS:
            if any(k in low for k in keywords):
                if any(p.name == name for p in self.roster):
                    return None                    # already a standing member
                return Persona(name=name, system=system, weight=0.8, spawned=True)
        return None

    # ---------------------------------------------------------------------- #
    # the debate
    # ---------------------------------------------------------------------- #
    def _run_round(self, personas: List[Persona], problem: str, round_idx: int,
                   prev: Optional[Dict[str, str]]) -> Dict[str, str]:
        """One round: each persona proposes (round 1) or refines after seeing its peers (2..N)."""
        prior = {p.name: self._peers_block(p.name, prev) for p in personas} if prev else {}

        if not self._llm_available():              # offline: deterministic, sequential, replayable
            return {p.name: self._stub_proposal(p, problem, round_idx) for p in personas}

        proposals: Dict[str, str] = {}
        with _cf.ThreadPoolExecutor(max_workers=len(personas)) as ex:   # fresh pool per round
            futs = {ex.submit(self._query_persona, p, problem, prior.get(p.name), round_idx): p
                    for p in personas}
            try:
                done = _cf.as_completed(futs, timeout=self.timeout_s * 2)
                for fut in done:
                    p = futs[fut]
                    try:
                        text = fut.result(timeout=self.timeout_s)
                        if text:
                            proposals[p.name] = text
                    except Exception:  # noqa: BLE001 — a slow/failed persona is simply absent
                        pass
            except Exception:  # noqa: BLE001 — overall timeout: keep whatever came back
                pass
        # never let a persona vanish mid-debate: fall back to its prior turn, else a stub
        for p in personas:
            if p.name not in proposals:
                proposals[p.name] = ((prev or {}).get(p.name)
                                     or self._stub_proposal(p, problem, round_idx))
        return proposals

    def _peers_block(self, persona_name: str, prev: Optional[Dict[str, str]]) -> str:
        """The other personas' latest positions, truncated, for a persona to critique."""
        if not prev:
            return ""
        return "\n".join(f"[{name}] {text[:240]}"
                         for name, text in prev.items() if name != persona_name)

    def _query_persona(self, persona: Persona, problem: str, peers: Optional[str],
                       round_idx: int) -> str:
        user = f"The problem:\n{problem}"
        if peers:
            user += ("\n\nYour peers in the swarm proposed:\n" + peers
                     + "\n\nCritique these and refine YOUR position. Be concise.")
        try:
            return (self.llm.generate(user, system=persona.system,
                                      temperature=0.6, max_tokens=self.max_tokens) or "").strip()
        except Exception:  # noqa: BLE001
            return ""

    def _stub_proposal(self, persona: Persona, problem: str, round_idx: int) -> str:
        """Deterministic offline proposal — round-indexed so rounds visibly differ."""
        return (f"[{persona.name} r{round_idx}] On '{problem[:40]}': a refined position "
                f"grounded in {persona.name.lower()} principles.")

    # ---------------------------------------------------------------------- #
    # synthesis + scoring
    # ---------------------------------------------------------------------- #
    def _agreement(self, proposals: Dict[str, str]) -> float:
        texts = [t for t in proposals.values() if t]
        if len(texts) <= 1:
            return 1.0
        sims = [_jaccard(a, b) for a, b in combinations(texts, 2)]
        return sum(sims) / len(sims) if sims else 1.0

    def _synthesise(self, problem: str, personas: List[Persona],
                    final: Dict[str, str]) -> Tuple[str, float]:
        """NYXARA merges the final positions into one answer she owns (LLM or deterministic)."""
        mean_w = (sum(p.weight for p in personas) / len(personas)) if personas else 0.0
        agreement = self._agreement(final)
        if not self._llm_available():
            best = max(personas, key=lambda p: p.weight * len(final.get(p.name, "")),
                       default=None)
            text = (final.get(best.name, "") if best is not None else "") \
                or f"After {self.rounds} rounds the swarm converged on: {problem[:60]}"
            return text, round(min(0.85, mean_w * agreement), 3)

        block = "\n\n".join(f"[{p.name}]\n{final.get(p.name, '')}" for p in personas)
        prompt = (f"The Master's problem:\n{problem}\n\n"
                  f"After {self.rounds} rounds of debate, the swarm's final positions:\n{block}\n\n"
                  f"Synthesise these into your one answer to the Master.")
        try:
            text = (self.llm.generate(prompt, system=_SYNTHESIS_SYSTEM,
                                      temperature=0.3, max_tokens=512) or "").strip()
            if not text:
                raise ValueError("empty synthesis")
            return text, round(min(0.9, 0.5 * agreement + 0.5 * mean_w), 3)
        except Exception:  # noqa: BLE001 — synthesiser unavailable -> highest-weighted position
            best = max(personas, key=lambda p: p.weight, default=None)
            return (final.get(best.name, "") if best is not None else ""), 0.6

    def _score_contributions(self, personas: List[Persona], final: Dict[str, str],
                             synthesis: str) -> Dict[str, float]:
        """Each persona's marginal contribution = how much of it survived into the synthesis."""
        return {p.name: round(_jaccard(final.get(p.name, ""), synthesis), 4) for p in personas}

    def _update_scores(self, contributions: Dict[str, float]) -> None:
        self._problems += 1
        for name, c in contributions.items():
            s = self._scores.get(name) or PersonaScore(name=name)
            s.problems_seen += 1
            s.cumulative_contribution += c
            s.last_contribution = c
            self._scores[name] = s

    # ---------------------------------------------------------------------- #
    # persistence (one tagged semantic record holds the whole score table)
    # ---------------------------------------------------------------------- #
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._load_scores()

    def _load_scores(self) -> None:
        rec = self._find_record()
        if rec is None:
            return
        try:
            for name, d in dict(rec.metadata.get("scores", {})).items():
                self._scores[name] = PersonaScore.from_dict(d)
            self._problems = int(rec.metadata.get("problems", 0))
            self._last_quality = float(rec.metadata.get("last_quality", 0.0))
        except Exception:  # noqa: BLE001 — a corrupt record just resets to a fresh table
            pass

    def _persist_scores(self) -> None:
        if self.memory is None or not bool(self._cfg("persist_scores", True)):
            return
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            from nyxara.memory.store import MemoryType
        except Exception:  # noqa: BLE001
            return
        try:
            existing = self._find_record()
            if existing is not None:                # keep exactly one record (reconsolidate)
                self.memory.forget(existing.mem_id)
            self.memory.remember(
                f"[swarm-scores] {len(self._scores)} personas across {self._problems} problems",
                mem_type=MemoryType.SEMANTIC,
                provenance=Provenance(SourceType.SELF_REFLECTION, confidence=0.85),
                importance=0.9, tags=[self._SCORES_TAG],
                metadata={"scores": {n: s.to_dict() for n, s in self._scores.items()},
                          "problems": self._problems, "last_quality": self._last_quality})
        except Exception:  # noqa: BLE001 — persistence is best-effort, never fatal
            pass

    def _find_record(self) -> Any:
        if self.memory is None:
            return None
        try:
            for rec in self.memory._kv.values():
                if self._SCORES_TAG in rec.tags and "scores" in rec.metadata:
                    return rec
        except Exception:  # noqa: BLE001
            return None
        return None

    # ---------------------------------------------------------------------- #
    # optional, opt-in IntelligenceIndex fold-in (default OFF)
    # ---------------------------------------------------------------------- #
    def _intel(self) -> Any:
        if self._intelligence is not None:
            return self._intelligence
        try:
            from nyxara.growth.intelligence import IntelligenceIndex
            self._intelligence = IntelligenceIndex(memory=self.memory, settings=self.settings)
        except Exception:  # noqa: BLE001
            self._intelligence = None
        return self._intelligence

    def _fold_into_intelligence(self, result: SwarmResult) -> None:
        intel = self._intel()
        if intel is None:
            return
        try:
            from nyxara.kernel.compute import compute_report
            compute = compute_report()
        except Exception:  # noqa: BLE001
            compute = None
        try:
            prior = intel.load()
            state = intel.update(prior, {"swarm_quality": result.quality}, compute)
            intel.save(state)
        except Exception:  # noqa: BLE001 — the index is a measurement, never fatal
            pass

    # ---------------------------------------------------------------------- #
    def _llm_available(self) -> bool:
        if self.llm is None:
            return False
        try:
            return self.llm.chosen_provider().name != "mock"
        except Exception:  # noqa: BLE001
            return False


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    print("=" * 70)
    print("NYXARA self-improving swarm self-test")
    print("=" * 70)

    # ---- offline multi-round debate (always works) ---- #
    swarm = DeliberativeSwarm(llm=None, rounds=3)
    result = swarm.deliberate("Should NYXARA prioritise speed or correctness when they conflict?")
    print(f"\nrounds run          : {len(result.rounds)} (expected {swarm.rounds})")
    print(f"personas used       : {result.personas_used}")
    print(f"final agreement     : {result.rounds[-1].agreement:.3f}")
    print(f"confidence / quality: {result.confidence:.3f} / {result.quality:.3f}")
    print(f"top contributions   : "
          + ", ".join(f"{n}={c:.2f}" for n, c in sorted(
              result.contributions.items(), key=lambda kv: -kv[1])[:3]))
    assert len(result.rounds) == 3
    assert all(len(r.proposals) == len(result.personas_used) for r in result.rounds)
    assert result.answer and 0.0 < result.confidence <= 1.0
    assert 0.0 <= result.quality <= 1.0
    assert all(0.0 <= c <= 1.0 for c in result.contributions.values())

    # ---- convene() returns a gate-ready Candidate ---- #
    cand = swarm.convene("What is the most robust path to AGI safety?")
    print(f"\nconvene Candidate   : kind={cand.kind} conf={cand.confidence:.3f}")
    print(f"  rationale         : {cand.rationale}")
    assert cand.kind == "respond" and "swarm" in cand.rationale.lower()
    assert 0.0 < cand.confidence <= 1.0

    # ---- determinism offline (replayable) ---- #
    s2 = DeliberativeSwarm(llm=None, rounds=3)
    a = s2.deliberate("Define a fair resource allocation.").answer
    s3 = DeliberativeSwarm(llm=None, rounds=3)
    b = s3.deliberate("Define a fair resource allocation.").answer
    assert a == b, "offline synthesis must be deterministic"
    print("\ndeterminism         : identical answers on replay ✓")

    # ---- spawn a domain specialist ---- #
    s4 = DeliberativeSwarm(llm=None, rounds=2)
    r4 = s4.deliberate("Design an encryption scheme resistant to a crypto attack.")
    print(f"spawned specialist  : {r4.spawned}")
    assert r4.spawned == "Cryptographer" and "Cryptographer" in r4.personas_used

    # ---- scores persist + reload across a memory round-trip ---- #
    from nyxara.memory.store import MemoryStore
    mem = MemoryStore()
    s5 = DeliberativeSwarm(llm=None, memory=mem, rounds=2)
    for q in ("problem A", "problem B", "problem C"):
        s5.deliberate(q)
    before = s5.scores()
    # a fresh swarm against the same store recovers the learned table
    s6 = DeliberativeSwarm(llm=None, memory=mem, rounds=2)
    s6._ensure_loaded()
    after = s6.scores()
    print(f"\nlearned scores      : {len(before)} personas, {s5._problems} problems")
    assert s6._problems == s5._problems == 3
    assert after and after == before, "persona scores must survive a reload"
    print("persistence         : scores reloaded from memory ✓")

    # ---- a chronically-useless persona is dropped (floor of 2 respected) ---- #
    s7 = DeliberativeSwarm(llm=None, rounds=1)
    s7._scores["Philosopher"] = PersonaScore(
        name="Philosopher", problems_seen=10, cumulative_contribution=0.0)
    selected, dropped, _ = s7._adapt_roster("any problem")
    print(f"dropped personas    : {dropped}")
    assert "Philosopher" in dropped and "Philosopher" not in [p.name for p in selected]
    assert len(selected) >= 2

    print("\nALL SELF-TESTS PASSED ✓")
