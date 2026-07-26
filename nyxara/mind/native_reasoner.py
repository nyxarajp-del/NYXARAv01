"""NYXARA · mind/native_reasoner.py — HER OWN chain of thought (👑, not the LLM's).

Prompt-driven "chain of thought" is the LLM writing prose that *looks* like thinking.
This module is the opposite: NYXARA reasons by **running her own faculties** and the
trace is the record of real computation — every step is something she actually did
(queried her learned causal graph, walked her knowledge graph, ran a symbolic
derivation, executed and certified a program, mapped structure between domains) and
every conclusion carries a machine-checkable artifact or is not emitted at all.

The pipeline (each stage appends :class:`~nyxara.mind.explain.ReasoningStep`\\ s):

1. **Parse** — classify the question's intent (why / what-if / does-X-cause-Y /
   counterfactual / how-do-I-achieve / computational / factual / analogy / chat).
2. **Decompose (multi-hop blackboard)** — a compound question is split into parts;
   a part's derived conclusion becomes a fact the next part can consume (pronouns
   are resolved against it), so sub-engines genuinely COMPOSE.
3. **Dispatch** — the intent's sub-engines run in an order chosen by a persisted
   UCB1 bandit (she *learns which of her engines pays off for which question shape*):
   causal graph queries, intervention planning, verifiable chains/first principles,
   compute-and-verify, knowledge-graph answering, structure-mapping analogy.
   Every engine abstains honestly (``None``) — nothing is ever fabricated.
4. **Verify** — where an oracle exists (:mod:`~nyxara.mind.grounded_verifier`), a
   contradicted answer is dropped, not softened.
5. **Calibrate** — a persisted per-intent :class:`~nyxara.mind.uncertainty.Calibrator`
   recalibrates raw confidence from lived outcomes; an
   :class:`~nyxara.mind.uncertainty.AbstentionPolicy` decides answer-vs-abstain.

Beyond answering, the system **improves itself** (:class:`NativeReasoningImprover`):
every reasoning act is logged to a bounded replay file; on growth ticks NYXARA
proposes small parameter mutations (abstention floors), REPLAYS her own recorded
turns under each proposal, and applies a mutation only when it *measurably scores
better* — the proof-carrying-improvement contract, within hard safety bounds.
Verified conclusions are also promoted into the knowledge graph, so her knowledge
compounds from her own verified reasoning — genuine learning beyond training data.

Pure standard library; every heavy faculty is import-guarded and optional.
The LLM appears nowhere in this module.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from nyxara.mind.explain import ReasoningStep, StepKind

__all__ = [
    "NativeConclusion",
    "NativeReasoner",
    "RungBandit",
    "NativeReasoningImprover",
    "classify_intent",
]

# hard safety bounds for anything the self-improvement loop may tune
_FLOOR_BOUNDS = (0.2, 0.9)


# --------------------------------------------------------------------------- #
# Intent parsing — what shape of question is this?
# --------------------------------------------------------------------------- #
_RE_COUNTERFACTUAL = re.compile(
    r"\bwhat\s+if\b.*\b(had\s+not|hadn't|had\s+never|not\s+happened)\b"
    r"|\bhad\s+.+\s+not\s+(happened|occurred)\b", re.I)
_RE_CAUSAL_PAIR = re.compile(
    r"\b(?:does|did|do|can|could|will)\s+(.+?)\s+cause[sd]?\s+(.+?)\s*[?.]*\s*$", re.I)
_RE_NECESSITY = re.compile(
    r"\b(?:did|was|is|were)\s+(.+?)\s+(?:really\s+)?"
    r"(?:matter(?:ed)?|necessary|essential|required|sufficient)\s+"
    r"(?:for|to)\s+(.+?)\s*[?.]*\s*$", re.I)
_RE_WHY = re.compile(r"^\s*why\b", re.I)
_RE_WHATIF = re.compile(
    r"\bwhat\s+(?:happens|happened|will\s+happen|would\s+happen)\s+(?:if|when)\b"
    r"|\b(?:the\s+)?effects?\s+of\b|\bwhat\s+if\s+i\b", re.I)
_RE_INTERVENTION = re.compile(
    r"\bhow\s+(?:can|do|could|should|would|to)\s*(?:i|we|you|one)?\s*"
    r"(make|achieve|cause|get|ensure|prevent|stop|avoid|reduce|increase|bring about)\b", re.I)
_RE_ANALOGY = re.compile(r"\bis\s+like\b|\banalog(?:y|ous)\b|\bsimilar\s+to\b", re.I)
_RE_FACTUAL = re.compile(r"^\s*(?:who|what|where|which|whose)\b", re.I)
_COMPUTE_CUES = ("compute", "calculate", "how many", "how much", "count", "solve",
                 "sum of", "product of", "average", "median", "gcd", "lcm", "modulo",
                 "remainder", "percent", "derivative", "integral", "evaluate", "prime",
                 "factor", "probability", "digits")


def classify_intent(text: str) -> str:
    """Classify a question into the intent class that picks its sub-engines."""
    t = (text or "").strip()
    low = t.lower()
    if not t:
        return "chat"
    if _RE_COUNTERFACTUAL.search(t):
        return "counterfactual"
    if _RE_NECESSITY.search(t):
        return "causal_necessity"
    if _RE_CAUSAL_PAIR.search(t):
        return "causal_pair"
    if _RE_WHY.search(t):
        return "causal_why"
    if _RE_INTERVENTION.search(t):
        return "intervention"
    if _RE_WHATIF.search(t):
        return "causal_whatif"
    if re.search(r"\d", t) and re.search(r"[-+*/%^=]|\*\*", t):
        return "computational"
    if any(c in low for c in _COMPUTE_CUES):
        return "computational"
    if _RE_ANALOGY.search(t):
        return "analogy"
    if _RE_FACTUAL.search(t):
        return "factual"
    return "chat"


# which sub-engines may answer which intent (bandit re-orders within the list).
# "law" is offered first on quantitative/factual intents: if she has discovered a governing law
# that fits the question it grounds the answer; otherwise it abstains and the others take over.
_INTENT_ENGINES: Dict[str, List[str]] = {
    "causal_why": ["law", "causal"],
    "causal_whatif": ["law", "causal"],
    "causal_pair": ["causal"],
    "causal_necessity": ["causal"],
    "counterfactual": ["causal"],
    "intervention": ["law", "intervention"],
    "computational": ["law", "chain", "compute"],
    "factual": ["law", "graph"],
    "analogy": ["analogy", "graph"],
    # "law" is offered on the catch-all too so a yes/no proposition she has settled ("does X …?")
    # can be answered from her own belief; it abstains on ordinary chat, so nothing else changes.
    "chat": ["law"],
}


# --------------------------------------------------------------------------- #
# The conclusion — an answer plus the real trace that produced it
# --------------------------------------------------------------------------- #
@dataclass
class NativeConclusion:
    """An answer NYXARA reasoned to HERSELF, with the inspectable step trace."""

    answer: str
    confidence: float
    source: str                 # causal | intervention | chain | compute | graph | analogy
    intent: str = "chat"
    steps: List[ReasoningStep] = field(default_factory=list)
    verified: bool = False
    certificate: str = ""

    def trace_lines(self) -> List[str]:
        return [f"{i + 1}. [{s.kind.value}] {s.summary}"
                + (f" — {s.detail}" if s.detail else "")
                for i, s in enumerate(self.steps)]

    def render(self, *, with_trace: bool = True) -> str:
        if not with_trace or not self.steps:
            return self.answer
        body = "\n".join(f"  {line}" for line in self.trace_lines())
        return f"{self.answer}\n\n— my own reasoning ({self.source}):\n{body}"

    def to_dict(self) -> Dict[str, Any]:
        return {"answer": self.answer, "confidence": round(self.confidence, 4),
                "source": self.source, "intent": self.intent,
                "verified": self.verified, "certificate": self.certificate,
                "steps": [s.to_dict() for s in self.steps]}


# --------------------------------------------------------------------------- #
# UCB1 rung bandit — she learns WHICH of her engines pays off for what
# --------------------------------------------------------------------------- #
class RungBandit:
    """Per-intent UCB1 over sub-engines, persisted so the learning compounds."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._arms: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._load()

    def _arm(self, intent: str, engine: str) -> Dict[str, float]:
        return self._arms.setdefault(intent, {}).setdefault(
            engine, {"n": 0.0, "wins": 0.0})

    def order(self, intent: str, engines: List[str]) -> List[str]:
        """Engines sorted by UCB1 score (untried arms explored first)."""
        if len(engines) < 2:
            return list(engines)
        total = sum(self._arm(intent, e)["n"] for e in engines) or 1.0

        def _score(e: str) -> float:
            a = self._arm(intent, e)
            if a["n"] <= 0:
                return float("inf")               # explore what she has never tried
            return a["wins"] / a["n"] + math.sqrt(2.0 * math.log(total) / a["n"])

        return sorted(engines, key=_score, reverse=True)

    def record(self, intent: str, engine: str, success: bool) -> None:
        a = self._arm(intent, engine)
        a["n"] += 1.0
        a["wins"] += 1.0 if success else 0.0
        self._save()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                self._arms = json.load(fh)
        except Exception:  # noqa: BLE001 — a corrupt file never bricks the mind
            self._arms = {}

    def _save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._arms, fh)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001 — persistence is best-effort
            pass


# --------------------------------------------------------------------------- #
# Persisted per-intent calibration
# --------------------------------------------------------------------------- #
class _CalibrationStore:
    """Per-intent :class:`~nyxara.mind.uncertainty.Calibrator`\\ s, persisted as JSON."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._cals: Dict[str, Any] = {}
        self._load()

    def calibrator(self, intent: str) -> Any:
        from nyxara.mind.uncertainty import Calibrator
        cal = self._cals.get(intent)
        if cal is None:
            cal = Calibrator()
            self._cals[intent] = cal
        return cal

    def recalibrate(self, intent: str, confidence: float) -> float:
        cal = self.calibrator(intent)
        if cal.samples < 5:              # too little lived evidence to correct anything
            return confidence
        return float(cal.recalibrate(confidence))

    def record(self, intent: str, confidence: float, correct: bool) -> None:
        self.calibrator(intent).add(float(confidence), bool(correct))
        self._save()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            from nyxara.mind.uncertainty import Calibrator
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for intent, d in raw.items():
                cal = Calibrator(n_bins=len(d.get("count", [])) or 10)
                cal._sum_conf = [float(x) for x in d.get("sum_conf", cal._sum_conf)]
                cal._sum_correct = [float(x) for x in d.get("sum_correct", cal._sum_correct)]
                cal._count = [int(x) for x in d.get("count", cal._count)]
                cal._n = int(d.get("n", 0))
                cal._brier_sum = float(d.get("brier_sum", 0.0))
                self._cals[intent] = cal
        except Exception:  # noqa: BLE001
            self._cals = {}

    def _save(self) -> None:
        if not self.path:
            return
        try:
            raw = {intent: {"sum_conf": cal._sum_conf, "sum_correct": cal._sum_correct,
                            "count": cal._count, "n": cal._n, "brier_sum": cal._brier_sum}
                   for intent, cal in self._cals.items()}
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(raw, fh)
            os.replace(tmp, self.path)
        except Exception:  # noqa: BLE001
            pass


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _norm_key(text: str) -> str:
    """A stable normalized key for a problem statement (verified-answer cache)."""
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    return " ".join(toks)[:160]


# --------------------------------------------------------------------------- #
# The native reasoner
# --------------------------------------------------------------------------- #
class NativeReasoner:
    """NYXARA's own chain-of-thought orchestrator. The LLM authors nothing here."""

    def __init__(self, *, knowledge_graph: Any = None, causal_model: Any = None,
                 memory: Any = None, compute: Any = None,
                 law_discovery: Any = None, belief_model: Any = None,
                 settings: Any = None, data_dir: Any = None) -> None:
        if settings is None:
            from nyxara.kernel.config import get_settings
            settings = get_settings()
        self.settings = settings
        self.cfg = getattr(settings, "native_reasoning", None)
        self.graph = knowledge_graph
        self.causal = causal_model
        self.memory = memory
        self.compute = compute
        # the empirical/physical laws she DISCOVERED herself (law_discovery) and her settled
        # BeliefModel — so self-learning actually grounds her answers instead of only living on
        # disk. Both optional; the rung abstains cleanly when neither applies. No LLM.
        self.law_discovery = law_discovery
        self.belief_model = belief_model

        base = data_dir
        if base is None:
            base = getattr(getattr(settings, "paths", None), "data_dir", None)
        self._dir = str(base) if base else None

        def _p(name: str) -> Optional[str]:
            return os.path.join(self._dir, name) if self._dir else None

        self.bandit = RungBandit(_p("native_rungs.json")) if self._flag("bandit") else None
        self.calibration = (_CalibrationStore(_p("native_calibration.json"))
                            if self._flag("calibration") else None)
        self.replay_path = _p("native_replay.jsonl")
        self.tuning_path = _p("native_tuning.json")
        self._tuning: Dict[str, Any] = {}
        self.reload_tuning()
        # the last emitted (intent, engine, confidence) awaiting a lived outcome
        self._pending: Optional[Dict[str, Any]] = None
        self._flushes = 0

    # ------------------------------------------------------------------ #
    # config / tuning
    # ------------------------------------------------------------------ #
    def _flag(self, name: str, default: bool = True) -> bool:
        return bool(getattr(self.cfg, name, default))

    def _floor(self, intent: str) -> float:
        """The abstention floor: config default, overridden by her own PROVEN tuning."""
        base = float(getattr(self.cfg, "min_confidence", 0.35))
        base = float(self._tuning.get("min_confidence", base))
        base = float(self._tuning.get("intent_floors", {}).get(intent, base))
        return _clamp(base, *_FLOOR_BOUNDS)

    def reload_tuning(self) -> None:
        """Load the self-improvement loop's accepted parameters (bounded overrides)."""
        self._tuning = {}
        if self.tuning_path and os.path.exists(self.tuning_path):
            try:
                with open(self.tuning_path, "r", encoding="utf-8") as fh:
                    self._tuning = json.load(fh)
            except Exception:  # noqa: BLE001
                self._tuning = {}

    # ------------------------------------------------------------------ #
    # the reasoning act
    # ------------------------------------------------------------------ #
    def reason(self, stimulus: str, mems: Optional[List[str]] = None
               ) -> Optional[NativeConclusion]:
        """Answer with her own faculties, or ``None`` — an honest abstain. Never raises."""
        try:
            return self._reason(stimulus, mems or [])
        except Exception:  # noqa: BLE001 — native reasoning must never crash a turn
            return None

    def _reason(self, stimulus: str, mems: List[str]) -> Optional[NativeConclusion]:
        text = (stimulus or "").strip()
        if not text or self.cfg is not None and not self._flag("enabled"):
            return None
        steps: List[ReasoningStep] = []
        intent = classify_intent(text)
        self._step(steps, StepKind.PERCEIVE, f"classified the question as {intent!r}",
                   detail=text[:120])
        if mems:
            self._step(steps, StepKind.RECALL,
                       f"recalled {len(mems)} relevant memories", evidence=mems[:3])

        parts = self._decompose(text) if self._flag("multi_hop") else [text]
        if len(parts) > 1:
            self._step(steps, StepKind.SELECT,
                       f"decomposed into {len(parts)} sub-questions (multi-hop)",
                       detail=" | ".join(p[:60] for p in parts))

        answers: List[Tuple[str, float, str, bool, str]] = []
        focus_label: Optional[str] = None
        max_steps = int(getattr(self.cfg, "max_steps", 24))
        for part in parts:
            if len(steps) >= max_steps:
                break
            if focus_label is not None:
                # blackboard: the previous hop's subject resolves this hop's pronouns
                resolved = re.sub(r"\b(it|that|this)\b",
                                  focus_label.replace("_", " "), part, flags=re.I)
                if resolved != part:
                    self._step(steps, StepKind.REASON,
                               f"resolved the pronoun against the previous hop: "
                               f"{focus_label!r}", detail=resolved[:100])
                    part = resolved
            part_intent = classify_intent(part) if len(parts) > 1 else intent
            hit = self._dispatch(part, part_intent, steps)
            if hit is None:
                return self._abstain(steps, intent, "a sub-question found no "
                                                    "verifiable answer")
            answer, conf, engine, verified, cert = hit
            answers.append(hit)
            focus_label = self._focus_of(part, part_intent) or focus_label

        if not answers:
            return self._abstain(steps, intent, "no engine could verify an answer")

        answer = "  ".join(a for a, _, _, _, _ in answers)
        confidence = min(c for _, c, _, _, _ in answers)
        sources = []
        for _, _, e, _, _ in answers:
            if e not in sources:
                sources.append(e)
        source = "+".join(sources)
        verified = all(v for _, _, _, v, _ in answers)
        certificate = "; ".join(c for _, _, _, _, c in answers if c)

        # verify against an oracle where one exists — a contradicted answer is DROPPED
        ground = self._ground_truth(text)
        if ground is not None:
            ok = self._carries_truth(ground, answer)
            self._step(steps, StepKind.GATE,
                       "checked against the grounded oracle",
                       detail="consistent" if ok else f"contradicted (truth: {ground})")
            if not ok:
                return self._abstain(steps, intent, "the grounded oracle contradicted "
                                                    "the answer")
            verified, confidence = True, max(confidence, 0.95)

        # calibrate from lived outcomes, then decide answer-vs-abstain
        raw_conf = confidence
        if self.calibration is not None:
            confidence = _clamp(self.calibration.recalibrate(intent, confidence))
        decision_floor = self._floor(intent)
        from nyxara.mind.uncertainty import AbstentionPolicy
        decision = AbstentionPolicy(min_confidence=decision_floor).decide(
            raw_conf, calibrated=confidence)
        self._step(steps, StepKind.GATE,
                   f"calibrated confidence {raw_conf:.2f}→{confidence:.2f} "
                   f"(floor {decision_floor:.2f})", detail=decision.reason)
        if decision.should_abstain:
            return self._abstain(steps, intent, decision.reason)

        conclusion = NativeConclusion(answer=answer, confidence=confidence,
                                      source=source, intent=intent, steps=steps,
                                      verified=verified, certificate=certificate)
        self._step(steps, StepKind.DECIDE,
                   f"concluded via {source} (verified={verified})",
                   confidence=confidence)
        self._promote(conclusion)
        self._log_reasoning(conclusion, answers[0][2])
        return conclusion

    def _abstain(self, steps: List[ReasoningStep], intent: str,
                 reason: str) -> None:
        self._step(steps, StepKind.ABSTAIN, reason)
        return None

    @staticmethod
    def _step(steps: List[ReasoningStep], kind: StepKind, summary: str, *,
              detail: str = "", confidence: Optional[float] = None,
              evidence: Optional[List[str]] = None) -> None:
        steps.append(ReasoningStep(kind=kind, summary=summary, detail=detail,
                                   confidence=confidence, evidence=evidence or [],
                                   index=len(steps)))

    # ------------------------------------------------------------------ #
    # decomposition (multi-hop blackboard)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _decompose(text: str) -> List[str]:
        """Split a compound question before a question word — never inside a noun phrase."""
        parts = re.split(
            r"\s+(?:and|then|;)\s+(?=(?:why|how|what|does|did|is|are|can|could|should|"
            r"who|where|which)\b)", text.strip(), flags=re.I)
        return [p.strip().rstrip(",") for p in parts if p.strip()] or [text]

    def _focus_of(self, part: str, intent: str) -> Optional[str]:
        """The causal label this part was about (feeds pronoun resolution downstream)."""
        if self.causal is None:
            return None
        try:
            if intent in ("causal_why", "causal_whatif", "counterfactual",
                          "intervention"):
                return self.causal.match_label(self._topic_segment(part, intent))
        except Exception:  # noqa: BLE001
            return None
        return None

    @staticmethod
    def _topic_segment(text: str, intent: str) -> str:
        """The slice of the question that names its subject (strips the wh-scaffold)."""
        low = text.strip()
        if intent == "causal_why":
            return re.sub(r"^\s*why\s+(?:did|do|does|is|was|are|were)?\s*", "", low,
                          flags=re.I)
        if intent == "counterfactual":
            m = re.search(r"\bif\s+(.+?)\s+had\b", low, flags=re.I)
            return m.group(1) if m else low
        if intent == "causal_whatif":
            m = re.search(r"\b(?:if|when|of)\s+(?:i\s+)?(.+?)\s*[?.]*\s*$", low, flags=re.I)
            return m.group(1) if m else low
        if intent == "intervention":
            m = _RE_INTERVENTION.search(low)
            if m:
                return low[m.end():].strip(" ?.")
        return low

    # ------------------------------------------------------------------ #
    # dispatch — bandit-ordered sub-engines, first verifiable answer wins
    # ------------------------------------------------------------------ #
    def _dispatch(self, part: str, intent: str, steps: List[ReasoningStep]
                  ) -> Optional[Tuple[str, float, str, bool, str]]:
        # her own verified past work answers repeats instantly (compounding loop).
        # ordinary conversation has no durable "verified answer" — a chat turn keyed on the
        # whole utterance must never replay a stale reply, so the cache is skipped for it.
        cached = self._cached_answer(part) if intent != "chat" else None
        if cached is not None:
            answer, conf = cached
            self._step(steps, StepKind.RECALL,
                       "answered from my own previously VERIFIED work "
                       "(knowledge graph)", detail=answer[:100], confidence=conf)
            return answer, conf, "graph", True, "previously certified by my own compute"

        engines = list(_INTENT_ENGINES.get(intent, ()))
        if not engines:
            return None
        if self.bandit is not None:
            engines = self.bandit.order(intent, engines)
        if len(engines) > 1:
            self._step(steps, StepKind.SELECT,
                       f"engine order for {intent!r} (learned): {', '.join(engines)}")
        table: Dict[str, Callable[[str, str, List[ReasoningStep]],
                                  Optional[Tuple[str, float, bool, str]]]] = {
            "law": self._law_answer,
            "causal": self._causal_answer,
            "intervention": self._intervention_answer,
            "chain": self._chain_answer,
            "compute": self._compute_answer,
            "graph": self._graph_answer,
            "analogy": self._analogy_answer,
        }
        for engine in engines:
            fn = table.get(engine)
            if fn is None:
                continue
            try:
                hit = fn(part, intent, steps)
            except Exception:  # noqa: BLE001 — one engine failing defers, never crashes
                hit = None
            if hit is not None:
                answer, conf, verified, cert = hit
                return answer, conf, engine, verified, cert
        return None

    # ------------------------------------------------------------------ #
    # engine: the laws SHE discovered — self-learning that grounds an answer
    # ------------------------------------------------------------------ #
    def _law_answer(self, part: str, intent: str, steps: List[ReasoningStep]
                    ) -> Optional[Tuple[str, float, bool, str]]:
        """Answer from a law NYXARA discovered herself (``law_discovery``) or a belief she settled
        (``belief_model``). This is the closed feedback loop: what her autonomous science learned
        actually changes what she says. Abstains (``None``) when nothing she discovered applies, so
        the other engines take over cleanly. No LLM — she evaluates her OWN corroborated law."""
        eng = self.law_discovery
        if eng is not None:
            try:
                var_values = self._extract_var_values(part)
                # 1) quantitative: plug the given inputs into the best-fitting discovered law
                if var_values:
                    res = eng.apply(var_values, query=part)
                    if res is not None:
                        target = str(res.get("target", "the result"))
                        value = float(res.get("value"))
                        expr = str(res.get("expression", ""))
                        inputs = ", ".join(f"{k}={v:g}" for k, v in
                                           (res.get("inputs") or {}).items())
                        self._step(steps, StepKind.REASON,
                                   f"applied a law I discovered myself: {expr}",
                                   detail=f"with {inputs}", confidence=0.9,
                                   evidence=[expr])
                        answer = (f"{target} = {value:.6g} — from the law I discovered "
                                  f"myself: {expr} (with {inputs}).")
                        return answer, 0.9, True, \
                            f"evaluated my own corroborated law {expr}"
                # 2) qualitative: name the governing law she found for this topic
                law = eng.recall(part)
                if law is not None:
                    self._step(steps, StepKind.REASON,
                               "recalled a governing law from my own discovery tower",
                               detail=law.expression, confidence=0.8,
                               evidence=[law.expression])
                    answer = (f"From my own experiments I discovered the law "
                              f"{law.expression}.")
                    return answer, 0.8, True, \
                        f"corroborated law {law.signature()} from my discovery tower"
            except Exception:  # noqa: BLE001 — a discovery engine hiccup defers, never crashes
                pass
        # 3) a settled belief answers a yes/no proposition she has investigated
        belief_hit = self._belief_answer(part, steps)
        if belief_hit is not None:
            return belief_hit
        return None

    _RE_VAR_ASSIGN = re.compile(
        r"([A-Za-z_]\w*)\s*(?:=|:|\bis\b|\bof\b|\bequals?\b|\bequal to\b)\s*"
        r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")

    def _extract_var_values(self, text: str) -> Dict[str, float]:
        """Pull ``name = number`` style assignments out of a question (``m=3``, ``g of 9.8``,
        ``t is 2``). The keys are matched against a discovered law's variables in ``apply``."""
        out: Dict[str, float] = {}
        for name, num in self._RE_VAR_ASSIGN.findall(text or ""):
            try:
                out[name.lower()] = float(num)
            except (TypeError, ValueError):
                continue
        return out

    def _belief_answer(self, part: str, steps: List[ReasoningStep]
                       ) -> Optional[Tuple[str, float, bool, str]]:
        """Answer a proposition from a belief she settled through her own investigations. Matches
        conservatively (the belief's statement must overlap the question strongly) so she never
        launders an unrelated belief into an answer."""
        bm = self.belief_model
        beliefs = getattr(bm, "beliefs", None)
        if not beliefs:
            return None
        ql = (part or "").lower()
        q_tokens = {w.strip("?.,!:;\"'()") for w in ql.split() if len(w) > 3}
        if not q_tokens:
            return None
        best = None
        best_overlap = 0.0
        for belief in beliefs.values():
            stmt = (getattr(belief, "statement", "") or "").lower()
            s_tokens = {w.strip("?.,!:;\"'()") for w in stmt.split() if len(w) > 3}
            if not s_tokens:
                continue
            overlap = len(q_tokens & s_tokens) / max(1, len(s_tokens))
            if overlap > best_overlap:
                best_overlap, best = overlap, belief
        # require a strong match and a belief that is actually settled (not still at 0.5)
        if best is None or best_overlap < 0.6 or getattr(best, "verdict", "") == "inconclusive":
            return None
        verdict = getattr(best, "verdict", "inconclusive")
        conf = float(getattr(best, "confidence", 0.5) or 0.5)
        self._step(steps, StepKind.REASON,
                   f"recalled a belief I settled through my own investigation: "
                   f"{verdict!r}", detail=getattr(best, "statement", "")[:100],
                   confidence=conf, evidence=[getattr(best, "last_reasoning", "")[:80]])
        yn = {"supported": "Yes", "refuted": "No"}.get(verdict, "Uncertain")
        answer = (f"{yn} — from my own investigation of "
                  f"{getattr(best, 'statement', 'this')!r}: {verdict} "
                  f"(confidence {conf:.0%}).")
        return answer, _clamp(max(conf, 0.4)), True, \
            f"settled belief {verdict} from {getattr(best, 'evidence_count', 1)} investigations"

    # ------------------------------------------------------------------ #
    # engine: the learned causal graph — why / what-if / pair / counterfactual
    # ------------------------------------------------------------------ #
    def _causal_answer(self, part: str, intent: str, steps: List[ReasoningStep]
                       ) -> Optional[Tuple[str, float, bool, str]]:
        if self.causal is None or not self._flag("causal_queries"):
            return None
        cwm = self.causal
        if intent == "causal_pair":
            m = _RE_CAUSAL_PAIR.search(part)
            if not m:
                return None
            a = cwm.match_label(m.group(1))
            b = cwm.match_label(m.group(2))
            if a is None or b is None:
                return None
            v = cwm.is_causal(a, b)
            self._step(steps, StepKind.REASON,
                       f"queried my causal graph: {a!r} → {b!r} is {v.verdict}",
                       detail=v.reason, confidence=v.confidence,
                       evidence=[f"{k}={val}" for k, val in v.evidence.items()][:4])
            if v.is_causal:
                answer = f"Yes — {a!r} causes {b!r}: {v.reason}."
            else:
                answer = f"No — {v.verdict}: {v.reason}."
            return answer, _clamp(max(v.confidence, 0.4)), True, \
                f"causal verdict {v.verdict} from {v.evidence.get('events', 0)} lived events"

        if intent == "causal_necessity":
            m = _RE_NECESSITY.search(part)
            if not m:
                return None
            a = cwm.match_label(m.group(1))
            b = cwm.match_label(m.group(2))
            if a is None or b is None:
                return None
            result = cwm.necessity_sufficiency(a, b)
            if result is None:
                self._step(steps, StepKind.REASON,
                           f"no fitted structural mechanism between {a!r} and {b!r} yet")
                return (f"I don't have enough learned structure between {a!r} and "
                        f"{b!r} yet to say honestly whether it was necessary or "
                        f"sufficient."), 0.4, True, \
                    "honest abstention — no fitted mechanism to sample counterfactual noise from"
            pn, ps, pns = result["PN"], result["PS"], result["PNS"]
            self._step(steps, StepKind.SIMULATE,
                       f"ran {result['samples']} Monte Carlo structural counterfactuals "
                       f"(abduction over fitted noise) for {a!r} -> {b!r}",
                       evidence=[f"PN={pn:.2f}", f"PS={ps:.2f}", f"PNS={pns:.2f}"])
            answer = (f"Necessity {pn:.0%} (without {a!r}, {b!r} likely would not have "
                      f"happened), sufficiency {ps:.0%} (with {a!r}, {b!r} likely would "
                      f"have) — necessary-and-sufficient in {pns:.0%} of the simulated "
                      f"instances.")
            return answer, _clamp(max(pn, ps, 0.4)), True, \
                f"Pearl PN/PS/PNS from {result['samples']} structural Monte Carlo draws"

        topic = self._topic_segment(part, intent)
        label = cwm.match_label(topic)
        if label is None:
            return None

        if intent == "causal_why":
            links = cwm.why(label)
            if not links:
                self._step(steps, StepKind.REASON,
                           f"queried my causal graph for causes of {label!r}: none "
                           f"established yet")
                return (f"I have no established cause for {label!r} yet — I have not "
                        f"observed enough evidence to say honestly."), 0.5, True, \
                    "honest no-cause verdict from the causal graph"
            chain = self._causal_chain_to(label)
            for link in links[:3]:
                self._step(steps, StepKind.REASON,
                           f"cause found: {link.describe()}",
                           confidence=link.confidence,
                           evidence=[f"ΔP={link.evidence.get('delta_p')}",
                                     f"lag={link.lag:.1f}s"])
            top = links[0]
            answer = cwm.explain(label)
            if len(chain) > 2:
                self._step(steps, StepKind.REASON,
                           "traced the causal chain " + " → ".join(chain))
                answer += " Full chain: " + " → ".join(chain) + "."
            return answer, top.confidence, True, \
                f"causal link {top.cause}→{label} (conf {top.confidence:.0%})"

        if intent == "causal_whatif":
            effects = cwm.predict_effects(label)
            if not isinstance(effects, dict) or not effects:
                self._step(steps, StepKind.REASON,
                           f"no established downstream effects of {label!r} yet")
                return (f"I have no established effects of {label!r} yet."), 0.5, True, \
                    "honest no-effect verdict from the causal graph"
            ranked = sorted(effects.items(), key=lambda kv: abs(kv[1]), reverse=True)
            self._step(steps, StepKind.SIMULATE,
                       f"propagated do({label}) through my causal graph",
                       evidence=[f"{e}: {sz:+.2f}" for e, sz in ranked[:4]])
            listing = ", ".join(f"{e!r} ({sz:+.2f})" for e, sz in ranked[:4])
            return (f"If {label!r} happens, my causal model predicts: {listing}."), \
                0.75, True, f"do-propagation over {len(ranked)} discovered links"

        if intent == "counterfactual":
            target = self._counterfactual_target(part, label)
            if target is None:
                return None
            # ground it in what actually happened, not just the population average:
            # abduction (recover THIS episode's realized noise) -> action -> prediction.
            evidence = cwm.latest_evidence(cwm.labels())
            cf = cwm.counterfactual(label, target, factual=1.0, counter=0.0,
                                    evidence=evidence)
            if cf.abducted:
                self._step(steps, StepKind.SIMULATE,
                           f"ran a STRUCTURAL counterfactual for {label!r} on {target!r}: "
                           f"abducted this episode's realized state, then do({label}=absent), "
                           f"then re-predicted forward — not a population average",
                           detail=cf.describe(), confidence=0.75)
            else:
                self._step(steps, StepKind.SIMULATE,
                           f"ran the counterfactual do({label}=absent) on {target!r} "
                           f"(population-average — no fitted mechanism to abduct from yet)",
                           detail=cf.describe(), confidence=0.6)
            source = ("structural counterfactual (abduction → action → prediction) over "
                      "this episode's own state" if cf.abducted else
                      "population-average interventional contrast over the discovered graph")
            return (f"Counterfactual: {cf.describe()}."), (0.75 if cf.abducted else 0.6), True, \
                f"{source} (Δ={cf.delta:+.3f})"
        return None

    def _causal_chain_to(self, effect: str, *, max_depth: int = 4) -> List[str]:
        """Walk the strongest cause-of-cause chain back from ``effect`` (root → effect)."""
        chain = [effect]
        seen = {effect}
        cur = effect
        for _ in range(max_depth):
            links = self.causal.why(cur)
            links = [l for l in links if l.cause not in seen]
            if not links:
                break
            cur = links[0].cause
            seen.add(cur)
            chain.append(cur)
        chain.reverse()
        return chain

    def _counterfactual_target(self, part: str, cause: str) -> Optional[str]:
        """The target of a counterfactual: a second label in the text, else the
        strongest known downstream effect of the cause."""
        rest = re.sub(re.escape(cause.replace("_", " ")), " ", part, flags=re.I)
        other = self.causal.match_label(rest)
        if other and other != cause:
            return other
        effects = self.causal.effects_of(cause)
        if effects:
            return effects[0].effect
        preds = self.causal.predict_effects(cause)
        if isinstance(preds, dict) and preds:
            return max(preds.items(), key=lambda kv: abs(kv[1]))[0]
        return None

    # ------------------------------------------------------------------ #
    # engine: intervention planning — the do-operator used prescriptively
    # ------------------------------------------------------------------ #
    def _intervention_answer(self, part: str, intent: str, steps: List[ReasoningStep]
                             ) -> Optional[Tuple[str, float, bool, str]]:
        if self.causal is None or not self._flag("intervention_planning"):
            return None
        m = _RE_INTERVENTION.search(part)
        if not m:
            return None
        verb = m.group(1).lower()
        prevent = verb in ("prevent", "stop", "avoid", "reduce")
        rest = part[m.end():]

        # a multi-goal ask ("how do I achieve X and Y") — try each conjunct as its own
        # goal label before falling back to matching the whole remainder as one label.
        segments = [s for s in re.split(r"\band\b|,", rest) if s.strip()]
        goals: List[str] = []
        for seg in (segments if len(segments) > 1 else []):
            g = self.causal.match_label(seg)
            if g is not None and g not in goals:
                goals.append(g)
        if len(goals) < 2:
            goal = self.causal.match_label(rest)
            if goal is None:
                return None
            goals = [goal]

        per_goal: List[Tuple[str, str, float, List[str]]] = []   # (goal, lever, strength, chain)
        for goal in goals:
            paths = self._causal_ancestor_paths(goal)
            if not paths:
                continue
            paths.sort(key=lambda p: p[1], reverse=True)
            lever, strength, chain = paths[0]
            per_goal.append((goal, lever, strength, chain))

        if not per_goal:
            self._step(steps, StepKind.REASON,
                       f"no known causal lever for {' / '.join(goals)} in my graph")
            return (f"I know of no established cause I could act on to "
                    f"{'prevent' if prevent else 'bring about'} {' and '.join(goals)} "
                    f"yet."), 0.5, True, "honest no-lever verdict from the causal graph"

        if len(per_goal) == 1 or len({lever for _, lever, _, _ in per_goal}) == 1:
            goal, lever, strength, chain = per_goal[0]
            self._step(steps, StepKind.REASON,
                       f"searched my causal graph backward from {goal!r}: best lever "
                       f"is {lever!r}", detail=" → ".join(chain),
                       confidence=_clamp(strength))
            action = ("suppress" if prevent else "bring about")
            answer = (f"To {'prevent' if prevent else 'achieve'} {goal!r}, {action} "
                      f"{lever!r} — my causal model shows {' → '.join(chain)} "
                      f"(path confidence {strength:.0%}).")
            return answer, _clamp(max(strength, 0.45)), True, \
                f"interventional plan over discovered causal path {' → '.join(chain)}"

        # genuinely distinct levers for distinct goals — check for interaction (do they
        # reinforce or conflict when pulled TOGETHER?) via the multi-variable do-operator,
        # rather than just reporting two independent single-lever plans.
        interventions = {lever: 1.0 for _, lever, _, _ in per_goal}
        lines = []
        for goal, lever, strength, chain in per_goal:
            joint = self.causal.do(interventions, target=goal)
            solo = self.causal.predict_effects(lever, target=goal)
            note = ("reinforcing" if abs(joint) > abs(solo) + 1e-9 else
                    "no extra conflict" if abs(joint - solo) < 1e-9 else "some conflict")
            lines.append(f"{goal!r} via {lever!r} (path {' → '.join(chain)}, "
                         f"joint effect {joint:+.2f} vs. acting alone {solo:+.2f}: {note})")
        self._step(steps, StepKind.SIMULATE,
                   f"joint do({', '.join(interventions)}) checked for cross-goal interaction",
                   evidence=lines[:4])
        avg_strength = sum(s for _, _, s, _ in per_goal) / len(per_goal)
        answer = ("To " + ("prevent" if prevent else "achieve") + " both: " +
                  "; ".join(lines) + ".")
        return answer, _clamp(max(avg_strength, 0.45)), True, \
            f"multi-variable do-operator interaction check over {len(per_goal)} goals"

    def _causal_ancestor_paths(self, goal: str, *, max_depth: int = 4
                               ) -> List[Tuple[str, float, List[str]]]:
        """All ancestor levers of ``goal``: (lever, path confidence, lever→goal chain)."""
        out: List[Tuple[str, float, List[str]]] = []

        def _walk(node: str, conf: float, chain: List[str], depth: int) -> None:
            if depth >= max_depth:
                return
            for link in self.causal.why(node):
                if link.cause in chain:
                    continue
                c = conf * link.confidence
                path = [link.cause] + chain
                out.append((link.cause, c, path))
                _walk(link.cause, c, path, depth + 1)

        _walk(goal, 1.0, [goal], 0)
        return out

    # ------------------------------------------------------------------ #
    # engine: verifiable chains & first principles
    # ------------------------------------------------------------------ #
    def _chain_answer(self, part: str, intent: str, steps: List[ReasoningStep]
                      ) -> Optional[Tuple[str, float, bool, str]]:
        try:
            from nyxara.mind.reasoning_chain import solve_chain
            chain = solve_chain(part)
        except Exception:  # noqa: BLE001
            chain = None
        if chain is not None:
            for s in chain.steps:
                self._step(steps, StepKind.REASON, s)
            self._step(steps, StepKind.GATE, "each step exactly computed (no guessing)",
                       confidence=chain.confidence)
            return chain.answer, chain.confidence, chain.confidence >= 0.9, \
                "multi-step exact derivation"
        try:
            from nyxara.mind.reasoning_faculties import solve_verifiable
            fac = solve_verifiable(part)
        except Exception:  # noqa: BLE001
            fac = None
        if fac is not None:
            answer, conf = fac
            self._step(steps, StepKind.REASON,
                       "solved with a verifiable faculty (exact computation)",
                       detail=str(answer)[:100], confidence=conf)
            return str(answer), float(conf), conf >= 0.9, "verifiable faculty (exact)"
        try:
            from nyxara.mind.first_principles import derive_from_first_principles
            der = derive_from_first_principles(part)
        except Exception:  # noqa: BLE001
            der = None
        if der is not None and der.result:
            for s in der.steps[:6]:
                self._step(steps, StepKind.REASON, s.expression,
                           detail=f"{s.rule}: {s.justification}".strip(": "))
            self._step(steps, StepKind.GATE,
                       f"derived from {der.domain} first principles "
                       f"(verified={der.verified})", confidence=der.confidence)
            return der.result, der.confidence, der.verified, \
                f"first-principles derivation ({der.domain}, {len(der.steps)} steps)"
        return None

    # ------------------------------------------------------------------ #
    # engine: compute-and-verify (reduce → run → certify)
    # ------------------------------------------------------------------ #
    def _compute_answer(self, part: str, intent: str, steps: List[ReasoningStep]
                        ) -> Optional[Tuple[str, float, bool, str]]:
        if self.compute is None:
            return None
        res = self.compute.solve(part)
        if res is None:
            return None
        self._step(steps, StepKind.REASON,
                   f"reduced the problem to a computation and ran it ({res.method})",
                   detail=str(res.answer)[:100], confidence=res.confidence)
        if res.certificate:
            self._step(steps, StepKind.GATE, "certified the result",
                       detail=res.certificate[:140])
        return str(res.answer), float(res.confidence), bool(res.verified), \
            res.certificate or res.method

    # ------------------------------------------------------------------ #
    # engine: the knowledge graph (provenanced facts, multi-hop paths)
    # ------------------------------------------------------------------ #
    def _graph_answer(self, part: str, intent: str, steps: List[ReasoningStep]
                      ) -> Optional[Tuple[str, float, bool, str]]:
        if self.graph is None or not self._flag("graph_answers"):
            return None
        ans = self.graph.ask(part)
        if (ans.interpretation == "unparsed" or not ans.text
                or ans.confidence < 0.45 or ans.text.startswith("No known connection")
                or " but no relations" in ans.text):
            return None
        evidence = [f"{t.subject} -[{t.predicate}]-> {t.object}"
                    for t in ans.triples[:4]]
        self._step(steps, StepKind.RECALL,
                   f"answered from my knowledge graph ({ans.interpretation})",
                   detail=ans.text[:120], confidence=ans.confidence,
                   evidence=evidence)
        for p in ans.paths[:2]:
            self._step(steps, StepKind.REASON,
                       "multi-hop path: " + " → ".join(p.nodes),
                       confidence=p.score)
        return ans.text, float(ans.confidence), False, \
            f"knowledge-graph provenance ({len(ans.triples)} triples)"

    # ------------------------------------------------------------------ #
    # engine: analogy by structure mapping over graph neighborhoods
    # ------------------------------------------------------------------ #
    def _analogy_answer(self, part: str, intent: str, steps: List[ReasoningStep]
                        ) -> Optional[Tuple[str, float, bool, str]]:
        if self.graph is None or not self._flag("analogy"):
            return None
        m = re.search(r"\b(?:how\s+)?(?:is\s+)?(.+?)\s+(?:is\s+)?like\s+(.+?)\s*[?.]*\s*$",
                      part, flags=re.I)
        if not m:
            return None
        try:
            from nyxara.mind.analogy import StructureMapper, entity, relation
        except Exception:  # noqa: BLE001
            return None
        base_rels = self._graph_relations(m.group(2), entity, relation)
        target_rels = self._graph_relations(m.group(1), entity, relation)
        if len(base_rels) < 2 or len(target_rels) < 2:
            return None
        result = StructureMapper(analogical=True).map(base_rels, target_rels)
        if result.structural_score < 0.5 or not result.relation_matches:
            return None
        pairs = [f"{c.base}↔{c.target}" for c in result.relation_matches[:4]]
        self._step(steps, StepKind.REASON,
                   f"mapped the structure of {m.group(2)!r} onto {m.group(1)!r} "
                   f"(score {result.structural_score:.2f})", evidence=pairs)
        mapping = ", ".join(f"{b} plays the role of {t}"
                            for b, t in list(result.entity_mapping.items())[:3])
        answer = (f"{m.group(1).strip()} is like {m.group(2).strip()} structurally: "
                  f"{mapping}.")
        conf = _clamp(0.4 + 0.4 * result.structural_score, 0.0, 0.85)
        return answer, conf, False, \
            f"structure mapping (score {result.structural_score:.2f}, " \
            f"{len(result.relation_matches)} correspondences)"

    def _graph_relations(self, name: str, entity: Any, relation: Any) -> List[Any]:
        """The knowledge-graph neighborhood of ``name`` as structure-mapper predicates."""
        eid = self.graph.resolve(name.strip())
        if eid is None:
            return []
        out = []
        for t in self.graph.triples_of(eid, direction="out"):
            out.append(relation(t.predicate, entity(t.subject), entity(t.object)))
        return out

    # ------------------------------------------------------------------ #
    # verified-answer cache + promotion (the compounding knowledge loop)
    # ------------------------------------------------------------------ #
    def _cached_answer(self, part: str) -> Optional[Tuple[str, float]]:
        if self.graph is None or not self._flag("promote_knowledge"):
            return None
        key = _norm_key(part)
        if not key:
            return None
        try:
            from nyxara.memory.graph import GraphQuery
            hits = self.graph.match(GraphQuery(subject=key,
                                               predicate="has_verified_answer"))
        except Exception:  # noqa: BLE001
            return None
        # only genuinely verified work may replay: her own certified compute, or a triple a
        # verifier actually proved. Teacher-inferred distillations (an LLM's best guess baked
        # by epistemic distillation) are knowledge, not certainty — replaying one as VERIFIED
        # freezes a possibly-wrong answer forever, so they are filtered out here (which also
        # neutralizes any such triples already persisted by earlier versions).
        hits = [t for t in hits if self._replayable(t)]
        if not hits:
            return None
        best = max(hits, key=lambda t: t.confidence)
        try:
            answer = self.graph.get_entity(best.object).name
        except Exception:  # noqa: BLE001
            return None
        return answer, float(best.confidence)

    @staticmethod
    def _replayable(triple: Any) -> bool:
        """True iff this ``has_verified_answer`` triple was genuinely verified."""
        attrs = getattr(triple, "attributes", None) or {}
        if attrs.get("source") == "native_reasoner":     # her own certified compute (_promote)
            return True
        if attrs.get("verdict") == "proven":             # a verifier actually proved it
            return True
        method = getattr(getattr(triple, "provenance", None), "method", "")
        return method == "verified"

    def _promote(self, conclusion: NativeConclusion) -> None:
        """Write a VERIFIED conclusion back into the knowledge graph (bounded)."""
        if (self.graph is None or not self._flag("promote_knowledge")
                or not conclusion.verified or conclusion.confidence < 0.7):
            return
        budget = int(getattr(self.cfg, "max_promotions_per_turn", 3))
        if budget <= 0:
            return
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            prov = Provenance(source=SourceType.SELF_REFLECTION,
                              confidence=conclusion.confidence, method="inferred",
                              detail="native_reasoner")
        except Exception:  # noqa: BLE001
            prov = None
        try:
            if conclusion.intent in ("causal_why", "causal_pair") and self.causal is not None:
                # causal knowledge becomes graph-queryable → multi-hop causal chains
                for link in self.causal.links()[:budget]:
                    self.graph.add_triple(link.cause, "causes", link.effect,
                                          confidence=_clamp(link.confidence),
                                          provenance=prov,
                                          attributes={"source": "causal_world_model"})
            elif conclusion.intent == "computational" and len(conclusion.answer) <= 120:
                # her own certified compute answers repeat questions instantly
                key = _norm_key(conclusion.steps[0].detail
                                if conclusion.steps else conclusion.answer)
                if key:
                    self.graph.add_triple(key, "has_verified_answer", conclusion.answer,
                                          confidence=conclusion.confidence,
                                          provenance=prov,
                                          attributes={"source": "native_reasoner"})
        except Exception:  # noqa: BLE001 — promotion is additive, never fatal
            pass

    # ------------------------------------------------------------------ #
    # lived outcomes: replay log, bandit, calibration (she learns from life)
    # ------------------------------------------------------------------ #
    def _log_reasoning(self, conclusion: NativeConclusion, engine: str) -> None:
        self._flush_pending()
        self._pending = {"t": round(time.time(), 2), "intent": conclusion.intent,
                         "engine": engine,
                         "confidence": round(conclusion.confidence, 4),
                         "verified": conclusion.verified, "outcome": None}

    def record_outcome(self, success: bool) -> None:
        """Feed the lived outcome of the last native answer back into her learning."""
        if self._pending is None:
            return
        self._pending["outcome"] = bool(success)
        intent = self._pending["intent"]
        if self.calibration is not None:
            self.calibration.record(intent, self._pending["confidence"], bool(success))
        if self.bandit is not None:
            self.bandit.record(intent, self._pending["engine"], bool(success))
        self._flush_pending()

    def _flush_pending(self) -> None:
        if self._pending is None or not self.replay_path:
            self._pending = None
            return
        try:
            os.makedirs(os.path.dirname(self.replay_path) or ".", exist_ok=True)
            with open(self.replay_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(self._pending) + "\n")
            self._flushes += 1
            cap = int(getattr(self.cfg, "replay_log_cap", 2000))
            if self._flushes % 200 == 0:
                _truncate_jsonl(self.replay_path, cap)
        except Exception:  # noqa: BLE001
            pass
        self._pending = None

    def self_improve(self) -> Dict[str, Any]:
        """Run one measure→propose→prove→apply cycle on her own reasoning parameters."""
        if not self._flag("self_improve") or not self.replay_path:
            return {"applied": False, "reason": "self-improvement disabled"}
        improver = NativeReasoningImprover(replay_path=self.replay_path,
                                           tuning_path=self.tuning_path)
        report = improver.improve()
        if report.get("applied"):
            self.reload_tuning()
        return report

    def report(self) -> Dict[str, Any]:
        return {"tuning": dict(self._tuning),
                "bandit": dict(self.bandit._arms) if self.bandit else {},
                "replay_log": self.replay_path or ""}

    # ------------------------------------------------------------------ #
    # grounded oracle (import-guarded)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ground_truth(text: str) -> Optional[str]:
        try:
            from nyxara.mind.grounded_verifier import ground_truth
            return ground_truth(text)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _carries_truth(ground: str, answer: str) -> bool:
        try:
            from nyxara.mind.grounded_verifier import answer_carries_truth
            return answer_carries_truth(ground, answer)
        except Exception:  # noqa: BLE001
            return True


def _truncate_jsonl(path: str, cap: int) -> None:
    """Keep only the newest ``cap`` lines of a jsonl file (bounded replay log)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        if len(lines) <= cap:
            return
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.writelines(lines[-cap:])
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Self-improvement — she measures, mutates, PROVES better, then applies
# --------------------------------------------------------------------------- #
class NativeReasoningImprover:
    """Propose small parameter mutations and accept ONLY what replays better.

    The proof-carrying-improvement contract (growth/proof_carrying.py, applied to
    her own reasoner): each candidate abstention floor is scored by replaying her
    RECORDED reasoning outcomes — answers she gave that turned out right earn, wrong
    answers cost double, and abstentions are near-neutral — and a mutation is written
    to the tuning file only when it strictly beats the current parameters. Hard
    bounds keep every tunable inside a safe envelope; deleting the tuning file
    restores config defaults (fully reversible)."""

    def __init__(self, *, replay_path: str, tuning_path: Optional[str],
                 min_evidence: int = 20) -> None:
        self.replay_path = replay_path
        self.tuning_path = tuning_path
        self.min_evidence = int(min_evidence)

    def _records(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not self.replay_path or not os.path.exists(self.replay_path):
            return out
        try:
            with open(self.replay_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    if rec.get("outcome") is not None:
                        out.append(rec)
        except Exception:  # noqa: BLE001
            return []
        return out

    @staticmethod
    def score(records: Iterable[Dict[str, Any]], floor: float) -> float:
        """Replay utility of a floor: right answers earn, wrong answers cost double,
        abstaining is near-neutral (slightly bad when the answer would have been right)."""
        total = 0.0
        for rec in records:
            conf = float(rec.get("confidence", 0.0))
            correct = bool(rec.get("outcome"))
            if conf >= floor:
                total += 1.0 if correct else -2.0
            else:
                total += -0.25 if correct else 0.25
        return total

    def _current(self) -> Dict[str, Any]:
        if self.tuning_path and os.path.exists(self.tuning_path):
            try:
                with open(self.tuning_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:  # noqa: BLE001
                pass
        return {}

    def improve(self) -> Dict[str, Any]:
        records = self._records()
        if len(records) < self.min_evidence:
            return {"applied": False,
                    "reason": f"too little lived evidence "
                              f"({len(records)}/{self.min_evidence} outcomes)"}
        tuning = self._current()
        lo, hi = _FLOOR_BOUNDS
        current = _clamp(float(tuning.get("min_confidence", 0.35)), lo, hi)
        base_score = self.score(records, current)
        # grid-search the whole safe envelope — the replay proof gate is the protection,
        # and the hard bounds are the envelope, so the search may be exhaustive
        grid = [x / 100.0 for x in range(int(lo * 100), int(hi * 100) + 1, 5)]
        best, best_score = current, base_score
        for cand in grid:
            s = self.score(records, cand)
            if s > best_score:
                best, best_score = cand, s

        # per-intent floors where an intent has enough of its own evidence
        intent_floors: Dict[str, float] = {}
        by_intent: Dict[str, List[Dict[str, Any]]] = {}
        for rec in records:
            by_intent.setdefault(str(rec.get("intent", "chat")), []).append(rec)
        for intent, recs in by_intent.items():
            if len(recs) < 12:
                continue
            ibest, ibest_score = best, self.score(recs, best)
            for cand in grid:
                s = self.score(recs, cand)
                if s > ibest_score:
                    ibest, ibest_score = cand, s
            if ibest != best:
                intent_floors[intent] = ibest

        changed = (best != current) or (intent_floors != tuning.get("intent_floors", {}))
        if not changed:
            return {"applied": False, "reason": "current parameters already replay best",
                    "score": base_score, "evidence": len(records)}
        new_tuning = dict(tuning)
        new_tuning["min_confidence"] = best
        if intent_floors:
            new_tuning["intent_floors"] = intent_floors
        else:
            new_tuning.pop("intent_floors", None)
        if self.tuning_path:
            try:
                os.makedirs(os.path.dirname(self.tuning_path) or ".", exist_ok=True)
                tmp = self.tuning_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(new_tuning, fh)
                os.replace(tmp, self.tuning_path)
            except Exception:  # noqa: BLE001
                return {"applied": False, "reason": "could not persist tuning"}
        return {"applied": True, "min_confidence": best,
                "intent_floors": intent_floors,
                "score_before": base_score, "score_after": best_score,
                "evidence": len(records)}


# --------------------------------------------------------------------------- #
# self-test / demo — her own reasoning, no LLM anywhere
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import random
    import tempfile

    from nyxara.mind.causal_world_model import CausalWorldModel

    rng = random.Random(7)
    cwm = CausalWorldModel(window=10.0)
    for k in range(300):
        base = k * 100.0
        if rng.random() < 0.5:
            cwm.observe("rain", at=base)
            if rng.random() < 0.9:
                cwm.observe("wet_ground", at=base + 1)
                if rng.random() < 0.85:
                    cwm.observe("slippery", at=base + 2)
    cwm.discover()

    try:
        from nyxara.memory.graph import KnowledgeGraph
        kg = KnowledgeGraph()
        kg.add_triple("NYXARA", "owned_by", "Jaypal")
        kg.add_triple("Jaypal", "created", "NYXARA")
    except Exception:  # noqa: BLE001
        kg = None

    with tempfile.TemporaryDirectory() as tmp:
        nr = NativeReasoner(knowledge_graph=kg, causal_model=cwm, data_dir=tmp)
        for q in ("why is the ground slippery?",
                  "does rain cause slippery?",
                  "what happens if rain?",
                  "how can I prevent slippery?",
                  "why is the ground slippery and how can I prevent it?",
                  "who is the owner of NYXARA?",
                  "how are you today?"):
            c = nr.reason(q)
            print(f"\nQ: {q}")
            if c is None:
                print("  → (honest abstain)")
            else:
                print("  →", c.render().replace("\n", "\n  "))

        # the self-improvement loop, proven on synthetic lived outcomes
        for i in range(30):
            nr._pending = {"t": 0, "intent": "factual", "engine": "graph",
                           "confidence": 0.5, "verified": False,
                           "outcome": i % 4 == 0}   # mostly wrong at conf 0.5
            nr._flush_pending()
        print("\nself-improve:", nr.self_improve())
    print("\nNATIVE REASONER SELF-TEST DONE ✓")
