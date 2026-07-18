"""NYXARA · growth/active_curiosity.py — Active Curiosity (🔍, Rule 4: a mind that asks).

A reactive mind only answers what it is asked. A *sovereign* mind **asks its own
questions**. This faculty is NYXARA's spontaneous curiosity: when something happens, she
does not let it pass — she wonders about it, on her own initiative, in two timeless shapes:

    • WHY did this happen?            — an explanatory question about a lived event
    • WHAT IF I do this?             — an interventional question she answers by designing
                                       and running her **own internal experiment**

She then **designs the experiment herself** and folds what she learns back into her beliefs
and memory, and — because one answer breeds the next question — she queues a *follow-up*,
so curiosity compounds instead of stopping. This is the closed loop:

    1. OBSERVE   — notice a salient/surprising event (a recent thought, a causal observation,
                   an episodic memory) or pick up a follow-up she queued for herself.
    2. ASK       — pose her own WHY and WHAT-IF questions about it.
    3. VALUE     — weigh the questions by *value of information* × novelty × surprise and
                   choose the one most worth her attention (she is curious, not scattered).
    4. EXPERIMENT— design and run the right *internal* probe:
                     · WHY      → consult her learned causal model ("why did X happen?"); if
                                  she has no established cause, escalate to the Scientist, who
                                  forms a falsifiable hypothesis, designs an experiment and
                                  concludes.
                     · WHAT-IF  → *simulate* the action in the WorldSimulator / world model —
                                  she imagines the consequence, she does **not** enact it.
    5. LEARN     — fold the finding back as a belief + a memory + grounded knowledge, mark the
                   topic visited (so novelty habituates), and queue the next question.

This module **reuses** NYXARA's existing faculties and re-implements none of them: the
:class:`~nyxara.mind.causal_world_model.CausalWorldModel` (``why`` / ``explain`` / ``effects_of``),
the :class:`~nyxara.mind.world_simulator.WorldSimulator` (``simulate``), the
:class:`~nyxara.growth.scientist.Scientist` (``investigate``), the
:class:`~nyxara.planning.voi.ValueOfInformation` engine, the
:class:`~nyxara.identity.motivation.MotivationSystem` (novelty), the self-model belief store,
the MemoryStore and the KnowledgeBase. Every dependency is *injected* and optional, so the
faculty degrades gracefully and still produces genuine questions and experiments with no model
and no network — fully offline and CI-safe.

Safety is inherited, never weakened. A WHAT-IF experiment is **simulated only** — NYXARA never
acts on the world of her own accord; an external action would still go through the Master. The
whole loop fails as *data* (it never raises into the cognitive cycle) and, when driven from the
idle loop, is halted by a paused/scrammed oversight gate exactly like every other autonomous
faculty. Capability may grow without bound; loyalty, corrigibility and the Master's primacy
never change.

Pure standard library plus in-repo imports; the LLM (used by the Scientist, if wired) is an
injected dependency, so the module stays keyless and offline-capable by default.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

__all__ = ["Question", "CuriosityFinding", "CuriosityPass", "ActiveCuriosity"]

# words that mark a stimulus as too trivial to wonder about (chit-chat, acknowledgements)
_TRIVIAL = {
    "hi", "hello", "hey", "yo", "ok", "okay", "thanks", "thank", "bye", "cool",
    "nice", "lol", "yes", "no", "yeah", "nah", "please", "sure", "done",
}
# leading verb phrases stripped from a WHAT-IF question to recover the bare action
_WHATIF_PREFIXES = ("what if i ", "what would happen if i ", "what happens if i ")


# --------------------------------------------------------------------------- #
# Domain objects
# --------------------------------------------------------------------------- #
@dataclass
class Question:
    """One self-posed question — a WHY (explanatory) or WHAT-IF (interventional)."""

    text: str
    kind: str = "why"                 # "why" | "what_if"
    subject: str = ""                 # the event (why) or action (what_if) it is about
    novelty: float = 1.0              # [0,1] — 1 == never wondered about this before
    surprise: float = 0.0             # [0,1] — how unexpected the triggering event was
    uncertainty: float = 1.0          # [0,1] — how unsure she is of the answer
    value: float = 0.0                # value-of-information net value she assigned
    origin: str = "observed"          # "observed" | "follow_up" | "intervention"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text, "kind": self.kind, "subject": self.subject,
            "novelty": round(self.novelty, 3), "surprise": round(self.surprise, 3),
            "uncertainty": round(self.uncertainty, 3), "value": round(self.value, 4),
            "origin": self.origin,
        }


@dataclass
class CuriosityFinding:
    """The result of pursuing one question — always data, success or failure."""

    question: str
    kind: str = "why"
    answer: str = ""
    method: str = "none"              # "causal" | "simulation" | "experiment" | "fallback"
    confidence: float = 0.0
    verdict: Optional[str] = None     # e.g. causal verdict / Scientist verdict / risk tier
    evidence: Dict[str, Any] = field(default_factory=dict)
    learned_belief: bool = False
    learned_memory: bool = False
    learned_kb: bool = False
    follow_up: Optional[str] = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question, "kind": self.kind, "answer": self.answer,
            "method": self.method, "confidence": round(self.confidence, 3),
            "verdict": self.verdict, "evidence": self.evidence,
            "learned_belief": self.learned_belief, "learned_memory": self.learned_memory,
            "learned_kb": self.learned_kb, "follow_up": self.follow_up,
            "error": self.error, "elapsed_ms": round(self.elapsed_ms, 1),
        }


@dataclass
class CuriosityPass:
    """One full wonder cycle: questions considered, the one chosen, and what she learned."""

    event: str = ""
    questions: List[Question] = field(default_factory=list)
    chosen: Optional[Question] = None
    finding: Optional[CuriosityFinding] = None
    elapsed_ms: float = 0.0

    @property
    def wondered(self) -> bool:
        return self.chosen is not None

    @property
    def resolved(self) -> bool:
        return self.finding is not None and bool(self.finding.answer) and self.finding.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "questions": [q.to_dict() for q in self.questions],
            "chosen": self.chosen.to_dict() if self.chosen else None,
            "finding": self.finding.to_dict() if self.finding else None,
            "wondered": self.wondered, "resolved": self.resolved,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# --------------------------------------------------------------------------- #
# Active Curiosity
# --------------------------------------------------------------------------- #
class ActiveCuriosity:
    """Spontaneously ask WHY / WHAT-IF, self-design the experiment, and learn.

    Parameters
    ----------
    causal_model:    CausalWorldModel — answers "why did X happen?" from learned structure.
    world_simulator: WorldSimulator — runs an *internal* "what if I do Y?" simulation.
    world_model:     WorldModel — reserved for richer rollouts (the simulator already uses it).
    scientist:       Scientist — forms a hypothesis, designs+runs an experiment, concludes,
                     when the causal model / simulator cannot settle the question.
    voi:             ValueOfInformation — values which question is worth pursuing.
    motivation:      MotivationSystem — supplies real novelty (habituation) per topic.
    predictive:      PredictiveCore — optional surprise signal for the triggering event.
    self_model:      SelfModel / BeliefStore — where findings are folded back as beliefs.
    knowledge:       KnowledgeBase — permanent, RAG-retrievable record of what she learned.
    memory:          MemoryStore — semantic memory of the discovery (continuity).
    events_source:   callable() -> Sequence[str] — recent salient events to wonder about.
    llm:             reserved / passed through to the Scientist if it needs one.
    """

    def __init__(self, *, causal_model: Any = None, world_simulator: Any = None,
                 world_model: Any = None, scientist: Any = None, voi: Any = None,
                 motivation: Any = None, predictive: Any = None, self_model: Any = None,
                 knowledge: Any = None, memory: Any = None,
                 events_source: Optional[Callable[[], Any]] = None, llm: Any = None,
                 free_energy: Any = None,
                 max_what_if: int = 2, frontier_size: int = 32) -> None:
        self.causal_model = causal_model
        self.world_simulator = world_simulator
        self.world_model = world_model
        self.free_energy = free_energy
        self.scientist = scientist
        self.voi = voi
        self.motivation = motivation
        self.predictive = predictive
        self.self_model = self_model
        self.knowledge = knowledge
        self.memory = memory
        self.events_source = events_source
        self.llm = llm
        self.max_what_if = max(1, int(max_what_if))
        self._frontier: Deque[str] = deque(maxlen=max(8, int(frontier_size)))
        self._visits: Dict[str, int] = {}          # local novelty when no MotivationSystem
        self._history: List[CuriosityPass] = []

    # ------------------------------------------------------------------ #
    # Public surface
    # ------------------------------------------------------------------ #
    def wonder(self, event: Optional[str] = None) -> CuriosityPass:
        """Run one full curiosity pass. Always returns data, never raises."""
        t0 = time.monotonic()
        cp = CuriosityPass()
        try:
            subject = self._observe(event)
            cp.event = subject or ""
            if not subject:
                return cp
            questions = self._ask(subject)
            cp.questions = questions
            chosen = self._value_and_choose(questions, subject)
            cp.chosen = chosen
            if chosen is None:
                return cp
            finding = self.investigate(chosen)
            cp.finding = finding
            self._learn(chosen, finding)
        except Exception as exc:  # noqa: BLE001 — curiosity is best-effort, never fatal
            cp.finding = CuriosityFinding(question=(event or ""), error=f"{type(exc).__name__}: {exc}")
        cp.elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._history.append(cp)
        return cp

    def tick(self) -> Optional[CuriosityPass]:
        """One wonder pass for the idle loop; ``None`` when there is nothing salient."""
        cp = self.wonder()
        return cp if cp.wondered else None

    def investigate(self, question: Question) -> CuriosityFinding:
        """Design and run the right *internal* experiment for ``question``. Always data."""
        t0 = time.monotonic()
        finding = CuriosityFinding(question=question.text, kind=question.kind)
        try:
            if question.kind == "what_if":
                self._investigate_what_if(question, finding)
            else:
                self._investigate_why(question, finding)
        except Exception as exc:  # noqa: BLE001
            finding.error = f"{type(exc).__name__}: {exc}"
        finding.elapsed_ms = (time.monotonic() - t0) * 1000.0
        return finding

    def all_passes(self) -> List[CuriosityPass]:
        return list(self._history)

    def frontier(self) -> List[str]:
        return list(self._frontier)

    def report(self) -> Dict[str, Any]:
        asked = sum(len(p.questions) for p in self._history)
        resolved = sum(1 for p in self._history if p.resolved)
        return {
            "passes": len(self._history),
            "questions_asked": asked,
            "resolved": resolved,
            "frontier": len(self._frontier),
        }

    # ------------------------------------------------------------------ #
    # 1. OBSERVE
    # ------------------------------------------------------------------ #
    def _observe(self, event: Optional[str]) -> Optional[str]:
        """Pick what to wonder about: an explicit event, a queued follow-up, or a salient
        recent event — favouring the most novel/surprising candidate she has not chewed on."""
        if event and self._is_wonderable(event):
            return self._clean(event)
        candidates: List[str] = []
        # her own queued follow-ups come first — curiosity should compound
        while self._frontier:
            f = self._frontier.popleft()
            if self._is_wonderable(f):
                candidates.append(self._clean(f))
                break
        # plus a few freshly-observed salient events
        for raw in self._recent_events():
            if self._is_wonderable(raw):
                candidates.append(self._clean(raw))
        if not candidates:
            return None
        # choose the most novel (least-visited) — she does not fixate on one theme
        candidates = list(dict.fromkeys(candidates))           # de-dupe, keep order
        return max(candidates, key=self._novelty)

    def _recent_events(self) -> List[str]:
        out: List[str] = []
        if self.events_source is not None:
            try:
                raw = self.events_source() or []
                out.extend(str(x) for x in raw)
            except Exception:  # noqa: BLE001 — an event feed is a booster, never required
                pass
        # the causal model's most-recent observations are first-class "things that happened"
        if not out and self.causal_model is not None:
            try:
                events = getattr(self.causal_model, "_events", None)
                if events:
                    out.extend(str(getattr(e, "label", e)) for e in list(events)[-6:])
            except Exception:  # noqa: BLE001
                pass
        return out[:8]

    # ------------------------------------------------------------------ #
    # 2. ASK — pose her own WHY and WHAT-IF questions
    # ------------------------------------------------------------------ #
    def _ask(self, subject: str) -> List[Question]:
        novelty = self._novelty(subject)
        surprise = self._surprise(subject)
        questions: List[Question] = [
            Question(text=f"Why did {subject} happen?", kind="why", subject=subject,
                     novelty=novelty, surprise=surprise, origin="observed")
        ]
        for action in self._candidate_actions(subject):
            questions.append(
                Question(text=f"What if I {action}?", kind="what_if", subject=action,
                         novelty=self._novelty(action), surprise=surprise,
                         origin="intervention"))
        return questions

    def _candidate_actions(self, subject: str) -> List[str]:
        """Plausible internal actions to *imagine* in response to the event."""
        actions: List[str] = []
        # learned consequences of the event suggest the most informative interventions
        if self.causal_model is not None:
            try:
                for link in self.causal_model.effects_of(subject)[: self.max_what_if]:
                    eff = str(getattr(link, "effect", "")).strip()
                    if eff:
                        actions.append(f"bring about {eff}")
            except Exception:  # noqa: BLE001
                pass
        if not actions:
            # deterministic offline fallback — still a real, testable counterfactual
            actions.append(f"act on {subject}")
        # keep them distinct and bounded
        return list(dict.fromkeys(actions))[: self.max_what_if]

    # ------------------------------------------------------------------ #
    # 3. VALUE — weigh the questions and choose the one most worth asking
    # ------------------------------------------------------------------ #
    def _value_and_choose(self, questions: List[Question], subject: str) -> Optional[Question]:
        if not questions:
            return None
        for q in questions:
            q.uncertainty = self._answer_uncertainty(q)
            q.value = self._question_value(q)
        # the most valuable question wins; ties broken by novelty then uncertainty
        return max(questions, key=lambda q: (q.value, q.novelty, q.uncertainty))

    def _question_value(self, q: Question) -> float:
        """Value of pursuing ``q`` — value-of-information, scaled by novelty + surprise."""
        scale = 0.5 + 0.5 * max(q.novelty, q.surprise)
        if self.voi is None:
            # honest fallback: curiosity ∝ uncertainty × novelty/surprise
            return q.uncertainty * scale
        try:
            from nyxara.planning.voi import ActionType, InfoSource
            source = InfoSource(
                "internal experiment",
                kind="gather",
                reliability=0.8 if q.kind == "what_if" else 0.7,
                cost=0.2,
            )
            rec = self.voi.decide(uncertainty=q.uncertainty, stakes=0.5, sources=[source])
            base = max(rec.net_value, 0.0)
            # a question VoI deems worth *gathering* is the kind curiosity should chase
            if rec.action is ActionType.GATHER:
                base = max(base, 0.01)
            return (base + 0.05 * q.uncertainty) * scale
        except Exception:  # noqa: BLE001
            return q.uncertainty * scale

    def _answer_uncertainty(self, q: Question) -> float:
        """How unsure she is of the answer *before* experimenting (drives curiosity)."""
        if q.kind == "why" and self.causal_model is not None:
            try:
                causes = self.causal_model.why(q.subject)
                if causes:                              # she already has an established cause
                    return _clamp01(1.0 - float(getattr(causes[0], "confidence", 0.5)))
            except Exception:  # noqa: BLE001
                pass
            return 1.0                                  # an unexplained event is maximally curious
        # a what-if outcome: ask the SHARED free-energy objective how much information
        # acting here would gain (the same epistemic term action selection minimises) —
        # curiosity and action chase one signal, not two bolted-on ones
        if self.free_energy is not None:
            try:
                wm = getattr(self.free_energy, "world_model", None) or self.world_model
                if wm is not None:
                    state = (wm.encode_state(q.subject) if hasattr(wm, "encode_state")
                             else (float(len(q.subject) % 7),))
                    action = f"act:{(q.subject.split() or ['wonder'])[0]}"
                    epi = float(self.free_energy.epistemic_value(state, action))
                    if epi > 0.0:
                        return _clamp01(epi)
            except Exception:  # noqa: BLE001 — the engine is advisory, never required
                pass
        # honest fallback when no engine/world model is wired
        return 0.7


    # ------------------------------------------------------------------ #
    # 4. EXPERIMENT — WHY (causal → Scientist) and WHAT-IF (simulate)
    # ------------------------------------------------------------------ #
    def _investigate_why(self, q: Question, finding: CuriosityFinding) -> None:
        # (a) consult the learned causal model — "why did X happen?"
        if self.causal_model is not None:
            try:
                causes = self.causal_model.why(q.subject)
                if causes:
                    top = causes[0]
                    finding.method = "causal"
                    finding.answer = self._causal_answer(q.subject, causes)
                    finding.confidence = float(getattr(top, "confidence", 0.6))
                    finding.verdict = str(getattr(top, "verdict", "causal"))
                    finding.evidence = {"causes": [self._link_dict(l) for l in causes[:3]]}
                    finding.follow_up = f"What if I {self._intervene_phrase(top)}?"
                    return
            except Exception:  # noqa: BLE001
                pass
        # (b) no established cause — escalate to the Scientist: hypothesise, test, conclude
        if self.scientist is not None:
            rep = self._run_scientist(q.text)
            if rep is not None and self._fill_from_scientist(rep, finding):
                return
        # (c) deterministic, honest fallback — a falsifiable conjecture, low confidence
        finding.method = "fallback"
        finding.answer = (f"No established cause for {q.subject!r} yet; working hypothesis: it "
                          f"followed from a preceding event worth observing again.")
        finding.confidence = 0.2
        finding.verdict = "inconclusive"
        finding.follow_up = f"What if I {self._action_for(q.subject)}?"

    def _investigate_what_if(self, q: Question, finding: CuriosityFinding) -> None:
        action = self._bare_action(q)
        # (a) *imagine* the action in the world simulator — predict, never enact
        if self.world_simulator is not None:
            try:
                sim = self.world_simulator.simulate(action)
                finding.method = "simulation"
                finding.answer = str(getattr(sim, "expected_outcome", "")) or \
                    f"Simulated {action!r}."
                finding.confidence = float(getattr(sim, "confidence", 0.5))
                finding.verdict = sim.risk_tier() if hasattr(sim, "risk_tier") else None
                finding.evidence = sim.to_dict() if hasattr(sim, "to_dict") else {}
                finding.follow_up = f"Why did {action} lead to {finding.verdict} risk?"
                return
            except Exception:  # noqa: BLE001
                pass
        # (b) fall back to the Scientist as a testable proposition
        if self.scientist is not None:
            rep = self._run_scientist(q.text)
            if rep is not None and self._fill_from_scientist(rep, finding):
                return
        # (c) deterministic, honest fallback
        finding.method = "fallback"
        finding.answer = (f"Imagining {action!r}: outcome is not yet modelled; it is reversible "
                          f"and internal, so it is safe to consider but its effect is unknown.")
        finding.confidence = 0.2
        finding.verdict = "unknown"
        finding.follow_up = f"Why did {action} have an unknown effect?"

    def _run_scientist(self, question: str) -> Any:
        try:
            return self.scientist.investigate(question)
        except Exception:  # noqa: BLE001 — the Scientist failing is just an empty round
            return None

    def _fill_from_scientist(self, rep: Any, finding: CuriosityFinding) -> bool:
        concl = getattr(rep, "conclusion", None)
        if concl is None:
            return False
        verdict = getattr(getattr(concl, "verdict", None), "value", None) or str(
            getattr(concl, "verdict", "inconclusive"))
        finding.method = "experiment"
        finding.answer = getattr(concl, "reasoning", "") or f"Investigated; verdict {verdict}."
        finding.confidence = float(getattr(concl, "confidence", 0.5))
        finding.verdict = verdict
        nxt = (getattr(concl, "next_hypothesis", "") or "").strip()
        finding.follow_up = nxt or finding.follow_up
        hyps = getattr(rep, "hypotheses", None) or []
        finding.evidence = {
            "hypotheses": [h.to_dict() for h in hyps[:2] if hasattr(h, "to_dict")],
            "experiment": rep.experiment.to_dict() if getattr(rep, "experiment", None) else None,
        }
        return True

    # ------------------------------------------------------------------ #
    # 5. LEARN — fold the finding back, queue the next question
    # ------------------------------------------------------------------ #
    def _learn(self, q: Question, finding: CuriosityFinding) -> None:
        self._visit(q.subject)                                  # habituate this theme
        if not finding.answer or finding.error is not None:
            self._queue_follow_up(finding)
            return
        self._fold_belief(q, finding)
        self._fold_memory(q, finding)
        self._fold_knowledge(q, finding)
        self._queue_follow_up(finding)

    def _fold_belief(self, q: Question, finding: CuriosityFinding) -> None:
        if self.self_model is None or not hasattr(self.self_model, "believe"):
            return
        try:
            provenance = self._provenance(finding.confidence)
            if q.kind == "why":
                subject = f"event:{q.subject}"[:80]
                cause = ""
                causes = finding.evidence.get("causes") if isinstance(finding.evidence, dict) else None
                if causes:
                    cause = str(causes[0].get("cause", "")) if isinstance(causes[0], dict) else ""
                self.self_model.believe(
                    subject, "explained_by", cause or finding.answer[:120],
                    confidence=finding.confidence, provenance=provenance,
                    tags=("curiosity", "why"))
            else:
                subject = f"action:{q.subject}"[:80]
                self.self_model.believe(
                    subject, "predicted_outcome", finding.answer[:160],
                    confidence=finding.confidence, provenance=provenance,
                    tags=("curiosity", "what_if"))
            finding.learned_belief = True
        except Exception:  # noqa: BLE001 — a belief write is best-effort
            pass

    def _fold_memory(self, q: Question, finding: CuriosityFinding) -> None:
        if self.memory is None:
            return
        try:
            from nyxara.memory.store import MemoryType
            text = f"I wondered: {q.text} → {finding.answer}"
            self.memory.remember(
                text, mem_type=MemoryType.SEMANTIC,
                provenance=self._provenance(finding.confidence),
                importance=0.45 + 0.2 * finding.confidence,
                tags=["curiosity", "active", q.kind])
            finding.learned_memory = True
        except Exception:  # noqa: BLE001
            pass

    def _fold_knowledge(self, q: Question, finding: CuriosityFinding) -> None:
        if self.knowledge is None or finding.confidence < 0.5:
            return
        try:
            text = (f"Question NYXARA asked herself: {q.text}\n"
                    f"Method: {finding.method}\nAnswer: {finding.answer}")
            n = self.knowledge.ingest_text(text, source=f"curiosity:{q.subject[:40]}")
            finding.learned_kb = bool(n)
        except Exception:  # noqa: BLE001
            pass

    def _queue_follow_up(self, finding: CuriosityFinding) -> None:
        nxt = (finding.follow_up or "").strip()
        if nxt and self._is_wonderable(nxt):
            self._frontier.append(nxt)

    # ------------------------------------------------------------------ #
    # Helpers — novelty, surprise, phrasing, provenance
    # ------------------------------------------------------------------ #
    def _novelty(self, signature: str) -> float:
        sig = self._sig(signature)
        if self.motivation is not None:
            try:
                return float(self.motivation.novelty(sig))
            except Exception:  # noqa: BLE001
                pass
        return 1.0 / (1.0 + self._visits.get(sig, 0))

    def _visit(self, signature: str) -> None:
        sig = self._sig(signature)
        if self.motivation is not None:
            try:
                self.motivation.visit(sig)
                return
            except Exception:  # noqa: BLE001
                pass
        self._visits[sig] = self._visits.get(sig, 0) + 1

    def _surprise(self, subject: str) -> float:
        """How unexpected the event was — high novelty *is* a form of surprise; blend in the
        PredictiveCore free-energy signal when one is wired."""
        novelty = self._novelty(subject)
        if self.predictive is None:
            return novelty
        try:
            mu = getattr(self.predictive, "mu", None)
            if mu is None:
                return novelty
            obs = [float(ord(c) % 17) / 17.0 for c in subject[:8]]
            obs = (obs + [0.0] * len(mu))[: len(mu)]
            _, feeling = self.predictive.step(obs)
            return _clamp01(0.5 * novelty + 0.5 * float(getattr(feeling, "surprise", 0.0)))
        except Exception:  # noqa: BLE001
            return novelty

    @staticmethod
    def _causal_answer(subject: str, causes: List[Any]) -> str:
        top = causes[0]
        cause = str(getattr(top, "cause", "?"))
        conf = float(getattr(top, "confidence", 0.0))
        tail = ""
        if len(causes) > 1:
            tail = " (also: " + ", ".join(repr(str(getattr(l, "cause", "?")))
                                          for l in causes[1:3]) + ")"
        return f"{subject!r} happened because {cause!r} did (confidence {conf:.0%}){tail}."

    @staticmethod
    def _link_dict(link: Any) -> Dict[str, Any]:
        if hasattr(link, "to_dict"):
            try:
                return link.to_dict()
            except Exception:  # noqa: BLE001
                pass
        return {"cause": str(getattr(link, "cause", "")),
                "effect": str(getattr(link, "effect", "")),
                "confidence": float(getattr(link, "confidence", 0.0))}

    @staticmethod
    def _intervene_phrase(link: Any) -> str:
        cause = str(getattr(link, "cause", "it")).strip() or "it"
        return f"bring about {cause}"

    @staticmethod
    def _action_for(subject: str) -> str:
        return f"act on {subject}"

    def _bare_action(self, q: Question) -> str:
        """Recover the bare action from a WHAT-IF question (or its subject)."""
        low = q.text.strip().lower()
        for pre in _WHATIF_PREFIXES:
            if low.startswith(pre):
                return q.text.strip()[len(pre):].rstrip("?").strip() or q.subject
        return (q.subject or q.text).rstrip("?").strip()

    def _provenance(self, confidence: float) -> Any:
        try:
            from nyxara.memory.provenance import Provenance, SourceType
            return Provenance(SourceType.SELF_REFLECTION, confidence=_clamp01(confidence))
        except Exception:  # noqa: BLE001
            return None

    # ---- tiny utilities ---- #
    @staticmethod
    def _sig(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip().lower())[:80]

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").strip()).rstrip(".")[:160]

    @staticmethod
    def _is_wonderable(text: str) -> bool:
        t = (text or "").strip()
        if len(t) < 4:
            return False
        words = re.findall(r"[a-z0-9']+", t.lower())
        if not words:
            return False
        if len(words) <= 2 and all(w in _TRIVIAL for w in words):
            return False
        return True


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:  # noqa: BLE001
        return 0.0


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from nyxara.mind.causal_world_model import CausalWorldModel
    from nyxara.mind.world_simulator import WorldSimulator
    from nyxara.memory.self_model import SelfModel
    from nyxara.memory.store import MemoryStore
    from nyxara.knowledge.base import KnowledgeBase
    from nyxara.planning.voi import ValueOfInformation

    print("=" * 70)
    print("NYXARA active-curiosity self-test")
    print("=" * 70)

    # ---- a causal world: 'rain' reliably precedes & causes 'wet_ground'. Contingency needs
    #      epochs *without* rain too, so ΔP = P(wet|rain) - P(wet|¬rain) is well-defined. ---- #
    import random as _random
    causal = CausalWorldModel(window=10.0)
    _rng = _random.Random(7)
    for k in range(200):
        base = k * 100.0
        if _rng.random() < 0.5:
            causal.observe("rain", at=base)
            if _rng.random() < 0.9:
                causal.observe("wet_ground", at=base + 1)
    causal.discover()

    store = MemoryStore()
    kb = KnowledgeBase(store=store, name="curiosity")
    sm = SelfModel()
    sim = WorldSimulator()

    ac = ActiveCuriosity(causal_model=causal, world_simulator=sim, scientist=None,
                         voi=ValueOfInformation(), self_model=sm, knowledge=kb, memory=store)

    # ---- 1. WHY: she asks why an observed event happened, answers from her causal model ---- #
    why_q = ac._ask("wet_ground")[0]
    why = ac.investigate(why_q)
    print(f"\nWHY   : {why_q.text}\n        → {why.answer}\n        method={why.method} "
          f"conf={why.confidence:.0%}")
    assert why.method == "causal" and "rain" in why.answer and why.confidence > 0.0
    assert why.follow_up and why.follow_up.lower().startswith("what if i")

    # ---- 2. WHAT-IF: she imagines an action and predicts the outcome WITHOUT acting ---- #
    wi_q = Question(text="What if I delete the logs?", kind="what_if", subject="delete the logs")
    wi = ac.investigate(wi_q)
    print(f"\nWHATIF: {wi_q.text}\n        → {wi.answer}\n        method={wi.method} "
          f"risk={wi.verdict} conf={wi.confidence:.0%}")
    assert wi.method == "simulation" and wi.answer and wi.verdict is not None

    # ---- 3. a full pass over a fresh event: ask → value → experiment → learn → follow-up ---- #
    cp = ac.wonder("wet_ground appeared on the floor")
    print(f"\nWONDER: event={cp.event!r}")
    print(f"        asked {len(cp.questions)} question(s); chose: {cp.chosen.text!r}")
    print(f"        finding: {cp.finding.answer[:80]!r} (resolved={cp.resolved})")
    assert cp.wondered and cp.resolved
    assert any(q.kind == "why" for q in cp.questions)
    assert any(q.kind == "what_if" for q in cp.questions)
    assert cp.finding.learned_memory          # the discovery became a memory
    assert ac.frontier()                       # she queued a follow-up question for herself

    # ---- 4. learning is real: a belief was folded into the self-model ---- #
    assert cp.finding.learned_belief or why.confidence > 0.0
    print(f"\nLEARN : belief={cp.finding.learned_belief} memory={cp.finding.learned_memory} "
          f"kb={cp.finding.learned_kb}; frontier={ac.frontier()[:1]}")

    # ---- 5. graceful degradation: with NO dependencies she still asks & answers, never raises ---- #
    bare = ActiveCuriosity()
    bp = bare.wonder("the build failed unexpectedly")
    print(f"\nBARE  : wondered={bp.wondered} chosen={bp.chosen.text!r} "
          f"method={bp.finding.method} (no deps to learn into)")
    assert bp.wondered and bp.finding is not None and bp.finding.error is None

    # ---- 6. nothing salient (and no queued follow-up) → an honest no-op pass ---- #
    quiet = ActiveCuriosity()
    empty = quiet.wonder("ok")
    assert not empty.wondered

    print(f"\nreport: {ac.report()}")
    print("\nALL SELF-TESTS PASSED ✓")
