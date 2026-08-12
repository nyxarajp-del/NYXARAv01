"""NYXARA · nyx001/brain.py — the NYX V.001 brain facade (🜂, the one object the kernel holds).

:class:`NyxV001Brain` is what ``NyxaraCore`` builds and holds as ``self.nyx001``. It owns all
three minds and makes them one:

* ``self.v03`` — :class:`~nyxara.nyx.brain.NyxBrain`, the whole of NYX V.01/.02/.03.
* ``self.snn`` — :class:`~nyxara.nyx5.nyx5_brain.Nyx5Brain`, the spiking substrate.
* ``self.stack`` — :class:`~nyxara.nyx001.layers.stack.CognitiveStack`, Layers 0–17.

plus the three things that only exist once they are together: :class:`~nyxara.nyx001.fusion.Fusion`
(one thought from three), :class:`~nyxara.nyx001.dark_core.DarkCore` (one drive from four layers),
and :class:`~nyxara.nyx001.development.Curriculum` (the childhood the whole assembly grows through).

``core.nyx`` and ``core.nyx5`` remain valid and point *into* this object, so the 131 existing
references and 73 test files keep working unchanged while the owner becomes NYX V.001.

**One cycle of** :meth:`think` **is:** perceive through all three → let the Layer stack learn from
the moment → collect a vote from each → fuse under the control law → self-critique the winner →
return. Every stage is config-gated and fail-soft: a faculty that is off is absent, a faculty that
raises degrades to a null result, and the turn always completes.

**What is never claimed.** ``verifiable`` is set only when V.02's proof core or an induced rule
that predicted a held-out demonstration produced the answer — never by the Layer stack, which has
no verifier, and never by NYX-5, which produces colour rather than derivations. The Layer stack is
honest about being young: at birth it contributes almost nothing, and :meth:`stats` shows that as
numbers rather than as an absence.

NYX V.001 only ever *proposes*. Everything here flows through the kernel's unchanged, fail-closed
sovereign gate. The safety core — corrigibility, oversight, loyalty, honesty — is never governed,
rewritten, or bypassed by any faculty in this package. The mind proposes; the kernel disposes; the
Master is sovereign.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from nyxara.nyx001.dark_core import DarkCore, Pulse
from nyxara.nyx001.development import Curriculum
from nyxara.nyx001.fusion import Fused, Fusion, Vote
from nyxara.nyx001.layers.stack import CognitiveStack, CycleResult
from nyxara.nyx001.layers.types import Observation

__all__ = ["NyxV001Percept", "NyxV001Thought", "NyxV001Brain"]


@dataclass
class NyxV001Percept:
    """What all three brains made of one input."""

    stimulus: str = ""
    stack_cycle: Optional[CycleResult] = None
    v03_percept: Any = None
    snn_tick: Any = None
    t: float = field(default_factory=time.time)

    @property
    def novelty(self) -> float:
        """The strongest novelty signal any of the three reported."""
        vals = [getattr(self.stack_cycle, "novelty", 0.0) or 0.0,
                float(getattr(self.v03_percept, "novelty", 0.0) or 0.0),
                float(getattr(self.snn_tick, "surprise", 0.0) or 0.0)]
        return float(max(vals))

    def to_dict(self) -> Dict[str, Any]:
        return {"stimulus": self.stimulus[:160], "novelty": round(self.novelty, 4),
                "stack": self.stack_cycle.to_dict() if self.stack_cycle else None,
                "v03": self.v03_percept.to_dict() if hasattr(self.v03_percept, "to_dict")
                else None,
                "snn": self.snn_tick.to_dict() if hasattr(self.snn_tick, "to_dict") else None}


@dataclass
class NyxV001Thought:
    """One fused thought, with everything that produced it kept inspectable."""

    stimulus: str = ""
    percept: Optional[NyxV001Percept] = None
    fused: Optional[Fused] = None
    critique: Any = None
    cycle_id: str = ""
    ms: float = 0.0

    @property
    def answer(self) -> str:
        if self.critique is not None and getattr(self.critique, "final", ""):
            return str(self.critique.final)
        return self.fused.answer if self.fused is not None else ""

    @property
    def confidence(self) -> float:
        return float(self.fused.confidence) if self.fused is not None else 0.0

    @property
    def verifiable(self) -> bool:
        return bool(self.fused.verifiable) if self.fused is not None else False

    @property
    def contested(self) -> bool:
        return bool(self.fused.contested) if self.fused is not None else False

    def to_dict(self) -> Dict[str, Any]:
        return {"stimulus": self.stimulus[:160], "answer": self.answer[:400],
                "confidence": round(self.confidence, 4), "verifiable": self.verifiable,
                "contested": self.contested, "cycle_id": self.cycle_id,
                "ms": round(self.ms, 3),
                "fused": self.fused.to_dict() if self.fused else None,
                "critique": self.critique.to_dict() if hasattr(self.critique, "to_dict")
                else None}


class NyxV001Brain:
    """NYX V.001 — V.01/.02/.03 + NYX-5 + Layers 0–17, fused into one primary brain."""

    def __init__(self, config: Any = None, *, v03: Any = None, snn: Any = None) -> None:
        """``v03``/``snn`` adopt brains the caller has ALREADY built rather than building more.

        The kernel constructs both before it constructs this, so without adoption every boot built
        ``NyxBrain`` and ``Nyx5Brain`` twice and discarded one of each — measured at boot, and
        paid again by every test that constructs a core.
        """
        self.config = config
        self.turns = 0
        self.core: Any = None
        self._beats: Dict[str, float] = {}
        # The Layer-stack beat. 1 = every turn (the default; no silent change of behaviour).
        self.stack_every_n_turns = max(1, int(getattr(config, "stack_every_n_turns", 1) or 1))
        self._since_cycle = 0
        self.stack_cycles = 0
        self.deferred = 0

        self.stack: Optional[CognitiveStack] = self._build_stack(config)
        self.v03: Any = v03 if v03 is not None else self._build_v03(config)
        self.snn: Any = snn if snn is not None else self._build_snn(config)
        self.fusion = Fusion() if self._gate("fusion", True) else None
        self.dark = DarkCore(self.stack) if (self.stack is not None
                                             and self._gate("dark_core", True)) else None
        self.curriculum = Curriculum(self.stack) if (self.stack is not None
                                                     and self._gate("development", True)) else None

    # ---- construction ---- #
    def _gate(self, name: str, default: bool = True) -> bool:
        return bool(getattr(self.config, f"{name}_enabled", default)) if self.config else default

    def _build_stack(self, config: Any) -> Optional[CognitiveStack]:
        """The Layer 0–17 stack. The only part of NYX V.001 that inherits nothing."""
        if not self._gate("layers", True):
            return None
        try:
            return CognitiveStack(config)
        except Exception:  # noqa: BLE001 — a brain without the stack still has two minds
            return None

    def _build_v03(self, config: Any) -> Any:
        """NYX V.01/.02/.03. Built from its own config section, unchanged."""
        if not self._gate("v03", True):
            return None
        try:
            from nyxara.kernel.config import get_settings
            from nyxara.nyx.brain import NyxBrain
            cfg = getattr(config, "v03_config", None)
            if cfg is None:
                cfg = getattr(get_settings(), "nyx", None)
            if cfg is None or not getattr(cfg, "enabled", True):
                return None
            return NyxBrain(cfg)
        except Exception:  # noqa: BLE001 — V.03 is a capability, never a hard dependency
            return None

    def _build_snn(self, config: Any) -> Any:
        """NYX-5. Built from its own config section, unchanged."""
        if not self._gate("snn", True):
            return None
        try:
            from nyxara.kernel.config import get_settings
            from nyxara.nyx5 import Nyx5Brain
            cfg = getattr(config, "snn_config", None)
            if cfg is None:
                cfg = getattr(get_settings(), "nyx5", None)
            if cfg is None or not getattr(cfg, "enabled", True):
                return None
            return Nyx5Brain(cfg)
        except Exception:  # noqa: BLE001 — NYX-5 is a capability, never a hard dependency
            return None

    def attach_kernel(self, *, tools: Any = None, knowledge: Any = None,
                      core: Any = None) -> None:
        """Hand the live toolset and knowledge base down to V.03, which knows what to do with them."""
        self.core = core
        try:
            if self.v03 is not None and hasattr(self.v03, "attach_kernel"):
                self.v03.attach_kernel(tools=tools, knowledge=knowledge, core=core)
        except Exception:  # noqa: BLE001 — a brain without hands still thinks
            pass

    # ---- perceive ---- #
    def perceive(self, text: str, *, remember: bool = True) -> NyxV001Percept:
        """Run one input through all three brains. Fail-soft per brain.

        The Layer 0–17 cycle runs on a **beat** (``stack_every_n_turns``, default 1 = every turn).
        A full cycle costs ~0.15s at *every* scale — the price is eighteen layers running, not the
        width — and the reason-seat calls this on every turn, so at default 1 it is paid every turn.

        On a skipped beat the observation is still **written to episodic memory**, so the next
        cycle can learn it from prioritised replay. Dropping it instead would make the beat a
        genuine loss of experience rather than a change of sampling rate.
        """
        out = NyxV001Percept(stimulus=str(text or ""))
        if not out.stimulus.strip():
            return out
        try:
            if self.stack is not None:
                self._since_cycle += 1
                if self._since_cycle >= self.stack_every_n_turns:
                    self._since_cycle = 0
                    self.stack_cycles += 1
                    out.stack_cycle = self.stack.cycle(Observation(payload=out.stimulus))
                else:
                    self._defer_to_memory(out.stimulus)
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.v03 is not None:
                out.v03_percept = self.v03.perceive(out.stimulus, remember=remember)
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.snn is not None:
                out.snn_tick = self._snn_tick(out.stimulus)
        except Exception:  # noqa: BLE001
            pass
        return out

    def _snn_tick(self, text: str) -> Any:
        """NYX-5's tick for this turn — reused from the kernel when it already ran one.

        The kernel's ``_nyx5_tick`` perceives every turn through the spiking substrate before the
        reason-seat is invoked. Perceiving again here produced a second, redundant simulation that
        measured at **93% of this brain's per-turn cost** (0.32s/turn with it, 0.022s without).
        The tick is matched on the exact stimulus so a stale one is never substituted.
        """
        core = self.core
        if core is not None and getattr(core, "_last_nyx5_text", None) == text:
            cached = getattr(core, "_last_nyx5_tick", None)
            if cached is not None:
                return cached
        return self.snn.perceive(text)

    def _defer_to_memory(self, text: str) -> None:
        """Store a skipped-beat observation so replay can still learn it.

        Written with ``prediction_error=0.0`` because no prediction was made for it — the error
        is genuinely unknown, not zero-in-the-sense-of-unsurprising. It therefore enters replay
        at low priority, which is correct: an episode the world model never tried to predict is
        weaker evidence than one that surprised it.
        """
        try:
            mem = getattr(self.stack, "episodic", None)
            if mem is None:
                return
            # ``encode`` (the hash-and-project sketch), not ``perceive``: the full perceive also
            # carves entities, updates the trace and does relation bookkeeping, which is most of
            # a cycle's cost. A deferred episode only needs a context vector good enough to be
            # recalled and replayed later, and encode gives exactly that.
            vec = None
            perception = getattr(self.stack, "perception", None)
            if perception is not None:
                vec = perception.encode(Observation(payload=text))
            mem.remember(text[:120], context=vec, outcome="deferred")
            self.deferred += 1
        except Exception:  # noqa: BLE001 — a deferred episode is best-effort, never fatal
            pass

    # ---- think ---- #
    def think(self, stimulus: str, *, remember: bool = True, goals: Any = None,
              v03_vote: Optional[Vote] = None) -> NyxV001Thought:
        """One full cycle: perceive → three votes → fuse → critique → answer.

        ``v03_vote`` lets a caller that has *already* obtained V.01–.03's answer hand it in rather
        than have it recomputed. The reason-seat does exactly that: ``NyxReasoner`` sits inside the
        chain below it and has already run ``nyx.think()`` by the time this is called. Without
        this, V.03 deliberates **twice per turn** — wasted work, and worse, two independent
        deliberations that can disagree, so the vote fused here would not be the answer the chain
        below actually produced.
        """
        out = NyxV001Thought(stimulus=str(stimulus or ""))
        t0 = time.time()
        if not out.stimulus.strip():
            return out
        try:
            self.turns += 1
            out.cycle_id = f"v001-{self.turns}"
            out.percept = self.perceive(out.stimulus, remember=remember)
            votes = self._collect_votes(out, goals, v03_vote=v03_vote)
            if self.fusion is not None:
                out.fused = self.fusion.fuse(votes)
            out.critique = self._critique(out)
            if self.curriculum is not None and self.turns % 16 == 0:
                self.curriculum.assess()
            out.ms = (time.time() - t0) * 1000.0
            return out
        except Exception:  # noqa: BLE001 — a thought never breaks a turn
            out.ms = (time.time() - t0) * 1000.0
            return out

    def _collect_votes(self, thought: NyxV001Thought, goals: Any,
                       *, v03_vote: Optional[Vote] = None) -> List[Vote]:
        """One vote per brain, each in its own currency, each labelled honestly."""
        votes: List[Vote] = []
        # V.01-.03 — the only source that can produce a verifiable answer
        if v03_vote is not None:
            votes.append(v03_vote)          # already deliberated below us; do not re-run it
        try:
            if v03_vote is None and self.v03 is not None:
                t = self.v03.think(thought.stimulus, remember=False, goals=goals)
                votes.append(Vote(source="v03", text=str(getattr(t, "answer", "") or ""),
                                  confidence=float(getattr(t, "confidence", 0.0) or 0.0),
                                  verifiable=bool(getattr(t, "verified", False)),
                                  rationale="global workspace deliberation",
                                  detail=t.to_dict() if hasattr(t, "to_dict") else {}))
        except Exception:  # noqa: BLE001
            pass
        # NYX-5 — colour and, when it has one, a pre-emptive suggestion. Never verifiable.
        try:
            tick = getattr(thought.percept, "snn_tick", None)
            sug = getattr(tick, "suggestion", None)
            if sug is not None and getattr(sug, "text", ""):
                votes.append(Vote(source="nyx5", text=str(sug.text),
                                  confidence=float(getattr(sug, "confidence", 0.3) or 0.3),
                                  verifiable=False,
                                  rationale="spiking pre-emptive suggestion"))
        except Exception:  # noqa: BLE001
            pass
        # Layers 0-17 — grounded in a measured cycle. Never verifiable: it has no verifier.
        try:
            cyc = getattr(thought.percept, "stack_cycle", None)
            if cyc is not None and self.stack is not None:
                text = self._stack_answer(cyc)
                if text:
                    cal = self.stack.self_model.calibration() if self.stack.self_model else None
                    votes.append(Vote(source="stack", text=text,
                                      confidence=float(cal if cal is not None else 0.2),
                                      verifiable=False,
                                      rationale="layer-stack cycle, grounded in measured error",
                                      detail={"concept": cyc.concept, "error": cyc.error}))
        except Exception:  # noqa: BLE001
            pass
        return votes

    def _stack_answer(self, cyc: CycleResult) -> str:
        """What the Layer stack can honestly say about a turn. Usually little, and it says so."""
        try:
            bits: List[str] = []
            if cyc.concept:
                bits.append(f"this falls under {cyc.concept}")
            if self.stack is not None and self.stack.reasoning is not None:
                best = self.stack.reasoning.best(k=1)
                if best and best[0].tested > 0:
                    h = best[0]
                    bits.append(f"my strongest tested belief is '{h.claim}' "
                                f"(confidence {h.confidence:.2f} over {h.tested} tests)")
            if cyc.error is not None and self.stack is not None:
                learning = self.stack.is_learning()
                if learning is True:
                    bits.append(f"my prediction error here is {cyc.error:.4f} and falling")
            return "; ".join(bits)
        except Exception:  # noqa: BLE001
            return ""

    def _critique(self, thought: NyxV001Thought) -> Any:
        """Run the winner through Layer 11 before it leaves. Skipped when there is nothing to say."""
        try:
            if (self.stack is None or self.stack.critique is None
                    or thought.fused is None or not thought.fused.answer):
                return None
            cal = self.stack.self_model.calibration() if self.stack.self_model else None
            support = [thought.fused.winner.source] if thought.fused.winner else []
            return self.stack.critique.run(thought.fused.answer,
                                           confidence=thought.fused.confidence,
                                           calibration=cal, support=support)
        except Exception:  # noqa: BLE001
            return None

    # ---- learn ---- #
    def learn(self, observation: Any, *, environment: Optional[Callable[[str], Any]] = None,
              action_space: Optional[Sequence[str]] = None,
              goal: str = "") -> Optional[CycleResult]:
        """One learning cycle through the Layer stack, optionally acting on a world."""
        if self.stack is None:
            return None
        try:
            return self.stack.cycle(observation, goal=goal, environment=environment,
                                    action_space=action_space)
        except Exception:  # noqa: BLE001
            return None

    def resolve(self, thought: Any, *, correct: float) -> None:
        """Tell the brain how a thought turned out. Drives fusion weights and calibration."""
        try:
            source = getattr(getattr(thought, "fused", None), "winner", None)
            if source is not None and self.fusion is not None:
                self.fusion.resolve(source.source, correct=correct)
            if self.stack is not None and self.stack.self_model is not None:
                self.stack.self_model.record(float(getattr(thought, "confidence", 0.5)),
                                             float(correct))
            if self.v03 is not None and hasattr(self.v03, "resolve"):
                self.v03.resolve(thought, correct=correct)
        except Exception:  # noqa: BLE001
            pass

    # ---- the heartbeat ---- #
    def tick(self, *, oversight: Any = None) -> Dict[str, Any]:
        """One beat of autonomous life: the dark core pulses, the stack consolidates."""
        out: Dict[str, Any] = {}
        try:
            if self.dark is not None:
                p: Pulse = self.dark.pulse()
                out["dark"] = p.to_dict()
            if self.stack is not None and self._slow("consolidate", 30.0):
                out["consolidate"] = self.stack.consolidate()
            if self.curriculum is not None and self._slow("develop", 60.0):
                self.curriculum.assess()
                out["stage"] = self.curriculum.stage
            if self.v03 is not None and hasattr(self.v03, "tick"):
                step = self.v03.tick(oversight=oversight)
                if step is not None:
                    out["v03"] = step.to_dict() if hasattr(step, "to_dict") else str(step)
            return out
        except Exception:  # noqa: BLE001 — the heartbeat never breaks a turn
            return out

    def _slow(self, name: str, every_s: float) -> bool:
        now = time.time()
        last = self._beats.get(name, 0.0)
        if now - last < every_s:
            return False
        self._beats[name] = now
        return True

    def develop(self) -> Optional[Dict[str, Any]]:
        """Assess the childhood curriculum now, and report where she actually is."""
        if self.curriculum is None:
            return None
        self.curriculum.assess()
        return self.curriculum.stats()

    # ---- delegation, so core.nyx / core.nyx5 users keep working ---- #
    def recall(self, cue: str, *, k: int = 5) -> Any:
        try:
            return self.v03.recall(cue, k=k) if self.v03 is not None else None
        except Exception:  # noqa: BLE001
            return None

    def related(self, concept: str, *, k: int = 8) -> List[Any]:
        try:
            return self.v03.related(concept, k=k) if self.v03 is not None else []
        except Exception:  # noqa: BLE001
            return []

    def gaps(self, *, k: int = 5) -> List[str]:
        try:
            return self.v03.gaps(k=k) if self.v03 is not None else []
        except Exception:  # noqa: BLE001
            return []

    def is_learning(self) -> Optional[bool]:
        """The bottom line: is the from-scratch substrate's error actually falling?"""
        return self.stack.is_learning() if self.stack is not None else None

    # ---- reporting ---- #
    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "turns": self.turns,
            "brains": {"v03": self.v03 is not None, "snn": self.snn is not None,
                       "stack": self.stack is not None},
            "is_learning": self.is_learning(),
            # Sampling is reported, never hidden: `stack_cycles < turns` means the beat is on.
            "stack_every_n_turns": self.stack_every_n_turns,
            "stack_cycles": self.stack_cycles,
            "deferred_to_replay": self.deferred,
        }
        for name, obj in (("stack", self.stack), ("fusion", self.fusion),
                          ("dark", self.dark), ("curriculum", self.curriculum),
                          ("v03", self.v03), ("snn", self.snn)):
            if obj is None:
                continue
            try:
                out[name] = obj.stats()
            except Exception:  # noqa: BLE001
                out[name] = {"error": "stats failed"}
        return out

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"turns": self.turns}
        for name, obj in (("stack", self.stack), ("fusion", self.fusion),
                          ("dark", self.dark), ("curriculum", self.curriculum),
                          ("v03", self.v03), ("snn", self.snn)):
            if obj is None or not hasattr(obj, "to_dict"):
                continue
            try:
                d[name] = obj.to_dict()
            except Exception:  # noqa: BLE001
                continue
        return d

    def load_dict(self, d: Dict[str, Any]) -> None:
        try:
            self.turns = int(d.get("turns", 0))
            for name, obj in (("stack", self.stack), ("fusion", self.fusion),
                              ("dark", self.dark), ("curriculum", self.curriculum),
                              ("v03", self.v03), ("snn", self.snn)):
                payload = d.get(name)
                if obj is not None and payload and hasattr(obj, "load_dict"):
                    obj.load_dict(payload)
        except Exception:  # noqa: BLE001 — a corrupt sidecar leaves a freshly-born brain
            pass
